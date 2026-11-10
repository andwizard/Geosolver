import os
import json
import argparse
from pathlib import Path

from tqdm import tqdm

from eval.agent import RemoteSensingEvaluator
import config

# Paths with config defaults (overridable via CLI)
REF_SOLUTIONS_DIR = config.DATA_ROOT / "solutions"


def load_checkpoint(checkpoint_path):
    """Load previously completed results from checkpoint."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(results, checkpoint_path):
    """Save current results as checkpoint."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_final_results(results, path):
    """Save final evaluation results."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated solutions")
    parser.add_argument("--solutions-json", type=str,
                        default=str(config.RUNS_DIR / "generated_solutions.json"),
                        help="Path to generated solutions JSON file")
    parser.add_argument("--solutions-root", type=str, default=None,
                        help="Root directory for generated solution files (default: parent of --solutions-json)")
    parser.add_argument("--ref-solutions-dir", type=str,
                        default=str(REF_SOLUTIONS_DIR),
                        help="Directory containing reference solutions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="./eval_results.json",
                        help="Output path for results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file path (default: .eval_checkpoint.json in output dir)")
    args = parser.parse_args()

    solutions_json = args.solutions_json
    solutions_root = args.solutions_root or str(Path(solutions_json).parent)
    ref_solutions_dir = args.ref_solutions_dir
    output_path = args.output
    checkpoint_path = args.checkpoint or str(Path(output_path).parent / ".eval_checkpoint_geoagent.json")

    evaluator = RemoteSensingEvaluator(
        api_key=config.OPENAI_API_KEY,
        api_base=config.OPENAI_API_BASE,
        model=config.OPENAI_MODEL,
        multimodal_model=config.OPENAI_MODEL,
    )

    with open(solutions_json, "r") as f:
        data = json.load(f)

    tasks = data["tasks"]

    # Resume logic
    results = load_checkpoint(checkpoint_path) if args.resume else []
    completed_ids = {r.get("task_id") for r in results}
    skipped = 0

    # Filter tasks to process
    pending_tasks = []
    for task in tasks:
        if task["task_ID"] not in completed_ids:
            pending_tasks.append(task)

    if args.resume and completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)} tasks already done, {len(pending_tasks)} remaining.\n")

    for task in tqdm(pending_tasks, desc="Evaluating"):
        task_id = task["task_ID"]
        task_text = task["task_text"]
        used_dataset = task["used_dataset"]
        Expected_Output = task['Expected_Output']
        reference_answer = task.get("reference_answer", "")
        generated_answer = task.get("generated_solution", {}).get("answer", "")

        gen_dir = os.path.join(solutions_root, task_id)
        ref_dir = os.path.join(ref_solutions_dir, task_id, "Outputs")

        if not os.path.exists(gen_dir) and not os.path.exists(ref_dir):
            skipped += 1
            results.append({
                "task_id": task_id,
                "skipped": True,
                "reason": "No output files found",
                "overall_score": 0.0,
            })
            save_checkpoint(results, checkpoint_path)
            continue
        try:
            result = evaluator.evaluate(
                task_text=task_text,
                task_id=task_id,
                used_dataset=used_dataset,
                Expected_Output=Expected_Output,
                reference_solution=ref_dir if os.path.exists(ref_dir) else None,
                generated_solution=gen_dir if os.path.exists(gen_dir) else None,
                reference_answer=reference_answer,
                generated_answer=generated_answer,
            )
            result_dict = result.to_dict()
            result_dict["task_id"] = task_id
            result_dict["token_usage"] = evaluator.get_token_usage()
            results.append(result_dict)
        except Exception as e:
            results.append({
                "task_id": task_id,
                "error": str(e),
                "overall_score": 0.0,
            })

        # Save checkpoint after each task
        save_checkpoint(results, checkpoint_path)

    # Build summary statistics
    valid_results = [r for r in results if not r.get("skipped") and not r.get("error")]
    scores = [r["overall_score"] for r in valid_results if isinstance(r.get("overall_score"), (int, float))]

    def _score_dist(items):
        dist = {}
        for s in items:
            key = str(int(s)) if s == int(s) else str(s)
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items()))

    summary = {
        "total_tasks": len(tasks),
        "evaluated": len(valid_results),
        "skipped": skipped,
        "errors": len(results) - len(valid_results),
    }
    if scores:
        summary["average_score"] = round(sum(scores) / len(scores), 2)
        summary["min_score"] = min(scores)
        summary["max_score"] = max(scores)
        summary["score_distribution"] = _score_dist(scores)

    # Per-dimension averages and distributions
    dim_keys = ["task_consistency", "result_completeness", "accuracy", "visualization_quality"]
    for dk in dim_keys:
        dim_scores = [
            r[dk]["score"] for r in valid_results
            if isinstance(r.get(dk), dict) and isinstance(r[dk].get("score"), (int, float))
        ]
        if dim_scores:
            summary[f"{dk}_average"] = round(sum(dim_scores) / len(dim_scores), 2)
            summary[f"{dk}_distribution"] = _score_dist(dim_scores)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Evaluated: {len(valid_results)}, Skipped: {skipped}, Errors: {len(results) - len(valid_results)}")
    print(f"Results saved to {output_path}")

    if scores:
        print(f"Average score: {summary['average_score']:.2f}")
        print(f"Min score: {summary['min_score']:.2f}")
        print(f"Max score: {summary['max_score']:.2f}")
        print(f"Score distribution: {summary['score_distribution']}")
        print(f"\n--- Dimension Averages ---")
        for dk in dim_keys:
            key = f"{dk}_average"
            if key in summary:
                print(f"  {dk}: {summary[key]:.2f}")

    # Print token usage summary
    token_usages = [r.get("token_usage", {}) for r in results if r.get("token_usage")]
    if token_usages:
        total_prompt = sum(t.get("prompt_tokens", 0) for t in token_usages)
        total_completion = sum(t.get("completion_tokens", 0) for t in token_usages)
        total_tokens = sum(t.get("total_tokens", 0) for t in token_usages)
        total_calls = sum(t.get("call_count", 0) for t in token_usages)
        print(f"\n--- Token Usage ---")
        print(f"API calls: {total_calls}")
        print(f"Prompt tokens: {total_prompt:,}")
        print(f"Completion tokens: {total_completion:,}")
        print(f"Total tokens: {total_tokens:,}")

    # Save final results as a dict with summary + tasks
    final_output = {
        "summary": summary,
        "tasks": results,
    }
    save_final_results(final_output, output_path)


if __name__ == "__main__":
    main()
