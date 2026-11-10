"""Evaluation entry point.

Evaluates generated solutions against reference solutions using an LLM-based
evaluator, producing scores across multiple dimensions with checkpoint/resume support.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from Eval.eval.agent import RemoteSensingEvaluator
import config


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> list[dict]:
    """Load previously completed results from checkpoint."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict], checkpoint_path: str) -> None:
    """Save current results as checkpoint."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_final_results(results: dict, path: str) -> None:
    """Save final evaluation results with summary and task details."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Summary builder ──────────────────────────────────────────────────────────

def _score_distribution(items: list) -> dict:
    """Build a sorted distribution of scores."""
    dist = {}
    for s in items:
        key = str(int(s)) if s == int(s) else str(s)
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def build_summary(tasks: list, results: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    valid_results = [r for r in results if not r.get("skipped") and not r.get("error")]
    scores = [r["overall_score"] for r in valid_results
              if isinstance(r.get("overall_score"), (int, float))]

    skipped = sum(1 for r in results if r.get("skipped"))
    errors = sum(1 for r in results if r.get("error"))

    summary = {
        "total_tasks": len(tasks),
        "evaluated": len(valid_results),
        "skipped": skipped,
        "errors": errors,
    }

    if scores:
        summary["average_score"] = round(sum(scores) / len(scores), 2)
        summary["min_score"] = min(scores)
        summary["max_score"] = max(scores)
        summary["score_distribution"] = _score_distribution(scores)

    # Per-dimension averages and distributions
    dim_keys = ["task_consistency", "result_completeness", "accuracy", "visualization_quality"]
    for dk in dim_keys:
        dim_scores = [
            r[dk]["score"] for r in valid_results
            if isinstance(r.get(dk), dict) and isinstance(r[dk].get("score"), (int, float))
        ]
        if dim_scores:
            summary[f"{dk}_average"] = round(sum(dim_scores) / len(dim_scores), 2)
            summary[f"{dk}_distribution"] = _score_distribution(dim_scores)

    return summary


# ── Token usage summary ──────────────────────────────────────────────────────

def print_token_usage(results: list[dict]) -> None:
    """Print aggregated token usage across all evaluations."""
    token_usages = [r.get("token_usage", {}) for r in results if r.get("token_usage")]
    if not token_usages:
        return

    total_prompt = sum(t.get("prompt_tokens", 0) for t in token_usages)
    total_completion = sum(t.get("completion_tokens", 0) for t in token_usages)
    total_tokens = sum(t.get("total_tokens", 0) for t in token_usages)
    total_calls = sum(t.get("call_count", 0) for t in token_usages)

    print(f"\n--- Token Usage ---")
    print(f"API calls: {total_calls}")
    print(f"Prompt tokens: {total_prompt:,}")
    print(f"Completion tokens: {total_completion:,}")
    print(f"Total tokens: {total_tokens:,}")


# ── Summary printer ──────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print evaluation summary to console."""
    print(f"\n{'=' * 60}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Evaluated: {summary['evaluated']}, "
          f"Skipped: {summary['skipped']}, "
          f"Errors: {summary['errors']}")

    if "average_score" in summary:
        print(f"Average score: {summary['average_score']:.2f}")
        print(f"Min score: {summary['min_score']:.2f}")
        print(f"Max score: {summary['max_score']:.2f}")
        print(f"Score distribution: {summary['score_distribution']}")
        print(f"\n--- Dimension Averages ---")
        for dk in ["task_consistency", "result_completeness", "accuracy", "visualization_quality"]:
            key = f"{dk}_average"
            if key in summary:
                print(f"  {dk}: {summary[key]:.2f}")


# ── Single task evaluation ───────────────────────────────────────────────────

def evaluate_single_task(evaluator: RemoteSensingEvaluator, task: dict,
                         solutions_root: str, ref_solutions_dir: str) -> dict:
    """Evaluate a single task and return the result dict."""
    task_id = task["task_ID"]

    # Directly assign 0 for TIMEOUT tasks without LLM evaluation
    generated_solution = task.get("generated_solution", {})
    if isinstance(generated_solution, dict) and generated_solution.get("response") == "TIMEOUT":
        return {
            "task_id": task_id,
            "reason": "Task timed out",
            "overall_score": 0.0,
        }

    gen_dir = os.path.join(solutions_root, task_id)
    # Reference outputs live under each task's "workspace" subfolder
    # (e.g. modify/solutions/<task_id>/workspace/*). The evaluator scans the
    # given directory recursively and matches by filename, so pointing at the
    # workspace dir is enough.
    ref_dir = os.path.join(ref_solutions_dir, task_id, "workspace")

    if not os.path.exists(gen_dir) and not os.path.exists(ref_dir):
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": "No output files found",
            "overall_score": 0.0,
        }

    try:
        result = evaluator.evaluate(
            task_text=task["task_text"],
            task_id=task_id,
            used_dataset=task["used_dataset"],
            Expected_Output=task["Expected_Output"],
            reference_solution=ref_dir if os.path.exists(ref_dir) else None,
            generated_solution=gen_dir if os.path.exists(gen_dir) else None,
            reference_answer=task.get("reference_answer", ""),
            generated_answer=task.get("generated_solution", {}).get("answer", ""),
        )
        result_dict = result.to_dict()
        result_dict["task_id"] = task_id
        result_dict["token_usage"] = evaluator.get_token_usage()
        return result_dict
    except Exception as e:
        return {
            "task_id": task_id,
            "error": str(e),
            "overall_score": 0.0,
        }


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_eval(solutions_json: str, solutions_root: str,
             ref_solutions_dir: str, output_path: str,
             checkpoint_path: str, resume: bool = False) -> None:
    """Run evaluation on all generated solutions."""
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
    results = load_checkpoint(checkpoint_path) if resume else []
    completed_ids = {r.get("task_id") for r in results}

    pending_tasks = [t for t in tasks if t["task_ID"] not in completed_ids]

    if resume and completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)} tasks done, "
              f"{len(pending_tasks)} remaining.\n")

    for task in tqdm(pending_tasks, desc="Evaluating"):
        result = evaluate_single_task(evaluator, task, solutions_root, ref_solutions_dir)
        results.append(result)
        save_checkpoint(results, checkpoint_path)

    summary = build_summary(tasks, results)
    print_summary(summary)
    print_token_usage(results)
    print(f"\nResults saved to {output_path}")

    save_final_results({"summary": summary, "tasks": results}, output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated solutions")
    parser.add_argument(
        "--solutions-root", type=str,
        default=str(config.RUNS_DIR),
        help="Root directory containing generated solutions")
    parser.add_argument(
        "--ref-solutions-dir", type=str,
        default=str(config.REF_SOLUTIONS_DIR),
        help="Directory containing reference solutions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    solutions_root = args.solutions_root
    solutions_json = str(Path(solutions_root) / "generated_solutions.json")
    output_path = str(Path(solutions_root) / "eval_result.json")
    checkpoint_path = str(Path(solutions_root) / ".eval_checkpoint.json")

    run_eval(
        solutions_json=solutions_json,
        solutions_root=solutions_root,
        ref_solutions_dir=args.ref_solutions_dir,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
