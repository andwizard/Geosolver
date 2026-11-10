"""Convert old eval_results.json (list) to new format (dict with summary + tasks)."""

import json

INPUT_PATH = "eval_results.json"
OUTPUT_PATH = "eval_results_new.json"


def _score_dist(items):
    dist = {}
    for s in items:
        key = str(int(s)) if s == int(s) else str(s)
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def convert():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both old (list) and new (dict) formats
    tasks_list = data.get("tasks", data) if isinstance(data, dict) else data
    tasks_list = [t for t in tasks_list if isinstance(t, dict)]

    valid = [t for t in tasks_list if not t.get("skipped") and not t.get("error")]
    scores = [t["overall_score"] for t in valid if isinstance(t.get("overall_score"), (int, float))]

    summary = {
        "total_tasks": len(tasks_list),
        "evaluated": len(valid),
        "skipped": sum(1 for t in tasks_list if t.get("skipped")),
        "errors": sum(1 for t in tasks_list if t.get("error")),
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
            t[dk]["score"] for t in valid
            if isinstance(t.get(dk), dict) and isinstance(t[dk].get("score"), (int, float))
        ]
        if dim_scores:
            summary[f"{dk}_average"] = round(sum(dim_scores) / len(dim_scores), 2)
            summary[f"{dk}_distribution"] = _score_dist(dim_scores)

    result = {
        "summary": summary,
        "tasks": tasks_list,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(tasks_list)} tasks")
    print(f"Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert()
