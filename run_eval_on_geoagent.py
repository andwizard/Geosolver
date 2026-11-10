#!/usr/bin/env python3
"""
Evaluate geosolver-on-geoagent task outputs using Claude Code CLI.

Each task's output directory (under the solutions root) is submitted to
`claude -p` for manual-style evaluation against the standardized geospatial
output criteria. There is no ground truth — Claude Code independently reads
the task outputs and source data to produce its judgment.

Each criterion is judged as **[Compliant / Non-Compliant]** with a brief
supporting rationale.

Usage:
    python run_eval_on_geoagent.py                           # process all tasks
    python run_eval_on_geoagent.py --max 5                   # limit to first 5 tasks
    python run_eval_on_geoagent.py --resume                  # resume from checkpoint
    python run_eval_on_geoagent.py --timeout 600             # 10 min timeout per task
    python run_eval_on_geoagent.py --solutions-root Output/runs/<run_id>
"""

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATASETS_FILE = BASE_DIR / "Data" / "GeoAgent" / "datasets.json"

DATA_DATASETS_DIR = BASE_DIR / "Data" / "GeoAgent" / "datasets"

# ── JSON schema for structured output ─────────────────────────────────────────
OUTPUT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "geospatial_data_files": {
            "type": "object",
            "properties": {
                "spatial_reference": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "data_structure": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "attribute_completeness": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "spatial_constraint": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
            },
            "required": ["spatial_reference", "data_structure", "attribute_completeness", "spatial_constraint"],
        },
        "visual_outputs": {
            "type": "object",
            "properties": {
                "data_correctness": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "symbolization": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "map_elements": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
            },
            "required": ["data_correctness", "symbolization", "map_elements"],
        },
        "analytical_conclusion": {
            "type": "object",
            "properties": {
                "core_question_coverage": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
                "analytical_validity": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["Compliant", "Non-Compliant", "N/A"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                },
            },
            "required": ["core_question_coverage", "analytical_validity"],
        },
        "overall_verdict": {
            "type": "string",
            "enum": ["Compliant", "Non-Compliant"],
            "description": "Compliant only if ALL applicable criteria are Compliant. Non-Compliant if any criterion is Non-Compliant.",
        },
        "overall_rationale": {
            "type": "string",
            "description": "Concise summary of the evaluation, highlighting key strengths and critical issues (2-4 sentences).",
        },
    },
    "required": [
        "geospatial_data_files", "visual_outputs", "analytical_conclusion",
        "overall_verdict", "overall_rationale",
    ],
})

# ── evaluation rubric ─────────────────────────────────────────────────────────
EVALUATION_RUBRIC = """\
## Geospatial Data Files Evaluation Criteria:

1. **Spatial Reference Correctness**: The Coordinate Reference System (CRS) of the output is explicitly defined, fully consistent with task specifications, and aligned with the spatial reference of input data, with no undefined settings, deviations, or conflicts.

2. **Data Structure and Type Validity**: The data format accurately represents corresponding geospatial entities. For vector data, the geometry type is semantically correct and matches entity attributes. For raster data, spatial resolution, grid dimensions, and band count fully meet analytical requirements without format disorder or parameter mismatches.

3. **Attribute and Value Completeness**: Vector data contains all required attribute fields with valid and semantically clear information. Raster data has complete and valid pixel value ranges and statistical metrics that conform to task requirements, without missing fields, abnormal values, or invalid data.

4. **Spatial Constraint Compliance**: All spatial constraints specified in the task are fully satisfied, including spatial filtering results, study area boundaries, buffer distances, spatial overlay outputs, and raster computation results. All spatial operations are accurate with no spatial deviation, computational error, or unmet constraints.

## Visual Outputs Evaluation Criteria

1. **Data Correctness**: Visualizations are generated from accurate and complete data sources, faithfully and precisely reflecting underlying geospatial analytical results. Mapped values and categories are fully consistent with computed statistics, without data distortion, result deviation, or presentation errors.

2. **Symbolization Logic Soundness**: Visual encoding schemes (including color schemes, symbol sizes, and line weights) are scientifically standardized. Visual elements effectively distinguish spatial patterns, geographic categories, and data hierarchies with clear logic and no misleading presentation, color confusion, symbol misuse, or hierarchical ambiguity.

3. **Essential Map Elements Completeness**: Visualizations contain fundamental cartographic elements including legends, titles, and key labels. The complete information system enables independent interpretation and understanding of core map content without missing critical elements or unclear annotations.

## Analytical Conclusion Evaluation Criteria

1. **Core Question Coverage**: The conclusion closely aligns with predefined analytical objectives and research questions, fully addressing all core analytical requirements without omission of key issues, theme deviation, or irrelevant content.

2. **Analytical Validity**: All conclusions are derived from standardized and correct geospatial analytical procedures, and are highly consistent with final geospatial datasets and computational results. Conclusions are objective and well-supported, free from subjective speculation, contradiction with results, or procedural errors.\
"""

# ── prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a senior GIS analyst performing a rigorous manual evaluation of a
geospatial task solution. You must independently examine the task, the source
data files, and the generated outputs, then evaluate the solution against the
standardized criteria below.

**Task Instruction**: Conduct a comprehensive manual verification and evaluation
on the outputs of geospatial tasks. Assess compliance item by item according to
the standardized criteria below. Output only **[Compliant / Non-Compliant]** with
a brief supporting rationale. The evaluation covers three major modules:
Geospatial Data Files, Visual Outputs, and Analytical Conclusions.

**Evaluation Rules**: Verify each indicator strictly against official task
specifications, input data standards, and analytical requirements. An output is
compliant only if all detailed criteria are fully satisfied; any omission, error,
or inconsistency in core criteria results in a non-compliant judgment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task ID: {task_id}
Task Description: {task_text}

Source Data (relative to {data_dir}):
{data_paths}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT DIRECTORY TO EVALUATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{output_dir}

This is the root directory containing all outputs produced by the task solution.
The exact contents depend on the task — explore freely to discover what is there.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU MUST EXAMINE ALL OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 0 — Explore the output directory freely:
  a) Start by listing the entire directory tree recursively (including
     subdirectories) to discover all files and their organization
  b) Read any scripts (.py), logs (.log, .txt), or other text files to
     understand what was executed and whether any errors occurred
  c) For each output file you find, inspect it thoroughly according to its type:
     - Vector files (shp/geojson/gpkg): CRS, geometry type, attribute fields,
       row count, and sample values
     - Raster files (tif/img/grd): CRS, resolution, bounds, band count,
       and pixel value statistics
     - Visualizations (png/jpg/svg): view the image and examine its content
     - Tabular files (csv/xlsx): columns and sample rows

Step 1 — Explore the source data:
  a) Navigate to {data_dir}
  b) Locate and inspect the source data files listed above (they may be in
     subdirectories — search recursively if needed)
  c) Understand the input data structure, CRS, attributes, extent, etc.

Step 2 — Cross-reference and verify:
  a) Does the output correctly address the task description?
  b) Are the spatial operations correct given the input data?
  c) Are numerical/categorical results consistent with the data?
  d) Are visualizations accurate and well-designed?

Step 3 — Judge each criterion as **[Compliant / Non-Compliant]** with a brief
rationale. Use **N/A** only if the criterion genuinely does not apply to this
task (e.g. visualization criteria when the task requires no visualization).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + EVALUATION_RUBRIC + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- **Compliant**: ALL applicable criteria (excluding N/A) are Compliant.
- **Non-Compliant**: ANY applicable criterion is Non-Compliant.

Return ONLY the JSON object — no explanation, no markdown fences.
"""


# ── helpers ────────────────────────────────────────────────────────────────────

_should_stop = False


def _signal_handler(signum, frame):
    global _should_stop
    _should_stop = True
    print("\nInterrupt received. Stopping after current task...")


def _format_data_paths(data_path):
    """Format the data_path field for display in the prompt."""
    if isinstance(data_path, list):
        return "\n".join(f"  - {p}" for p in data_path)
    else:
        return f"  - {data_path}"


def build_prompt(task: dict, output_dir: str) -> str:
    """Build the evaluation prompt for a single task."""
    task_id = f"task_{task['id']}"
    task_text = task["task"]
    data_paths_formatted = _format_data_paths(task.get("data_path", ""))

    return PROMPT_TEMPLATE.format(
        task_id=task_id,
        task_text=task_text,
        data_dir=str(DATA_DATASETS_DIR),
        data_paths=data_paths_formatted,
        output_dir=output_dir,
    )


def run_claude(output_dir: str, task: dict, timeout: int) -> dict:
    """Call `claude -p` and return parsed evaluation dict."""
    prompt = build_prompt(task, output_dir)

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--json-schema", OUTPUT_SCHEMA,
        "--dangerously-skip-permissions",
        "--verbose",
    ]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        def _watcher(p=proc):
            while p.poll() is None:
                if _should_stop:
                    p.kill()
                    return
                time.sleep(0.2)

        t = threading.Thread(target=_watcher, daemon=True)
        t.start()

        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            stdout, _ = proc.communicate()
        return {"error": "claude timeout"}
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        return {"error": f"subprocess error: {exc}"}

    if _should_stop:
        raise KeyboardInterrupt("Interrupted by user")

    # Parse stream-json looking for StructuredOutput
    structured_output = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                    inp = block.get("input", {})
                    structured_output = {
                        "geospatial_data_files": inp.get("geospatial_data_files", {}),
                        "visual_outputs": inp.get("visual_outputs", {}),
                        "analytical_conclusion": inp.get("analytical_conclusion", {}),
                        "overall_verdict": inp.get("overall_verdict", "Non-Compliant"),
                        "overall_rationale": inp.get("overall_rationale", ""),
                    }

    if structured_output is not None:
        return structured_output

    # Fallback: regex extract JSON from raw output
    m = re.search(r'\{[\s\S]*\}', stdout or "")
    if m:
        try:
            obj = json.loads(m.group())
            return {
                "geospatial_data_files": obj.get("geospatial_data_files", {}),
                "visual_outputs": obj.get("visual_outputs", {}),
                "analytical_conclusion": obj.get("analytical_conclusion", {}),
                "overall_verdict": obj.get("overall_verdict", "Non-Compliant"),
                "overall_rationale": obj.get("overall_rationale", ""),
            }
        except json.JSONDecodeError:
            pass

    return {"error": f"could not parse output: {(stdout or '')[:500]}"}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> list:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, checkpoint_path)


def save_final_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── summary builder ──────────────────────────────────────────────────────────

def build_summary(tasks: list, results: list) -> dict:
    """Compute summary statistics from evaluation results."""
    valid_results = [r for r in results if not r.get("error")]
    error_results = [r for r in results if r.get("error")]

    compliant = sum(1 for r in valid_results if r.get("overall_verdict") == "Compliant")
    non_compliant = len(valid_results) - compliant

    summary = {
        "total_tasks": len(tasks),
        "evaluated": len(valid_results),
        "errors": len(error_results),
        "compliant": compliant,
        "non_compliant": non_compliant,
    }

    if valid_results:
        summary["compliance_rate"] = round(compliant / len(valid_results) * 100, 1)

    # Per-criterion compliance counts
    criterion_counts = {}
    for module_key in ["geospatial_data_files", "visual_outputs", "analytical_conclusion"]:
        all_criteria = {}
        for r in valid_results:
            module = r.get(module_key, {})
            for criterion, details in module.items():
                if isinstance(details, dict):
                    if criterion not in all_criteria:
                        all_criteria[criterion] = {"Compliant": 0, "Non-Compliant": 0, "N/A": 0}
                    verdict = details.get("verdict", "N/A")
                    if verdict in all_criteria[criterion]:
                        all_criteria[criterion][verdict] += 1
        criterion_counts[module_key] = all_criteria

    summary["criterion_compliance"] = criterion_counts

    # Difficulty breakdown
    diff_groups = {}
    for t in tasks:
        level = t.get("task_level", "unknown")
        if level not in diff_groups:
            diff_groups[level] = {"total": 0, "compliant": 0, "evaluated": 0}
        tid = f"task_{t['id']}"
        diff_groups[level]["total"] += 1
        for r in valid_results:
            if r.get("task_id") == tid:
                diff_groups[level]["evaluated"] += 1
                if r.get("overall_verdict") == "Compliant":
                    diff_groups[level]["compliant"] += 1

    for level, stats in diff_groups.items():
        if stats["evaluated"] > 0:
            stats["compliance_rate"] = round(stats["compliant"] / stats["evaluated"] * 100, 1)
        else:
            stats["compliance_rate"] = 0.0

    summary["difficulty_breakdown"] = diff_groups

    return summary


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Evaluate geoagent task outputs using Claude Code CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--solutions-root", type=str,
        default=str(BASE_DIR / "Output" / "runs"),
        help="Root directory containing task run outputs")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Specific run ID under solutions-root. "
             "If not provided, uses the most recent run directory.")
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of tasks to process")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Seconds per task (default: 1800)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from task index (0-based)")
    args = parser.parse_args()

    # ── resolve run directory ─────────────────────────────────────────────────
    solutions_root = Path(args.solutions_root)
    if args.run_id:
        run_dir = solutions_root / args.run_id
    else:
        run_dirs = sorted(
            [d for d in solutions_root.iterdir() if d.is_dir()],
            key=lambda d: d.name, reverse=True,
        )
        if not run_dirs:
            print("No run directories found under", solutions_root)
            return
        run_dir = run_dirs[0]
        print(f"Using most recent run directory: {run_dir.name}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    output_path = str(run_dir / "eval_result.json")
    checkpoint_path = str(run_dir / ".eval_checkpoint.json")

    # ── load tasks ────────────────────────────────────────────────────────────
    with open(DATASETS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.start > 0:
        tasks = tasks[args.start:]
    if args.max:
        tasks = tasks[:args.max]

    # ── resume logic ──────────────────────────────────────────────────────────
    results = load_checkpoint(checkpoint_path) if args.resume else []
    completed_ids = {r.get("task_id") for r in results}
    pending_tasks = [t for t in tasks if f"task_{t['id']}" not in completed_ids]

    if args.resume and completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)} tasks done, "
              f"{len(pending_tasks)} remaining.\n")

    stats = {"evaluated": 0, "errors": 0}

    print(f"Processing {len(pending_tasks)} tasks")
    print(f"  Solutions root: {run_dir}")
    print(f"  Datasets: {DATASETS_FILE}")
    print(f"  Output: {output_path}\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    with tqdm(pending_tasks, unit="task", dynamic_ncols=True) as pbar:
        for task in pbar:
            task_id = f"task_{task['id']}"
            pbar.set_description(f"[{task_id}] {task['task'][:55]}", refresh=False)

            output_dir = str(run_dir / task_id)
            if not os.path.isdir(output_dir):
                result = {
                    "task_id": task_id,
                    "error": f"Output directory not found: {output_dir}",
                }
            else:
                result = run_claude(output_dir, task, timeout=args.timeout)

            result["task_id"] = task_id
            result["difficulty"] = task.get("task_level", "unknown")

            results.append(result)
            save_checkpoint(results, checkpoint_path)

            if result.get("error"):
                stats["errors"] += 1
            else:
                stats["evaluated"] += 1

            pbar.set_postfix(
                evaluated=stats["evaluated"],
                errors=stats["errors"],
                refresh=True,
            )

    # ── build and save final output ───────────────────────────────────────────
    summary = build_summary(tasks, results)
    print(f"\n{'=' * 60}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Evaluated: {summary['evaluated']}, Errors: {summary['errors']}")
    print(f"Compliant: {summary.get('compliant', 'N/A')}, "
          f"Non-Compliant: {summary.get('non_compliant', 'N/A')}")
    if "compliance_rate" in summary:
        print(f"Compliance Rate: {summary['compliance_rate']}%")

    final_output = {"summary": summary, "tasks": results}
    save_final_results(final_output, output_path)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
