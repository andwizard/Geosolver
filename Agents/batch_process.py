#!/usr/bin/env python3
"""
Batch process GeoDatasets/datasets.json tasks using Claude Code CLI.

Each task is submitted to `claude -p` with a structured JSON schema,
and the result is appended one-by-one to a fresh output JSON file.
REJECT tasks are written directly without calling Claude.
Tasks outside the --filter scope are also written directly.

Usage:
    python batch_process.py                          # process all ACCEPT tasks
    python batch_process.py --filter POSSIBLE        # process POSSIBLE tasks
    python batch_process.py --filter ALL             # process all non-REJECT tasks
    python batch_process.py --max 10                 # limit to first 10 tasks
    python batch_process.py --resume                 # re-run FAILED, skip only ACCEPT
    python batch_process.py --resume --resume-skip-failed  # skip all written tasks
    python batch_process.py --output my_out.json     # custom output path
    python batch_process.py --timeout 180            # 3 min timeout per task
"""

import argparse
import json
import re
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASETS_FILE = BASE_DIR / "datastes_new_modified_fix_1.json"
DEFAULT_OUTPUT = BASE_DIR / "datasets_solutions.json"
DATA_BASE_DIR = BASE_DIR / "Data"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"

# ── JSON schema for structured output ─────────────────────────────────────────
OUTPUT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "response": {"type": "string", "enum": ["ACCEPT", "FAILED"]},
        "script":   {"type": "string"},
        "answer":   {"type": "string"},
    },
    "required": ["response", "script"],
})

# ── prompt template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a GIS analyst. Your job is to generate a complete, runnable Python script \
for the following spatial analysis task.

TASK:
{task_text}

REQUIRED DATA (all files are under {data_dir}/):
{file_list}

EXEPTED OUTPUT:
{expected_output}

INSTRUCTIONS:
1. Write a complete Python script in "script" that:
   - Loads data with absolute paths under {data_dir}/
   - Saves any output figures to {output_dir}/ with descriptive filenames
   - Uses dark-theme matplotlib (background #1a1a2e, text white)
2. If you cannot produce a valid script, set "response" to "FAILED" and "script" to "".
3. Otherwise set "response" to "ACCEPT".
4. To fullfill the EXPECTED OUTPUT, your script should:
   - Print the final answer to the console (e.g. optimal locations, statistics, etc.)
   - Save any required figures to {output_dir}/ with descriptive filenames.

FORRMAT:
Output a JSON object with the following fields:
    "response": "ACCEPT" or "FAILED",
    "script": "The Python script path or content",
    "answer": "any textual answer to the task"

Return ONLY the JSON object — no explanation, no markdown fences.
"""


# ── helpers ────────────────────────────────────────────────────────────────────

# Global flag to track if we should stop
_should_stop = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _should_stop
    _should_stop = True
    print("\nInterrupt received. Stopping after current task...")
    # Do NOT raise here — communicate() is a blocking syscall and Python 3.5+
    # (PEP 475) retries EINTR, so a raise inside the signal handler is silently
    # swallowed until communicate() returns on its own.  Instead we set the flag
    # and kill the subprocess from the watcher thread (see run_claude).

def build_prompt(task_text: str, used_files: list, data_dir: Path, expected_output: dict, output_dir: Path) -> str:
    file_list = "\n".join(f"  - {f}" for f in used_files)
    return PROMPT_TEMPLATE.format(
        task_text=task_text,
        data_dir=str(data_dir),
        file_list=file_list,
        expected_output=json.dumps(expected_output),
        output_dir=str(output_dir),
    )


def run_claude(task_text: str, used_files: list, data_dir: Path, expected_output: dict,
               task_dir: Path, timeout: int) -> dict:
    """Call `claude -p` with --output-format json and return the parsed solution dict.

    When --json-schema is used, Claude emits a StructuredOutput tool_use block
    whose input IS the payload we want.  We capture it from the stream-json
    events; if that fails we fall back to regex-extracting JSON from plain text.

    The generated script is saved to {task_dir}/solution.py.

    Returns dict with 'response', 'script', 'trajectory', and optionally 'usage' (token counts).
    """
    prompt = build_prompt(task_text, used_files, data_dir, expected_output, task_dir)

    # ── primary: stream-json so we can pull the StructuredOutput tool input ──
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
            cwd=str(task_dir),
            # New session so the claude subprocess does NOT receive terminal
            # SIGINT directly — we kill it explicitly when _should_stop is set.
            start_new_session=True,
        )

        # Watcher thread: kill the subprocess as soon as Ctrl+C is detected.
        # Capture proc in the default argument so the closure holds a stable
        # reference even after the outer `proc` variable is set to None.
        import threading, time
        def _watcher(p=proc):
            while p.poll() is None:
                if _should_stop:
                    p.kill()
                    return
                time.sleep(0.2)
        t = threading.Thread(target=_watcher, daemon=True)
        t.start()

        stdout, stderr = proc.communicate(timeout=timeout)
        proc = None  # Mark as completed successfully
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            stdout, stderr = proc.communicate()
        # Parse whatever output was collected before timeout
        trajectory = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                trajectory.append(event)
            except json.JSONDecodeError:
                continue
        return {"response": "FAILED", "script": "timeout", "answer": "", "trajectory": trajectory}
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        return {"response": "FAILED", "script": f"subprocess error: {exc}", "answer": ""}

    if _should_stop:
        raise KeyboardInterrupt("Interrupted by user")

    if proc and proc.returncode != 0:
        err = (stderr or stdout or "unknown error")[:400]
        return {"response": "FAILED", "script": f"claude exit {proc.returncode}: {err}", "answer": ""}

    # Parse the NDJSON stream looking for StructuredOutput tool_use input
    # NOTE: assistant event arrives BEFORE result event, so collect both
    # before returning to ensure usage is populated.
    usage = None
    structured_output = None
    trajectory = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            trajectory.append(event)
        except json.JSONDecodeError:
            continue
        # Track token usage from result event
        if event.get("type") == "result":
            usage = event.get("usage")
        # assistant message containing tool_use blocks
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                    inp = block.get("input", {})
                    structured_output = {
                        "response": inp.get("response", "FAILED"),
                        "script":   inp.get("script", ""),
                        "answer":   inp.get("answer", ""),
                        "trajectory": trajectory,
                    }
                    # Validate script content
                    if structured_output["script"] and len(structured_output["script"]) < 50:
                        # Likely a filename instead of actual script, mark as FAILED
                        structured_output["response"] = "FAILED"
                        structured_output["script"] = f"Invalid script output: {structured_output['script']}"

    if structured_output is not None:
        if usage:
            structured_output["usage"] = usage
        return structured_output

    # ── fallback: try to extract any JSON object from the raw output ──────────
    m = re.search(r'\{[\s\S]*?\}', stdout or "")
    if m:
        try:
            obj = json.loads(m.group())
            result = {
                "response": obj.get("response", "FAILED"),
                "script":   obj.get("script", ""),
                "answer":   obj.get("answer", ""),
                "trajectory": trajectory,
            }
            if usage:
                result["usage"] = usage
            return result
        except json.JSONDecodeError:
            pass

    result = {
        "response": "FAILED",
        "script": f"could not parse output: {stdout[:300]}",
        "answer": "",
        "trajectory": trajectory,
    }
    if usage:
        result["usage"] = usage
    return result


def load_source() -> dict:
    with open(DATASETS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_output(output_path: Path) -> dict:
    """Load existing output file, or return an empty structure."""
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    return {"metadata": {}, "tasks": [], "generation_stats": {}}


def save(data: dict, output_path: Path) -> None:
    """Atomic write: write to .tmp then rename."""
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(output_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Batch-generate GIS solutions via Claude Code CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
 
    parser.add_argument("--max",     type=int, default=None, help="Max number of tasks to process")
    parser.add_argument("--timeout", type=int, default=120,  help="Seconds per task (default: 120)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, but re-run FAILED tasks (only skip ACCEPT tasks)",
    )
    parser.add_argument(
        "--resume-skip-failed",
        action="store_true",
        dest="resume_skip_failed",
        help="Resume and skip FAILED tasks too (only skip already-written tasks)",
    )
    parser.add_argument("--output",  type=Path, default=None, help=f"Output JSON path (default: {{runs_dir}}/datasets_solutions.json)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help=f"Run output directory (default: {DEFAULT_RUNS_DIR})")
    args = parser.parse_args()

    # ── determine run directory (resume or new) ───────────────────────────────
    run_dir = args.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Default output goes into the run directory
    if args.output is None:
        args.output = run_dir / "datasets_solutions.json"

    # In resume mode with default output, look for the latest run's output
    if args.resume and args.output == run_dir / "datasets_solutions.json":
        if args.runs_dir.exists():
            runs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
            if runs:
                latest_run = runs[-1]
                latest_output = latest_run / "datasets_solutions.json"
                if latest_output.exists():
                    run_dir = latest_run
                    args.output = latest_output
                    print(f"Resume mode: using latest run directory: {run_dir}\n")

    # Now create the run directory (only if it doesn't already exist from resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}\n")

    # ── load source ───────────────────────────────────────────────────────────
    source = load_source()
    all_tasks = source["tasks"]

    # ── init output (fresh or resume) ─────────────────────────────────────────
    if args.resume and args.output.exists():
        output = load_output(args.output)
        # Build a map of task_ID -> solution status
        task_status = {}
        for t in output["tasks"]:
            if "generated_solution" in t:
                status = t["generated_solution"].get("response", "")
            else:
                status = None
            task_status[t["task_ID"]] = status

        # Default resume: skip only ACCEPT tasks, re-run FAILED
        if args.resume_skip_failed:
            # Skip all already-written tasks including FAILED
            written_ids = {t["task_ID"] for t in output["tasks"]}
            print(f"Resume mode (skip failed): {len(written_ids)} already written, "
                  f"{len(all_tasks) - len(written_ids)} remaining.")
        else:
            # Only skip ACCEPT tasks, re-run FAILED
            written_ids = {tid for tid, status in task_status.items() if status == "ACCEPT"}
            failed_ids = {tid for tid, status in task_status.items() if status == "FAILED"}
            print(f"Resume mode: {len(written_ids)} ACCEPT tasks skipped, "
                  f"{len(failed_ids)} FAILED tasks will be re-run, "
                  f"{len(all_tasks) - len(written_ids) - len(failed_ids)} remaining.")
    else:
        output = {"metadata": source.get("metadata", {}), "tasks": [], "generation_stats": {}}
        written_ids = set()

    # ── apply --max before filtering ──────────────────────────────────────────
    if args.max:
        all_tasks = all_tasks[:args.max]

    pending = [t for t in all_tasks if t["task_ID"] not in written_ids]

    if not pending:
        print("No tasks to process.")
        return

    # Build a map for updating existing tasks
    output_task_map = {t["task_ID"]: t for t in output["tasks"]}

    # ── stats ─────────────────────────────────────────────────────────────────
    n_accept = n_failed = n_skipped = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    print(f"Processing {len(pending)} tasks  |  output → {args.output}\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    with tqdm(
        pending,
        unit="task",
        dynamic_ncols=True,
        postfix={"accept": 0, "failed": 0, "skip": 0},
    ) as pbar:
        for task in pbar:
            task_id   = task["task_ID"]
            task_text = task["task_text"]

            pbar.set_description(f"[{task_id[-8:]}] {task_text[:55]}", refresh=False)

            source = task.get("source", "")
            data_dir = DATA_BASE_DIR / source

            # Create task-specific directory
            task_dir = run_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            solution = run_claude(task_text, task.get("used_dataset", []), data_dir, task.get("Expected_Output", {}), task_dir, timeout=args.timeout)

            # Save the generated script to task directory
            # script_content = solution.get("script", "")
            # if script_content and script_content != "timeout" and not script_content.startswith("subprocess error:") and not script_content.startswith("could not parse output:"):
            #     script_path = task_dir / "solution.py"
            #     with open(script_path, "w", encoding="utf-8") as f:
            #         f.write(script_content)
            #     solution["script_file"] = str(script_path)

            # Accumulate token usage before removing it
            if "usage" in solution:
                usage = solution["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens += input_tokens + output_tokens
                del solution["usage"]

            # Keep only required fields
            solution["used_dataset"] = task.get("used_dataset", [])
            task_out = dict(task)
            task_out["generated_solution"] = solution

            # Update or append task in output
            if task_id in output_task_map:
                # Update existing task (resume mode)
                idx = next(i for i, t in enumerate(output["tasks"]) if t["task_ID"] == task_id)
                output["tasks"][idx] = task_out
                output_task_map[task_id] = task_out
            else:
                # New task, append to output
                output["tasks"].append(task_out)
                output_task_map[task_id] = task_out

            if solution["response"] == "ACCEPT":
                n_accept += 1
            else:
                n_failed += 1

            # update stats
            output.setdefault("generation_stats", {}).update({
                "processed":       n_accept + n_failed,
                "accept":          n_accept,
                "failed":          n_failed,
                "skipped":         n_skipped,
                "input_tokens":    total_input_tokens,
                "output_tokens":   total_output_tokens,
                "total_tokens":    total_tokens,
                "last_updated":    datetime.now().isoformat(),
            })

            save(output, args.output)
            pbar.set_postfix(
                accept=n_accept,
                failed=n_failed,
                skip=n_skipped,
                tokens=total_tokens,
                refresh=True
            )

    print(f"\nFinished.  ACCEPT={n_accept}  FAILED={n_failed}  SKIP={n_skipped}")
    print(f"Token usage: Input={total_input_tokens:,}  Output={total_output_tokens:,}  Total={total_tokens:,}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
