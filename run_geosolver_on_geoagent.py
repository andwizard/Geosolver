"""Batch run GeoSolver agent against the GeoAgent task dataset.

Mirrors the directory/output conventions of run_agent.py:
  - Timestamped output under Output/runs/
  - Per-task: {output_path}/task_{id}/
      task.log         – full console output (tee'd)
      workspace/       – agent scratch space (contains solution.py, trajectory.txt)
  - Aggregate: {output_path}/generated_solutions.json  (with checkpoint/resume)
"""

import asyncio
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config
from Agents.build_geoagent import build_geoagent, geoagent_invoke
from Agents.build_geosolver import (
    build_geosolver,
    aclose_geosolver_client,
)
from Agents.tools.graph import reset_graph
from Agents.tools.verify_output import set_workspace_dir

# ---------------------------------------------------------------------------
# Module-level globals that build_geosolver tools read from.
# We inject these directly (same pattern as geo_asolve in build_geosolver.py).
# ---------------------------------------------------------------------------
import Agents.build_geosolver as _geosolver_mod


# ── Stdout tee ─────────────────────────────────────────────────────────────────

class _TeeWriter:
    """Write to both the real stdout and a log file simultaneously."""

    def __init__(self, log_path: str, console_verbose: bool = True):
        self._real_stdout = sys.stdout
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._console_verbose = console_verbose

    def write(self, data):
        self._log_file.write(data)
        if self._console_verbose:
            self._real_stdout.write(data)

    def flush(self):
        self._log_file.flush()
        if self._console_verbose:
            self._real_stdout.flush()

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._real_stdout, name)


# ── Task loading ───────────────────────────────────────────────────────────────

def load_tasks():
    """Load tasks from Data/GeoAgent/datasets.json (flat list format)."""
    tasks_path = Path(__file__).parent / "Data" / "GeoAgent" / "datasets.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Data path resolution ───────────────────────────────────────────────────────

def resolve_data_files(data_path, base_dir: str) -> list:
    """Resolve a GeoAgent-format data_path entry to absolute file paths.

    data_path can be:
      - A string: directory path → all files under it; file path → [that file]
      - A list of strings: each resolved recursively.
    """
    if isinstance(data_path, list):
        files = []
        for p in data_path:
            files.extend(resolve_data_files(p, base_dir))
        return files

    full = Path(base_dir) / data_path
    if full.is_file():
        return [str(full)]
    if full.is_dir():
        return sorted(str(f) for f in full.iterdir() if f.is_file())
    # Try with .shp extension (common shorthand in GeoAgent tasks)
    shp = Path(base_dir) / (data_path + ".shp")
    if shp.is_file():
        return [str(shp)]
    return []


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint(json_result_path: str) -> tuple[list, set, int, int]:
    """Load existing checkpoint data.

    Returns (results_list, processed_ids, failed_count, timeout_count).
    """
    results: list = []
    processed_ids: set = set()
    failed_count = 0
    timeout_count = 0

    if not os.path.exists(json_result_path):
        return results, processed_ids, failed_count, timeout_count

    print(f"Found existing output: {json_result_path}, resuming...")
    try:
        with open(json_result_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        for entry in existing:
            tid = entry["task_id"]
            if entry.get("status") == "completed":
                processed_ids.add(tid)
                results.append(entry)
            elif entry.get("status") == "failed":
                failed_count += 1
            elif entry.get("status") == "timeout":
                timeout_count += 1

        print(f"Resumed {len(processed_ids)} completed tasks, "
              f"{failed_count} failed, {timeout_count} timeout.")
    except json.JSONDecodeError:
        print("Existing file is corrupt or empty, starting fresh.")

    return results, processed_ids, failed_count, timeout_count


def save_checkpoint(results: list, json_result_path: str) -> None:
    """Persist current progress."""
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


# ── Single task processing ─────────────────────────────────────────────────────

async def process_task(agent, method: str, task: dict, data_base_dir: str,
                       task_folder: str, timeout_seconds: int) -> dict:
    """Process a single GeoAgent-format task.

    Supports two methods:
      - geosolver: local LangGraph agent (streaming, trajectory capture)
      - geoagent:  remote HTTP API (submit + poll)
    """
    task_id = task["id"]
    task_text = task["task"]

    os.makedirs(task_folder, exist_ok=True)

    # Tee stdout to task.log
    log_path = os.path.join(task_folder, "task.log")
    tee = _TeeWriter(log_path, console_verbose=True)
    real_stdout = sys.stdout
    sys.stdout = tee

    try:
        print(f"Task: {task_id}")
        print(task_text)

        # ── geoagent path (remote API) ──────────────────────────────────
        if method == "geoagent":
            # Adapt GeoAgent task format: geoagent_invoke expects task_ID
            task["task_ID"] = str(task_id)
            task["task_text"] = task_text
            response = geoagent_invoke(agent, task, data_base_dir, task_folder, timeout_seconds)
            status = response.get("status", "")
            if status == "ACCEPT":
                result_status = "completed"
            elif status == "FAILED":
                result_status = "failed"
            else:
                result_status = "timeout"
            return {
                "task_id": task_id,
                "task_text": task_text,
                "status": result_status,
                "answer": response.get("answer", ""),
                "script": "",
                "trajectory": "",
            }

        # ── geosolver path (local agent) ────────────────────────────────
        data_path = task.get("data_path", "")
        workspace_dir = os.path.join(task_folder, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        # Set globals that GeoSolver tools read
        set_workspace_dir(workspace_dir)
        reset_graph()

        file_list = resolve_data_files(data_path, data_base_dir)

        # Inject task context so explore_data / plan_and_scaffold / generate_code
        # can read it without relying on the LLM to pass values correctly.
        _geosolver_mod._task_context = {
            "data_path": str(Path(data_base_dir) / (data_path if isinstance(data_path, str) else "")),
            "task_description": task_text,
            "expected_output": "See task description above.",
            "output_dir": workspace_dir,
            "file_list": "\n".join(file_list),
        }
        _geosolver_mod._task_output = {}
        _geosolver_mod._exploration_findings = ""
        _geosolver_mod._exploration_metadata = {}
        _geosolver_mod._script_dir = workspace_dir

        # Build prompt
        file_list_str = "\n".join(f"  - {f}" for f in file_list) if file_list else "  (none)"
        prompt = f"""Task: {task_text}

Data files:
{file_list_str}

Output directory: {workspace_dir}

Please complete this task using the standard GeoSolver workflow (explore, plan, scaffold, code, verify)."""

        trajectory_buf = io.StringIO()

        class _TeeStream(io.StringIO):
            """Capture trajectory while also writing to the real (tee'd) stdout."""
            def __init__(self, tee_writer):
                super().__init__()
                self._tee = tee_writer

            def write(self, s):
                self._tee.write(s)
                return super().write(s)

        tee_stream = _TeeStream(tee)

        original_dir = os.getcwd()
        os.chdir(task_folder)

        try:
            input_message = {"role": "user", "content": prompt}

            # Redirect stdout through TeeStream for trajectory capture during
            # agent.pretty_print() calls — EVERYTHING goes to both task.log
            # and trajectory buffer.
            sys.stdout = tee_stream

            try:
                stream = agent.astream(
                    {"messages": [input_message]},
                    stream_mode="updates",
                )
                async for step in stream:
                    for _, update in step.items():
                        if update and (messages := update.get("messages")) and isinstance(messages, list):
                            for message in messages:
                                # Skip verify_output messages (verbose base64 data)
                                if hasattr(message, 'name') and message.name == 'verify_output':
                                    continue
                                message.pretty_print()
            except asyncio.CancelledError:
                await stream.aclose()
                raise
        except asyncio.TimeoutError:
            print(f"Task timed out after {timeout_seconds}s")
            await asyncio.sleep(2)
            return {
                "task_id": task_id,
                "task_text": task_text,
                "status": "timeout",
                "answer": "",
                "trajectory": trajectory_buf.getvalue(),
            }
        except Exception as e:
            print(f"Task failed: {e}")
            return {
                "task_id": task_id,
                "task_text": task_text,
                "status": "failed",
                "answer": str(e),
                "trajectory": trajectory_buf.getvalue(),
            }
        finally:
            os.chdir(original_dir)
            sys.stdout = tee  # restore tee for final prints

        trajectory = trajectory_buf.getvalue()

        # Read solution.py if it was written
        script = ""
        solution_path = os.path.join(workspace_dir, "solution.py")
        if os.path.exists(solution_path):
            with open(solution_path, "r", encoding="utf-8") as f:
                script = f.read()

        # Extract answer from send_output if available
        answer = _geosolver_mod._task_output.get("answer", "")

        print("Status: completed")
        print(f"Answer: {answer[:200] if answer else '(empty)'}")

        return {
            "task_id": task_id,
            "task_text": task_text,
            "status": "completed",
            "answer": answer[:5000] if answer else "",
            "script": script,
            "trajectory": trajectory,
        }
    finally:
        sys.stdout = real_stdout
        tee.close()


# ── Main orchestrator ──────────────────────────────────────────────────────────

async def async_main(method: str = "geosolver"):
    # Data directory for GeoAgent dataset
    data_base_dir = str(Path(__file__).parent / "Data" / "GeoAgent" / "datasets")
    timeout_seconds = 3600

    # Timestamped output directory
    now = datetime.now()
    dir_name = now.strftime("%Y%m%d_%H%M%S")
    output_path = str(config.RUNS_DIR / f"{dir_name}_{method}_on_geoagent")
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    print(f"Method: {method}\n")

    # Checkpoint
    json_result_path = os.path.join(output_path, "generated_solutions.json")
    results, processed_ids, failed_count, timeout_count = \
        load_checkpoint(json_result_path)

    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from GeoAgent dataset")

    agent = None

    pbar = tqdm(tasks, desc="Processing tasks", unit="task")

    for task in pbar:
        task_id = task["id"]

        if task_id in processed_ids:
            pbar.set_postfix({
                "status": "skipped",
                "failed": failed_count,
                "timeout": timeout_count,
            })
            continue

        # Fresh agent per task (isolated connection pool)
        await aclose_geosolver_client()
        if method == "geoagent":
            agent = build_geoagent(config.GEOAGENT_BASE_URL)
        else:
            agent = build_geosolver(config.MODEL_NAME)

        task_folder = os.path.join(output_path, f"task_{task_id}")

        # geoagent is a sync blocking call with its own timeout; skip outer
        # asyncio timeout wrapping (same pattern as run_agent.py).
        if method == "geoagent":
            result = await process_task(agent, method, task, data_base_dir, task_folder, timeout_seconds)
        else:
            try:
                result = await asyncio.wait_for(
                    process_task(agent, method, task, data_base_dir, task_folder, timeout_seconds),
                    timeout=timeout_seconds + 60,  # extra margin over inner timeout
                )
            except asyncio.TimeoutError:
                print(f"Task {task_id} outer timeout")
                result = {
                    "task_id": task_id,
                    "task_text": task["task"],
                    "status": "timeout",
                    "answer": "",
                    "trajectory": "",
                }

        if result["status"] == "failed":
            failed_count += 1
        elif result["status"] == "timeout":
            timeout_count += 1

        results.append(result)
        processed_ids.add(task_id)

        pbar.set_postfix({
            "status": result["status"],
            "failed": failed_count,
            "timeout": timeout_count,
        })

        save_checkpoint(results, json_result_path)

    pbar.close()

    await aclose_geosolver_client()

    total = len(tasks)
    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"\nProcessing complete!")
    print(f"Total: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed_count}")
    print(f"Timeout: {timeout_count}")
    print(f"Results saved to {json_result_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run geosolver/geoagent on GeoAgent task dataset")
    parser.add_argument("--method", choices=["geosolver", "geoagent"], default="geosolver",
                        help="Agent method (default: geosolver)")
    args = parser.parse_args()
    asyncio.run(async_main(method=args.method))