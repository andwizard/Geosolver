"""Batch task execution entry point.

Runs geospatial analysis agents (CodeAct, deepagent, geosolver, geoagent, claude_code)
against a task dataset, producing generated solutions with checkpoint/resume support.
"""

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import os
import sys
from datetime import datetime

from tqdm import tqdm

from Agents.build_ClaudeCode import build_claude_code, claude_code_ainvoke
from Agents.build_deepagent import build_deepagent, deepagent_ainvoke
from Agents.build_geoagent import build_geoagent, geoagent_invoke
from Agents.build_geosolver import build_geosolver, geo_asolve, aclose_geosolver_client
from Agents.build_geosolver_without_contract import (
    build_geosolver_without_contract,
    geo_asolve_without_contract,
    aclose_geosolver_without_contract_client,
)
from Agents.build_codeact import build_codeact, codeact_ainvoke
from Agents.build_geosolver_without_dfg import (
    build_geosolver_without_dfg,
    geosolver_without_dfg_ainvoke,
)
import config


# ── Agent factory ────────────────────────────────────────────────────────────

async def build_agent(method: str):
    """Build and return the agent for the given method."""
    if method == "CodeAct":
        return await build_codeact(config.MODEL_NAME)
    elif method == "Geosolver_without_DFG":
        return await build_geosolver_without_dfg(config.MODEL_NAME)
    elif method == "deepagent":
        return build_deepagent(config.MODEL_NAME, config.DEEPAGENT_ROOT_DIR, config.DEEPAGENT_ENV_DICT)
    elif method == "geosolver":
        return build_geosolver(config.MODEL_NAME)
    elif method == "geosolver_without_contract":
        return build_geosolver_without_contract(config.MODEL_NAME)
    elif method == "geoagent":
        return build_geoagent(config.GEOAGENT_BASE_URL)
    elif method == "claude_code":
        return build_claude_code()
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Task loading ─────────────────────────────────────────────────────────────

def load_tasks(task_path: str) -> dict:
    """Load task dataset from JSON file."""
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(json_result_path: str) -> tuple[dict, set, int, int]:
    """Load existing checkpoint data.

    Returns (generated_data, processed_questions, failed_count, timeout_count).
    If no checkpoint exists, returns an empty initial state.
    """
    generated_data: dict = {"metadata": {}, "tasks": []}
    processed_questions: set = set()
    failed_count = 0
    timeout_count = 0

    if not os.path.exists(json_result_path):
        return generated_data, processed_questions, failed_count, timeout_count

    print(f"Found existing output: {json_result_path}, resuming...")
    try:
        with open(json_result_path, "r", encoding="utf-8") as f:
            existing_content = json.load(f)

        if "metadata" in existing_content:
            generated_data["metadata"] = existing_content["metadata"]

        if "tasks" in existing_content:
            generated_data["tasks"] = []
            for t in existing_content["tasks"]:
                q_text = t["task_text"]
                if t["generated_solution"]["response"] != "FAILED":
                    processed_questions.add(q_text)
                    generated_data["tasks"].append(t)

        failed_count = existing_content.get("metadata", {}).get("failed_count", 0)
        timeout_count = existing_content.get("metadata", {}).get("timeout_count", 0)
        generated_data["metadata"]["failed_count"] = 0

        print(f"Resumed {len(processed_questions)} tasks, will retry {failed_count} failed tasks.")
    except json.JSONDecodeError:
        print("Existing file is corrupt or empty, starting fresh.")

    return generated_data, processed_questions, failed_count, timeout_count


def save_checkpoint(generated_data: dict, json_result_path: str,
                    failed_count: int, timeout_count: int) -> None:
    """Persist current progress to the checkpoint file."""
    generated_data["metadata"]["failed_count"] = failed_count
    generated_data["metadata"]["timeout_count"] = timeout_count
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=4, ensure_ascii=False)


# ── Agent invocation ─────────────────────────────────────────────────────────

async def invoke_agent(agent, method: str, task: dict,
                       data_root: str, task_folder: str,
                       timeout_seconds: int, verbose: bool = True) -> dict:
    """Invoke the appropriate agent method with timeout handling.

    Returns the agent's response dict, or a dict with status TIMEOUT/FAILED.
    """

    try:
        with contextlib.redirect_stdout(sys.stdout if verbose else None):
            if method == "CodeAct":
                response = await asyncio.wait_for(
                    codeact_ainvoke(agent, task, data_root, task_folder),
                    timeout=timeout_seconds,
                )
            elif method == "Geosolver_without_DFG":
                response = await asyncio.wait_for(
                    geosolver_without_dfg_ainvoke(agent, task, data_root, task_folder),
                    timeout=timeout_seconds,
                )
            elif method == "deepagent":
                response = await asyncio.wait_for(
                    deepagent_ainvoke(agent, task, data_root, task_folder),
                    timeout=timeout_seconds,
                )
            elif method == "geosolver":
                response = await asyncio.wait_for(
                    geo_asolve(agent, task, data_root, task_folder),
                    timeout=timeout_seconds,
                )
            elif method == "geosolver_without_contract":
                response = await asyncio.wait_for(
                    geo_asolve_without_contract(agent, task, data_root, task_folder),
                    timeout=timeout_seconds,
                )
            elif method == "geoagent":
                # geoagent is synchronous and enforces its own limit server-side,
                # so pass our configured timeout as its time_limit (defaults to
                # 600s otherwise) to keep it consistent with the other methods.
                response = geoagent_invoke(agent, task, data_root, os.getcwd(),
                                           time_limit=timeout_seconds)
            elif method == "claude_code":
                response = await claude_code_ainvoke(
                    task, data_root, task_folder, timeout_seconds,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
    except asyncio.TimeoutError:
        print(f"Task timed out after {timeout_seconds}s")
        # Yield control to let cancelled coroutines clean up async generators
        # and release HTTP connections before the next task starts.
        await asyncio.sleep(2)
        response = {"status": "TIMEOUT"}
    except Exception as e:
        print(f"Task failed: {e}")
        response = {"status": "FAILED", "answer": str(e)}

    return response


# ── Stdout tee (captures console output to a log file while still printing) ──

class _TeeWriter:
    """Write to both the real stdout and a log file simultaneously.

    When *console_verbose* is False, output is still written to the log file
    but suppressed on the console — so the log always captures the full trace.
    """

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

    # Forward any attribute access to the real stdout so that code
    # checking sys.stdout.isatty(), .encoding, etc. still works.
    def __getattr__(self, name):
        return getattr(self._real_stdout, name)


# ── Single task processing ───────────────────────────────────────────────────

async def process_task(agent, method: str, task: dict,
                       data_root: str, output_path: str,
                       timeout_seconds: int, verbose: bool = True) -> dict:
    """Process a single task end-to-end and write outputs to disk.

    All console output during the task (verbose agent messages, prints, etc.)
    is tee'd to a ``task.log`` file inside the task folder.

    Returns the generated_solution dict.
    """
    task_id = task["task_ID"]
    task_folder = os.path.join(output_path, task_id)
    os.makedirs(task_folder, exist_ok=True)

    # Tee all stdout to a per-task log file.
    # Log file ALWAYS captures full output; console display follows verbose flag.
    log_path = os.path.join(task_folder, "task.log")
    tee = _TeeWriter(log_path, console_verbose=verbose)
    real_stdout = sys.stdout
    sys.stdout = tee

    try:
        print(f"Task: {task_id}")
        print(task["task_text"])

        original_dir = os.getcwd()
        os.chdir(task_folder)

        try:
            # Always verbose=True so agent output flows through the tee.
            # The tee itself controls whether to show it on the console.
            response = await invoke_agent(agent, method, task, data_root, task_folder, timeout_seconds, verbose=True)
        finally:
            os.chdir(original_dir)

        script = response.get("script", "")
        try_counts = response.get("try_counts", 0)
        trajectory = response.get("trajectory", "")

        generated_solution = {
            "response": response.get("status", "FAILED"),
            "answer": response.get("answer", ""),
        }

        print("Response:", generated_solution["response"])
        print("Answer:", generated_solution["answer"])

        # Persist solution outputs
        with open(os.path.join(task_folder, "solution.py"), "w", encoding="utf-8") as f:
            f.write(script)

        with open(os.path.join(task_folder, "trajectory.txt"), "w", encoding="utf-8") as f:
            f.write(f"Try Counts: {try_counts}\n")
            f.write(trajectory)

        return generated_solution
    finally:
        sys.stdout = real_stdout
        tee.close()


# ── Single task with retries (self-contained: builds + tears down its agent) ──

async def process_task_with_retries(method: str, task: dict,
                                    data_root: str, output_path: str,
                                    timeout_seconds: int, verbose: bool,
                                    max_retries: int) -> dict:
    """Build an agent, run one task with retries, and tear the agent down.

    Fully self-contained so it can run either inline (serial mode) or inside
    an isolated worker process (parallel mode). Each call gets its own agent
    and connection pool; a fresh agent is rebuilt before every retry to clear
    any state corrupted by a timeout/failure.

    Returns the generated_solution dict.
    """
    # Fresh agent (and fresh connection pool). Close any pool left over from a
    # prior task in this same process so half-open connections can't leak.
    await aclose_geosolver_client()
    await aclose_geosolver_without_contract_client()
    agent = await build_agent(method)

    generated_solution = None
    for attempt in range(1 + max_retries):
        if attempt > 0:
            print(f"\nRetrying {task['task_ID']} (attempt {attempt + 1}/{1 + max_retries})...")

        generated_solution = await process_task(
            agent, method, task, data_root,
            output_path, timeout_seconds, verbose,
        )

        # Success — no need to retry
        if generated_solution["response"] not in ("FAILED", "TIMEOUT"):
            break

        # Timeout/failure can corrupt async generators and leak HTTP
        # connections. Always rebuild the agent before retrying or before
        # moving on to the next task.
        print(f"Rebuilding agent after {generated_solution['response']}...")
        await aclose_geosolver_client()
        await aclose_geosolver_without_contract_client()
        agent = await build_agent(method)

    # Release this task's connection pool.
    await aclose_geosolver_client()
    await aclose_geosolver_without_contract_client()

    return generated_solution


def _worker(payload: tuple) -> tuple:
    """Process-pool entry point: run a single task in an isolated process.

    Must be a module-level function so the ``spawn`` start method (macOS
    default) can import and call it. Running each task in its own process
    naturally isolates the agents' module-global state, the per-task
    ``os.chdir``, and the global ``sys.stdout`` replacement.

    ``payload`` is (task_index, method, task, data_root, output_path,
    timeout_seconds, verbose, max_retries) — all picklable.

    Returns (task_index, task, generated_solution).
    """
    (task_index, method, task, data_root, output_path,
     timeout_seconds, verbose, max_retries) = payload

    generated_solution = asyncio.run(process_task_with_retries(
        method, task, data_root, output_path,
        timeout_seconds, verbose, max_retries,
    ))
    return task_index, task, generated_solution


# ── Main orchestrator ────────────────────────────────────────────────────────

async def run_agent(method: str, task_path: str, output_path: str,
                    verbose: bool = True, max_retries: int = 1,
                    workers: int = 1, timeout_seconds: int | None = None) -> None:
    """Run all tasks in the dataset through the chosen agent method.

    Uses a single event loop for all tasks so that agent internal resources
    (LLM client connection pools, etc.) remain valid across tasks.  A timeout
    on one task only cancels that single coroutine — the event loop survives.

    Args:
        max_retries: Number of times to retry a task that times out or fails.
            On retry the agent is rebuilt to clear any corrupted state.
        workers: Number of parallel worker processes. 1 (default) runs tasks
            serially in this process, identical to the original behavior. >1
            runs each task in an isolated process so the agents' module-global
            state, per-task os.chdir, and global stdout swap don't collide.
        timeout_seconds: Per-task time limit. Falls back to
            config.EVAL_TIMEOUT_SECONDS when None.
    """
    if timeout_seconds is None:
        timeout_seconds = config.EVAL_TIMEOUT_SECONDS
    task_path = os.path.abspath(task_path)
    output_path = os.path.abspath(output_path)

    json_result_path = os.path.join(output_path, "generated_solutions.json")
    generated_data, processed_questions, failed_count, timeout_count = \
        load_checkpoint(json_result_path)

    data = load_tasks(task_path)
    generated_data["metadata"] = data["metadata"]
    tasks = data["tasks"]

    # Original dataset position for each task, used to keep the output ordered
    # by the dataset even when tasks finish out of order in parallel mode.
    task_order = {t["task_text"]: i for i, t in enumerate(tasks)}

    # Tasks not yet completed in a previous run.
    pending = [t for t in tasks if t["task_text"] not in processed_questions]
    print(f"{len(pending)} task(s) to run, {len(processed_questions)} already done.")

    def record(task: dict, generated_solution: dict) -> None:
        """Fold one finished task into the accumulated results + checkpoint."""
        nonlocal failed_count, timeout_count
        if generated_solution["response"] == "FAILED":
            failed_count += 1
        elif generated_solution["response"] == "TIMEOUT":
            timeout_count += 1

        task["generated_solution"] = generated_solution
        generated_data["tasks"].append(task)
        processed_questions.add(task["task_text"])
        save_checkpoint(generated_data, json_result_path, failed_count, timeout_count)

    if workers <= 1:
        # ── Serial path (original behavior) ──
        pbar = tqdm(pending, desc="Processing tasks", unit="task")
        for task in pbar:
            generated_solution = await process_task_with_retries(
                method, task, str(config.DATA_ROOT), output_path,
                timeout_seconds, verbose, max_retries,
            )
            record(task, generated_solution)
            pbar.set_postfix({
                "status": "done",
                "failed": failed_count,
                "timeout": timeout_count,
            })
        pbar.close()
    else:
        # ── Parallel path: one isolated process per task ──
        # Console output is suppressed to avoid interleaving; each task's full
        # trace is still captured in its own task.log by the tee.
        payloads = [
            (i, method, task, str(config.DATA_ROOT), output_path,
             timeout_seconds, False, max_retries)
            for i, task in enumerate(pending)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            pbar = tqdm(total=len(futures), desc="Processing tasks", unit="task")
            for fut in concurrent.futures.as_completed(futures):
                _, task, generated_solution = fut.result()
                record(task, generated_solution)
                pbar.update(1)
                pbar.set_postfix({
                    "status": "done",
                    "failed": failed_count,
                    "timeout": timeout_count,
                })
            pbar.close()

    # Restore dataset ordering (parallel completion is out of order; already
    # loaded checkpoint tasks are ordered too). Unknown tasks sort to the end.
    generated_data["tasks"].sort(
        key=lambda t: task_order.get(t["task_text"], len(task_order))
    )
    save_checkpoint(generated_data, json_result_path, failed_count, timeout_count)

    print(f"\nProcessing complete!")
    print(f"Total failed: {failed_count}")
    print(f"Total timeout: {timeout_count}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run geospatial analysis agents on task datasets")
    parser.add_argument(
        "--method",
        choices=["CodeAct", "Geosolver_without_DFG", "deepagent", "geosolver", "geosolver_without_contract", "geoagent", "claude_code"],
        required=True,
        help="Agent method to use",
    )
    parser.add_argument("--task_path", required=True, help="Path to task JSON file")
    parser.add_argument("--output_path", default="", help="Output directory path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print agent internal messages (default: on)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Suppress agent internal messages")
    parser.add_argument("--retry", type=int, default=1, help="Max retries per task on timeout/failure (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes (default: 1 = serial)")
    parser.add_argument("--timeout", type=int, default=None, help="Per-task time limit in seconds (default: config EVAL_TIMEOUT_SECONDS)")

    args = parser.parse_args()

    if args.output_path == "":
        now = datetime.now()
        dir_name = now.strftime("%Y%m%d_%H%M%S")
        output_path = str(config.RUNS_DIR / f"{dir_name}_{args.method}")
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = args.output_path

    asyncio.run(run_agent(
        method=args.method, task_path=args.task_path,
        output_path=output_path, verbose=args.verbose,
        max_retries=args.retry, workers=args.workers,
        timeout_seconds=args.timeout,
    ))
