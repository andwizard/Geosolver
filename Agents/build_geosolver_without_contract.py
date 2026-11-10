"""
GeoSolver without contract — based on 79e3df2 architecture.

Simple pipeline: graph_plan → inspect (explore DAG) → code_generate → send_output
No contract generation, no explore_data separation, no verify_output.
"""

import asyncio
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Literal

import httpx

from deepagents.backends import LocalShellBackend
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from Agents.tools.graph import from_dict, inspect as graph_inspect, reset_graph
from Agents.tools.sandbox_env import execute_code
from Agents.prompts.geosolver_prompt_without_contract import (
    GEOSOLVER_PROMPT,
    GRAPH_PLANNER_PROMPT,
    CODE_GENERATOR_PROMPT,
)
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from config import LLM_REQUEST_TIMEOUT

# ============================================================================
# Global state
# ============================================================================

_task_output: dict = {}
_geosolver_http_client: httpx.AsyncClient | None = None


async def aclose_geosolver_without_contract_client() -> None:
    """Close the httpx client from the previous build before rebuilding."""
    global _geosolver_http_client
    if _geosolver_http_client is not None:
        with contextlib.suppress(Exception):
            await _geosolver_http_client.aclose()
        _geosolver_http_client = None


# ============================================================================
# Tools
# ============================================================================

@tool
def send_output(status: Literal["ACCEPT", "FAILED"], script: str = "", answer: str = "") -> str:
    """Submit the final task result. Call this when you have completed all analysis.

    Args:
        status: "ACCEPT" if the task was completed successfully, "FAILED" otherwise.
        script: The complete Python script that was generated (if status is ACCEPT).
        answer: A summary of the results/outputs produced by the script.

    Returns:
        Confirmation string.
    """
    global _task_output
    _task_output = {
        "status": status,
        "script": script,
        "answer": answer,
    }
    return f"Output submitted. Status: {status}"


# ============================================================================
# Builder
# ============================================================================

def build_geosolver_without_contract(model_name):
    """Build a GeoSolver agent WITHOUT contract/verification (79e3df2-style).

    Tool set: inspect, send_output, code_generate, graph_plan
    Pipeline: graph_plan → inspect (explore DAG) → code_generate → send_output
    """
    global _geosolver_http_client
    _geosolver_http_client = httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT)
    llm = init_chat_model(
        model_name,
        request_timeout=LLM_REQUEST_TIMEOUT,
        http_async_client=_geosolver_http_client,
    )

    # -- Sub-agent: Graph Planner (data exploration + DAG construction) --
    graph_planner = create_agent(
        llm,
        tools=[from_dict, execute_code],
        system_prompt=GRAPH_PLANNER_PROMPT,
        middleware=[
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=10000,
                        keep=3,
                        clear_tool_inputs=True,
                        exclude_tools=[],
                        placeholder="[cleared]",
                    ),
                ],
            ),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
    )

    # -- Sub-agent: Code Generator --
    code_generator = create_agent(
        llm,
        tools=[execute_code],
        system_prompt=CODE_GENERATOR_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(thread_limit=10, run_limit=5),
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=10000,
                        keep=3,
                        clear_tool_inputs=True,
                        exclude_tools=[],
                        placeholder="[cleared]",
                    ),
                ],
            ),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
    )

    @tool
    async def graph_plan(request: str) -> str:
        """
        Capable of exploring data and generating task plans in the form of data flow diagrams.
        Please select it when you need to view data information and conduct task planning.
        """
        result = await graph_planner.ainvoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    @tool
    async def code_generate(request: str) -> str:
        """
        Complete tasks by generating code based on data flow diagram prompts.
        Choose it when you need to write code.
        """
        result = await code_generator.ainvoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    # -- Main GeoSolver agent --
    geosolver_agent = create_agent(
        llm,
        tools=[graph_inspect, send_output, code_generate, graph_plan],
        system_prompt=GEOSOLVER_PROMPT,
        middleware=[
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
    )

    return geosolver_agent


# ============================================================================
# Async solver (compatible with run_agent.py interface)
# ============================================================================

async def geo_asolve_without_contract(agent, task, data_path, output_path):
    """Run the GeoSolver without-contract agent on a single task.

    Args:
        agent: The compiled geosolver agent (from build_geosolver_without_contract).
        task: Task dict with keys: task_text, source, used_dataset, Expected_Output, task_ID.
        data_path: Root data directory.
        output_path: Per-task output directory.

    Returns:
        Dict with keys: status, script, answer, trajectory.
    """
    global _task_output
    _task_output = {}

    # Reset graph state to prevent cross-task leakage
    reset_graph()

    data_dir = str(Path(data_path) / task['source'])
    file_list = [f"{data_dir}/{file_name}" for file_name in task.get('used_dataset', [])]
    workspace_dir = os.path.join(output_path, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    message = """
    TASK:
    {task_text}

    REQUIRED DATA (all files are under {data_dir}/):
    {file_list}

    EXEPTED OUTPUT:
    {expected_output}

    OUTPUT_DIR:
    {output_dir}

    INSTRUCTIONS:
    1. Please call the send_output tool to return the result after you complete all tasks.
    2. If you cannot produce a valid script, set "task_status" to "FAILED". Otherwise set "task_status" to "ACCEPT".
    """

    message = message.format(
        task_text=task['task_text'],
        data_dir=data_dir,
        file_list="\n".join(file_list),
        expected_output=json.dumps(task.get('Expected_Output', {})),
        output_dir=workspace_dir,
    )

    print(message)

    input_message = {"role": "user", "content": message}

    # Tee stdout to capture trajectory while still printing to console
    trajectory_buf = io.StringIO()
    _original_stdout = sys.stdout

    class _TeeWriter:
        """Write to both the real stdout and a capture buffer."""
        def __init__(self, original, capture):
            self._original = original
            self._capture = capture

        def write(self, s):
            self._original.write(s)
            self._capture.write(s)
            return len(s)

        def flush(self):
            self._original.flush()

    sys.stdout = _TeeWriter(_original_stdout, trajectory_buf)

    stream = agent.astream(
        {"messages": [input_message]},
        stream_mode="updates",
    )
    try:
        async for step in stream:
            for _, update in step.items():
                if update and (messages := update.get("messages")) and isinstance(messages, list):
                    for message in messages:
                        message.pretty_print()
    except asyncio.CancelledError:
        await stream.aclose()
        raise
    finally:
        sys.stdout = _original_stdout
        trajectory = trajectory_buf.getvalue()
        _task_output['trajectory'] = trajectory

    return _task_output
