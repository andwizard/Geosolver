"""
CodeAct — single-LLM baseline agent.

A single LLM equipped with a code-execution tool directly generates and runs a
Python script for the task, then submits the result via ``send_output``. There
is no separate planning stage (see ``build_geosolver_without_dfg`` for the
plan-then-code variant).
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from Agents.prompts.vanilla_prompts import CODE_PROMPT_VANILLA
from Agents.tools.sandbox_env import execute_code
from config import LLM_REQUEST_TIMEOUT


class State(TypedDict):
    """Type definition for the graph state"""
    basic_prompt: str
    answer: str
    messages: Annotated[list, add_messages]
    status: str
    script: str
    output_dir: str


_task_output = {}


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


async def build_codeact(model_name):
    """Build the CodeAct agent: a single LLM that writes and runs code directly."""

    # Create a model instance with request_timeout so LLM API calls
    # don't hang indefinitely when the API is unresponsive.
    llm = init_chat_model(model_name, request_timeout=LLM_REQUEST_TIMEOUT)

    workflow = StateGraph(State)

    async def _code(state: State):
        print("=" * 30, "codeact", "=" * 30)

        agent = create_agent(
            llm,
            tools=[execute_code, send_output],
            system_prompt=CODE_PROMPT_VANILLA,
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

        messages = [
            HumanMessage(content=state['basic_prompt']),
        ]
        try:
            response = None
            async for chunk in agent.astream({"messages": messages}):
                # astream yields {node_name: state_update}; pick whichever node
                # emitted a message update (e.g. "model", "tools").
                for node_update in chunk.values():
                    if isinstance(node_update, dict) and node_update.get("messages"):
                        response = node_update
                        last_msg = node_update["messages"][-1]
                        if hasattr(last_msg, "content") and last_msg.content:
                            print(last_msg.content, end="", flush=True)
            print()

            state['status'] = _task_output['status']
            state['script'] = _task_output['script']
            state['answer'] = response['messages'][-1].content if response else ""

        except Exception as e:
            print("error: " + str(e))
            state['status'] = "FAILED"
            state['script'] = f"{str(e)}"
            state['answer'] = f"Error occurred during execution: {e}"

        return state

    workflow.add_node("codeact", _code)
    workflow.add_edge(START, "codeact")
    workflow.add_edge("codeact", END)

    return workflow.compile()


async def codeact_ainvoke(agent, task, data_path, output_path):
    from Agents.prompts.basic_prompts import BASIC_PROMPT

    data_dir = Path(data_path) / task['source']
    file_list = [f"{data_dir}/{file_name}" for file_name in task.get('used_dataset', [])]
    workspace_dir = f"{output_path}/workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    basic_prompt = BASIC_PROMPT.format(
        task_text=task['task_text'],
        file_list=file_list,
        expected_output=json.dumps(task.get('Expected_Output', {})),
        output_dir=workspace_dir,
    )

    return await agent.ainvoke({"basic_prompt": basic_prompt, "output_dir": output_path})


async def as_main():

    import configparser
    _cfg = configparser.ConfigParser()
    _cfg.read(Path(__file__).parent.parent / "config.ini")
    model_name = _cfg.get("Model", "name", fallback="openai:qwen3.5-plus")
    data_root = _cfg.get("PATH", "DATA_PATH", fallback="./Data")

    # single test
    agent = await build_codeact(model_name)
    task_id = "TASK_250309_135125_304935"
    with open("../Data/datasets.json", "r") as f:
        data = json.load(f)
        tasks = data['tasks']
        for task in tasks:
            if task['task_ID'] == task_id:
                break

    print(task['task_text'])
    response = await codeact_ainvoke(agent, task, data_path=os.path.abspath(data_root), output_path=os.path.abspath("."))
    print("FINAL ANSWER")
    print(response['answer'])


if __name__ == "__main__":
    asyncio.run(as_main())
