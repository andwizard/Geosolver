import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
import csv
import io
from langchain_core.utils.uuid import uuid7
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import InMemorySaver

from langchain.tools import tool
from Agents.prompts.basic_prompts import BASIC_PROMPT
from Agents.prompts.deepagent_prompts import PROMPT

output = {}
answer = ""
status= False

@tool(parse_docstring=True)
def send_output(final_answer: str, task_status: bool) -> dict:
    """Send output to user

    Args:
        final_answer: (str) final answer of the output
        task_status: (bool) Was the task completed successfully? True for success, False for failure.
    """
    global answer
    global status
    answer = final_answer
    status = task_status
    return final_answer, task_status


def build_deepagent(model_name, root_dir, env_dict):
    backend = LocalShellBackend(root_dir=root_dir, env=env_dict, virtual_mode=False)

    agent = create_deep_agent(
        model=init_chat_model(model_name),
        tools = [send_output],
        backend=backend,

    )

    return agent


async def deepagent_ainvoke(agent, task, data_path, output_path):
    global answer, status
    answer = ""
    status = False

    from pathlib import Path

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

    prompt = basic_prompt + PROMPT

    # print(prompt)
    # quit()
    input_message = {
    "role": "user",
    "content": prompt,
}
    script = ""
    trajectory = ""
    for step in agent.stream(
        {"messages": [input_message]},
        stream_mode="updates",
    ):
        for _, update in step.items():
            if update and (messages := update.get("messages")) and isinstance(messages, list):
                for message in messages:
                    if hasattr(message, "tool_calls"):
                        if message.tool_calls:
                            if message.tool_calls[0]['name'] == "write_file" and '.py' in message.tool_calls[0]['args']['file_path']:
                                script = message.tool_calls[0]['args']['content']
                    buffer = io.StringIO()
                    sys.stdout = buffer
                    message.pretty_print()
                    sys.stdout = sys.__stdout__
                    trajectory += buffer.getvalue()
                    message.pretty_print()

    return {
            "status": "ACCEPT" if status else "FAILED",
            "script": script,
            "answer": answer,
            "trajectory": trajectory
    }



async def as_main():
    data_root = "../Data"
    model_name = "openai:qwen3.5-plus"
    root_dir = "../"
    env_dict = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


    task_id = "TASK_250309_135125_304935"
    agent = build_deepagent(model_name, root_dir, env_dict)
    with open( "../Data/datasets.json", "r") as f:
        data = json.load(f)
        tasks = data['tasks']
        for task in tasks:
            if task['task_ID'] == task_id :
                result = await deepagent_ainvoke(agent, task, data_path=os.path.abspath(data_root), output_path=os.path.abspath('.'))
                break

    print("Final result:", result)

if __name__ == "__main__":
    asyncio.run(as_main())