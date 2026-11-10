import configparser
from datetime import datetime
import os
from pathlib import Path
from typing import Dict
import uuid
import time
from langsmith import Client
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
import json
from utils.utils import copy_files
from utils.logger import get_logger_with_session
from langgraph.managed import IsLastStep


def setup_environment() -> Dict[str, str]:
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H_%M_%S")
    session_id = str(uuid.uuid4())[:8]
    working_dir = Path("data_storage") / f"{date_str}_{time_str}_{session_id}"
    working_dir.mkdir(parents=True, exist_ok=True)
    env_vars = {

        "LANGCHAIN_TRACING_V2": config.get("Langsmith", "LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_ENDPOINT": config.get("Langsmith", "LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": config.get("Langsmith", "LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": config.get("Langsmith", "LANGCHAIN_PROJECT"),

        "SESSION_ID": f"{date_str}_{time_str}_{session_id}",
        
        "WORKING_DIRECTORY": str(working_dir),
        "CONDA_PATH": config.get("Tool", "CONDA_PATH"),
        "CONDA_ENV": config.get("Tool", "CONDA_ENV"),
        
        "OPENAI_BASE_URL": config.get("LLM", "OPENAI_BASE_URL"),
        "ANTHROPIC_BASE_URL": config.get("LLM", "ANTHROPIC_BASE_URL"),
        "API_KEY": config.get("LLM", "API_KEY"),
        "DASHSCOPE_API_KEY": config.get("LLM", "DASHSCOPE_API_KEY"),
        "DEEPSEEK_API_KEY": config.get("LLM", "DEEPSEEK_API_KEY"),
        "NVIDIA_API_KEY": config.get("LLM", "NVIDIA_API_KEY"),
        "VOLC_ACCESSKEY": config.get("LLM", "VOLC_ACCESSKEY"),
        "VOLC_SECRETKEY": config.get("LLM", "VOLC_SECRETKEY"),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value

    return env_vars

with open("datasets/contest.json", "r", encoding="utf-8") as f:
    task_list = json.load(f)

logger = get_logger_with_session()

print(len(task_list))

for task_num in range(1, 2):
    try:
        # setup environment
        env_config = setup_environment()

        # create_graph after setup_environment()
        from geo_agent.graph import create_graph

        # setup logger
        logger = get_logger_with_session()

        graph, graph_config = create_graph(task_num)

        logger.info(f"-------------------- Graph created with thread_id {task_num} --------------------")
        logger.info(f"-------------------- Task {task_num}: {task_list[task_num-1]['task']} --------------------")

        time.sleep(10)

        user_input = task_list[task_num-1]["task"] + ", Data files are stored in the data directory like data/.../..."
        data_path = task_list[task_num-1]["data_path"]
        if isinstance(data_path, list):
            data_path = [os.path.join("datasets/Datasets", path) for path in data_path]
        else:
            data_path = os.path.join("datasets/Datasets", data_path)

        target_directory = os.path.join(os.environ["WORKING_DIRECTORY"], "data")
        os.makedirs(target_directory, exist_ok=True)
        copy_files(data_path, target_directory)

        _input = {
                "messages": [
                    HumanMessage(
                        content=user_input
                    ),
                ],
                "user_query": user_input,
                "user_guide": "",
                "task_category": "",
                "task_plan":"",
                "process_decision": "",
                "coder_state": "",
                "reporter_state": "",
                "cartographer_state": "",
                "code_reviewer_state": "",
                "map_reviewer_state": "",
                "report_reviewer_state": "",
                "needs_revision": False,
                "is_last_step": IsLastStep()
            }

        events = graph.stream(
            _input,
            graph_config,
            stream_mode="values",
            debug=False
        )

        def print_stream(stream):
            for s in stream:
                message = s["messages"][-1]
                if isinstance(message.content, list):
                    content = "\n".join(str(item) for item in message.content)
                else:
                    content = message.content
                logger.info(content)

        print_stream(events)
    except Exception as e:
        logger.error(f"Task {task_num} failed with error: {str(e)}")
        continue