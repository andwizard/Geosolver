import os
import shutil
import uuid
import yaml
import configparser
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage

repo_path = os.path.join(os.path.dirname(__file__), "../")

def _extract_message_content(message_or_dict) -> str:
    """Extract plain text content from complex message objects or dictionaries"""
    # logger.info(f"Extracting content from type: {type(message_or_dict)}")
    
    if isinstance(message_or_dict, dict):
        if "messages" in message_or_dict:
            messages = message_or_dict["messages"]
            if isinstance(messages, list) and messages:
                first_message = messages[-1]
                if isinstance(first_message, (AIMessage, HumanMessage, BaseMessage)):
                    return first_message.content
    
    if isinstance(message_or_dict, (AIMessage, HumanMessage, BaseMessage)):
        return message_or_dict.content
    
    if isinstance(message_or_dict, str):
        return message_or_dict
    
    return str(message_or_dict)

def read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as file:
                return yaml.safe_load(file)
        except UnicodeDecodeError:
            raise ValueError("Failed to read the YAML file with UTF-8 and GBK encodings.")
    except Exception as e:
        raise ValueError(f"YAML loading failed due to: {e}")
    
def copy_files(src_paths: Union[str, List[str]], dst_dir: str) -> None:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    if isinstance(src_paths, str):
        src_paths = [src_paths]
    
    for src_path in src_paths:
        if not os.path.exists(src_path):
            raise ValueError(f"Source path does not exist: {src_path}")
            
        base_name = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, base_name)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


def get_working_directory() -> str:
    working_directory = os.environ.get("WORKING_DIRECTORY")
    
    if working_directory is None:
        raise EnvironmentError("The 'WORKING_DIRECTORY' environment variable is not set.")
    
    return working_directory