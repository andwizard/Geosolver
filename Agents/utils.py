import math
import re
import statistics


from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, create_model
import inspect
from inspect import Parameter, Signature
from functools import wraps
from typing import Callable, Any, Annotated, Tuple

from dashscope.tokenizers.tokenizer import root_path
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pandas.core.sample import preprocess_weights
from pydantic_core import PydanticUndefined

# from tools import *
from langchain_core.documents import Document
from langchain_core.tools import tool, StructuredTool
import re
import json
from typing import List, Dict, Any


from enum import Enum
from typing import Optional

def extract_markdown_code_blocks(text: str) -> list[str]:
    """Extract multi-line Markdown code blocks (wrapped in ```)."""
    # Regex explanation:
    # ^```.*$: match the opening ``` (optionally with a language tag, e.g. ```python)
    # [\s\S]*?: non-greedy match of everything in between (including newlines)
    # ^```$: match the closing ``` on its own line
    pattern = r'^```.*?$(?s)(.*?)^```$'
    # re.MULTILINE: make ^/$ match the start/end of each line; re.DOTALL: make . match newlines
    # (equivalent to (?s) in the pattern)
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    # Strip leading/trailing blank lines (keep indentation inside the code), filter empty results
    return [match.strip('\n\r') for match in matches if match.strip()]

def write_codes_to_single_file(code: str, filename: str = "output.py"):
    """
    Write all code blocks into a single Python file
    :param code: the extracted code blocks
    :param filename: output filename (default: output.py)
    """
    # Write to file (encoding="utf-8" avoids encoding issues)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"✅ Code blocks written to {filename}")

def inject_structured_reasoning(original_func: Callable) -> Callable:
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Capture the reasoning dict
        reasoning_dict = kwargs.pop('reasoning', None)

        if reasoning_dict:

            task_reason = reasoning_dict.get("__task__", "not specified")
            print(f"\n🔹 [Tool Call] {original_func.__name__}")
            print(f"   └─ 🎯 Task purpose: {task_reason}")

            # Iterate over the reasons for specific parameters
            for param_name, reason in reasoning_dict.items():
                if param_name != "__task__":
                    # Get the actual passed value for comparison
                    actual_value = kwargs.get(param_name, "default value")
                    print(f"   └─ ⚙️ Parameter {param_name}={actual_value}")
                    print(f"      └─ 💡 Reason: {reason}")
            # ----------------------------------------

        return original_func(*args, **kwargs)

    # Modify the signature
    sig = inspect.signature(original_func)
    parameters = list(sig.parameters.values())

    # [Core change]: change the type annotation to Dict[str, str]
    reasoning_param = Parameter(
        name="reasoning",
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation= Annotated[Dict[str, str], "Must be a dictionary.\n- Contains the key '__task__': explain the overall goal of calling this tool.\n- Contains keys matching other parameter names: explain where specific parameter values come from.\n"]
    )
    if "reasoning" not in sig.parameters.keys():
        parameters.append(reasoning_param)
        wrapper.__signature__ = sig.replace(parameters=parameters)

    # [Core change]: docstring guides the model to write a Dict
    doc_addendum = (
        "\n\n    Args:\n"
        "        reasoning (Dict[str, str]): Must be a dictionary.\n"
        "            - Contains the key '__task__': explain the overall goal of calling this tool.\n"
        "            - Contains keys matching other parameter names: explain where specific parameter values come from.\n"
        "            Example: {'__task__': 'compute interest', 'rate': 'the user specified a 5% annual rate'}"
    )

    if wrapper.__doc__:
        wrapper.__doc__ += doc_addendum
    else:
        wrapper.__doc__ = doc_addendum

    return wrapper

def merge_schema_into_signature(target_func, schema_cls: type[BaseModel]):
    # Get the original function signature
    old_sig = inspect.signature(target_func)
    new_params = []

    # Get the Pydantic V2 field definitions
    model_fields = schema_cls.model_fields

    for param in old_sig.parameters.values():
        param_name = param.name

        # If this parameter is defined in the schema
        if param_name in model_fields:
            field_info = model_fields[param_name]

            # 1. Extract the type annotation
            new_annotation = field_info.annotation

            # 2. Extract the default value (handle PydanticUndefined)
            # Strategy: if the schema has a default, prefer the schema's;
            # if the schema marks it required (Undefined) but the function had a default, keep the function's.
            new_default = inspect.Parameter.empty
            if field_info.default is not PydanticUndefined:
                new_default = field_info.default
            elif param.default is not inspect.Parameter.empty:
                # The schema has no default, but the function does — keep the function's
                new_default = param.default

            # Use .replace() to create a new Parameter object (Parameter is immutable)
            new_param = param.replace(
                annotation=Annotated[new_annotation, field_info.description],
                default=new_default
            )
            new_params.append(new_param)

        else:
            # Parameter not in the schema; keep it as is (e.g. request_context)
            new_params.append(param)

    # Build the new signature and overwrite
    target_func.__signature__ = old_sig.replace(parameters=new_params)

    # This is a compatibility step: also update the __annotations__ dict,
    # since many simple tools only look at __annotations__ and do not parse the signature
    for param in new_params:
        if param.annotation is not inspect.Parameter.empty:
            target_func.__annotations__[param.name] = param.annotation



class SignatureParser:
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)

    def extract_functions(self) -> List[Dict]:
        """
        Extract the full source of all function definitions (including decorators, signature, docstring)
        """
        functions = []
        lines = self.source_code.splitlines()

        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                # Get the start/end line numbers of the function in the source
                start_line = node.lineno - 1
                end_line = node.end_lineno

                # If there are decorators, move the start line up
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1

                # Extract the complete function stub code.
                # Note: this extracts the raw text, i.e. def foo(): ...
                func_source = "\n".join(lines[start_line:end_line])

                # Get the docstring
                docstring = ast.get_docstring(node) or "No description provided."


                functions.append({
                    "name": node.name,
                    "source": func_source,  # Full def ... : """doc""" pass
                    "docstring": docstring
                })
        return functions


import ast
import sys


class ImportLifter(ast.NodeTransformer):
    """
    AST transformer:
    1. Traverse all nodes in the code.
    2. Extract all import and from ... import statements.
    3. Remove these import nodes from their original locations (including inside functions).
    """

    def __init__(self):
        self.collected_imports = set()

    def visit_Import(self, node):
        # Use ast.unparse to convert the node back to a string (Python 3.9+)
        self.collected_imports.add(ast.unparse(node))
        return None  # Returning None deletes the node from the tree

    def visit_ImportFrom(self, node):
        # Ignore from __future__ ...
        if node.module == '__future__':
            return node
        self.collected_imports.add(ast.unparse(node))
        return None  # Returning None deletes the node from the tree


def clean_and_organize_code(file_path):
    print(f"🧹 Cleaning code file: {file_path} ...")

    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        # 1. Parse the source code into an AST
        tree = ast.parse(source_code)

        # 2. Run the transformer
        lifter = ImportLifter()
        cleaned_tree = lifter.visit(tree)
        ast.fix_missing_locations(cleaned_tree)  # Fix line-number information

        # 3. Process the import statements (deduplicate + sort)
        # Separating stdlib from third-party imports is hard to detect automatically,
        # so just sort alphabetically here
        sorted_imports = sorted(list(lifter.collected_imports))

        # 4. Regenerate the code
        imports_block = "\n".join(sorted_imports)
        body_code = ast.unparse(cleaned_tree)

        # 5. Combine the final content
        final_content = f"# Optimized Imports\n{imports_block}\n\n{body_code}"

        # 6. Write back to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"✨ Cleanup complete! Extracted {len(sorted_imports)} import statements.")

    except Exception as e:
        print(f"❌ Error parsing code: {e}")

def window_prob(input: List[str], tagets: List[str]) -> List[float]:
    confidences = []  # stores (start index, end index)
    conf = []
    current_list = 0
    start = False
    start_idx = -1

    def logprob_to_confidence(logprob):
        """Convert log probability to confidence (probability value, 0~1)"""
        # Handle the case where logprob is 0 (probability = 1)
        if logprob == 0:
            return 1.0
        # Prevent exp overflow for very small logprobs by clamping to a minimum
        min_logprob = -100
        logprob = max(logprob, min_logprob)
        return math.exp(logprob)

    for idx, item in enumerate(input):
        token = item['token']
        prob = logprob_to_confidence(item['logprob'])

        if '[' in token and not start:
            start = True

        if start and token in tagets[current_list]:
            conf.append(prob)
            # print(token)
        elif len(conf) > 0 and token not in tagets[current_list] and current_list < len(tagets) - 1:
            # print("----", current_list)
            current_list += 1
            confidences.append(statistics.mean(conf))
            conf = []

        if idx == len(input) - 1:
            confidences.append(statistics.mean(conf))

    print(confidences)
    return confidences





def extract_json_from_md_code_blocks(md_text: str) -> List[Dict[str, Any]]:
    """
    Extract all valid JSON content from code blocks in Markdown text

    Args:
        md_text: the raw Markdown text (containing code blocks)

    Returns:
        a list of all valid JSON objects (each element is a parsed JSON dict/list)
    """
    # Store the final extracted JSON results
    valid_jsons = []

    # Step 1: regex-match Markdown code blocks (compatible with ```json/```javascript/``` etc.)
    # Regex explanation:
    # - ```[\s\S]*?``` matches all code blocks (non-greedy)
    # - (?<=```) positive lookbehind: match content after ```
    # - (?:json|javascript|js)? match the optional language tag (json/js/javascript)
    # - \s* match newlines/spaces
    # - ([\s\S]*?) capture the code block content (non-greedy)
    # - (?=```) positive lookahead: match content before ```
    code_block_pattern = re.compile(r'(?<=```)(?:json|javascript|js)?\s*([\s\S]*?)(?=```)', re.IGNORECASE)
    code_blocks = code_block_pattern.findall(md_text)

    if not code_blocks:
        print("No Markdown code blocks detected")
        return valid_jsons

    # Step 2: iterate over each code block, extract and validate JSON
    for idx, code in enumerate(code_blocks):
        # Clean irrelevant content from the code block (comments, blank lines, leading/trailing whitespace)
        cleaned_code = _clean_code_block(code)

        if not cleaned_code:
            print(f"Code block #{idx + 1} is empty, skipping")
            continue

        # Try to parse JSON and handle exceptions
        try:
            json_data = json.loads(cleaned_code)
            valid_jsons.append(json_data)
            print(f"Code block #{idx + 1} JSON parsed successfully")
        except json.JSONDecodeError as e:
            print(f"Code block #{idx + 1} JSON parse failed: {e.msg} (position: {e.pos})")
            print(f"Failed content preview: {cleaned_code[:100]}...")
        except Exception as e:
            print(f"Code block #{idx + 1} processing exception: {str(e)}")

    return valid_jsons


def _clean_code_block(code: str) -> str:
    """Helper function: clean irrelevant content from a code block"""
    # 1. Remove single-line comments (starting with //)
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    # 2. Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # 3. Remove leading/trailing whitespace, newlines, tabs
    code = code.strip()
    # 4. Remove blank lines
    code = '\n'.join([line for line in code.split('\n') if line.strip()])
    return code
