import json
import os
from langchain_core.tools import tool
import pandas as pd
from typing import Dict, Optional, Annotated, List
from utils.logger import get_logger_with_session
import uuid
import chardet

@tool
def create_document(
    points: Annotated[List[str], "List of points to be included in the document"],
    file_name: Annotated[str, "Name and path of the file to save the document"]
) -> str:
    """
    Create and save a text document in Markdown format.

    This function takes a list of points and writes them as numbered items in a Markdown file.
    
    Returns:
    str: A message indicating where the outline was saved or an error message.
    """
    logger = get_logger_with_session()
    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory not in file_name:
            file_path = os.path.join(working_directory, file_name)
        logger.info(f"(create_document) Creating document: {file_name}")
        with open(file_path, "w", encoding='utf-8') as file:
            for i, point in enumerate(points):
                file.write(f"{i + 1}. {point}\n")
        logger.info(f"(create_document) Document created successfully: {file_name}")
        return f"Outline saved to {file_path}"
    except Exception as e:
        logger.error(f"Error while saving outline: {str(e)}")
        return f"Error while saving outline: {str(e)}"

@tool
def write_document(
    content: Annotated[str, "Content to be written to the document"],
    file_name: Annotated[str, "Name and path of the file to save the document"]
) -> str:
    """
    Create and save a Markdown document.

    This function takes a string of content and writes it to a file.
    It ensures the file is saved in Markdown (.md) format and handles errors gracefully.
    """
    logger = get_logger_with_session()

    try:
        if not file_name.lower().endswith('.md'):
            return "Error: The file must have a .md (Markdown) extension."

        working_directory = os.getenv("WORKING_DIRECTORY")
        if not working_directory:
            return "Error: 'WORKING_DIRECTORY' environment variable is not set."
        
        if working_directory not in file_name:
            file_path = os.path.join(working_directory, file_name)
        else:
            file_path = file_name

        # logger.info(f"(write_document) Writing document to: {file_name}")

        with open(file_path, "w", encoding='utf-8') as file:
            file.write(content)

        logger.info(f"(write_document) Document written successfully to: {file_name}")
        return f"Document saved to {file_path}"

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {str(fnf_error)}")
        return f"Error: {str(fnf_error)}"
    
    except PermissionError as perm_error:
        logger.error(f"Permission error: {str(perm_error)}")
        return f"Error: {str(perm_error)}"
    
    except Exception as e:
        logger.error(f"Unexpected error while saving document: {str(e)}")
        return f"Error while saving document: {str(e)}"

@tool
def read_document(
    file_name: Annotated[str, "Name and path of the file to read"],
    start: Annotated[Optional[int], "Starting line number to read from"] = None,
    end: Annotated[Optional[int], "Ending line number to read to"] = None
) -> str:
    """
    Read the specified document.

    This function reads a document from the specified file and returns its content.
    Optionally, it can return a specific range of lines.
    """
    logger = get_logger_with_session()

    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory not in file_name:
            file_path = os.path.join(working_directory, file_name)

        with open(file_path, 'rb') as f:
            raw_data = f.read(1000)
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        # chardet only sniffs the first 1KB, so a file whose head is pure
        # ASCII is often misdetected as 'ascii' and then blows up on multi-byte
        # UTF-8 bytes later in the file. ASCII is a subset of UTF-8, so decode
        # as UTF-8; on any remaining decode error, retry leniently instead of
        # failing the whole read.
        if encoding.lower() == 'ascii':
            encoding = 'utf-8'
        try:
            with open(file_path, "r", encoding=encoding) as file:
                lines = file.readlines()
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"(read_document) {encoding} decode failed ({str(e)}); retrying as utf-8 with errors='replace'")
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                lines = file.readlines()

        if start is None:
            start = 0

        content = "\n".join(lines[start:end])
        logger.info(f"(read_document) Document read successfully: {file_name}")
        return content

    except FileNotFoundError:
        logger.error(f"(read_document) File not found: {file_name}")
        return f"Error: The file {file_name} was not found."
    except Exception as e:
        logger.error(f"(read_document) Error while reading document: {str(e)}")
        return f"Error while reading document: {str(e)}"

@tool
def read_plan_document(
    role: Annotated[str, "Role to retrieve the plan document for (e.g., 'Planner')"]
) -> str:
    """
    Read the latest plan document for the specified role.
    """
    logger = get_logger_with_session()

    try:
        working_directory = os.environ.get("WORKING_DIRECTORY")
        if not working_directory:
            return "Error: 'WORKING_DIRECTORY' environment variable is not set."

        role_directory = os.path.join(working_directory, role.lower())
        
        if not os.path.exists(role_directory):
            return f"Error: Directory not found for role '{role}' at {role_directory}"

        markdown_files = [
            f for f in os.listdir(role_directory) 
            if f.lower().endswith('.md')
        ]
        
        if not markdown_files:
            return f"Error: No markdown files found in directory for role '{role}'"

        latest_file = max(
            markdown_files,
            key=lambda f: os.path.getctime(os.path.join(role_directory, f))
        )
        
        latest_file_path = os.path.join(role_directory, latest_file)

        with open(latest_file_path, "r", encoding="utf-8") as file:
            content = file.read()

        logger.info(f"(read_plan_document) Successfully read latest document: {latest_file_path}")
        return content

    except Exception as e:
        logger.error(f"Unexpected error while reading plan document: {str(e)}")
        return f"Unexpected error: {str(e)}"

@tool
def edit_document(
    file_name: Annotated[str, "Name and path of the file to edit"],
    inserts: Annotated[Dict[int, str], "Dictionary of line numbers and text to insert"]
) -> str:
    """
    Edit a document by inserting text at specific line numbers.

    This function reads an existing document, inserts new text at specified line numbers,
    and saves the modified document.

    Args:
        file_name (str): Name of the file to edit.
        inserts (Dict[int, str]): Dictionary where keys are line numbers and values are text to insert.

    Returns:
        str: A message indicating the result of the operation.

    Example:
        file_name = "example.txt"
        inserts = {
            1: "This is the first line to insert.",
            3: "This is the third line to insert."
        }
        result = edit_document(file_name=file_name, inserts=inserts)
        print(result)
        # Output: "Document edited and saved to /path/to/example.txt"
    """
    logger = get_logger_with_session()
    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory not in file_name:
            file_path = os.path.join(working_directory, file_name)
        logger.info(f"(edit_document) Editing document: {file_path}")
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()

        sorted_inserts = sorted(inserts.items())
        for line_number, text in sorted_inserts:
            if 1 <= line_number <= len(lines) + 1:
                lines.insert(line_number - 1, text + "\n")
            else:
                logger.error(f"Line number out of range: {line_number}")
                return f"Error: Line number {line_number} is out of range."

        with open(file_path, "w", encoding='utf-8') as file:
            file.writelines(lines) 
    
        logger.info(f"Document edited successfully: {file_path}")
        return f"Document edited and saved to {file_path}"
    except FileNotFoundError:
        logger.error(f"File not found: {file_name}")
        return f"Error: The file {file_name} was not found."
    except Exception as e:
        logger.error(f"Error while editing document: {str(e)}")
        return f"Error while editing document: {str(e)}"

@tool
def write_case_document(
    content: Annotated[str, "Content to write to the document"],
    doc_path: Annotated[str, "Full document path including filename"]
) -> str:
    """
    Create and save a Markdown document.

    This function takes a string of content and writes it to a file.
    """
    logger = get_logger_with_session()
    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory not in doc_path:
            file_path = os.path.join(working_directory, doc_path)
        # Ensure target directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"Writing document: {file_path}")
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Document written successfully: {file_path}")
        return f"Document saved to {file_path}"
    except Exception as e:
        logger.error(f"Error while saving document: {str(e)}")
        return f"Error while saving document: {str(e)}"