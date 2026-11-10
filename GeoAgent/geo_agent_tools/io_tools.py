from datetime import datetime
import json
import os
from pathlib import Path
from typing import Annotated
from filelock import FileLock
from langchain.tools import tool
from utils.logger import get_logger_with_session

@tool
def list_directory_contents(list_directory: str) -> str:
    """
    List the contents of the specified directory. If the directory does not exist, create it.
    
    Args:
        list_directory (str): The path to the directory to list. Defaults to the data storage directory.
    
    Returns:
        str: A string representation of the directory contents.
    """
    logger = get_logger_with_session()
    # logger.info(f"Listing directory contents: {directory}")
    working_dir = Path(os.environ['WORKING_DIRECTORY'])
    directory = working_dir / list_directory
    try:
        if not os.path.exists(directory):
            logger.info(f"Directory does not exist. Creating directory: {directory}")
            os.makedirs(directory)
            return f"Directory was not found and has been created: {directory}"
        
        logger.info(f"Listing contents of directory: {directory}")
        contents = os.listdir(directory)
        logger.info(f"Directory contents: {contents}")
        return f"Directory contents:\n" + "\n".join(contents)
    except PermissionError as e:
        logger.error(f"Permission denied: {str(e)}")
        return f"Error: Permission denied: {str(e)}"
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")
        return f"Error listing directory contents: {str(e)}"

@tool
def list_working_directory_files(all_files: Annotated[bool, "whether to list the files in the subfolder, default is True"]) -> str:
    """
    Lists the names of the files and directories in the current working directory and returns a success status.
    
    Args:
        all_files (bool): When True, includes files and directories in subdirectories; 
                         When False, only lists items in the working directory.

    Returns:
        str: A JSON formatted string showing all files and directories with a success status.
    """
    logger = get_logger_with_session()
    
    try:
        # Get the working directory from environment variable
        working_dir = Path(os.environ['WORKING_DIRECTORY'])
        logger.info(f"(list_working_directory_files) Listing contents of working directory.")
        
        def format_list(items: list) -> str:
            """Helper function to format list items with quotes"""
            normalized_items = [item.replace(os.path.sep, '/') for item in sorted(items)]
            return "[" + ", ".join(f'"{item}"' for item in normalized_items) + "]"
        
        # Lists to hold the file and directory names
        all_files_list = []
        all_dirs_list = []

        if all_files:
            # Recursively collect files and directories
            for root, dirs, files in os.walk(working_dir):
                rel_root = Path(root).relative_to(working_dir)
                
                # Add files
                all_files_list.extend(
                    str(rel_root / file) if str(rel_root) != '.' else file 
                    for file in files
                )
                
                # Add directories
                all_dirs_list.extend(
                    str(rel_root / dir_name) if str(rel_root) != '.' else dir_name 
                    for dir_name in dirs
                )

            # Log the results
            logger.info(f"📁 Directories: {format_list(all_dirs_list)}\n")
            logger.info(f"📄 Files: {format_list(all_files_list)}")

            # Check if there are no files or directories found
            if not all_files_list and not all_dirs_list:
                result = {
                    "success": False,
                    "message": "No files or directories found.",
                    "directories": [],
                    "files": []
                }
            else:
                # Return the formatted JSON output
                result = {
                    "success": True,
                    "message": "Directory contents listed successfully.",
                    "directories": all_dirs_list,
                    "files": all_files_list
                }
        else:
            # Handle the case where only top-level files/directories are listed (if implemented)
            result = {
                "success": False,
                "message": "Only top-level listing is not yet implemented.",
                "directories": [],
                "files": []
            }

        # Convert the result to JSON formatted string
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        # Handle errors and return a failure status
        logger.error(f"Error in listing directory files: {e}")
        result = {
            "success": False,
            "message": f"Error occurred: {str(e)}",
            "directories": [],
            "files": []
        }
        return json.dumps(result, ensure_ascii=False)
    

@tool
def create_directory(
    folder_name: Annotated[str, "Name of the folder to create"],
    path: Annotated[str, "Path where the folder will be created"] = ""
) -> str:
    """
    Create a new directory at the specified path.
    
    This function creates a new directory at the given path, which will be joined with working_directory.
    If path is empty, the folder will be created directly under working_directory.
    If the folder already exists, a notification will be returned.
    
    Returns:
        str: A message indicating the operation result
    """
    logger = get_logger_with_session()
    try:
        # Join path with working_directory
        working_directory = os.environ["WORKING_DIRECTORY"]
        full_path = os.path.join(working_directory, path, folder_name) if path else os.path.join(working_directory, folder_name)
        
        # Check if directory already exists
        if os.path.exists(full_path):
            logger.info(f"(create_directory) Directory already exists: {full_path}")
            return f"Directory already exists at {full_path}"
            
        # Create directory
        os.makedirs(full_path)
        logger.info(f"(create_directory) Directory created successfully: {full_path}")
        return f"Directory created at {full_path}"
        
    except Exception as e:
        logger.error(f"Error while creating directory: {str(e)}")
        return f"Error while creating directory: {str(e)}"
    
@tool
def find_file_in_working_directory(filename: str) -> str:
    """
    Search for the specified file in the working directory and all its subdirectories.
    Provides detailed feedback if the file is found or not.

    Args:
        filename (str): The name of the file to search for in the working directory, without the path as prefix.

    Returns:
        str: A message indicating whether the file exists, or if there was an error during the search.
    """
    logger = get_logger_with_session()
    try:
        working_dir = Path(os.environ["WORKING_DIRECTORY"])

        if not working_dir.exists():
            logger.error(f"Working directory does not exist: {working_dir}")
            return f"Error: Working directory does not exist: {working_dir}"

        # logger.info(f"(find_file_in_working_directory) Searching for file '{filename}' in the working directory.")
        
        for root, dirs, files in os.walk(working_dir):
            if filename in files:
                file_path = Path(root) / filename
                logger.info(f"(find_file_in_working_directory) File found: {filename}")
                return f"File '{filename}' found at: {filename}"

        logger.info(f"(find_file_in_working_directory) File '{filename}' not found in the working directory")
        return f"File '{filename}' not found in the working directory"

    except PermissionError as e:
        logger.error(f"Permission denied: {str(e)}")
        return f"Error: Permission denied while searching for '{filename}': {str(e)}"

    except Exception as e:
        logger.error(f"Error while searching for file: {str(e)}")
        return f"Error while searching for file '{filename}': {str(e)}"
    
@tool
def create_image_review_queue_file(
    image_filenames: Annotated[list[str], "List of image filenames to be written to the txt file"]
) -> str:
    """
    Create a .txt file named 'image_review_queue.txt' containing a list of image filenames, each on a new line.
    
    This function takes a list of image filenames and writes them to a fixed-name .txt file 'image_review_queue.txt'
    in the cartographer directory. Each image filename will be written on a new line in the txt file. 
    If the txt file already exists, it will be overwritten.
    
    Args:
        image_filenames (list[str]): List of image filenames to be written to the file.
        
    Returns:
        str: A message indicating the result of the operation.
    """
    logger = get_logger_with_session()
    try:
        working_dir = Path(os.environ["WORKING_DIRECTORY"])
        cartographer_dir = working_dir / "cartographer"
        if not os.path.exists(cartographer_dir):
            os.makedirs(cartographer_dir)
            
        full_path = os.path.join(cartographer_dir, "image_review_queue.txt")

        # Write the image filenames to the txt file
        with open(full_path, 'w') as f:
            for filename in image_filenames:
                f.write(f"cartographer/{filename}\n")
        
        # Log the result
        logger.info(f"(create_image_review_queue_file) Successfully created image_review_queue.txt in cartographer directory.")
        return f"File image_review_queue.txt created successfully in cartographer directory."

    except Exception as e:
        # Log any errors
        logger.error(f"Error while creating image list file: {str(e)}")
        return f"Error while creating image list file: {str(e)}"
    

def get_next_image_from_queue() -> str:
    """
    Read the first image filename from 'image_review_queue.txt', return it,
    and remove it from the file.

    This function reads the first image filename from the 'image_review_queue.txt' file
    in the working_dir/cartographer directory, returns the filename, and removes it from 
    the file to process it. If the file is empty, it returns a message indicating that 
    the queue is empty.

    Returns:
        str: The name of the first image in the queue or a message indicating the queue is empty.
    """
    logger = get_logger_with_session()
    try:
        working_dir = Path(os.environ["WORKING_DIRECTORY"])
        full_path = working_dir / "cartographer" / "image_review_queue.txt"
        
        # Check if the file exists
        if not os.path.exists(full_path):
            logger.error(f"(get_next_image_from_queue) The file {full_path} does not exist.")
            return f"Error: The file {full_path} does not exist."
        
        # Read the file and get the first image filename
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # If the file is empty, return an appropriate message
        if not lines:
            return "No images in the review queue."
        
        # Get the first image filename (strip newline)
        first_image = lines[0].strip()

        # Remove the first image from the file
        with open(full_path, 'w') as f:
            f.writelines(lines[1:])  # Write the remaining images back to the file
        
        # Log and return the result
        logger.info(f"(process_next_image_from_queue) Image {first_image} is ready for processing.")
        return f"{first_image}"

    except Exception as e:
        # Log any errors
        logger.error(f"Error while processing the image queue: {str(e)}")
        return f"Error while processing the image queue: {str(e)}"

def get_image_queue_count() -> str:
    """
    Get the number of images remaining in the image review queue.

    This function reads the 'image_review_queue.txt' file from the working_dir/cartographer
    directory and returns the count of remaining images in the queue.

    Returns:
        str: A message indicating the number of images in the queue.
    """
    logger = get_logger_with_session()
    try:
        full_path = os.path.join(os.environ["WORKING_DIRECTORY"], "cartographer", "image_review_queue.txt")
        
        # Check if the file exists
        if not os.path.exists(full_path):
            logger.error(f"(get_image_queue_count) The file {full_path} does not exist.")
            return f"Error: The file {full_path} does not exist."
        
        # Read the file and count lines
        with open(full_path, 'r') as f:
            # 过滤掉空行后计数
            count = sum(1 for line in f if line.strip())
        
        logger.info(f"(get_image_queue_count) Found {count} images in the queue.")
        return count

    except Exception as e:
        logger.error(f"Error while counting images in queue: {str(e)}")
        return f"Error while counting images in queue: {str(e)}"

@tool
def save_file_metadata(role: str, file_name: str, file_path: str, description: str):
    """
    This function saves metadata about a file generated by a specific role.
    If `file_metadata.json` does not exist, it creates the file and adds the entry.
    It supports multiple files per role and accepts the description of each file as a required parameter.

    Args:
        role (str): The role (e.g., 'User', 'Planner', 'Coder', 'Cartographer', 'Reporter'). Capitalize the first letter of the role.
        file_name (str): The name of the file being saved.
        file_path (str): The full path of the file.
        description (str): A brief description of the file's purpose or role.
    """
    logger = get_logger_with_session()

    # Get the working directory from environment variable
    working_path = os.environ.get("WORKING_DIRECTORY", "")
    if not working_path:
        raise EnvironmentError("(save_file_metadata) WORKING_DIRECTORY environment variable is not set.")

    # Ensure the directory exists
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    # Full path for the metadata file
    metadata_file_path = os.path.join(working_path, "file_metadata.json")

    # Define a lock for the metadata file
    lock = FileLock(metadata_file_path + ".lock")

    # Try to acquire the lock and perform file operations
    with lock:
        try:
            # Check if metadata file exists, if not, create it
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Initialize the metadata structure
                data = {
                    "metadata": {
                        "last_updated": datetime.utcnow().isoformat(),
                        "roles": []
                    },
                    "files": {}
                }
        except json.JSONDecodeError:
            logger.error(f"(save_file_metadata) Invalid JSON format in {metadata_file_path}. Creating a new file.")
            data = {
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "roles": []
                },
                "files": {}
            }

        # Ensure the role exists in the file structure
        if role not in data["files"]:
            data["files"][role] = []

        # Prepare the file entry
        file_metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Append the file metadata to the role's list if not already added
        if not any(file['file_name'] == file_name for file in data["files"][role]):
            data["files"][role].append(file_metadata)

        # Update the last_updated timestamp
        data["metadata"]["last_updated"] = datetime.utcnow().isoformat()

        # Check if the role exists in the roles list, add it if not
        if role not in data["metadata"]["roles"]:
            data["metadata"]["roles"].append(role)

        # Save the updated metadata to the file
        try:
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"(save_file_metadata) File metadata for {role} -> {file_name} has been saved.")
        except Exception as e:
            logger.error(f"(save_file_metadata) Error saving metadata: {e}")

@tool
def query_files_by_role(role: str) -> str:
    """
    Query all files generated by a specified role and return their names, paths and descriptions as a string suitable for LLM input.
    
    Args:
        role (str): The role to query (e.g. 'User', 'Planner', 'Coder', 'Cartographer', 'Reporter'). Capitalize the first letter of the role.
    
    Returns:
        str: A formatted string containing the name, path and description of each file.
    """
    logger = get_logger_with_session()
    working_path = os.environ.get("WORKING_DIRECTORY", "")
    if not working_path:
        raise EnvironmentError("(query_files_by_role) WORKING_DIRECTORY environment variable is not set.")

    metadata_file_path = os.path.join(working_path, "file_metadata.json")

    if not os.path.exists(metadata_file_path):
        raise FileNotFoundError(f"Metadata file {metadata_file_path} not found.")

    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if role in data["files"]:
        file_info_str = f"Files for role '{role}':\n"
        for file_metadata in data["files"][role]:
            file_info_str += f"- **File Name**: {file_metadata['file_name']}\n"
            file_info_str += f"  **File Path**: {file_metadata['file_path']}\n"
            file_info_str += f"  **Description**: {file_metadata['description']}\n\n"
        return file_info_str
    else:
        return f"(query_files_by_role) No files found for the role '{role}'."