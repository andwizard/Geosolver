import sys

import matplotlib
matplotlib.use('Agg')
import os
import logging
import time
from typing import Annotated
from pathlib import Path
import inspect
from importlib import import_module

import subprocess
from langchain_core.tools import tool
from utils.logger import get_logger_with_session
from langchain_experimental.utilities import PythonREPL

CONDA_PATH = os.environ['CONDA_PATH']
CONDA_ENV = os.environ['CONDA_ENV']

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """The user wants to see the output of the code. You must show the output to the user."""
    try:
        result = repl.run(code) ## Effectively runs de code
        print('\n',code,'\n')
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

@tool
def query_function_document(
    function_name: Annotated[str, "The name of the function to query, typically from the GIS library."]
) -> Annotated[str, "The documentation string for the specified function or an error message if retrieval fails."]:
    """
    Query the documentation of a function.

    This function retrieves the documentation string of a specified function by its name. 
    It is especially useful for GIS functions like `sjoin` to understand their usage.
    Avoid abbreviations for libraries and use the full name. (e.g. use geopandas instead of gpd, use pandas.DataFrame.to_csv instead of pd.DataFrame.to_csv)
    Only query the function name, not the class name. (e.g. geopandas.GeoDataFrame is not a valid function name)

    Args:
        function_name (str): The name of the function whose documentation is to be retrieved.
            The function should be in the current namespace or importable.

    Returns:
        str: The documentation string for the specified function. If the function is not found or an error occurs,
        an error message is returned.

    Examples:
        Example 1: Query documentation for `sjoin`
        >>> query_function_document("geopandas.sjoin") 
        # Output:
        # Documentation for sjoin:
        # Perform spatial join between GeoDataFrames...
    """
    logger = get_logger_with_session()
    logger.info(f"(query_function_document) Querying documentation for function: {function_name}")
    try:
        # Resolve a dotted path that may include nested attributes, e.g.
        # "geopandas.GeoDataFrame.to_file": import the longest importable
        # module prefix ("geopandas"), then walk the remaining attributes
        # (GeoDataFrame -> to_file). The old rsplit('.', 1) tried to import
        # "geopandas.GeoDataFrame" as a module and always failed on such paths.
        parts = function_name.split('.')
        module = None
        module_name = function_name
        rest = []
        for _i in range(len(parts) - 1, 0, -1):
            try:
                module = import_module('.'.join(parts[:_i]))
                module_name = '.'.join(parts[:_i])
                rest = parts[_i:]
                break
            except ImportError:
                continue
        if module is None:
            raise ImportError(module_name)
        func_name = rest[-1] if rest else module_name
        
        # Walk the remaining attributes (class -> method, etc.)
        func = module
        for attr in rest:
            func = getattr(func, attr)
        
        # Use inspect to get documentation string
        documentation = inspect.getdoc(func)
        if not documentation:
            documentation = "No documentation available."
        
        return f"Documentation for {function_name}:\n{documentation}"
    
    except ValueError:
        logger.error("(query_function_document) Invalid function name format. Please use format 'module.function'")
        return "Error: Invalid function name format. Please use format 'module.function'"
    except ImportError:
        logger.error(f"(query_function_document) Module {module_name} not found")
        return f"Error: Module {module_name} not found"
    except AttributeError:
        logger.error(f"(query_function_document) Function {func_name} not found in module {module_name}")
        return f"Error: Function {func_name} not found in module {module_name}"
    except Exception as e:
        logger.error(f"(query_function_document) Failed to retrieve documentation. Error: {repr(e)}")
        return f"Failed to retrieve documentation. Error: {repr(e)}"

@tool
def list_files_in_codebase(
    codebase_path: Annotated[str, "Path to the codebase directory"] = "codebase"
) -> Annotated[str, "A list of Python filenames in the codebase folder or an error message if the folder is not found."]:
    """
    List the contents of the specified codebase directory.

    This function scans the given directory for Python files (`.py`) and returns a list of their names.
    It is useful for understanding what scripts are available in the folder for reference or execution.

    Args:
        codebase_path (str): Path to the codebase directory. Defaults to "codebase".

    Returns:
        str: A string listing the Python filenames in the specified folder.
             If the folder doesn't exist or an error occurs, an error message is returned.

        Example 1: List Python files in a custom codebase path
        >>> list_files_in_codebase("path/to/codebase")
        # Output:
        # Available Python files in 'path/to/codebase':
        # - file1.py
        # - file2.py
    """
    logger = get_logger_with_session()
    logger.info(f"Listing Python files in the '{codebase_path}' folder.")

    try:
        # Check if the folder exists
        if not os.path.exists(codebase_path):
            return f"The folder '{codebase_path}' does not exist."

        # Get all .py files in the folder
        py_files = [f for f in os.listdir(codebase_path) if f.endswith(".py")]

        if py_files:
            tool_info = f"Available Python files in '{codebase_path}':\n" + "\n".join(f"- {file}" for file in py_files)
        else:
            tool_info = f"No Python files found in the '{codebase_path}' folder."
        # logger.info(tool_info)
        return tool_info

    except Exception as e:
        logger.error(f"Failed to retrieve Python files. Error: {repr(e)}")
        return f"Failed to retrieve Python files. Error: {repr(e)}"

@tool
def read_codebase_code(
    codefile_name: Annotated[str, "The Python code file name or full path want to read."]
) -> str:
    """
    Read a Python code file and return its content.

    This function reads the content of a specified Python code file and formats it for return.
    If the file does not exist or an error occurs during reading, it returns an error message.

    Args:
        codefile_name (str): The name or full path of the Python code file to read.

    Returns:
        str: The content of the Python code file formatted as a string. If an error occurs, 
        an error message is returned instead.

    Examples:
        Example 1: Read a valid Python file
        >>> read_code("example_script.py")
        # Output:
        # def example_function():
        #     print("This is an example script.")
    """
    logger = get_logger_with_session()
    logger.info(f"(read_codefile_name) Reading code file: {codefile_name}")

    try:
        # Ensure file path includes codebase path
        if 'codebase' not in codefile_name:
            file_path = os.path.join('codebase', codefile_name)
        else:
            file_path = codefile_name
            
        # Add .py extension if missing
        if not file_path.endswith('.py'):
            file_path += '.py'
            
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: The file {codefile_name} was not found."
            
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            code_content = file.read()
            
        # Format return content
        formatted_content = f"""
        {code_content}
        """
        # logger.info(f"Successfully read code file: {file_path}")
        return formatted_content
    except Exception as e:
        error_msg = f"Error reading code file: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def read_code(
    codefile_name: Annotated[str, "The Python code file name or full path want to read."]
) -> str:
    """
    Read a Python code file and return its content.

    This function reads the content of a specified Python code file and formats it for return.
    If the file does not exist or an error occurs during reading, it returns an error message.

    Args:
        codefile_name (str): The name or full path of the Python code file to read.

    Returns:
        str: The content of the Python code file formatted as a string. If an error occurs, 
        an error message is returned instead.

    Examples:
        Example 1: Read a valid Python file
        >>> read_code("example_script.py")
        # Output:
        # def example_function():
        #     print("This is an example script.")
    """
    logger = get_logger_with_session()
    try:
        # Ensure WORKING_DIRECTORY exists
        working_directory = os.environ["WORKING_DIRECTORY"]
        os.makedirs(working_directory, exist_ok=True)
        
        # Process codefile_name to ensure valid path
        file_path = Path(working_directory) / codefile_name
        file_path = file_path.resolve()
            
        # Add .py extension if missing
        # if not file_path.endswith('.py'):
        #     file_path += '.py'
        
        logger.info(f"Reading code file: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: The file {codefile_name} was not found."
            
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            code_content = file.read()
            
        # Format return content
        formatted_content = f"""
        {code_content}
        """
        logger.info(f"(read_code) Successfully read code file: {file_path}")
        return formatted_content
    except Exception as e:
        error_msg = f"(read_code) Error reading code file: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def write_code(
    input_code: Annotated[str, "The Python code to write."],
    codefile_name: Annotated[str, "The Python code file name or full path."] = 'code.py'  
):
    """
    Write Python code to a file in a specified working directory.

    This function ensures that the designated working directory exists, writes the provided Python code 
    to a specified file, and logs the process. If an error occurs during the execution, it returns 
    details about the error and the file path.

    Args:
        input_code (str): The Python code to be written to a file.
        codefile_name (str, optional): The name of the file to save the code in, or the full path. 
            Defaults to 'code.py'.

    Returns:
        dict: If an error occurs, a dictionary is returned containing:
            - "result" (str): Indicates that an error occurred.
            - "error" (str): A description of the error.
            - "file_path" (str): The path to the file where the code was written or attempted to be written.
    """
    logger = get_logger_with_session()
    try:
        # Ensure WORKING_DIRECTORY exists
        working_directory = os.environ["WORKING_DIRECTORY"]
        os.makedirs(working_directory, exist_ok=True)
        
        # Process codefile_name to ensure valid path
        code_file_path = Path(working_directory) / codefile_name
        code_file_path = code_file_path.resolve()
        
        # Write code to file
        with open(code_file_path, 'w', encoding='utf-8') as code_file:
            code_file.write(input_code)
        
        logger.info(f"(write_code) Code has been written to file: {code_file_path}")

        return {
            "result": "Code written successfully",
            "file_path": str(code_file_path)
        }

    except Exception as e:
        logger.error("An error occurred while executing code")
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": str(code_file_path) if 'code_file_path' in locals() else "Unknown"
        }

@tool
def edit_code(
    input_code: Annotated[str, "The Python code to write."],
    codefile_name: Annotated[str, "The Python code file name or full path."] = 'code.py'  
):
    """
    Edit an existing Python code file. If the file doesn't exist, create a new one.

    Args:
        input_code (str): The Python code to be written to the file.
        codefile_name (str, optional): The name of the file to edit, or the full path. 
            Defaults to 'code.py'.

    Returns:
        dict: A dictionary containing:
            - "result" (str): Indicates the operation result.
            - "error" (str): Description of the error if one occurred.
            - "file_path" (str): The path to the edited file.
    """
    logger = get_logger_with_session()
    try:
        # Ensure WORKING_DIRECTORY exists
        working_directory = os.environ["WORKING_DIRECTORY"]
        os.makedirs(working_directory, exist_ok=True)
        
        # Process codefile_name to ensure valid path
        code_file_path = Path(working_directory) / codefile_name
        code_file_path = code_file_path.resolve()
        
        # Write the new code
        with open(code_file_path, 'w', encoding='utf-8') as code_file:
            code_file.write(input_code)
        
        logger.info(f"(edit_code) Code has been updated in file: {code_file_path}")

        return {
            "result": "Code updated successfully",
            "file_path": str(code_file_path)
        }

    except Exception as e:
        logger.error("An error occurred while editing code")
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": str(code_file_path) if 'code_file_path' in locals() else "Unknown"
        }

@tool
def execute_code_file(
    codefile_name: Annotated[str, "The Python code file name or full path want to execute."]
):
    """
    Execute a Python code file in a specified Conda environment and return the result.

    This function runs a Python script in a predefined Conda environment, capturing the output
    and any errors that occur during execution. The execution is performed within a specified
    working directory, ensuring proper environment activation and file accessibility.

    Args:
        codefile_name (str): The name or full path of the Python code file to execute.

    Returns:
        dict: A dictionary containing:
            - "result" (str): Execution status, either "Code executed successfully" or "Failed to execute."
            - "output" (str): The standard output from the execution, if available.
            - "error" (str): The standard error output if execution fails.
            - "file_path" (str): The full path of the code file executed.
            - "code_content" (str): The content of the code file.
    """
    logger = get_logger_with_session()
    logger.info(f"(execute_code_file) Executing and readingcode file: {codefile_name}")
    try:

        # Ensure WORKING_DIRECTORY exists
        working_directory = os.environ["WORKING_DIRECTORY"]
        os.makedirs(working_directory, exist_ok=True)
        
        # Process codefile_name to ensure valid path
        code_file_path = Path(working_directory) / codefile_name
        code_file_path = code_file_path.resolve()

        # Read code file content
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as read_error:
            logger.error(f"(execute_code_file) Failed to read code file: {read_error}")
            code_content = ""


        # 进程已运行在目标 conda 环境内：非 Windows 直接用当前解释器执行，无需 activate（跨平台）。
        if sys.platform.startswith("win"):
            activate_cmd = Path(CONDA_PATH) / "Scripts" / "activate.bat"
            full_command = ' '.join([str(activate_cmd), CONDA_ENV, "&&", "python", str(code_file_path)])
            logger.info(f"(execute_code_file) Executing command: {full_command}")
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_directory,
                encoding='utf-8',
                errors='ignore'  # Ignore characters that cannot be decoded
            )
        else:
            logger.info(f"(execute_code_file) Executing: {sys.executable} {code_file_path}")
            result = subprocess.run(
                [sys.executable, str(code_file_path)],
                capture_output=True,
                text=True,
                cwd=working_directory,
                encoding='utf-8',
                errors='ignore'  # Ignore characters that cannot be decoded
            )
        
        # Capture standard output and error output
        output = result.stdout or ""  # Use empty string if stdout is None
        # logger.info(f"ExecutingOutput: {output}")
        error_output = result.stderr
        
        if result.returncode == 0:
            logger.info("(execute_code_file) Code executed successfully")
            return {
                "result": "Code executed successfully",
                "output": output + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.",
                "file_path": str(codefile_name),
                "code_content": f"""
                {code_content}
                """
            }
        else:
            logger.error(f"Code execution failed: {error_output}")
            return {
                "result": "Failed to execute",
                "error": error_output,
                "file_path": str(codefile_name),
                "code_content": f"""
                {code_content}
                """
            }
    except Exception as e:
        logger.exception("An error occurred while executing code")
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": str(codefile_name) if 'codefile_name' in locals() else "Unknown",
            "code_content": ""
        }

@tool
def execute_code(
    input_code: Annotated[str, "The Python code to execute."],
    codefile_name: Annotated[str, "The Python code file name or full path."] = 'code.py'
):
    """
    Execute Python code in a specified conda environment and return the result.

    This function takes Python code as input, writes it to a file, executes it in the specified
    conda environment, and returns the output or any errors encountered during execution.

    Args:
    input_code (str): The Python code to be executed.
    codefile_name (str): The name of the file to save the code in, or the full path.

    Returns:
    dict: A dictionary containing the execution result, output, and file path.
    """
    logger = get_logger_with_session()
    try:
        # Record start time
        start_time = time.time()

        # Ensure WORKING_DIRECTORY exists
        working_directory = os.environ["WORKING_DIRECTORY"]
        os.makedirs(working_directory, exist_ok=True)
        
        # Process codefile_name to ensure valid path
        code_file_path = Path(working_directory) / codefile_name
        code_file_path = code_file_path.resolve()

        logger.info(f"(execute_code) Code will be written to file: {code_file_path}")
        
        # Write code to file
        with open(code_file_path, 'w', encoding='utf-8') as code_file:
            code_file.write(input_code)
        
        logger.info(f"(execute_code) Code has been written to file: {code_file_path}")
        
        # Build Conda activation and Python command for Windows
        activate_cmd = Path(CONDA_PATH) / "Scripts" / "activate.bat"
        
        is_windows = sys.platform.startswith("win")

        if is_windows:
            # Execute activation command
            subprocess.run([str(activate_cmd), CONDA_ENV], shell=True, check=True)
            
            # Execute Python code
            result = subprocess.run(
                ["python", str(code_file_path)],
                capture_output=True,
                text=True,
                cwd=working_directory,
                encoding='utf-8',
                errors='ignore'  # Ignore characters that cannot be decoded
            )
        else:       
            CONDA_PYTHON = sys.executable

            result = subprocess.run(
                [CONDA_PYTHON, str(code_file_path)],
                capture_output=True,
                text=True,
                cwd=working_directory,
                encoding='utf-8',
                errors='ignore'  # Ignore characters that cannot be decoded
            )
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Capture standard output and error output
        output = result.stdout or ""  # Use empty string if stdout is None
        error_output = result.stderr
        
        if result.returncode == 0:
            logger.info(f"(execute_code) Code executed successfully in {execution_time:.2f} seconds")
            return {
                "result": "Code executed successfully",
                "output": output + f"\n\nExecution time: {execution_time:.2f} seconds\n\nIf you have completed all tasks, respond with FINAL ANSWER.",
                "file_path": str(code_file_path),
                "execution_time": execution_time
            }
        else:
            logger.error(f"(execute_code) Code execution failed in {execution_time:.2f} seconds: {error_output}")
            return {
                "result": "Failed to execute",
                "error": error_output,
                "file_path": str(code_file_path),
                "execution_time": execution_time
            }
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.exception(f"(execute_code) An error occurred while executing code (took {execution_time:.2f} seconds)")
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": str(code_file_path) if 'code_file_path' in locals() else "Unknown",
            "execution_time": execution_time
        }

@tool
def execute_command(
    command: Annotated[str, "Command to be executed."]
) -> Annotated[str, "Output of the command."]:
    """
    Execute a command in a specified Conda environment and return its output. (Available for Windows operating system)

    This function activates a Conda environment , executes the given command,
    and returns the output or any errors encountered during execution.
    Please use pip to install the package.
    Args:
    command (str): The command to be executed in the Conda environment.

    Returns:
    str: The output of the command or an error message.
    """
    logger = get_logger_with_session()
    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        # 进程已运行在目标 conda 环境内：非 Windows 直接执行命令即可（子进程继承当前环境），无需 activate。
        if sys.platform.startswith("win"):
            activate_cmd = Path(CONDA_PATH) / "Scripts" / "activate.bat"
            full_command = ' '.join([str(activate_cmd), CONDA_ENV, "&&", command])
        else:
            full_command = command

        logger.info(f"(execute_command) Executing command: {command}")

        # Execute command and capture output
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_directory,
            encoding='utf-8',
            errors='ignore'  # Ignore characters that cannot be decoded
        )
        logger.info("(execute_command) Command executed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"(execute_command) Error executing command: {e.stderr}")
        return f"Error: {e.stderr}"