import asyncio
import os
import signal
import sys
import tempfile

from langchain_core.tools import tool


class SandboxExecutor:
    @staticmethod
    async def execute_code(code: str = None, work_dir: str = None, timeout: int = 1200, file_path: str = None) -> dict:
        """Execute code in an isolated subprocess and capture stdout/stderr.

        Uses asyncio subprocess so that cancellation (e.g. from an outer
        asyncio.wait_for timeout) properly propagates and kills the child.

        Supports two modes:
        - file_path: Execute an existing Python file directly (no prelude).
        - code: Write code string to a temp file with prelude imports, then execute.

        Returns: {"success": bool, "stdout": str, "stderr": str}
        """
        if work_dir is None:
            work_dir = os.getcwd()

        if file_path is not None:
            # Execute an existing file directly — no temp file, no prelude
            temp_script_path = file_path
            owns_temp_file = False
        else:
            # Write code to a temporary file with prelude imports
            if code is None:
                raise ValueError("Either 'code' or 'file_path' must be provided.")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=work_dir, delete=False) as temp_file:
                pre_code = "import matplotlib\nmatplotlib.use('Agg')\nimport pandas as pd\nimport geopandas as gpd\nimport json\nimport zipfile\nimport rasterio\nimport os\n\n"
                temp_file.write(pre_code + code)
                temp_script_path = temp_file.name
            owns_temp_file = True

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_script_path,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""
            success = proc.returncode == 0
            return {
                "success": success,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }

        except (asyncio.TimeoutError, TimeoutError):
            # Graceful SIGTERM first, then SIGKILL after 5s
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        await proc.wait()
                except ProcessLookupError:
                    pass
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution Timed Out after {timeout} seconds.",
            }

        except asyncio.CancelledError:
            # Outer task was cancelled — forcibly kill the subprocess
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            raise

        except Exception as e:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
            }

        finally:
            # Clean up the temporary script file (only if we created it)
            if owns_temp_file and os.path.exists(temp_script_path):
                os.remove(temp_script_path)


@tool
async def execute_code(code: str, work_dir: str = None) -> str:
    """Execute Python code in an isolated sandbox environment and return stdout/stderr.

    Args:
        code: Python code string to execute.
        work_dir: Working directory for execution (defaults to current directory).

    Returns:
        A string containing stdout, stderr, and success/failure status.
    """
    result = await SandboxExecutor.execute_code(code, work_dir=work_dir, timeout=600)
    status = "SUCCESS" if result["success"] else "FAILED"
    output = [f"[{status}]"]
    MAX_STDOUT_LENGTH = 200000
    if result["stdout"]:
        stdout_content = result["stdout"]
        if len(stdout_content) > MAX_STDOUT_LENGTH:
            stdout_content = stdout_content[:MAX_STDOUT_LENGTH] + "\n...[stdout too long, truncated]"
        output.append(f"--- stdout ---\n{stdout_content}")

    if result["stderr"]:
        stderr_content = result["stderr"]
        if len(stderr_content) > MAX_STDOUT_LENGTH:
            stderr_content = stderr_content[:MAX_STDOUT_LENGTH] + "\n...[stderr too long, truncated]"
        output.append(f"--- stderr ---\n{stderr_content}")
    return "\n".join(output)


@tool
async def execute_file(file_path: str, work_dir: str = None) -> str:
    """Execute an existing Python file directly and return stdout/stderr.

    Unlike execute_code (which writes a code string to a temp file with prelude imports),
    this tool runs a .py file that already exists on disk. No prelude imports are prepended
    — the file must already contain all necessary imports.

    Use this to test a scaffold file after making edits with edit_code.

    Args:
        file_path: Path to an existing .py file to execute.
        work_dir: Working directory for execution (defaults to current directory).

    Returns:
        A string containing stdout, stderr, and success/failure status.
    """
    result = await SandboxExecutor.execute_code(work_dir=work_dir, file_path=file_path)
    status = "SUCCESS" if result["success"] else "FAILED"
    output = [f"[{status}]"]
    MAX_STDOUT_LENGTH = 200000
    if result["stdout"]:
        stdout_content = result["stdout"]
        if len(stdout_content) > MAX_STDOUT_LENGTH:
            stdout_content = stdout_content[:MAX_STDOUT_LENGTH] + "\n...[stdout too long, truncated]"
        output.append(f"--- stdout ---\n{stdout_content}")

    if result["stderr"]:
        stderr_content = result["stderr"]
        if len(stderr_content) > MAX_STDOUT_LENGTH:
            stderr_content = stderr_content[:MAX_STDOUT_LENGTH] + "\n...[stderr too long, truncated]"
        output.append(f"--- stderr ---\n{stderr_content}")
    return "\n".join(output)
