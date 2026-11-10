"""Sandbox for running Python code with the current Python interpreter."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    """Result of code execution."""

    stdout: str
    stderr: str
    return_code: int
    files_created: list[Path] = field(default_factory=list)
    error: str | None = None


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    scratch_dir: Path = Path("scratch")
    timeout: int = 60


class Sandbox:
    """Sandbox for running Python code.

    This sandbox executes Python code with the current Python interpreter
    (``sys.executable``) and saves any generated files to a scratch directory.
    """

    def __init__(self, config: SandboxConfig | None = None):
        """Initialize the sandbox.

        Args:
            config: Sandbox configuration. Uses defaults if not provided.
        """
        self.config = config or SandboxConfig()
        self._scratch_dir = self.config.scratch_dir
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self, session_id: str | None = None) -> Path:
        """Get the output directory for a session.

        Args:
            session_id: Optional session identifier.

        Returns:
            Path to the session's output directory.
        """
        if session_id:
            output_dir = self._scratch_dir / session_id
        else:
            output_dir = self._scratch_dir / next(tempfile._get_candidate_names())
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    async def run(
        self,
        code: str,
        session_id: str | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Run Python code in the sandbox.

        Args:
            code: Python code to execute.
            session_id: Optional session identifier for grouping files.
            timeout: Optional timeout in seconds. Uses config default if not provided.

        Returns:
            ExecutionResult containing stdout, stderr, return code, and files created.
        """
        timeout = timeout or self.config.timeout
        output_dir = self._get_output_dir(session_id)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=output_dir,
            delete=False,
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                sys.executable,
                script_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=output_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    return_code=-1,
                    error="timeout",
                )

            files_created = list(output_dir.glob("*"))
            files_created = [
                f for f in files_created if f.name != Path(script_path).name
            ]

            return ExecutionResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode,
                files_created=files_created,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                error=str(e),
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    def run_sync(
        self,
        code: str,
        session_id: str | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Synchronous version of run().

        Args:
            code: Python code to execute.
            session_id: Optional session identifier for grouping files.
            timeout: Optional timeout in seconds. Uses config default if not provided.

        Returns:
            ExecutionResult containing stdout, stderr, return code, and files created.
        """
        timeout = timeout or self.config.timeout
        output_dir = self._get_output_dir(session_id)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=output_dir,
            delete=False,
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                sys.executable,
                script_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=output_dir,
            )

            files_created = list(output_dir.glob("*"))
            files_created = [
                f for f in files_created if f.name != Path(script_path).name
            ]

            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                files_created=files_created,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                return_code=-1,
                error="timeout",
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                error=str(e),
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    def cleanup(self, session_id: str | None = None) -> None:
        """Clean up scratch directory.

        Args:
            session_id: Optional session ID. If provided, only cleans that session's directory.
        """
        if session_id:
            session_dir = self._scratch_dir / session_id
            if session_dir.exists():
                import shutil

                shutil.rmtree(session_dir)
        else:
            import shutil

            if self._scratch_dir.exists():
                shutil.rmtree(self._scratch_dir)
            self._scratch_dir.mkdir(parents=True, exist_ok=True)
