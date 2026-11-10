"""Claude Code CLI agent builder and invocation.

Uses the Claude Code CLI subprocess to solve geospatial analysis tasks.
Follows the same BASIC_PROMPT + specialized prompt pattern as other agents.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Agents.prompts.basic_prompts import BASIC_PROMPT
from Agents.prompts.claude_code_prompts import CLAUDE_CODE_PROMPT


# ── Structured output schema for Claude Code CLI ─────────────────────────────

CLAUDE_CODE_OUTPUT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "response": {"type": "string", "enum": ["ACCEPT", "FAILED"]},
        "script":   {"type": "string"},
        "answer":   {"type": "string"},
    },
    "required": ["response", "script"],
})


# ── Prompt construction ─────────────────────────────────────────────────────

def _build_claude_prompt(task_text: str, used_files: list, data_dir: str,
                         expected_output: dict, output_dir: str) -> str:
    """Build the prompt string for the Claude Code CLI.

    Combines the shared BASIC_PROMPT (used by all agents) with the
    Claude Code-specific CLAUDE_CODE_PROMPT.
    """
    file_list = "\n".join(f"  - {f}" for f in used_files)
    basic = BASIC_PROMPT.format(
        task_text=task_text,
        file_list=file_list,
        expected_output=json.dumps(expected_output),
        output_dir=str(output_dir),
    )
    return basic + "\n" + CLAUDE_CODE_PROMPT


# ── Output parsing ──────────────────────────────────────────────────────────

def _parse_claude_stream_output(stdout_text: str) -> dict:
    """Parse Claude's stream-json NDJSON output.

    Extracts StructuredOutput tool_use input and token usage from result events.
    Falls back to regex JSON extraction if the structured output is not found.

    Returns dict with 'response', 'script', 'answer', 'trajectory', and
    optionally 'usage'.
    """
    usage = None
    structured_output = None
    trajectory = []

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            trajectory.append(event)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "result":
            usage = event.get("usage")

        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                    inp = block.get("input", {})
                    structured_output = {
                        "response": inp.get("response", "FAILED"),
                        "script":   inp.get("script", ""),
                        "answer":   inp.get("answer", ""),
                        "trajectory": trajectory,
                    }
                    # Heuristic: a very short script value is likely a filename, not code
                    if structured_output["script"] and len(structured_output["script"]) < 50:
                        structured_output["response"] = "FAILED"
                        structured_output["script"] = f"Invalid script output: {structured_output['script']}"

    if structured_output is not None:
        if usage:
            structured_output["usage"] = usage
        return structured_output

    # ── fallback: try to extract any JSON object from the raw output ──────────
    m = re.search(r'\{[\s\S]*?\}', stdout_text)
    if m:
        try:
            obj = json.loads(m.group())
            result = {
                "response": obj.get("response", "FAILED"),
                "script":   obj.get("script", ""),
                "answer":   obj.get("answer", ""),
                "trajectory": trajectory,
            }
            if usage:
                result["usage"] = usage
            return result
        except json.JSONDecodeError:
            pass

    result = {
        "response": "FAILED",
        "script": f"could not parse output: {stdout_text[:300]}",
        "answer": "",
        "trajectory": trajectory,
    }
    if usage:
        result["usage"] = usage
    return result


# ── Agent invocation ────────────────────────────────────────────────────────

async def claude_code_ainvoke(task: dict, data_root: str, task_folder: str,
                              timeout_seconds: int) -> dict:
    """Run a single task via the Claude Code CLI subprocess.

    Returns a dict with 'status', 'script', 'answer', 'trajectory', and
    optionally 'usage'.
    """
    task_text = task["task_text"]
    used_files = task.get("used_dataset", [])
    expected_output = task.get("Expected_Output", {})
    source = task.get("source", "")
    data_dir = os.path.join(data_root, source) if source else data_root

    prompt = _build_claude_prompt(task_text, used_files, data_dir,
                                  expected_output, task_folder)

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--json-schema", CLAUDE_CODE_OUTPUT_SCHEMA,
        "--dangerously-skip-permissions",
        "--verbose",
    ]

    proc = None
    try:
        proc = await asyncio.subprocess.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds,
        )
        stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        return {"status": "TIMEOUT", "script": "", "answer": "", "trajectory": ""}
    except Exception as exc:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return {"status": "FAILED", "script": "", "answer": str(exc), "trajectory": ""}

    if proc.returncode != 0:
        err = (stderr_text or stdout_text or "unknown error")[:400]
        return {"status": "FAILED", "script": "", "answer": f"claude exit {proc.returncode}: {err}", "trajectory": ""}

    result = _parse_claude_stream_output(stdout_text)
    trajectory_raw = result.get("trajectory", [])
    trajectory_str = json.dumps(trajectory_raw, indent=2, ensure_ascii=False) if isinstance(trajectory_raw, list) else str(trajectory_raw)
    return {
        "status":     result.get("response", "FAILED"),
        "script":     result.get("script", ""),
        "answer":     result.get("answer", ""),
        "trajectory": trajectory_str,
        "usage":      result.get("usage"),
    }


def build_claude_code():
    """No-op: claude_code uses the CLI subprocess, no persistent agent needed."""
    return None
