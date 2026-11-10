"""File read/edit tools for the Code Generator agent.

Provides targeted file operations so the Code Generator can:
- Read specific parts of a persistent scaffold file to understand current state
- Make surgical edits to fix individual TODO blocks
- Iterate on the file without regenerating the entire script each time

Mirrors Claude Code's Read/Edit tool semantics.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# read_code
# ---------------------------------------------------------------------------

@tool
def read_code(file_path: str) -> str:
    """Read a Python file from disk and return its full contents.

    Use this to inspect the scaffold file before making edits, or to check
    specific sections of code after an execution failure.

    Args:
        file_path: Absolute or relative path to the Python file to read.

    Returns:
        The file contents with a header showing file path and line count.
        Returns an error message if the file is not found.
    """
    path = Path(file_path)
    if not path.exists():
        return f"ERROR: File not found: {file_path}"
    if not path.is_file():
        return f"ERROR: Path is not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    lines = content.split("\n")
    line_count = len(lines)
    MAX_LINES = 1000
    MAX_CHARS = 100000

    if line_count > MAX_LINES or len(content) > MAX_CHARS:
        # Truncate and show which parts are included
        truncated_lines = lines[:MAX_LINES]
        truncated_content = "\n".join(truncated_lines)
        if len(truncated_content) > MAX_CHARS:
            truncated_content = truncated_content[:MAX_CHARS]
        header = (
            f"=== {file_path} ===\n"
            f"Lines: {line_count} total (showing first ~{len(truncated_content.split(chr(10)))} due to size limit)\n"
            f"Chars: {len(content):,} total (truncated to {len(truncated_content):,})\n\n"
        )
        return header + truncated_content + "\n\n... [file truncated]"

    header = f"=== {file_path} ===\nLines: {line_count}\n\n"
    return header + content


# ---------------------------------------------------------------------------
# edit_code
# ---------------------------------------------------------------------------

@tool
def edit_code(file_path: str, old_string: str, new_string: str) -> str:
    """Replace a specific string in a file with a new string.

    Mirrors Claude Code's Edit tool semantics:
    - old_string must exist exactly once in the file (unique match).
    - If old_string is not unique, the edit is rejected with an error listing
      the line numbers of all matches.
    - If old_string is not found, the edit is rejected.

    Use this to replace TODO blocks with actual implementations, fix errors,
    or update specific sections without rewriting the entire file.

    Args:
        file_path: Absolute or relative path to the file to edit.
        old_string: The exact text to replace (must be unique in the file).
        new_string: The replacement text.

    Returns:
        Confirmation of the change with surrounding context, or an error
        message if the edit could not be applied.
    """
    path = Path(file_path)
    if not path.exists():
        return f"ERROR: File not found: {file_path}"
    if not path.is_file():
        return f"ERROR: Path is not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    # Count occurrences
    count = content.count(old_string)
    if count == 0:
        return (
            f"ERROR: old_string not found in {file_path}.\n"
            f"Make sure the string matches exactly (whitespace, indentation, etc.)."
        )
    if count > 1:
        # Find line numbers of all matches for diagnostics
        lines = content.split("\n")
        match_lines = []
        for i, line in enumerate(lines, start=1):
            if old_string in line:
                match_lines.append(str(i))
        return (
            f"ERROR: old_string is not unique — found {count} occurrences in {file_path}.\n"
            f"Matching lines: {', '.join(match_lines[:20])}"
            + (f" ... and {count - 20} more" if count > 20 else "") + "\n"
            f"Make old_string more specific (include more surrounding context) to target a single occurrence."
        )

    # Apply the edit
    new_content = content.replace(old_string, new_string, 1)
    path.write_text(new_content, encoding="utf-8")

    # Build context for confirmation
    old_lines = old_string.strip().split("\n")
    new_lines = new_string.strip().split("\n")
    preview_old = "\n".join(old_lines[:3])
    preview_new = "\n".join(new_lines[:3])
    if len(old_lines) > 3:
        preview_old += f"\n... ({len(old_lines)} lines total)"
    if len(new_lines) > 3:
        preview_new += f"\n... ({len(new_lines)} lines total)"

    return (
        f"Successfully edited {file_path}.\n\n"
        f"--- Replaced ---\n{preview_old}\n\n"
        f"--- With ---\n{preview_new}"
    )
