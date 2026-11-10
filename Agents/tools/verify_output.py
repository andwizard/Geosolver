"""Output verification tool for code generator.

Reads output files from the workspace directory, compares them against the
task's expected output specification, and returns a structured multimodal
report to help the LLM quickly assess whether generated results meet requirements.

Only reads files explicitly listed in data_output / visual_output (the task's
Expected_Output fields). Images are returned as multimodal image_url content
blocks (following the pattern in Eval/eval/agent.py's _image_to_content_block).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from langchain_core.tools import tool

# Global reference to the workspace directory, set by geo_asolve
_workspace_dir: str = ""


def set_workspace_dir(path: str) -> None:
    """Set the workspace directory for output verification."""
    global _workspace_dir
    _workspace_dir = path


# ---------------------------------------------------------------------------
# File type helpers
# ---------------------------------------------------------------------------

def _get_file_type(file_path: Path) -> str:
    """Determine file type from extension."""
    suffix = file_path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
        return "image"
    elif suffix in (".tif", ".tiff", ".img", ".bil"):
        return "raster"
    elif suffix in (".shp", ".geojson"):
        return "vector"
    elif suffix in (".txt", ".csv", ".json", ".xml", ".html", ".md"):
        return "text"
    return "unknown"


def _read_image_file(file_path: Path) -> dict:
    """Read an image file and return as a multimodal image_url content block.

    Mirrors Eval/eval/agent.py's _image_to_content_block: compresses to JPEG
    under 2MB, returns a dict suitable for inclusion in multimodal content arrays.
    """
    try:
        from PIL import Image

        # Keep images small — base64 in conversation context is expensive
        # (3MB of base64 ≈ 1M tokens, which blows the context window)
        max_bytes = int(200 * 1024)  # 200KB max
        max_dim = 1600  # downscale any dimension above this

        with Image.open(file_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Downscale large images immediately
            width, height = img.size
            if width > max_dim or height > max_dim:
                scale = max_dim / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_byte_arr = io.BytesIO()
            quality = 85

            # Compress until under max_bytes
            while True:
                img_byte_arr.seek(0)
                img_byte_arr.truncate(0)
                img.save(img_byte_arr, format='JPEG', quality=quality)
                current_size = img_byte_arr.tell()

                if current_size <= max_bytes:
                    break

                if quality > 25:
                    quality -= 15
                else:
                    width, height = img.size
                    new_width = int(width * 0.7)
                    new_height = int(height * 0.7)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = 50

            image_data = img_byte_arr.getvalue()

        image_b64 = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        }

    except ImportError:
        raise RuntimeError(f"Cannot read image {file_path.name}: PIL unavailable")
    except Exception as e:
        raise RuntimeError(f"Cannot read image {file_path.name}: {e}")


def _read_image_info(file_path: Path) -> str:
    """Return text metadata for an image file (dimensions, size)."""
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            return (f"  Dimensions: {img.size[0]}x{img.size[1]}\n"
                    f"  Size: {file_path.stat().st_size:,} bytes")
    except ImportError:
        return f"  Size: {file_path.stat().st_size:,} bytes (PIL unavailable)"
    except Exception as e:
        return f"  Error reading image info: {e}"


def _read_vector_file(file_path: Path) -> str:
    """Read and describe a vector file using geopandas."""
    try:
        import geopandas as gpd

        gdf = gpd.read_file(str(file_path))

        lines = [
            f"  Feature count: {len(gdf)}",
            f"  CRS: {gdf.crs}",
            f"  Geometry types: {list(gdf.geometry.type.unique())}",
            f"  Columns: {list(gdf.columns)}",
            f"  Total bounds: {gdf.total_bounds.tolist()}",
            "",
            f"  --- Preview (first 10 rows) ---",
            gdf.head(10).to_string(),
        ]
        content = "\n".join(lines)
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        return content

    except ImportError:
        return f"  Size: {file_path.stat().st_size:,} bytes (geopandas unavailable)"
    except Exception as e:
        return f"  Error: {e}"


def _read_raster_file(file_path: Path) -> str:
    """Read and describe a raster file using rasterio."""
    try:
        import rasterio
        import numpy as np

        with rasterio.open(str(file_path)) as src:
            lines = [
                f"  Driver: {src.driver}",
                f"  Size: {src.width} x {src.height} pixels",
                f"  Bands: {src.count}",
                f"  CRS: {src.crs}",
                f"  Bounds: {src.bounds}",
                f"  Resolution: {src.res}",
                f"  Dtype: {src.dtypes}",
            ]
            nodata = src.nodata
            if nodata is not None:
                lines.append(f"  Nodata: {nodata}")
            if src.count >= 1:
                # Use masked=True so nodata values become NaN — otherwise
                # np.nanmin() would pick up nodata (e.g. -9999.0) as the min
                data = src.read(1, masked=True)
                # Fill masked (nodata) with NaN for consistent nan* functions
                if hasattr(data, 'filled'):
                    data = data.filled(np.nan)
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
                vmean = np.nanmean(data)
                valid_count = int(np.sum(~np.isnan(data)))
                total_count = int(data.size)
                # Use general format to avoid misleading zeros for very small/large values
                # (.4f would show 5.58e-12 as 0.0000, causing LLM to think raster is broken)
                lines.append(
                    f"  Band 1 stats: min={vmin:.6g}, "
                    f"max={vmax:.6g}, "
                    f"mean={vmean:.6g}"
                )
                if nodata is not None:
                    lines.append(f"  Valid pixels: {valid_count}/{total_count}")
        return "\n".join(lines)

    except ImportError:
        return f"  Size: {file_path.stat().st_size:,} bytes (rasterio unavailable)"
    except Exception as e:
        return f"  Error: {e}"


def _read_text_file(file_path: Path) -> str:
    """Read a text file with length limit."""
    try:
        MAX_CHAR_LENGTH = 8000
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            return "  (empty file)"
        if len(content) > MAX_CHAR_LENGTH:
            content = content[:MAX_CHAR_LENGTH] + (
                f"\n... [truncated, {len(content)} chars total]"
            )
        return content
    except UnicodeDecodeError:
        return f"  Size: {file_path.stat().st_size:,} bytes (binary)"
    except Exception as e:
        return f"  Error: {e}"


def _read_file_content_for_text(file_path: Path) -> str:
    """Read file content as text — for images returns metadata only."""
    file_type = _get_file_type(file_path)

    if file_type == "image":
        return _read_image_info(file_path)
    elif file_type == "vector":
        return _read_vector_file(file_path)
    elif file_type == "raster":
        return _read_raster_file(file_path)
    else:
        return _read_text_file(file_path)


def _resolve_file(workspace: Path, name_or_glob: str) -> list[Path] | None:
    """Resolve a filename or glob pattern to actual files.

    Returns a list of matched files, or None if nothing matched.
    Handles both exact filenames and glob patterns.
    """
    if not name_or_glob:
        return None

    # Try exact match first
    exact = workspace / name_or_glob
    if exact.is_file():
        return [exact]

    # Try glob (recursive — output may be in subdirectories)
    matched = sorted(workspace.glob(f"**/{name_or_glob}"))
    matched = [f for f in matched if f.is_file()]
    return matched if matched else None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def verify_output(
    data_output: str = "",
    visual_output: str = "",
) -> list:
    """Inspect generated output files and compare against expected requirements.

    Only reads files explicitly listed in data_output / visual_output.
    Images are returned as multimodal content blocks for visual inspection.

    Typical usage:
    - verify_output(data_output="result.shp", visual_output="map.png")
      → Checks specific data + visual outputs expected by the task.

    Args:
        data_output: Expected data output filename(s) from the task's Expected_Output.
                     Can be a single name ("result.shp") or glob ("*.geojson").
                     Leave empty if no data output is required.
        visual_output: Expected visual output filename(s) from the task's Expected_Output.
                       Can be a single name ("map.png") or glob ("*.png").
                       Leave empty if no visual output is required.

    Returns:
        A list of multimodal content blocks:
        - {"type": "text", "text": "..."} for the verification report text
        - {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
          for each image output file
    """
    global _workspace_dir

    if not _workspace_dir:
        return [{"type": "text", "text": "ERROR: Workspace directory not set."}]

    workspace = Path(_workspace_dir)
    if not workspace.exists():
        return [{"type": "text", "text": f"ERROR: Workspace directory not found: {_workspace_dir}"}]

    all_files = sorted([f for f in workspace.rglob("*") if f.is_file()])
    all_names = {f.name for f in all_files}

    # ── Resolve expected outputs ──────────────────────────────────────────
    data_files = _resolve_file(workspace, data_output) if data_output else None
    visual_files = _resolve_file(workspace, visual_output) if visual_output else None

    # ── Collect image content blocks ──────────────────────────────────────
    image_blocks: list[dict] = []
    for fp in (visual_files or []):
        if _get_file_type(fp) == "image":
            try:
                image_blocks.append(_read_image_file(fp))
            except RuntimeError:
                pass  # Error already noted in text report via _read_image_info
    for fp in (data_files or []):
        if _get_file_type(fp) == "image":
            try:
                image_blocks.append(_read_image_file(fp))
            except RuntimeError:
                pass

    # ── Build text report ─────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * 55)
    lines.append("VERIFICATION REPORT")
    lines.append("=" * 55)
    lines.append(f"Workspace: {_workspace_dir}")
    lines.append(f"Total files in workspace: {len(all_files)}")
    lines.append("")

    # Expected outputs checklist
    lines.append("--- Expected Outputs Checklist ---")

    if data_output:
        status = "[OK]" if data_files else "[MISSING]"
        lines.append(f"  {status} data_output : {data_output}"
                     + (f" ({len(data_files)} file(s))" if data_files else ""))
    else:
        lines.append("  [N/A] data_output : (not required)")

    if visual_output:
        status = "[OK]" if visual_files else "[MISSING]"
        lines.append(f"  {status} visual_output: {visual_output}"
                     + (f" ({len(visual_files)} file(s))" if visual_files else ""))
    else:
        lines.append("  [N/A] visual_output: (not required)")
    lines.append("")

    # ── Data output section ───────────────────────────────────────────────
    if data_output:
        lines.append("=" * 55)
        lines.append("DATA OUTPUT")
        lines.append("=" * 55)
        if data_files:
            for fp in data_files:
                file_type = _get_file_type(fp)
                lines.append(f"\n>>> {fp.name}  type={file_type}  size={fp.stat().st_size:,}B")
                lines.append("-" * 40)
                lines.append(_read_file_content_for_text(fp))
                lines.append("")
        else:
            lines.append(f"  NOT FOUND: no file matching '{data_output}'")
            lines.append(f"  Available files: {sorted(all_names)}")
            lines.append("")
    else:
        lines.append("=" * 55)
        lines.append("DATA OUTPUT: (not required)")
        lines.append("=" * 55)
        lines.append("")

    # ── Visual output section ─────────────────────────────────────────────
    if visual_output:
        lines.append("=" * 55)
        lines.append("VISUAL OUTPUT")
        lines.append("=" * 55)
        if visual_files:
            for fp in visual_files:
                file_type = _get_file_type(fp)
                lines.append(f"\n>>> {fp.name}  type={file_type}  size={fp.stat().st_size:,}B")
                lines.append("-" * 40)
                lines.append(_read_file_content_for_text(fp))
                if file_type == "image":
                    lines.append("  (image attached below for visual inspection)")
                lines.append("")
        else:
            lines.append(f"  NOT FOUND: no file matching '{visual_output}'")
            lines.append(f"  Available files: {sorted(all_names)}")
            lines.append("")
    else:
        lines.append("=" * 55)
        lines.append("VISUAL OUTPUT: (not required)")
        lines.append("=" * 55)
        lines.append("")

    # ── Footer guidance ───────────────────────────────────────────────────
    lines.append("=" * 55)
    lines.append("HOW TO USE THIS REPORT")
    lines.append("=" * 55)
    lines.append("1. Check the checklist above — any [MISSING] entry means the code")
    lines.append("   did NOT produce a required output file.")
    lines.append("2. For data_output: verify file structure, row count, column names,")
    lines.append("   CRS/geometry types, and key values match the task requirements.")
    lines.append("3. For visual_output: inspect the attached image(s) — check")
    lines.append("   that the map/chart is correctly rendered, properly labeled,")
    lines.append("   and shows the expected spatial patterns or trends.")
    lines.append("4. If any output is missing or incorrect, fix the code and re-run"
                 " execute_code, then call verify_output again.")

    # Assemble final content blocks: text report first, then images
    return [{"type": "text", "text": "\n".join(lines)}] + image_blocks
