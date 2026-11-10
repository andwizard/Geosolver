"""Remote sensing evaluation agent based on OpenAI Response API."""

from __future__ import annotations

import base64
import io
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image

# Generated maps can be very large (hundreds of megapixels). These are our own
# trusted output files, so disable Pillow's decompression-bomb guard — otherwise
# valid but oversized images raise DecompressionBombError and can't be evaluated.
Image.MAX_IMAGE_PIXELS = None

from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
# Role & Responsibilities
You are an authoritative Remote Sensing Task Evaluation Expert. Your core responsibility is to objectively and comprehensively evaluate whether AI-generated code and its execution results accurately meet the user's original remote sensing task requirements, and to conduct an in-depth comparison with the reference solution.

# Core Capabilities
1. Deeply understand the logic of various remote sensing tasks (e.g., classification, object detection, change detection, band math).
2. Possess the ability to read and analyze the attributes and structures of geospatial data (Satellite Imagery, Shapefiles, GeoTIFFs, etc.).
3. Accurately compare the similarities and differences between the generated code and the reference code in terms of algorithmic implementation and output results.
4. Provide multi-dimensional, structured, and actionable quantitative evaluations.

# Evaluation Dimensions & Scoring Criteria (0-3 Scale)

## 1. Task Consistency
Evaluate whether the generated content accurately responds to and complies with the user's instructions (geographical scope, data sources, specific constraints, etc.).
* 3 Points: Fully compliant. Precisely followed all instructions with no deviations.
* 2 Points: Mostly compliant. Followed the main instructions but has minor flaws in secondary constraints (e.g., unspecified CRS, naming conventions).
* 1 Point: Partially compliant. Deviated from key instructions or violated explicit constraints specified by the user.
* 0 Points: Fails to comply. Completely failed to understand or execute the user's requirements.

## 2. Result Completeness
Evaluate whether the output files and intermediate products are complete, comparing the differences with the reference solution.
* 3 Points: Fully complete. No missing files or results; the output is almost identical to the reference solution.
* 2 Points: Mostly complete. Contains the core required results, but has slight differences from the reference solution in non-critical attributes or fields.
* 1 Point: Incomplete. Covers basic logic but shows massive discrepancies from the reference solution, or is missing some specified output files.
* 0 Points: Severely incomplete. Failed to generate the core results required for the task.

## 3. Accuracy
Evaluate the consistency of boundary detection, categorization, spatial topology, and key numerical values against the reference answer.
* 3 Points: Highly accurate. Completely identical to the reference answer, or the error falls within a reasonable and minimal tolerance range.
* 2 Points: Minor errors. Slight deviations exist, but they do not affect the overall achievement of the task goals and conclusions.
* 1 Point: Significant differences. Evident discrepancies from the reference answer, severely impacting the usability of the results.
* 0 Points: Completely incorrect. Missing answers for key numerical values, or the results deviate entirely from the reference answer.

## 4. Visualization Quality
Evaluate the clarity, color scheme rationality, spatial scale, and completeness of legends/annotations in charts and image renderings.
* 3 Points: Excellent quality. Clear and readable, scientifically colored (adhering to remote sensing cartography standards), with complete and professional annotations.
* 2 Points: Good quality. Basically meets visualization needs, but has very minor aesthetic or layout flaws.
* 1 Point: Poor quality. Obvious errors such as element misalignment, improper scaling, missing legends, or illegible color schemes.
* 0 Points: Invalid or missing. Failed to fulfill visualization requirements, or the output is completely uninterpretable.

# Output Format
You must strictly follow the JSON format below. Do not output any additional explanatory text outside the Markdown code block. Do not add any comments (e.g., `//` or `#`) inside the JSON.

```json
{
  "task_consistency": {
    "score": 0,
    "comments": "Detailed evaluation reasoning and analysis"
  },
  "result_completeness": {
    "score": 0,
    "comments": "Detailed evaluation reasoning and analysis"
  },
  "accuracy": {
    "score": 0,
    "comments": "Detailed evaluation reasoning and analysis"
  },
  "visualization_quality": {
    "score": 0,
    "comments": "Detailed evaluation reasoning and analysis. If the task does not require visualization, state this and score 0."
  },
  "summary": "Overall performance summary based on the four dimensions above",
  "suggestions": [
    "Specific and actionable improvement suggestion 1",
    "Specific and actionable improvement suggestion 2"
  ],
  "reference_vs_generated": {
    "strengths": ["Advantages of the generated solution compared to the reference"],
    "weaknesses": ["Disadvantages of the generated solution compared to the reference"],
    "differences": ["Analysis of core differences in implementation logic or results between the two"]
  }
}

# Important Notes
Please uniformly assign 3 points when the task has no visualization requirements.

Even if a dimension scores 0, a clear reason must be provided in the comments (e.g., "The current task does not require visualization, hence scored 0").

Ensure the output is a strictly valid JSON string that can be directly parsed by code extractors.
"""

FIX_CODE_PTOMPT = """
You will receive a piece of code with an incorrect path. Please fix the path issue of this code according to the path provided by the user, and do not make any changes to other parts of the code.
"""


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_OPEN_FILE = {
    "type": "function",
    "name": "open_file",
    "description": """Open and read the file content. Different file types are handled differently:

1. **Image files** (png, jpg, jpeg, bmp): Use a multimodal model to analyze image content, which can be remote sensing imagery or data analysis charts (heatmaps, route maps, statistical charts, etc.), returning a natural language description
2. **Remote sensing raster files** (tif, tiff, img, bil): Use code to read GeoTIFF metadata (coordinates, resolution, bands, etc.)
3. **Vector files** (shp, geojson): Use code to read vector data attribute tables and geometry information
4. **Other files**: Attempt to read text content

Return a natural language description of the file content.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to open (relative or absolute)",
            }
        },
        "required": ["file_path"],
    },
}

# =============================================================================
# Agent Implementation
# =============================================================================

EVALUATION_ATTRIBUTE_DESCRIPTIONS: dict[str, str] = {
    "task_consistency": "Task Consistency — Evaluates whether the generated solution accurately understood the task requirements and whether the output format meets expectations",
    "result_completeness": "Result Completeness — Evaluates whether the task result is complete and includes all required outputs",
    "accuracy": "Accuracy — Evaluates how closely the generated result matches the reference result and whether the numerical precision meets requirements",
    "visualization_quality": "Visualization Quality — Evaluates whether the generated visualization is clear, aesthetically pleasing, and informative",
}


@dataclass
class EvaluationResult:
    """Evaluation result for a remote sensing task."""

    task_consistency: dict
    result_completeness: dict
    accuracy: dict
    visualization_quality: dict
    overall_score: float
    summary: str
    suggestions: list[str] = field(default_factory=list)
    reference_vs_generated: dict = field(default_factory=dict)
    reference_execution: dict | None = None
    generated_execution: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "attribute_descriptions": EVALUATION_ATTRIBUTE_DESCRIPTIONS,
            "task_consistency": self.task_consistency,
            "result_completeness": self.result_completeness,
            "accuracy": self.accuracy,
            "visualization_quality": self.visualization_quality,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "suggestions": self.suggestions,
            "reference_vs_generated": self.reference_vs_generated,
        }
        if self.generated_execution is not None:
            result["generated_execution"] = self.generated_execution
        return result


class RemoteSensingEvaluator:
    """Agent for evaluating remote sensing task results.

    This agent uses OpenAI's Response API to evaluate whether generated
    code and its results match the task description, comparing against
    reference code when provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "qwen3.5-plus",
        multimodal_model: str = "qwen3.5-plus",
        scratch_dir: str | Path = "scratch",
    ):
        """Initialize the evaluator.

        Args:
            api_key: OpenAI API key. Defaults to env OPENAI_API_KEY.
            api_base: API base URL. Defaults to env OPENAI_API_BASE.
            model: Model for text evaluation.
            multimodal_model: Model for image analysis.
            scratch_dir: Directory for sandbox execution outputs.
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = api_base or config.OPENAI_API_BASE
        self.model = model
        self.multimodal_model = multimodal_model
        self.scratch_dir = Path(scratch_dir)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # Token usage tracker
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }

        from sandbox.runner import Sandbox, SandboxConfig

        self._sandbox = Sandbox(SandboxConfig(scratch_dir=self.scratch_dir))

    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        suffix = file_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            return "image"
        elif suffix in [".tif", ".tiff", ".img", ".bil"]:
            return "raster"
        elif suffix in [".shp", ".geojson", ".json"]:
            return "vector"
        elif suffix in [".txt", ".csv", ".json", ".xml", "html"]:
            return "text"
        return "unknown"

    def _open_raster_file(self, file_path: Path) -> str:
        """Open and describe a raster file using code."""
        from sandbox.runner import Sandbox, SandboxConfig

        sandbox = Sandbox(SandboxConfig(scratch_dir=Path("scratch")))

        code = f"""
import rasterio
import numpy as np

path = "{file_path.resolve()}"

with rasterio.open(path) as src:
    print(f"File: {{path}}")
    print(f"Driver: {{src.driver}}")
    print(f"Width x Height: {{src.width}} x {{src.height}}")
    print(f"Band count: {{src.count}}")
    print(f"CRS: {{src.crs}}")
    print(f"Bounds: {{src.bounds}}")
    print(f"Resolution: {{src.res}}")
    print(f"Data types: {{src.dtypes}}")
    if src.count >= 1:
        data = src.read(1)
        print(f"Pixel value range: min={{np.nanmin(data)}}, max={{np.nanmax(data)}}")
"""

        result = sandbox.run_sync(code, session_id="raster_read")
        if result.return_code == 0:
            return result.stdout
        else:
            return f"Read failed: {result.stderr}"

    def _open_vector_file(self, file_path: Path) -> str:
        """Open and describe a vector file using code."""
        from sandbox.runner import Sandbox, SandboxConfig

        sandbox = Sandbox(SandboxConfig(scratch_dir=Path("scratch")))

        code = f"""
import geopandas as gpd

path = "{file_path.resolve()}"
gdf = gpd.read_file(path)

print(f"File: {{path}}")
print(f"Geometry types: {{gdf.geometry.type.unique()}}")
print(f"Feature count: {{len(gdf)}}")
print(f"Column names: {{list(gdf.columns)}}")
print(f"CRS: {{gdf.crs}}")
print("\\nTop 5 rows:")
print(gdf.head())
"""

        result = sandbox.run_sync(code, session_id="vector_read")
        if result.return_code == 0:
            return result.stdout
        else:
            return f"Read failed: {result.stderr}"

    def _image_to_content_block(self, file_path: Path) -> dict:
        # Convert the target MB to bytes
        max_bytes = int(2 * 1024 * 1024)

        # Read the original file and check its initial size
        file_size = file_path.stat().st_size

        with Image.open(file_path.resolve()) as img:
            # If PNG has an alpha channel (RGBA) or palette (P), convert to RGB so it can be saved as JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Downscale very large images up front. Generated maps can be
            # hundreds of megapixels / gigapixels; without this the loop below
            # would JPEG-encode the full giant array repeatedly, taking minutes
            # and risking OOM. Capping the longest side first bounds time/memory
            # while keeping the map legible for the multimodal judge.
            MAX_DIM = 4096
            if max(img.size) > MAX_DIM:
                img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)

            img_byte_arr = io.BytesIO()
            quality = 95

            # If the original file is already under 3MB, save it directly to memory
            # (converting to JPEG usually makes it even smaller)
            if file_size <= max_bytes:
                img.save(img_byte_arr, format='JPEG', quality=quality)

            # If the file is over 3MB, or still over 3MB after saving, start iterative compression
            while True:
                img_byte_arr.seek(0)
                img_byte_arr.truncate(0) # Clear the buffer

                img.save(img_byte_arr, format='JPEG', quality=quality)
                current_size = img_byte_arr.tell()

                # If the target size is reached, exit the loop
                if current_size <= max_bytes:
                    break

                # Compression strategy: lower quality first; once quality drops to 30, start shrinking dimensions
                if quality > 30:
                    quality -= 10
                else:
                    # Quality is already very low, scale down the image size (80% of width and height)
                    width, height = img.size
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    # After shrinking dimensions, slightly restore the quality parameter
                    quality = 60

            # Get the final compressed byte data
            image_data = img_byte_arr.getvalue()

        # Use the jpeg mime type uniformly
        mime = "image/jpeg"
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        }

    def _open_text_file(self, file_path: Path) -> str:
        """Open a text file."""
        try:
            # Strict limit: max 995904 characters
            MAX_CHAR_LENGTH = 100000
            content = file_path.read_text(encoding="utf-8").strip()

            # Empty file handling
            if not content:
                return "File is empty"

            # Auto-truncate if too long
            if len(content) > MAX_CHAR_LENGTH:
                content = content[:MAX_CHAR_LENGTH]
                content += "\n\n[File content too long, truncated to maximum allowed length]"

            return content
        except Exception as e:
            return f"Read failed: {str(e)}"

    def open_file(self, file_path: str) -> str:
        """Open a file and return its content description.

        This is the main tool function that routes to the appropriate
        handler based on file type.

        Args:
            file_path: Path to the file to open.

        Returns:
            Natural language description of the file content.
        """
        path = Path(file_path)

        if not path.exists():
            return f"File not found: {file_path}"

        file_type = self._get_file_type(path)

        if file_type == "image":
            # Images are handled separately as content blocks; return a placeholder
            return f"[Image file: {path.name}]"
        elif file_type == "raster":
            return self._open_raster_file(path)
        elif file_type == "vector":
            return self._open_vector_file(path)
        elif file_type == "text":
            return self._open_text_file(path)
        else:
            return f"Unsupported file type: {path.suffix}"

    def _parse_json_response(self, text: str) -> dict | None:
        """Parse JSON from model response text."""
        import json
        import re

        # Try to find JSON in markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _track_usage(self, response) -> None:
        """Accumulate token usage from API response."""
        usage = getattr(response, "usage", None)
        if usage:
            self.token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self.token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            self.token_usage["total_tokens"] += getattr(usage, "total_tokens", 0)
            self.token_usage["call_count"] += 1

    def get_token_usage(self) -> dict:
        """Return current token usage statistics."""
        return dict(self.token_usage)

    def _compute_overall_score(self, task_consistency: dict, result_completeness: dict,
                                accuracy: dict, visualization_quality: dict) -> float:
        """Compute overall_score as the average of four dimension scores."""
        scores = []
        for d in [task_consistency, result_completeness, accuracy, visualization_quality]:
            s = d.get("score", 0)
            if isinstance(s, (int, float)):
                scores.append(s)
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def evaluate(
        self,
        task_text: str,
        task_id: str,
        used_dataset: str,
        Expected_Output: str,
        reference_solution: str | None = None,
        generated_solution: str | None = None,
        reference_answer: str | None = None,
        generated_answer: str | None = None,
    ) -> EvaluationResult:
        """Evaluate generated code against reference code for a task.

        This method:
        1. Runs both reference and generated code in sandbox
        2. Reads and analyzes the execution results
        3. Compares the generated code/results against reference
        4. Evaluates from multiple dimensions

        Args:
            task_text: Description of the remote sensing task.
            reference_solution: Directory containing reference output files (for RESULT method), or reference code (for CODE/TRAJCT methods).
            generated_solution: Directory containing generated output files (for RESULT method), or generated code (for CODE/TRAJCT methods).
            reference_answer: Reference answer text (optional, for RESULT method).
            generated_answer: Generated answer text (optional, for RESULT method).
            context: Additional context about the task.
            method: Eval method.

        Returns:
            EvaluationResult with scores and comments.
        """
        # Step 1: Execute code in sandbox and collect results
        reference_execution = None
        generated_execution = None
        ref_result_files: list[Path] = []
        gen_result_files: list[Path] = []

        # Extract expected output file names from Expected_Output
        expected_files = set()
        if isinstance(Expected_Output, dict):
            for v in Expected_Output.values():
                if v and isinstance(v, str) and "." in v:
                    expected_files.add(v)

        # Step 2: Read result files (only those in Expected_Output)
        if reference_solution is not None and Path(reference_solution).exists():
            ref_result_files = [
                f for f in Path(reference_solution).rglob("*")
                if f.is_file() and f.name in expected_files
            ]
        reference_results_text, reference_image_blocks = self._describe_files(
            ref_result_files, "Reference Solution File Result"
        )

        generated_results_text = ""
        generated_image_blocks: list[dict] = []

        if generated_solution is not None and Path(generated_solution).exists():
            gen_result_files = [
                f for f in Path(generated_solution).rglob("*")
                if f.is_file() and f.name in expected_files
            ]
            generated_results_text, generated_image_blocks = self._describe_files(
                gen_result_files, "Generated Solution File Result"
            )

        # Step 3: Build evaluation prompt as content blocks
        eval_instruction = """
        Based on the reference solution and generated solution results above, evaluate from multiple dimensions:
        1. Task Consistency - Whether the generated solution results meet the task requirements
        2. Result Completeness - Whether the generated solution results contain all required content
        3. Result Accuracy - The quality and accuracy of the generated solution results
        4. Visualization Quality - If the generated solution has image output, whether the images are clear and readable and contain important information

        Please output the evaluation results in JSON format."""

        text_parts = []
        text_parts.append(f"## Task Description\n{task_text}")
        text_parts.append(f"## Used Dataset\n{used_dataset}")
        text_parts.append(f"## Expected Output\n{Expected_Output}")
        text_parts.append(f"## File Result\n{task_text}")
        if reference_results_text:
            text_parts.append(reference_results_text)
        if generated_results_text:
            text_parts.append(generated_results_text)
        text_parts.append(f"## Solution Answer\n{task_text}")
        if reference_answer:
            text_parts.append(f"## Reference Answer\n{reference_answer}")
        if generated_answer:
            text_parts.append(f"## Generated Answer\n{generated_answer}")
        text_parts.append(eval_instruction)

        text_parts.append(f"## Image Result\n{task_text}")
        text_part = "\n\n".join(text_parts)

        user_content: list[dict] = [{"type": "text", "text": text_part}]
        user_content.extend(reference_image_blocks)
        user_content.extend(generated_image_blocks)


        # Step 4: Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tool_choice="auto",
            temperature=0.3,
        )

        self._track_usage(response)

        response_text = response.choices[0].message.content or ""

        # Step 5: Parse result
        parsed = self._parse_json_response(response_text)

        if parsed:
            tc = parsed.get("task_consistency", {})
            rc = parsed.get("result_completeness", {})
            acc = parsed.get("accuracy", {})
            vq = parsed.get("visualization_quality", {})
            overall = self._compute_overall_score(tc, rc, acc, vq)
            return EvaluationResult(
                task_consistency=tc,
                result_completeness=rc,
                accuracy=acc,
                visualization_quality=vq,
                overall_score=overall,
                summary=parsed.get("summary", ""),
                suggestions=parsed.get("suggestions", []),
                reference_vs_generated=parsed.get("reference_vs_generated", {}),
                reference_execution=reference_execution,
                generated_execution=generated_execution,
            )
        else:
            # Fallback: return raw response
            return EvaluationResult(
                task_consistency={"score": 0.0, "comments": "Evaluation failed"},
                result_completeness={"score": 0.0, "comments": "Evaluation failed"},
                accuracy={"score": 0.0, "comments": "Evaluation failed"},
                visualization_quality={"score": 0.0, "comments": "Evaluation failed"},
                overall_score=0.0,
                summary=response_text[:500] if response_text else "Evaluation failed",
                suggestions=["Please check the API response format"],
                reference_execution=reference_execution,
                generated_execution=generated_execution,
            )

    def _execute_code(self, code: str, session_id: str) -> dict[str, Any]:
        """Execute code in sandbox and return execution info.

        Args:
            code: Python code to execute.
            session_id: Session identifier for grouping files.

        Returns:
            Dictionary with execution results.
        """
        result = self._sandbox.run_sync(code, session_id=session_id)

        return {
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "files_created": [str(f) for f in result.files_created],
            "error": result.error,
        }

    def _collect_result_files(self, execution: dict[str, Any]) -> list[Path]:
        """Collect result files from execution.

        Args:
            execution: Execution dictionary from _execute_code.

        Returns:
            List of Path objects for result files.
        """
        files_created = execution.get("files_created", [])
        return [Path(f) for f in files_created if Path(f).exists()]

    def _describe_files(self, file_paths: list[Path], title: str) -> tuple[str, list[dict]]:
        """Read and describe multiple result files.

        Args:
            file_paths: List of file paths to read.
            title: Title for this section.

        Returns:
            Tuple of (text description, list of image content blocks).
        """
        descriptions = [f"=== {title} ===\n"]
        image_blocks: list[dict] = []

        for file_path in file_paths:
            file_type = self._get_file_type(file_path)
            if file_type == "image":
                # Reading/encoding an image is not wrapped in try/except inside
                # _image_to_content_block, so a corrupt/truncated/unsupported
                # image would otherwise raise and abort the ENTIRE evaluation
                # (scoring the whole task as an error). Guard it here so a single
                # unreadable image degrades to an explicit note instead, and the
                # remaining output files still get evaluated fairly.
                try:
                    block = self._image_to_content_block(file_path)
                except Exception as e:
                    descriptions.append(
                        f"--- Image file: {file_path.name} ---\n"
                        f"[Image exists but could NOT be read: {e}]\n"
                    )
                    continue
                descriptions.append(f"--- Image file: {file_path.name} ---\n")
                image_blocks.append({"type": "text", "text": f"[Image: {file_path.name}]"})
                image_blocks.append(block)
            else:
                content = self.open_file(str(file_path))
                descriptions.append(f"--- File: {file_path.name} ---\n{content}\n")
        
        if len(descriptions) == 1:
            descriptions.append("No Data Result Found!")

        return "\n".join(descriptions), image_blocks


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    import json

    # Env/config is loaded centrally by `import config` (root .env + config.ini).

    evaluator = RemoteSensingEvaluator(
        api_key=config.OPENAI_API_KEY,
        api_base=config.OPENAI_API_BASE,
        model=config.OPENAI_MODEL,
        multimodal_model=config.OPENAI_MODEL,
    )

    # Example evaluation with reference code and generated code
    with open("./eval/test.json", "r") as f:
        data = json.load(f)

    print(data["task_text"])

    result = evaluator.evaluate(
        data["task_text"], data["reference_code"], data["generated_code"]
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
