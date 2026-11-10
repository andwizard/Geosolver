import asyncio
import contextlib
import io
import json
import os
import sys
import time  # DEBUG_TIMING
from pathlib import Path
from typing import Literal

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagents.backends import LocalShellBackend
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from Agents.tools.graph import set_DAG, update_node, remove_node, add_edge, remove_edge, show_dag, _get_graph, reset_graph, DATA_TYPES, OPERATOR_TYPES
from Agents.tools.sandbox_env import execute_code, execute_file
from Agents.tools.code_framework import scaffold_code as scaffold_code_fn
from Agents.tools.file_ops import read_code, edit_code
from Agents.tools.verify_output import verify_output, set_workspace_dir
from Agents.prompts.geosolver_prompt import GEOSOLVER_PROMPT, GRAPH_PLANNER_PROMPT, CODE_GENERATOR_PROMPT, DATA_EXPLORER_PROMPT, CONTRACT_GENERATOR_PROMPT
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit, ModelCallLimitMiddleware, ModelRetryMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware
from config import LLM_REQUEST_TIMEOUT


# Global to store the final output
_task_output: dict = {}
_timings: list = []  # DEBUG_TIMING
_script_dir: str = ""
_task_context: dict = {}  # Injected before agent runs — tools read from this
_exploration_findings: str = ""  # Cached from explore_data, auto-injected to plan_and_scaffold
_exploration_metadata: dict = {}  # Structured JSON extracted from Data Explorer output

# Dedicated httpx client for the current build. Passed explicitly to the model so
# langchain's lru_cache'd shared pool is bypassed — each build gets its own pool,
# preventing a poisoned/half-open connection from leaking across tasks.
_geosolver_http_client: httpx.AsyncClient | None = None


async def aclose_geosolver_client() -> None:
    """Close the httpx client from the previous build before rebuilding."""
    global _geosolver_http_client
    if _geosolver_http_client is not None:
        with contextlib.suppress(Exception):
            await _geosolver_http_client.aclose()
        _geosolver_http_client = None

# Per-sub-agent timeout (seconds).
# DISABLED — sub-agents now run without timeout.
# SUB_AGENT_TIMEOUT_SECONDS = 300

def set_script_dir(path: str) -> None:
    """Set the directory where solution.py is located."""
    global _script_dir
    _script_dir = path


@tool
def send_output(status: Literal["ACCEPT", "FAILED"], answer: str = "") -> str:
    """Submit the final task result. Call this when you have completed all analysis.

    The generated script (solution.py) is automatically read from the workspace
    directory — you do NOT need to pass it.

    Args:
        status: "ACCEPT" if the task was completed successfully, "FAILED" otherwise.
        answer: A summary of the results/outputs produced by the script.

    Returns:
        Confirmation string.
    """
    global _task_output, _script_dir
    script = ""
    if _script_dir:
        script_path = os.path.join(_script_dir, "solution.py")
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script = f.read()
        except Exception:
            script = ""
    _task_output = {
        "status": status,
        "script": script,
        "answer": answer,
    }
    return f"Output submitted. Status: {status}"


def build_geosolver(model_name):
    # Create a model instance with request_timeout so LLM API calls
    # don't hang indefinitely when the API is unresponsive.
    # Pass a dedicated httpx client so langchain's lru_cache'd shared connection
    # pool is bypassed — each build gets its own pool, so a poisoned connection
    # from one task cannot leak into the next.
    global _geosolver_http_client
    _geosolver_http_client = httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT)
    llm = init_chat_model(
        model_name,
        request_timeout=LLM_REQUEST_TIMEOUT,
        http_async_client=_geosolver_http_client,
    )

    graph_planner = create_agent(
        llm,
        tools=[set_DAG, update_node, remove_node, add_edge, remove_edge, show_dag, execute_code],
        system_prompt=GRAPH_PLANNER_PROMPT,
        middleware=[
                ContextEditingMiddleware(
                    edits=[
                        ClearToolUsesEdit(
                            trigger=10000,
                            keep=5,
                            clear_tool_inputs=True,
                            exclude_tools=[],
                            placeholder="[cleared]",
                        ),
                    ],
                ),
                # ModelRetryMiddleware(
                #     max_retries=3,
                #     backoff_factor=2.0,
                #     initial_delay=1.0,
                # ),
                # ToolRetryMiddleware(
                #     max_retries=3,
                #     backoff_factor=2.0,
                #     initial_delay=1.0,
                # ),
            ],
    )
    #TODO
    code_generator = create_agent(
        llm,
        tools=[execute_code, execute_file, read_code, edit_code],
        system_prompt=CODE_GENERATOR_PROMPT,
        middleware=[
                # ToolCallLimitMiddleware(thread_limit=10, run_limit=5),
                ContextEditingMiddleware(
                    edits=[
                        ClearToolUsesEdit(
                            trigger=20000,
                            keep=5,
                            clear_tool_inputs=True,
                            exclude_tools=[],
                            placeholder="[cleared]",
                        ),
                    ],
                ),
                ModelRetryMiddleware(
                    max_retries=3,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                )
            ],
    )

    data_explorer = create_agent(
        llm,
        tools=[execute_code],
        system_prompt=DATA_EXPLORER_PROMPT,
        middleware=[
                # ContextEditingMiddleware(
                #     edits=[
                #         ClearToolUsesEdit(
                #             trigger=10000,
                #             keep=3,
                #             clear_tool_inputs=True,
                #             exclude_tools=[],
                #             placeholder="[cleared]",
                #         ),
                #     ],
                # ),
                ModelRetryMiddleware(
                    max_retries=3,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
                ToolRetryMiddleware(
                    max_retries=3,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
            ],
    )

    @tool
    async def explore_data(instruction: str = "") -> str:
        """
        Explore all input data files and report their structure.
        Call this FIRST before any code generation.

        All task context (data paths, task description, expected outputs) is
        injected automatically. Usually you call this with an empty instruction.

        This tool ONLY explores and reports — it does NOT construct a DAG.
        Use this for ALL tasks (both simple and complex) to understand your data.

        Returns the exploration report AND the complete code execution
        trajectories (both successful AND failed). Review both:
        - Failed trajectories often reveal data quirks (encoding, skiprows, CRS issues)
        - Use these insights to guide downstream plan_and_scaffold or generate_code

        Args:
            instruction: Optional guidance for the data explorer, e.g.
                "focus on CRS compatibility between files" or
                "CSV needs skiprows=4, encoding may be latin-1".
                Leave empty if no special guidance is needed.
        """
        global _task_context
        global _exploration_findings
        _t0 = time.time()
        solver_instruction = instruction

        data_path = _task_context.get("data_path", "")
        task_description = _task_context.get("task_description", "")
        expected_output = _task_context.get("expected_output", "")
        output_dir = _task_context.get("output_dir", "")
        file_list = _task_context.get("file_list", "")

        sub_prompt = f"""## Data Directory
        {data_path}

        ## Task Description
        {task_description}

        ## Input Files
        {file_list}

        ## Expected Output Specification
        {expected_output}

        ## Output Directory
        {output_dir}

        ## Solver Instruction
        {solver_instruction if solver_instruction else "No special guidance — explore all files thoroughly."}

        ---

        Based on the above information, please explore EACH input data file and report its structure.
        IMPORTANT: Only explore and report. Do NOT construct a DAG.
        For each file, report: format, CRS, geometry type, columns/dimensions, row count, and any special notes."""

        result_text = ""
        execute_results = []  # Collect ALL (code, result) pairs — successes AND failures
        pending_code: dict = {}  # tool_call_id -> code string
        print(f"[explore_data] Starting data_explorer sub-agent at {time.time() - _t0:.1f}s...")
        sys.stdout.flush()
        de_stream = data_explorer.astream(
            {"messages": [{"role": "user", "content": sub_prompt}]},
            stream_mode="updates",
        )
        try:
            async for step in de_stream:
                for _, update in step.items():
                    if update and (messages := update.get("messages")) and isinstance(messages, list):
                        for message in messages:
                            msg_type = type(message).__name__
                            print(f"[explore_data] Received {msg_type} at {time.time() - _t0:.1f}s")
                            sys.stdout.flush()
                            message.pretty_print()
                            # Capture tool_calls from AIMessage (stores code parameter by tool_call_id)
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                for tc in message.tool_calls:
                                    if tc.get('name') == 'execute_code':
                                        pending_code[tc.get('id')] = tc.get('args', {}).get('code', '')
                            # Capture ToolMessage results from execute_code, paired with code
                            # Record ALL trajectories (successes AND failures) so the solver
                            # can analyze failures to extract data quirks (encoding, skiprows, etc.)
                            if hasattr(message, 'name') and message.name == 'execute_code':
                                content = message.content if hasattr(message, 'content') else str(message)
                                tool_call_id = getattr(message, 'tool_call_id', '')
                                code = pending_code.pop(tool_call_id, '')
                                status = "[SUCCESS]" if "[SUCCESS]" in str(content) else "[FAILED]"
                                if code:
                                    full_msg = f"[STATUS] {status}\n[CODE]\n{code}\n[RESULT]\n{str(content)}"
                                else:
                                    full_msg = f"[STATUS] {status}\n[RESULT]\n{str(content)}"
                                execute_results.append(full_msg)
                            if hasattr(message, 'text') and message.text:
                                result_text = message.text
        except asyncio.CancelledError:
            await de_stream.aclose()
            raise
        finally:
            _timings.append(f"explore_data: {time.time() - _t0:.1f}s")
        print(f"[explore_data] Sub-agent finished at {time.time() - _t0:.1f}s")
        sys.stdout.flush()
        # Store ALL execute_code trajectories (both SUCCESS and FAILED) as exploration
        # findings. Failed attempts are especially valuable — they reveal data quirks
        # (encoding, skiprows, CRS format) that downstream agents must account for.
        _exploration_findings = "\n\n---\n\n".join(execute_results) if execute_results else result_text

        # Append the complete trajectory log to the result so the solver can
        # analyze both successes and failures and extract actionable insights
        # for plan_and_scaffold / generate_code instructions.
        # print(result_text)
        # quit()
        # if execute_results:
        #     success_count = sum(1 for r in execute_results if "[STATUS] [SUCCESS]" in r)
        #     fail_count = len(execute_results) - success_count
        #     trajectory_report = f"\n\n=== EXPLORATION TRAJECTORIES ({len(execute_results)} total: {success_count} success, {fail_count} failed) ===\n"
        #     trajectory_report += "\n\n---\n\n".join(execute_results)
        #     trajectory_report += "\n=== END TRAJECTORIES ==="
        #     result_text = result_text + trajectory_report

        # Extract structured JSON metadata from Data Explorer's output
        # This bypasses the need for GeoSolver Phase 1.5 manual synthesis
        global _exploration_metadata
        _exploration_metadata = {}
        try:
            json_str = _extract_json_from_llm_response(result_text)
            _exploration_metadata = json.loads(json_str)
            print(f"[explore_data] Extracted structured metadata for {len(_exploration_metadata.get('files', []))} file(s)")
        except Exception as e:
            print(f"[explore_data] Could not extract structured JSON: {e}")
        
        return result_text

    # -------------------------------------------------------------------
    # Phase 2 helpers: per-layer contract generation (LLM-based)
    # -------------------------------------------------------------------

    def _extract_json_from_llm_response(text: str) -> str:
        """Strip markdown fences and extract JSON from an LLM response."""
        import re as _re
        match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, _re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _format_exploration_metadata(metadata: dict) -> str:
        """Format structured exploration metadata into a prompt section.

        Replaces the old Phase 1.5 manual synthesis by GeoSolver.
        """
        if not metadata or not metadata.get("files"):
            return ""

        lines = ["## Per-File Metadata (extracted by Data Explorer)", ""]
        for f in metadata["files"]:
            lines.append(f"### File: {f.get('name', '?')}")
            lines.append(f"- format: {f.get('format', '?')}")
            if f.get("crs"):
                lines.append(f"- crs: {f['crs']}")
            if f.get("geom_type"):
                lines.append(f"- geom_type: {f['geom_type']}")
            if f.get("schema_fields"):
                lines.append(f"- schema_fields: {f['schema_fields']}")
            if f.get("row_count") is not None:
                lines.append(f"- row_count: {f['row_count']}")
            if f.get("loading_code"):
                lines.append(f"- loading_code: {f['loading_code']}")
            if f.get("notes"):
                lines.append(f"- notes: {f['notes']}")
            lines.append("")

        if metadata.get("crs_compatibility"):
            lines.append(f"### CRS Compatibility")
            lines.append(str(metadata["crs_compatibility"]))
            lines.append("")

        if metadata.get("data_pitfalls"):
            lines.append(f"### Data Pitfalls (from failed attempts)")
            lines.append(str(metadata["data_pitfalls"]))
            lines.append("")

        return "\n".join(lines)

    def _format_data_node_metadata(node, var_name):
        """Format a single data node's metadata for the LLM prompt."""
        parts = [f"  - id={node.get('id')}, name=\"{node.get('name')}\", type={node.get('type')}"]
        parts.append(f"    variable: `{var_name}`")
        if node.get("crs"):
            parts.append(f"    crs: {node['crs']}")
        if node.get("geom_type"):
            parts.append(f"    geom_type: {node['geom_type']}")
        if node.get("schema_fields"):
            parts.append(f"    schema_fields: {node['schema_fields']}")
        if node.get("resolution"):
            parts.append(f"    resolution: {node['resolution']}")
        if node.get("bands"):
            parts.append(f"    bands: {node['bands']}")
        if node.get("dimensions"):
            parts.append(f"    dimensions: {node['dimensions']}")
        if node.get("variables"):
            parts.append(f"    variables: {node['variables']}")
        if node.get("spatial_keys"):
            parts.append(f"    spatial_keys: {node['spatial_keys']}")
        return "\n".join(parts)

    def _format_operator_node_metadata(node, out_var):
        """Format a single operator node for the LLM prompt."""
        parts = [f"  - id={node.get('id')}, name=\"{node.get('name')}\", type={node.get('type')}"]
        parts.append(f"    output variable: `{out_var}`")
        if node.get("tag"):
            parts.append(f"    tag: {node['tag']}")
        if node.get("function_description"):
            parts.append(f"    description: {node['function_description']}")
        return "\n".join(parts)

    def _build_all_contracts_prompt(nodes, edges, node_map, layers, var_names, task_context):
        """Build a single prompt for generating ALL contracts at once.

        Shows the full DAG organized by topological layers so the LLM can see
        the complete data flow and write cross-layer contracts in one pass.
        """
        task_description = task_context.get("task_description", "")
        expected_output = task_context.get("expected_output", "")

        # Build depth→node_id mapping
        node_depth: dict = {}
        for depth, layer_ids in enumerate(layers):
            for nid in layer_ids:
                node_depth[nid] = depth

        max_depth = len(layers) - 1

        # Group nodes by layer for structured presentation
        layer_blocks = []
        for depth, layer_ids in enumerate(layers):
            lines = [f"### Layer {depth}" + (" (FINAL OUTPUT)" if depth == max_depth else "")]
            for nid in layer_ids:
                node = node_map.get(nid, {})
                ntype = node.get("type", "?")
                if ntype in DATA_TYPES:
                    lines.append(_format_data_node_metadata(node, var_names.get(nid, nid)))
                else:
                    lines.append(_format_operator_node_metadata(node, var_names.get(nid, nid)))
                # Show edges for this node — use VARIABLE NAME (not node ID)
                # so the LLM writes correct Python variable references in contracts
                for e in edges:
                    if e.get("target") == nid and e.get("source") in node_map:
                        src = node_map[e.get("source")]
                        src_var = var_names.get(e['source'], e['source'])
                        role = e.get("role", "")
                        role_str = f", role={role}" if role else ""
                        lines.append(f"    ← `{src_var}` [{src.get('type','?')}{role_str}]")
            layer_blocks.append("\n".join(lines))

        dag_structure = "\n\n".join(layer_blocks)

        return f"""## Task Description
{task_description}

## Expected Output Specification
{expected_output}

## Full DAG (topological order, {len(layers)} layers)

{dag_structure}

---
Generate assertions for ALL nodes in the DAG above. Think about the ENTIRE pipeline:

- **Root data nodes** (Layer 0 data nodes): 0-1 post-loading assertion each
  - CRS not what was expected? Critical column missing? Geometry type different?
- **Operators** (each processing step): 2-4 assertions TOTAL per operator (pre+post)
  - Pre: only if inputs could be incompatible (CRS mismatch, missing column)
  - Post: what must be true for downstream to work?
- **Final output layer** (Layer {max_depth}): post-conditions MUST match Expected Output Specification

Ask for each operator: "if this step silently fails, what downstream breaks?"

Return ONLY a JSON object keyed by node ID. Use exact variable names from above.
```json
{{
  "<node-id>": {{
    "pre": ["assert ...", ...],
    "post": ["assert ...", ...]
  }}
}}
```
"""

    # -------------------------------------------------------------------

    @tool
    async def plan_and_scaffold(instruction: str = "") -> str:
        """
        Construct DAG and generate code scaffold. Only for COMPLEX tasks.

        This tool:
        1. Constructs a GISDataFlowGraph (DAG) based on exploration findings
           (auto-injected from explore_data — you do NOT need to pass them;
           includes BOTH successful and failed trajectories for full context)
        2. Generates a code scaffold with embedded contracts and TODO blocks
        3. Returns the DAG summary AND the full scaffold code for your review

        Review the scaffold code carefully before passing it to generate_code.
        Check that: the DAG covers all task requirements, CRS/geom_type contracts
        are correct, and the output layer produces the expected files.

        Args:
            instruction: Optional DAG design notes, e.g.
                "Make sure all CRS are unified to EPSG:4326" or
                "Add a deduplication step before the spatial join".
                Leave empty if no special guidance is needed.
        """
        global _task_context
        global _exploration_findings
        _t0 = time.time()

        # Auto-inject cached exploration findings (includes both SUCCESS and FAILED
        # trajectories) — LLM never passes them. Failed trajectories are valuable
        # for understanding data quirks (encoding, skiprows, CRS format).
        # data_specs = _exploration_findings or "No exploration findings available."

        data_path = _task_context.get("data_path", "")
        task_description = _task_context.get("task_description", "")
        expected_output = _task_context.get("expected_output", "")
        output_dir = _task_context.get("output_dir", "")
        file_list = _task_context.get("file_list", "")

        # Step 1: Run graph_planner with structured metadata auto-injected
        # Structured metadata (from Data Explorer's JSON output) is the PRIMARY
        # source for populating DAG node metadata. Raw trajectories are dropped —
        # they were only useful for GeoSolver's now-removed Phase 1.5 synthesis.
        solver_instruction_block = ""
        if instruction:
            solver_instruction_block = f"""
        ## Solver Instructions (DAG Design Guidance)
        {instruction}

        """
        # Build the structured metadata section
        metadata_section = _format_exploration_metadata(_exploration_metadata)
        if not metadata_section:
            metadata_section = "No structured metadata available. Use execute_code to inspect files if needed."

        graph_instruction = f"""## Data Directory
{data_path}

## Task Description
{task_description}

## Input Files
{file_list}

## Expected Output Specification
{expected_output}

## Output Directory
{output_dir}

{metadata_section}
{solver_instruction_block}
"""
        result_text = ""
        # print(graph_instruction)
        # quit()
        gp_stream = graph_planner.astream(
            {"messages": [{"role": "user", "content": graph_instruction}]},
            stream_mode="updates",
        )
        try:
            async for step in gp_stream:
                for _, update in step.items():
                    if update and (messages := update.get("messages")) and isinstance(messages, list):
                        for message in messages:
                            message.pretty_print()
                            if hasattr(message, 'text') and message.text:
                                result_text = message.text
        except asyncio.CancelledError:
            await gp_stream.aclose()
            raise
        finally:
            _timings.append(f"plan_and_scaffold(graph_plan): {time.time() - _t0:.1f}s")

        # Step 1.5: Generate ALL contracts in ONE LLM call
        graph = _get_graph()
        if graph is not None:
            try:
                from Agents.tools.code_framework import _resolve_var_names
                nodes = graph._data.get("nodes", [])
                edges = graph._data.get("edges", [])
                node_map = {n.get("id"): n for n in nodes}
                layers = graph.topological_sort()
                var_names = _resolve_var_names(graph)

                full_prompt = _build_all_contracts_prompt(
                    nodes, edges, node_map, layers, var_names, _task_context,
                )
                full_message = CONTRACT_GENERATOR_PROMPT + "\n\n" + full_prompt
                
                # print(full_message)
                # quit()

                response = await llm.ainvoke([
                    {"role": "user", "content": full_message},
                ])
                all_contracts = json.loads(_extract_json_from_llm_response(response.content))
                total_contracts = 0
                for node_dict in nodes:
                    nid = node_dict.get("id")
                    if nid in all_contracts:
                        node_dict["contracts"] = all_contracts[nid]
                        total_contracts += 1

                result_text += f"\nContracts generated for {total_contracts} node(s) in a single pass."
            except Exception as e:
                result_text += f"\nWARNING: Contract generation failed — {e}"
        else:
            return result_text + "\n\nFAILED: No active DAG graph. The graph_planner did not register a valid DAG."

        # Step 2: Generate scaffold from the DAG
        try:
            code = scaffold_code_fn(graph, data_path, output_dir)
            file_path = os.path.join(output_dir, "solution.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            result_text += f"\n\nScaffold written to: {file_path}"
            # Include full scaffold code so Solver can review it
            result_text += f"\n\n--- SCAFFOLD CODE (review before calling generate_code) ---\n{code}\n--- END SCAFFOLD ---"
        except Exception as e:
            result_text += f"\n\nFAILED: Could not generate code framework — {e}"
        # quit()
        _timings.append(f"plan_and_scaffold(total): {time.time() - _t0:.1f}s")
        # quit()
        return result_text

    @tool
    async def generate_code(instruction: str) -> str:
        """
        Generate or modify the solution code based on your instruction.

        The code_generator sub-agent works on workspace/solution.py. It has:
        - read_code: read the current solution.py
        - edit_code: make surgical edits to solution.py
        - execute_code: run a temporary code snippet (with prelude imports) for quick testing
        - execute_file: run solution.py and report errors/output

        It does NOT verify outputs — you will call verify_output yourself after.

        Use this tool for ALL code generation needs:
        - Simple task (first attempt): provide a concrete task plan, e.g.
          "Load X.csv with pd.read_csv, convert lon/lat to GeoDataFrame,
           compute Y, save result.shp to OUTPUT_DIR, print answer."
        - Complex task (first attempt): after reviewing the scaffold from
          plan_and_scaffold, provide guidance, e.g.
          "The scaffold looks good. Fill all TODOs.
           Note: node-3 needs CRS reprojection to EPSG:3857."
        - Retry after failure: provide specific fix guidance, e.g.
          "node-5 spatial join returned empty — check CRS mismatch
           between node-3 (EPSG:4326) and node-4 (EPSG:3857)."

        Args:
            instruction: Your complete instruction to the code generator.
                Include the task plan, scaffold review notes, or fix guidance.
                Be specific and actionable.
        """
        global _task_context
        _cg_t0 = time.time()

        output_dir = _task_context.get("output_dir", "")
        scaffold_path = os.path.join(output_dir, "solution.py")
        task_description = _task_context.get("task_description", "")
        expected_output = _task_context.get("expected_output", "")
        data_path = _task_context.get("data_path", "")
        file_list = _task_context.get("file_list", "")

        # Ensure solution.py exists so edit_code can work on it
        if not os.path.exists(scaffold_path):
            with open(scaffold_path, "w", encoding="utf-8") as f:
                f.write("# solution.py - to be filled by code_generator\n")

        sub_instruction = f"""## Workspace
        Scaffold/solution file: {scaffold_path}

        ## Data Directory
        {data_path}

        ## Task Description
        {task_description}

        ## Input Files
        {file_list}

        ## Expected Output
        {expected_output}

        ## Output Directory
        {output_dir}

        ## Solver Instruction (FOLLOW THIS)
        {instruction}

        ---

        Work on {scaffold_path}:
        1. Use `read_code` to inspect the current state of the file
        2. Use `edit_code` to implement changes according to the Solver Instruction above
        3. Use `execute_file` to test after each major change
        4. If execution fails, read the error, fix it with `edit_code`, and re-execute
        5. When the code runs successfully (no errors), report back — do NOT verify outputs"""

        # print("==================")
        # print(sub_instruction)
        # quit()
        all_messages = []
        result_text = ""
        cg_stream = code_generator.astream(
            {"messages": [{"role": "user", "content": sub_instruction}]},
            stream_mode="updates",
        )
        try:
            async for step in cg_stream:
                for _, update in step.items():
                    if update and (messages := update.get("messages")) and isinstance(messages, list):
                        for message in messages:
                            message.pretty_print()
                            all_messages.append(message)
                            if hasattr(message, 'text') and message.text:
                                result_text = message.text
        except asyncio.CancelledError:
            await cg_stream.aclose()
            raise

        # Extract last execute_file result from sub-agent's tool messages
        last_exec_result = None
        for msg in all_messages:
            if hasattr(msg, 'name') and msg.name == 'execute_file':
                last_exec_result = msg.content if hasattr(msg, 'content') else str(msg)

        if last_exec_result:
            exec_output = last_exec_result[:5000] if len(str(last_exec_result)) > 5000 else str(last_exec_result)
            result_text += f"\n\n[Execution Result]\n{exec_output}"

        _timings.append(f"generate_code: {time.time() - _cg_t0:.1f}s")
        return result_text

    geosolver_agent = create_agent(
        llm,
        # tools=[inspect, send_output, code_generate, graph_plan, scaffold_code],
        tools=[send_output, explore_data, plan_and_scaffold, generate_code, verify_output],
        system_prompt=GEOSOLVER_PROMPT,
        middleware=[
                ContextEditingMiddleware(
                    edits=[
                        ClearToolUsesEdit(
                            trigger=50000,
                            keep=8,
                            clear_tool_inputs=True,
                            exclude_tools=[],
                            placeholder="[cleared]",
                        ),
                    ],
                ),
                ModelRetryMiddleware(
                    max_retries=2,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
                ToolRetryMiddleware(
                    max_retries=2,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
            ],
    )

    return geosolver_agent


async def geo_asolve(agent, task, data_path, output_path):
    global _task_output, _timings, _task_context, _exploration_findings  # DEBUG_TIMING
    _task_output = {}
    _timings.clear()  # DEBUG_TIMING
    _exploration_findings = ""  # Clear cached findings for new task
    _exploration_metadata = {}  # Clear cached metadata for new task

    _t_total = time.time()  # DEBUG_TIMING

    # Reset global graph state to prevent cross-task leakage
    reset_graph()

    from Agents.prompts.basic_prompts import BASIC_PROMPT
    from pathlib import Path

    data_dir = str(Path(data_path) / task['source'])
    file_list = [f"{data_dir}/{file_name}" for file_name in task.get('used_dataset', [])]
    workspace_dir = f"{output_path}/workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    set_workspace_dir(workspace_dir)
    set_script_dir(workspace_dir)

    # Inject task context so explore_data/plan_and_scaffold/generate_code can
    # read it directly without relying on the LLM to pass these values correctly.
    _task_context = {
        "data_path": data_dir,
        "task_description": task['task_text'],
        "expected_output": json.dumps(task.get('Expected_Output', {})),
        "output_dir": workspace_dir,
        "file_list": "\n".join(file_list),
    }

    basic_prompt = BASIC_PROMPT.format(
        task_text=task['task_text'],
        file_list=file_list,
        expected_output=json.dumps(task.get('Expected_Output', {})),
        output_dir=workspace_dir,
    )

    print(basic_prompt)
    # quit()

    input_message = {
    "role": "user",
    "content": basic_prompt,
}
    # Redirect stdout through a Tee so ALL console output (including sub-agent
    # internal prints and pretty_print calls) is captured to trajectory buffer.
    trajectory_buf = io.StringIO()
    _original_stdout = sys.stdout

    class _Tee(io.StringIO):
        """Write to both the real stdout and a capture buffer."""
        def __init__(self, original, capture):
            super().__init__()
            self._original = original
            self._capture = capture

        def write(self, s):
            self._original.write(s)
            self._capture.write(s)
            return len(s)

        def flush(self):
            self._original.flush()

    sys.stdout = _Tee(_original_stdout, trajectory_buf)

    stream = agent.astream(
        {"messages": [input_message]},
        stream_mode="updates",
    )
    try:
        async for step in stream:
            for _, update in step.items():
                if update and (messages := update.get("messages")) and isinstance(messages, list):
                    for message in messages:
                        # Skip printing verify_output ToolMessages — they contain
                        # verbose base64 image data that clutters the output.
                        if hasattr(message, 'name') and message.name == 'verify_output':
                            continue
                        message.pretty_print()
    except asyncio.CancelledError:
        await stream.aclose()
        raise
    finally:
        sys.stdout = _original_stdout
        # Always save trajectory and print timing, even on SystemExit (quit)
        trajectory = trajectory_buf.getvalue()
        _task_output['trajectory'] = trajectory
        trajectory_path = Path(__file__).parent / "workspace" / "trajectory.txt"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trajectory_path, "w", encoding="utf-8") as f:
            f.write(trajectory)
        # DEBUG_TIMING: print all sub-agent timings at the end
        _timings.append(f"total: {time.time() - _t_total:.1f}s")
        print("\n[DEBUG_TIMING] " + " | ".join(_timings))
        # DEBUG_TIMING_END

    return _task_output



async def as_main():
    import configparser
    from pathlib import Path
    _cfg = configparser.ConfigParser()
    _cfg.read(Path(__file__).parent.parent / "config.ini")
    model_name = "openai:qwen3.5-plus"
    data_root = "../Data"

    agent = build_geosolver(model_name)
    task_id = "TASK_250309_135125_476928"
    with open(Path(__file__).parent.parent / "Data" / "datasets.json", "r") as f:
        data = json.load(f)
        tasks = data['tasks']
        for task in tasks:
            if task['task_ID'] == task_id :
                result = await geo_asolve(agent, task, data_path=os.path.abspath(data_root), output_path=os.path.abspath('.'))
                break

    # print("Final Answer:", result)

if __name__ == "__main__":
    asyncio.run(as_main())