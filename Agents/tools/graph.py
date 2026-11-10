from __future__ import annotations

import copy
import json as _json
import os as _os
from typing import List, Dict, Optional, Any, Union, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Strict base model
# ---------------------------------------------------------------------------
class StrictBaseModel(BaseModel):
    """Strict mode: forbid extra fields so DAG JSON drift is caught early."""

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Pydantic schema (internal validation only)
# ---------------------------------------------------------------------------

class BaseGraphNode(StrictBaseModel):
    """Base for all nodes."""

    id: Union[str, int] = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Optional description")
    contracts: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="LLM-generated contracts. Keys: 'pre', 'post'. "
                    "Values are lists of Python assert statements.",
    )


class VectorNode(BaseGraphNode):
    type: Literal["vector"] = "vector"
    format_hint: Optional[str] = Field(default=None)
    crs: str = Field(...)
    geom_type: str = Field(...)
    schema_fields: List[str] = Field(default_factory=list)


class RasterNode(BaseGraphNode):
    type: Literal["raster"] = "raster"
    format_hint: Optional[str] = Field(default=None)
    crs: str = Field(...)
    resolution: tuple[float, float] = Field(...)
    bands: Dict[str, str] = Field(...)


class MultidimNode(BaseGraphNode):
    type: Literal["multidim"] = "multidim"
    format_hint: Optional[str] = Field(default=None)
    dimensions: List[str] = Field(...)
    variables: List[str] = Field(...)


class TabularNode(BaseGraphNode):
    type: Literal["tabular"] = "tabular"
    format_hint: Optional[str] = Field(default=None)
    schema_fields: List[str] = Field(default_factory=list)
    spatial_keys: Optional[List[str]] = Field(default=None)


class OperatorNode(BaseGraphNode):
    """Operator base: has tag and function_description."""

    function_description: str = Field(
        ...,
        description="Short description of what this operator does, used as a code comment",
    )
    custom_assertions: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Custom assertions for tag='other' operators. Keys: 'pre', 'post'. "
                    "Assertions use role names for inputs and {output} for the operator output.",
    )


class PreprocessingOperator(OperatorNode):
    """Preprocessing: format conversion, CRS unification, reprojection, data filtering/cleaning, file I/O."""
    type: Literal["preprocessing"] = "preprocessing"
    tag: Literal["reproject", "format_convert", "filter_attribute", "zonal_stats", "data_clean", "other"]


class SpatialOperator(OperatorNode):
    """Spatial: spatial join, buffer, overlay (union/intersection/diff), clip, dissolve, kernel density."""
    type: Literal["spatial"] = "spatial"
    tag: Literal["buffer", "clip", "intersection", "union", "difference", "dissolve", "spatial_join", "other"]


class TemporalOperator(OperatorNode):
    """Temporal: time aggregation, resample, rolling window, climatology, time-series analysis."""
    type: Literal["temporal"] = "temporal"
    tag: Literal["resample", "rolling_window", "time_aggregate", "other"]


class StatisticalOperator(OperatorNode):
    """Statistical: groupby/agg, regression, clustering, numerical extraction, descriptive stats."""
    type: Literal["statistical"] = "statistical"
    tag: Literal["groupby_aggregate", "descriptive_stats", "regression", "clustering", "other"]


class VisualizationOperator(OperatorNode):
    """Visualization: map/chart/time-series figure generation."""
    type: Literal["visualization"] = "visualization"
    tag: Literal["choropleth", "scatter", "time_series", "histogram", "other"]


class ComputeOperator(OperatorNode):
    """General-purpose compute: any operation not covered by the other five operator types."""
    type: Literal["compute"] = "compute"
    tag: Literal["column_arithmetic", "area_calculation", "raster_math", "other"]


class Edge(StrictBaseModel):
    """A directed binding in the data-flow graph.

    Edges carry NO data contract (crs / geom_type / schema_fields / bands / ...).
    Those live on the data nodes, which are the single source of truth.The two ends of an edge must be different types of nodes.
    """

    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    role: Optional[str] = Field(
        default=None,
        description="Operator parameter name the source binds to on data->operator "
                    "edges, e.g. 'data', 'mask', 'clipper', 'left', 'right', 'weights', "
                    "'by'. Required on data->operator edges.",
    )



class _Schema(StrictBaseModel):
    """Pydantic schema for GISDataFlowGraph — used only for .model_validate()."""

    version: str = "1.0"
    nodes: List[Union[VectorNode, RasterNode, MultidimNode, TabularNode,
                      PreprocessingOperator, SpatialOperator, TemporalOperator,
                      StatisticalOperator, VisualizationOperator, ComputeOperator]]
    edges: List[Edge]


# ---------------------------------------------------------------------------
# TaskStep
# ---------------------------------------------------------------------------

class TaskStep:
    """Single step in a task sequence produced by GISDataFlowGraph.generate_task_sequence()."""

    def __init__(self, depth: int, node_id: str, node_info: dict, deps: list[dict]):
        self.depth = depth
        self.node_id = node_id
        self.node_info = node_info
        self.deps = deps

    def __repr__(self):
        return (
            f"TaskStep(depth={self.depth}, node_id={self.node_id!r}, "
            f"deps={self.deps}, node_info={self.node_info})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_id(node: dict) -> str:
    return str(node.get("node_id") or node.get("id"))


def _edge_sources(edge: dict) -> list[str]:
    sources = edge.get("source_node_ids") or edge.get("source_node_id") or edge.get("source")
    if isinstance(sources, str):
        sources = [sources]
    if sources is None:
        sources = []
    return [str(s) for s in sources]


def _edge_target(edge: dict) -> str:
    return str(edge.get("target_node_id") or edge.get("target"))


# ---------------------------------------------------------------------------
# Geographic CRS detection (semantic validation)
# ---------------------------------------------------------------------------

_GEOGRAPHIC_CRS_PATTERNS: set[str] = {
    "EPSG:4326", "EPSG:4269", "EPSG:4979", "CRS84", "OGC:CRS84",
}


def _is_geographic_crs(crs: str) -> bool:
    """Check if a CRS string represents a geographic (unprojected) system.

    Uses a simple heuristic — checks against known geographic codes and
    proj strings containing ``longlat`` / ``lonlat``. This is pragmatic,
    not a full CRS database lookup.
    """
    if not crs:
        return False
    crs_upper = crs.upper()
    if crs_upper in _GEOGRAPHIC_CRS_PATTERNS:
        return True
    crs_lower = crs.lower()
    if "longlat" in crs_lower or "lonlat" in crs_lower:
        return True
    return False


# ---------------------------------------------------------------------------
# GISDataFlowGraph — main class
# ---------------------------------------------------------------------------

class GISDataFlowGraph:
    """Unified DAG graph, encapsulating state management, validation,
    topological sort and queries."""

    # -- node-type validation rules (ported from _validate_dag) --
    _DATA_TYPES = {"vector", "raster", "multidim", "tabular"}
    _OPERATOR_TYPES = {
        "preprocessing", "spatial", "temporal", "statistical", "visualization", "compute",
    }
    _VALID_TYPES = _DATA_TYPES | _OPERATOR_TYPES

    _TYPE_REQUIRED_FIELDS: dict[str, list[str]] = {
        "vector": ["crs", "geom_type", "schema_fields"],
        "raster": ["crs", "resolution", "bands"],
        "multidim": ["dimensions", "variables"],
        "tabular": ["schema_fields"],
        "preprocessing": ["tag", "function_description"],
        "spatial": ["tag", "function_description"],
        "temporal": ["tag", "function_description"],
        "statistical": ["tag", "function_description"],
        "visualization": ["tag", "function_description"],
        "compute": ["tag", "function_description"],
    }

    def __init__(self):
        self._data: dict = {"version": "1.0", "nodes": [], "edges": []}

    @classmethod
    def from_dict(cls, data: dict) -> GISDataFlowGraph:
        """Initialize from a full DAG dict (nodes + edges).
        Automatically validates on construction."""
        g = cls()
        g._data = copy.deepcopy(data)
        g.validate(raise_on_error=True)
        return g

    # -- validation --

    def validate(self, raise_on_error: bool = False,
                 strict: bool = True) -> tuple[list[str], list[str]]:
        """Validate graph structure. Returns (errors, warnings).

        Combines business-rule validation (node types, edge rules, cycle check),
        semantic validation, and Pydantic schema validation.

        Args:
            raise_on_error: if True, raise ValueError when any errors found.
            strict: if False, skip completeness checks (isolated nodes, missing
                    operator outputs, cycle detection, semantic validation).
                    Use strict=False during incremental DAG construction.
        """
        errors: list[str] = []
        warnings: list[str] = []
        data = self._data

        if not isinstance(data, dict):
            errors.append(f"Not a dict, got {type(data).__name__}")
            return errors

        for key in ("version", "nodes", "edges"):
            if key not in data:
                errors.append(f"Missing required field '{key}', keys: {list(data.keys())}")
        if errors:
            if raise_on_error:
                raise ValueError("\n".join(errors))
            return errors

        nodes = data["nodes"]
        edges = data["edges"]

        if not isinstance(nodes, list) or not nodes:
            errors.append("'nodes' must be a non-empty list")
            return errors
        if not isinstance(edges, list):
            errors.append("'edges' must be a list")
            return errors

        node_ids: set[str] = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"nodes[{i}] is not a dict")
                continue

            nid = _node_id(node)
            if not nid:
                errors.append(f"nodes[{i}] missing 'id' or 'node_id'")
            elif nid in node_ids:
                errors.append(f"Duplicate node id '{nid}'")
            node_ids.add(nid)

            ntype = node.get("type")
            if ntype not in self._VALID_TYPES:
                errors.append(f"Node '{nid}' has invalid type='{ntype}', must be one of {sorted(self._VALID_TYPES)}")
                continue

            for field in self._TYPE_REQUIRED_FIELDS.get(ntype, []):
                if field not in node:
                    errors.append(f"{ntype} node '{nid}' missing required field '{field}'")

        # edge validation: source/target existence
        referenced: set[str] = set()
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"edges[{j}] is not a dict")
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            referenced.add(src)
            referenced.add(tgt)
            if src not in node_ids:
                errors.append(f"Edge[{j}] source='{src}' does not exist")
            if tgt not in node_ids:
                errors.append(f"Edge[{j}] target='{tgt}' does not exist")

        # build node_id -> type map
        node_type_map: dict[str, str] = {}
        for node in nodes:
            if isinstance(node, dict):
                nid = _node_id(node)
                ntype = node.get("type")
                if nid and ntype:
                    node_type_map[nid] = ntype

        # operator-directly-connected-to-operator check
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if (node_type_map.get(src) in self._OPERATOR_TYPES
                    and node_type_map.get(tgt) in self._OPERATOR_TYPES):
                errors.append(
                    f"Edge[{j}] violates 'no operator-to-operator direct connection': "
                    f"source='{src}' (type={node_type_map.get(src)}) -> "
                    f"target='{tgt}' (type={node_type_map.get(tgt)})"
                )

        # data-directly-connected-to-data check
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if (node_type_map.get(src) in self._DATA_TYPES
                    and node_type_map.get(tgt) in self._DATA_TYPES):
                errors.append(
                    f"Edge[{j}] violates 'no data-to-data direct connection': "
                    f"source='{src}' (type={node_type_map.get(src)}) -> "
                    f"target='{tgt}' (type={node_type_map.get(tgt)})"
                )

        # operator output must go to data node
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if (node_type_map.get(src) in self._OPERATOR_TYPES
                    and node_type_map.get(tgt) not in self._DATA_TYPES):
                errors.append(
                    f"Edge[{j}] violates 'operator output must target data node': "
                    f"source='{src}' (type={node_type_map.get(src)}) -> "
                    f"target='{tgt}' (type={node_type_map.get(tgt)})"
                )

        # data -> operator edges must declare a role (the operator parameter name)
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if (node_type_map.get(src) in self._DATA_TYPES
                    and node_type_map.get(tgt) in self._OPERATOR_TYPES
                    and not edge.get("role")):
                errors.append(
                    f"Edge[{j}] data->operator must declare 'role' (operator parameter name): "
                    f"source='{src}' -> target='{tgt}'"
                )

        # every operator must be a source at least once (strict only)
        if strict:
            ops_as_source: set[str] = set()
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("source")
                    if src in node_type_map and node_type_map[src] in self._OPERATOR_TYPES:
                        ops_as_source.add(src)
            for nid, ntype in node_type_map.items():
                if ntype in self._OPERATOR_TYPES and nid not in ops_as_source:
                    errors.append(f"Operator node '{nid}' is not a source of any edge (missing output)")

            # isolated node check
            if len(node_ids) > 1:
                isolated = node_ids - referenced
                for nid in sorted(isolated):
                    errors.append(f"Isolated node '{nid}' not participating in any edge")

            # cycle detection
            if not errors:
                try:
                    self.topological_sort()
                except ValueError as e:
                    errors.append(f"Graph contains a cycle: {e}")

            # semantic validation (before Pydantic)
            sem_errors, sem_warnings = self._semantic_validate()
            errors.extend(sem_errors)
            warnings.extend(sem_warnings)

        # Pydantic final validation
        if not errors:
            try:
                _Schema.model_validate(data)
            except Exception as e:
                errors.append(f"Pydantic validation failed: {e}")

        if raise_on_error and errors:
            raise ValueError("\n".join(errors))
        return errors, warnings

    # -- semantic validation --

    def _semantic_validate(self) -> tuple[list[str], list[str]]:
        """Semantic validation of operator inputs/outputs using DAG fields.

        Returns (errors, warnings). Errors represent factually impossible
        DAGs; warnings flag suspicious patterns the LLM should self-correct.
        """
        errors: list[str] = []
        warnings: list[str] = []

        nodes = self._data.get("nodes", [])
        edges = self._data.get("edges", [])

        node_map: dict[str, dict] = {}
        node_type_map: dict[str, str] = {}
        for node in nodes:
            if isinstance(node, dict):
                nid = _node_id(node)
                node_map[nid] = node
                node_type_map[nid] = node.get("type", "")

        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = _node_id(node)
            ntype = node.get("type", "")
            if ntype not in self._OPERATOR_TYPES:
                continue

            tag = node.get("tag", "")

            # find input data nodes
            input_nodes: list[dict] = []
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                if edge.get("target") == nid:
                    src = edge.get("source")
                    if src in node_map and node_type_map.get(src) in self._DATA_TYPES:
                        input_nodes.append(node_map[src])

            # find output data nodes
            output_nodes: list[dict] = []
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                if edge.get("source") == nid:
                    tgt = edge.get("target")
                    if tgt in node_map and node_type_map.get(tgt) in self._DATA_TYPES:
                        output_nodes.append(node_map[tgt])

            if tag == "reproject":
                self._check_reproject(nid, input_nodes, output_nodes, warnings)
            elif tag == "buffer":
                self._check_buffer(nid, input_nodes, warnings)
            elif tag == "area_calculation":
                self._check_area_calculation(nid, input_nodes, warnings)
            elif tag == "zonal_stats":
                self._check_zonal_stats(nid, input_nodes, warnings)
            elif tag in ("clip", "intersection", "union", "difference"):
                self._check_binary_spatial(nid, tag, input_nodes, warnings)
            elif tag == "filter_attribute":
                self._check_filter_attribute(nid, input_nodes, output_nodes, warnings)
            elif tag == "column_arithmetic":
                self._check_column_arithmetic(nid, input_nodes, output_nodes, warnings)
            elif tag == "format_convert":
                self._check_format_convert(nid, input_nodes, output_nodes, warnings)
            elif tag == "dissolve":
                self._check_dissolve(nid, input_nodes, output_nodes, warnings)

        # Cross-branch efficiency analysis
        eff_warnings = self._check_efficiency()
        warnings.extend(eff_warnings)

        return errors, warnings

    def _check_efficiency(self) -> list[str]:
        """Cross-branch efficiency analysis.

        Detects patterns where large datasets undergo expensive spatial
        operations (buffer, reproject) on a PARALLEL branch that never
        benefits from a filter/reduction applied elsewhere, and the two
        branches only meet at a spatial join — the filter's output could
        have been used to first scope/clip the large datasets, reducing
        the data volume for the expensive operations.
        """
        warnings: list[str] = []
        nodes = self._data.get("nodes", [])
        edges = self._data.get("edges", [])

        # ── Build lookup maps ──────────────────────────────────────────
        node_map: dict[str, dict] = {}
        node_type_map: dict[str, str] = {}
        for node in nodes:
            if isinstance(node, dict):
                nid = _node_id(node)
                node_map[nid] = node
                node_type_map[nid] = node.get("type", "")

        downstream: dict[str, list[str]] = {_node_id(n): [] for n in nodes}
        upstream: dict[str, list[str]] = {_node_id(n): [] for n in nodes}
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src in downstream and tgt in upstream:
                downstream[src].append(tgt)
                upstream[tgt].append(src)

        # ── Transitive-closure helpers ──────────────────────────────────
        def _all_upstream(nid: str) -> set[str]:
            result: set[str] = set()
            stack = [nid]
            while stack:
                cur = stack.pop()
                for pred in upstream.get(cur, []):
                    if pred not in result:
                        result.add(pred)
                        stack.append(pred)
            return result

        def _all_downstream(nid: str) -> set[str]:
            result: set[str] = set()
            stack = [nid]
            while stack:
                cur = stack.pop()
                for succ in downstream.get(cur, []):
                    if succ not in result:
                        result.add(succ)
                        stack.append(succ)
            return result

        # ── 1. Identify reduction nodes (stable tag-based) ─────────────
        # Tags whose operators reliably reduce feature/row count.
        _REDUCE_TAGS = {"filter_attribute", "clip"}
        reduction_nodes: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = _node_id(node)
            ntype = node.get("type", "")
            if ntype not in self._OPERATOR_TYPES:
                continue
            if node.get("tag") in _REDUCE_TAGS:
                reduction_nodes.add(nid)

        if not reduction_nodes:
            return warnings

        # ── 2. Per-reduction cross-branch analysis ─────────────────────
        # Expensive spatial ops — only meaningful on vector data.
        _EXPENSIVE_TAGS = {"buffer", "reproject"}

        for r_nid in sorted(reduction_nodes):
            r_node = node_map.get(r_nid, {})
            r_name = r_node.get("name", r_nid)

            # Output data node(s) of the reduction operator
            r_outputs = [
                tgt for tgt in downstream.get(r_nid, [])
                if node_type_map.get(tgt) in self._DATA_TYPES
            ]
            if not r_outputs:
                continue

            r_output_nid = r_outputs[0]
            r_downstream_set = _all_downstream(r_output_nid) | {r_output_nid, r_nid}
            r_upstream_set = _all_upstream(r_nid) | {r_nid}

            # Find spatial_join nodes downstream of the reduction output
            for ds_nid in r_downstream_set:
                ds_type = node_type_map.get(ds_nid, "")
                ds_node = node_map.get(ds_nid, {})
                if ds_type != "spatial":
                    continue
                if ds_node.get("tag") != "spatial_join":
                    continue
                sj_name = ds_node.get("name", ds_nid)

                # Examine each data input to the spatial join
                sj_inputs = [
                    pred for pred in upstream.get(ds_nid, [])
                    if node_type_map.get(pred) in self._DATA_TYPES
                ]

                for sj_inp in sj_inputs:
                    # Skip inputs that are part of the reduction chain
                    if sj_inp in r_downstream_set or sj_inp in r_upstream_set:
                        continue

                    inp_upstream = _all_upstream(sj_inp)

                    # Expensive ops in this input's branch
                    expensive_ops: list[tuple[str, str]] = []
                    for up_nid in inp_upstream:
                        up_type = node_type_map.get(up_nid, "")
                        if up_type not in self._OPERATOR_TYPES:
                            continue
                        up_tag = node_map.get(up_nid, {}).get("tag", "")
                        if up_tag in _EXPENSIVE_TAGS:
                            expensive_ops.append((up_nid, up_tag))

                    if not expensive_ops:
                        continue

                    # Root data nodes in this branch (full datasets).
                    # Only vector data — buffer/reproject operate on vectors.
                    root_data: list[str] = []
                    for up_nid in inp_upstream | {sj_inp}:
                        up_type = node_type_map.get(up_nid, "")
                        if up_type not in self._DATA_TYPES:
                            continue
                        if up_type != "vector":
                            continue
                        has_op_producer = any(
                            node_type_map.get(p, "") in self._OPERATOR_TYPES
                            for p in upstream.get(up_nid, [])
                        )
                        if not has_op_producer:
                            root_data.append(up_nid)

                    # Exclude root data that the filter already operates on
                    extraneous_root = [d for d in root_data if d not in r_upstream_set]
                    if not extraneous_root:
                        continue

                    # Does the reduction already scope this branch?
                    reduction_scopes_branch = bool(
                        inp_upstream & r_downstream_set
                    )
                    if reduction_scopes_branch:
                        continue

                    root_names = [node_map.get(d, {}).get("name", d) for d in extraneous_root]
                    ops_desc = ", ".join(
                        f"{nid}({tag})" for nid, tag in expensive_ops
                    )

                    warnings.append(
                        f"[efficiency] '{sj_name}' ({ds_nid}): "
                        f"input from {root_names} undergoes [{ops_desc}] "
                        f"on full data in parallel with '{r_name}' ({r_nid}). "
                        f"Consider using '{r_name}' output to clip/scope "
                        f"{root_names} BEFORE these operations."
                    )

        return warnings

    def _check_reproject(self, nid: str, input_nodes: list[dict],
                         output_nodes: list[dict], warnings: list[str]) -> None:
        input_crs = input_nodes[0].get("crs") if input_nodes else None
        output_crs = output_nodes[0].get("crs") if output_nodes else None
        if input_crs and output_crs and input_crs == output_crs:
            warnings.append(
                f"[semantic] Operator '{nid}' (tag='reproject'): "
                f"input CRS ({input_crs}) equals output CRS ({output_crs}). "
                f"Reprojection should change the coordinate reference system."
            )

    def _check_buffer(self, nid: str, input_nodes: list[dict],
                      warnings: list[str]) -> None:
        for inp in input_nodes:
            crs = inp.get("crs", "")
            if _is_geographic_crs(crs):
                warnings.append(
                    f"[semantic] Operator '{nid}' (tag='buffer'): "
                    f"input '{inp.get('id')}' has geographic CRS ({crs}). "
                    f"Buffering in degrees produces meaningless results; "
                    f"reproject to a projected CRS first."
                )

    def _check_area_calculation(self, nid: str, input_nodes: list[dict],
                                warnings: list[str]) -> None:
        for inp in input_nodes:
            crs = inp.get("crs", "")
            if _is_geographic_crs(crs):
                warnings.append(
                    f"[semantic] Operator '{nid}' (tag='area_calculation'): "
                    f"input '{inp.get('id')}' has geographic CRS ({crs}). "
                    f"Area calculated in degrees is meaningless; "
                    f"reproject to a projected CRS first."
                )

    def _check_zonal_stats(self, nid: str, input_nodes: list[dict],
                           warnings: list[str]) -> None:
        crs_list = [inp.get("crs") for inp in input_nodes if inp.get("crs")]
        if len(crs_list) >= 2 and len(set(crs_list)) > 1:
            warnings.append(
                f"[semantic] Operator '{nid}' (tag='zonal_stats'): "
                f"inputs have mismatched CRS: {crs_list}. "
                f"Vector and raster inputs must share the same CRS."
            )

    def _check_binary_spatial(self, nid: str, tag: str, input_nodes: list[dict],
                              warnings: list[str]) -> None:
        crs_list = [inp.get("crs") for inp in input_nodes if inp.get("crs")]
        if len(crs_list) >= 2 and len(set(crs_list)) > 1:
            warnings.append(
                f"[semantic] Operator '{nid}' (tag='{tag}'): "
                f"inputs have mismatched CRS: {crs_list}. "
                f"Both inputs must share the same CRS."
            )

    def _check_filter_attribute(self, nid: str, input_nodes: list[dict],
                                output_nodes: list[dict],
                                warnings: list[str]) -> None:
        if not input_nodes or not output_nodes:
            return
        input_fields = set(input_nodes[0].get("schema_fields", []))
        output_fields = set(output_nodes[0].get("schema_fields", []))
        if input_fields and output_fields:
            extra = output_fields - input_fields
            if extra:
                warnings.append(
                    f"[semantic] Operator '{nid}' (tag='filter_attribute'): "
                    f"output has fields not in input: {sorted(extra)}. "
                    f"Filter operations should not create new columns."
                )

    def _check_column_arithmetic(self, nid: str, input_nodes: list[dict],
                                 output_nodes: list[dict],
                                 warnings: list[str]) -> None:
        if not input_nodes or not output_nodes:
            return
        input_fields = set(input_nodes[0].get("schema_fields", []))
        output_fields = set(output_nodes[0].get("schema_fields", []))
        if input_fields and output_fields:
            missing = input_fields - output_fields
            if missing:
                warnings.append(
                    f"[semantic] Operator '{nid}' (tag='column_arithmetic'): "
                    f"output is missing input fields: {sorted(missing)}. "
                    f"Column arithmetic should retain original columns plus new ones."
                )

    def _check_format_convert(self, nid: str, input_nodes: list[dict],
                              output_nodes: list[dict],
                              warnings: list[str]) -> None:
        if not input_nodes or not output_nodes:
            return
        input_fields = set(input_nodes[0].get("schema_fields", []))
        output_fields = set(output_nodes[0].get("schema_fields", []))
        if input_fields and output_fields and input_fields != output_fields:
            warnings.append(
                f"[semantic] Operator '{nid}' (tag='format_convert'): "
                f"output fields differ from input fields. "
                f"Format conversion should preserve all schema fields."
            )

    def _check_dissolve(self, nid: str, input_nodes: list[dict],
                        output_nodes: list[dict], warnings: list[str]) -> None:
        if not input_nodes or not output_nodes:
            return
        input_fields = set(input_nodes[0].get("schema_fields", []))
        output_fields = set(output_nodes[0].get("schema_fields", []))
        if input_fields and output_fields:
            extra = output_fields - input_fields
            if extra:
                warnings.append(
                    f"[semantic] Operator '{nid}' (tag='dissolve'): "
                    f"output has fields not in input: {sorted(extra)}. "
                    f"Dissolve should not create new columns."
                )

    # -- topological sort --

    def topological_sort(self) -> list[list[str]]:
        """Layered topological sort. Returns list of layers, each a sorted
        list of node IDs that can execute in parallel.

        Raises:
            ValueError: if the graph has a cycle.
        """
        nodes = self._data.get("nodes", [])
        edges = self._data.get("edges", [])

        adjacency: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}

        for node in nodes:
            nid = _node_id(node)
            adjacency[nid] = []
            in_degree[nid] = 0

        for edge in edges:
            for src in _edge_sources(edge):
                tgt = _edge_target(edge)
                adjacency[src].append(tgt)
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        layers: list[list[str]] = []
        zero_in = [nid for nid, deg in in_degree.items() if deg == 0]

        while zero_in:
            layers.append(sorted(zero_in))
            next_zero: list[str] = []
            for nid in zero_in:
                for neighbor in adjacency[nid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_zero.append(neighbor)
            zero_in = next_zero

        total = sum(len(layer) for layer in layers)
        if total != len(nodes):
            raise ValueError(
                f"Graph contains a cycle, cannot topologically sort. "
                f"Sorted {total}/{len(nodes)} nodes."
            )

        return layers

    def generate_task_sequence(self) -> list[list[TaskStep]]:
        """Generate a task sequence with dependency context, based on
        topological sort."""
        nodes = self._data.get("nodes", [])
        edges = self._data.get("edges", [])

        node_map: dict[str, dict] = {}
        for node in nodes:
            nid = _node_id(node)
            node_map[nid] = node

        incoming: dict[str, list[dict]] = {nid: [] for nid in node_map}
        for edge in edges:
            for src in _edge_sources(edge):
                tgt = _edge_target(edge)
                incoming.setdefault(tgt, []).append(node_map.get(src, {}))

        layers = self.topological_sort()
        task_sequence: list[list[TaskStep]] = []

        for depth, layer_ids in enumerate(layers):
            steps: list[TaskStep] = []
            for nid in layer_ids:
                step = TaskStep(
                    depth=depth,
                    node_id=nid,
                    node_info=node_map[nid],
                    deps=incoming.get(nid, []),
                )
                steps.append(step)
            task_sequence.append(steps)

        return task_sequence

    # -- graph queries --

    def inspect(self, identifier: str) -> dict:
        """Query by node ID or name. Returns detailed info including
        predecessors, successors, in/out degree and node type."""
        nodes = self._data.get("nodes", [])
        edges = self._data.get("edges", [])

        # find the node
        target = None
        for node in nodes:
            nid = _node_id(node)
            if nid == identifier or node.get("name") == identifier:
                target = node
                break
        if target is None:
            raise KeyError(f"No node with id or name '{identifier}'")

        nid = _node_id(target)
        predecessors: list[dict] = []
        successors: list[dict] = []
        in_degree = 0
        out_degree = 0

        node_map: dict[str, dict] = {_node_id(n): n for n in nodes}

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            edge_info = {
                "source": src,
                "target": tgt,
                "role": edge.get("role"),
            }
            # include format_hint if present
            for k, v in edge.items():
                if k not in ("source", "target", "role"):
                    edge_info[k] = v

            if tgt == nid:
                in_degree += 1
                pred_node = node_map.get(src, {})
                predecessors.append({
                    "node_id": src,
                    "node_name": pred_node.get("name"),
                    "node_type": pred_node.get("type"),
                    "node_info": pred_node,
                    "edge": edge_info,
                })
            if src == nid:
                out_degree += 1
                succ_node = node_map.get(tgt, {})
                successors.append({
                    "node_id": tgt,
                    "node_name": succ_node.get("name"),
                    "node_type": succ_node.get("type"),
                    "node_info": succ_node,
                    "edge": edge_info,
                })

        return {
            "id": nid,
            "name": target.get("name"),
            "type": target.get("type"),
            "info": target,
            "predecessors": predecessors,
            "successors": successors,
            "in_degree": in_degree,
            "out_degree": out_degree,
        }

    # -- serialization --

    def to_dict(self) -> dict:
        """Return a deep copy of internal dict for JSON serialization."""
        return copy.deepcopy(self._data)

    # -- save to local files --

    def save_local(self, output_dir: str = ".", filename_prefix: str = "dag") -> str:
        """Save the DAG JSON and a visualization image to local files.

        Writes two files:
          - ``<filename_prefix>.json`` — the full DAG JSON representation
          - ``<filename_prefix>.png`` — a visual rendering of the graph

        Args:
            output_dir: Directory to save the files (default: current directory).
            filename_prefix: Base name for output files (default: ``"dag"``).

        Returns:
            A summary string with saved file paths, or a partial error message.
        """
        _os.makedirs(output_dir, exist_ok=True)

        # 1. Save DAG JSON
        json_path = _os.path.join(output_dir, f"{filename_prefix}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # 2. Generate and save visualization
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import networkx as nx
            from collections import defaultdict

            data = self.to_dict()
            nx_g = nx.DiGraph()

            # node type -> color / label mapping
            type_colors = {
                "vector": "#4CAF50",
                "raster": "#2196F3",
                "multidim": "#9C27B0",
                "tabular": "#FF9800",
                "preprocessing": "#00BCD4",
                "spatial": "#F44336",
                "temporal": "#E91E63",
                "statistical": "#795548",
                "visualization": "#FF5722",
                "compute": "#607D8B",
            }
            type_labels = {
                "vector": "vector",
                "raster": "raster",
                "multidim": "multidim",
                "tabular": "tabular",
                "preprocessing": "preprocessing",
                "spatial": "spatial",
                "temporal": "temporal",
                "statistical": "statistical",
                "visualization": "visualization",
                "compute": "compute",
            }

            # add nodes
            for node in data.get("nodes", []):
                nid = _node_id(node)
                ntype = node.get("type", "unknown")
                name = node.get("name", nid)
                nx_g.add_node(nid, label=name, node_type=ntype, color=type_colors.get(ntype, "#9E9E9E"))

            # add edges
            for edge in data.get("edges", []):
                src = edge.get("source")
                tgt = edge.get("target")
                nx_g.add_edge(src, tgt, label=edge.get("role", ""))

            if not nx_g.nodes:
                return f"SUCCESS: DAG JSON saved to {json_path}.\n  PNG:  skipped (empty graph)."

            # layered layout via topological generations
            try:
                layers = {node: i for i, layer in enumerate(nx.topological_generations(nx_g)) for node in layer}
            except nx.NetworkXUnfeasible:
                layers = {node: 0 for node in nx_g.nodes}

            layer_nodes_map = defaultdict(list)
            for node, layer in layers.items():
                layer_nodes_map[layer].append(node)

            pos = {}
            for layer, nodes in sorted(layer_nodes_map.items()):
                for i, node in enumerate(nodes):
                    pos[node] = (i - (len(nodes) - 1) / 2, -layer)

            # auto-detect CJK font
            cjk_font = None
            for family in ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
                           "Source Han Sans CN", "Microsoft YaHei", "SimHei"]:
                if fm.findfont(family, fallback_to_default=False) != fm.findfont("DejaVu Sans"):
                    cjk_font = family
                    break
            if cjk_font:
                plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, ax = plt.subplots(
                figsize=(max(10, len(nx_g.nodes) * 1.5), max(6, len(set(layers.values())) * 2))
            )

            # draw nodes as rounded boxes
            for node, attrs in nx_g.nodes(data=True):
                x, y = pos[node]
                color = attrs.get("color", "#9E9E9E")
                ntype = attrs.get("node_type", "")
                tag = f"[{type_labels.get(ntype, ntype)}]" if ntype else ""
                label = f"{attrs['label']}\n{tag}"
                ax.annotate(
                    label, xy=(x, y), fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#333", alpha=0.9),
                    color="white", fontweight="bold",
                )

            # draw edges
            edge_labels = {}
            for u, v, attrs in nx_g.edges(data=True):
                role = attrs.get("label", "")
                if role:
                    edge_labels[(u, v)] = role

            nx.draw_networkx_edges(nx_g, pos, ax=ax, arrows=True, arrowsize=20,
                                   edge_color="#666", width=1.5, node_size=3000)
            if edge_labels:
                nx.draw_networkx_edge_labels(nx_g, pos, ax=ax, edge_labels=edge_labels,
                                             font_size=8, font_color="#555")

            ax.set_title(f"DAG Data Flow Graph — {filename_prefix}", fontsize=14, fontweight="bold", pad=20)
            ax.axis("off")
            plt.tight_layout()

            png_path = _os.path.join(output_dir, f"{filename_prefix}.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight", transparent=False)
            plt.close(fig)

            return f"SUCCESS: DAG saved.\n  JSON: {json_path}\n  PNG:  {png_path}"

        except ImportError:
            return (
                f"SUCCESS: DAG JSON saved to {json_path}.\n"
                "Visualization skipped: install 'networkx' and 'matplotlib' for PNG output."
            )
        except Exception as e:
            return (
                f"PARTIAL: DAG JSON saved to {json_path}.\n"
                f"Visualization failed: {e}"
            )


# ---------------------------------------------------------------------------
# Public constants (importable by other modules)
# ---------------------------------------------------------------------------

OPERATOR_TYPES = GISDataFlowGraph._OPERATOR_TYPES
DATA_TYPES = GISDataFlowGraph._DATA_TYPES


# ---------------------------------------------------------------------------
# Module-level graph instance (shared across tool calls)
# ---------------------------------------------------------------------------

_current_graph: Optional[GISDataFlowGraph] = None


def _get_graph() -> Optional[GISDataFlowGraph]:
    return _current_graph


def reset_graph() -> None:
    """Reset the module-level graph instance between tasks."""
    global _current_graph
    _current_graph = None


@tool
def from_dict(graph_json: str) -> str:
    """Create a GISDataFlowGraph from a JSON string and store it as the active graph.

    This tool constructs a data flow diagram from a JSON description containing
    nodes (vector, raster, multidim, tabular, + 6 operator types) and edges.

    Args:
        graph_json: A JSON string with "version", "nodes", and "edges" fields.
                    Example: '{"version":"1.0","nodes":[...],"edges":[...]}'

    Returns:
        A summary string with node count, edge count, and validation status.
        If validation fails, returns the error details.
    """
    global _current_graph

    try:
        data = _json.loads(graph_json)
    except _json.JSONDecodeError as e:
        return f"FAILED: Invalid JSON — {e}"

    graph = GISDataFlowGraph()
    graph._data = copy.deepcopy(data)

    errors, warnings = graph.validate()
    if errors:
        error_msg = "\n".join(errors)
        return f"FAILED: Graph validation errors:\n{error_msg}"

    _current_graph = graph
    graph.save_local()

    node_count = len(data.get("nodes", []))
    edge_count = len(data.get("edges", []))
    node_types = {}
    for node in data.get("nodes", []):
        ntype = node.get("type", "unknown")
        node_types[ntype] = node_types.get(ntype, 0) + 1

    type_summary = ", ".join(f"{k}: {v}" for k, v in node_types.items())
    msg = (
        f"SUCCESS: Graph created with {node_count} nodes, {edge_count} edges.\n"
        f"Node types: {type_summary}\n"
        f"Graph is ready for inspection."
    )
    if warnings:
        msg += "\n\nWarnings:\n" + "\n".join(warnings)
    return msg


@tool
def inspect(identifier: str, depth: int = None) -> str:
    """Inspect a node in the active GISDataFlowGraph by its ID or name.

    Returns detailed information including predecessors, successors,
    in/out degree and node type.

    Args:
        identifier: The node ID (e.g. "node-1") or node name to inspect.
        depth: Optional. If provided, inspect all nodes at this depth layer
               in the topological sort instead of a specific node.

    Returns:
        A formatted string with the node's detailed information,
        or a summary of all nodes at the specified depth layer.
    """
    graph = _get_graph()
    if graph is None:
        return "FAILED: No active graph. Use set_DAG to create one, then fix errors with update_node/add_edge."

    if depth is not None:
        try:
            layers = graph.topological_sort()
            if depth < 0 or depth >= len(layers):
                return f"FAILED: Depth {depth} out of range. Graph has {len(layers)} layers (0-{len(layers)-1})."
            layer_ids = layers[depth]
            lines = [f"Layer {depth} ({len(layer_ids)} nodes):"]
            for nid in layer_ids:
                info = graph.inspect(nid)
                lines.append(f"  - {nid} (type={info['type']}, name={info['name']})")
            return "\n".join(lines)
        except ValueError as e:
            return f"FAILED: {e}"

    try:
        info = graph.inspect(identifier)
        lines = [
            f"Node: {info['id']}",
            f"Name: {info['name']}",
            f"Type: {info['type']}",
            f"In-degree: {info['in_degree']}",
            f"Out-degree: {info['out_degree']}",
        ]
        # node detail fields
        node_info = info.get("info", {})
        extra_fields = {k: v for k, v in node_info.items()
                        if k not in ("id", "node_id", "name", "description", "type")}
        if extra_fields:
            lines.append("Fields:")
            for k, v in extra_fields.items():
                lines.append(f"  {k}: {v!r}")

        if info["predecessors"]:
            lines.append("Predecessors:")
            for pred in info["predecessors"]:
                p_lines = [
                    f"  - Node: {pred['node_id']}",
                    f"    Name: {pred['node_name']}",
                    f"    Type: {pred['node_type']}",
                ]
                edge = pred.get("edge", {})
                if edge:
                    p_lines.append(f"    Edge:")
                    p_lines.append(f"      role: {edge.get('role')!r}")
                    for ek, ev in edge.items():
                        if ek not in ("source", "target", "role"):
                            p_lines.append(f"      {ek}: {ev!r}")
                lines.append("\n".join(p_lines))
        if info["successors"]:
            lines.append("Successors:")
            for succ in info["successors"]:
                s_lines = [
                    f"  - Node: {succ['node_id']}",
                    f"    Name: {succ['node_name']}",
                    f"    Type: {succ['node_type']}",
                ]
                edge = succ.get("edge", {})
                if edge:
                    s_lines.append(f"    Edge:")
                    s_lines.append(f"      role: {edge.get('role')!r}")
                    for ek, ev in edge.items():
                        if ek not in ("source", "target", "role"):
                            s_lines.append(f"      {ek}: {ev!r}")
                lines.append("\n".join(s_lines))
        return "\n".join(lines)
    except KeyError as e:
        return f"FAILED: {e}"


# ---------------------------------------------------------------------------
# Incremental DAG editing tools (graph_planner)
# ---------------------------------------------------------------------------

def _dag_summary(graph: GISDataFlowGraph) -> str:
    """Return a human-readable summary of the DAG."""
    nodes = graph._data.get("nodes", [])
    edges = graph._data.get("edges", [])

    lines = [f"DAG Summary: {len(nodes)} nodes, {len(edges)} edges"]

    # Node type breakdown
    type_counts: dict = {}
    for n in nodes:
        nt = n.get("type", "unknown")
        type_counts[nt] = type_counts.get(nt, 0) + 1
    type_parts = []
    for nt in sorted(type_counts.keys()):
        type_parts.append(f"{nt}: {type_counts[nt]}")
    if type_parts:
        lines.append(f"  Node types: {', '.join(type_parts)}")

    # All nodes brief
    lines.append("  Nodes:")
    for n in nodes:
        nid = _node_id(n)
        nt = n.get("type", "?")
        name = n.get("name", "")
        extra = ""
        if nt in GISDataFlowGraph._DATA_TYPES:
            crs = n.get("crs", "")
            extra = f", crs={crs}" if crs else ""
        elif nt in GISDataFlowGraph._OPERATOR_TYPES:
            tag = n.get("tag", "")
            extra = f", tag={tag}" if tag else ""
        lines.append(f"    {nid} [{nt}] {name}{extra}")

    # All edges
    if edges:
        lines.append("  Edges:")
        for e in edges:
            src = e.get("source", "?")
            tgt = e.get("target", "?")
            role = e.get("role", "")
            role_str = f" ({role})" if role else ""
            lines.append(f"    {src} -> {tgt}{role_str}")
    else:
        lines.append("  Edges: (none)")

    # Topological layers
    try:
        layers = graph.topological_sort()
        lines.append(f"  Topological layers ({len(layers)}):")
        for depth, layer_ids in enumerate(layers):
            ids = ", ".join(layer_ids)
            lines.append(f"    Layer {depth}: {ids}")
    except ValueError as e:
        lines.append(f"  Topology: ERROR — {e}")

    return "\n".join(lines)


@tool
def set_DAG(dag_json: str) -> str:
    """Set the entire DAG from a JSON string — use for initial creation or full rebuilds.

    Pass a complete DAG JSON with "nodes" and "edges" arrays. The JSON is
    validated for correct structure before being applied. This is the fastest
    way to create or replace the entire DAG.

    Validation runs automatically — no need to call validate_DAG separately.
    For small fixes (changing a single CRS, adding one edge), use edit_DAG instead.

    Args:
        dag_json: Complete DAG JSON string, e.g.
            {"version":"1.0","nodes":[...],"edges":[...]}

    Returns:
        Status message with DAG summary on success, or error details on failure.
    """
    global _current_graph

    # Parse JSON
    try:
        new_data = _json.loads(dag_json)
    except _json.JSONDecodeError as e:
        return f"FAILED: Invalid JSON — {e}"

    # Basic structure check
    if not isinstance(new_data, dict):
        return "FAILED: DAG must be a JSON object"
    if "nodes" not in new_data:
        return "FAILED: DAG must have a 'nodes' key"
    if "edges" not in new_data:
        return "FAILED: DAG must have an 'edges' key"
    if not isinstance(new_data["nodes"], list):
        return "FAILED: 'nodes' must be an array"
    if not isinstance(new_data["edges"], list):
        return "FAILED: 'edges' must be an array"

    # Ensure version
    if "version" not in new_data:
        new_data["version"] = "1.0"

    # Create or replace graph
    if _current_graph is None:
        _current_graph = GISDataFlowGraph()
    _current_graph._data = new_data
    _current_graph.save_local()

    nodes = new_data["nodes"]
    edges = new_data["edges"]
    lines = [f"OK: DAG set ({len(nodes)} nodes, {len(edges)} edges)"]

    # Auto-validate after set_DAG
    validation_result = validate_DAG()
    lines.append("\n--- Validation ---")
    lines.append(validation_result)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Incremental repair tools — fix specific nodes/edges without regenerating
# the entire DAG. Use after set_DAG when validation reports errors.
# ---------------------------------------------------------------------------

@tool
def update_node(node_id: str, updates_json: str) -> str:
    """Update a single node's fields in the DAG. Use to fix validation errors without regenerating the entire DAG.

    After set_DAG validation reports errors like "node-3 missing crs" or
    "node-5 wrong geom_type", use this to surgically fix only the broken fields.

    Args:
        node_id: The ID of the node to update, e.g. "node-3"
        updates_json: JSON object with the fields to update, e.g.
            {"crs": "EPSG:4326"} or {"schema_fields": ["name", "pop"]}

    Returns:
        Status message with the update result.
    """
    global _current_graph
    if _current_graph is None:
        return "FAILED: No active DAG. Use set_DAG first."

    try:
        updates = _json.loads(updates_json)
    except _json.JSONDecodeError as e:
        return f"FAILED: Invalid JSON — {e}"

    if not isinstance(updates, dict):
        return "FAILED: updates_json must be a JSON object"

    nodes = _current_graph._data.get("nodes", [])
    target_node = None
    for node in nodes:
        nid = _node_id(node)
        if nid == node_id:
            target_node = node
            break

    if target_node is None:
        return f"FAILED: Node '{node_id}' not found. Existing IDs: {', '.join(_node_id(n) for n in nodes)}"

    # Apply updates
    changed = []
    for key, value in updates.items():
        if key in ("id", "node_id", "type"):
            return f"FAILED: Cannot change core field '{key}' via update_node. Use remove_node + set_DAG instead."
        old = target_node.get(key)
        target_node[key] = value
        changed.append(f"  {key}: {_json.dumps(old)} → {_json.dumps(value)}")

    _current_graph.save_local()

    # Lightweight validation (schema only, not graph structure)
    errors, warnings = _current_graph.validate(strict=False)
    result = [f"OK: Updated node '{node_id}' ({len(changed)} field(s)):"]
    result.extend(changed)
    if errors:
        result.append(f"\nWARNING — {len(errors)} schema error(s):")
        for e in errors:
            result.append(f"  [ERROR] {e}")
    if warnings:
        result.append(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            result.append(f"  [WARNING] {w}")

    return "\n".join(result)


@tool
def remove_node(node_id: str) -> str:
    """Remove a node and all its associated edges from the DAG.

    Use to delete an incorrectly created node. All edges connected to this
    node are also removed. After removal, recheck validation with validate_DAG.

    Args:
        node_id: The ID of the node to remove, e.g. "node-3"

    Returns:
        Status message with removal result.
    """
    global _current_graph
    if _current_graph is None:
        return "FAILED: No active DAG. Use set_DAG first."

    nodes = _current_graph._data.get("nodes", [])
    edges = _current_graph._data.get("edges", [])

    target = None
    for node in nodes:
        if _node_id(node) == node_id:
            target = node
            break

    if target is None:
        return f"FAILED: Node '{node_id}' not found. Existing IDs: {', '.join(_node_id(n) for n in nodes)}"

    # Remove node
    nodes.remove(target)

    # Remove all edges connected to this node
    removed_edges = []
    kept_edges = []
    for edge in edges:
        if edge.get("source") == node_id or edge.get("target") == node_id:
            removed_edges.append(f"  {edge.get('source')} → {edge.get('target')}")
        else:
            kept_edges.append(edge)
    _current_graph._data["edges"] = kept_edges
    _current_graph.save_local()

    result = [f"OK: Removed node '{node_id}' (type={target.get('type')}, name={target.get('name')})"]
    if removed_edges:
        result.append(f"Also removed {len(removed_edges)} edge(s):")
        result.extend(removed_edges)
    result.append(f"Remaining: {len(nodes)} nodes, {len(kept_edges)} edges")

    return "\n".join(result)


@tool
def add_edge(edge_json: str) -> str:
    """Add a single edge to the DAG. Use to connect nodes when validation reports missing edges.

    Args:
        edge_json: JSON object with source, target, and optional role, e.g.
            {"source": "node-3", "target": "node-5", "role": "left"}
            or {"source": "node-5", "target": "node-7"}

    Returns:
        Status message with the result.
    """
    global _current_graph
    if _current_graph is None:
        return "FAILED: No active DAG. Use set_DAG first."

    try:
        edge = _json.loads(edge_json)
    except _json.JSONDecodeError as e:
        return f"FAILED: Invalid JSON — {e}"

    if not isinstance(edge, dict):
        return "FAILED: edge_json must be a JSON object"

    src = edge.get("source")
    tgt = edge.get("target")
    if not src:
        return "FAILED: Edge must have a 'source' key"
    if not tgt:
        return "FAILED: Edge must have a 'target' key"

    nodes = _current_graph._data.get("nodes", [])
    node_ids = {_node_id(n) for n in nodes}
    if src not in node_ids:
        return f"FAILED: Source '{src}' does not exist. Existing IDs: {', '.join(sorted(node_ids))}"
    if tgt not in node_ids:
        return f"FAILED: Target '{tgt}' does not exist. Existing IDs: {', '.join(sorted(node_ids))}"

    # Check for duplicate edge
    existing_edges = _current_graph._data.get("edges", [])
    for existing in existing_edges:
        if existing.get("source") == src and existing.get("target") == tgt:
            return f"FAILED: Edge {src} → {tgt} already exists"

    new_edge = {"source": src, "target": tgt}
    if "role" in edge:
        new_edge["role"] = edge["role"]

    existing_edges.append(new_edge)
    _current_graph.save_local()

    # Lightweight validation
    errors, warnings = _current_graph.validate(strict=False)
    role_str = f", role={edge['role']}" if "role" in edge else ""
    result = [f"OK: Added edge {src} → {tgt}{role_str} ({len(existing_edges)} total edges)"]
    if errors:
        result.append(f"\nWARNING — {len(errors)} error(s):")
        for e in errors:
            result.append(f"  [ERROR] {e}")
    if warnings:
        result.append(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            result.append(f"  [WARNING] {w}")

    return "\n".join(result)


@tool
def remove_edge(source: str, target: str) -> str:
    """Remove a specific edge from the DAG.

    Args:
        source: The source node ID of the edge to remove
        target: The target node ID of the edge to remove

    Returns:
        Status message with removal result.
    """
    global _current_graph
    if _current_graph is None:
        return "FAILED: No active DAG. Use set_DAG first."

    edges = _current_graph._data.get("edges", [])
    found = None
    for i, edge in enumerate(edges):
        if edge.get("source") == source and edge.get("target") == target:
            found = edges.pop(i)
            break

    if found is None:
        edge_list = ", ".join(f"{e.get('source')}→{e.get('target')}" for e in edges)
        return f"FAILED: Edge {source} → {target} not found. Existing edges: [{edge_list}]"

    _current_graph.save_local()
    role_str = f", role={found['role']}" if found.get("role") else ""
    return f"OK: Removed edge {source} → {target}{role_str} ({len(edges)} edges remaining)"


@tool
def show_dag() -> str:
    """Display the current DAG state — all nodes and edges organized by topological layer.

    Use to inspect the DAG before final validation. Shows node types, names,
    and the edge connections between them.

    Returns:
        Formatted DAG summary string.
    """
    graph = _get_graph()
    if graph is None:
        return "No active DAG. Use set_DAG to create one."

    nodes = graph._data.get("nodes", [])
    edges = graph._data.get("edges", [])

    lines = [f"DAG Summary: {len(nodes)} nodes, {len(edges)} edges\n"]

    # Try topological sort for structured display
    try:
        layers = graph.topological_sort()
        lines.append(f"Topological layers ({len(layers)}):")
        node_map = {_node_id(n): n for n in nodes}
        for depth, layer_ids in enumerate(layers):
            layer_label = " (FINAL OUTPUT)" if depth == len(layers) - 1 else ""
            lines.append(f"  Layer {depth}{layer_label}:")
            for nid in layer_ids:
                node = node_map.get(nid, {})
                ntype = node.get("type", "?")
                name = node.get("name", "?")
                extra = ""
                if ntype in DATA_TYPES:
                    crs = node.get("crs", "")
                    if crs:
                        extra = f" [crs={crs}]"
                elif ntype in OPERATOR_TYPES:
                    tag = node.get("tag", "")
                    if tag:
                        extra = f" [tag={tag}]"
                lines.append(f"    {nid} ({ntype}): {name}{extra}")
    except ValueError as e:
        lines.append(f"Topology: ERROR — {e}")
        lines.append("\nNodes:")
        for node in nodes:
            nid = _node_id(node)
            lines.append(f"  {nid} ({node.get('type', '?')}): {node.get('name', '?')}")

    lines.append(f"\nEdges ({len(edges)}):")
    for edge in edges:
        src = edge.get("source", "?")
        tgt = edge.get("target", "?")
        role = edge.get("role", "")
        role_str = f" [{role}]" if role else ""
        lines.append(f"  {src} → {tgt}{role_str}")

    return "\n".join(lines)


def validate_DAG() -> str:
    """Validate the current DAG for structural and semantic correctness.

    Runs the full validation suite: schema validation, graph structure checks,
    and semantic checks (e.g. CRS consistency, geographic CRS warnings for
    buffer/area operations).

    When valid, also displays the topological layer breakdown.

    Called automatically by set_DAG — use update_node/add_edge/remove_node/remove_edge to fix reported errors.

    Returns:
        "VALID" with topological layers, or a list of errors and warnings.
    """
    graph = _get_graph()
    if graph is None:
        return "No active DAG. Use set_DAG or edit_DAG to start building one."

    errors, warnings = graph.validate(strict=True)
    lines = []

    if errors:
        lines.append(f"INVALID — {len(errors)} error(s):")
        for e in errors:
            lines.append(f"  [ERROR] {e}")
    else:
        lines.append("VALID — DAG is structurally and semantically correct.")

    if warnings:
        lines.append(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            lines.append(f"  [WARNING] {w}")

    # When valid, show topological layers
    if not errors:
        try:
            layers = graph.topological_sort()
            lines.append(f"\nTopological layers ({len(layers)}):")
            for depth, layer_ids in enumerate(layers):
                lines.append(f"  Layer {depth}: {', '.join(layer_ids)}")
        except ValueError as e:
            lines.append(f"\nTopology: ERROR — {e}")

    return "\n".join(lines)
