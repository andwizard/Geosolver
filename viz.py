"""Visualization helpers for the GeoSolver Streamlit frontend.

Three groups of helpers:
  * dataset / result preview — render vector, raster and tabular files
    (vectors and raster footprints go on an interactive Leaflet map);
  * DAG rendering — turn a geosolver DAG dict into Graphviz DOT.

Folium maps are embedded via their raw HTML (`_repr_html_`) so we don't need
the extra `streamlit-folium` dependency.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import streamlit as st

VECTOR_EXTS = {".shp", ".geojson", ".gpkg", ".kml", ".gml"}
RASTER_EXTS = {".tif", ".tiff", ".img", ".vrt"}
TABLE_EXTS = {".csv", ".xlsx", ".xls"}
MAX_FEATURES = 1500  # cap features drawn on a map for responsiveness


def file_kind(path: Path) -> str:
    """Classify a file into 'vector' | 'raster' | 'table' | 'zip' | 'other'."""
    ext = path.suffix.lower()
    if ext in VECTOR_EXTS:
        return "vector"
    if ext in RASTER_EXTS:
        return "raster"
    if ext in TABLE_EXTS:
        return "table"
    if ext == ".zip":
        return "zip"
    if ext == ".json":  # could be GeoJSON or plain JSON — decided at read time
        return "vector"
    return "other"


# ── Folium embedding ──────────────────────────────────────────────────────────

def show_map(m, height: int = 460) -> None:
    """Embed a folium map into the current Streamlit container.

    Renders the map to a standalone HTML document and hands it to
    ``st.iframe`` via a ``data:`` URL (the supported replacement for the
    deprecated ``st.components.v1.html``).
    """
    doc = m.get_root().render()
    src = "data:text/html;base64," + base64.b64encode(doc.encode()).decode()
    st.iframe(src, height=height, width="stretch")


def _base_map():
    import folium
    return folium.Map(location=[10, 0], zoom_start=2, tiles="CartoDB positron")


# ── Vector ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _read_vector(path: str):
    import geopandas as gpd
    return gpd.read_file(path)


def vector_to_map(gdf, name: str = "layer"):
    """Build a folium map for a GeoDataFrame (reprojected to WGS84)."""
    import folium

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return None, 0
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf.to_crs(4326)

    total = len(gdf)
    if total > MAX_FEATURES:
        gdf = gdf.iloc[:MAX_FEATURES]

    # Stringify non-geometry columns so to_json() can't choke on dates/NaN.
    safe = gdf.copy()
    geom_col = safe.geometry.name
    fields = [c for c in safe.columns if c != geom_col]
    for c in fields:
        safe[c] = safe[c].astype(str)

    minx, miny, maxx, maxy = safe.total_bounds
    m = _base_map()
    tooltip = folium.GeoJsonTooltip(fields=fields[:6]) if fields else None
    folium.GeoJson(
        json.loads(safe.to_json()),
        name=name,
        tooltip=tooltip,
        marker=folium.CircleMarker(radius=4, fill=True, fill_opacity=0.7),
    ).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m, total


def preview_vector(path: Path, key: str) -> None:
    try:
        gdf = _read_vector(str(path))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read as vector: {e}")
        return
    crs = gdf.crs.to_string() if gdf.crs else "unknown"
    geom_types = ", ".join(sorted(gdf.geom_type.dropna().unique())) or "—"
    st.caption(f"**{len(gdf)}** features · CRS `{crs}` · geometry: {geom_types}")
    m, drawn = vector_to_map(gdf, name=path.stem)
    if m is not None:
        if drawn > MAX_FEATURES:
            st.info(f"Showing first {MAX_FEATURES} of {drawn} features.")
        show_map(m)
    else:
        st.info("No drawable geometries.")
    with st.expander("Attribute table (first 100 rows)"):
        st.dataframe(gdf.drop(columns=gdf.geometry.name).head(100),
                     width="stretch")


# ── Raster ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _raster_preview(path: str, max_dim: int = 800):
    """Read raster metadata + a downsampled RGB/grayscale thumbnail (as PNG bytes)."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))
        arr = src.read(out_shape=(src.count, out_h, out_w),
                       resampling=Resampling.average).astype("float32")
        meta = {
            "size": f"{src.width} × {src.height}",
            "bands": src.count,
            "crs": src.crs.to_string() if src.crs else "unknown",
            "res": f"{abs(src.res[0]):.4g}, {abs(src.res[1]):.4g}",
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }
        bounds84 = None
        try:
            if src.crs:
                bounds84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception:  # noqa: BLE001
            bounds84 = None

    # Build a visual: RGB if >=3 bands, else colormapped single band.
    def _stretch(a):
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros_like(a)
        lo, hi = np.nanpercentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1
        return np.clip((a - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6 * out_h / out_w))
    ax.axis("off")
    if arr.shape[0] >= 3:
        rgb = np.dstack([_stretch(arr[i]) for i in range(3)])
        ax.imshow(rgb)
    else:
        ax.imshow(_stretch(arr[0]), cmap="viridis")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=110)
    plt.close(fig)
    return meta, buf.getvalue(), bounds84


def preview_raster(path: Path, key: str) -> None:
    try:
        meta, png, bounds84 = _raster_preview(str(path))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read as raster: {e}")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(png, caption=path.name, width="stretch")
    with c2:
        st.write({k: str(v) for k, v in meta.items()})
    if bounds84:
        import folium
        w, s, e, n = bounds84
        m = _base_map()
        folium.Rectangle([[s, w], [n, e]], color="#e6194B", weight=2,
                         fill=True, fill_opacity=0.08,
                         tooltip="raster footprint").add_to(m)
        m.fit_bounds([[s, w], [n, e]])
        st.caption("Footprint (WGS84)")
        show_map(m, height=360)


# ── Tabular ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _read_table(path: str, nrows: int = 500):
    import pandas as pd
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def preview_table(path: Path, key: str) -> None:
    try:
        df = _read_table(str(path))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read as table: {e}")
        return
    st.caption(f"Showing up to {len(df)} rows · {df.shape[1]} columns")
    st.dataframe(df, width="stretch")
    # If it looks like point data, offer a quick map.
    lat = next((c for c in df.columns if c.lower() in {"lat", "latitude", "y"}), None)
    lon = next((c for c in df.columns if c.lower() in {"lon", "lng", "longitude", "x"}), None)
    if lat and lon:
        import pandas as pd
        pts = df[[lat, lon]].apply(pd.to_numeric, errors="coerce").dropna()
        if not pts.empty:
            import folium
            m = _base_map()
            for _, r in pts.head(MAX_FEATURES).iterrows():
                folium.CircleMarker([r[lat], r[lon]], radius=3,
                                    fill=True, fill_opacity=0.7).add_to(m)
            m.fit_bounds([[pts[lat].min(), pts[lon].min()],
                          [pts[lat].max(), pts[lon].max()]])
            st.caption(f"Point map from `{lat}`/`{lon}`")
            show_map(m)


# ── Zip (best effort) ─────────────────────────────────────────────────────────

def preview_zip(path: Path, key: str) -> None:
    try:
        names = zipfile.ZipFile(path).namelist()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not open zip: {e}")
        return
    st.caption(f"{len(names)} entries in archive")
    with st.expander("Archive contents"):
        st.code("\n".join(names[:200]), language="text")
    shp = next((n for n in names if n.lower().endswith(".shp")), None)
    csv = next((n for n in names if n.lower().endswith(".csv")), None)
    if shp:  # zipped shapefile — geopandas reads it directly
        try:
            gdf = _read_vector(f"zip://{path}!{shp}")
            st.caption(f"Shapefile `{shp}` — {len(gdf)} features")
            m, drawn = vector_to_map(gdf, name=Path(shp).stem)
            if m is not None:
                show_map(m)
        except Exception as e:  # noqa: BLE001
            st.info(f"Found shapefile but could not render: {e}")
    elif csv:
        try:
            import pandas as pd
            with zipfile.ZipFile(path) as z, z.open(csv) as f:
                df = pd.read_csv(f, nrows=500, on_bad_lines="skip")
            st.caption(f"CSV `{csv}` (first {len(df)} rows)")
            st.dataframe(df, width="stretch")
        except Exception as e:  # noqa: BLE001
            st.info(f"Found CSV but could not render: {e}")


def preview_file(path: Path, key: str) -> None:
    """Dispatch to the right previewer based on file kind."""
    if not path.exists():
        st.warning(f"File not found: {path}")
        return
    kind = file_kind(path)
    if kind == "vector":
        preview_vector(path, key)
    elif kind == "raster":
        preview_raster(path, key)
    elif kind == "table":
        preview_table(path, key)
    elif kind == "zip":
        preview_zip(path, key)
    else:
        st.info(f"No preview for `{path.suffix}` files.")


# ── DAG rendering ──────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "vector": "#4CAF50", "raster": "#2196F3", "multidim": "#9C27B0",
    "tabular": "#FF9800", "preprocessing": "#607D8B", "spatial": "#E91E63",
    "temporal": "#00BCD4", "statistical": "#795548", "visualization": "#3F51B5",
    "compute": "#F44336",
}
_DATA_TYPES = {"vector", "raster", "multidim", "tabular"}


def _esc(s: str) -> str:
    return str(s).replace('"', '\\"').replace("\n", "\\n")


def dag_to_dot(dag: dict) -> str:
    """Convert a geosolver DAG dict ({nodes, edges}) to Graphviz DOT."""
    lines = [
        "digraph DAG {",
        '  rankdir=LR;',
        '  node [style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#888888"];',
    ]
    for node in dag.get("nodes", []):
        nid = _esc(node.get("id", "?"))
        ntype = node.get("type", "compute")
        color = _TYPE_COLORS.get(ntype, "#9E9E9E")
        shape = "ellipse" if ntype in _DATA_TYPES else "box"
        label = _esc(node.get("name", nid))
        tag = node.get("tag")
        sub = ntype + (f" · {tag}" if tag else "")
        lines.append(
            f'  "{nid}" [label="{label}\\n({_esc(sub)})", '
            f'shape={shape}, fillcolor="{color}", fontcolor="white"];'
        )
    for edge in dag.get("edges", []):
        src, tgt = _esc(edge.get("source", "")), _esc(edge.get("target", ""))
        role = edge.get("role")
        lbl = f' [label="{_esc(role)}"]' if role else ""
        lines.append(f'  "{src}" -> "{tgt}"{lbl};')
    lines.append("}")
    return "\n".join(lines)


def render_dag(dag: dict) -> None:
    """Render a DAG dict as an interactive Graphviz chart with details."""
    nodes = dag.get("nodes", [])
    edges = dag.get("edges", [])
    if not nodes:
        st.info("DAG is empty.")
        return
    st.caption(f"{len(nodes)} nodes · {len(edges)} edges")
    st.graphviz_chart(dag_to_dot(dag), width="stretch")

    # Legend
    present = sorted({n.get("type", "compute") for n in nodes})
    st.markdown(" ".join(
        f"<span style='background:{_TYPE_COLORS.get(t, '#9E9E9E')};color:white;"
        f"padding:2px 8px;border-radius:8px;margin-right:4px;font-size:12px'>{t}</span>"
        for t in present), unsafe_allow_html=True)

    with st.expander("Node details & contracts"):
        for n in nodes:
            head = f"**{n.get('id')}** · `{n.get('type')}`"
            if n.get("tag"):
                head += f" · `{n['tag']}`"
            st.markdown(head + f" — {n.get('name', '')}")
            if n.get("function_description"):
                st.caption(n["function_description"])
            contracts = n.get("contracts") or {}
            for phase in ("pre", "post"):
                for a in contracts.get(phase, []) or []:
                    st.code(f"[{phase}] {a}", language="python")
