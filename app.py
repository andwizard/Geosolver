"""GeoSolver – simple Streamlit frontend.

Type a natural-language geospatial task (or load one from the benchmark),
pick an agent method, run it, and inspect the generated code / answer /
visualization — all from the browser.

    streamlit run app.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
import viz

# ── Constants ────────────────────────────────────────────────────────────────

METHODS = [
    "CodeAct",
    "Geosolver_without_DFG",
    "geosolver",
    "geosolver_without_contract",
    "deepagent",
    "geoagent",
    "claude_code",
]

DATA_ROOT = Path(config.DATA_ROOT)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

# Where UI-triggered runs are written.
APP_RUNS_DIR = Path(config.RUNS_DIR) / "app"


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_sources() -> list[str]:
    """Data source folders directly under Data/ (each maps to a task 'source')."""
    if not DATA_ROOT.exists():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


@st.cache_data(show_spinner=False)
def list_files(source: str) -> list[str]:
    """Data file names inside a given source folder (non-recursive)."""
    src = DATA_ROOT / source
    if not src.exists():
        return []
    return sorted(p.name for p in src.iterdir() if p.is_file())


@st.cache_data(show_spinner=False)
def load_benchmark() -> list[dict]:
    """All benchmark tasks from Data/datasets.json."""
    path = DATA_ROOT / "datasets.json"
    if not path.exists():
        return []
    return json.load(open(path, encoding="utf-8")).get("tasks", [])


def stream_task(method: str, task: dict, timeout: int, retries: int):
    """Run one task via the run_agent CLI in a subprocess, streaming its output.

    Yields ("line", str) for each console line as the agent produces it, then a
    final ("done", {"solution": dict, "folder": str}). Zero changes to the agent
    internals: we reuse the existing CLI, which already tees the full trajectory
    to stdout and writes solution.py / dag.json / outputs into the task folder.
    """
    # Unique output dir per run so the CLI's resume-checkpoint never skips a
    # repeated (benchmark) task, and runs don't clobber each other.
    run_dir = APP_RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    task_folder = run_dir / task["task_ID"]

    # A self-contained single-task dataset for the CLI.
    task_file = run_dir / "_task.json"
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "tasks": [task]}, f, ensure_ascii=False)

    cmd = [
        sys.executable, "-u", "run_agent.py",   # -u => unbuffered => smooth stream
        "--method", method,
        "--task_path", str(task_file),
        "--output_path", str(run_dir),
        "--timeout", str(timeout),
        "--retry", str(retries),
        "--workers", "1",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(Path(__file__).parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:  # blocks per line as the subprocess emits them
        yield "line", line.rstrip("\n")
    proc.wait()

    yield "done", {"solution": _read_solution(run_dir, task, proc.returncode),
                   "folder": str(task_folder)}


def _read_solution(run_dir: Path, task: dict, returncode: int) -> dict:
    """Recover the generated_solution the CLI wrote, or synthesize an error."""
    result_file = run_dir / "generated_solutions.json"
    if result_file.exists():
        try:
            data = json.load(open(result_file, encoding="utf-8"))
            for t in data.get("tasks", []):
                if t.get("task_ID") == task["task_ID"]:
                    return t.get("generated_solution", {})
        except Exception:  # noqa: BLE001
            pass
    return {"response": "FAILED",
            "answer": f"No result produced (subprocess exit code {returncode})."}


def input_paths(task: dict) -> list[Path]:
    """Absolute paths of a task's declared input files under Data/<source>/."""
    return [DATA_ROOT / task["source"] / f for f in task.get("used_dataset", [])]


def find_outputs(task_folder: Path) -> tuple[list[Path], list[Path]]:
    """Return (images, other files) generated anywhere under the task folder."""
    images, others = [], []
    for p in sorted(task_folder.rglob("*")):
        if not p.is_file() or p.name in {"solution.py", "trajectory.txt", "task.log"}:
            continue
        (images if p.suffix.lower() in IMAGE_EXTS else others).append(p)
    return images, others


# ── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GeoSolver", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoSolver")
st.caption("Run a geospatial-analysis agent on a natural-language task.")

# --- Sidebar: run settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.selectbox("Agent method", METHODS, index=0)
    timeout = st.number_input("Per-task timeout (s)", 30, 3600,
                              min(600, config.EVAL_TIMEOUT_SECONDS), step=30)
    retries = st.number_input("Retries on failure", 0, 5, 1)
    st.divider()
    st.write(f"**Model:** `{config.MODEL_NAME}`")
    st.write(f"**Data root:** `{DATA_ROOT}`")
    if method == "geoagent":
        st.info("`geoagent` needs the GeoAgent service running "
                f"({config.GEOAGENT_BASE_URL}).")

# --- Task definition ---
sources = list_sources()
if not sources:
    st.error(f"No data source folders found under {DATA_ROOT}.")
    st.stop()

mode = st.radio("Task source", ["✍️ Custom task", "📚 Benchmark task"],
                horizontal=True)

task: dict | None = None

if mode == "✍️ Custom task":
    task_text = st.text_area(
        "Task (natural language)",
        placeholder="e.g. Which Eastern African countries had the highest net migration in 2019?",
        height=110,
    )
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Data source folder", sources)
    with col2:
        files = st.multiselect("Input files", list_files(source),
                               help="Files under Data/<source>/ the agent may use.")

    with st.expander("Expected output (optional)"):
        c1, c2, c3 = st.columns(3)
        number_output = c1.text_input("Number output name", "")
        visual_output = c2.text_input("Visual output name", "")
        data_output = c3.text_input("Data output name", "")

    if task_text.strip():
        task = {
            "task_ID": datetime.now().strftime("APP_%Y%m%d_%H%M%S"),
            "task_text": task_text.strip(),
            "source": source,
            "used_dataset": files,
            "Expected_Output": {
                "number_output": number_output,
                "visual_output": visual_output,
                "data_output": data_output,
            },
        }

else:  # Benchmark task
    tasks = load_benchmark()
    if not tasks:
        st.error("Data/datasets.json not found or empty.")
        st.stop()
    src_filter = st.selectbox(
        "Filter by source", ["(all)"] + sorted({t["source"] for t in tasks})
    )
    pool = [t for t in tasks if src_filter == "(all)" or t["source"] == src_filter]
    labels = [f"[{t.get('difficulty', '?')}] {t['task_text'][:90]}" for t in pool]
    idx = st.selectbox("Task", range(len(pool)),
                       format_func=lambda i: labels[i]) if pool else None
    if idx is not None:
        task = pool[idx]
        st.text_area("Task text", task["task_text"], height=90, disabled=True)
        c1, c2 = st.columns(2)
        c1.write(f"**Source:** {task['source']}")
        c1.write(f"**Files:** {', '.join(task.get('used_dataset', [])) or '—'}")
        c2.write(f"**Expected:** `{json.dumps(task.get('Expected_Output', {}), ensure_ascii=False)}`")
        if task.get("reference_answer"):
            c2.write(f"**Reference answer:** {task['reference_answer']}")

# --- Input data preview ---
if task is not None:
    paths = input_paths(task)
    if paths:
        with st.expander("🔍 Preview input data", expanded=False):
            names = [p.name for p in paths]
            sel = st.selectbox("File", names, key="preview_select")
            viz.preview_file(DATA_ROOT / task["source"] / sel, key=f"in_{sel}")

# --- Run ---
run = st.button("▶️ Run", type="primary", disabled=task is None)

if run and task is not None:
    st.subheader("🔴 Live agent output")
    log_box = st.empty()
    lines: list[str] = []
    result = None
    try:
        with st.status(f"Running `{method}` …", expanded=True) as status_box:
            for kind, payload in stream_task(method, task, int(timeout), int(retries)):
                if kind == "line":
                    lines.append(payload)
                    # Render a scrolling tail; cap for responsiveness.
                    log_box.code("\n".join(lines[-400:]), language="text")
                else:  # done
                    result = payload
            ok = result and result["solution"].get("response") not in ("FAILED", "TIMEOUT")
            status_box.update(
                label="✅ Finished" if ok else "⚠️ Finished with errors",
                state="complete" if ok else "error", expanded=False,
            )
    except Exception as e:  # noqa: BLE001 – surface any launch/stream error
        st.error(f"Run failed: {e}")
        st.stop()

    if result is not None:
        st.session_state["last"] = {
            "solution": result["solution"], "folder": result["folder"],
            "method": method, "full_log": "\n".join(lines),
        }

# --- Results ---
last = st.session_state.get("last")
if last:
    st.divider()
    st.subheader("Results")
    solution = last["solution"]
    task_folder = Path(last["folder"])

    status = solution.get("response", "?")
    (st.success if status not in ("FAILED", "TIMEOUT") else st.error)(f"Status: {status}")
    st.markdown("**Answer**")
    st.code(str(solution.get("answer", "")) or "(empty)", language="text")

    images, others = find_outputs(task_folder)
    geo_outputs = [f for f in others
                   if viz.file_kind(f) in {"vector", "raster", "table", "zip"}]
    tabs = st.tabs(["🖼️ Visualizations", "🗺️ Map", "🔀 DAG",
                    "📄 Generated code", "📁 Files", "📜 Log"])

    with tabs[0]:
        if images:
            for img in images:
                st.image(str(img), caption=img.name, width="stretch")
        else:
            st.info("No image outputs.")

    with tabs[1]:
        if geo_outputs:
            names = [str(f.relative_to(task_folder)) for f in geo_outputs]
            pick = st.selectbox("Output file", names, key="out_map_select")
            viz.preview_file(task_folder / pick, key=f"out_{pick}")
        else:
            st.info("No geospatial/data outputs to map.")

    with tabs[2]:
        # geosolver's DAG tools auto-save dag.json / dag.png into the task
        # folder (their cwd during the run).
        dag_json = next(iter(task_folder.rglob("dag.json")), None)
        dag_png = next(iter(task_folder.rglob("dag.png")), None)
        if dag_json is not None:
            viz.render_dag(json.loads(dag_json.read_text(encoding="utf-8")))
        elif dag_png is not None:
            st.image(str(dag_png), width="stretch")
        else:
            st.info("No DAG for this run (only `geosolver` methods produce one).")

    with tabs[3]:
        sol_py = task_folder / "solution.py"
        st.code(sol_py.read_text(encoding="utf-8") if sol_py.exists()
                else "(no solution.py)", language="python")

    with tabs[4]:
        if others:
            for f in others:
                st.download_button(f"⬇️ {f.relative_to(task_folder)}",
                                   data=f.read_bytes(), file_name=f.name,
                                   key=str(f))
        else:
            st.info("No data-file outputs.")

    with tabs[5]:
        log = task_folder / "task.log"
        if log.exists():
            text = log.read_text(encoding="utf-8")
        else:  # fall back to the streamed console output
            text = last.get("full_log", "(no log)")
        st.code(text[-20000:], language="text")

    st.caption(f"Output folder: `{task_folder}`")
