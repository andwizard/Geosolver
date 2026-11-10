# GeoAgent Service

A FastAPI service that runs a **multi-agent geospatial analysis pipeline** built on
LangGraph. Given a task (natural-language description + dataset files + expected
output), it plans, writes and executes Python code, produces maps/reports, and
returns the generated code and images.

Within the parent **GeoSolver** project this service is the backend for the
`--method geoagent` runner: `Agents/build_geoagent.py` is a thin HTTP client that
submits tasks to this service at `config.GEOAGENT_BASE_URL` (default
`http://127.0.0.1:8002`).

> **Note on modifications.** The upstream source could not be run as-is in this
> project and did not fit our evaluation setup, so the code here has been
> **modified as appropriate** from the original (e.g. removing hard-coded paths and
> secrets, making the dataset root configurable, and adapting the service for use
> with GeoSolver). The original, unmodified source is archived at
> <https://doi.org/10.6084/m9.figshare.29145008> and is licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This adapted
> version is distributed under the same license; attribution and modification
> details are in the root [`NOTICE`](../NOTICE).

---

## Configuration

Copy the template and fill it in (the real `config.ini` is git-ignored because it
holds your API key):

```bash
cp config_template.ini config.ini
```

`config.ini` sections:

```ini
[Langsmith]
LANGCHAIN_TRACING_V2 = false          # set true + fill keys to enable LangSmith tracing

[LLM]
OPENAI_BASE_URL = https://.../v1      # OpenAI-compatible endpoint
API_KEY = <your key>
PLAN_LLM = <model>                    # per-role model names (Planner, Coder, reviewers, ...)
...

[Tool]
WORKING_DIRECTORY = data_storage      # default scratch dir (overridden per task by the API)
CONDA_PATH = /path/to/conda           # only used on Windows; kept non-empty to satisfy imports
CONDA_ENV  = geoagent
```

### Dataset location

`pipline.py` resolves each task's data files under a dataset root chosen as:

1. the `GEOAGENT_DATA_ROOT` environment variable if set, otherwise
2. the parent project's `Data/` directory (`../Data` relative to this folder).

So by default it reads datasets straight from the GeoSolver `Data/<source>/` tree.

---

## Running

```bash
# from this directory, using the geoagent environment
python main.py
# or, without the auto-reloader:
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```

Concurrency is controlled by `GEOAGENT_MAX_WORKERS` (process-pool size; each task
runs in its own subprocess so per-task environment/working-dir never collide):

```bash
GEOAGENT_MAX_WORKERS=4 python main.py
```

`restart.sh` wraps stop → start → health-check for convenience.

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/run` | Run a task synchronously, return the result |
| POST | `/api/v1/run/async` | Submit a task, run in background |
| GET  | `/api/v1/status/{task_id}` | Poll an async task's status/result |
| GET  | `/health` | Health check → `{"status":"ok"}` |
| GET  | `/docs` | Swagger UI |

Request body (`AgentTaskRequest`):

```json
{
  "task": { "task_ID": "...", "task_text": "...", "used_dataset": ["..."], "source": "GeoBenchX", "...": "..." },
  "output_path": "./service_outputs/<task>",
  "time_limit": 600
}
```

---

## Use from the parent GeoSolver project

1. Start this service (above).
2. Point the client at it — `config.ini` in the GeoSolver root:
   ```ini
   [GeoAgent]
   base_url = http://127.0.0.1:8002
   ```
3. Run tasks through the standard runner:
   ```bash
   python run_agent.py --method geoagent --task_path Data/datasets.json
   ```

---

## Notes

- **Recursion limit**: the LangGraph run uses `recursion_limit = 20`
  (`geo_agent/graph.py`), and the code-debug sub-agent has its own cap. On hard
  tasks the review/debug loop can hit this limit before returning a clean terminal
  status even though intermediate outputs were produced. Raise the limit in
  `graph.py` if you need longer loops.
- This service was ported from a standalone project and **modified as appropriate**
  to run here (see the note at the top). Only source code was brought over; its
  original experiment datasets are **not** included, so the `eval_*.py` scripts
  require you to supply their data before they will run. The original unmodified
  source is at <https://doi.org/10.6084/m9.figshare.29145008>.

---

## License

The upstream GeoAgent source is licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
This adapted version is distributed under the same license. The original work is:

> *GeoAgent: a hierarchical LLM-based multi-agent architecture for autonomous
> spatial analysis* — Lin, archived at
> <https://doi.org/10.6084/m9.figshare.29145008>.

Changes for GeoSolver include removing hard-coded paths and secrets, making the
dataset root configurable, and adapting the service to the parent project's
evaluation setup. See the root [`NOTICE`](../NOTICE) for the full attribution.

The rest of the GeoSolver repository is licensed separately under Apache-2.0.
