import configparser
import json
import os
from pathlib import Path

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.resolve()

# Single source of config for the whole project: a root-level .env (secrets/API
# keys) plus config.ini (paths, model, timeouts). load_dotenv does not override
# already-set env vars.
load_dotenv(_BASE_DIR / ".env")

# Load config.ini
_cfg = configparser.ConfigParser()
_cfg.read(_BASE_DIR / "config.ini")

# Paths
DATA_ROOT = _BASE_DIR / _cfg.get("PATH", "DATA_PATH", fallback="./Data")
Output_PATH = _BASE_DIR / _cfg.get("PATH", "Output_PATH", fallback="./Output")

# Sub-dataset directories
GEO_BENCH_X_DIR = DATA_ROOT / "GeoBenchX"
GEO_ANALYST_BENCH_DIR = DATA_ROOT / "GeoAnalystBench"
GEO_COLABS_DIR = DATA_ROOT / "GeoColabs"
SANDBOX_SCRATCH = Output_PATH / "scratch"
RUNS_DIR = Output_PATH / "runs"
EVAL_OUTPUT_DIR = Output_PATH / "eval"
REF_SOLUTIONS_DIR = Output_PATH / "solutions"

# Model config
MODEL_NAME = _cfg.get("Model", "name", fallback="openai:qwen3.5-plus")
MODEL_TEMPERATURE = _cfg.getfloat("Model", "temperature", fallback=0.0)
# Per-request timeout (seconds) for LLM API calls, shared by all agent builders.
LLM_REQUEST_TIMEOUT = _cfg.getint("Model", "request_timeout", fallback=120)

# GeoAgent config
GEOAGENT_BASE_URL = _cfg.get("GeoAgent", "base_url", fallback="http://127.0.0.1:8000")

# DeepAgent config
DEEPAGENT_ROOT_DIR = _cfg.get("DeepAgent", "root_dir", fallback="./")
_env_dict_str = _cfg.get("DeepAgent", "env_dict", fallback="{}")
try:
    DEEPAGENT_ENV_DICT = json.loads(_env_dict_str)
except json.JSONDecodeError:
    DEEPAGENT_ENV_DICT = {}

# Eval config
EVAL_TIMEOUT_SECONDS = _cfg.getint("Eval", "timeout_seconds", fallback=1200)
EVAL_MAX_WORKERS = _cfg.getint("Eval", "max_workers", fallback=4)

# API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# OPENAI_API_BASE="https://api.deepseek.com"