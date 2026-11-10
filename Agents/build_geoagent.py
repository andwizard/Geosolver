import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import time


def build_geoagent(base_url):
    """Build a geoagent config dict.

    Args:
        base_url: Base URL for the geoagent API.

    Returns:
        A config dict with the base_url.
    """
    return {"base_url": base_url}


def geoagent_invoke(agent, task, data_path, output_path, time_limit=600):

    base_url = agent["base_url"]

    payload = {
        "task": task,
        "output_path": output_path,
        "time_limit": time_limit
    }

    # 1. Submit the task
    print(f"--- Submitting experiment task: {task['task_ID']} ---")
    print(f"--- Output path: {task['task_ID']} ---")
    resp = requests.post(f"{base_url}/api/v1/run/async", json=payload, timeout=60)
    resp.raise_for_status()
    print("Task submitted successfully; the service has started computing in the background.")

    # 2. Poll for results
    # Client-side hard deadline: the server is supposed to enforce ``time_limit``
    # itself, but if a job hangs server-side it never returns a terminal status
    # and this loop would otherwise poll forever.  Give the server a grace margin
    # past its own limit, then give up locally and report TIMEOUT.
    deadline = time.time() + time_limit + 180
    print("Waiting for results (checking every 60 seconds)...")
    while True:
        try:
            status_resp = requests.get(f"{base_url}/api/v1/status/{task['task_ID']}", timeout=60)
            data = status_resp.json()
        except requests.RequestException as e:
            # Transient network/server hiccup — log and keep polling until the
            # deadline rather than crashing the whole task.
            print(f"\nError polling status, will retry later: {e}")
            data = {}

        status = data.get('status', "")
        if status == "ACCEPT":
            print("\n✅ Experiment completed successfully!")
            print("Result data:", data["answer"])

            return data

        elif status == "FAILED":
            print("\n❌ Experiment execution error:")
            print(data["error"])

            return data
        elif status == "TIMEOUT":
            print("\n❌ Experiment execution timed out:")
            print(data["error"])

            return data
        elif time.time() >= deadline:
            # Server never returned a terminal status within the grace window.
            print(f"\n❌ Client wait timed out (server did not return results within {time_limit + 180}s)")
            return {"status": "TIMEOUT",
                    "error": f"client-side polling deadline exceeded ({time_limit + 180}s)"}
        else:
            # Still running; print a dot as a heartbeat
            print(".", end="", flush=True)
            time.sleep(60)

if __name__ == "__main__":
    import configparser
    from pathlib import Path
    _cfg = configparser.ConfigParser()
    _cfg.read(Path(__file__).parent.parent / "config.ini")
    base_url = _cfg.get("GeoAgent", "base_url", fallback="http://127.0.0.1:8002")

    task = {
        "task_ID": "TASK_250309_135125_811107",
        "task_cls": "Thematic Geographic Analysis",
        "task_text": "Calculate the total railway length in Brazilian municipalities where population density exceeds 50 people per km², and analyze the correlation between railway accessibility and population density quartiles.",
        "used_dataset": [
            "hotosm_bra_railways_lines_shp.shp",
            "bra_ppp_2018_1km_Aggregated_UNadj.tif",
            "bra_admbnda_adm2_ibge_2020.shp"
        ],
        "Expected_Output": {
            "number_output": "Pearson correlation coefficient",
            "visual_output": "",
            "data_output": "analysis_result.geojson, quartile_stats.csv, correlation.txt"
        },
        "source": "GeoBenchX",
        "reference_answer": "Pearson correlation coefficient: 0.2006"
    }

    agent = build_geoagent(base_url)
    geoagent_invoke(agent, task, data_path=str(Path(__file__).parent.parent / "Data"), output_path="../Output/runs/test")