import configparser
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List
import uuid
import time
import json
import re
import base64
import asyncio  # 新增 asyncio 导入
from dotenv import load_dotenv

# from langsmith import Client
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from utils.utils import copy_files
from utils.logger import get_logger_with_session
from langgraph.managed import IsLastStep

# setup logger
logger = get_logger_with_session()

load_dotenv()


def setup_environment(task_num: str, output_path) -> Dict[str, str]:
    config = configparser.ConfigParser()
    config.read("config.ini")

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H_%M_%S")
    session_id = str(uuid.uuid4())[:8]
    working_dir = Path(output_path)
    working_dir.mkdir(parents=True, exist_ok=True)
    env_vars = {
        "LANGCHAIN_TRACING_V2": config.get("Langsmith", "LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_ENDPOINT": config.get("Langsmith", "LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": config.get("Langsmith", "LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": config.get("Langsmith", "LANGCHAIN_PROJECT"),

        "SESSION_ID": f"{date_str}_{time_str}_{task_num}",

        "WORKING_DIRECTORY": str(working_dir),
        "CONDA_PATH": config.get("Tool", "CONDA_PATH"),
        "CONDA_ENV": config.get("Tool", "CONDA_ENV"),

        "OPENAI_BASE_URL": config.get("LLM", "OPENAI_BASE_URL"),
        "API_KEY": config.get("LLM", "API_KEY"),
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    return env_vars


async def run_agent(task, data_paths: List[str]):
    """
    Args:
        task_num: graph 的 thread_id
        task: 任务描述
        data_paths: 数据集路径，格式为 [数据集来源]/x  （x为数字）
    """
    # setup environment
    task_id = task["task_ID"]

    # create_graph after setup_environment()
    from geo_agent.graph import create_graph

    graph, graph_config = create_graph(task_id)

    logger.info(f"-------------------- Graph created with thread_id {task_id} --------------------")

    message = """
    TASK:
    {task_text}

    REQUIRED DATA:
    {file_list}

    EXPECTED OUTPUT:
    {expected_output}


    INSTRUCTIONS:
    1. Please call the send_output tool to return the result after you complete all tasks.
    2. If you cannot produce a valid script, set "task_status" to "FAILED". Otherwise set "task_status" to "ACCEPT".
    """.format(
        task_text=task['task_ID'],
        file_list=str(task['used_dataset']),
        expected_output=json.dumps(task['Expected_Output']),
    )

    target_directory = os.path.join(os.environ["WORKING_DIRECTORY"], "data")
    os.makedirs(target_directory, exist_ok=True)
    
    # 使用 to_thread 防止文件拷贝操作阻塞事件循环
    await asyncio.to_thread(copy_files, data_paths, target_directory)

    _input = {
        "messages": [
            HumanMessage(
                content=message
            ),
        ],
        "user_query": message,
        "user_guide": "",
        "task_category": "",
        "task_plan": "",
        "process_decision": "",
        "coder_state": "",
        "reporter_state": "",
        "cartographer_state": "",
        "code_reviewer_state": "",
        "map_reviewer_state": "",
        "report_reviewer_state": "",
        "needs_revision": False,
        "is_last_step": IsLastStep()
    }

    logger.info(f"-------------------- Task {task_id}: \n{message} --------------------")
    
    # 替换为异步 stream
    events = graph.astream(
        _input,
        graph_config,
        stream_mode="values",
        debug=False
    )

    message = HumanMessage(content="")

    # 替换为异步循环
    async for event in events:
        message = event["messages"][-1]
        if isinstance(message.content, list):
            content = "\n".join(str(item) for item in message.content)
        else:
            content = message.content
        logger.info(content)

    return message.content


# ==========================================
# 核心函数：运行 Agent 并返回指定 schema 的 JSON
# ==========================================
async def run_agent_to_json(task: dict, output_path: str) -> str:
    """
    执行 run_agent 并将生成的代码和图片结果打包为 JSON 字符串返回。
    返回的 JSON 结构:
    {
        "code": "[合并的 Python 代码]",
        "images": ["[图片的 base64 编码字符串1]", "[图片的 base64 编码字符串2]", ...]
    }
    """
    task_num = task["task_ID"]
    used_dataset = task["used_dataset"]
    source = task['source']

    logger.info(f"Executing run_agent_to_json for task_num: {task_num}")

    # Dataset root: env override, else the parent project's Data/ dir
    # (this file lives at <repo>/GeoAgent/pipline.py, so parent.parent = <repo>).
    data_root = os.environ.get(
        "GEOAGENT_DATA_ROOT",
        str(Path(__file__).resolve().parent.parent / "Data"),
    )
    base_dir = Path(data_root) / source

    # 用于存放最终结果的列表（或使用集合去重）
    data_paths_set = set()

    for filename in used_dataset:
        # 解析 prefix 和 format
        prefix, ext = os.path.splitext(filename)  # prefix="roads_1", ext=".shp"
        fmt = ext[1:] if ext.startswith('.') else ext  # "shp"

        # 在目标目录下匹配所有 prefix.fmt 文件
        for p in base_dir.glob(f"{prefix}.*"):
            data_paths_set.add(str(p))

    # 转回列表
    data_paths = list(data_paths_set)

    env_config = setup_environment(task['task_ID'], output_path)

    # 1. 异步等待 agent 运行结束
    final_message = await run_agent(task, data_paths)
    logger.info(f"Agent finished. Working dir: {output_path}")

    work_dir_path = Path(output_path)
    if not work_dir_path.exists() or not work_dir_path.is_dir():
        logger.error(f"Working directory not found: {output_path}")
        raise RuntimeError("Working directory not found after execution.")

    result_dict = final_message

    logger.info("Successfully assembled JSON response.")
    return final_message


async def main():
    # 本地测试样例
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
        "reference_solutions": [
            "import geopandas as gpd\nimport rasterio\nfrom rasterstats import zonal_stats\nimport pandas as pd\nimport numpy as np\nfrom scipy.stats import pearsonr\nimport os\n\n# 设置路径\nGeoBenchX_PATH = \"./Data/GeoBenchX\"\nSCRATCH_PATH = \"./workspace\"\nos.makedirs(SCRATCH_PATH, exist_ok=True)\n\n# 1. 加载行政区数据并设置/转换坐标系\nadm = gpd.read_file(f\"{GeoBenchX_PATH}/bra_admbnda_adm2_ibge_2020.shp\")\n# 该数据坐标系未定义，根据地理范围判断应为WGS84，先设置再转换\nadm = adm.set_crs(epsg=4326)\n\n# 2. 计算人口密度\nraster_path = f\"{GeoBenchX_PATH}/bra_ppp_2018_1km_Aggregated_UNadj.tif\"\nwith rasterio.open(raster_path) as src:\n    transform = src.transform\nstats = zonal_stats(\n    adm,\n    raster_path,\n    stats=\"mean\",\n    nodata=-99999.0,\n    affine=transform\n)\nadm['pop_density'] = [s['mean'] if s['mean'] is not None else 0 for s in stats]\n\n# 3. 筛选高人口密度区域\nhigh_density = adm[adm['pop_density'] > 50].copy()\nprint(f\"Number of high population density areas: {len(high_density)}\")\n\n# 4. 加载铁路数据\nrailways = gpd.read_file(f\"{GeoBenchX_PATH}/hotosm_bra_railways_lines_shp.shp\")\nrailways = railways.to_crs(epsg=4326)\n\n# 5. 空间连接并计算铁路长度\n# 将数据投影到适合巴西的投影坐标系以准确计算长度（如EPSG:5462）\nrailways_proj = railways.to_crs(epsg=5462)\nrailways_proj['length_km'] = railways_proj.geometry.length / 1000  # 米转公里\nhigh_density_proj = high_density.to_crs(epsg=5462)\njoined = gpd.overlay(railways_proj, high_density_proj, how='intersection')\n# joined = gpd.sjoin(high_density_proj, railways_proj, how=\"inner\", predicate=\"intersects\")\nrail_length = joined.groupby('ADM2_PCODE')['length_km'].sum().reset_index()\nrail_length.columns = ['ADM2_PCODE', 'railway_length_km']\n\n# 6. 合并数据\nmerged = high_density.merge(rail_length, on='ADM2_PCODE', how='left')\nmerged['railway_length_km'] = merged['railway_length_km'].fillna(0)\ntotal_railway_length_km = merged['railway_length_km'].sum()\nprint(f\"Total railway length in high population density areas: {total_railway_length_km:.2f} km\")\n\n# 7. 人口密度四分位分组\nmerged['pop_density_quartile'] = pd.qcut(merged['pop_density'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])\n\n# 8. 统计分析\nquartile_stats = merged.groupby('pop_density_quartile')['railway_length_km'].agg(['mean', 'median'])\ncorr, _ = pearsonr(merged['pop_density'], merged['railway_length_km'])\n\n# 保存结果\nmerged.to_file(f\"{SCRATCH_PATH}/analysis_result.geojson\", driver=\"GeoJSON\")\nquartile_stats.to_csv(f\"{SCRATCH_PATH}/quartile_stats.csv\")\nwith open(f\"{SCRATCH_PATH}/correlation.txt\", 'w') as f:\n    f.write(f\"Pearson correlation coefficient: {corr:.4f}\")\n\nprint(f\"Analysis complete. Results saved to {SCRATCH_PATH}\")\nprint(f\"Correlation coefficient: {corr:.4f}\")\nprint(quartile_stats)"
        ],
        "source": "GeoBenchX",
        "reference_answer": "Pearson correlation coefficient: 0.2006"
    }
    
    result_json = await run_agent_to_json(
        task,
        output_path="./test"
    )
    print(result_json)

if __name__ == "__main__":
    # 使用 asyncio 运行主异步函数
    asyncio.run(main())