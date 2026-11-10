import asyncio
import os
import json
import logging
import multiprocessing as mp
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from fastapi.openapi.docs import get_swagger_ui_html

# 核心逻辑 run_agent_to_json 在子进程 worker 内按需 import（见 _run_task_worker），
# 父进程不再于模块顶层加载重依赖。

# ================= 配置与日志 =================
logger = logging.getLogger("GeoAgentService")

# ================= 数据模型 (Schema) =================
# 定义 API 输入的 JSON 结构
class AgentTaskRequest(BaseModel):
    task: Dict[str, Any] = Field(..., description="包含 task_ID, used_dataset 等的完整任务字典")
    output_path: Optional[str] = Field(default="./service_outputs", description="结果存储根目录")
    time_limit: int = Field(default=300, description="时间限制")

# 定义 API 输出的结构
class AgentTaskResponse(BaseModel):
    status: str
    task_id: str
    answer: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================= 进程级任务隔离 =================
# run_agent_to_json 通过 os.environ 传递 WORKING_DIRECTORY / SESSION_ID 等 per-task 上下文，
# 而 os.environ 是进程全局的。若在单进程内并发多个任务，后启动的任务会覆盖前者的环境变量，
# 造成写错目录 / 日志串号。这里把每个任务放进独立子进程执行：os.environ、模块级 logger、
# graph 状态天然按进程隔离，工具代码无需任何改动即可安全并发。
# 并发度 = 进程池大小，可用环境变量 GEOAGENT_MAX_WORKERS 调整。
executor: Optional[ProcessPoolExecutor] = None


def _run_task_worker(task: Dict[str, Any], output_path: str, time_limit: int) -> str:
    """在独立子进程中运行单个 agent 任务。

    子进程拥有独立的 os.environ / 模块状态，因此并发任务之间不会互相污染
    WORKING_DIRECTORY / SESSION_ID。返回 run_agent_to_json 的结果字符串。
    超时由 asyncio.wait_for 在子进程内强制执行（抛 asyncio.TimeoutError）。
    """
    # 延迟到子进程内 import，避免父进程 spawn 时的重复开销
    from pipline import run_agent_to_json as _run

    async def _run_with_timeout():
        return await asyncio.wait_for(_run(task, output_path), timeout=time_limit)

    return asyncio.run(_run_with_timeout())


# ================= 服务生命周期 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global executor
    max_workers = int(os.environ.get("GEOAGENT_MAX_WORKERS", "4"))
    # 用 spawn 上下文：子进程是干净解释器，避免 fork 已运行的 asyncio 事件循环 / 线程导致的死锁
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    logger.info(f"Geo Agent Service 正在启动... 进程池 workers={max_workers}")
    yield
    # 关闭时执行
    executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Geo Agent Service 已关闭。")

# ================= 初始化 FastAPI 应用 =================
# 1. 先把 app 初始化出来
app = FastAPI(
    title="Geo Agent API",
    description="地理空间分析 Agent 服务接口",
    version="1.0.0",
    lifespan=lifespan,
    # 先把默认 docs 关掉，我们自己挂载
    docs_url=None,
    redoc_url=None
)

# 2. 自己定义 /docs，用国内 CDN
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Geo Agent API",
        # 国内可用 CDN（静态文件）
        swagger_js_url="https://cdn.staticfile.org/swagger-ui/5.11.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.staticfile.org/swagger-ui/5.11.0/swagger-ui.css",
    )

# ================= API 端点 =================

@app.post("/api/v1/run", response_model=AgentTaskResponse)
async def run_task(request: AgentTaskRequest):
    """
    同步执行 Agent 任务。任务在独立子进程中运行，多个请求可安全并发。
    """
    task_id = request.task.get("task_ID", "unknown_task")

    try:
        # 1. 为该任务创建独立的输出目录
        task_output_path = Path(request.output_path) / task_id
        task_output_path.mkdir(parents=True, exist_ok=True)

        # 2. 调用核心逻辑（在进程池中执行，不阻塞事件循环、不共享 os.environ）
        logger.info(f"开始处理任务: {task_id}")
        loop = asyncio.get_running_loop()
        answer_json_str = await loop.run_in_executor(
            executor, _run_task_worker, request.task, str(task_output_path), request.time_limit
        )

        # 3. 解析结果并返回
        answer_data = json.loads(answer_json_str)

        return AgentTaskResponse(
            status="ACCEPT",
            task_id=task_id,
            answer=answer_data
        )

    except asyncio.TimeoutError:
        logger.error(f"任务超时 {task_id} (limit={request.time_limit}s)")
        raise HTTPException(status_code=504, detail=f"任务超时 ({request.time_limit}s)")
    except Exception as e:
        logger.error(f"任务执行失败 {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

tasks_repo = {}

@app.post("/api/v1/run/async")
async def run_task_async(request: AgentTaskRequest, background_tasks: BackgroundTasks):
    """
    后台方式执行。任务在独立子进程中运行，多个请求可安全并发。
    """
    task_id = request.task.get("task_ID", "exp_task_default")
    task_output_path = Path(request.output_path)
    time_limit = request.time_limit

    task_output_path.mkdir(parents=True, exist_ok=True)

    # 记录初始状态
    tasks_repo[task_id] = {"status": "PROCESSING", "answer": None, "error": None}

    async def runner():
        loop = asyncio.get_running_loop()
        try:
            logger.info(f"实验开始: {task_id}")
            answer = await loop.run_in_executor(
                executor, _run_task_worker, request.task, str(task_output_path), time_limit
            )
            tasks_repo[task_id] = {
                "status": "ACCEPT",
                "answer": answer,
                "error": None
            }
            logger.info(f"实验完成: {task_id}")
        except asyncio.TimeoutError as te:
            tasks_repo[task_id] = {"status": "TIMEOUT", "answer": None, "error": str(te)}
        except Exception as e:
            tasks_repo[task_id] = {"status": "FAILED", "answer": None, "error": str(e)}

    background_tasks.add_task(runner)
    return {"task_id": task_id, "status": "ACCEPT"}

@app.get("/api/v1/status/{task_id}")
async def get_task_status(task_id: str):
    # 返回该任务的最新进度
    if task_id not in tasks_repo:
        raise HTTPException(status_code=404, detail="任务 ID 不存在")
    return tasks_repo[task_id]

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "geo_agent"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，监听 0.0.0.0:8002
    # 注意：reload=True 为开发模式；生产建议关闭 reload。
    # 并发由进程池承担，单 uvicorn worker 即可，不要再叠加 --workers（会导致 tasks_repo 不一致）。
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
