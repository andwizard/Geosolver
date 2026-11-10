#!/usr/bin/env bash
# GeoAgent 服务一键重启脚本
# 流程: config校验 -> 停旧服务(含worker子进程) -> 启新服务 -> 健康检查 -> 端到端LLM验证
# 用法:
#   ./restart.sh          改完 config.ini 后敲一下, 完整重启+校验
#   ./restart.sh --fast    跳过端到端 LLM 调用(不花 token, 只保证服务起来)
#
# 设计要点:
#   - 第0步先校验 config.ini 能否解析, 挡住"乱码/格式错误"(之前踩过两次: base_url多尾巴、首行粘乱码)
#   - 停服务时连 ProcessPoolExecutor 的 worker 子进程一起清理, 避免孤儿进程残留
#   - 并发度沿用环境变量 GEOAGENT_MAX_WORKERS(默认 32)

set -u
PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ" || { echo "无法进入项目目录 $PROJ"; exit 1; }

# Python interpreter to run the service with. Override via GEOAGENT_PYTHON,
# e.g. the python from a conda env that has the geospatial stack installed.
PY="${GEOAGENT_PYTHON:-python}"
PORT=8002
WORKERS="${GEOAGENT_MAX_WORKERS:-32}"
LOG="$PROJ/logs/service_stdout.log"
CFG="$PROJ/config.ini"
SKIP_LLM=0
[ "${1:-}" = "--fast" ] && SKIP_LLM=1

echo "==================== GeoAgent 重启 ===================="

# ---- 0) config 解析校验 (gate) ----
echo "[0/4] 校验 config.ini ..."
$PY - "$CFG" <<'PYEOF' || { echo "  config 解析失败, 已中止 —— 请先修 config.ini"; exit 1; }
import configparser, sys
c = configparser.ConfigParser()
try:
    c.read(sys.argv[1])
    m = c.get("LLM", "PLAN_LLM"); b = c.get("LLM", "OPENAI_BASE_URL"); k = c.get("LLM", "API_KEY")
    assert m and b and k, "LLM 段缺少必填项"
    if b.rstrip('/').endswith('/chat/completions'):
        print("  base_url 以 /chat/completions 结尾, ChatOpenAI 会重复拼接 -> 应只填到 /v1"); sys.exit(1)
    print(f"  OK | 模型={m} | base_url={b} | key尾={k[-6:]}")
except SystemExit: raise
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:160]}"); sys.exit(1)
PYEOF

# ---- 1) 停止旧服务(含 worker 子进程) ----
echo "[1/4] 停止旧服务 ..."
SVC=$(lsof -t -i :$PORT -s TCP:LISTEN 2>/dev/null)
if [ -n "$SVC" ]; then
    CHILDREN=$(pgrep -P "$SVC" 2>/dev/null)
    kill "$SVC" 2>/dev/null; sleep 3
    ps -p "$SVC" >/dev/null 2>&1 && { kill -9 "$SVC" 2>/dev/null; sleep 1; }
    for c in $CHILDREN; do ps -p "$c" >/dev/null 2>&1 && kill -9 "$c" 2>/dev/null; done
    echo "  已停止旧服务(PID=$SVC)及其 worker 子进程"
else
    echo "  (当前无运行中的服务)"
fi
sleep 1
if lsof -i :$PORT -s TCP:LISTEN >/dev/null 2>&1; then
    echo "  端口 $PORT 仍被占用, 中止"; exit 1
fi

# ---- 2) 启动新服务 ----
# 线程数硬限制: numpy/GDAL 等库默认按 CPU 核数(本机 104 核)开线程,
# 单个栅格运算子进程即可打满共享宿主机的 CPU 与机械盘 IOPS(历史上曾致整机冻结, 需物理重启)。
# 这些变量由 uvicorn -> worker 子进程(spawn 继承环境) -> 代码执行子进程 层层继承。
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GDAL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
echo "[2/4] 启动新服务 (workers=$WORKERS, threads<=4/进程) ..."
GEOAGENT_MAX_WORKERS=$WORKERS nohup $PY -m uvicorn main:app --host 0.0.0.0 --port $PORT > "$LOG" 2>&1 &
echo "  新服务 PID=$!"

# ---- 3) 健康检查(最多等 ~20s) ----
echo "[3/4] 健康检查 ..."
OK=""
for i in $(seq 1 10); do
    sleep 2
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        OK=1; echo "  健康检查通过 (~$((i*2))s)"; break
    fi
done
[ -z "$OK" ] && { echo "  健康检查超时, 请查看日志: $LOG"; exit 1; }

# ---- 4) 端到端 LLM 验证 ----
if [ "$SKIP_LLM" = "1" ]; then
    echo "[4/4] 跳过端到端 LLM 验证 (--fast)"
else
    echo "[4/4] 端到端验证 LLM (走服务相同的 create_llm 链路) ..."
    $PY - <<'PYEOF'
import warnings; warnings.filterwarnings("ignore")
import os, configparser
c = configparser.ConfigParser(); c.read("config.ini")
os.environ['OPENAI_BASE_URL'] = c.get("LLM", "OPENAI_BASE_URL")
os.environ['API_KEY'] = c.get("LLM", "API_KEY")
try:
    from model.llm import create_llm
    r = create_llm("PLAN_LLM").invoke("只回复两个字:正常")
    print(f"  LLM 可用 | 模型={c.get('LLM','PLAN_LLM')} | 回复={r.content!r}")
except Exception as e:
    print(f"  LLM 调用失败: {type(e).__name__}: {str(e).replace(chr(10),' ')[:220]}")
    raise SystemExit(1)
PYEOF
    [ $? -ne 0 ] && { echo "  端到端失败 —— 服务已起, 但当前 config 的 KEY/模型/base_url 有问题"; exit 1; }
fi

echo "======================================================"
echo "重启完成, 服务就绪: http://0.0.0.0:$PORT (workers=$WORKERS)"
