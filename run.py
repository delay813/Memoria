"""
心忆 · Memoria 跨平台启动入口(可从任意工作目录运行)

用法:
    python run.py

等价于 `cd agent && python -m uvicorn server:app --host 127.0.0.1 --port 8080`
"""
import os
import sys

# 把 agent/ 目录加入模块搜索路径, 使 agent 内部 `import config` /
# `from agent import Agent` 等导入在任意 cwd 下都能工作(不依赖启动时的工作目录)
AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import uvicorn  # noqa: E402

import server  # noqa: E402  (FastAPI app 定义在 agent/server.py)


if __name__ == "__main__":
    print("=" * 50)
    print("心忆 · Memoria 启动")
    print("=" * 50)
    print("页面访问:       http://127.0.0.1:8080/")
    print("NPC列表:        GET  /agents")
    print("世界状态:       GET  /world")
    print("对话接口:       POST /chat   body: {\"agent_id\": \"npc_01\", \"question\": \"...\"}")
    uvicorn.run(server.app, host="127.0.0.1", port=8080)
