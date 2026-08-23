#!/usr/bin/env bash
# 心忆 · Memoria 启动脚本(Linux / macOS)
set -e
cd "$(dirname "$0")"

# 优先使用项目虚拟环境, 否则回退到系统 python3
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" run.py
