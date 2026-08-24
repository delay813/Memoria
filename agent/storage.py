"""
统一持久化工具: 原子写 JSON。
所有"先写临时文件再 os.replace 覆盖"的落盘统一走这里, 避免并发/崩溃时写出半截 JSON 导致下次加载静默清空。
"""
import json
import os
import tempfile


def save_json(path, obj, indent=2):
    """原子写入 JSON 到 path: 先写同目录临时文件, 再 os.replace 原子覆盖。

    - 同目录临时文件保证 rename 在同一文件系统上(原子性)。
    - 任何一步失败都尽量清理临时文件, 不影响原文件。
    """
    path = os.fspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise
