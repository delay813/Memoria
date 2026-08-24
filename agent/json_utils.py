"""
从大模型输出中稳健解析 JSON 的统一工具。

原本 agent.py 的 _parse_json、longterm_memory.forget、agent._summarize_session、
agent._generate_topics 各自内联了"去代码围栏 + 正则提取 + json.loads"的重复逻辑, 现统一至此。
"""
import json
import re


def _strip_fences(content):
    """去掉模型常带出的 ```json / ``` 围栏。"""
    return re.sub(r"```(?:json)?", "", content or "")


def parse_object(content):
    """解析一个 JSON 对象; 失败安全降级为空 dict。"""
    if not content:
        return {}
    content = _strip_fences(content)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_array(content):
    """解析一个 JSON 数组; 失败安全降级为空 list。"""
    if not content:
        return []
    content = _strip_fences(content)
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except Exception:
        return []
    return arr if isinstance(arr, list) else []
