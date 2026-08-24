"""
统一 LLM 调用封装(DashScope Generation)。

- call(): 非流式调用, 带超时重试, 成功返回 response, 失败返回 None
- call_text(): 非流式调用并直接返回助手文本(去首尾空白), 失败返回空串

原本 agent.py 的 _call_generation 与 longterm_memory.py 的 _call_small 各自维护了一份
"重试 + 退避"逻辑, 现统一至此, 消除重复。
"""
import time

from dashscope import Generation

import config


def call(messages, model=None):
    """非流式 Generation 调用, 带超时重试; 成功返回 response, 失败返回 None。"""
    model = model or config.CHAT_MODEL
    for attempt in range(config.MODEL_MAX_RETRIES + 1):
        try:
            resp = Generation.call(
                api_key=config.API_KEY,
                model=model,
                messages=messages,
                result_format="message",
            )
            if resp.status_code == 200:
                return resp
            print(f"模型调用返回非200(status={resp.status_code}), 第{attempt+1}次")
        except Exception as e:
            print(f"模型调用异常(第{attempt+1}次): {e}")
        if attempt < config.MODEL_MAX_RETRIES:
            time.sleep(config.MODEL_RETRY_DELAY)
    return None


def call_text(messages, model=None):
    """非流式调用并直接返回助手文本(去首尾空白); 失败返回空串。"""
    resp = call(messages, model=model)
    if not resp:
        return ""
    try:
        return (resp.output.choices[0].message.content or "").strip()
    except Exception:
        return ""
