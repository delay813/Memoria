"""
多NPC对话系统 - Web服务API

架构:
- 总控(Orchestrator): 加载NPC + 现实时间世界时钟(节日/每日问候)
- 从Agent(NPC): 独立人格, 会话历史/摘要/画像/长期记忆完全隔离
- 世界时钟: 现实时间推进, 特殊日期(节日/生日)触发角色特殊对话
"""
import json
import os
import threading

import uvicorn
from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

import config
from orchestrator import get_orchestrator

# ============================================================
# 一、Web服务入口
# ============================================================

app = FastAPI(
    title="心忆 · Memoria",
    description="多角色对话, 每个角色独立人格与记忆, 现实时间世界",
)

# 挂载静态文件目录(用绝对路径, 不依赖启动时工作目录)
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")

# 启动总控: 加载NPC + 初始化现实时间世界时钟
orch = get_orchestrator()


def _prefetch_topics():
    """后台预生成各角色开场卡(命中缓存), 让首次加载也秒开; 失败不影响服务。"""
    if not config.TOPIC_PREFETCH:
        return
    for aid in list(orch.agents.keys()):
        try:
            orch.suggest_topics(aid)
        except Exception as e:  # noqa: BLE001
            print(f"[开场卡预热] {aid} 生成失败: {e}")


threading.Thread(target=_prefetch_topics, daemon=True).start()


# ============================================================
# 二、接口
# ============================================================

@app.get("/agents")
async def list_agents():
    """列出所有可对话的NPC(含生日等信息)"""
    return {"agents": orch.list_agents()}


@app.get("/agents/{agent_id}/history")
async def agent_history(agent_id: str):
    """返回指定NPC的对话历史(供前端恢复会话)"""
    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}"}
    return {"agent_id": agent_id, "name": agent.name, "messages": orch.get_history(agent_id)}


@app.get("/agents/{agent_id}/diary")
async def agent_diary(agent_id: str):
    """返回指定NPC的日历史(隔夜整理的梦境日记), 供前端回看"""
    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}"}
    return {"agent_id": agent_id, "name": agent.name, "diary": agent.get_daily_log()}


@app.get("/agents/{agent_id}/memory")
async def agent_memory(agent_id: str):
    """返回指定NPC的记忆卡("她眼中的你"): 事实库 + 用户画像 + 关系状态"""
    data = orch.get_memory_profile(agent_id)
    if not data:
        return {"error": f"未找到角色: {agent_id}"}
    agent = orch.get_agent(agent_id)
    return {"agent_id": agent_id, "name": agent.name, **data}


@app.get("/agents/{agent_id}/topics")
async def agent_topics(agent_id: str):
    """为指定角色生成开场话题(线程池执行, 避免模型调用阻塞事件循环)"""
    topics = await run_in_threadpool(orch.suggest_topics, agent_id)
    if topics is None:
        return {"error": f"未找到角色: {agent_id}"}
    return {"agent_id": agent_id, "topics": topics}


@app.get("/agents/{agent_id}/schedule")
async def agent_schedule(agent_id: str):
    """返回指定NPC的今日行程(带当前时段高亮)"""
    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}"}
    return {"agent_id": agent_id, "schedule": agent.schedule_today()}


@app.get("/user")
async def get_user():
    """当前用户档案(昵称)"""
    return {"nickname": orch.user.get_nickname()}


@app.post("/user")
async def set_user(request: Request):
    """设置用户昵称(称呼随关系演进用)"""
    body = await request.json()
    nickname = str(body.get("nickname") or "")[:20]
    orch.user.set_nickname(nickname)
    return {"nickname": orch.user.get_nickname()}


@app.get("/achievements")
async def achievements():
    """全部成就及解锁状态(供前端成就墙)"""
    return {"achievements": orch.get_achievements()}


@app.get("/inbox")
def inbox_summary():
    """用户上线时检查主动消息(惰性生成), 并返回各角色未读数

    注意: 必须用普通 def, 不能 async def —— check_proactive() 内部会同步调用模型
    生成主动消息(单次数秒)。async 端点的同步代码直接在事件循环里执行, 前端每 30s
    轮询一次本接口会把整个服务阻塞数秒(表现为全站周期性卡顿)。普通 def 会被
    FastAPI 自动放进线程池, 不阻塞事件循环。
    """
    orch.check_proactive()
    return orch.inbox_summary()


@app.get("/agents/{agent_id}/inbox")
async def agent_inbox(agent_id: str):
    """返回指定NPC的主动消息收件箱"""
    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}"}
    return {"agent_id": agent_id, "messages": agent.get_inbox()}


@app.post("/agents/{agent_id}/inbox/read")
async def agent_inbox_read(agent_id: str):
    """标记指定NPC的主动消息为已读"""
    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}"}
    agent.mark_read()
    return {"status": "ok"}


@app.get("/world")
async def world_state():
    """当前世界状态: 现实日期/星期/经过天数/今日节日/事件时间轴"""
    return orch.snapshot()


@app.post("/reset")
async def reset_all(request: Request):
    """初始化所有角色: 清空对话记录/记忆/好感度/日记, 重置世界状态

    破坏性操作(不可恢复): 必须带 ?confirm=yes 才会真正执行, 防止误触/脚本误调清空全部数据。
    """
    if request.query_params.get("confirm") != "yes":
        return {"error": "重置会清空所有角色的对话/记忆/好感度/日记，不可恢复。确认请调用 POST /reset?confirm=yes"}
    return orch.reset_all()


@app.post("/chat")
async def chat_stream(request: Request):
    """与指定NPC对话, SSE流式返回(注入今日世界上下文与特殊问候)"""
    body = await request.json()
    agent_id = body.get("agent_id")
    # 限长 2000 字符: 超长输入会直接进入对话历史/向量化/系统提示词, 造成 token 与费用爆炸
    question = str(body.get("question") or "")[:2000]

    if not question:
        return {"error": "问题不能为空"}

    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}, 可用角色见 GET /agents"}

    def generation():
        for event in orch.chat_stream(agent_id, question):
            # 事件统一为 dict: {type: cognition/content/favor/milestone/error, data: ...}
            if isinstance(event, dict):
                data = event
            else:
                data = {"type": "content", "data": event}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/")
async def root():
    return RedirectResponse("/static/chat.html")


# ============================================================
# 三、启动入口
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("心忆 · Memoria 启动")
    print("=" * 50)
    print("页面访问:       http://127.0.0.1:8080/")
    print("NPC列表:        GET  /agents")
    print("世界状态:       GET  /world")
    print("对话接口:       POST /chat   body: {\"agent_id\": \"npc_01\", \"question\": \"...\"}")
    uvicorn.run(app, host="127.0.0.1", port=8080)
