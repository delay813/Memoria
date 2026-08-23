"""
多NPC对话系统 - Web服务API

架构:
- 总控(Orchestrator): 加载NPC + 现实时间世界时钟(节日/每日问候)
- 从Agent(NPC): 独立人格, 会话历史/摘要/画像/长期记忆完全隔离
- 世界时钟: 现实时间推进, 特殊日期(节日/生日)触发角色特殊对话
"""
import json
import os

import uvicorn
from fastapi import FastAPI, Request
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


@app.get("/inbox")
async def inbox_summary():
    """用户上线时检查主动消息(惰性生成), 并返回各角色未读数"""
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
async def reset_all():
    """初始化所有角色: 清空对话记录/记忆/好感度, 重置世界状态"""
    return orch.reset_all()


@app.post("/chat")
async def chat_stream(request: Request):
    """与指定NPC对话, SSE流式返回(注入今日世界上下文与特殊问候)"""
    body = await request.json()
    agent_id = body.get("agent_id")
    question = body.get("question")

    if not question:
        return {"error": "问题不能为空"}

    agent = orch.get_agent(agent_id)
    if not agent:
        return {"error": f"未找到角色: {agent_id}, 可用角色见 GET /agents"}

    def generation():
        for chunk in orch.chat_stream(agent_id, question):
            data = json.dumps({"content": chunk, "done": False}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'content': '', 'done': True}, ensure_ascii=False)}\n\n"

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
