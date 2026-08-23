"""
总控主Agent: 加载预定义从Agent(NPC) + 世界时钟
- 把"今日世界上下文"(现实日期/节日)注入每个NPC的对话
- 特定日期(节日/角色生日)首次对话时, 触发角色特殊问候
"""
import threading
import time
from datetime import date

import config
from agent import Agent
from narrator import WorldClock


class Orchestrator:
    def __init__(self):
        cfg = config.load_agents_config()
        self.agents = {}
        for a in cfg.get("agents", []):
            agent = Agent(
                a["id"], a["name"], a.get("persona", ""),
                description=a.get("description", ""),
                birthday=a.get("birthday", ""),
            )
            self.agents[agent.agent_id] = agent

        self.world = WorldClock()

    def list_agents(self):
        return [
            {
                "id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "birthday": a.birthday,
                "favorability": a.get_favor(),
                "mood": a.get_mood(),
            }
            for a in self.agents.values()
        ]

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)

    def get_history(self, agent_id):
        agent = self.get_agent(agent_id)
        if not agent:
            return None
        return agent.get_history()

    def _today_special_names(self, agent):
        """今天对该角色而言的特殊日期名称列表(全局节日 + 角色生日)"""
        d = self.world.today()
        names = [f["name"] for f in self.world.festivals_on(d)]
        if agent.birthday:
            try:
                bm, bd = (int(x) for x in agent.birthday.split("-"))
                if (bm, bd) == (d.month, d.day):
                    names.append(f"{agent.name}的生日")
            except Exception:
                pass
        return names

    def chat_stream(self, agent_id, question):
        """与指定NPC对话(流式): 注入今日世界上下文 + 特殊问候"""
        agent = self.get_agent(agent_id)
        if not agent:
            yield f"❌ 未找到角色: {agent_id}"
            return
        world_context = self.world.today_context()
        special_names = self._today_special_names(agent)
        greet_hint = self.world.greeting_hint(agent_id, special_names)
        yield from agent.chat_stream(
            question, world_context=world_context, greet_hint=greet_hint
        )

    def snapshot(self):
        return self.world.snapshot()

    def reset_all(self):
        """初始化所有角色: 清空对话/记忆/好感度, 重置世界"""
        for a in self.agents.values():
            a.reset()
        self.world.reset()
        return {"status": "ok", "message": "所有角色已初始化"}

    # ============================================================
    # 主动消息(惰性生成: 用户上线时才判断是否该发)
    # ============================================================
    def check_proactive(self):
        """用户上线时调用: 判断每个角色是否该主动发消息"""
        today = self.world.today()
        today_iso = today.isoformat()
        now = time.time()
        for a in self.agents.values():
            # 1) 节日/生日: 当天主动送祝福(每天一次)
            names = self._today_special_names(a)
            if names and a.get_last_proactive().get("festival") != today_iso:
                content = a.generate_proactive(f"今天是{'、'.join(names)}，想主动送上节日祝福或问候")
                if content:
                    a.add_proactive("festival", content)
                    a.mark_proactive("festival", today_iso)
                    print(f"[{a.name}] 主动消息(节日): {content}")
                continue
            # 2) 普通主动(想念/想起): 与好感度挂钩 + 随时间衰减 + 冷却
            if a.should_send_checkin(now):
                days = 0
                if a.get_last_seen():
                    try:
                        days = (today - date.fromisoformat(a.get_last_seen())).days
                    except Exception:
                        days = 0
                content = a.generate_proactive(f"已经有{days}天没和对方联系了，心里有点想念")
                if content:
                    a.add_proactive("checkin", content)
                    a.mark_proactive("checkin", now)
                    print(f"[{a.name}] 主动消息(想念): {content}")

    def inbox_summary(self):
        """各角色未读主动消息数, 供前端红点提示"""
        return {
            "agents": [
                {"id": a.agent_id, "name": a.name, "unread": a.unread_count()}
                for a in self.agents.values()
            ]
        }


_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
