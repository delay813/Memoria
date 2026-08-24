"""
总控主Agent: 加载预定义从Agent(NPC) + 世界时钟
- 把"今日世界上下文"(现实日期/节日)注入每个NPC的对话
- 特定日期(节日/角色生日)首次对话时, 触发角色特殊问候
"""
import random
import time

import config
import time_utils
from agent import Agent
from narrator import WorldClock
from user_profile import UserProfile


class Orchestrator:
    def __init__(self):
        cfg = config.load_agents_config()
        self.user = UserProfile(config.USER_FILE)
        self.agents = {}
        for a in cfg.get("agents", []):
            agent = Agent(
                a["id"], a["name"], a.get("persona", ""),
                description=a.get("description", ""),
                birthday=a.get("birthday", ""),
                unlock_cb=self.user.unlock,
            )
            self.agents[agent.agent_id] = agent

        self.world = WorldClock()

    def list_agents(self):
        result = []
        for a in self.agents.values():
            st = a.current_status()
            visual = config.NPC_VISUALS.get(a.agent_id, {})
            result.append({
                "id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "birthday": a.birthday,
                "avatar": visual.get("avatar", ""),
                "card": visual.get("card", ""),
                "en": visual.get("en", ""),
                "tags": visual.get("tags", []),
                "theme": visual.get("theme", {}),
                "favorability": a.get_favor(),
                "stage": a.get_stage(),
                "mood": a.get_mood(),
                "status": st["label"],
                "activity": st["activity"],
                "busy": st["busy"],
                "tension_label": a.tension_label(),
                "quote": a.dynamic_quote(),
            })
        return result

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

    def _social_context(self, agent_id):
        """跨角色互动: 告知当前角色"用户最近也和其他角色有互动"及其他角色近况, 让角色自然关心/转述"""
        today = self.world.today()
        others = []
        for aid, a in self.agents.items():
            if aid == agent_id:
                continue
            last = a.get_last_seen()
            if not last:
                continue
            days = time_utils.days_since(last, ref=today)
            if 0 <= days <= 2:
                st = a.current_status()
                ev = a.life.today_event()
                line = f"{a.name}（{st['label']}）"
                if ev:
                    line += f"，她今天{ev.get('text', '')}"
                others.append(line)
        if not others:
            return ""
        detail = "\n".join(f"- {o}" for o in others)
        return (
            "【你听到的关于其他角色和用户的消息】\n"
            f"{detail}\n"
            "你可以自然地在合适的时候提起、关心、好奇或转述这些，但不要每句话都说、不要显得刻意。"
        )

    def _relay_context(self, agent):
        """低概率随机挑一位其他角色, 返回其近况(供普通主动消息"转述"); 无则返回 None。"""
        others = [a for aid, a in self.agents.items() if aid != agent.agent_id]
        if not others:
            return None
        if random.random() >= config.CROSS_NPC_RELAY_PROBABILITY:
            return None
        other = random.choice(others)
        st = other.current_status()
        ev = other.life.today_event()
        if ev:
            return f"你最近听说{other.name}今天{ev.get('text', '')}，想顺便和对方聊聊这件事"
        return f"你最近见到{other.name}在{st['activity']}，想顺便和对方提一句"

    def _achievement_event(self, key):
        """解锁成就并返回事件数据; 未新解锁则返回 None"""
        if self.user.unlock(key):
            a = self.user.find_achievement(key)
            if a:
                return {"type": "achievement", "data": {
                    "key": a["key"], "emoji": a["emoji"],
                    "label": a["label"], "desc": a["desc"],
                }}
        return None

    def chat_stream(self, agent_id, question):
        """与指定NPC对话(流式): 注入世界上下文/昵称/跨角色社交上下文 + 成就解锁事件"""
        agent = self.get_agent(agent_id)
        if not agent:
            yield {"type": "error", "data": f"未找到角色: {agent_id}"}
            return
        agent.sync_life()
        agent.ensure_daily_script()
        world_context = self.world.today_context()
        world_context += "\n" + agent.daily_context()
        social_context = self._social_context(agent_id)

        nickname = self.user.get_nickname()

        # "初次相遇"成就: 只有真正产生一轮对话(非深夜沉睡秒回)才解锁
        if not agent.current_status().get("sleepy"):
            ev = self._achievement_event("first_chat")
            if ev:
                yield ev

        milestone_data = None
        reconciled = False
        for event in agent.chat_stream(
            question, world_context=world_context,
            social_context=social_context or None, user_nickname=nickname,
        ):
            if event.get("type") == "favor":
                data = event.get("data") or {}
                fav = data.get("favorability", 0)
                for key, threshold in (("favor_45", 45), ("favor_70", 70)):
                    if fav >= threshold:
                        ev = self._achievement_event(key)
                        if ev:
                            yield ev
                if data.get("reconciled"):
                    reconciled = True
            if event.get("type") == "milestone":
                milestone_data = event.get("data") or {}
            yield event

        # 锁已释放: 生成剧情消息(里程碑/和好)并送进收件箱 + 透出给前端
        if milestone_data:
            content = agent.generate_milestone_message(
                milestone_data.get("from", ""), milestone_data.get("to", ""))
            if content:
                agent.add_proactive("milestone", content)
                yield {"type": "milestone_story", "data": {"name": agent.name, "content": content}}
        if reconciled:
            content = agent.generate_reconcile_message()
            if content:
                agent.add_proactive("reconcile", content)
                yield {"type": "reconcile", "data": {"name": agent.name, "content": content}}

        # 和全部角色都聊过天
        if len(self.agents) > 1 and all(a.get_last_seen() for a in self.agents.values()):
            ev = self._achievement_event("three_agents")
            if ev:
                yield ev

    def get_memory_profile(self, agent_id):
        """返回指定NPC的记忆卡("她眼中的你"): 事实库 + 画像 + 关系状态"""
        agent = self.get_agent(agent_id)
        if not agent:
            return None
        return agent.get_memory_profile()

    def suggest_topics(self, agent_id):
        """为指定角色生成 3 个开场话题"""
        agent = self.get_agent(agent_id)
        if not agent:
            return None
        return agent.suggest_topics()

    def get_achievements(self):
        """返回全部成就及解锁状态(供前端成就墙)"""
        return self.user.all_achievements()

    def snapshot(self):
        snap = self.world.snapshot()
        npcs = []
        for a in self.agents.values():
            st = a.current_status()
            npcs.append({
                "id": a.agent_id,
                "name": a.name,
                "status": st["label"],
                "activity": st["activity"],
                "busy": st["busy"],
                "events": a.life.recent_events(2),
            })
        snap["npcs"] = npcs
        return snap

    def reset_all(self):
        """初始化所有角色: 清空对话/记忆/好感度/日记, 重置世界"""
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
            # 0) 惰性推进世界模拟(作息/随机事件): 回填离开期间的日子
            a.sync_life()
            # 0.5) 惰性生成今天的命运大纲(剧本): 决定今天做什么 + 要不要主动
            a.ensure_daily_script()
            # 1) 惰性触发隔夜整理(深睡): 用户上线后检查, 无新互动则免做
            a.maybe_dream()
            # 1.5) 补回深夜积压的消息: 她已醒来且不再睡觉时, 生成延迟回复进收件箱
            if a.has_pending_replies() and not a.current_status()["sleepy"]:
                content = a.generate_delayed_reply()
                if content:
                    a.add_proactive("reply", content)
                    print(f"[{a.name}] 补回深夜消息: {content}")
            # 2) 节日/生日: 当天主动送祝福(每天一次); 深夜睡着先不送(等醒), 与聊天侧一致
            names = self._today_special_names(a)
            if names and not a.current_status().get("sleepy") and a.try_claim_festival(today_iso):
                # try_claim_festival 已原子占位, 防止 30s 轮询并发重复生成同一条祝福
                content = a.generate_proactive(f"今天是{'、'.join(names)}，想主动送上节日祝福或问候")
                if content:
                    a.add_proactive("festival", content)
                    self.user.unlock("first_proactive")
                    print(f"[{a.name}] 主动消息(节日): {content}")
                continue
            # 3) 普通主动(想念/想起): 与好感度挂钩 + 随时间衰减 + 冷却
            if a.try_claim_checkin(now):
                # try_claim_checkin 已原子占位(写入冷却), 防止并发轮询重复生成同一条"想念"
                days = time_utils.days_since(a.get_last_seen(), ref=today)
                reason = f"已经有{days}天没和对方联系了，心里有点想念"
                script = a.get_daily_script() or {}
                if script.get("outline"):
                    reason += f"；你今天的安排是：{script['outline']}"
                relay = self._relay_context(a)
                if relay:
                    reason += f"；另外，{relay}"
                content = a.generate_proactive(reason)
                if content:
                    a.add_proactive("checkin", content)
                    self.user.unlock("first_proactive")
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
