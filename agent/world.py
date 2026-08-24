"""
世界模拟接入(WorldMixin): 作息状态 + 随机事件注入对话 + 每日命运大纲(剧本)。

- current_status / schedule_today: 供前端展示角色此刻在做什么
- daily_context: 汇总"今日"上下文(状态 + 大纲 + 事件)一次性注入对话
- sync_life: 惰性推进世界, 回填离开期间的日子
"""
import config
import json_utils
import llm
import time_utils
from world_sim import format_schedule, resolve_schedule


class WorldMixin:
    """作息状态 / 随机事件 / 命运大纲。混入 Agent。"""

    # ============================================================
    # 世界模拟: 作息状态 + 低概率随机事件(不在场时的生命感)
    # ============================================================
    def current_status(self):
        """当前作息状态(上课中/社团活动中/睡觉中/空闲等), 供前端展示。"""
        slot = resolve_schedule(self.agent_id)
        return {
            "activity": slot.get("activity", ""),
            "label": slot.get("label", ""),
            "busy": bool(slot.get("busy", False)),
            "sleepy": bool(slot.get("sleepy", False)),
        }

    def schedule_today(self):
        """今日行程(带当前时段高亮), 供前端角色卡展示。"""
        return format_schedule(self.agent_id)

    def status_context(self):
        """把当前作息状态写成提示词片段, 注入对话: 忙时自然说"晚点回你", 深夜带睡意。"""
        st = self.current_status()
        label = st.get("label", "") or ""
        activity = st.get("activity", "") or ""
        if st.get("sleepy"):
            return (
                f"【你此刻的状态】你现在{activity}（{label}），已经很晚了。"
                "深夜回复要迷迷糊糊、轻柔简短，可以带一点睡意或迷迷糊糊的语气。"
            )
        if st.get("busy"):
            return (
                f"【你此刻的状态】你现在{activity}（{label}），有点忙，但你还是抽空回复了对方。"
                "回复要简短，可以自然带一句自己在忙、是抽空回的消息，"
                "例如'我在上课，偷偷回你一下'；要显得真心、不敷衍。"
            )
        return f"【你此刻的状态】你现在{activity}（{label}），比较有空，可以正常聊。"

    def life_events_context(self):
        """今天/最近发生在角色身上的随机事件, 注入对话, 让世界有"不在场时的生命感"。"""
        today = time_utils.today_iso()
        ev = self.life.today_event(today)
        if ev:
            return (
                f"【今天发生在你身上的事】{ev.get('text', '')}。"
                "你会在对话中自然流露这种心情，但不要生硬地原句复述。"
            )
        recents = self.life.recent_events(k=2)
        if recents and recents[0].get("date") != today:
            parts = "；".join(e.get("text", "") for e in recents if e.get("text"))
            if parts:
                return f"【你最近经历的事】{parts}。如果话题相关，可以自然提起。"
        return ""

    # ============================================================
    # 命运大纲(每日剧本): 旁白每天为角色写一条极简"今天做什么 + 要不要主动"
    # ============================================================
    def _generate_daily_script(self, today_iso):
        """生成今天的极简命运大纲。返回 {date, outline, reach_out, reason} 或 {}。"""
        persona = self._current_persona()
        facts = self.fact_memory.all()
        facts_text = "、".join(facts) if facts else "暂无"
        profile = self._load_memory().get("user_profile", "") or "暂无"
        sched = format_schedule(self.agent_id)
        sched_text = "；".join(f"{s['time']} {s['label']}" for s in sched)
        ev = self.life.today_event(today_iso)
        ev_text = ev.get("text", "") if ev else "无特别事件"

        prompt = f"""你是「{self.name}」的旁白编剧。请为「{self.name}」写今天({today_iso})的极简命运大纲。

【她的核心人格】
{persona}

【她今天的时间安排】{sched_text}
【今天发生在她身上的事】{ev_text}
【当前与用户的关系】好感度 {self._favor}/100，关系阶段{self._favor_stage()}；此刻心情 {self._mood_label()}
【她对用户的了解】{facts_text}
【她对用户的印象】{profile}

请只做两件事:
1. 用 1 句话写她"今天的心境/特别打算"的极简大纲(例如"课程较满，有点累，想早点休息"或"今天格外想你")。不要复述固定的作息表(上课/社团/睡觉等系统已掌握), 只写今天有别于平常、或最能代表她今日心境的那一点; 必须符合人设与事实, 绝不能写出与身份、性格矛盾的事。
2. 判断她今天"要不要主动找用户聊天"(reach_out): 综合好感度高低、关系阶段、今天忙不忙、以及她对用户的了解与想念程度。给一句极简理由(可空)。

严格输出JSON(不要任何其他文字):
{{"outline":"...","reach_out":true,"reason":"..."}}
"""
        resp = llm.call([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if not resp:
            return {}
        info = json_utils.parse_object(resp.output.choices[0].message.content)
        if not isinstance(info, dict):
            return {}
        outline = str(info.get("outline", "")).strip()
        if not outline:
            return {}
        raw = info.get("reach_out")
        if isinstance(raw, str):
            reach_out = raw.strip().lower() in ("true", "1", "yes", "是", "想", "要")
        else:
            reach_out = bool(raw)
        reason = str(info.get("reason", "")).strip()
        return {"date": today_iso, "outline": outline, "reach_out": reach_out, "reason": reason}

    def ensure_daily_script(self):
        """惰性生成今天的命运大纲(每天一次, 幂等)。网络调用在锁外进行。"""
        today = time_utils.today_iso()
        script = self.life.get_script() or {}
        if script.get("date") == today:
            return script
        if self._scripting:
            return script or {}
        self._scripting = True
        try:
            script = self._generate_daily_script(today)
            if script:
                self.life.set_script(script)
            return script or {}
        finally:
            self._scripting = False

    def get_daily_script(self):
        """返回当天的命运大纲 dict(可能为空)。"""
        return dict(self.life.get_script() or {})

    def script_context(self):
        """把当天的命运大纲写成提示词片段, 注入对话, 让角色"遵从大纲"行动。"""
        script = self.life.get_script() or {}
        today = time_utils.today_iso()
        if script.get("date") == today and script.get("outline"):
            return (
                f"【你今天的命运大纲】{script['outline']}。"
                "这是你今天的行动与心境的依据，请自然地带入对话，但不要逐字复述大纲。"
            )
        return ""

    def daily_context(self):
        """汇总"今日"上下文(当前状态 + 命运大纲 + 当天/最近事件), 一次性注入对话。"""
        parts = [self.status_context()]
        script = self.script_context()
        if script:
            parts.append(script)
        events = self.life_events_context()
        if events:
            parts.append(events)
        return "\n".join(parts)

    def sync_life(self):
        """惰性推进世界: 回填离开期间的日子、掷随机事件, 并把事件带来的心情变化应用到自身。"""
        with self._lock:
            try:
                events = self.life.advance()
            except Exception as e:
                print(f"[{self.name}] 世界模拟推进失败: {e}")
                return
            if not events:
                return
            for ev in events:
                self._apply_mood(int(ev.get("mood", 0)))
            self._unlock_cb("life_event")
