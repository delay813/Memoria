"""
世界模拟: NPC 作息 + 低概率随机事件(不在场时的生命感)
- 每个NPC有一张作息表(按小时划分), 决定其当前"状态"(上课/社团/值班/睡觉/空闲)
- 每天(惰性)为每个NPC按低概率触发一条随机事件, 影响其心情并注入对话
- 用户离开期间世界照常运转: 回填错过的日子(最多 LIFE_BACKFILL_MAX_DAYS 天), 形成生命感
- 本模块只做"纯逻辑 + 状态机", 心情的应用由 Agent 完成, 不直接调用模型
"""
import json
import os
import random
import time
from datetime import date, datetime, timedelta

import config
import storage
import time_utils


def _in_slot(hour, start, end):
    """判断 hour(0~24, 可含小数) 是否落在 [start, end) 区间; end<=start 表示跨午夜(复用 time_utils)。"""
    return time_utils.in_interval(hour, start, end)


def resolve_schedule(agent_id, now=None):
    """返回该角色当前时间点的作息槽(activity/label/busy/sleepy)。"""
    now = now or datetime.now()
    schedule = config.NPC_SCHEDULES.get(agent_id) or config.DEFAULT_SCHEDULE
    hour = now.hour + now.minute / 60.0
    for slot in schedule:
        if _in_slot(hour, slot.get("start", 0), slot.get("end", 24)):
            return slot
    return schedule[-1]


def schedule_for(agent_id):
    """返回该角色的作息表(原始槽位列表)。"""
    return config.NPC_SCHEDULES.get(agent_id) or config.DEFAULT_SCHEDULE


def format_schedule(agent_id, now=None):
    """返回今日行程(带当前时段高亮), 供前端角色卡展示。"""
    now = now or datetime.now()
    hour = now.hour + now.minute / 60.0
    out = []
    for slot in schedule_for(agent_id):
        start, end = slot.get("start", 0), slot.get("end", 24)
        out.append({
            "time": f"{start:02d}:00-{end:02d}:00",
            "label": slot.get("label", ""),
            "activity": slot.get("activity", ""),
            "busy": bool(slot.get("busy", False)),
            "sleepy": bool(slot.get("sleepy", False)),
            # 按「当前小时是否落在该槽位」标记, 避免同名 label(如两个"上课中")被同时高亮
            "current": _in_slot(hour, start, end),
        })
    return out


def pick_random_event(agent_id, rng=None):
    """从通用 + 该角色专属事件池中随机选一条事件(低概率触发时调用)。"""
    rng = rng or random
    pool = list(config.RANDOM_EVENTS) + list(config.NPC_RANDOM_EVENTS.get(agent_id, []))
    if not pool:
        return None
    return dict(rng.choice(pool))


class LifeSim:
    """单个NPC的"生活状态机": 记录每天是否发生随机事件(惰性推进, 幂等)。"""

    def __init__(self, agent_id, file_path):
        self.agent_id = agent_id
        self.file_path = file_path
        self.last_sim_date = None       # "YYYY-MM-DD", 上次模拟到哪天
        self.daily_events = []          # 事件列表(按时间正序追加, 读取时倒序取最近)
        self._pending_followups = []    # 已排期、尚未触发的后续事件链
        self.daily_script = {}          # 当日"命运大纲" {date, outline, reach_out, reason}
        self._load()

    # ============================================================
    # 持久化
    # ============================================================
    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.last_sim_date = data.get("last_sim_date")
                    daily = data.get("daily_events", [])
                    if isinstance(daily, list):
                        self.daily_events = daily
                    pending = data.get("pending_followups", [])
                    if isinstance(pending, list):
                        self._pending_followups = pending
                    script = data.get("daily_script", {})
                    if isinstance(script, dict):
                        self.daily_script = script
        except Exception as e:
            print(f"[世界模拟] {self.agent_id} 生活状态加载失败: {e}")

    def _save(self):
        try:
            storage.save_json(self.file_path, {
                "last_sim_date": self.last_sim_date,
                "daily_events": self.daily_events,
                "pending_followups": self._pending_followups,
                "daily_script": self.daily_script,
            })
        except Exception as e:
            print(f"[世界模拟] {self.agent_id} 生活状态保存失败: {e}")

    # ============================================================
    # 掷事件 / 推进
    # ============================================================
    def _emit_event(self, date_iso, ev):
        """把一条事件(随机或后续)落库, 并处理其后续链。返回事件记录。"""
        record = {
            "date": date_iso,
            "key": ev.get("key", ""),
            "kind": ev.get("kind", ""),
            "text": ev.get("text", ""),
            "mood": int(ev.get("mood", 0)),
            "ts": time.time(),
        }
        self.daily_events.append(record)
        self.daily_events = self.daily_events[-config.LIFE_EVENTS_KEEP:]
        fup = ev.get("followup")
        if fup and fup.get("key") and fup.get("text"):
            delay = max(1, int(fup.get("delay_days", 1)))
            due = (date.fromisoformat(date_iso) + timedelta(days=delay)).isoformat()
            self._pending_followups.append({
                "date": due,
                "key": fup.get("key", ""),
                "kind": fup.get("kind", ""),
                "text": fup.get("text", ""),
                "mood": int(fup.get("mood", 0)),
                "followup": fup.get("followup"),
            })
        return record

    def _roll_day(self, date_iso, rng=None):
        """为某天产生一条事件: 优先触发到期后续链, 否则掷随机。命中返回记录, 否则 None。"""
        rng = rng or random
        # 1) 先触发已排期的后续事件(每天最多一条, 保持克制)
        due = [f for f in self._pending_followups if f.get("date") == date_iso]
        self._pending_followups = [f for f in self._pending_followups if f.get("date") != date_iso]
        if due:
            return self._emit_event(date_iso, due[0])
        # 2) 否则掷随机事件
        if rng.random() >= config.RANDOM_EVENT_PROBABILITY:
            return None
        ev = pick_random_event(self.agent_id, rng=rng)
        if not ev:
            return None
        return self._emit_event(date_iso, ev)

    def advance(self, today=None, rng=None):
        """推进世界到今天: 回填上次模拟以来的每一天(最多 LIFE_BACKFILL_MAX_DAYS 天), 返回新事件列表。

        幂等: 同一天多次调用不会重复生成; 首次运行只掷今天, 不回填历史。
        """
        today = today or time_utils.today_iso()
        if self.last_sim_date is None:
            ev = self._roll_day(today, rng=rng)
            self.last_sim_date = today
            self._save()
            return [ev] if ev else []
        try:
            last = date.fromisoformat(self.last_sim_date)
            cur = date.fromisoformat(today)
        except Exception:
            self.last_sim_date = today
            self._save()
            return []
        if cur <= last:
            return []
        days = (cur - last).days
        start = last + timedelta(days=1)
        if days > config.LIFE_BACKFILL_MAX_DAYS:
            # 离开太久只模拟最近这几天(否则会一次性堆出大量事件)
            start = cur - timedelta(days=config.LIFE_BACKFILL_MAX_DAYS - 1)
        new_events = []
        d = start
        while d <= cur:
            ev = self._roll_day(d.isoformat(), rng=rng)
            if ev:
                new_events.append(ev)
            d += timedelta(days=1)
        self.last_sim_date = today
        self._save()
        return new_events

    # ============================================================
    # 查询
    # ============================================================
    def today_event(self, today=None):
        """今天发生的事件(可能为 None)。"""
        today = today or time_utils.today_iso()
        for ev in reversed(self.daily_events):
            if ev.get("date") == today:
                return ev
        return None

    def recent_events(self, k=3):
        """最近 k 条事件(按时间倒序)。"""
        return list(reversed(self.daily_events[-k:]))

    # ============================================================
    # 命运大纲(每日剧本): 由旁白/模型生成, 这里只负责存取
    # ============================================================
    def get_script(self):
        """返回当前保存的命运大纲 dict(可能为 None)。"""
        return dict(self.daily_script) if self.daily_script else None

    def set_script(self, script):
        """保存命运大纲(覆盖)。"""
        self.daily_script = dict(script or {})
        self._save()

    def reset(self):
        """清空生活状态并删除持久化文件。"""
        self.last_sim_date = None
        self.daily_events = []
        self._pending_followups = []
        self.daily_script = {}
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
        except Exception as e:
            print(f"[世界模拟] 删除生活状态失败: {e}")
