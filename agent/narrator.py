"""
世界时钟(旁白): 基于现实时间推进世界, 维护事件时间轴
- 现实一天 = 世界一天, 使用真实日历日期(不再用"游戏内第N天"加速)
- 维护特殊日期日历(节日等), 供角色在特定日期触发特殊对话(如圣诞祝福)
- 跨天时记录"新的一天"事件; 记录每个角色每天是否已发送过节日问候
"""
import json
import os
import threading
import time
from datetime import date

import config


class WorldClock:
    """世界时钟: 现实时间 + 节日日历 + 每日问候跟踪"""

    def __init__(self, state_file=None, calendar_file=None):
        self.state_file = state_file or config.WORLD_STATE_FILE
        self.calendar_file = calendar_file or config.CALENDAR_FILE
        self.lock = threading.Lock()

        self.special_dates = self._load_calendar()
        self.start_date = None   # 世界起始日期 "YYYY-MM-DD"
        self.last_date = None    # 上次同步到的日期
        self.days_elapsed = 0    # 从世界起始起经过的天数
        self.greeted = {}        # {agent_id: "YYYY-MM-DD"} 每个角色最近一次节日问候的日期
        self.events = []         # 事件时间轴
        self._event_id = 0

        self._load()
        self._sync_today()

    # ============================================================
    # 加载 / 持久化
    # ============================================================
    def _load_calendar(self):
        try:
            if os.path.exists(self.calendar_file):
                with open(self.calendar_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("special_dates", [])
        except Exception as e:
            print(f"[旁白] 特殊日期日历加载失败: {e}")
        return []

    def _load(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.start_date = state.get("start_date")
                self.last_date = state.get("last_date")
                self.days_elapsed = state.get("days_elapsed", 0)
                self.greeted = state.get("greeted", {})
                self.events = state.get("events", [])[-50:]
                self._event_id = max((e.get("id", 0) for e in self.events), default=0)
        except Exception as e:
            print(f"[旁白] 世界状态加载失败: {e}")

    def _save(self):
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "start_date": self.start_date,
                    "last_date": self.last_date,
                    "days_elapsed": self.days_elapsed,
                    "greeted": self.greeted,
                    "events": self.events[-50:],
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[旁白] 世界状态保存失败: {e}")

    # ============================================================
    # 现实时间
    # ============================================================
    @staticmethod
    def today():
        return date.today()

    @staticmethod
    def today_iso():
        return date.today().isoformat()

    @staticmethod
    def weekday_cn(d=None):
        d = d or date.today()
        return config.WEEKDAY_NAMES[d.weekday()]

    def _sync_today(self):
        """用现实日期同步世界: 首次运行记录起始日; 跨天则记录"新的一天"事件"""
        with self.lock:
            today = self.today_iso()
            if self.start_date is None:
                self.start_date = today
                self.last_date = today
                self.days_elapsed = 0
                self._add_event("world_start", f"【旁白】世界于 {today}（{self.weekday_cn()}）开启。")
                self._save()
                return
            if today != self.last_date:
                delta = (date.fromisoformat(today) - date.fromisoformat(self.last_date)).days
                self.days_elapsed += delta
                self.last_date = today
                self._add_event("new_day", f"【旁白】时间来到 {today}（{self.weekday_cn()}）。")
                self._save()

    # ============================================================
    # 事件
    # ============================================================
    def _add_event(self, kind, content):
        self._event_id += 1
        evt = {
            "id": self._event_id,
            "kind": kind,
            "date": self.today_iso(),
            "ts": time.time(),
            "content": content,
        }
        self.events.append(evt)
        if len(self.events) > 50:
            self.events = self.events[-50:]
        print(f"[旁白] {content}")
        return evt

    # ============================================================
    # 特殊日期
    # ============================================================
    def festivals_on(self, d=None):
        """返回某天的节日列表(元素含 name/description)"""
        d = d or self.today()
        return [f for f in self.special_dates
                if f.get("month") == d.month and f.get("day") == d.day]

    # ============================================================
    # 今日上下文(注入角色系统提示词)
    # ============================================================
    def today_context(self):
        """今日世界上下文文本, 注入角色系统提示词"""
        self._sync_today()
        d = self.today()
        festivals = [f["name"] for f in self.festivals_on(d)]
        lines = [f"今天是 {self.today_iso()}（{self.weekday_cn(d)}）。"]
        if festivals:
            lines.append("今天是" + "、".join(festivals) + "。")
        return "【今日世界】\n" + "\n".join(lines)

    # ============================================================
    # 每日特殊问候
    # ============================================================
    def greeting_hint(self, agent_id, special_names):
        """今天有特殊日期且该角色今天尚未问候过时, 返回问候提示文本并记录; 否则返回 None"""
        with self.lock:
            if not special_names:
                return None
            today = self.today_iso()
            if self.greeted.get(agent_id) == today:
                return None
            self.greeted[agent_id] = today
            self._save()
            return (f"【今日特别】今天是{'、'.join(special_names)}，"
                    f"请结合你的性格，自然地先送上节日祝福或问候，再回应对方。")

    # ============================================================
    # 世界快照(供前端轮询)
    # ============================================================
    def snapshot(self):
        self._sync_today()
        with self.lock:
            d = self.today()
            festivals = [f["name"] for f in self.festivals_on(d)]
            return {
                "today": self.today_iso(),
                "weekday": self.weekday_cn(d),
                "days_elapsed": self.days_elapsed,
                "festivals": festivals,
                "events": list(self.events),
            }

    def reset(self):
        """重置世界状态到初始(清空事件/问候/天数), 并重新记录起始日"""
        with self.lock:
            self.start_date = None
            self.last_date = None
            self.days_elapsed = 0
            self.greeted = {}
            self.events = []
            self._event_id = 0
            try:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
            except Exception as e:
                print(f"[旁白] 删除世界状态失败: {e}")
        self._sync_today()
