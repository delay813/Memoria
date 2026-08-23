"""world_sim(世界模拟: 作息 + 随机事件)纯逻辑单元测试, 不依赖网络。

运行: pytest -q   (需先在项目根目录执行 pip install -r requirements-dev.txt)
"""
import os
import random
from datetime import datetime

# 在 import 前注入占位 Key, 避免 config.py 因缺 Key 而抛错
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-ci")

import config
from world_sim import LifeSim, _in_slot, format_schedule, pick_random_event, resolve_schedule


def test_in_slot_plain():
    assert _in_slot(9, 8, 12)
    assert not _in_slot(12, 8, 12)  # 右端点不含
    assert not _in_slot(7, 8, 12)


def test_in_slot_wraps_midnight():
    assert _in_slot(23, 22, 6)
    assert _in_slot(3, 22, 6)
    assert not _in_slot(12, 22, 6)


def test_resolve_schedule_by_hour():
    assert resolve_schedule("npc_01", datetime(2026, 1, 1, 10, 0))["label"] == "上课中"
    assert resolve_schedule("npc_01", datetime(2026, 1, 1, 23, 0))["label"] == "睡觉中"
    assert resolve_schedule("npc_01", datetime(2026, 1, 1, 20, 0))["label"] == "空闲"


def test_resolve_schedule_unknown_agent_uses_default():
    assert resolve_schedule("unknown", datetime(2026, 1, 1, 9, 0))["label"] == "忙碌中"


def test_advance_idempotent_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 1.0)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    rng = random.Random(42)
    first = life.advance(today="2026-01-05", rng=rng)
    second = life.advance(today="2026-01-05", rng=rng)
    assert len(first) == 1
    assert second == []
    assert life.last_sim_date == "2026-01-05"


def test_advance_backfills_missed_days(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 1.0)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    rng = random.Random(7)
    life.advance(today="2026-01-01", rng=rng)
    events = life.advance(today="2026-01-03", rng=rng)
    assert {e["date"] for e in events} == {"2026-01-02", "2026-01-03"}


def test_advance_backfill_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 1.0)
    monkeypatch.setattr(config, "LIFE_BACKFILL_MAX_DAYS", 3)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    rng = random.Random(3)
    life.advance(today="2026-01-01", rng=rng)
    events = life.advance(today="2026-01-10", rng=rng)
    assert {e["date"] for e in events} == {"2026-01-08", "2026-01-09", "2026-01-10"}


def test_no_event_when_probability_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 0.0)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    assert life.advance(today="2026-01-05") == []
    assert life.today_event("2026-01-05") is None


def test_pick_random_event_valid():
    ev = pick_random_event("npc_01", rng=random.Random(1))
    assert isinstance(ev, dict)
    assert ev.get("text")
    assert isinstance(ev.get("mood"), int)


def test_reset_clears_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 1.0)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    life.advance(today="2026-01-05", rng=random.Random(5))
    life.reset()
    assert life.last_sim_date is None
    assert life.daily_events == []


def test_format_schedule_marks_current():
    sched = format_schedule("npc_01", datetime(2026, 1, 1, 10, 0))
    current = [s for s in sched if s["current"]]
    assert len(current) == 1
    assert current[0]["label"] == "上课中"
    assert all("time" in s for s in sched)


def test_followup_scheduled_and_triggered(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RANDOM_EVENT_PROBABILITY", 0.0)
    life = LifeSim("npc_01", str(tmp_path / "life.json"))
    ev = {"key": "x", "text": "起因", "mood": -1, "kind": "低落",
          "followup": {"key": "y", "delay_days": 2, "text": "后续", "mood": 1, "kind": "开心"}}
    life._emit_event("2026-01-01", ev)
    assert len(life._pending_followups) == 1
    assert life._pending_followups[0]["date"] == "2026-01-03"
    life.last_sim_date = "2026-01-02"
    events = life.advance(today="2026-01-03", rng=random.Random(1))
    assert any(e["key"] == "y" for e in events)
    assert life._pending_followups == []
