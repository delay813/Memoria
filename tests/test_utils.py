"""公共工具(text_utils / json_utils / time_utils)的单元测试, 不依赖网络与 API Key。"""
from datetime import date

from json_utils import parse_array, parse_object
from text_utils import plain_text, tokenize, tokenize_set, worth_remembering
from time_utils import days_since, in_interval, today_iso


# ---------- text_utils ----------
def test_tokenize_english_and_chinese_bigram():
    tokens = tokenize("Hello 世界")
    assert "hello" in tokens
    assert "世界" in tokens
    assert "world" not in tokens


def test_tokenize_set_is_set():
    assert tokenize_set("你好") == {"你好"}


def test_plain_text_strips_markdown():
    assert plain_text("## 标题\n**加粗**\n- 列表项") == "标题\n加粗\n列表项"


def test_worth_remembering_filters_filler_and_short():
    assert not worth_remembering("嗯")
    assert not worth_remembering("你好")
    assert worth_remembering("我今天面试通过了")


# ---------- json_utils ----------
def test_parse_object_extracts_json():
    assert parse_object('```json\n{"delta": 3}\n```') == {"delta": 3}


def test_parse_object_fails_safe():
    assert parse_object("不是JSON") == {}


def test_parse_array_extracts_json():
    assert parse_array('```json\n["a", "b"]\n```') == ["a", "b"]


def test_parse_array_fails_safe():
    assert parse_array("没有数组") == []


# ---------- time_utils ----------
def test_in_interval_plain():
    assert in_interval(9, 8, 12)
    assert not in_interval(12, 8, 12)
    assert not in_interval(7, 8, 12)


def test_in_interval_wraps_midnight():
    assert in_interval(23, 22, 6)
    assert in_interval(3, 22, 6)
    assert not in_interval(12, 22, 6)


def test_days_since():
    assert days_since(today_iso()) == 0
    assert days_since("") == 0
    assert days_since("bad-date") == 0


def test_days_since_with_ref():
    assert days_since("2026-01-01", ref=date(2026, 1, 4)) == 3
