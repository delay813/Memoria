"""LongTermMemory 去重纯逻辑的单元测试, 不依赖网络。

运行: pytest -q   (需先在项目根目录执行 pip install -r requirements-dev.txt)
"""
import os

# 在 import longterm_memory 前注入占位 Key, 避免 config.py 因缺 Key 而抛错
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-ci")

import config
from longterm_memory import LongTermMemory


def test_cosine_identical_is_one():
    a = [1.0, 2.0, 3.0]
    assert abs(LongTermMemory._cosine(a, a) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert LongTermMemory._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_opposite_is_minus_one():
    assert abs(LongTermMemory._cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9


def test_cosine_mismatched_length_returns_zero():
    assert LongTermMemory._cosine([1.0, 2.0], [1.0]) == 0.0


def test_classify_by_score_bands(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DEDUP_HIGH", 0.9)
    monkeypatch.setattr(config, "MEMORY_DEDUP_LOW", 0.5)
    assert LongTermMemory._classify_by_score(0.95) == "duplicate"
    assert LongTermMemory._classify_by_score(0.30) == "new"
    assert LongTermMemory._classify_by_score(0.70) == "gray"
