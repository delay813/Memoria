"""FactMemory(结构化事实库)的单元测试, 不依赖网络与 API Key。

运行: pytest -q   (需先在项目根目录执行 pip install -r requirements-dev.txt)
"""
from fact_memory import FactMemory


def test_tokenize_english_and_chinese_bigram():
    tokens = FactMemory._tokenize("Hello 世界")
    assert "hello" in tokens
    assert "世界" in tokens
    assert "world" not in tokens


def test_replace_all_filters_empty_and_returns_all(tmp_path):
    fm = FactMemory(str(tmp_path / "facts.json"))
    fm.replace_all(["喜欢喝咖啡", "是程序员", ""])
    assert fm.all() == ["喜欢喝咖啡", "是程序员"]


def test_retrieve_returns_all_when_few(tmp_path):
    fm = FactMemory(str(tmp_path / "facts.json"))
    fm.replace_all(["喜欢喝咖啡", "养了一只猫"])
    # 事实条数 <= k 时直接全量返回
    assert fm.retrieve("随便问点什么", k=6) == ["喜欢喝咖啡", "养了一只猫"]


def test_retrieve_ranks_by_keyword_overlap(tmp_path):
    fm = FactMemory(str(tmp_path / "facts.json"))
    fm.replace_all(["喜欢喝咖啡", "喜欢跑步", "在杭州工作"])
    result = fm.retrieve("你喝咖啡吗", k=2)
    assert result[0] == "喜欢喝咖啡"


def test_reset_clears_all(tmp_path):
    fm = FactMemory(str(tmp_path / "facts.json"))
    fm.replace_all(["某条事实"])
    fm.reset()
    assert fm.all() == []
