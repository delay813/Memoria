"""
共享时间工具。

- today / today_iso: 现实日期(现实一天 = 世界一天)
- days_since: 距某日期过去的天数(集中了散落各处的 date.fromisoformat 计算)
- in_interval: 跨午夜区间判断(world_sim 的作息槽 与 narrator 的睡眠窗口共用)

集中后方便将来替换为"可注入时钟"(测试无需 monkeypatch date.today)。
"""
from datetime import date


def today():
    """现实今天(本地时区)。"""
    return date.today()


def today_iso():
    """现实今天的 ISO 字符串(YYYY-MM-DD)。"""
    return date.today().isoformat()


def days_since(iso, ref=None):
    """距 ref(默认今天)过去的天数; iso 为空或解析失败返回 0。"""
    if not iso:
        return 0
    try:
        return ((ref or date.today()) - date.fromisoformat(iso)).days
    except Exception:
        return 0


def in_interval(hour, start, end):
    """判断 hour(0~24, 可含小数)是否落在 [start, end); end<=start 表示跨午夜。"""
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
