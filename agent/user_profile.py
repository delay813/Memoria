"""
单用户档案: 成就系统(全局, 跨角色)
- 成就: 跨角色的收集/图鉴, 解锁后持久化
"""
import json
import os
import threading
import time


# 成就定义(按展示顺序)
ACHIEVEMENTS = [
    {"key": "first_chat", "emoji": "👋", "label": "初次相遇", "desc": "第一次和角色聊天"},
    {"key": "three_agents", "emoji": "🎭", "label": "广结缘", "desc": "和全部角色都聊过天"},
    {"key": "favor_45", "emoji": "💛", "label": "渐渐熟悉", "desc": "有角色好感度达到 45"},
    {"key": "favor_70", "emoji": "💖", "label": "亲密无间", "desc": "有角色好感度达到 70"},
    {"key": "first_proactive", "emoji": "📮", "label": "被惦记", "desc": "第一次收到角色的主动消息"},
    {"key": "first_dream", "emoji": "🌙", "label": "入梦", "desc": "第一次触发角色的隔夜整理(梦境日记)"},
    {"key": "life_event", "emoji": "🍃", "label": "她的日常", "desc": "见证一位角色生活中的随机事件"},
]


class UserProfile:
    """单用户档案: 昵称 + 成就, 持久化到 JSON 文件"""

    def __init__(self, file_path):
        self.file_path = file_path
        # 成就解锁可能从后台线程回调(_run_compact/_run_dream/sync_life 等), 防并发写坏文件
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[用户档案] 加载失败: {e}")
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            # 原子写: 先写临时文件再 rename, 防止并发写导致 JSON 截断/损坏
            tmp = self.file_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.file_path)
        except Exception as e:
            print(f"[用户档案] 保存失败: {e}")

    # ---------- 成就 ----------
    def unlock(self, key):
        """解锁成就; 返回是否为新解锁(便于前端即时提示)。线程安全。"""
        with self._lock:
            ach = self._data.setdefault("achievements", {})
            if ach.get(key):
                return False
            ach[key] = time.time()
            self._save()
            return True

    def unlocked(self, key):
        return bool(self._data.get("achievements", {}).get(key))

    def all_achievements(self):
        """返回全部成就及解锁状态(供前端成就墙展示)"""
        ach = self._data.get("achievements", {})
        result = []
        for a in ACHIEVEMENTS:
            ts = ach.get(a["key"])
            result.append({
                "key": a["key"],
                "emoji": a["emoji"],
                "label": a["label"],
                "desc": a["desc"],
                "unlocked": bool(ts),
                "unlocked_at": ts,
            })
        return result

    def find_achievement(self, key):
        for a in ACHIEVEMENTS:
            if a["key"] == key:
                return a
        return None

    # ---------- 用户昵称(称呼随关系演进用) ----------
    def get_nickname(self):
        """返回用户昵称; 未设置则空串。"""
        return str(self._data.get("nickname", "") or "").strip()

    def set_nickname(self, nickname):
        """设置用户昵称(限长 20 字), 返回最终昵称。线程安全。"""
        with self._lock:
            nickname = str(nickname or "").strip()[:20]
            self._data["nickname"] = nickname
            self._save()
            return nickname
