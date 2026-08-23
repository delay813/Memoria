"""
结构化事实库(语义记忆): 每个NPC独立, 从对话中抽取关于用户的稳定事实(身份/性格/偏好/约定/重要事件等)
- 与长期记忆(情景记忆)互补: 事实库存"稳定结论", 长期记忆存"具体事件"
- 检索: 轻量关键词重叠打分(事实量小, 无需额外向量库)
- 更新: 每次记忆压缩时由大模型整体合并去重, 自动完成新增/修正/淘汰
"""
import json
import os
import re
import time


class FactMemory:
    """单个NPC专属的结构化事实库(语义记忆)"""

    def __init__(self, file_path):
        self.file_path = file_path
        self._facts = self._load()

    # ============================================================
    # 存取
    # ============================================================
    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[事实库] 加载失败: {e}")
        return []

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[事实库] 保存失败: {e}")

    # ============================================================
    # 分词(英文单词 + 中文双字词, 用于轻量相关性打分)
    # ============================================================
    @staticmethod
    def _tokenize(text):
        text = (text or "").lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        # 中文双字词(去掉标点/空白后按相邻两字聚合)
        han = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        for i in range(len(han) - 1):
            tokens.add(han[i:i + 2])
        return tokens

    # ============================================================
    # 更新 / 检索
    # ============================================================
    def replace_all(self, statements):
        """用大模型抽取出的最新事实列表整体替换(已由大模型完成合并去重与淘汰)。"""
        now = time.time()
        updated = []
        for s in statements:
            s = str(s or "").strip()
            if not s:
                continue
            updated.append({
                "statement": s,
                "tags": [],
                "confidence": 0.8,
                "created_ts": now,
                "updated_ts": now,
            })
        self._facts = updated
        self._save()

    def all(self):
        return [f.get("statement", "") for f in self._facts]

    def retrieve(self, query, k=6):
        """按关键词重叠召回最相关事实; 事实很少时直接全量返回。"""
        statements = self.all()
        if not statements:
            return []
        if len(statements) <= k:
            return statements
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return statements[:k]
        scored = []
        for f in self._facts:
            s = f.get("statement", "")
            overlap = len(q_tokens & self._tokenize(s))
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return [s for _, s in scored[:k]]
        return statements[:k]

    def reset(self):
        """清空事实库"""
        self._facts = []
        self._save()
