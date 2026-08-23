"""
每个从Agent(NPC)独立的长期记忆: 基于Chroma的向量存储
检索策略: small-to-big + hybrid search(稠密向量 + BM25稀疏) + RRF融合 + rerank重排 + 四维加权
四维加权: 时近性(指数衰减) + 重要性(写入打分) + 访问频率(反复复习巩固) + 语义相似度
遗忘机制: 小模型定期评估, 低价值记忆做适当遗忘, 避免向量库只增不减
"""
import json
import math
import os
import re
import time
import uuid

from dashscope import Generation, TextReRank
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config


class SimpleBM25:
    """轻量级BM25稀疏检索(纯Python, 无额外依赖), 支持中英文混合"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.corpus = [self._tokenize(d.page_content) for d in docs]
        self.doc_len = [len(t) for t in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.n = len(self.corpus)
        self.df = {}
        for tokens in self.corpus:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", (text or "").lower())

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query, k=10):
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.corpus:
            return []
        scored = []
        for i, tokens in enumerate(self.corpus):
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for term in set(query_tokens):
                f = freq.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl) if self.avgdl else 1.0
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, self.docs[i]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]


class LongTermMemory:
    """单个NPC专属的长期记忆库(独立persist目录, 与其他NPC完全隔离)"""

    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.embedding = DashScopeEmbeddings(
            dashscope_api_key=config.API_KEY,
            model=config.EMBEDDING_MODEL,
        )
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name="longterm_memory",
        )
        # 记忆注册表: parent_id -> {text, created_ts, last_access_ts, importance, frequency}
        self.registry_file = os.path.join(persist_dir, "registry.json")
        self._registry = self._load_registry()

        # 小块(检索单位)的Document列表 + BM25稀疏索引
        self._docs = []
        self.bm25 = None
        self._rebuild_bm25()

    # ============================================================
    # 注册表(记忆元数据, 按父块=整条消息维度记录)
    # ============================================================
    def _load_registry(self):
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[长期记忆] 注册表加载失败: {e}")
        return {}

    def _save_registry(self):
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[长期记忆] 注册表保存失败: {e}")

    # ============================================================
    # 打分辅助
    # ============================================================
    # 情绪/重要事件关键词: 人脑对"有情绪、有意义"的事记得更牢, 命中则提升写入重要性
    _EMOTION_KEYWORDS = (
        "喜欢", "爱", "讨厌", "恨", "生气", "难过", "伤心", "开心", "激动", "害怕", "担心",
        "约定", "承诺", "秘密", "告白", "生日", "考试", "面试", "工作", "生病", "家人", "家里",
        "梦想", "分手", "吵架", "道歉", "结婚", "决定", "重要", "想你了", "好久不见",
    )

    @staticmethod
    def _heuristic_importance(text):
        """写入时的重要性打分(1-10, 启发式): 长度为基础, 命中情绪关键词额外加分"""
        text = str(text or "").strip()
        length = len(text)
        base = max(1, min(10, 3 + length // 15))
        bonus = config.MEMORY_EMOTION_BONUS if any(k in text for k in LongTermMemory._EMOTION_KEYWORDS) else 0
        return max(1, min(10, base + bonus))

    def _recency_score(self, entry, now):
        """时近性: 新记忆权重高, 随时间指数衰减"""
        age_days = (now - entry.get("created_ts", now)) / 86400.0
        return math.exp(-age_days / config.MEMORY_RECENCY_HALF_LIFE_DAYS)

    def _frequency_score(self, entry, now):
        """访问频率: 调取次数越多越高; 长期不访问权重下降"""
        freq = entry.get("frequency", 0)
        idle_days = (now - entry.get("last_access_ts", now)) / 86400.0
        freq_norm = min(1.0, freq / config.MEMORY_FREQ_NORMALIZE)
        idle_decay = math.exp(-idle_days / config.MEMORY_FREQ_HALF_LIFE_DAYS)
        return freq_norm * idle_decay

    def _call_small(self, prompt):
        """调用小模型(遗忘判断用), 带重试"""
        for attempt in range(config.MODEL_MAX_RETRIES + 1):
            try:
                resp = Generation.call(
                    api_key=config.API_KEY,
                    model=config.FORGET_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    result_format="message",
                )
                if resp.status_code == 200:
                    return resp
            except Exception as e:
                print(f"[长期记忆] 遗忘模型调用异常(第{attempt+1}次): {e}")
            if attempt < config.MODEL_MAX_RETRIES:
                time.sleep(config.MODEL_RETRY_DELAY)
        return None

    # ============================================================
    # 小块切分
    # ============================================================
    @staticmethod
    def _split_small(text):
        text = str(text or "").strip()
        if not text:
            return []
        if len(text) <= config.MEMORY_SMALL_CHUNK_SIZE:
            return [text]
        parts = re.split(r"(?<=[。！？!?\n])", text)
        chunks, buf = [], ""
        for p in parts:
            if not p.strip():
                continue
            if len(buf) + len(p) <= config.MEMORY_SMALL_CHUNK_SIZE:
                buf += p
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = p
        if buf.strip():
            chunks.append(buf.strip())
        return chunks or [text]

    # ============================================================
    # 写入
    # ============================================================
    def add(self, text, metadata=None):
        """写入单条记忆(内部走批量通道)"""
        self.add_many([text], [metadata])

    def add_many(self, texts, metadatas=None):
        """批量写入多条记忆(只重建一次BM25索引, 避免逐条写入的O(n^2)性能问题)。

        texts:     原始文本列表
        metadatas: 与 texts 等长的元数据字典列表(可为 None)
        """
        texts = list(texts or [])
        if not texts:
            return
        metadatas = list(metadatas or [{}] * len(texts))
        if len(metadatas) < len(texts):
            metadatas += [{}] * (len(texts) - len(metadatas))

        now = time.time()
        metas, ids, docs = [], [], []
        for text, meta in zip(texts, metadatas):
            text = str(text or "").strip()
            if not text:
                continue
            meta = dict(meta or {})
            parent_id = uuid.uuid4().hex
            smalls = self._split_small(text)
            for i, s in enumerate(smalls):
                m = dict(meta)
                m["parent_id"] = parent_id
                m["parent_text"] = text
                m["parent_seq"] = i
                metas.append(m)
                ids.append(f"{parent_id}:{i}")
                docs.append(Document(page_content=s, metadata=m))

            self._registry[parent_id] = {
                "text": text,
                "created_ts": now,
                "last_access_ts": now,
                "importance": self._heuristic_importance(text),
                "frequency": 0,
                "status": "active",
            }

        if not ids:
            return
        try:
            self.store.add_texts([d.page_content for d in docs], metadatas=metas, ids=ids)
            self._docs.extend(docs)
            self._build_bm25()
            self._save_registry()
        except Exception as e:
            print(f"[长期记忆] 批量写入失败: {e}")

    # ============================================================
    # BM25 索引
    # ============================================================
    def _rebuild_bm25(self):
        try:
            data = self.store.get(include=["documents", "metadatas"])
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            self._docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metas)
            ]
        except Exception as e:
            print(f"[长期记忆] BM25重建失败, 仅用向量检索: {e}")
            self._docs = []
        self._build_bm25()

    def _build_bm25(self):
        try:
            self.bm25 = SimpleBM25(self._docs) if self._docs else None
        except Exception as e:
            print(f"[长期记忆] BM25构建失败: {e}")
            self.bm25 = None

    def reset(self):
        """清空长期记忆(向量库 + 注册表 + 内存索引)"""
        try:
            ids = self.store.get()["ids"]
            if ids:
                self.store.delete(ids=ids)
        except Exception as e:
            print(f"[长期记忆] 清空失败: {e}")
        self._docs = []
        self.bm25 = None
        self._registry = {}
        self._save_registry()

    # ============================================================
    # 检索: hybrid -> RRF -> small-to-big -> rerank -> 四维加权
    # ============================================================
    def recall(self, query, k=None):
        k = k or config.MEMORY_RECALL_K
        try:
            fused = self._hybrid_search(query, top_n=config.MEMORY_HYBRID_TOPN)
            parents = self._group_to_parents(fused, limit=config.MEMORY_RERANK_TOPN)  # [(pid, text, rrf)]
            if not parents:
                return []
            text_to_parent = {p[1]: p for p in parents}
            texts = [p[1] for p in parents]
            reranked, sim_scores = self._rerank(query, texts, top_n=len(texts))

            now = time.time()
            weighted = []
            for i, text in enumerate(reranked):
                p = text_to_parent.get(text)
                if not p:
                    continue
                pid = p[0]
                entry = self._registry.get(pid, {})
                # 软遗忘: 已"想不起来"的记忆不注入召回
                if entry.get("status") == "forgotten":
                    continue
                raw_sim = sim_scores[i] if sim_scores and sim_scores[i] is not None else None
                # 相关性过低(低于阈值)的记忆不注入上下文
                if raw_sim is not None and raw_sim < config.MEMORY_RECALL_THRESHOLD:
                    continue
                sim = raw_sim if raw_sim is not None else 0.5
                rec = self._recency_score(entry, now)
                imp = entry.get("importance", 5) / 10.0
                freq = self._frequency_score(entry, now)
                total = (config.MEMORY_W_RECENCY * rec
                         + config.MEMORY_W_IMPORTANCE * imp
                         + config.MEMORY_W_FREQUENCY * freq
                         + config.MEMORY_W_SIMILARITY * sim)
                age_days = (now - entry.get("created_ts", now)) / 86400.0
                weighted.append((total, pid, text, age_days))

            weighted.sort(key=lambda x: x[0], reverse=True)
            top = weighted[:k]

            # 访问强化: 被调取的记忆 frequency+1, 更新 last_access
            for _, pid, _, _ in top:
                if pid in self._registry:
                    self._registry[pid]["frequency"] = self._registry[pid].get("frequency", 0) + 1
                    self._registry[pid]["last_access_ts"] = now
            self._save_registry()

            # 附上时间标签, 让角色能自然说出"你上次……/那天……"
            return [self._format_memory(text, age) for _, _, text, age in top]
        except Exception as e:
            print(f"[长期记忆] 召回失败: {e}")
            return []

    @staticmethod
    def _format_memory(text, age_days):
        """给久远记忆加时间标签(近1天内不加), 贴合人脑情景记忆的"何时"维度"""
        if age_days and age_days >= 1:
            days = int(round(age_days))
            return f"[约{days}天前] {text}"
        return text

    def _hybrid_search(self, query, top_n=10):
        vec_docs = self.store.similarity_search(query, k=top_n)
        bm_docs = self.bm25.search(query, k=top_n) if self.bm25 else []
        return self._rrf([vec_docs, bm_docs])

    @staticmethod
    def _rrf(result_lists, rrf_k=60):
        scores, doc_map = {}, {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map.setdefault(key, doc)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(doc_map[key], score) for key, score in ranked]

    def _group_to_parents(self, fused, limit=9):
        parent_scores, parent_texts = {}, {}
        for doc, score in fused:
            pid = doc.metadata.get("parent_id")
            if pid is None:
                pid = doc.page_content
                parent_text = doc.page_content
            else:
                parent_text = doc.metadata.get("parent_text") or doc.page_content
            parent_scores[pid] = max(parent_scores.get(pid, 0.0), score)
            parent_texts[pid] = parent_text
        ranked = sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(pid, parent_texts[pid], score) for pid, score in ranked]

    def _rerank(self, query, candidates, top_n=3):
        """重排模型对候选父块精排; 返回(文本列表, 分数列表)。失败时分数为None"""
        if not candidates:
            return [], []
        try:
            resp = TextReRank.call(
                model=config.RERANK_MODEL,
                api_key=config.API_KEY,
                query=query,
                documents=candidates,
                top_n=top_n,
            )
            if resp.status_code == 200 and getattr(resp, "output", None) and resp.output.results:
                results = sorted(resp.output.results, key=lambda r: r.relevance_score, reverse=True)
                reranked, scores, seen = [], [], set()
                for r in results:
                    if r.index < len(candidates) and r.index not in seen:
                        reranked.append(candidates[r.index])
                        scores.append(r.relevance_score)
                        seen.add(r.index)
                return reranked, scores
            print(f"[长期记忆] 重排失败(status={resp.status_code}), 退回聚合排序")
        except Exception as e:
            print(f"[长期记忆] 重排异常: {e}, 退回聚合排序")
        n = min(top_n, len(candidates))
        return candidates[:n], [None] * n

    # ============================================================
    # 遗忘机制(小模型驱动)
    # ============================================================
    def forget(self, agent_name=""):
        """评估候选记忆, 低价值则遗忘(删除), 避免向量库只增不减"""
        now = time.time()
        candidates = []
        for pid, entry in list(self._registry.items()):
            if entry.get("status") == "forgotten":
                continue  # 已软遗忘的不再重复评估
            age_days = (now - entry.get("created_ts", now)) / 86400.0
            rec = self._recency_score(entry, now)
            freq = entry.get("frequency", 0)
            if (age_days >= config.FORGET_MIN_AGE_DAYS
                    and rec < config.FORGET_RECENCY_THRESHOLD
                    and freq < config.FORGET_FREQ_THRESHOLD):
                candidates.append((pid, entry))
        if not candidates:
            return 0

        batch = candidates[:config.FORGET_BATCH_SIZE]
        items = [{
            "id": pid,
            "text": (entry.get("text") or "")[:200],
            "importance": entry.get("importance", 5),
            "age_days": round((now - entry.get("created_ts", now)) / 86400.0, 1),
            "frequency": entry.get("frequency", 0),
        } for pid, entry in batch]

        prompt = f"""你是「{agent_name}」的记忆管理助手。下面是候选遗忘的记忆(含编号/内容/当前重要性/年龄天数/访问次数)。请判断每条是否该遗忘：
- 琐碎、寒暄、过时、重复、对未来对话无长期价值的 → "forget"
- 关于用户的关键信息、重要事件、稳定偏好、有长期价值的 → "keep"

严格输出JSON数组(不要任何其他文字):
[{{"id":"...","decision":"forget"或"keep","importance":1到10的整数}}]

【候选记忆】
{json.dumps(items, ensure_ascii=False)}
"""
        resp = self._call_small(prompt)
        if not resp:
            return 0
        content = resp.output.choices[0].message.content
        content = re.sub(r"```(?:json)?", "", content)
        match = re.search(r"\[.*\]", content, re.DOTALL)
        try:
            decisions = json.loads(match.group(0)) if match else []
        except Exception:
            decisions = []

        forgotten = 0
        for d in decisions:
            pid = d.get("id")
            if pid not in self._registry:
                continue
            if d.get("decision") == "forget":
                entry = self._registry[pid]
                # 软遗忘: 先标记"想不起来"(不注入召回), 而非物理删除
                entry["status"] = "forgotten"
                entry["importance"] = max(1, min(10, (entry.get("importance", 5) or 5) - 4))
                # 重要度已降到谷底, 才物理删除
                if entry["importance"] <= 1:
                    try:
                        self.store.delete(where={"parent_id": pid})
                    except Exception as e:
                        print(f"[长期记忆] 删除记忆 {pid} 失败: {e}")
                    self._registry.pop(pid, None)
                forgotten += 1
            else:
                try:
                    imp = int(d.get("importance", 5) or 5)
                    self._registry[pid]["importance"] = max(1, min(10, imp))
                    self._registry[pid]["status"] = "active"
                except Exception:
                    pass

        self._save_registry()
        if forgotten:
            self._rebuild_bm25()
            print(f"[长期记忆] {agent_name} 遗忘了 {forgotten} 条低价值记忆")
        return forgotten
