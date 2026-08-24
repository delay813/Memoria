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

from dashscope import TextReRank
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

import config
import json_utils
import llm
import storage
import text_utils


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
        """英文/数字按单词, 中文按双字词(bigram)切分; 复用 text_utils。"""
        return text_utils.tokenize(text)

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
        # topic -> 父块id集合 的反查索引(话题联想扩展用)
        self._topic_index = {}
        self._last_registry_save = 0.0
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
            storage.save_json(self.registry_file, self._registry)
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
        """批量写入多条记忆, 并做去重(避免同一件事反复追加、堆积重复)。

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
        # 1) 预处理: 清理空文本 + 提取话题标签
        cleaned = []
        for text, meta in zip(texts, metadatas):
            text = str(text or "").strip()
            if not text:
                continue
            meta = dict(meta or {})
            # 话题标签: 从meta中取出并只存进registry(不放进Chroma metadata, 避免list类型报错)
            topics_raw = meta.pop("topics", None) or []
            if isinstance(topics_raw, str):
                topics_raw = [topics_raw]
            topics = [str(t).strip() for t in topics_raw if str(t).strip()]
            cleaned.append({"text": text, "meta": meta, "topics": topics})
        if not cleaned:
            return

        # 2) 去重分流: 每条 → new(新增) / duplicate(强化旧记忆) / merge(保历史合并)
        new_items, merges, reinforce = self._dedupe(cleaned, now)

        # 3) 强化(duplicate): 只更新旧记忆元数据, 不写新块、不覆盖旧文本
        for pid in reinforce:
            entry = self._registry.get(pid)
            if entry:
                entry["frequency"] = entry.get("frequency", 0) + 1
                entry["last_access_ts"] = now

        # 4) 合并(merge): 保历史式合并, 更新旧父块文本(旧事实+新事实都保留)
        for pid, merged_text, merged_topics in merges:
            self._merge_into(pid, merged_text, merged_topics, now)

        # 5) 新增(new): 批量写入
        self._write_new(new_items, now)

        # 6) 统一持久化 + 重建索引(BM25 + topic反查)
        self._save_registry()
        self._rebuild_bm25()

    # ============================================================
    # 记忆去重(写入时): embedding 相似度分流 + 保历史合并
    # ============================================================
    @staticmethod
    def _cosine(a, b):
        """两个向量的余弦相似度(纯Python, 零依赖)"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    @staticmethod
    def _classify_by_score(score):
        """按相似度分档: duplicate(几乎复述) / new(明显新事) / gray(需小模型判断)"""
        if score >= config.MEMORY_DEDUP_HIGH:
            return "duplicate"
        if score <= config.MEMORY_DEDUP_LOW:
            return "new"
        return "gray"

    def _write_chunks(self, parent_id, text, base_meta=None):
        """把一段父块文本切小块写入Chroma, 返回是否成功"""
        base_meta = base_meta or {}
        smalls = self._split_small(text)
        metas, ids, docs = [], [], []
        for i, s in enumerate(smalls):
            m = dict(base_meta)
            m["parent_id"] = parent_id
            m["parent_text"] = text
            m["parent_seq"] = i
            metas.append(m)
            ids.append(f"{parent_id}:{i}")
            docs.append(Document(page_content=s, metadata=m))
        try:
            self.store.add_texts([d.page_content for d in docs], metadatas=metas, ids=ids)
            return True
        except Exception as e:
            print(f"[长期记忆] 写入父块 {parent_id} 失败: {e}")
            return False

    def _best_parent_match(self, emb):
        """用向量查最相似的小块并归组到父块, 返回 (parent_id, parent_text, cosine) 或 None"""
        try:
            hits = self.store.similarity_search_by_vector(emb, k=config.MEMORY_DEDUP_TOPK)
        except Exception as e:
            print(f"[长期记忆] 去重检索失败: {e}")
            return None
        if not hits:
            return None
        try:
            embeds = self.embedding.embed_documents([h.page_content for h in hits])
        except Exception as e:
            print(f"[长期记忆] 去重候选嵌入失败: {e}")
            return None
        best_pid = best_text = None
        best_score = -1.0
        for doc, e in zip(hits, embeds):
            pid = doc.metadata.get("parent_id")
            if not pid:
                continue
            score = self._cosine(emb, e)
            if score > best_score:
                best_pid = pid
                best_text = doc.metadata.get("parent_text") or doc.page_content
                best_score = score
        if best_pid is None:
            return None
        return best_pid, best_text, best_score

    def _llm_dedup_judge(self, new_text, old_text):
        """灰区判断: 让小模型输出 duplicate / merge / new。失败安全降级为 new(不丢信息)"""
        prompt = f"""你是记忆去重助手。判断【新记忆】与【已有记忆】的关系, 只输出一个词:
- duplicate: 新记忆是已有记忆的纯重复/重申, 没有任何新信息
- merge: 是同一件事, 但新记忆带来新进展/新细节(如时间、地点、状态、心情的变化), 需要合并
- new: 是不同的事, 不应合并

特别注意: 涉及时间/地点/状态等"变化"的(如"昨天在上海, 今天在北京"、"之前喜欢、现在不喜欢"), 即使话题相似也必须是 merge 或 new, 绝不能当作 duplicate 把旧信息丢掉。

【已有记忆】
{old_text}

【新记忆】
{new_text}

只输出一个词:"""
        resp = llm.call([{"role": "user", "content": prompt}], model=config.FORGET_MODEL)
        if not resp:
            return "new"
        content = resp.output.choices[0].message.content.strip().lower()
        if "merge" in content:
            return "merge"
        if "duplicate" in content:
            return "duplicate"
        return "new"

    def _llm_merge(self, new_text, old_text):
        """生成保历史合并文本: 旧事实+新事实都保留。失败返回 None(降级为新增)"""
        prompt = f"""你是记忆合并助手。把【已有记忆】和【新记忆】合并成一条完整记忆。硬性要求:
1. 必须保留【已有记忆】里的全部历史事实, 尤其是"过去曾处于/曾发生"的信息(如"曾在上海")。
2. 若是同一属性的变化(地点/工作/状态/喜好等), 用"之前…后来…/现在…"把新旧都写出来, 严禁用新信息覆盖旧信息。
3. 若是补充细节, 把细节自然并入。
4. 输出一条通顺的合并记忆, 80字以内, 不要任何前缀或解释。

【已有记忆】
{old_text}

【新记忆】
{new_text}"""
        resp = llm.call([{"role": "user", "content": prompt}], model=config.FORGET_MODEL)
        if not resp:
            return None
        return resp.output.choices[0].message.content.strip()

    def _dedupe(self, cleaned, now):
        """去重分流。返回 (new_items, merges, reinforce)。
        new_items: 要新增的项; merges: [(pid, merged_text, merged_topics)]; reinforce: 要强化的pid集合。
        去重失败/关闭时安全降级为"全部新增"(绝不因去重丢信息)。"""
        if not config.MEMORY_DEDUP_ENABLED or not self._registry:
            return cleaned, [], set()

        new_items, merges, reinforce = [], [], set()
        try:
            embeds = self.embedding.embed_documents([c["text"] for c in cleaned])
        except Exception as e:
            print(f"[长期记忆] 去重embedding失败, 退化为全部新增: {e}")
            return cleaned, [], set()

        for item, emb in zip(cleaned, embeds):
            match = self._best_parent_match(emb)
            if match is None:
                new_items.append(item)
                continue
            pid, old_text, score = match
            cls = self._classify_by_score(score)
            if cls == "new":
                new_items.append(item)
                continue
            if cls == "duplicate":
                # 高分复述: 直接强化旧记忆(不新增)
                reinforce.add(pid)
                continue
            # 灰区: 小模型判断
            verdict = self._llm_dedup_judge(item["text"], old_text)
            if verdict == "duplicate":
                reinforce.add(pid)
            elif verdict == "merge":
                merged = self._llm_merge(item["text"], old_text)
                if merged:
                    merges.append((pid, merged, item["topics"]))
                else:
                    new_items.append(item)   # 合并失败 → 安全降级为新增(保留旧记忆原样)
            else:
                new_items.append(item)

        return new_items, merges, reinforce

    def _write_new(self, items, now):
        """批量写入"新"记忆(小块切分 + Chroma写入 + 注册表)"""
        for item in items:
            text, meta, topics = item["text"], item["meta"], item["topics"]
            parent_id = uuid.uuid4().hex
            if not self._write_chunks(parent_id, text, base_meta=meta):
                continue   # 写入失败则跳过(不写注册表, 保持一致性)
            self._registry[parent_id] = {
                "text": text,
                "created_ts": now,
                "last_access_ts": now,
                "importance": self._heuristic_importance(text),
                "frequency": 0,
                "status": "active",
                "topics": topics,
            }

    def _merge_into(self, pid, merged_text, merged_topics, now):
        """保历史合并: 删除旧父块的小块, 写入合并文本(同parent_id), 更新注册表(保留created_ts)"""
        entry = self._registry.get(pid)
        if not entry:
            return
        old_text = entry.get("text", "")
        try:
            self.store.delete(where={"parent_id": pid})
        except Exception as e:
            print(f"[长期记忆] 合并删除旧块失败 {pid}: {e}")
            return
        if not self._write_chunks(pid, merged_text):
            # 写入失败则回滚旧文本, 避免记忆丢失
            print(f"[长期记忆] 合并写入失败, 回滚旧文本 {pid}")
            self._write_chunks(pid, old_text)
            return
        entry["text"] = merged_text
        entry["importance"] = max(entry.get("importance", 5), self._heuristic_importance(merged_text))
        entry["frequency"] = entry.get("frequency", 0) + 1
        entry["last_access_ts"] = now
        entry["updated_ts"] = now
        entry["topics"] = list(dict.fromkeys((entry.get("topics", []) or []) + list(merged_topics)))

    # ============================================================
    # BM25 索引
    # ============================================================
    def _rebuild_bm25(self):
        try:
            data = self.store.get(include=["documents", "metadatas"])
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            docs = []
            for t, m in zip(texts, metas):
                meta = m or {}
                pid = meta.get("parent_id")
                # 软遗忘(想不起来)的记忆不进入稀疏索引, 减少检索噪声(与召回层过滤一致)
                if pid and self._registry.get(pid, {}).get("status") == "forgotten":
                    continue
                docs.append(Document(page_content=t, metadata=meta))
            self._docs = docs
        except Exception as e:
            print(f"[长期记忆] BM25重建失败, 仅用向量检索: {e}")
            self._docs = []
        self._build_bm25()
        self._rebuild_topic_index()

    def _build_bm25(self):
        try:
            self.bm25 = SimpleBM25(self._docs) if self._docs else None
        except Exception as e:
            print(f"[长期记忆] BM25构建失败: {e}")
            self.bm25 = None

    def _rebuild_topic_index(self):
        """topic -> 父块id集合 的反查索引, 从registry构建(软遗忘的记忆不参与联想)"""
        self._topic_index = {}
        for pid, entry in self._registry.items():
            if entry.get("status") == "forgotten":
                continue
            for t in entry.get("topics", []):
                if t:
                    self._topic_index.setdefault(t, set()).add(pid)

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
        self._topic_index = {}
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
            # 话题联想扩展: 与已召回父块共享话题的其他父块加入候选(多跳式联想)
            parents, topic_hits = self._expand_by_topic(parents, limit=config.MEMORY_TOPIC_EXPAND)
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
                # 相关性过低(低于阈值)的记忆不注入上下文; 但话题命中的记忆给保底分, 不被误过滤
                if raw_sim is not None and raw_sim < config.MEMORY_RECALL_THRESHOLD:
                    if pid in topic_hits:
                        raw_sim = config.MEMORY_TOPIC_SIM_FLOOR
                    else:
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
            # 访问计数节流落盘: 不必每轮对话都写一次注册表(避免频繁磁盘写)
            if now - self._last_registry_save >= config.MEMORY_ACCESS_SAVE_INTERVAL:
                self._save_registry()
                self._last_registry_save = now

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

    def _expand_by_topic(self, parents, limit=3):
        """话题联想扩展: 对已召回的父块, 用其话题标签反查同话题的其他父块。
        返回 (扩展后的父块列表, 被扩展命中的父块id集合)。扩展只扩大候选池, 不增加最终注入条数。"""
        if not self._topic_index:
            return parents, set()
        seen = {p[0] for p in parents}
        hit = set()
        extra = []
        for pid, _text, _score in parents:
            for t in self._registry.get(pid, {}).get("topics", []):
                for nb in self._topic_index.get(t, ()):
                    if nb in seen or nb not in self._registry:
                        continue
                    if self._registry[nb].get("status") == "forgotten":
                        continue
                    seen.add(nb)
                    hit.add(nb)
                    extra.append((nb, self._registry[nb].get("text", ""), 0.0))
                    if len(extra) >= limit:
                        return parents + extra, hit
                if len(extra) >= limit:
                    return parents + extra, hit
        return parents + extra, hit

    def _rerank(self, query, candidates, top_n=3):
        """重排模型对候选父块精排; 返回(文本列表, 分数列表)。失败时分数为None"""
        if not candidates:
            return [], []
        # 省流开关: 关闭重排时跳过 rerank API 调用, 分数置 None(下游退化为统一相似度),
        # 由时近性/重要性/访问频率三维继续排序, 与"重排失败"的既有降级路径一致。
        if not config.MEMORY_RERANK_ENABLED:
            n = min(top_n, len(candidates))
            return candidates[:n], [None] * n
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
    def _mark_forgotten(self, pid, penalty=4, hard_delete=True):
        """软遗忘的统一入口: 标记 status=forgotten 并降低重要度。

        重要度降到谷底(<=1)时, 视 hard_delete 决定是否物理删除。
        forget(低价值遗忘)与 suppress_conflicts(事实冲突)共用此逻辑。
        返回是否真正执行了遗忘(已遗忘/不存在则 False)。
        """
        entry = self._registry.get(pid)
        if not entry:
            return False
        if entry.get("status") == "forgotten":
            return False
        entry["status"] = "forgotten"
        entry["importance"] = max(1, min(10, (entry.get("importance", 5) or 5) - penalty))
        if hard_delete and entry["importance"] <= 1:
            try:
                self.store.delete(where={"parent_id": pid})
            except Exception as e:
                print(f"[长期记忆] 删除记忆 {pid} 失败: {e}")
            self._registry.pop(pid, None)
        return True

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
        resp = llm.call([{"role": "user", "content": prompt}], model=config.FORGET_MODEL)
        if not resp:
            return 0
        content = resp.output.choices[0].message.content
        decisions = json_utils.parse_array(content)

        forgotten = 0
        for d in decisions:
            pid = d.get("id")
            if pid not in self._registry:
                continue
            if d.get("decision") == "forget":
                if self._mark_forgotten(pid, penalty=4):
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

    def suppress_conflicts(self, statements, top_k=3):
        """跨记忆一致性: 把"已被纠正/过时"的旧事实对应的情景记忆做软遗忘。

        facts.json 是权威的"当前结论"; 当一条事实被更新/淘汰时, 语义检索出与其相关的旧情景记忆,
        并标记为 forgotten(不再注入召回), 让事实库优先于向量库, 避免角色"旧事重提"自相矛盾。
        仅对语义相似度超过阈值的记忆软遗忘(不物理删除), 降低误伤。返回软遗忘的记忆条数。
        """
        cleaned = [str(s or "").strip() for s in (statements or []) if str(s or "").strip()]
        if not cleaned:
            return 0
        try:
            stmt_embeds = self.embedding.embed_documents(cleaned)
        except Exception as e:
            print(f"[长期记忆] 冲突语句嵌入失败: {e}")
            return 0

        suppressed = set()
        for stmt, emb in zip(cleaned, stmt_embeds):
            try:
                hits = self.store.similarity_search_by_vector(emb, k=top_k)
            except Exception as e:
                print(f"[长期记忆] 冲突检索失败: {e}")
                continue
            if not hits:
                continue
            try:
                hit_embeds = self.embedding.embed_documents([h.page_content for h in hits])
            except Exception as e:
                print(f"[长期记忆] 冲突候选嵌入失败: {e}")
                continue
            for doc, he in zip(hits, hit_embeds):
                if self._cosine(emb, he) < config.MEMORY_FACT_CONFLICT_THRESHOLD:
                    continue
                pid = doc.metadata.get("parent_id")
                if not pid or pid not in self._registry:
                    continue
                if self._mark_forgotten(pid, penalty=3, hard_delete=False):
                    suppressed.add(pid)

        if suppressed:
            self._save_registry()
            self._rebuild_bm25()
            print(f"[长期记忆] 因事实更新软遗忘了 {len(suppressed)} 条相关旧记忆")
        return len(suppressed)
