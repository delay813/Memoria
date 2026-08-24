"""
记忆抽取与压缩(MemoryMixin): 滚动式"画像 + 摘要 + 人格成长 + 稳定事实"提取, 浅睡压缩。

- memory.json 存 user_profile(画像) + conversation_summary(摘要) + last_compact_turns
- 主题漂移或达到兜底轮数时后台异步压缩: 提取→应用→(必要时)重写人格→遗忘→跨记忆一致性
"""
import json
import os
import threading

import config
import json_utils
import llm
import storage
import text_utils


class MemoryMixin:
    """记忆抽取 / 压缩 / 画像。混入 Agent。"""

    # ============================================================
    # 记忆(摘要+画像)持久化
    # ============================================================
    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[{self.name}] 记忆加载失败: {e}")
            return {}

    def _save_memory(self, profile=None, summary=None, last_compact_turns=None):
        memory = self._load_memory()
        if profile is not None:
            memory["user_profile"] = profile
        if summary is not None:
            memory["conversation_summary"] = summary
        if last_compact_turns is not None:
            memory["last_compact_turns"] = last_compact_turns
        try:
            storage.save_json(self.memory_file, memory)
        except Exception as e:
            print(f"[{self.name}] 记忆保存失败: {e}")

    @staticmethod
    def _clean_text(content):
        if not content:
            return ""
        return str(content).strip()[:config.MEMORY_INPUT_MAX_CHARS]

    def get_memory_profile(self):
        """返回"她眼中的你": 关系状态 + 关于用户的事实库 + 她对用户的印象画像(供前端记忆卡展示)。"""
        memory = self._load_memory()
        return {
            "favorability": self._favor,
            "stage": self._favor_stage(),
            "mood": self._mood_label(),
            "tension": self._tension,
            "tension_label": self._tension_label(),
            "facts": self.fact_memory.all(),
            "profile": memory.get("user_profile", ""),
        }

    # ============================================================
    # 检索意图 & 最近对话
    # ============================================================
    def _build_recall_query(self, prompt):
        """用"当前消息 + 最近上下文 + 最近话题标签"做检索, 更贴近人脑的联想式激活。"""
        recent = [m for m in self.history if m.get("role") in ("user", "assistant")][-3:]
        parts = []
        for m in recent:
            parts.append(self._clean_text(m.get("content", "")))
            topic = m.get("topic")
            if topic:
                parts.append(str(topic))
        parts.append(self._clean_text(prompt))
        text = " ".join(p for p in parts if p).strip()
        return text[-config.MEMORY_INPUT_MAX_CHARS:]

    def _recent_history_text(self, n=6):
        """最近对话文本(不含当前这条用户消息, 供认知步骤使用)。"""
        recent = [m for m in self.history[1:]
                  if m.get("role") in ("user", "assistant") and not m.get("interrupted")][-(n + 1):-1]
        return "\n".join(f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in recent)

    # ============================================================
    # 记忆提取(滚动式: 画像 + 摘要 + 人格成长 + 稳定事实)
    # ============================================================
    def _extract_memory(self, history=None):
        history = history if history is not None else self.history
        recent = [m for m in history[1:]
                  if m.get("role") in ("user", "assistant") and not m.get("interrupted")][
            -config.MEMORY_INPUT_MAX_TURNS:
        ]
        history_text = "\n".join(
            f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in recent
        )

        old = self._load_memory()
        old_profile = old.get("user_profile", "暂无")
        old_summary = old.get("conversation_summary", "暂无")
        old_facts = self.fact_memory.all()
        facts_text = "\n".join(f"- {f}" for f in old_facts) or "暂无"

        prompt = f"""
你是「{self.name}」的记忆与成长提取助手。请阅读"已有记忆"和"新的对话历史", 在已有记忆基础上合并更新, 严格输出JSON(不要任何其他文字):

1. user_profile(用户画像): 从「{self.name}」的视角, 记录关于用户的身份、性格、关注点、与「{self.name}」的关系等长期稳定的印象, 100字以内, 没有则写"暂无"
2. conversation_summary(对话摘要): 将旧摘要与新对话合并, 概括这段对话的主题和已发生的事, 150字以内
3. persona_growth(人格成长): 这段对话让「{self.name}」产生的新认知或成长(新的世界观、对用户的新了解、观念变化等), 一句话以内, 没有则写""
4. facts(关于用户的稳定事实): 抽取/合并关于用户的稳定事实(身份、性格、偏好、约定、重要事件等), 每条{config.FACT_MAX_CHARS}字以内, 去掉已过时或已被纠正的信息, 与旧事实合并去重, 最多{config.FACT_MAX_COUNT}条, 没有则给空数组[]
5. tags(话题标签): 从这段对话提取 3~5 个话题关键词/标签(用于记忆检索与联想, 如 "面试" "换工作" "焦虑"), 没有则给空数组[]
6. removed_facts(被删除的旧事实): 列出你从"已有事实"里删除的那些"已过时或被明确纠正"的旧事实原文(仅当旧事实确实已过时/被纠正时才列; 措辞微调、合并改写而仍有效的信息不要列), 没有则给空数组[]

要求: 保留旧记忆中仍然有效的信息; 不要遗漏旧记忆中的重要内容

【已有用户画像】
{old_profile}

【已有对话摘要】
{old_summary}

【已有事实】
{facts_text}

【新的对话历史】
{history_text}

输出格式:
{{"user_profile":"...","conversation_summary":"...","persona_growth":"...","facts":["...","..."],"removed_facts":["..."],"tags":["...","..."]}}
"""
        resp = llm.call([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if not resp:
            print(f"[{self.name}] 记忆提取失败")
            return {}
        info = json_utils.parse_object(resp.output.choices[0].message.content)
        if not isinstance(info, dict):
            info = {}
        facts = info.get("facts", []) or []
        if not isinstance(facts, list):
            facts = [facts]
        tags = info.get("tags", []) or []
        if not isinstance(tags, list):
            tags = [tags]
        removed_facts = info.get("removed_facts", []) or []
        if not isinstance(removed_facts, list):
            removed_facts = [removed_facts]
        return {
            "user_profile": (info.get("user_profile") or "").strip(),
            "conversation_summary": (info.get("conversation_summary") or "").strip(),
            "persona_growth": (info.get("persona_growth") or "").strip(),
            "facts": [str(x).strip() for x in facts if str(x).strip()],
            "removed_facts": [str(x).strip() for x in removed_facts if str(x).strip()],
            "tags": [str(x).strip() for x in tags if str(x).strip()],
        }

    # ============================================================
    # 上下文压缩(异步): 更新画像+摘要+事实+人格成长, 有价值对话写入长期记忆
    # ============================================================
    def _should_compact(self, topic_shift=False):
        """浅睡触发: 主题漂移(上一段话题结束) 或 达到兜底轮数。"""
        user_turns = len([m for m in self.history if m.get("role") == "user"])
        memory = self._load_memory()
        since_last = user_turns - memory.get("last_compact_turns", 0)
        if since_last >= config.MEMORY_UPDATE_THRESHOLD:
            return True
        if topic_shift and since_last >= config.MEMORY_SHIFT_MIN_TURNS:
            return True
        return False

    def _maybe_compact_async(self, topic_shift=False):
        """需要压缩时, 丢到后台线程执行(调用方已持有 self._lock)。"""
        if not self._should_compact(topic_shift) or self._compacting:
            return
        self._compacting = True
        threading.Thread(target=self._run_compact, daemon=True).start()

    def _run_compact(self):
        try:
            # 1) 提取(大模型, 先拿快照, 不在锁内做网络调用)
            with self._lock:
                snapshot = list(self.history)
            mem = self._extract_memory(snapshot)
            # 2) 应用(加锁, 快速完成) + 判断是否需要重写内核
            with self._lock:
                self._apply_compact(
                    mem.get("user_profile", ""),
                    mem.get("conversation_summary", ""),
                    mem.get("persona_growth", ""),
                    mem.get("facts", []),
                    mem.get("tags", []),
                )
                should_rewrite = self._should_rewrite_persona()
            # 3) 重写内核(大模型, 锁外执行)
            if should_rewrite:
                new_kernel = self._rewrite_persona()
                if new_kernel:
                    with self._lock:
                        self._save_persona_md(new_kernel)
                        print(f"[{self.name}] 人格内核已重写(成长)")
            # 4) 遗忘机制(小模型, 锁外执行, 避免向量库只增不减)
            try:
                self.long_term.forget(self.name)
            except Exception as e:
                print(f"[{self.name}] 遗忘机制异常: {e}")
            # 4.5) 跨记忆一致性(锁外执行): 被纠正/过时的旧事实 → 软遗忘相关情景记忆,
            #      让事实库(权威结论)优先于向量库(旧证据), 避免角色"旧事重提"自相矛盾
            removed = mem.get("removed_facts", []) or []
            if removed:
                try:
                    self.long_term.suppress_conflicts(removed)
                except Exception as e:
                    print(f"[{self.name}] 事实冲突软遗忘异常: {e}")
        except Exception as e:
            print(f"[{self.name}] 异步压缩失败: {e}")
        finally:
            self._compacting = False

    def _apply_compact(self, profile, summary, growth, facts, tags=None):
        """把结晶记忆/事实/人格成长落盘 + 有价值对话写入长期记忆 + 截断历史(必须在锁内调用)。"""
        if profile or summary:
            self._save_memory(profile, summary)
            print(f"[{self.name}] 记忆已更新")

        if facts:
            self.fact_memory.replace_all(facts)
            print(f"[{self.name}] 事实库已更新({len(facts)}条)")

        if growth:
            self._append_growth(growth)
            print(f"[{self.name}] 人格成长: {growth}")

        if len(self.history) > config.MEMORY_KEEP_TURNS * 2:
            removed = self.history[1:-(config.MEMORY_KEEP_TURNS * 2)]
            batch_texts, batch_metas = [], []
            extra_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
            for m in removed:
                if (m.get("content") and text_utils.worth_remembering(m.get("content", ""))
                        and not m.get("interrupted")):  # 残片不写入长期记忆
                    topics = []
                    if m.get("topic"):
                        topics.append(str(m["topic"]).strip())
                    topics += extra_tags
                    batch_texts.append(m.get("content", ""))
                    batch_metas.append({
                        "agent": self.agent_id,
                        "role": m.get("role", ""),
                        "topics": topics,
                    })
            if batch_texts:
                self.long_term.add_many(batch_texts, batch_metas)
            self.history = [self.history[0]] + self.history[-(config.MEMORY_KEEP_TURNS * 2):]
            self._save_history()

        remaining = len([m for m in self.history if m.get("role") == "user"])
        self._save_memory(last_compact_turns=remaining)
