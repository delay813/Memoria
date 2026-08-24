"""
从Agent(NPC): 独立人格 + 隔离会话历史 + 摘要压缩 + 该NPC视角用户画像 + 独立长期记忆。

本文件只保留 Agent 的"骨架": 状态初始化/持久化、短期历史、reset 与流式对话编排;
各职责拆分到独立混入(mixin)模块:
  - persona.py      人格内核(蒸馏/成长/重写)
  - relationship.py 好感度/心情/关系张力/称呼/语录
  - memory.py       记忆抽取与压缩(画像+摘要+事实)
  - dream.py        深睡(隔夜整理/回忆日记)
  - proactive.py    主动消息/延迟回复/里程碑剧情
  - cognition.py    读心/内心独白/系统提示词构造
  - world.py        作息状态/随机事件/命运大纲
  - topics.py       开场话题
"""
import json
import os
import threading
import time

from dashscope import Generation

import config
import prompts
import storage
import time_utils
from cognition import CognitionMixin
from dream import DreamMixin
from fact_memory import FactMemory
from longterm_memory import LongTermMemory
from memory import MemoryMixin
from persona import PersonaMixin
from proactive import ProactiveMixin
from relationship import RelationshipMixin
from topics import TopicsMixin
from world import WorldMixin
from world_sim import LifeSim


class Agent(PersonaMixin, RelationshipMixin, MemoryMixin, DreamMixin,
            ProactiveMixin, CognitionMixin, WorldMixin, TopicsMixin):
    """一个NPC角色实例, 所有状态与其他NPC完全隔离。"""

    def __init__(self, agent_id, name, persona, description="", birthday="", unlock_cb=None):
        self.agent_id = agent_id
        self.name = name
        self.persona = persona or ""          # 完整初始设定(静态)
        self.description = description or ""
        self.birthday = birthday or ""
        self._unlock_cb = unlock_cb or (lambda key: None)   # 成就解锁回调(由总控注入)

        # 隔离存储目录
        self.storage_dir = os.path.join(config.AGENTS_DIR, agent_id)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.history_file = os.path.join(self.storage_dir, "chat.json")
        self.memory_file = os.path.join(self.storage_dir, "memory.json")
        self.persona_file = os.path.join(self.storage_dir, config.PERSONA_FILE_NAME)
        self.anchor_file = os.path.join(self.storage_dir, config.PERSONA_ANCHOR_FILE_NAME)
        self.state_file = os.path.join(self.storage_dir, config.STATE_FILE_NAME)
        self.inbox_file = os.path.join(self.storage_dir, config.INBOX_FILE_NAME)
        self.fact_file = os.path.join(self.storage_dir, config.FACT_FILE_NAME)
        self.daily_log_file = os.path.join(self.storage_dir, config.DREAM_DAILY_LOG_NAME)
        self.life_file = os.path.join(self.storage_dir, config.LIFE_FILE_NAME)

        # 世界模拟: 作息 + 每日随机事件(不在场时的生命感)
        self.life = LifeSim(agent_id, self.life_file)

        # 短期记忆: 当前对话上下文(首条是人格system message)
        self.history = [{"role": "system", "content": self.persona}]

        # 长期记忆: 独立RAG向量库(情景记忆)
        self.long_term = LongTermMemory(os.path.join(self.storage_dir, "chroma_db"))

        # 结构化事实库(语义记忆)
        self.fact_memory = FactMemory(self.fact_file)

        # 每个NPC一把锁, 防止并发请求互相污染历史。
        # 锁纪律(重要): threading.Lock 不可重入。持锁期间严禁调用任何会再次获取本锁的方法
        # (add_proactive/mark_read/reset/sync_life/generate_delayed_reply 等); 需要拿锁的
        # 动作一律放到锁释放之后, 或先记标记、离开临界区后再执行。网络调用必须在锁外进行。
        self._lock = threading.Lock()

        # 后台任务标志(防止重复触发)
        self._compacting = False
        self._distilling = False
        self._dreaming = False
        self._scripting = False

        # 人格内核缓存(按文件mtime失效)
        self._persona_mtime = -1
        self._persona_content = ""

        # 话题开场卡缓存(带时间戳, 过期后重新生成)
        self._topics_cache = None

        # 好感度 / 心情
        self._favor = config.FAVOR_INITIAL
        self._mood = 0.0
        self._load_state()

        self._load_history()

    # ============================================================
    # 动态状态持久化(state.json: 好感度/心情/最后互动/主动占位/深睡/待回复/张力)
    # ============================================================
    def _load_state(self):
        self._last_seen = ""
        self._last_proactive = {}
        self._mood = 0.0
        self._last_dream_ts = 0
        self._pending_replies = []
        self._tension = 0
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._favor = int(state.get("favorability", config.FAVOR_INITIAL))
                self._last_seen = state.get("last_seen", "")
                self._last_proactive = state.get("last_proactive", {})
                self._last_dream_ts = state.get("last_dream_ts", 0)
                pending = state.get("pending_replies", [])
                if isinstance(pending, list):
                    self._pending_replies = pending
                try:
                    self._tension = int(state.get("tension", 0))
                except (TypeError, ValueError):
                    self._tension = 0
                self._tension = max(0, min(config.TENSION_MAX, self._tension))
                try:
                    self._mood = float(state.get("mood", 0.0))
                except (TypeError, ValueError):
                    self._mood = 0.0
        except Exception as e:
            print(f"[{self.name}] 状态加载失败: {e}")

    def _save_state(self):
        try:
            storage.save_json(self.state_file, {
                "favorability": self._favor,
                "mood": self._mood,
                "last_seen": self._last_seen,
                "last_proactive": self._last_proactive,
                "last_dream_ts": self._last_dream_ts,
                "pending_replies": self._pending_replies,
                "tension": self._tension,
            })
        except Exception as e:
            print(f"[{self.name}] 状态保存失败: {e}")

    # ============================================================
    # 历史持久化
    # ============================================================
    def _load_history(self):
        if not os.path.exists(self.history_file) or os.path.getsize(self.history_file) == 0:
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not loaded:
                return
            # 人格以最新配置为准, 只保留历史正文
            if loaded[0].get("role") == "system":
                self.history = [{"role": "system", "content": self.persona}] + loaded[1:]
            else:
                self.history = [{"role": "system", "content": self.persona}] + loaded
        except Exception as e:
            print(f"[{self.name}] 历史加载失败: {e}")

    def _save_history(self):
        try:
            storage.save_json(self.history_file, self.history)
        except Exception as e:
            print(f"[{self.name}] 历史保存失败: {e}")

    def get_history(self):
        """返回该NPC的对话历史(不含system人格), 供前端恢复会话。

        附带 ts(秒级时间戳)供前端显示真实发送时间; 过滤掉流式中断产生的残片(interrupted)。
        """
        with self._lock:
            return [
                {"role": m["role"], "content": m["content"], "ts": m.get("ts")}
                for m in self.history[1:]
                if m.get("role") in ("user", "assistant") and not m.get("interrupted")
            ]

    def reset(self):
        """初始化该角色: 清空对话历史/结晶记忆/好感度/心情/长期记忆/事实库/收件箱/日记(保留人格内核与锚点)。"""
        with self._lock:
            self.history = [{"role": "system", "content": self.persona}]
            self._favor = config.FAVOR_INITIAL
            self._mood = 0.0
            self._last_seen = ""
            self._last_proactive = {}
            self._last_dream_ts = 0
            self._pending_replies = []
            self._tension = 0
            self._compacting = False
            self._distilling = False
            self._dreaming = False
            self._scripting = False
            self._topics_cache = None
            for f in (self.history_file, self.memory_file, self.state_file,
                      self.inbox_file, self.fact_file, self.daily_log_file, self.life_file):
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    print(f"[{self.name}] 删除 {f} 失败: {e}")
            self.long_term.reset()
            self.fact_memory.reset()
            self.life.reset()

    # ============================================================
    # 对话(流式)
    # ============================================================
    def chat_stream(self, prompt, world_context=None, social_context=None, user_nickname=None):
        self._lock.acquire()
        try:
            # 0) 惰性触发深睡(隔夜整理): 满足条件才后台执行
            self._maybe_dream_async()

            # 0.5) 深夜睡着时: 不立即回复, 存入待回复队列, 等她醒来再补回
            status = self.current_status()
            if status.get("sleepy"):
                self.history.append({"role": "user", "content": prompt, "ts": time.time()})
                self._save_history()
                self._queue_pending_reply(prompt)
                self._last_seen = time_utils.today_iso()
                self._save_state()
                yield {
                    "type": "sleep",
                    "data": {
                        "status": status.get("label", ""),
                        "activity": status.get("activity", ""),
                        "note": "她睡得很沉，明早会回复你。",
                    },
                }
                return

            # 1) 检索: 用"当前消息 + 最近上下文"做查询(联想式激活)
            query = self._build_recall_query(prompt)
            recalled = self.long_term.recall(query)
            facts = self.fact_memory.retrieve(query, k=config.FACT_RECALL_K)

            # 2) 用户消息入短期历史
            self.history.append({"role": "user", "content": prompt, "ts": time.time()})

            # 3) 用户轮次先落盘(浅睡触发移到认知后, 见步骤5)
            self._save_history()

            # 4) 认知步骤: 读心 + 内心独白 + 心情变化(便宜模型, 可开关)
            cognition = self._cognize(prompt, recalled, facts, world_context)

            # 5) 本轮话题回流 + 主题漂移触发浅睡(压缩+去重, 不阻塞回复)
            topic = (cognition or {}).get("topic") or ""
            if topic and self.history and self.history[-1].get("role") == "user":
                self.history[-1]["topic"] = topic
            self._maybe_compact_async((cognition or {}).get("topic_shift") is True)

            # 6) 应用心情变化
            self._apply_mood(cognition.get("mood_shift", 0))

            # 6.5) 透出认知(心声/读心/心情变化)给前端: 让用户"看见"角色此刻的内心
            yield {
                "type": "cognition",
                "data": {
                    "internal_thought": (cognition or {}).get("internal_thought", ""),
                    "user_emotion": (cognition or {}).get("user_emotion", ""),
                    "user_intent": (cognition or {}).get("user_intent", ""),
                    "topic": (cognition or {}).get("topic", ""),
                    "mood_shift": (cognition or {}).get("mood_shift", 0),
                    "mood": self._mood_label(),
                },
            }

            # 7) 构造消息: 人格+准则+关系+心情+世界+问候+画像+事实+独白+召回记忆 + 历史正文
            messages = [{
                "role": "system",
                "content": self._build_system_message(
                    recalled, facts, world_context, cognition,
                    social_context=social_context, user_nickname=user_nickname,
                ),
            }]
            # 只传 role/content 给模型, 剥离历史消息上的内部字段(如 topic), 避免污染API调用
            # 同时跳过流式中断产生的残片(interrupted), 防止截断内容进入模型上下文
            messages += [
                {"role": m["role"], "content": m["content"]}
                for m in self.history[1:]
                if not m.get("interrupted")
            ]

            # 8) 当轮用户消息末尾追加硬性风格约束(只影响本次调用, 不入历史)
            style = (cognition or {}).get("style") or "chat"
            if messages and messages[-1]["role"] == "user":
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + "\n\n" + prompts.tail_hint(style),
                }

            full_answer = ""
            interrupted = False
            attempt = 0
            while True:
                try:
                    resps = Generation.call(
                        api_key=config.API_KEY,
                        model=config.CHAT_MODEL,
                        messages=messages,
                        result_format="message",
                        stream=True,
                        incremental_output=True,
                    )
                    for resp in resps:
                        if resp.status_code != 200:
                            raise RuntimeError(f"模型返回异常状态 {resp.status_code}")
                        chunk = resp.output.choices[0].message.content
                        full_answer += chunk
                        yield {"type": "content", "data": chunk}
                    break  # 正常完成
                except Exception as e:
                    print(f"[{self.name}] 对话调用异常(第{attempt+1}次): {e}")
                    if full_answer:
                        # 已输出部分内容, 不重试, 追加一句收尾
                        note = "（刚才好像断了一下）"
                        full_answer += note
                        interrupted = True
                        yield {"type": "content", "data": note}
                        break
                    if attempt >= config.MODEL_MAX_RETRIES:
                        fallback = "（刚刚有点走神，能再说一遍吗？）"
                        full_answer = fallback
                        interrupted = True
                        yield {"type": "content", "data": fallback}
                        break
                    attempt += 1
                    time.sleep(config.MODEL_RETRY_DELAY)

            # 助手回复入历史并保存(带本轮话题标签)
            # 流式中断/失败的残片打 interrupted 标记: 构建模型消息与历史返回时会过滤,
            # 避免截断内容污染后续对话与记忆提取
            msg = {"role": "assistant", "content": full_answer, "topic": topic, "ts": time.time()}
            if interrupted:
                msg["interrupted"] = True
            self.history.append(msg)
            self._save_history()

            # 每轮同步结算好感度(带理由)并透出给前端即时反馈 + 关系里程碑
            favor_result = self._settle_favor()
            yield {"type": "favor", "data": favor_result}
            if favor_result.get("milestone"):
                yield {"type": "milestone", "data": favor_result["milestone"]}
        finally:
            self._lock.release()
