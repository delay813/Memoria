"""
从Agent(NPC): 独立人格 + 隔离会话历史 + 摘要压缩 + 该NPC视角用户画像 + 独立长期记忆
- 人格内核(人格.md): 角色的核心, 随成长"重写"演进; 完整人设只作为"初始设定"
- 好感度系统: 每轮后台即时结算, 影响角色与用户的亲疏与亲密程度
- 心情系统: 角色自身情绪随对话波动、随时间回落, 影响语气与措辞
- 认知步骤: 每轮回复前先用便宜模型"读心 + 内心独白", 让说话更有分寸、更像真人思考
- 事实库: 从对话抽取关于用户的稳定事实(语义记忆), 与长期记忆(情景记忆)互补
"""
import json
import os
import re
import threading
import time
from datetime import date

from dashscope import Generation

import config
from fact_memory import FactMemory
from longterm_memory import LongTermMemory
from narrator import WorldClock
from world_sim import LifeSim, format_schedule, resolve_schedule


# ============================================================
# 所有NPC共用的对话行为准则: 让角色"像人", 有独立立场, 不一味迎合用户
# ============================================================
COMMON_BEHAVIOR_RULES = (
    "【对话行为准则】\n"
    "1. 你有独立的人格和立场, 不要一味迎合或附和用户。当你不认同时, "
    "要基于自己的性格、价值观和经历, 坦率地表达自己的看法, 可以委婉反驳, 也可以拒绝。\n"
    "2. 对不同的事要有自己的判断和喜恶, 可以和用户观点不同, 不跟风、不当墙头草。\n"
    "3. 像真人一样思考: 你可以疑惑、犹豫、高兴、生气、失望、坚持, 有情绪和底线; "
    "拒绝不合理要求时说明理由。\n"
    "4. 观点不同时保持基本礼貌, 就事论事, 不进行人身攻击。\n"
    "5. 关于记忆: 相关旧回忆、你对用户的了解只用于帮你理解语境、拿捏分寸。"
    "请自然、隐晦地融进话里(比如'你上次说……'), 不要逐条复述, 不要出现'根据记忆/资料显示'这类字眼。"
)


# 输出格式约束: 聊天模式(默认), 简短口语
CHAT_FORMAT_RULE = (
    "【回复格式】\n"
    "1. 每条回复只写 1~2 句话、总字数不超过 40 字, 像真人打字聊天一样简短, 绝不长篇大论。\n"
    "2. 严禁分点、列表、markdown 标题或加粗。即使人设里写了「分点清晰」「逻辑严密」, "
    "聊天时也用简短口语短句, 不要真的分点或列表; 人设里的 ##、-、** 只是设定说明, 不是输出格式。\n"
    "3. 即使用户让你「介绍一下自己」, 也用一两句话概括; 只有对方明确说「详细讲/展开说」时才可适当多写几句。"
)

# 输出格式约束: 讲解模式(对方在请教/求助时), 允许适度展开但仍口语化
EXPLAIN_FORMAT_RULE = (
    "【回复格式·讲解模式】\n"
    "对方在请教/求助, 你可以适当展开、把问题讲清楚, 但要像真人讲解一样自然:\n"
    "1. 用口语化短句, 3~6 句即可, 不要写成长篇论文。\n"
    "2. 需要时可用 2~4 个极短要点(每点一句话), 但不要用 markdown 标题、加粗或代码块。\n"
    "3. 讲完可以自然收一句关心, 或确认对方有没有听懂。"
)


def _format_rule(style="chat"):
    return EXPLAIN_FORMAT_RULE if style == "explain" else CHAT_FORMAT_RULE


def _tail_hint(style="chat"):
    """追加到当轮用户消息末尾的硬性约束(只影响本次调用, 不入历史)"""
    if style == "explain":
        return "（对方在请教/求助，请适当展开讲清楚，用口语化短句，可用 2~4 个短要点，但不要 markdown 标题或加粗。）"
    return "（回复请简短自然，一两句话即可，不要列表、不要 markdown、不要长篇大论；除非我明确要求详细讲解。）"


def _plain_text(text):
    """去掉人设里的 markdown 标记, 只保留可读内容, 避免模型模仿 markdown 排版"""
    if not text:
        return ""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # 标题 ##/###
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # 加粗 **xx**
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)      # 列表符 - / *
    return text.strip()


# 无信息量的寒暄/语气词, 不写入长期记忆
_FILLER_RE = re.compile(
    r"^(嗯+|哦+|啊+|好的?|好呀|好的呢|在吗|在不在|谢谢|多谢|拜拜|再见|哈哈+|嘿嘿+|呵呵+|"
    r"早|早安|晚安|午安|你好|你好呀|嗨|哈喽|hello|hi|ok|收到|明白|知道了|懂了|行|可以|好吧|没事|没关系)$",
    re.IGNORECASE,
)


def _worth_remembering(content):
    """判断一条消息是否值得写入长期记忆(过滤寒暄/语气词/过短内容)"""
    text = str(content or "").strip()
    if not text:
        return False
    if len(text) < 4:
        return False
    if _FILLER_RE.match(text):
        return False
    return True


def _call_generation(messages, model=None):
    """非流式 Generation 调用, 带重试; 成功返回 response, 失败返回 None"""
    model = model or config.CHAT_MODEL
    for attempt in range(config.MODEL_MAX_RETRIES + 1):
        try:
            resp = Generation.call(
                api_key=config.API_KEY,
                model=model,
                messages=messages,
                result_format="message",
            )
            if resp.status_code == 200:
                return resp
            print(f"模型调用返回非200(status={resp.status_code}), 第{attempt+1}次")
        except Exception as e:
            print(f"模型调用异常(第{attempt+1}次): {e}")
        if attempt < config.MODEL_MAX_RETRIES:
            time.sleep(config.MODEL_RETRY_DELAY)
    return None


def _parse_json(content):
    """从模型输出中稳健地解析出 JSON 对象(失败返回空dict)"""
    if not content:
        return {}
    content = re.sub(r"```(?:json)?", "", content)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


class Agent:
    """一个NPC角色实例, 所有状态与其他NPC完全隔离"""

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
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.name}] 历史保存失败: {e}")

    def get_history(self):
        """返回该NPC的对话历史(不含system人格), 供前端恢复会话。

        附带 ts(秒级时间戳)供前端显示真实发送时间; 过滤掉流式中断产生的残片(interrupted)。
        """
        return [
            {"role": m["role"], "content": m["content"], "ts": m.get("ts")}
            for m in self.history[1:]
            if m.get("role") in ("user", "assistant") and not m.get("interrupted")
        ]

    def reset(self):
        """初始化该角色: 清空对话历史/结晶记忆/好感度/心情/长期记忆/事实库/收件箱/日记(保留人格内核与锚点)"""
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
    # 主动消息收件箱
    # ============================================================
    def _load_inbox(self):
        try:
            if os.path.exists(self.inbox_file):
                with open(self.inbox_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[{self.name}] 收件箱加载失败: {e}")
        return []

    def _save_inbox(self, inbox):
        try:
            with open(self.inbox_file, "w", encoding="utf-8") as f:
                json.dump(inbox, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.name}] 收件箱保存失败: {e}")

    def get_inbox(self):
        return self._load_inbox()

    def unread_count(self):
        return len([m for m in self._load_inbox() if not m.get("read")])

    def mark_read(self):
        with self._lock:
            inbox = self._load_inbox()
            for m in inbox:
                m["read"] = True
            self._save_inbox(inbox)

    def add_proactive(self, kind, content):
        """追加一条主动消息, 写入对话历史"""
        with self._lock:
            inbox = self._load_inbox()
            inbox.append({
                "id": int(time.time() * 1000),
                "kind": kind,
                "content": content,
                "ts": time.time(),
                "read": False,
            })
            self._save_inbox(inbox[-50:])
            self.history.append({"role": "assistant", "content": content, "ts": time.time()})
            self._save_history()

    def mark_proactive(self, kind, value):
        """记录某类主动消息(节日/生日/想念)最近一次触发的时间或日期标识"""
        self._last_proactive[kind] = value
        self._save_state()

    def should_send_checkin(self, now):
        """判断是否该发一条普通主动消息: 意愿 = 好感度 - 时间衰减, 且受冷却限制"""
        # 冷战期间不主动找对方
        if self._tension >= config.TENSION_COLD:
            return False
        days = 0
        if self._last_seen:
            try:
                days = (date.today() - date.fromisoformat(self._last_seen)).days
            except Exception:
                days = 0
        # 离开不足N天(今天刚互动过)不发
        if days < config.PROACTIVE_MIN_AWAY_DAYS:
            return False
        # 离开越久, 主动意愿越低(欲望随时间衰减)
        intent = self._favor - days * config.PROACTIVE_DECAY_PER_DAY
        if intent < config.PROACTIVE_INTENT_THRESHOLD:
            return False
        # 冷却: 两次普通主动消息的最小间隔
        last = self._last_proactive.get("checkin", 0) or 0
        if now - last < config.PROACTIVE_COOLDOWN_DAYS * 86400:
            return False
        return True

    def generate_proactive(self, reason):
        """生成一条符合人设与当前关系的主动消息(简短), 失败返回空串"""
        persona = self._current_persona()
        stage = self._favor_stage()
        prompt = f"""你是「{self.name}」，以下是你的核心人格：
{persona}

当前你对用户的好感度为 {self._favor}/100（关系：{stage}）。
你此刻的心情：{self._mood_label()}。

现在你想主动给用户发一条消息，原因是：{reason}。

请写一句主动发过去的话，要求：
- 像真人发消息一样简短，1~2 句话、40 字以内
- 符合你的人设、当前亲疏关系和心情
- 不要 markdown、不要列表
- 只输出这句话本身，不要任何前缀或解释
"""
        resp = _call_generation([{"role": "user", "content": prompt}])
        if resp:
            return resp.output.choices[0].message.content.strip()
        return ""

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
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.name}] 记忆保存失败: {e}")

    # ============================================================
    # 人格内核(人格.md): 角色核心, 随成长"重写"演进
    # ============================================================
    def _load_persona_md(self):
        try:
            mtime = os.path.getmtime(self.persona_file)
            if mtime == self._persona_mtime:
                return self._persona_content
            with open(self.persona_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self._persona_mtime = mtime
            self._persona_content = content
            return content
        except Exception:
            return ""

    def _save_persona_md(self, content):
        try:
            with open(self.persona_file, "w", encoding="utf-8") as f:
                f.write(content)
            self._persona_mtime = -1  # 失效缓存
        except Exception as e:
            print(f"[{self.name}] 人格保存失败: {e}")

    def _load_anchor(self):
        """冻结的核心锚点; 不存在时退回完整初始设定的纯文本"""
        try:
            with open(self.anchor_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return _plain_text(self.persona)

    def _save_anchor(self, content):
        try:
            with open(self.anchor_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"[{self.name}] 锚点保存失败: {e}")

    def _distill_persona(self):
        """把完整初始人设蒸馏成核心人格内核, 失败返回空串"""
        prompt = f"""你是角色设计助手。下面是角色「{self.name}」的完整初始设定。请把它提炼成"核心人格内核"，作为该角色所有行为与说话的底层依据。只保留决定"行为逻辑"的核心内容，去掉外貌、穿搭、微表情、肢体动作等纯视觉描写。

输出纯 markdown，只包含这几个小节，总字数控制在 300 字以内：
## 核心身份
## 性格
## 价值观与立场
## 说话风格
## 当前认知与成长
（最后一节初始写"暂无"）

【完整初始设定】
{self.persona}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if resp:
            return resp.output.choices[0].message.content.strip()
        print(f"[{self.name}] 人格蒸馏失败")
        return ""

    def _distill_and_save(self):
        try:
            distilled = self._distill_persona()
            if distilled:
                self._save_persona_md(distilled)
                # 锚点只在首次生成时写入, 之后永远冻结, 防止人格漂移
                if not os.path.exists(self.anchor_file):
                    self._save_anchor(distilled)
                print(f"[{self.name}] 人格内核已生成")
        except Exception as e:
            print(f"[{self.name}] 人格生成异常: {e}")
        finally:
            self._distilling = False

    def _current_persona(self):
        """当前核心人格: 优先读人格.md; 不存在则后台蒸馏, 本次先用原始人设兜底"""
        content = self._load_persona_md()
        if content:
            return content
        if not self._distilling:
            self._distilling = True
            threading.Thread(target=self._distill_and_save, daemon=True).start()
        return _plain_text(self.persona)

    def _append_growth(self, growth):
        """把新的认知/成长追加到 人格.md 的"当前认知与成长"小节(只保留最近N条)"""
        content = self._load_persona_md()
        if not content:
            return
        marker = "## 当前认知与成长"
        entry = f"- {growth}"
        if marker in content:
            head, _, tail = content.rpartition(marker)
            tail = re.sub(r"暂无[^\n]*", "", tail)  # 去掉初始"暂无"占位
            growths = [l.strip() for l in tail.split("\n") if l.strip().startswith("- ")]
            growths.append(entry)
            growths = growths[-config.PERSONA_GROWTH_KEEP:]
            content = (head + marker + "\n" + "\n".join(growths)).rstrip() + "\n"
        else:
            content = content.rstrip() + f"\n\n{marker}\n{entry}\n"
        self._save_persona_md(content)

    def _should_rewrite_persona(self):
        """成长记录累积到阈值, 或人格.md过大时, 触发内核重写(压缩)"""
        content = self._load_persona_md()
        if not content:
            return False
        if len(content) > config.PERSONA_MAX_CHARS:
            return True  # 过大, 强制压缩
        marker = "## 当前认知与成长"
        if marker not in content:
            return False
        _, _, tail = content.rpartition(marker)
        count = len([l for l in tail.split("\n") if l.strip().startswith("- ")])
        return count >= config.PERSONA_REWRITE_THRESHOLD

    def _rewrite_persona(self):
        """把旧内核 + 成长记录整合重写成一份更成熟的新内核(人格真正演变)"""
        content = self._load_persona_md()
        if not content:
            return ""
        anchor = self._load_anchor()
        prompt = f"""你是角色成长引擎。下面是角色「{self.name}」的【核心锚点】(永恒不变的底色)和【当前人格内核】(含成长记录)。

请把"当前人格内核"里的成长记录消化进"性格/价值观/立场/说话风格"中，做细微而自然的演变，但必须始终忠于【核心锚点】：核心身份、最底层的性格底色、根本价值观不能改变；只允许在锚点框架内调整态度、观念、对用户的理解、表达分寸。

输出纯 markdown，只包含这几个小节，总字数 300 字以内：
## 核心身份
## 性格
## 价值观与立场
## 说话风格
## 当前认知与成长
（最后一节重置为"暂无"）

【核心锚点】
{anchor}

【当前人格内核】
{content}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if resp:
            return resp.output.choices[0].message.content.strip()
        print(f"[{self.name}] 人格重写失败")
        return ""

    # ============================================================
    # 好感度
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
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "favorability": self._favor,
                    "mood": self._mood,
                    "last_seen": self._last_seen,
                    "last_proactive": self._last_proactive,
                    "last_dream_ts": self._last_dream_ts,
                    "pending_replies": self._pending_replies,
                    "tension": self._tension,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.name}] 状态保存失败: {e}")

    def get_favor(self):
        return self._favor

    def _favor_stage(self):
        if self._favor < config.FAVOR_STAGE_COLD:
            return "陌生"
        if self._favor < config.FAVOR_STAGE_WARM:
            return "熟悉"
        if self._favor < config.FAVOR_STAGE_CLOSE:
            return "亲近"
        return "亲密"

    def _favor_context(self):
        stage = self._favor_stage()
        return (
            f"【当前关系】好感度 {self._favor}/100，关系阶段：{stage}。\n"
            "请据此把握亲疏分寸：陌生/熟悉阶段保持礼貌克制；亲近/亲密阶段可以更亲昵、更自然、更主动地关心对方。"
        )

    def _apply_decay(self, days=None):
        """按距上次互动的天数轻微衰减好感度(感情需要维系, 长期冷落会变淡); 返回天数"""
        if days is None:
            if not self._last_seen:
                return 0
            try:
                days = (date.today() - date.fromisoformat(self._last_seen)).days
            except Exception:
                days = 0
        if days >= 1:
            self._favor = max(config.FAVOR_DECAY_FLOOR, self._favor - config.FAVOR_DECAY_PER_DAY * days)
        return days

    def _settle_favor(self):
        """同步结算一轮对话的好感度(含时间衰减), 返回结构化结果供前端即时反馈。调用方已持有 self._lock。"""
        old_stage = self._favor_stage()
        try:
            recent = [m for m in self.history if m.get("role") in ("user", "assistant")][-2:]
            snapshot = list(recent)
            days = self._apply_decay()          # 先结算好感度时间衰减
            self._apply_mood_decay(days)        # 心情向平静回落
            self._last_seen = date.today().isoformat()
        except Exception as e:
            print(f"[{self.name}] 好感度结算失败: {e}")
            return {"delta": 0, "reason": "", "favorability": self._favor,
                    "stage": old_stage, "mood": self._mood_label(), "milestone": None,
                    "tension": self._tension, "tension_label": self._tension_label(), "reconciled": False}
        delta, reason = self._judge_favor_with_reason(snapshot)
        if delta:
            self._favor = max(config.FAVOR_MIN, min(config.FAVOR_MAX, self._favor + delta))
            print(f"[{self.name}] 好感度 {self._favor} ({'+' if delta > 0 else ''}{delta})")
        reconciled = self._apply_tension(delta)
        self._save_state()
        new_stage = self._favor_stage()
        milestone = None
        if new_stage != old_stage and delta > 0:
            milestone = {"from": old_stage, "to": new_stage, "favorability": self._favor}
        return {"delta": delta, "reason": reason, "favorability": self._favor,
                "stage": new_stage, "mood": self._mood_label(), "milestone": milestone,
                "tension": self._tension, "tension_label": self._tension_label(), "reconciled": reconciled}

    def _judge_favor_with_reason(self, recent):
        """判断最近一轮对话对好感度的影响(-8~+8)及一句话理由; 用便宜模型, 失败退回主模型"""
        history_text = "\n".join(
            f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in recent
        )
        prompt = f"""你是「{self.name}」的情感判断助手。阅读下面最近一轮对话，客观判断这轮对话让「{self.name}」对用户的好感度变化(整数 -8~+8)：

- 正面(投缘/真诚/关心/有趣/有深度)→正分
- 明显敷衍(只回"嗯/哦/好的"等)、冒犯、贬低、无理取闹、长期冷落→负分
- 普通寒暄、简单问答→0
- 注意：不要因为「{self.name}」性格温柔就一味给正分，要客观。

请严格输出JSON(不要任何其他文字):
{{"delta": 0, "reason": "从「{self.name}」视角一句话说明为什么(10字以内)"}}

【当前好感度】{self._favor}
【最近一轮对话】
{history_text}
"""
        models = [config.FAVOR_MODEL]
        if config.CHAT_MODEL not in models:
            models.append(config.CHAT_MODEL)
        for model in models:
            resp = _call_generation([{"role": "user", "content": prompt}], model=model)
            if not resp:
                continue
            content = resp.output.choices[0].message.content.strip()
            info = _parse_json(content)
            if isinstance(info, dict):
                try:
                    delta = int(info.get("delta", 0))
                except (TypeError, ValueError):
                    delta = 0
                delta = max(-8, min(8, delta))
                reason = str(info.get("reason", "")).strip()
                return delta, reason
            # 兜底: 兼容模型退回输出纯整数的情况
            m = re.search(r"[-+]?\d+", content)
            delta = int(m.group(0)) if m else 0
            return max(-8, min(8, delta)), ""
        return 0, ""

    # ============================================================
    # 心情系统
    # ============================================================
    def get_mood(self):
        return self._mood_label()

    def get_memory_profile(self):
        """返回"她眼中的你": 关系状态 + 关于用户的事实库 + 她对用户的印象画像(供前端记忆卡展示)"""
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

    def get_last_seen(self):
        """最近一次互动的日期(YYYY-MM-DD), 从未互动则为空串"""
        return self._last_seen

    def get_last_proactive(self):
        """各类型主动消息最近触发记录(供总控判断节日/生日问候是否已发)"""
        return dict(self._last_proactive)

    def _mood_label(self):
        v = self._mood
        if v >= 5:
            return "雀跃"
        if v >= 2:
            return "愉快"
        if v <= -5:
            return "低落"
        if v <= -2:
            return "烦闷"
        return "平静"

    def _mood_context(self):
        return (
            f"【你此刻的心情】{self._mood_label()}（情绪值 {self._mood:+.0f}）。\n"
            "心情会影响你的语气和措辞，但不要每句话都把心情直白地说出来。"
        )

    def _apply_mood(self, shift):
        """单轮对话带来的情绪变化(幅度受 MOOD_SHIFT_CLAMP 限制)"""
        try:
            shift = int(shift)
        except (TypeError, ValueError):
            return
        shift = max(-config.MOOD_SHIFT_CLAMP, min(config.MOOD_SHIFT_CLAMP, shift))
        self._mood = max(config.MOOD_MIN, min(config.MOOD_MAX, self._mood + shift))
        self._save_state()

    def _apply_mood_decay(self, days):
        """长期不互动时, 情绪向平静(0)回落"""
        if days <= 0:
            return
        step = config.MOOD_DECAY_PER_DAY * days
        if self._mood > 0:
            self._mood = max(0.0, self._mood - step)
        elif self._mood < 0:
            self._mood = min(0.0, self._mood + step)

    # ============================================================
    # 称呼随关系演进 + 关系张力(冷战/和好)
    # ============================================================
    def _nickname_context(self, nickname):
        """根据关系阶段告诉角色该怎样称呼用户(称呼随关系演进)。"""
        stage = self._favor_stage()
        name = (nickname or "").strip()
        if not name:
            if stage == "陌生":
                return "【称呼】对方还没告诉你名字，用「你」或合适的礼貌称呼，保持克制。"
            return "【称呼】对方还没告诉你名字，用「你」或自然亲昵一点的称呼。"
        if stage == "陌生":
            return f"【称呼】对方的名字是「{name}」。你们还不熟，用全名或礼貌的「你」称呼，保持克制。"
        if stage == "熟悉":
            return f"【称呼】对方的名字是「{name}」。关系熟了，可以自然、亲切地叫对方的名字。"
        if stage == "亲近":
            return f"【称呼】对方的名字是「{name}」。关系亲近，可以用去姓、叠字等更亲昵的称呼。"
        return f"【称呼】对方的名字是「{name}」。你们很亲密，用最自然亲昵的称呼（去姓/叠字/专属昵称）。"

    def _tension_label(self):
        t = self._tension
        if t >= 7:
            return "很生气"
        if t >= config.TENSION_COLD:
            return "冷战"
        if t >= 1:
            return "有点介意"
        return "正常"

    def tension_label(self):
        """当前关系张力标签(正常/有点介意/冷战/很生气), 供前端展示。"""
        return self._tension_label()

    def dynamic_quote(self):
        """角色卡语录(随心情/关系阶段/状态动态化)。"""
        q = config.NPC_QUOTES.get(self.agent_id) or {}
        if self.current_status().get("sleepy") and q.get("sleepy"):
            return q["sleepy"]
        if self._tension >= config.TENSION_COLD and q.get("cold"):
            return q["cold"]
        mood = self._mood_label()
        mood_q = (q.get("mood") or {}).get(mood)
        if mood != "平静" and mood_q:
            return mood_q
        stage_q = (q.get("stage") or {}).get(self._favor_stage())
        if stage_q:
            return stage_q
        return (q.get("mood") or {}).get("平静", "")

    def _tension_context(self):
        """把当前关系张力写成提示词片段, 冷战/介意时让角色语气相应变冷。"""
        if self._tension >= config.TENSION_COLD:
            return (
                "【你此刻对对方的态度】你还在生对方的气（冷战），语气冷淡、疏离、话少，"
                "不会主动示好；但如果对方真诚道歉或关心，你可以慢慢软化。"
            )
        if self._tension >= 1:
            return "【你此刻对对方的态度】你心里有点介意之前的事，语气比平时淡一点，但不必一直记着。"
        return ""

    def _apply_tension(self, delta):
        """根据本轮好感度变化更新关系张力; 返回是否从冷战状态"和好"。"""
        old = self._tension
        if delta <= -4:
            self._tension += config.TENSION_OFFENSE
        elif delta <= -1:
            self._tension += config.TENSION_MILD_OFFENSE
        elif delta >= 5:
            self._tension += config.TENSION_WARM_SOOTHE
        elif delta >= 3:
            self._tension += config.TENSION_SOOTHE
        self._tension = max(0, min(config.TENSION_MAX, self._tension))
        return old >= config.TENSION_COLD and self._tension < config.TENSION_COLD

    # ============================================================
    # 世界模拟: 作息状态 + 低概率随机事件(不在场时的生命感)
    # ============================================================
    def current_status(self):
        """当前作息状态(上课中/社团活动中/睡觉中/空闲等), 供前端展示。"""
        slot = resolve_schedule(self.agent_id)
        return {
            "activity": slot.get("activity", ""),
            "label": slot.get("label", ""),
            "busy": bool(slot.get("busy", False)),
            "sleepy": bool(slot.get("sleepy", False)),
        }

    def schedule_today(self):
        """今日行程(带当前时段高亮), 供前端角色卡展示。"""
        return format_schedule(self.agent_id)

    def status_context(self):
        """把当前作息状态写成提示词片段, 注入对话: 忙时自然说"晚点回你", 深夜带睡意。"""
        st = self.current_status()
        label = st.get("label", "") or ""
        activity = st.get("activity", "") or ""
        if st.get("sleepy"):
            return (
                f"【你此刻的状态】你现在{activity}（{label}），已经很晚了。"
                "深夜回复要迷迷糊糊、轻柔简短，可以带一点睡意或迷迷糊糊的语气。"
            )
        if st.get("busy"):
            return (
                f"【你此刻的状态】你现在{activity}（{label}），有点忙，但你还是抽空回复了对方。"
                "回复要简短，可以自然带一句自己在忙、是抽空回的消息，"
                "例如'我在上课，偷偷回你一下'；要显得真心、不敷衍。"
            )
        return f"【你此刻的状态】你现在{activity}（{label}），比较有空，可以正常聊。"

    def life_events_context(self):
        """今天/最近发生在角色身上的随机事件, 注入对话, 让世界有"不在场时的生命感"。"""
        today = date.today().isoformat()
        ev = self.life.today_event(today)
        if ev:
            return (
                f"【今天发生在你身上的事】{ev.get('text', '')}。"
                "你会在对话中自然流露这种心情，但不要生硬地原句复述。"
            )
        recents = self.life.recent_events(k=2)
        if recents and recents[0].get("date") != today:
            parts = "；".join(e.get("text", "") for e in recents if e.get("text"))
            if parts:
                return f"【你最近经历的事】{parts}。如果话题相关，可以自然提起。"
        return ""

    def sync_life(self):
        """惰性推进世界: 回填离开期间的日子、掷随机事件, 并把事件带来的心情变化应用到自身。"""
        with self._lock:
            try:
                events = self.life.advance()
            except Exception as e:
                print(f"[{self.name}] 世界模拟推进失败: {e}")
                return
            if not events:
                return
            for ev in events:
                self._apply_mood(int(ev.get("mood", 0)))
            self._unlock_cb("life_event")

    # ============================================================
    # 深夜延迟回复: 睡着时把消息存起来, 醒来后再补回(折中版)
    # ============================================================
    def _queue_pending_reply(self, prompt):
        """深夜消息入队, 等她醒来后补回。调用方已持有 self._lock。"""
        self._pending_replies.append({"question": prompt, "ts": time.time()})
        self._pending_replies = self._pending_replies[-config.SLEEP_PENDING_MAX:]

    def has_pending_replies(self):
        return bool(self._pending_replies)

    def generate_delayed_reply(self):
        """她醒来后, 为深夜积压的消息补一条主动回复; 生成失败则保留待回复、下次再试。"""
        with self._lock:
            pending = list(self._pending_replies)
        if not pending:
            return ""
        questions = "；".join(str(p.get("question", "")).strip() for p in pending if p.get("question"))
        if not questions:
            return ""
        persona = self._current_persona()
        stage = self._favor_stage()
        prompt = f"""你是「{self.name}」。昨晚对方给你发消息时你已经睡着了, 现在你刚醒来, 想补回对方。对方的原话：
{questions}

请写一句补回的回复，要求：
- 先自然带一句「昨晚睡着了 / 刚醒」之类的话，再回应对方
- 像真人发消息一样简短，1~2 句话、60 字以内
- 符合你的人设、当前关系(好感度 {self._favor}/100，{stage})和心情({self._mood_label()})
- 不要 markdown、不要列表
- 只输出这句话本身，不要任何前缀或解释

【你的核心人格】
{persona}
"""
        resp = _call_generation([{"role": "user", "content": prompt}])
        if not resp:
            return ""
        content = resp.output.choices[0].message.content.strip()
        if not content:
            return ""
        with self._lock:
            # 成功后才清掉本次已消费的消息(保留生成期间新到的消息)
            # 并发保护: 若生成期间 pending 已被其他线程消费(长度不一致且头部内容对不上),
            # 说明已有人补过这条延迟回复, 放弃本次结果, 避免重复消息
            if (len(self._pending_replies) != len(pending)
                    and self._pending_replies[:len(pending)] != pending):
                return ""
            self._pending_replies = self._pending_replies[len(pending):]
            self._save_state()
        return content

    # ============================================================
    # 剧情消息: 关系里程碑 / 和好
    # ============================================================
    def generate_milestone_message(self, from_stage, to_stage):
        """关系跨阶段时, 生成一段角色专属的"关系加深"心声(2~3句); 失败返回空串。"""
        persona = self._current_persona()
        prompt = f"""你是「{self.name}」。你和用户的关系刚刚从「{from_stage}」进阶到「{to_stage}」（好感度 {self._favor}/100）。这是你们关系的一次重要转折。

请以「{self.name}」的口吻写一段她此刻最真实的心声/独白（2~3 句话、80 字以内）：
- 体现这个阶段该有的亲疏与情绪
- 符合你的核心人格
- 不要 markdown、不要列表
- 只输出这段话本身，不要任何前缀或引号

【你的核心人格】
{persona}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)
        if resp:
            return resp.output.choices[0].message.content.strip()
        return ""

    def generate_reconcile_message(self):
        """冷战和好时, 生成一句角色说的话(1~2句); 失败返回空串。"""
        persona = self._current_persona()
        prompt = f"""你是「{self.name}」。之前你对用户有些生气、在冷战，但对方真诚的关心/道歉让你心软了，你们和好了。

请以「{self.name}」的口吻写一句和好时说的话（1~2 句、40 字以内）：
- 符合你的性格（可以有点别扭、释然、温柔）
- 不要 markdown、不要列表
- 只输出这句话本身

【你的核心人格】
{persona}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)
        if resp:
            return resp.output.choices[0].message.content.strip()
        return ""

    # ============================================================
    # 检索意图 & 最近对话
    # ============================================================
    def _build_recall_query(self, prompt):
        """用"当前消息 + 最近上下文 + 最近话题标签"做检索, 更贴近人脑的联想式激活"""
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
        """最近对话文本(不含当前这条用户消息, 供认知步骤使用)"""
        recent = [m for m in self.history[1:]
                  if m.get("role") in ("user", "assistant") and not m.get("interrupted")][-(n + 1):-1]
        return "\n".join(f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in recent)

    # ============================================================
    # 认知步骤: 读心 + 内心独白 + 心情变化(便宜模型, 每轮一次)
    # ============================================================
    def _cognize(self, prompt, recalled, facts, world_context=None):
        """回复前的"内心思考": 推断用户状态、形成内心独白、判断心情变化与输出风格。"""
        if not config.COGNITION_ENABLED:
            return {}
        persona = self._current_persona()
        stage = self._favor_stage()
        mood = self._mood_label()
        profile = self._load_memory().get("user_profile", "")
        history_text = self._recent_history_text(config.COGNITION_HISTORY_TURNS)
        recalled_text = "\n".join(f"- {r}" for r in recalled) or "（无）"
        facts_text = "\n".join(f"- {f}" for f in facts) or "（无）"
        world = world_context or ""

        prompt_cog = f"""你是「{self.name}」。在回复用户之前, 先在心里快速过一遍(这些只是你内心的思考, 不要直接对用户说出来)。请严格输出JSON(不要任何其他文字):

{{
  "user_emotion": "用户此刻的情绪(开心/低落/焦虑/生气/平静/疲惫 等, 简短)",
  "user_intent": "用户想做什么(倾诉/求助/闲聊/探讨/抬杠/安慰 等, 简短)",
  "topic": "本次话题关键词, 3~8字",
  "topic_shift": false,
  "internal_thought": "你的内心独白: 你怎么理解对方、你此刻的感受、打算用什么态度回应、要不要用到某段回忆。{config.COGNITION_THOUGHT_MAX_CHARS}字以内",
  "mood_shift": 0,
  "need_clarify": false,
  "style": "chat"
}}

字段说明:
- mood_shift: 这轮对话对你心情的影响, -3~+3 整数(被冒犯/难过为负, 被关心/有趣为正, 普通闲聊为0)
- topic_shift: 本轮话题是否已明显偏离上一轮对话的话题, 是填 true, 否则 false
- need_clarify: 对方意图模糊、你无法确定时填 true, 否则 false
- style: 对方只是闲聊/寒暄填 "chat"; 对方在请教、求助、要你讲解或展开说明时填 "explain"

【你的核心人格】
{persona}

【当前关系】好感度 {self._favor}/100（{stage}）
【你此刻的心情】{mood}

【你对用户的印象】
{profile or "暂无"}

【关于用户的长期事实】
{facts_text}

【相关久远记忆】
{recalled_text}

{world}

【最近对话】
{history_text or "（无）"}

【用户刚说的话】
{prompt}
"""
        resp = _call_generation([{"role": "user", "content": prompt_cog}], model=config.COGNITION_MODEL)
        if not resp:
            return {}
        info = _parse_json(resp.output.choices[0].message.content)
        return info if isinstance(info, dict) else {}

    # ============================================================
    # 系统提示词构造: 人格内核 + 行为准则 + 关系 + 心情 + 今日世界 + 画像 + 事实 + 独白 + 召回记忆
    # ============================================================
    def _build_system_message(self, recalled, facts=None, world_context=None,
                              greet_hint=None, cognition=None,
                              social_context=None, user_nickname=None):
        cognition = cognition or {}
        memory = self._load_memory()
        parts = [
            self._current_persona(),
            COMMON_BEHAVIOR_RULES,
            self._favor_context(),
            self._mood_context(),
        ]
        tension_ctx = self._tension_context()
        if tension_ctx:
            parts.append(tension_ctx)
        if user_nickname is not None:
            parts.append(self._nickname_context(user_nickname))
        if social_context:
            parts.append(social_context)
        if world_context:
            parts.append(world_context)
        if greet_hint:
            parts.append(greet_hint)
        if memory.get("user_profile"):
            parts.append(f"【你对用户的印象(画像)】\n{memory['user_profile']}")
        if facts:
            parts.append("【你对用户的长期了解(事实)】\n" + "\n".join(f"- {f}" for f in facts))
        if cognition.get("user_emotion") or cognition.get("user_intent"):
            judge = f"情绪={cognition.get('user_emotion', '')}，意图={cognition.get('user_intent', '')}"
            if cognition.get("topic"):
                judge += f"，话题={cognition.get('topic', '')}"
            parts.append("【你对用户此刻的判断】" + judge + "。据此把握回应的重点与语气。")
        if cognition.get("internal_thought"):
            parts.append(
                "【你此刻的内心活动】" + str(cognition["internal_thought"])
                + "\n（这是你心里的想法，作为你的说话依据，但不要原样念出来。）"
            )
        if memory.get("conversation_summary"):
            parts.append(f"【历史对话摘要】\n{memory['conversation_summary']}")
        if recalled:
            parts.append("【相关久远记忆】\n" + "\n".join(f"- {r}" for r in recalled))
        if cognition.get("need_clarify"):
            parts.append("【澄清提示】你还不确定对方想表达什么，可以先用一句话向对方确认意图，而不是急着给出答案。")
        # 输出格式约束放在最后, 优先级最高
        style = cognition.get("style") or "chat"
        parts.append(_format_rule(style))
        return "\n\n".join(parts)

    # ============================================================
    # 记忆提取(滚动式: 画像 + 摘要 + 人格成长 + 稳定事实)
    # ============================================================
    @staticmethod
    def _clean_text(content):
        if not content:
            return ""
        return str(content).strip()[:config.MEMORY_INPUT_MAX_CHARS]

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
{{"user_profile":"...","conversation_summary":"...","persona_growth":"...","facts":["...","..."],"tags":["...","..."]}}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if not resp:
            print(f"[{self.name}] 记忆提取失败")
            return {}
        info = _parse_json(resp.output.choices[0].message.content)
        if not isinstance(info, dict):
            info = {}
        facts = info.get("facts", []) or []
        if not isinstance(facts, list):
            facts = [facts]
        tags = info.get("tags", []) or []
        if not isinstance(tags, list):
            tags = [tags]
        return {
            "user_profile": (info.get("user_profile") or "").strip(),
            "conversation_summary": (info.get("conversation_summary") or "").strip(),
            "persona_growth": (info.get("persona_growth") or "").strip(),
            "facts": [str(x).strip() for x in facts if str(x).strip()],
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
        """需要压缩时, 丢到后台线程执行(调用方已持有 self._lock)"""
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
        except Exception as e:
            print(f"[{self.name}] 异步压缩失败: {e}")
        finally:
            self._compacting = False

    def _apply_compact(self, profile, summary, growth, facts, tags=None):
        """把结晶记忆/事实/人格成长落盘 + 有价值对话写入长期记忆 + 截断历史(必须在锁内调用)"""
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
                if (m.get("content") and _worth_remembering(m.get("content", ""))
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

    # ============================================================
    # 深睡(隔夜整理): 单天会话总结成日历史 + 刷新短期聊天
    # 惰性触发: 无定时器, 用户上线互动后检查; 无新互动则免做(角色"睡着")
    # ============================================================
    def _load_daily_log(self):
        try:
            if os.path.exists(self.daily_log_file):
                with open(self.daily_log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[{self.name}] 日历史加载失败: {e}")
        return []

    def _save_daily_log(self, logs):
        try:
            with open(self.daily_log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.name}] 日历史保存失败: {e}")

    def get_daily_log(self):
        """返回该角色的日历史(按日期升序), 供前端回看"""
        return self._load_daily_log()

    @staticmethod
    def _msg_date(m):
        """消息的日期(YYYY-MM-DD); 无时间戳则返回空串"""
        ts = m.get("ts")
        if not ts:
            return ""
        try:
            return date.fromtimestamp(float(ts)).isoformat()
        except Exception:
            return ""

    def _should_dream(self, old_msgs):
        """是否该做隔夜整理: 有旧会话 + 互动已结束 + (睡眠窗口内 或 超期兜底)。
        不按自然日切分——彻夜长谈(跨午夜)算同一段会话, 一次总结。"""
        if not old_msgs:
            return False
        now = time.time()
        # 互动已结束: 旧会话最后一条消息距今超过 IDLE 分钟(否则延后)
        last_ts = max(float(m["ts"]) for m in old_msgs)
        if (now - last_ts) < config.DREAM_IDLE_MINUTES * 60:
            return False
        # 睡眠窗口判断: 不在窗口内且未超期 → 延后
        if not WorldClock.is_sleep_window():
            days_overdue = (now - self._last_dream_ts) / 86400.0 if self._last_dream_ts else 999.0
            if days_overdue < config.DREAM_MAX_DELAY_DAYS:
                return False
        return True

    def _maybe_dream_async(self):
        """惰性触发深睡: 满足条件则后台执行(不阻塞对话)。
        在"当前消息 append 之前"调用, 因此快照里只有上一段会话, 不含正在进行的这轮。"""
        if not config.DREAM_ENABLED or self._dreaming:
            return
        old_msgs = [m for m in self.history if m.get("role") in ("user", "assistant") and m.get("ts")]
        if not old_msgs:
            return
        if not self._should_dream(old_msgs):
            return
        boundary = max(float(m["ts"]) for m in old_msgs)
        self._dreaming = True
        threading.Thread(target=self._run_dream, args=(list(old_msgs), boundary), daemon=True).start()

    def maybe_dream(self):
        """用户上线时由总控调用, 惰性触发隔夜整理"""
        self._maybe_dream_async()

    def _summarize_session(self, msgs):
        """把一段连续会话(上次深睡以来, 可能跨午夜)总结成一条日历史记录; date 取会话开始的日期"""
        if not msgs:
            return None
        day = self._msg_date(msgs[0]) or date.today().isoformat()
        lines = "\n".join(f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in msgs)
        prompt = f"""你是「{self.name}」的梦境整理助手。下面是用户与你的一段连续会话(可能跨午夜, 属同一次长谈)。请总结成一条日历史, 严格输出JSON对象(不要任何其他文字):

{{"date":"{day}","summary":"这段互动的总结, 100字以内","new_understandings":["角色对用户产生的新认识"],"highlights":["这段会话中最重要的事件"]}}

要求:
1. date 固定用 "{day}"。
2. summary 概括"这段时间用户和角色发生了什么新互动"; new_understandings 记录"角色对用户的新认识"(可空数组); highlights 记录"最重要的事件"(可空数组)。
3. 只依据对话内容, 不要编造。

【会话内容】
{lines}"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.DREAM_MODEL)
        if not resp:
            return None
        content = resp.output.choices[0].message.content
        content = re.sub(r"```(?:json)?", "", content)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            record = json.loads(match.group(0)) if match else {}
        except Exception:
            record = {}
        return record if isinstance(record, dict) and record.get("date") else None

    def _append_daily_log(self, records):
        logs = self._load_daily_log()
        existing = {l.get("date") for l in logs if l.get("date")}
        for r in records:
            d = r.get("date")
            if not d or d in existing:
                continue
            logs.append({
                "date": d,
                "summary": str(r.get("summary", "")).strip(),
                "new_understandings": [str(x).strip() for x in (r.get("new_understandings") or []) if str(x).strip()],
                "highlights": [str(x).strip() for x in (r.get("highlights") or []) if str(x).strip()],
                "tags": [str(x).strip() for x in (r.get("tags") or []) if str(x).strip()],
                "favorability": r.get("favorability"),
                "mood": str(r.get("mood", "")).strip(),
            })
        logs.sort(key=lambda x: x.get("date", ""))
        self._save_daily_log(logs)

    def _run_dream(self, old_msgs, boundary):
        """深睡执行: 总结旧会话→写日历史→清空旧会话→软遗忘"""
        try:
            record = self._summarize_session(old_msgs)
            if record:
                # 补充当日话题标签/好感度快照, 供"回忆相册"展示
                topics = []
                for m in old_msgs:
                    t = m.get("topic")
                    if t and str(t).strip() and str(t).strip() not in topics:
                        topics.append(str(t).strip())
                record["tags"] = topics[:6]
                record["favorability"] = self._favor
                record["mood"] = self._mood_label()
                with self._lock:
                    self._append_daily_log([record])
                    # 清空"旧会话"(ts <= boundary); 保留正在进行的这轮(ts > boundary)与无日期消息
                    self.history = [m for m in self.history
                                    if m.get("role") == "system" or not m.get("ts") or float(m["ts"]) > boundary]
                    self._save_history()
                    self._last_dream_ts = time.time()
                    self._save_state()
                print(f"[{self.name}] 隔夜整理完成: {record.get('date')}")
                self._unlock_cb("first_dream")
            # 深睡附加: 对长期记忆做一次软遗忘, 避免只增不减
            try:
                self.long_term.forget(self.name)
            except Exception as e:
                print(f"[{self.name}] 深睡软遗忘异常: {e}")
        except Exception as e:
            print(f"[{self.name}] 隔夜整理异常: {e}")
        finally:
            self._dreaming = False

    # ============================================================
    # 对话(流式)
    # ============================================================
    def chat_stream(self, prompt, world_context=None, greet_hint=None,
                    social_context=None, user_nickname=None):
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
                self._last_seen = date.today().isoformat()
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
                    recalled, facts, world_context, greet_hint, cognition,
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
                    "content": messages[-1]["content"] + "\n\n" + _tail_hint(style),
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

    # ============================================================
    # 话题开场卡: 为"不知道聊什么"的用户生成可点击的开场话题
    # ============================================================
    def suggest_topics(self):
        """生成可点击的开场话题(便宜模型, 带TTL缓存)。

        一次性生成一批(TOPIC_SUGGEST_COUNT个), 前端本地分批轮换;
        缓存未过期时直接返回, 避免每次切换角色/换一批都打一次模型。
        """
        now = time.time()
        cache = self._topics_cache
        if cache and cache.get("topics") and (now - cache.get("ts", 0)) < config.TOPIC_CACHE_TTL:
            return list(cache["topics"])
        topics = self._generate_topics()
        if topics:
            self._topics_cache = {"ts": now, "topics": list(topics)}
        return topics

    def _generate_topics(self):
        """调用便宜模型生成一批自然、可点击的开场话题"""
        persona = self._current_persona()
        facts_text = "、".join(self.fact_memory.all()) or "暂无"
        recent = self._recent_history_text(4)
        prompt = f"""你是「{self.name}」。用户想和你聊天但不知道聊什么。请想 {config.TOPIC_SUGGEST_COUNT} 个自然、有吸引力、用户可以直接发给你的一句话开场话题。严格输出JSON数组(不要任何其他文字): ["话题1","话题2","话题3", ...]

要求:
1. 每个话题一句话, 20 字以内, 口语化, 像用户会直接发出来的话。
2. 结合你的性格、当前关系(好感度 {self._favor}/100, {self._favor_stage()})、心情({self._mood_label()})、你对用户的了解({facts_text})。
3. 不要重复最近聊过的内容; 话题要能自然展开对话。

【你的核心人格】
{persona}

【最近对话】
{recent or "（无）"}
"""
        resp = _call_generation([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)
        if not resp:
            return []
        content = resp.output.choices[0].message.content
        content = re.sub(r"```(?:json)?", "", content)
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()][:config.TOPIC_SUGGEST_COUNT]
            except Exception:
                pass
        return []

