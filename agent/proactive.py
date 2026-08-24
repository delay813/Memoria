"""
主动消息(ProactiveMixin): 收件箱 + 节日/想念/延迟回复 + 关系里程碑/和好剧情消息。

- 主动消息惰性生成: 用户上线时才判断是否该发(无定时器)
- 深夜睡着时把消息入队, 醒来后补回一条延迟回复
- 里程碑/和好剧情消息写入收件箱, 前端弹出卡片
"""
import json
import os
import time

import config
import llm
import storage
import time_utils


class ProactiveMixin:
    """主动消息收件箱 / 节日问候 / 想念 / 延迟回复 / 剧情消息。混入 Agent。"""

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
            storage.save_json(self.inbox_file, inbox)
        except Exception as e:
            print(f"[{self.name}] 收件箱保存失败: {e}")

    def get_inbox(self):
        with self._lock:
            return self._load_inbox()

    def unread_count(self):
        with self._lock:
            return len([m for m in self._load_inbox() if not m.get("read")])

    def mark_read(self):
        with self._lock:
            inbox = self._load_inbox()
            for m in inbox:
                m["read"] = True
            self._save_inbox(inbox)

    def add_proactive(self, kind, content):
        """追加一条主动消息, 写入对话历史。"""
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

    def _should_send_checkin_locked(self, now):
        """判断是否该发一条普通主动消息(调用方须已持有 self._lock)。"""
        # 冷战期间不主动找对方
        if self._tension >= config.TENSION_COLD:
            return False
        days = time_utils.days_since(self._last_seen) if self._last_seen else 0
        # 离开不足N天(今天刚互动过)不发
        if days < config.PROACTIVE_MIN_AWAY_DAYS:
            return False
        # 冷却: 两次普通主动消息的最小间隔
        last = self._last_proactive.get("checkin", 0) or 0
        if now - last < config.PROACTIVE_COOLDOWN_DAYS * 86400:
            return False
        # 命运大纲(剧本)决定今天要不要主动; 有当天大纲时以其为准, 无大纲退回数值判断
        script = self.life.get_script() or {}
        if script.get("date") == time_utils.today_iso():
            if script.get("reach_out") is False:
                return False
            if script.get("reach_out") is True:
                return True
        # 离开越久, 主动意愿越低(欲望随时间衰减)
        intent = self._favor - days * config.PROACTIVE_DECAY_PER_DAY
        return intent >= config.PROACTIVE_INTENT_THRESHOLD

    def try_claim_checkin(self, now):
        """原子判断+占位普通主动消息(想念): 满足意愿且冷却已过才返回 True, 防并发轮询重复生成。"""
        with self._lock:
            if not self._should_send_checkin_locked(now):
                return False
            self._last_proactive["checkin"] = now
            self._save_state()
            return True

    def try_claim_festival(self, today_iso):
        """原子判断+占位节日/生日问候: 当天尚未发过才返回 True, 防并发轮询重复发送。"""
        with self._lock:
            if self._last_proactive.get("festival") == today_iso:
                return False
            self._last_proactive["festival"] = today_iso
            self._save_state()
            return True

    def generate_proactive(self, reason):
        """生成一条符合人设与当前关系的主动消息(简短), 失败返回空串。"""
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
        return llm.call_text([{"role": "user", "content": prompt}])

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
        content = llm.call_text([{"role": "user", "content": prompt}])
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
        return llm.call_text([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)

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
        return llm.call_text([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)
