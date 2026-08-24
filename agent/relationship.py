"""
关系状态(RelationshipMixin): 好感度 / 关系阶段 / 心情 / 关系张力(冷战·和好) / 称呼 / 角色卡语录。

- 好感度: 每轮后台即时结算(-8~+8), 随时间轻微衰减
- 心情: 随对话波动、随时间向平静回落
- 张力: 被冒犯累积→冷战; 被真诚关心→和好
"""
import re

import config
import json_utils
import llm
import time_utils


class RelationshipMixin:
    """好感度 / 心情 / 关系张力 / 称呼 / 语录。混入 Agent。"""

    # ============================================================
    # 好感度
    # ============================================================
    def get_favor(self):
        with self._lock:
            return self._favor

    def get_stage(self):
        """当前关系阶段(陌生/熟悉/亲近/亲密), 供前端直接展示(避免前端重复阈值)。"""
        with self._lock:
            return self._favor_stage()

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
        """按距上次互动的天数轻微衰减好感度(感情需要维系, 长期冷落会变淡); 返回天数。"""
        if days is None:
            if not self._last_seen:
                return 0
            days = time_utils.days_since(self._last_seen)
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
            self._last_seen = time_utils.today_iso()
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
        # 仅 45(熟悉→亲近)/70(亲近→亲密) 两个关键阶段跨越才触发关系里程碑剧情(与成就/文档口径一致)
        if delta > 0 and ((new_stage == "亲近" and old_stage == "熟悉")
                          or (new_stage == "亲密" and old_stage == "亲近")):
            milestone = {"from": old_stage, "to": new_stage, "favorability": self._favor}
        return {"delta": delta, "reason": reason, "favorability": self._favor,
                "stage": new_stage, "mood": self._mood_label(), "milestone": milestone,
                "tension": self._tension, "tension_label": self._tension_label(), "reconciled": reconciled}

    def _judge_favor_with_reason(self, recent):
        """判断最近一轮对话对好感度的影响(-8~+8)及一句话理由; 用便宜模型, 失败退回主模型。"""
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
            resp = llm.call([{"role": "user", "content": prompt}], model=model)
            if not resp:
                continue
            content = resp.output.choices[0].message.content.strip()
            info = json_utils.parse_object(content)
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
        with self._lock:
            return self._mood_label()

    def get_last_seen(self):
        """最近一次互动的日期(YYYY-MM-DD), 从未互动则为空串。"""
        with self._lock:
            return self._last_seen

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
        """单轮对话带来的情绪变化(幅度受 MOOD_SHIFT_CLAMP 限制)。"""
        try:
            shift = int(shift)
        except (TypeError, ValueError):
            return
        shift = max(-config.MOOD_SHIFT_CLAMP, min(config.MOOD_SHIFT_CLAMP, shift))
        self._mood = max(config.MOOD_MIN, min(config.MOOD_MAX, self._mood + shift))
        self._save_state()

    def _apply_mood_decay(self, days):
        """长期不互动时, 情绪向平静(0)回落。"""
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
        with self._lock:
            return self._tension_label()

    def dynamic_quote(self):
        """角色卡语录(随心情/关系阶段/状态动态化)。"""
        with self._lock:
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
