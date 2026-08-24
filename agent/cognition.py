"""
认知步骤(CognitionMixin): 回复前的"读心 + 内心独白 + 心情变化 + 输出风格" + 系统提示词构造。

- _cognize: 每轮回复前用便宜模型先"想"一遍(理解对方/态度/要不要用回忆)
- _build_system_message: 组装 人格 + 准则 + 关系 + 心情 + 世界 + 画像 + 事实 + 独白 + 记忆
"""
import config
import json_utils
import llm
import prompts


class CognitionMixin:
    """读心 / 内心独白 / 系统提示词构造。混入 Agent。"""

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
        resp = llm.call([{"role": "user", "content": prompt_cog}], model=config.COGNITION_MODEL)
        if not resp:
            return {}
        info = json_utils.parse_object(resp.output.choices[0].message.content)
        return info if isinstance(info, dict) else {}

    # ============================================================
    # 系统提示词构造: 人格内核 + 行为准则 + 关系 + 心情 + 今日世界 + 画像 + 事实 + 独白 + 召回记忆
    # ============================================================
    def _build_system_message(self, recalled, facts=None, world_context=None,
                              cognition=None, social_context=None, user_nickname=None):
        cognition = cognition or {}
        memory = self._load_memory()
        parts = [
            self._current_persona(),
            prompts.COMMON_BEHAVIOR_RULES,
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
        if memory.get("user_profile"):
            parts.append(f"【你对用户的印象(画像)】\n{memory['user_profile']}")
        if facts:
            parts.append("【你对用户的长期了解(事实)】\n" + "\n".join(f"- {f}" for f in facts))
            parts.append("【记忆冲突处理】当上面的长期事实与【相关久远记忆】冲突时，以长期事实为准（事实是已更新的权威结论，久远记忆可能是过时的旧证据）。")
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
        parts.append(prompts.format_rule(style))
        return "\n\n".join(parts)
