"""
话题开场卡(TopicsMixin): 为"不知道聊什么"的用户生成可点击的开场话题(带 TTL 缓存)。
"""
import time

import config
import json_utils
import llm


class TopicsMixin:
    """开场话题生成 + 缓存。混入 Agent。"""

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
        """调用便宜模型生成一批自然、可点击的开场话题。"""
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
        resp = llm.call([{"role": "user", "content": prompt}], model=config.COGNITION_MODEL)
        if not resp:
            return []
        arr = json_utils.parse_array(resp.output.choices[0].message.content)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()][:config.TOPIC_SUGGEST_COUNT]
        return []
