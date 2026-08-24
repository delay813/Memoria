# 心忆 · Memoria 架构说明（结项版）

> 本文档描述重构后的模块布局与关键设计决策，作为后续维护的入口。

## 一、分层

```
用户浏览器 (static/chat.html)
      │  SSE 流式 / JSON
      ▼
 FastAPI (server.py)
      │
      ▼
 Orchestrator (总控)
   ├── WorldClock (narrator.py)   现实时间 / 节日 / 每日问候
   └── Agent × N                  每个 NPC 完全隔离
```

单个 `Agent` 由一组职责单一的 **mixin** 组合而成（全部方法作用于同一实例，`self` 共享状态）：

| 模块 | 职责 |
|---|---|
| `agent.py` | 骨架：状态/历史初始化、持久化、`reset`、流式对话编排 |
| `persona.py` | 人格内核（蒸馏 / 成长 / 重写 + 冻结锚点） |
| `relationship.py` | 好感度 / 关系阶段 / 心情 / 关系张力 / 称呼 / 语录 |
| `memory.py` | 记忆抽取与压缩（画像 + 摘要 + 事实 + 长期记忆写入） |
| `dream.py` | 深睡 · 隔夜整理（回忆日记） |
| `proactive.py` | 主动消息 / 延迟回复 / 里程碑·和好剧情 |
| `cognition.py` | 读心 / 内心独白 / 系统提示词构造 |
| `world.py` | 作息状态 / 随机事件 / 命运大纲接入 |
| `topics.py` | 开场话题生成 + TTL 缓存 |

## 二、共享基础模块（消除复制粘贴）

| 模块 | 职责 | 曾重复出现于 |
|---|---|---|
| `llm.py` | 统一 `Generation.call` 重试 / `call_text` | `agent._call_generation`、`longterm_memory._call_small` |
| `json_utils.py` | LLM 输出 JSON 稳健解析（对象/数组） | `agent._parse_json`、`forget`、`_summarize_session`、`_generate_topics` |
| `text_utils.py` | 中英文分词 / markdown 清洗 / 记忆价值判断 | `FactMemory._tokenize`、`SimpleBM25._tokenize`、`agent._plain_text` |
| `time_utils.py` | 现实日期 / 天数差 / 跨午夜区间 | `date.today()`×10、`fromisoformat`×4、`_in_slot` vs `is_sleep_window` |
| `prompts.py` | 共享行为准则 + 输出格式约束 | `agent.py` 顶部常量 |

## 三、数据外置

- **`agents.json`**：每个角色的完整配置（人设 + `schedule` 作息 + `quotes` 语录 + `random_events` 专属事件 + `avatar`/`card`/`en`/`tags`/`theme` 前端视觉资料）。新增角色只改这里。
- **`npc_common.json`**：兜底作息 `default_schedule` + 通用随机事件池 `random_events`。
- **`config.py`** 只保留「可调参数 + 路径 + 模型名」，角色/事件/语录等**数据**全部外置到 JSON。
- 前端角色视觉资料与主题色由 `/agents` 接口下发，`chat.html` 不再硬编码（`CHARACTERS`/`THEMES` 已移除）。

## 四、关键设计决策（为什么保留某些看似重叠的结构）

- **`facts.json`（稳定事实） vs `user_profile`（画像）**：两者粒度不同——事实是「可检索的短结论」，画像是「一段叙事印象」，且画像服务于日记/剧本等叙事场景。故保留，但在 `_extract_memory` 提示词中明确各自边界，避免交叉污染。
- **`conversation_summary`（滚动摘要） vs `daily_log`（回忆日记）**：前者是「注入每轮上下文的近期摘要」，后者是「按天归档、仅供回看的日记」。产物不同，故保留。
- **软遗忘统一入口**：`LongTermMemory._mark_forgotten(pid, penalty, hard_delete)` 同时服务 `forget()`（低价值遗忘）与 `suppress_conflicts()`（事实冲突），避免两处重复的「标记 forgotten + 降重要度」逻辑。

## 五、成本 / 延迟开关

- `COGNITION_ENABLED`：关闭「读心 + 内心独白」（省一次模型调用）。
- `MEMORY_RERANK_ENABLED`：关闭 rerank 精排（省一次 API 调用，由时近性/重要性/访问频率三维继续排序）。
- 好感度结算每轮同步执行（需要即时反馈 + 里程碑/和好检测）；如后续需要进一步降延迟，可改为异步并让前端从 `/inbox` 轮询补收剧情卡片。
