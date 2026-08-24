# 心忆 · Memoria

> 让 NPC 角色「像人一样记忆、像人一样思考」的多角色陪伴系统。

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="license">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/models-Qwen%20%7C%20DashScope-615ced" alt="models">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="framework">
</p>

**Memoria（心忆）** 是一个多角色 AI 陪伴系统。每个角色拥有**独立的人格内核、心情、好感度、语义记忆（事实库）与情景记忆（长期记忆 RAG）**，并在每次回复前先完成一次「读心 + 内心独白」的认知过程，再决定怎么说——不是背台词，而是真的「想」过之后才开口。

内置三位角色：**星野璃**（计算机系反差萌学姐）、**苏晚柠**（治愈系文学社长）、**白河祈**（清冷巫女学姐），可自由增删。

> 底层模型使用阿里云百炼（DashScope）的 Qwen 系列：`qwen-max` 做主对话，`qwen-turbo` 做辅助判断，`text-embedding-v1` 做向量化，`gte-rerank` 做重排。

---

## ✨ 特性

### 🧠 记忆：拟合人脑的记忆结构
- **语义记忆 / 情景记忆分离**：`facts.json` 存「关于用户的稳定事实」（偏好、身份、约定），`chroma_db` 存「具体发生过的事」。
- **四维加权召回**：时近性（指数衰减）＋ 重要性（情感显著性）＋ 访问频率（复习巩固）＋ 语义相似度。
- **Small-to-big + Hybrid + Rerank**：小句切分检索 → 稠密向量 + BM25 混合 → RRF 融合 → 重排精排。
- **软遗忘**：低价值记忆先被「想不起来」（不注入召回），重要度极低时才物理删除，而非粗暴硬删。
- **情感显著性**：带情绪 / 重要事件关键词的记忆写入时自动提升重要性，模拟「有情绪的事记得更牢」。
- **记忆去重（保历史合并）**：写入时用 embedding 相似度分流，只消除「真重复」；对「相似但带新信息」（如时间 / 地点 / 状态变化）做**保历史合并**——旧事实与新事实都保留，绝不覆盖「昨天在上海」这类既定信息。
- **时间标签**：召回记忆附「约 N 天前」，让角色能自然说出「你上次说……」。

### 💬 说话：拟合人脑的思考过程
- **内心独白（Two-pass）**：每轮回复前，角色先用便宜模型「在心里过一遍」（理解对方、感受、态度、要不要用某段回忆），再据此生成口语回复。
- **读心**：每轮推断用户此刻的**情绪 / 意图 / 话题**，让安慰、玩笑、拒绝都更「踩在点上」。
- **心情系统**：角色自身情绪随对话波动、随时间回落，反过来影响语气与措辞。
- **自适应输出格式**：闲聊用 40 字内的短句；对方请教 / 求助时自动切换到「讲解模式」，适度展开但依然口语化。
- **澄清能力**：意图模糊时角色会先反问确认，而不是硬答。

### 🎭 人格 & 关系
- **人格内核 + 冻结锚点**：完整人设蒸馏成「核心人格内核」，随成长重写演进，但永远忠于冻结的锚点（防漂移）。
- **好感度系统**：每轮后台即时结算（-8~+8），关系阶段（陌生 / 熟悉 / 亲近 / 亲密）影响亲疏分寸，长期冷落会轻微衰减。
- **主动消息**：节日 / 生日祝福，以及「想念」式主动问候，惰性生成。
- **世界时钟**：现实时间推进，特殊日期（节日）触发特殊对话。
- **深睡 · 隔夜整理（梦境）**：参考 OpenClaw 的 Dreaming 机制——白天「浅睡」按轮次用便宜模型快速去重；晚上「深睡」总结当天重要互动、角色对用户的新认识，沉淀为可回看的「回忆日记」，并刷新新一天的聊天框。全部惰性触发：用户不上线、无新互动就免做，角色「睡着」零消耗。

### 🌍 世界模拟（作息 + 随机事件）
- **作息 / 状态**：每个 NPC 有按小时划分的作息表（上课 / 社团 / 值班 / 睡觉 / 空闲）。前端实时展示角色此刻在做什么；对话时角色会自然带出状态——忙时会「抽空回你」（如「我在上课，偷偷回你一下」）。
- **深夜延迟回复**：深夜她睡着时**不立即回复**，而是把你的消息存起来，等她醒来（不再处于睡觉状态）再通过收件箱补回一条「昨晚睡着了，刚醒…」的主动消息。
- **低概率随机事件**：每天惰性地为每个角色掷一次随机事件（默认 25% 概率），如「今天考砸了，心情低落」「收到小礼物，很开心」，直接影响角色心情并自然流露进对话。
- **事件链**：部分事件带「后续」——「考砸了」过两天可能「成绩出来，没那么糟」，「卡 bug」次日「终于找到原因」，让世界像追剧一样有下文。
- **不在场时的生命感**：用户离开期间世界照常运转——再次上线时会回填错过的日子（最多 7 天），补上期间的随机事件与心情变化，角色会「想起」这几天发生的事。

### 💞 关系 & 可玩性
- **称呼随关系演进**：侧栏可设置你的昵称；随好感度阶段（陌生/熟悉/亲近/亲密），角色对你的称呼自然变亲昵（全名 → 名字 → 去姓/叠字 → 专属昵称）。
- **关系里程碑剧情**：好感度跨阶段（45/70）时，角色会写下一段「关系加深」的心声，作为主动消息送进收件箱，前端弹出卡片式提示。
- **冷战 / 和好**：被明显冒犯或敷衍会累积「关系张力」→ 进入冷战（语气变冷、暂停主动消息）；真诚关心/道歉累积 → 和好，触发一次「和好」剧情。
- **跨角色互动**：角色会知道你和其他角色的近况（对方此刻在做什么、今天发生了什么），自然关心或转述；普通主动消息有概率转述另一位角色的事。
- **桌面通知**：开启后，角色发来主动消息会弹系统通知（浏览器 `Notification`）。

---

## 🏗 架构

```
用户浏览器 (static/chat.html)
      │  SSE 流式 / JSON
      ▼
 FastAPI (server.py)
      │
      ▼
 Orchestrator (总控)
   ├── WorldClock       世界时钟 / 节日日历 / 主动消息
   └── Agent × N        每个 NPC 完全隔离
         ├── 人格内核        人格.md + 人格锚点.md（蒸馏 / 成长 / 重写）
         ├── 状态            state.json（好感度 + 心情）
         ├── 短期记忆         chat.json（会话上下文，滚动压缩）
         ├── 语义记忆         facts.json（关于用户的稳定事实）
         ├── 情景记忆         chroma_db（长期记忆 RAG：向量 + BM25 + rerank + 四维加权）
         └── 认知步骤         读心 + 内心独白 + 心情变化（便宜模型）
```

**单轮对话的数据流：**

```
用户消息
  → 联想检索（长期记忆 recall + 事实库 retrieve，查询 = 当前消息 + 最近上下文）
  → 认知步骤（读心 + 内心独白 + 心情变化，便宜模型）
  → 组装系统提示词（人格 + 行为准则 + 关系 + 心情 + 用户判断 + 内心活动 + 记忆 + 格式约束）
  → 主模型流式生成回复
  → 落历史
  → 后台异步：好感度结算 / 记忆压缩（画像 + 摘要 + 事实 + 人格成长 + 写长期记忆 + 遗忘）
```

---

## 📁 目录结构

```
.
├── agent/                        # 后端代码（Python 包）
│   ├── server.py                 # FastAPI 入口 + 全部 HTTP/SSE 接口
│   ├── orchestrator.py           # 总控：加载 NPC + 世界时钟 + 主动消息
│   ├── agent.py                  # 单个 NPC：人格/心情/好感度/认知/对话
│   ├── longterm_memory.py        # 长期记忆（Chroma + BM25 + rerank + 四维加权 + 软遗忘）
│   ├── fact_memory.py            # 结构化事实库（语义记忆）
│   ├── narrator.py               # 世界时钟（现实时间 + 节日日历）
│   ├── world_sim.py              # 世界模拟（作息 + 随机事件状态机）
│   ├── user_profile.py           # 单用户档案（昵称 + 成就系统）
│   ├── config.py                 # 全部配置（含 .env 加载）
│   ├── agents.json               # NPC 初始人设配置（新增角色在这里改）
│   ├── calendar.json             # 特殊日期（节日）日历
│   ├── world.json                # 世界状态（运行时生成，勿手动改）
│   ├── user.json                 # 用户档案数据（昵称/成就，运行时生成，已 gitignore）
│   ├── agents/                   # 每个 NPC 的运行时数据（自动生成，已 gitignore）
│   │   └── npc_01/
│   │       ├── 人格.md / 人格锚点.md
│   │       ├── chat.json / memory.json / state.json / inbox.json / facts.json / life.json
│   │       └── chroma_db/
│   └── static/                   # 前端（chat.html + 角色立绘/头像）
├── tests/                        # 单元测试
├── run.py                        # 跨平台启动入口(推荐)
├── run.sh                        # Linux / macOS 启动脚本
├── pyproject.toml                # 开发工具配置(ruff / pytest)
├── .gitattributes                # 换行符 / 二进制规范
├── .github/workflows/ci.yml      # CI 流水线
├── .env.example                  # 环境变量模板
├── requirements.txt              # 运行时依赖(已锁定版本)
├── requirements-dev.txt          # 开发 / 测试依赖
├── 启动服务.bat                   # Windows 一键启动脚本
└── LICENSE
```

> `agent/agents/`、`agent/world.json` 与 `agent/user.json` 都是运行时生成的数据，已在 `.gitignore` 中排除；克隆仓库后首次运行会自动生成。

---

## 🚀 快速开始

### 0. 前置条件
- Python 3.9+（开发使用 3.12）
- 一个阿里云百炼（DashScope）API Key：前往 [百炼控制台](https://bailian.console.aliyun.com/) 获取

### 1. 安装依赖
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置密钥
```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```
编辑 `.env`，填入你的 API Key：
```ini
API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 启动
```bash
# 推荐: 在项目根目录直接运行(自动处理路径, 跨平台)
python run.py

# Linux / macOS 也可
./run.sh

# 或 Windows 双击 启动服务.bat
# 或手动:
cd agent
python -m uvicorn server:app --host 127.0.0.1 --port 8080
```
浏览器打开 **http://127.0.0.1:8080/**。

> `requirements.txt` 已锁定为验证可运行的版本；如需开发/测试请额外执行 `pip install -r requirements-dev.txt`。

### 4. 运行测试
```bash
pip install -r requirements-dev.txt
pytest -q
```
> 仓库含 GitHub Actions（`.github/workflows/ci.yml`），每次 push / PR 会自动执行「依赖安装 + 语法检查 + 单元测试」。

---

## ⚙️ 配置说明

### 环境变量（`.env`）
| 变量 | 默认值 | 说明 |
|---|---|---|
| `API_KEY` | — | DashScope API Key（必填） |
| `CHAT_MODEL` | `qwen-max` | 主对话模型 |
| `EM_MODEL` | `text-embedding-v1` | 向量化模型 |
| `RERANK_MODEL` | `gte-rerank-v2` | 重排模型 |
| `FAVOR_MODEL` | `qwen-turbo` | 好感度判断模型 |
| `FORGET_MODEL` | `qwen-turbo` | 遗忘判断模型 |
| `COGNITION_MODEL` | `qwen-turbo` | 读心 / 内心独白模型 |

> 系统环境变量优先于 `.env` 文件；也兼容 `DASHSCOPE_API_KEY`。

### 关键参数（`agent/config.py`）
所有可调参数都集中在 `config.py`，带中文注释，常用项：

| 参数 | 默认 | 说明 |
|---|---|---|
| `MEMORY_RECALL_K` | 3 | 每轮注入的相关记忆条数 |
| `MEMORY_RECALL_THRESHOLD` | 0.1 | 记忆相关性下限 |
| `MEMORY_KEEP_TURNS` | 4 | 压缩后保留的近期轮数 |
| `COGNITION_ENABLED` | `True` | 是否开启「读心 + 内心独白」（关闭可降低延迟） |
| `MEMORY_EMOTION_BONUS` | 3 | 情绪记忆的重要性加成 |
| `MEMORY_DEDUP_HIGH` | 0.93 | 记忆相似度≥此值判「重复」，直接强化旧记忆（不新增） |
| `MEMORY_DEDUP_LOW` | 0.60 | 记忆相似度≤此值判「新事」，直接新增 |
| `FACT_MAX_COUNT` | 25 | 事实库总量上限 |
| `PERSONA_REWRITE_THRESHOLD` | 6 | 成长记录达到几条后重写人格内核 |
| `MEMORY_UPDATE_THRESHOLD` | 20 | 浅睡兜底轮数（话题一直不变时强制整理） |
| `MEMORY_SHIFT_MIN_TURNS` | 4 | 主题漂移触发浅睡的最小间隔轮数 |
| `DREAM_ENABLED` | `True` | 是否开启深睡（隔夜整理） |
| `DREAM_SLEEP_START` / `DREAM_SLEEP_END` | 21 / 6 | 睡眠窗口（小时），此时间段上线才深睡 |
| `DREAM_IDLE_MINUTES` | 30 | 距上次消息超此时长视为「互动结束」才整理 |
| `NPC_SCHEDULES` | — | 各角色的作息表（按小时划分，决定「上课中/睡觉中/空闲」等状态） |
| `RANDOM_EVENT_PROBABILITY` | 0.25 | 每个角色每天触发随机事件的概率 |
| `LIFE_BACKFILL_MAX_DAYS` | 7 | 用户离开后最多回填/模拟的天数 |
| `LIFE_EVENTS_KEEP` | 30 | 每个角色保留的最近随机事件条数 |
| `TENSION_COLD` | 4 | 关系张力达到此值进入「冷战」 |
| `CROSS_NPC_RELAY_PROBABILITY` | 0.2 | 普通主动消息中转述另一位角色近况的概率 |

### 新增 / 修改角色
编辑 `agent/agents.json`，每个角色包含 `id`、`name`、`description`、`birthday`（`MM-DD` 格式，可留空）和 `persona`（完整人设）。人格内核会在首次对话时自动蒸馏生成，无需手工维护 `人格.md`。

### 特殊日期
编辑 `agent/calendar.json`，按 `{"month": 12, "day": 25, "name": "圣诞节", "description": "..."}` 的格式添加节日。

---

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/agents` | 角色列表（含好感度、心情、当前作息状态） |
| `GET` | `/agents/{id}/history` | 指定角色的对话历史 |
| `GET` | `/agents/{id}/diary` | 指定角色的「回忆日记」（隔夜整理的日历史） |
| `GET` | `/agents/{id}/schedule` | 指定角色的「今日行程」（带当前时段高亮） |
| `GET` | `/user` | 当前用户昵称 |
| `POST` | `/user` | 设置用户昵称，body：`{"nickname"}` |
| `POST` | `/chat` | 对话（SSE 流式），body：`{"agent_id","question"}` |
| `GET` | `/inbox` | 各角色未读主动消息数（惰性触发主动消息） |
| `GET` | `/agents/{id}/inbox` | 指定角色的主动消息收件箱 |
| `POST` | `/agents/{id}/inbox/read` | 标记已读 |
| `GET` | `/world` | 世界状态（日期/星期/节日/事件 + 各角色作息状态与近期事件） |
| `POST` | `/reset` | 初始化所有角色（清空记忆与好感度，保留人格内核），破坏性操作需带 `?confirm=yes` |

---

## ⚠️ 已知限制 & 未来方向

- **单用户设计**：好感度 / 心情 / 记忆都以「单一用户」为前提，未做多用户隔离；服务默认只监听 `127.0.0.1`，无鉴权，请勿直接暴露公网。
- **锁粒度**：为避免并发污染历史，单个 NPC 的对话持有锁到流式结束；多角色并发不受影响。
- **前端角色硬编码**：`static/chat.html` 中角色头像、卡面、主题色、语录按 `agent_id` 写死，新增角色需同步修改前端的 `CHARACTERS` / `THEMES` 映射。
- **事实库为轻量实现**：事实检索用关键词重叠打分（事实量小，足够）；如需更强语义匹配可换向量检索。
- **记忆固化**：稳定事实通过「整体合并重写」沉淀进事实库；尚未做「反复回忆 → 自动晋升」的显式固化。
- **未来可做**：情绪效价显式回流到记忆加权、跨角色互动、多模态（语音/图片）、多用户隔离与鉴权、更细的对话修复策略。

---

## 📄 License

[MIT](./LICENSE)
