"""
全局配置: 路径 / 模型 / 世界时间(现实时间) / 特殊日期日历 / 从Agent(NPC)配置加载
"""
import json
import logging
import os
import re
import sys


def _fix_console_encoding():
    """修复 Windows 控制台打印中文/emoji 时的编码错误(UnicodeEncodeError)。

    控制台默认代码页常为 GBK(cp936), 而模型输出可能含 emoji、生僻字等 GBK 无法编码的字符,
    print 时就会抛 "UnicodeEncodeError: 'gbk' codec can't encode ..."。
    这里把 stdout/stderr 重配为 UTF-8 输出, 并对仍无法编码的字符降级为 '?'(而非抛错),
    从根源上消除命令行"文字解码错误"。任何入口(run.py / 启动服务.bat / uvicorn)都会先导入本模块。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_fix_console_encoding()


def _setup_logging():
    """统一日志: 控制台输出, INFO 级别, 带时间戳。各模块用 logging.getLogger(__name__) 记录。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


_setup_logging()

# 项目根目录(本文件所在目录), 不依赖启动时的工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 预定义从Agent(NPC)的配置文件
AGENTS_CONFIG_FILE = os.path.join(BASE_DIR, "agents.json")

# 特殊日期日历(节日等), 用于触发角色特殊对话
CALENDAR_FILE = os.path.join(BASE_DIR, "calendar.json")

# 每个NPC的隔离存储目录(chat历史 / 摘要画像 / 长期记忆RAG / 事实库)
AGENTS_DIR = os.path.join(BASE_DIR, "agents")

# 旁白(世界状态)持久化文件
WORLD_STATE_FILE = os.path.join(BASE_DIR, "world.json")

# 单用户档案(成就)持久化文件
USER_FILE = os.path.join(BASE_DIR, "user.json")


# ============================================================
# 环境变量加载: 支持从项目根目录的 .env 读取配置(不覆盖已存在的环境变量)
# 说明: 系统环境变量 / 启动时注入的变量 > .env 文件
# ============================================================
def _load_dotenv(path):
    """极简 .env 解析: 逐行读取 KEY=VALUE 注入 os.environ, 已存在的环境变量优先。"""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                # 去掉行内注释(仅当 # 前有空白时视为注释, 避免误伤值中的 #)
                line = re.split(r"\s+#", line)[0].strip()
                # 兼容 "export KEY=VALUE" 写法
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, _, value = line.partition("=")
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                # 去掉成对的单/双引号
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if key not in os.environ:  # 不覆盖系统/命令行已设置的环境变量
                    os.environ[key] = value
    except Exception as e:  # noqa: BLE001
        print(f"[config] .env 加载失败: {e}")


# 优先读项目根目录 .env(与本文件同级的上一级目录), 兼容本目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)
_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
_load_dotenv(os.path.join(BASE_DIR, ".env"))


# ============================================================
# 模型 / 密钥: 全部直接从环境变量读取(优先 DASHSCOPE_API_KEY, 兼容 API_KEY)
# ============================================================
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-max")
EMBEDDING_MODEL = os.getenv("EM_MODEL", "text-embedding-v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
FAVOR_MODEL = os.getenv("FAVOR_MODEL", "qwen-turbo")  # 好感度判断用便宜模型

# 关键凭证缺失时尽早给出清晰提示(否则会在初始化向量库时抛出难懂的 pydantic 校验错误)
if not API_KEY:
    raise RuntimeError(
        "未检测到 DashScope API Key。\n"
        "请在项目根目录的 .env 文件中填写: API_KEY=你的百炼密钥\n"
        "或设置环境变量 DASHSCOPE_API_KEY / API_KEY 后重新启动。"
    )

# ============================================================
# 模型调用: 超时重试
# ============================================================
MODEL_MAX_RETRIES = 2    # 模型调用失败时最多重试次数
MODEL_RETRY_DELAY = 1.0  # 每次重试间隔(秒)

# ============================================================
# 世界时间: 采用现实时间(现实一天 = 世界一天), 无游戏内加速
# ============================================================
WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# ============================================================
# 长期记忆: small-to-big + hybrid search + rerank 检索参数
# ============================================================
MEMORY_RECALL_K = 3            # 最终返回的父块(整条消息)数
MEMORY_RECALL_THRESHOLD = 0.1  # 相关性阈值: rerank分数低于此值的记忆不注入上下文
MEMORY_SMALL_CHUNK_SIZE = 150  # 小块切分粒度(字符), 超过则按句子切分
MEMORY_HYBRID_TOPN = 12        # 混合检索中, 向量/BM25 每路返回的小块数
MEMORY_RERANK_TOPN = 9         # 送入重排模型的父块候选数
MEMORY_RERANK_ENABLED = True   # 是否启用 rerank 精排(关闭可省一次 API 调用, 由三维加权继续排序)
MEMORY_ACCESS_SAVE_INTERVAL = 60  # 召回访问计数落盘节流(秒): 避免每轮对话都写一次注册表文件
MEMORY_FACT_CONFLICT_THRESHOLD = 0.72  # 事实更新时, 语义相似度≥此值的旧情景记忆判"冲突"并软遗忘

# ============================================================
# 记忆加权召回(四维: 时近性/重要性/访问频率/语义相似度)
# ============================================================
MEMORY_RECENCY_HALF_LIFE_DAYS = 7.0   # 时近性半衰期(天), 指数衰减模拟遗忘
MEMORY_FREQ_HALF_LIFE_DAYS = 14.0     # 访问频率半衰期(天), 长期不访问权重下降
MEMORY_FREQ_NORMALIZE = 5             # 访问次数归一化(达到该次数频率维度满分)
MEMORY_W_RECENCY = 0.2
MEMORY_W_IMPORTANCE = 0.3
MEMORY_W_FREQUENCY = 0.2
MEMORY_W_SIMILARITY = 0.3

# 情感显著性: 写入记忆时, 命中情绪/重要事件关键词的重要性加成(人脑对"有情绪的事"记得更牢)
MEMORY_EMOTION_BONUS = 3

# 话题联想扩展: 与已召回父块共享话题的其他父块加入候选(多跳式联想, 不增加最终注入条数)
MEMORY_TOPIC_EXPAND = 3
MEMORY_TOPIC_SIM_FLOOR = 0.4

# ============================================================
# 记忆去重(写入时): embedding 相似度分流, 仅"疑似相关"才调用小模型判断
# 原则: 只消除"真重复"; 对"相似但有新信息"(时间/地点/状态变化)一律保历史合并, 绝不覆盖旧事实
# ============================================================
MEMORY_DEDUP_ENABLED = True    # 是否开启写入去重(关闭则退化为纯追加)
MEMORY_DEDUP_HIGH = 0.93       # 相似度>=此值: 几乎一字不差的复述, 直接判"重复"(0 token)
MEMORY_DEDUP_LOW = 0.60        # 相似度<=此值: 明显新事, 直接新增(0 token)
MEMORY_DEDUP_TOPK = 3          # 每条新记忆去重时检索的最相似候选数

# ============================================================
# 记忆压缩参数
# ============================================================
MEMORY_UPDATE_THRESHOLD = 20  # 浅睡兜底轮数: 距上次压缩达到此轮数强制整理(即使话题未切换)
MEMORY_SHIFT_MIN_TURNS = 4    # 主题漂移触发浅睡的最小间隔轮数(上一段话题至少聊这么多轮才结算)
MEMORY_KEEP_TURNS = 4         # 压缩后保留最近对话轮数(过小会丢失近期语境)
MEMORY_INPUT_MAX_TURNS = 12   # 提取记忆时最多取的消息条数(降低单次压缩输入token)
MEMORY_INPUT_MAX_CHARS = 4000 # 单条消息截断长度
MEMORY_EXTRACT_MODEL = "qwen-turbo"  # 后台记忆整理(提取/人格重写)统一用便宜模型, 降低压缩成本

# ============================================================
# 结构化事实库(语义记忆): 关于用户的稳定事实/偏好/约定
# ============================================================
FACT_FILE_NAME = "facts.json"
FACT_RECALL_K = 6      # 每轮注入上下文的相关事实条数上限
FACT_MAX_COUNT = 25    # 事实库总量上限(抽取时整体合并去重, 自动淘汰过时信息)
FACT_MAX_CHARS = 40    # 单条事实最大长度

# ============================================================
# 人格内核 & 好感度
# ============================================================
PERSONA_FILE_NAME = "人格.md"             # 每个角色的核心内核文件(随成长演进)
PERSONA_ANCHOR_FILE_NAME = "人格锚点.md"  # 冻结的初始核心锚点(防漂移, 永不改写)
STATE_FILE_NAME = "state.json"            # 好感度/心情等动态状态
FAVOR_INITIAL = 20                        # 初始好感度
FAVOR_MIN = 0
FAVOR_MAX = 100
FAVOR_DECAY_PER_DAY = 1                   # 每满一天未互动, 好感度轻微下降
FAVOR_DECAY_FLOOR = 5                     # 衰减下限(不会降到0)
FAVOR_STAGE_COLD = 20                     # <20 陌生
FAVOR_STAGE_WARM = 45                     # <45 熟悉
FAVOR_STAGE_CLOSE = 70                    # <70 亲近, >=70 亲密
PERSONA_GROWTH_KEEP = 8                   # 人格成长记录保留最近条数
PERSONA_REWRITE_THRESHOLD = 6             # 成长记录累积到该条数后触发内核重写(人格真正演变)
PERSONA_MAX_CHARS = 800                   # 人格.md超过该长度强制压缩重写(防止过大)

# ============================================================
# 心情系统(角色自身的情绪状态, 随对话波动、随时间回落)
# ============================================================
MOOD_MIN = -10              # 情绪值下限
MOOD_MAX = 10               # 情绪值上限
MOOD_DECAY_PER_DAY = 1.0    # 每满一天未互动, 情绪值向平静(0)回落
MOOD_SHIFT_CLAMP = 3        # 单轮对话对情绪值的最大改变量

# ============================================================
# 认知步骤(读心 + 内心独白 + 心情变化): 每轮回复前先用便宜模型"想"一遍
# ============================================================
COGNITION_ENABLED = True                          # 关闭可减少一次模型调用、降低延迟
COGNITION_MODEL = os.getenv("COGNITION_MODEL", "qwen-turbo")
COGNITION_HISTORY_TURNS = 6                       # 认知时回看的最近对话条数
COGNITION_THOUGHT_MAX_CHARS = 60                  # 内心独白最大长度

# ============================================================
# 话题开场卡: 生成与缓存(输入栏上方的可点击开场话题)
# - 一次性多生成几个, 前端本地分批轮换, "换一批"不再每次都打模型
# - 生成结果带TTL缓存, 短时间内反复切换角色/刷新直接命中, 不再阻塞
# ============================================================
TOPIC_SUGGEST_COUNT = 9    # 每次生成的开场话题总数(前端每次展示3个, 轮换3批)
TOPIC_CACHE_TTL = 300      # 开场卡缓存有效期(秒): 过期后才重新调用模型
TOPIC_PREFETCH = True      # 服务启动时后台预生成各角色开场卡, 让首次加载也秒开

# ============================================================
# 遗忘机制(小模型驱动, 避免向量库只增不减)
# 注意: 采用"软遗忘"——先标记为想不起来(不注入召回), 重要度极低时才物理删除
# ============================================================
FORGET_MODEL = os.getenv("FORGET_MODEL", "qwen-turbo")  # 遗忘判断用小模型
FORGET_MIN_AGE_DAYS = 3         # 至少N天前的记忆才考虑遗忘
FORGET_RECENCY_THRESHOLD = 0.05 # 时近性低于此值进入遗忘候选
FORGET_FREQ_THRESHOLD = 1       # 访问次数低于此值进入候选
FORGET_BATCH_SIZE = 12          # 每次评估的候选条数上限

# ============================================================
# 主动消息(角色主动发起的离线事件, 惰性生成: 用户上线时才判断)
# ============================================================
INBOX_FILE_NAME = "inbox.json"  # 每个角色的主动消息收件箱
PROACTIVE_MIN_AWAY_DAYS = 1     # 至少离开N天才可能触发"想念"
PROACTIVE_INTENT_THRESHOLD = 40 # 主动意愿阈值(好感度-时间衰减 >= 此值才主动)
PROACTIVE_DECAY_PER_DAY = 5     # 每离开一天, 主动意愿降低的分数
PROACTIVE_COOLDOWN_DAYS = 2     # 两次普通主动消息的最小间隔(天)
CROSS_NPC_RELAY_PROBABILITY = 0.2  # 普通主动消息中"转述另一位角色近况"的概率

# ============================================================
# 深睡(隔夜整理): 晚上特定时间做"单天会话总结 + 深度去重 + 刷新短期聊天"
# 惰性触发: 无定时器, 只在用户上线互动后检查; 无新互动则免做(角色"睡着")
# ============================================================
DREAM_ENABLED = True            # 是否开启隔夜整理(深睡)
DREAM_MODEL = os.getenv("DREAM_MODEL", "qwen-turbo")  # 深睡总结用便宜模型
DREAM_SLEEP_START = 21          # 睡眠窗口开始(小时)
DREAM_SLEEP_END = 6             # 睡眠窗口结束(小时), 即 21:00 ~ 次日 06:00
DREAM_IDLE_MINUTES = 30         # 距上次消息超过此分钟数视为"互动已结束", 才可整理(延后未结束的互动)
DREAM_MAX_DELAY_DAYS = 2        # 超过此天数未整理则忽略睡眠窗口强制补做(防无限延后)
DREAM_DAILY_LOG_NAME = "daily_log.json"  # 每个角色的"日历史"文件


# ============================================================
# 世界模拟: NPC 作息 + 低概率随机事件(不在场时的生命感)
# - 每个NPC按小时有作息表, 决定其当前"状态"(上课/社团/值班/睡觉/空闲)
# - 每天惰性地为每个角色掷一次随机事件(低概率), 影响心情并注入对话
# - 用户离开期间世界照常运转: 回填错过的日子(最多 LIFE_BACKFILL_MAX_DAYS 天)
# ============================================================
LIFE_FILE_NAME = "life.json"     # 每个NPC的生活状态(每日随机事件)持久化文件
RANDOM_EVENT_PROBABILITY = 0.25  # 每个角色每天触发随机事件的概率(0~1)
LIFE_BACKFILL_MAX_DAYS = 7       # 用户离开后最多回填/模拟的天数(太久只模拟最近这几天)
LIFE_EVENTS_KEEP = 30            # 每个角色保留的最近事件条数
SLEEP_PENDING_MAX = 5            # 深夜睡着时最多积压几条待补回的消息

# ============================================================
# NPC 数据(外置到 JSON, 新增/修改角色与事件不再改 Python 代码)
# - agents.json: 每个角色的 schedule/random_events/quotes + 前端视觉资料(avatar/card/en/tags/theme)
# - npc_common.json: 兜底作息 DEFAULT_SCHEDULE + 通用随机事件池 RANDOM_EVENTS
# ============================================================
def _read_json_data(path, default):
    """读取 JSON 数据文件; 失败降级为 default 并打印提示。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[config] 数据文件加载失败 {path}: {e}")
        return default


_COMMON = _read_json_data(os.path.join(BASE_DIR, "npc_common.json"), {})

# 兜底作息(新增角色未单独配置作息时使用)
DEFAULT_SCHEDULE = _COMMON.get("default_schedule") or [
    {"start": 0, "end": 8, "activity": "在休息", "label": "休息中", "busy": True, "sleepy": True},
    {"start": 8, "end": 12, "activity": "在忙自己的事", "label": "忙碌中", "busy": True},
    {"start": 12, "end": 14, "activity": "在午休", "label": "午休中", "busy": False},
    {"start": 14, "end": 18, "activity": "在忙自己的事", "label": "忙碌中", "busy": True},
    {"start": 18, "end": 23, "activity": "在休息", "label": "空闲", "busy": False},
    {"start": 23, "end": 24, "activity": "在休息", "label": "休息中", "busy": True, "sleepy": True},
]

# 通用随机事件池(所有角色)
RANDOM_EVENTS = _COMMON.get("random_events") or []

# 每个角色独有的数据(作息/专属事件/语录/前端视觉资料)
_AGENTS_DATA = _read_json_data(AGENTS_CONFIG_FILE, {}).get("agents", [])
NPC_SCHEDULES = {a["id"]: a["schedule"] for a in _AGENTS_DATA if a.get("schedule")}
NPC_RANDOM_EVENTS = {a["id"]: a["random_events"] for a in _AGENTS_DATA if a.get("random_events")}
NPC_QUOTES = {a["id"]: a["quotes"] for a in _AGENTS_DATA if a.get("quotes")}
NPC_VISUALS = {
    a["id"]: {
        "avatar": a.get("avatar", ""),
        "card": a.get("card", ""),
        "en": a.get("en", ""),
        "tags": a.get("tags", []),
        "theme": a.get("theme", {}),
    }
    for a in _AGENTS_DATA
}


# ============================================================
# 关系张力(冷战/和好): 明显冒犯/敷衍累积→冷战, 真诚关心/道歉累积→和好
# ============================================================
TENSION_MAX = 10               # 张力上限
TENSION_COLD = 4               # >= 此值进入"冷战"(语气变冷/暂停普通主动消息)
TENSION_OFFENSE = 2            # 单轮好感度 <= -4 时的张力增量
TENSION_MILD_OFFENSE = 1       # 单轮好感度在 [-3,-1] 区间时的张力增量
TENSION_SOOTHE = -1            # 单轮好感度 >= +3 时的张力减量(关心/哄)
TENSION_WARM_SOOTHE = -2       # 单轮好感度 >= +5 时的张力减量(真诚道歉/很暖心)


def load_agents_config():
    """返回 agents.json 的配置字典(启动时已加载缓存)。"""
    return {"agents": _AGENTS_DATA}
