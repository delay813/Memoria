"""
全局配置: 路径 / 模型 / 世界时间(现实时间) / 特殊日期日历 / 从Agent(NPC)配置加载
"""
import json
import logging
import os
import re


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

# ============================================================
# 记忆压缩参数
# ============================================================
MEMORY_UPDATE_THRESHOLD = 6   # 用户提问次数达到该值触发记忆更新+压缩
MEMORY_KEEP_TURNS = 4         # 压缩后保留最近对话轮数(过小会丢失近期语境)
MEMORY_INPUT_MAX_TURNS = 20   # 提取记忆时最多取的消息条数
MEMORY_INPUT_MAX_CHARS = 4000 # 单条消息截断长度

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


def load_agents_config():
    """加载 agents.json, 返回配置字典"""
    with open(AGENTS_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
