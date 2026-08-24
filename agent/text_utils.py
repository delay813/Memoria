"""
共享文本处理工具。

- tokenize / tokenize_set: 中英文分词(英文/数字按单词, 中文按双字词)
- plain_text: 去掉人设里的 markdown 标记
- worth_remembering: 判断一条消息是否值得写入长期记忆(过滤寒暄/过短内容)

fact_memory.FactMemory 与 longterm_memory.SimpleBM25 曾各自维护一份相同的分词实现,
现统一至此。
"""
import re


def tokenize(text):
    """返回 token 列表(小写): 英文/数字按单词, 中文按相邻双字词(bigram)。"""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    tokens += [han[i:i + 2] for i in range(len(han) - 1)]
    return tokens


def tokenize_set(text):
    """tokenize 的集合版(用于关键词重叠判分)。"""
    return set(tokenize(text))


def plain_text(text):
    """去掉人设里的 markdown 标记, 只保留可读内容, 避免模型模仿 markdown 排版。"""
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


def worth_remembering(content):
    """判断一条消息是否值得写入长期记忆(过滤寒暄/语气词/过短内容)。"""
    text = str(content or "").strip()
    if not text:
        return False
    if len(text) < 4:
        return False
    if _FILLER_RE.match(text):
        return False
    return True
