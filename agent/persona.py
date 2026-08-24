"""
人格内核(PersonaMixin): 角色的核心人格内核(人格.md) + 冻结锚点(人格锚点.md)。

- 首次对话把完整人设蒸馏成核心内核, 之后随成长"重写"演进
- 锚点永远冻结, 防止人格漂移
"""
import os
import re
import threading

import config
import llm
import text_utils


class PersonaMixin:
    """人格内核: 蒸馏 / 加载 / 成长 / 重写。混入 Agent。"""

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
        """冻结的核心锚点; 不存在时退回完整初始设定的纯文本。"""
        try:
            with open(self.anchor_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return text_utils.plain_text(self.persona)

    def _save_anchor(self, content):
        try:
            with open(self.anchor_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"[{self.name}] 锚点保存失败: {e}")

    def _distill_persona(self):
        """把完整初始人设蒸馏成核心人格内核, 失败返回空串。"""
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
        content = llm.call_text([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if content:
            return content
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
        """当前核心人格: 优先读人格.md; 不存在则后台蒸馏, 本次先用原始人设兜底。"""
        content = self._load_persona_md()
        if content:
            return content
        if not self._distilling:
            self._distilling = True
            threading.Thread(target=self._distill_and_save, daemon=True).start()
        return text_utils.plain_text(self.persona)

    def _append_growth(self, growth):
        """把新的认知/成长追加到 人格.md 的"当前认知与成长"小节(只保留最近N条)。"""
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
        """成长记录累积到阈值, 或人格.md过大时, 触发内核重写(压缩)。"""
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
        """把旧内核 + 成长记录整合重写成一份更成熟的新内核(人格真正演变)。"""
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
        result = llm.call_text([{"role": "user", "content": prompt}], model=config.MEMORY_EXTRACT_MODEL)
        if result:
            return result
        print(f"[{self.name}] 人格重写失败")
        return ""
