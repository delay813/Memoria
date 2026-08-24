"""
深睡 · 隔夜整理(DreamMixin): 晚上特定时间把一段连续会话总结成"回忆日记", 并刷新短期聊天。

惰性触发: 无定时器, 只在用户上线互动后检查; 无新互动则免做(角色"睡着"零消耗)。
"""
import json
import os
import threading
import time
from datetime import date

import config
import json_utils
import llm
import storage
import time_utils
from narrator import WorldClock


class DreamMixin:
    """深睡(隔夜整理) / 回忆日记。混入 Agent。"""

    # ============================================================
    # 日历史(回忆日记)持久化
    # ============================================================
    def _load_daily_log(self):
        try:
            if os.path.exists(self.daily_log_file):
                with open(self.daily_log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[{self.name}] 日历史加载失败: {e}")
        return []

    def _save_daily_log(self, logs):
        try:
            storage.save_json(self.daily_log_file, logs)
        except Exception as e:
            print(f"[{self.name}] 日历史保存失败: {e}")

    def get_daily_log(self):
        """返回该角色的日历史(按日期升序), 供前端回看。"""
        return self._load_daily_log()

    @staticmethod
    def _msg_date(m):
        """消息的日期(YYYY-MM-DD); 无时间戳则返回空串。"""
        ts = m.get("ts")
        if not ts:
            return ""
        try:
            return date.fromtimestamp(float(ts)).isoformat()
        except Exception:
            return ""

    def _should_dream(self, old_msgs):
        """是否该做隔夜整理: 有旧会话 + 互动已结束 + (睡眠窗口内 或 超期兜底)。
        不按自然日切分——彻夜长谈(跨午夜)算同一段会话, 一次总结。"""
        if not old_msgs:
            return False
        now = time.time()
        # 互动已结束: 旧会话最后一条消息距今超过 IDLE 分钟(否则延后)
        last_ts = max(float(m["ts"]) for m in old_msgs)
        if (now - last_ts) < config.DREAM_IDLE_MINUTES * 60:
            return False
        # 睡眠窗口判断: 不在窗口内且未超期 → 延后
        if not WorldClock.is_sleep_window():
            days_overdue = (now - self._last_dream_ts) / 86400.0 if self._last_dream_ts else 999.0
            if days_overdue < config.DREAM_MAX_DELAY_DAYS:
                return False
        return True

    def _maybe_dream_async(self):
        """惰性触发深睡: 满足条件则后台执行(不阻塞对话)。
        在"当前消息 append 之前"调用, 因此快照里只有上一段会话, 不含正在进行的这轮。"""
        if not config.DREAM_ENABLED or self._dreaming:
            return
        old_msgs = [m for m in self.history if m.get("role") in ("user", "assistant") and m.get("ts")]
        if not old_msgs:
            return
        if not self._should_dream(old_msgs):
            return
        boundary = max(float(m["ts"]) for m in old_msgs)
        self._dreaming = True
        threading.Thread(target=self._run_dream, args=(list(old_msgs), boundary), daemon=True).start()

    def maybe_dream(self):
        """用户上线时由总控调用, 惰性触发隔夜整理。"""
        self._maybe_dream_async()

    def _summarize_session(self, msgs):
        """把一段连续会话(上次深睡以来, 可能跨午夜)总结成一条日历史记录; date 取会话开始的日期。"""
        if not msgs:
            return None
        day = self._msg_date(msgs[0]) or time_utils.today_iso()
        lines = "\n".join(f"{m['role']}: {self._clean_text(m.get('content', ''))}" for m in msgs)
        prompt = f"""你是「{self.name}」的梦境整理助手。下面是用户与你的一段连续会话(可能跨午夜, 属同一次长谈)。请总结成一条日历史, 严格输出JSON对象(不要任何其他文字):

{{"date":"{day}","summary":"这段互动的总结, 100字以内","new_understandings":["角色对用户产生的新认识"],"highlights":["这段会话中最重要的事件"]}}

要求:
1. date 固定用 "{day}"。
2. summary 概括"这段时间用户和角色发生了什么新互动"; new_understandings 记录"角色对用户的新认识"(可空数组); highlights 记录"最重要的事件"(可空数组)。
3. 只依据对话内容, 不要编造。

【会话内容】
{lines}"""
        resp = llm.call([{"role": "user", "content": prompt}], model=config.DREAM_MODEL)
        if not resp:
            return None
        record = json_utils.parse_object(resp.output.choices[0].message.content)
        return record if isinstance(record, dict) and record.get("date") else None

    def _append_daily_log(self, records):
        logs = self._load_daily_log()
        existing = {l.get("date") for l in logs if l.get("date")}
        for r in records:
            d = r.get("date")
            if not d or d in existing:
                continue
            logs.append({
                "date": d,
                "summary": str(r.get("summary", "")).strip(),
                "new_understandings": [str(x).strip() for x in (r.get("new_understandings") or []) if str(x).strip()],
                "highlights": [str(x).strip() for x in (r.get("highlights") or []) if str(x).strip()],
                "tags": [str(x).strip() for x in (r.get("tags") or []) if str(x).strip()],
                "favorability": r.get("favorability"),
                "mood": str(r.get("mood", "")).strip(),
            })
        logs.sort(key=lambda x: x.get("date", ""))
        self._save_daily_log(logs)

    def _run_dream(self, old_msgs, boundary):
        """深睡执行: 总结旧会话→写日历史→清空旧会话→软遗忘。"""
        try:
            record = self._summarize_session(old_msgs)
            if record:
                # 补充当日话题标签/好感度快照, 供"回忆相册"展示
                topics = []
                for m in old_msgs:
                    t = m.get("topic")
                    if t and str(t).strip() and str(t).strip() not in topics:
                        topics.append(str(t).strip())
                record["tags"] = topics[:6]
                record["favorability"] = self._favor
                record["mood"] = self._mood_label()
                with self._lock:
                    self._append_daily_log([record])
                    # 清空"旧会话"(ts <= boundary); 保留正在进行的这轮(ts > boundary)与无日期消息
                    self.history = [m for m in self.history
                                    if m.get("role") == "system" or not m.get("ts") or float(m["ts"]) > boundary]
                    self._save_history()
                    self._last_dream_ts = time.time()
                    self._save_state()
                print(f"[{self.name}] 隔夜整理完成: {record.get('date')}")
                self._unlock_cb("first_dream")
            # 深睡附加: 对长期记忆做一次软遗忘, 避免只增不减
            try:
                self.long_term.forget(self.name)
            except Exception as e:
                print(f"[{self.name}] 深睡软遗忘异常: {e}")
        except Exception as e:
            print(f"[{self.name}] 隔夜整理异常: {e}")
        finally:
            self._dreaming = False
