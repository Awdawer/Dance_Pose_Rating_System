import os
import json
import time
from typing import Optional, List, Dict
from datetime import datetime


class AICoachConfig:
    def __init__(self):
        self.api_key = "sk-f1f60c7c96e64d9db08831fd516c150b"
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-v4-pro"
        self.max_tokens = 1000
        self.temperature = 0.3
        self.enabled = True

    def is_configured(self) -> bool:
        return bool(self.api_key and self.enabled)

    def to_dict(self):
        return {
            "api_key": self.api_key[:8] + "***" if self.api_key else "",
            "base_url": self.base_url,
            "model": self.model,
            "enabled": self.enabled
        }


class AICoach:
    def __init__(self, config: Optional[AICoachConfig] = None):
        self.config = config or AICoachConfig()
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.config.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                print("[AI Coach] openai package not installed. Run: pip install openai")
                self.client = None

    def update_config(self, config: AICoachConfig):
        self.config = config
        self._init_client()

    def analyze_realtime_feedback(self, diffs: Dict[str, float], score: float, timing_hint: str = "") -> Optional[str]:
        if not self.config.is_configured() or not self.client:
            return None

        angle_names = {
            "leftShoulder": "左肩",
            "rightShoulder": "右肩",
            "leftElbow": "左肘",
            "rightElbow": "右肘",
            "leftHip": "左髋",
            "rightHip": "右髋",
            "leftKnee": "左膝",
            "rightKnee": "右膝"
        }

        problem_angles = []
        for key, diff in diffs.items():
            if diff > 15:
                name = angle_names.get(key, key)
                direction = "弯曲不够" if diff > 30 else "角度偏差"
                problem_angles.append(f"{name}{direction} {diff:.0f}度")

        if score >= 85:
            tone = "鼓励"
            feedback = "动作很标准！"
        elif score >= 70:
            tone = "建议"
            feedback = "整体不错，还有小改进空间"
        else:
            tone = "指导"
            feedback = "需要多练习"

        prompt = f"""你是一位专业舞蹈教练，请针对用户的肢体动作和节奏给出专业反馈（不超过50字）。

分析数据：
- 综合评分: {score:.0f}分
- 需要改进的肢体部位: {", ".join(problem_angles) if problem_angles else "各部位动作标准"}
- 节奏提示: {timing_hint if timing_hint else "节奏正常"}

请重点关注：
1. 肢体角度是否准确
2. 动作幅度是否到位
3. 节奏是否跟得上

用简洁专业的语言给出建议。"""

        return self._call_api(prompt)

    def analyze_bad_frame(self, time_str: str, score: float, diffs: Dict[str, float],
                          ref_action: str = "标准动作") -> Optional[str]:
        if not self.config.is_configured() or not self.client:
            return None

        angle_names = {
            "leftShoulder": "左肩",
            "rightShoulder": "右肩",
            "leftElbow": "左肘",
            "rightElbow": "右肘",
            "leftHip": "左髋",
            "rightHip": "右髋",
            "leftKnee": "左膝",
            "rightKnee": "右膝"
        }

        details = []
        for key, diff in sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:4]:
            name = angle_names.get(key, key)
            if diff > 10:
                if diff > 25:
                    details.append(f"{name}需要调整 {diff:.0f}度")
                else:
                    details.append(f"{name}偏差 {diff:.0f}度")

        prompt = f"""在{time_str}，用户完成"{ref_action}"时，综合评分只有{score:.0f}分。

问题详情：
{chr(10).join(details) if details else "整体姿态需要改进"}

请给出2个具体改进建议，帮助用户做得更好。回复要简洁、有针对性。"""

        return self._call_api(prompt)

    def generate_session_summary(self, session_data: Dict) -> Optional[str]:
        if not self.config.is_configured() or not self.client:
            return None

        total_frames = session_data.get("total_frames", 0)
        avg_score = session_data.get("avg_score", 0)
        bad_frames_count = session_data.get("bad_frames_count", 0)
        duration = session_data.get("duration", 0)

        prompt = f"""作为专业舞蹈教练，请为用户生成详细的课后总结报告。

练习数据：
- 练习时长: {duration:.0f}秒
- 总动作数: {total_frames}个
- 平均得分: {avg_score:.1f}分
- 需要改进的动作: {bad_frames_count}处

请按照以下结构回复：
1. 鼓励评语（1-2句）
2. 肢体动作改进建议（针对角度、幅度）
3. 节奏协调性建议
4. 下一步练习重点

保持专业但亲切的语气，回复控制在120字以内。"""

        return self._call_api(prompt)

    def _call_api(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        if not self.client:
            return None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业、温暖、鼓励人心的舞蹈教练。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[AI Coach] API call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)

        return None


class CoachingHistory:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.history: List[Dict] = []

    def add_feedback(self, timestamp: str, feedback: str, score: float):
        self.history.append({
            "timestamp": timestamp,
            "feedback": feedback,
            "score": score,
            "type": "realtime"
        })
        self._trim()

    def add_bad_frame_advice(self, time_str: str, advice: str, score: float):
        self.history.append({
            "time": time_str,
            "advice": advice,
            "score": score,
            "type": "bad_frame"
        })
        self._trim()

    def set_summary(self, summary: str):
        self.history.append({
            "summary": summary,
            "type": "summary"
        })

    def _trim(self):
        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size:]

    def get_recent_feedbacks(self, count: int = 5) -> List[Dict]:
        feedbacks = [h for h in self.history if h.get("type") == "realtime" and h.get("feedback")]
        return feedbacks[-count:]

    def get_bad_frame_advices(self) -> List[Dict]:
        return [h for h in self.history if h.get("type") == "bad_frame" and h.get("advice")]

    def get_summary(self) -> Optional[str]:
        for h in reversed(self.history):
            if h.get("type") == "summary":
                return h.get("summary")
        return None

    def clear(self):
        self.history.clear()
