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
            "leftShoulder": "Left Shoulder",
            "rightShoulder": "Right Shoulder",
            "leftElbow": "Left Elbow",
            "rightElbow": "Right Elbow",
            "leftHip": "Left Hip",
            "rightHip": "Right Hip",
            "leftKnee": "Left Knee",
            "rightKnee": "Right Knee"
        }

        problem_angles = []
        for key, diff in diffs.items():
            if diff > 15:
                name = angle_names.get(key, key)
                direction = "insufficient bend" if diff > 30 else "angle deviation"
                problem_angles.append(f"{name} {direction} {diff:.0f}°")

        if score >= 85:
            tone = "encouraging"
            feedback = "Great form!"
        elif score >= 70:
            tone = "suggestive"
            feedback = "Good overall, minor improvements needed"
        else:
            tone = "instructive"
            feedback = "Needs more practice"

        prompt = f"""You are a professional dance coach. Please provide professional feedback on the user's body movements and rhythm (no more than 50 words).

Analysis data:
- Overall Score: {score:.0f} points
- Body parts needing improvement: {", ".join(problem_angles) if problem_angles else "All parts in good form"}
- Rhythm hint: {timing_hint if timing_hint else "Rhythm normal"}

Please focus on:
1. Whether body angles are accurate
2. Whether movement amplitude is sufficient
3. Whether rhythm is maintained

Provide concise and professional suggestions in English."""

        return self._call_api(prompt)

    def analyze_bad_frame(self, time_str: str, score: float, diffs: Dict[str, float],
                          ref_action: str = "Standard Action") -> Optional[str]:
        if not self.config.is_configured() or not self.client:
            return None

        angle_names = {
            "leftShoulder": "Left Shoulder",
            "rightShoulder": "Right Shoulder",
            "leftElbow": "Left Elbow",
            "rightElbow": "Right Elbow",
            "leftHip": "Left Hip",
            "rightHip": "Right Hip",
            "leftKnee": "Left Knee",
            "rightKnee": "Right Knee"
        }

        details = []
        for key, diff in sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:4]:
            name = angle_names.get(key, key)
            if diff > 10:
                if diff > 25:
                    details.append(f"{name} needs adjustment {diff:.0f}°")
                else:
                    details.append(f"{name} deviation {diff:.0f}°")

        prompt = f"""At {time_str}, the user scored only {score:.0f} points while performing "{ref_action}".

Problem details:
{chr(10).join(details) if details else "Overall posture needs improvement"}

Please provide 2 specific improvement suggestions to help the user do better. Keep the response concise and targeted in English."""

        return self._call_api(prompt)

    def generate_session_summary(self, session_data: Dict) -> Optional[str]:
        if not self.config.is_configured() or not self.client:
            return None

        total_frames = session_data.get("total_frames", 0)
        avg_score = session_data.get("avg_score", 0)
        bad_frames_count = session_data.get("bad_frames_count", 0)
        duration = session_data.get("duration", 0)

        prompt = f"""As a professional dance coach, please generate a detailed post-session summary report for the user.

Practice data:
- Practice duration: {duration:.0f} seconds
- Total movements: {total_frames}
- Average score: {avg_score:.1f} points
- Movements needing improvement: {bad_frames_count}

Please structure your response as follows:
1. Encouraging comments (1-2 sentences)
2. Body movement improvement suggestions (focus on angles, amplitude)
3. Rhythm and coordination suggestions
4. Next practice focus areas

Maintain a professional yet friendly tone. Keep the response within 120 words in English."""

        return self._call_api(prompt)

    def _call_api(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        if not self.client:
            return None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a professional, warm, and encouraging dance coach. Always respond in English."},
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
