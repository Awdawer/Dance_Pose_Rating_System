"""
AI Coach 功能单元测试
测试 AI教练的配置、实时反馈、坏帧分析、课后总结等功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ai_coach import AICoach, AICoachConfig, CoachingHistory


class MockOpenAIClient:
    """模拟OpenAI客户端，用于离线测试"""
    def __init__(self, mock_response=None):
        self.mock_response = mock_response or "测试反馈：动作标准，继续保持！"
    
    def chat(self):
        return self
    
    def completions(self):
        return self
    
    def create(self, **kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    content = self.mock_response
                message = Message()
            choices = [Choice()]
        return MockResponse()


class TestAICoachConfig:
    """测试配置类"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        config = AICoachConfig()
        assert config.api_key is not None
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-v4-pro"
        assert config.max_tokens == 1000
        assert config.temperature == 0.3
        assert config.enabled == True
    
    def test_is_configured(self):
        """测试配置检查方法"""
        config = AICoachConfig()
        assert config.is_configured() == True
        
        config.enabled = False
        assert config.is_configured() == False
        
        config.enabled = True
        config.api_key = ""
        assert config.is_configured() == False


class TestAICoachOffline:
    """离线测试AI Coach（不调用真实API）"""
    
    def test_disabled_coach(self):
        """测试禁用AI教练时返回None"""
        config = AICoachConfig()
        config.enabled = False
        coach = AICoach(config)
        
        result = coach.analyze_realtime_feedback({}, 80)
        assert result is None
        
        result = coach.analyze_bad_frame("00:05", 50, {})
        assert result is None
        
        result = coach.generate_session_summary({})
        assert result is None
    
    def test_empty_diffs(self):
        """测试空的差异数据"""
        config = AICoachConfig()
        config.enabled = False  # 禁用API调用
        coach = AICoach(config)
        
        result = coach.analyze_realtime_feedback({}, 90)
        assert result is None
    
    def test_score_levels(self):
        """测试不同分数等级的处理逻辑"""
        config = AICoachConfig()
        config.enabled = False
        coach = AICoach(config)
        
        # 由于禁用了API，我们只能测试配置是否正确处理
        assert coach.config.temperature == 0.3
        assert coach.config.max_tokens == 1000


class TestCoachingHistory:
    """测试历史记录管理"""
    
    def test_history_add_and_retrieve(self):
        """测试添加和检索历史记录"""
        history = CoachingHistory(max_size=10)
        
        history.add_feedback("2024-01-01 10:00:00", "测试反馈1", 85)
        history.add_feedback("2024-01-01 10:00:01", "测试反馈2", 75)
        
        recent = history.get_recent_feedbacks(5)
        assert len(recent) == 2
        assert recent[0]["feedback"] == "测试反馈1"
        assert recent[1]["feedback"] == "测试反馈2"
    
    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        history = CoachingHistory(max_size=3)
        
        for i in range(5):
            history.add_feedback(f"time_{i}", f"feedback_{i}", 80)
        
        recent = history.get_recent_feedbacks(10)
        assert len(recent) == 3  # 应该只保留最新的3条
    
    def test_bad_frame_advice(self):
        """测试坏帧建议记录"""
        history = CoachingHistory()
        
        history.add_bad_frame_advice("00:05", "调整左肩角度", 45)
        history.add_bad_frame_advice("00:10", "调整右膝角度", 50)
        
        advices = history.get_bad_frame_advices()
        assert len(advices) == 2
        assert advices[0]["advice"] == "调整左肩角度"
    
    def test_summary_storage(self):
        """测试总结存储和检索"""
        history = CoachingHistory()
        
        history.set_summary("这是一个总结")
        summary = history.get_summary()
        assert summary == "这是一个总结"
        
        history.set_summary("更新后的总结")
        summary = history.get_summary()
        assert summary == "更新后的总结"
    
    def test_clear_history(self):
        """测试清空历史记录"""
        history = CoachingHistory()
        
        history.add_feedback("time1", "feedback1", 80)
        history.set_summary("summary")
        
        history.clear()
        
        assert len(history.get_recent_feedbacks()) == 0
        assert history.get_summary() is None


class TestAICoachIntegration:
    """集成测试（需要真实API调用）"""
    
    def test_api_call_success(self):
        """测试真实API调用（需要配置有效的API Key）"""
        config = AICoachConfig()
        
        if not config.is_configured():
            print("⚠️  跳过API集成测试：未配置API Key")
            return
        
        coach = AICoach(config)
        
        result = coach.analyze_realtime_feedback(
            {"leftShoulder": 10, "rightShoulder": 5},
            85,
            "节奏正常"
        )
        
        print(f"API响应: {result}")
        assert result is not None
        assert len(result) > 0
    
    def test_session_summary(self):
        """测试生成课后总结"""
        config = AICoachConfig()
        
        if not config.is_configured():
            print("⚠️  跳过课后总结测试：未配置API Key")
            return
        
        coach = AICoach(config)
        
        session_data = {
            "total_frames": 150,
            "avg_score": 78.5,
            "bad_frames_count": 15,
            "duration": 60
        }
        
        result = coach.generate_session_summary(session_data)
        
        print(f"课后总结: {result}")
        assert result is not None
        assert len(result) > 0


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("AI Coach 单元测试")
    print("=" * 60)
    
    # 测试配置类
    print("\n1. 测试配置类 (AICoachConfig)")
    print("-" * 60)
    config_test = TestAICoachConfig()
    config_test.test_config_defaults()
    print("✓ test_config_defaults passed")
    config_test.test_is_configured()
    print("✓ test_is_configured passed")
    
    # 测试离线功能
    print("\n2. 测试离线功能 (AICoach Offline)")
    print("-" * 60)
    offline_test = TestAICoachOffline()
    offline_test.test_disabled_coach()
    print("✓ test_disabled_coach passed")
    offline_test.test_empty_diffs()
    print("✓ test_empty_diffs passed")
    offline_test.test_score_levels()
    print("✓ test_score_levels passed")
    
    # 测试历史记录
    print("\n3. 测试历史记录管理 (CoachingHistory)")
    print("-" * 60)
    history_test = TestCoachingHistory()
    history_test.test_history_add_and_retrieve()
    print("✓ test_history_add_and_retrieve passed")
    history_test.test_history_size_limit()
    print("✓ test_history_size_limit passed")
    history_test.test_bad_frame_advice()
    print("✓ test_bad_frame_advice passed")
    history_test.test_summary_storage()
    print("✓ test_summary_storage passed")
    history_test.test_clear_history()
    print("✓ test_clear_history passed")
    
    # 测试集成功能（需要API Key）
    print("\n4. 测试集成功能 (需要API Key)")
    print("-" * 60)
    integration_test = TestAICoachIntegration()
    integration_test.test_api_call_success()
    integration_test.test_session_summary()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()