import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dtw_utils import (
    _standard_dtw, 
    _dtw_sakoe_chiba, 
    _dtw_path_sakoe_chiba,
    TimingAnalyzer,
    compute_weighted_features
)


def test_dtw_performance():
    """测试DTW算法性能对比"""
    print("=" * 60)
    print("DTW Performance Test")
    print("=" * 60)
    
    # 生成测试数据
    x = np.random.randn(60, 8).astype(np.float32)
    y = np.random.randn(60, 8).astype(np.float32)
    
    # 预热（编译JIT）
    print("Warming up JIT compiler...")
    _dtw_sakoe_chiba(x[:10], y[:10], window_size=5)
    
    # 测试标准DTW
    print("\nTesting Standard DTW...")
    start = time.time()
    for _ in range(10):
        _standard_dtw(x, y)
    standard_time = (time.time() - start) / 10
    
    # 测试Sakoe-Chiba约束DTW
    print("Testing Sakoe-Chiba DTW (window=5)...")
    start = time.time()
    for _ in range(10):
        _dtw_sakoe_chiba(x, y, window_size=5)
    sakoe_time = (time.time() - start) / 10
    
    # 测试Sakoe-Chiba约束DTW（更大窗口）
    print("Testing Sakoe-Chiba DTW (window=10)...")
    start = time.time()
    for _ in range(10):
        _dtw_sakoe_chiba(x, y, window_size=10)
    sakoe_time_10 = (time.time() - start) / 10
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Standard DTW:     {standard_time*1000:.2f} ms")
    print(f"  Sakoe-Chiba (w=5): {sakoe_time*1000:.2f} ms")
    print(f"  Sakoe-Chiba (w=10): {sakoe_time_10*1000:.2f} ms")
    print(f"  Speedup (w=5):     {standard_time/sakoe_time:.2f}x")
    print(f"  Speedup (w=10):    {standard_time/sakoe_time_10:.2f}x")
    print("=" * 60)


def test_dtw_path():
    """测试DTW路径计算"""
    print("\n" + "=" * 60)
    print("DTW Path Test")
    print("=" * 60)
    
    # 生成测试数据
    x = np.random.randn(30, 8).astype(np.float32)
    y = np.random.randn(30, 8).astype(np.float32)
    
    # 测试路径计算
    path = _dtw_path_sakoe_chiba(x, y, window_size=5)
    
    print(f"Path length: {len(path)}")
    print(f"First 5 points: {path[:5]}")
    print(f"Last 5 points: {path[-5:]}")
    print("=" * 60)


def test_timing_analyzer():
    """测试节奏分析器"""
    print("\n" + "=" * 60)
    print("Timing Analyzer Test")
    print("=" * 60)
    
    analyzer = TimingAnalyzer(history_len=60, smoothing_window=5, lag_threshold=3.0)
    
    # 生成基础序列（正弦波模拟舞蹈动作）
    t = np.linspace(0, 4*np.pi, 60)
    base_seq = []
    for i in range(8):
        base_seq.append(np.sin(t + i * 0.5))
    base_seq = np.array(base_seq).T.astype(np.float32)
    
    # 测试1: 正常同步
    print("\nTest 1: Normal synchronization")
    hint, conf, lag = analyzer.analyze_timing(base_seq.tolist(), base_seq.tolist())
    print(f"  Hint: {hint}, Confidence: {conf:.2f}, Lag: {lag:.2f}")
    
    # 测试2: 慢动作（延迟5帧）
    print("\nTest 2: Slow motion (5 frames behind)")
    slow_seq = np.vstack([base_seq[:5], base_seq[:-5]])
    analyzer.reset()
    hint, conf, lag = analyzer.analyze_timing(slow_seq.tolist(), base_seq.tolist())
    print(f"  Hint: {hint}, Confidence: {conf:.2f}, Lag: {lag:.2f}")
    
    # 测试3: 快动作（提前3帧）
    print("\nTest 3: Fast motion (3 frames ahead)")
    fast_seq = base_seq[3:]
    analyzer.reset()
    hint, conf, lag = analyzer.analyze_timing(fast_seq.tolist(), base_seq[:-3].tolist())
    print(f"  Hint: {hint}, Confidence: {conf:.2f}, Lag: {lag:.2f}")
    
    print("=" * 60)


def test_weighted_features():
    """测试加权特征提取"""
    print("\n" + "=" * 60)
    print("Weighted Features Test")
    print("=" * 60)
    
    # 模拟关节角度数据
    angles_list = [
        {
            "leftShoulder": 90, "rightShoulder": 85,
            "leftElbow": 120, "rightElbow": 115,
            "leftHip": 100, "rightHip": 95,
            "leftKnee": 130, "rightKnee": 125
        },
        {
            "leftShoulder": 95, "rightShoulder": 90,
            "leftElbow": 125, "rightElbow": 120,
            "leftHip": 105, "rightHip": 100,
            "leftKnee": 135, "rightKnee": 130
        }
    ]
    
    features = compute_weighted_features(angles_list)
    
    print(f"Number of frames: {len(features)}")
    print(f"Feature vector length: {len(features[0])}")
    print(f"Frame 1 features: {features[0]}")
    print(f"Frame 2 features: {features[1]}")
    print("=" * 60)


def test_integration():
    """集成测试：模拟实际使用场景"""
    print("\n" + "=" * 60)
    print("Integration Test")
    print("=" * 60)
    
    analyzer = TimingAnalyzer(history_len=60, smoothing_window=5, lag_threshold=3.0)
    
    # 模拟60帧的舞蹈动作
    t = np.linspace(0, 4*np.pi, 60)
    ref_seq = []
    user_seq = []
    
    for i in range(60):
        # 参考序列
        ref_angs = {
            "leftShoulder": 90 + 10*np.sin(t[i]),
            "rightShoulder": 85 + 10*np.sin(t[i] + 0.5),
            "leftElbow": 120 + 15*np.sin(t[i] + 1.0),
            "rightElbow": 115 + 15*np.sin(t[i] + 1.5),
            "leftHip": 100 + 8*np.sin(t[i] + 2.0),
            "rightHip": 95 + 8*np.sin(t[i] + 2.5),
            "leftKnee": 130 + 12*np.sin(t[i] + 3.0),
            "rightKnee": 125 + 12*np.sin(t[i] + 3.5)
        }
        ref_seq.append(ref_angs)
        
        # 用户序列（模拟轻微延迟）
        delay = 2  # 2帧延迟
        if i >= delay:
            user_angs = {
                "leftShoulder": 90 + 10*np.sin(t[i-delay]),
                "rightShoulder": 85 + 10*np.sin(t[i-delay] + 0.5),
                "leftElbow": 120 + 15*np.sin(t[i-delay] + 1.0),
                "rightElbow": 115 + 15*np.sin(t[i-delay] + 1.5),
                "leftHip": 100 + 8*np.sin(t[i-delay] + 2.0),
                "rightHip": 95 + 8*np.sin(t[i-delay] + 2.5),
                "leftKnee": 130 + 12*np.sin(t[i-delay] + 3.0),
                "rightKnee": 125 + 12*np.sin(t[i-delay] + 3.5)
            }
        else:
            user_angs = ref_angs
        user_seq.append(user_angs)
    
    # 提取特征
    user_features = compute_weighted_features(user_seq)
    ref_features = compute_weighted_features(ref_seq)
    
    # 分析节奏
    start = time.time()
    hint, conf, lag = analyzer.analyze_timing(user_features, ref_features)
    elapsed = time.time() - start
    
    print(f"Analysis time: {elapsed*1000:.2f} ms")
    print(f"Hint: {hint}")
    print(f"Confidence: {conf:.2f}")
    print(f"Average lag: {lag:.2f} frames")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DTW Utils Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_dtw_performance()
        test_dtw_path()
        test_timing_analyzer()
        test_weighted_features()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
