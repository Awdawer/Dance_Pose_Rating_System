"""
测试DTW对齐评分功能
验证当用户节奏与参考不同时，DTW评分是否能正确对齐
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dtw_utils import DTWAlignmentScorer, compute_weighted_features
from src.utils.geometry import compute_angles


def create_mock_landmarks(angle_dict):
    """创建模拟的关键点数据"""
    # 创建33个关键点，只设置我们关心的关节
    lms = [{"x": 0.5, "y": 0.5, "v": 1.0} for _ in range(33)]
    
    # 设置特定关节的角度（通过位置模拟）
    # 这里简化处理，实际应该根据角度计算坐标
    return lms


def test_dtw_alignment():
    """测试DTW对齐功能"""
    print("=" * 60)
    print("DTW对齐评分测试")
    print("=" * 60)
    
    scorer = DTWAlignmentScorer(window_size=15, min_window=3)
    
    # 模拟10帧的舞蹈动作
    # 参考序列：正弦波
    ref_angles_list = []
    for i in range(10):
        angle = {
            "leftShoulder": 90 + 10 * np.sin(i * 0.5),
            "rightShoulder": 85 + 10 * np.sin(i * 0.5 + 0.3),
            "leftElbow": 120 + 15 * np.sin(i * 0.5 + 0.6),
            "rightElbow": 115 + 15 * np.sin(i * 0.5 + 0.9),
            "leftHip": 100 + 8 * np.sin(i * 0.5 + 1.2),
            "rightHip": 95 + 8 * np.sin(i * 0.5 + 1.5),
            "leftKnee": 130 + 12 * np.sin(i * 0.5 + 1.8),
            "rightKnee": 125 + 12 * np.sin(i * 0.5 + 2.1)
        }
        ref_angles_list.append(angle)
    
    # 用户序列：慢2帧（延迟）
    user_angles_list = []
    delay = 2
    for i in range(10):
        ref_i = max(0, i - delay)  # 用户慢了2帧
        user_angles_list.append(ref_angles_list[ref_i].copy())
    
    print("\n测试场景：用户比参考慢2帧")
    print("-" * 60)
    
    # 逐帧测试
    for frame_idx in range(10):
        user_lm = create_mock_landmarks(user_angles_list[frame_idx])
        ref_lm = create_mock_landmarks(ref_angles_list[frame_idx])
        
        user_angs = user_angles_list[frame_idx]
        ref_angs = ref_angles_list[frame_idx]
        
        # 更新历史
        scorer.update_histories(user_lm, ref_lm, user_angs, ref_angs)
        
        # 查找最佳匹配
        best_r_lm, best_r_angs, info = scorer.find_best_match()
        
        if info.get("aligned", False):
            lag = info.get("lag", 0)
            ref_idx = info.get("ref_idx", frame_idx)
            print(f"帧 {frame_idx}: 用户动作 -> 匹配到参考帧 {ref_idx} (lag={lag})")
            
            # 验证：如果lag=2，说明正确识别了2帧延迟
            if frame_idx >= 5 and lag == 2:
                print(f"  ✓ 正确识别2帧延迟！")
        else:
            print(f"帧 {frame_idx}: 历史不足，未对齐")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def test_same_video():
    """测试相同视频的情况"""
    print("\n" + "=" * 60)
    print("相同视频测试（应该lag=0）")
    print("=" * 60)
    
    scorer = DTWAlignmentScorer(window_size=15, min_window=3)
    
    # 完全相同的序列
    angles_list = []
    for i in range(10):
        angle = {
            "leftShoulder": 90 + 10 * np.sin(i * 0.5),
            "rightShoulder": 85 + 10 * np.sin(i * 0.5 + 0.3),
            "leftElbow": 120 + 15 * np.sin(i * 0.5 + 0.6),
            "rightElbow": 115 + 15 * np.sin(i * 0.5 + 0.9),
            "leftHip": 100 + 8 * np.sin(i * 0.5 + 1.2),
            "rightHip": 95 + 8 * np.sin(i * 0.5 + 1.5),
            "leftKnee": 130 + 12 * np.sin(i * 0.5 + 1.8),
            "rightKnee": 125 + 12 * np.sin(i * 0.5 + 2.1)
        }
        angles_list.append(angle)
    
    for frame_idx in range(10):
        lm = create_mock_landmarks(angles_list[frame_idx])
        angs = angles_list[frame_idx]
        
        scorer.update_histories(lm, lm, angs, angs)  # 相同的关键点
        best_r_lm, best_r_angs, info = scorer.find_best_match()
        
        if info.get("aligned", False):
            lag = info.get("lag", 0)
            print(f"帧 {frame_idx}: lag={lag} (相同视频应该lag=0)")
    
    print("=" * 60)


if __name__ == "__main__":
    test_dtw_alignment()
    test_same_video()
