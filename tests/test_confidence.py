
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.geometry import compute_angles, compute_angle_confidence, LSH, LEL, LIH
from src.core.scoring import score_angles, _select_points_with_weights

def test_angle_confidence():
    print("Testing Angle Confidence...")
    # Create dummy landmarks (all 33 points)
    # LSH=11, LEL=13, LIH=23
    lms = [{"x": 0.5, "y": 0.5, "v": 0.9} for _ in range(33)]
    
    # Set Left Shoulder (11) visibility to low
    lms[LSH]["v"] = 0.1
    
    conf = compute_angle_confidence(lms)
    print(f"Confidence for leftShoulder: {conf['leftShoulder']}")
    print(f"Confidence for rightShoulder: {conf['rightShoulder']}")
    
    assert conf["leftShoulder"] == 0.1, "Left Shoulder confidence should be min(0.1, 0.9, 0.9) = 0.1"
    assert conf["rightShoulder"] == 0.9, "Right Shoulder confidence should be 0.9"
    print("PASS: Angle Confidence Calculation")

def test_score_angles_weighting():
    print("\nTesting Score Angles Weighting...")
    u_angles = {"leftShoulder": 90, "rightShoulder": 90}
    r_angles = {"leftShoulder": 0, "rightShoulder": 90} # 90 deg diff for left, 0 for right
    
    # Without weights (should be bad score because of leftShoulder)
    score_raw, _ = score_angles(u_angles, r_angles)
    print(f"Raw Score (should be < 100): {score_raw}")
    
    # With weights: ignore leftShoulder (v < 0.5)
    weights = {"leftShoulder": 0.1, "rightShoulder": 0.9}
    score_weighted, _ = score_angles(u_angles, r_angles, angle_weights=weights)
    print(f"Weighted Score (should be 100): {score_weighted}")
    
    assert score_weighted > score_raw, "Weighted score should be higher as bad angle is ignored"
    assert abs(score_weighted - 100.0) < 1e-6, f"Weighted score should be 100.0, got {score_weighted}"
    print("PASS: Angle Scoring Weighting")

def test_procrustes_weights():
    print("\nTesting Procrustes Weights...")
    lms = [{"x": 0.5, "y": 0.5, "v": 0.9} for _ in range(33)]
    
    # Set Left Wrist (15) to low visibility
    lms[15]["v"] = 0.1
    
    pts, weights = _select_points_with_weights(lms)
    
    # Check weight for point 15
    # _select_points_with_weights logic:
    # 11,12,13,14,15,16...
    # index in pts array matches the order of add_idx calls.
    # 0: 11, 1: 12, 2: 13, 3: 14, 4: 15 (Left Wrist)
    
    w_15 = weights[4]
    print(f"Weight for point 15 (Left Wrist) with v=0.1: {w_15}")
    
    assert w_15 == 0.0, "Weight for low visibility point should be 0.0"
    
    # Check normal point
    w_16 = weights[5] # Right Wrist (16)
    print(f"Weight for point 16 (Right Wrist) with v=0.9: {w_16}")
    assert w_16 > 0.0, "Weight for high visibility point should be > 0.0"
    
    print("PASS: Procrustes Weighting")

if __name__ == "__main__":
    try:
        test_angle_confidence()
        test_score_angles_weighting()
        test_procrustes_weights()
        print("\nALL TESTS PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
