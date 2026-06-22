import os
import sys
import math
import numpy as np
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from src.core.scoring import score_diff, _dist_to_score
from src.utils.geometry import angle_between, compute_angles, LSH, LEL, LIH, LAW, LAN, RSH, REL, RIH, RAW, RAN, LEK, REK
 
 
def test_score_diff_tolerance_and_bounds():
    assert score_diff(0.0, tolerance=3.0) == 100.0
    assert score_diff(3.0, tolerance=3.0) == 100.0
    s = score_diff(20.0, tolerance=3.0)
    assert abs(s - (100.0 - 1.5 * 17.0)) < 1e-9
    assert score_diff(1e9) == 0.0
 
 
def test_dist_to_score_tolerance_and_monotonicity():
    assert _dist_to_score(0.0, tolerance=0.02) == 100.0
    assert _dist_to_score(0.02, tolerance=0.02) == 100.0
    s1 = _dist_to_score(0.03, k=3.0, gamma=0.5, tolerance=0.02)
    s2 = _dist_to_score(0.10, k=3.0, gamma=0.5, tolerance=0.02)
    assert 0.0 <= s2 < s1 < 100.0
 
 
def test_angle_between_degenerate_and_extremes():
    assert angle_between((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == 0.0
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert abs(angle_between(a, b, c) - 0.0) < 1e-6
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (-1.0, 0.0)
    assert abs(angle_between(a, b, c) - 180.0) < 1e-6
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (0.0, 1.0)
    assert abs(angle_between(a, b, c) - 90.0) < 1e-6
 
 
def test_compute_angles_known_geometry():
    lms = [{"x": 0.0, "y": 0.0, "v": 1.0} for _ in range(33)]
 
    lms[LEL] = {"x": 0.0, "y": 0.0, "v": 1.0}
    lms[LAW] = {"x": 1.0, "y": 0.0, "v": 1.0}
    lms[LSH] = {"x": 0.0, "y": 1.0, "v": 1.0}
 
    lms[LIH] = {"x": 0.0, "y": 2.0, "v": 1.0}
    lms[LEK] = {"x": 1.0, "y": 2.0, "v": 1.0}
    lms[LAN] = {"x": 2.0, "y": 2.0, "v": 1.0}
 
    lms[REL] = {"x": 0.0, "y": 0.0, "v": 1.0}
    lms[RAW] = {"x": -1.0, "y": 0.0, "v": 1.0}
    lms[RSH] = {"x": 0.0, "y": 1.0, "v": 1.0}
 
    lms[RIH] = {"x": 0.0, "y": 2.0, "v": 1.0}
    lms[REK] = {"x": -1.0, "y": 2.0, "v": 1.0}
    lms[RAN] = {"x": -2.0, "y": 2.0, "v": 1.0}
 
    angs = compute_angles(lms)
 
    assert abs(angs["leftElbow"] - 90.0) < 1e-6
    assert abs(angs["rightElbow"] - 90.0) < 1e-6
 
    assert math.isfinite(angs["leftShoulder"])
    assert math.isfinite(angs["rightShoulder"])
    assert math.isfinite(angs["leftHip"])
    assert math.isfinite(angs["rightHip"])
    assert math.isfinite(angs["leftKnee"])
    assert math.isfinite(angs["rightKnee"])
 
 
if __name__ == "__main__":
    test_score_diff_tolerance_and_bounds()
    test_dist_to_score_tolerance_and_monotonicity()
    test_angle_between_degenerate_and_extremes()
    test_compute_angles_known_geometry()
    print("ALL TESTS PASSED")
