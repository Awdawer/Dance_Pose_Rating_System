import os
import sys
import math
import numpy as np
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from src.core.scoring import _select_points_with_weights, _w_procrustes_dist, _dist_to_score
 
 
def _make_landmarks(points_by_idx, vis_by_idx=None):
    lms = [{"x": 0.0, "y": 0.0, "v": 0.0} for _ in range(33)]
    for idx, (x, y) in points_by_idx.items():
        lms[idx] = {"x": float(x), "y": float(y), "v": 1.0}
    if vis_by_idx:
        for idx, v in vis_by_idx.items():
            if 0 <= idx < len(lms):
                lms[idx]["v"] = float(v)
    return lms
 
 
def _base_pose_landmarks():
    pts = {
        11: (0.40, 0.20),
        12: (0.60, 0.20),
        13: (0.35, 0.35),
        14: (0.65, 0.35),
        15: (0.30, 0.50),
        16: (0.70, 0.50),
        23: (0.43, 0.55),
        24: (0.57, 0.55),
        25: (0.42, 0.72),
        26: (0.58, 0.72),
        27: (0.41, 0.90),
        28: (0.59, 0.90),
        29: (0.40, 0.92),
        30: (0.60, 0.92),
        31: (0.39, 0.95),
        32: (0.61, 0.95),
    }
    return _make_landmarks(pts)
 
 
def _transform_landmarks(lms, scale=1.0, angle_rad=0.0, translation=(0.0, 0.0)):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    tx, ty = translation
 
    out = []
    for p in lms:
        x = float(p["x"])
        y = float(p["y"])
        xr = x * c - y * s
        yr = x * s + y * c
        out.append({"x": scale * xr + tx, "y": scale * yr + ty, "v": float(p.get("v", 1.0))})
    return out
 
 
def test_weighted_procrustes_invariant_to_similarity():
    a = _base_pose_landmarks()
    b = _transform_landmarks(a, scale=1.7, angle_rad=0.6, translation=(0.12, -0.08))
 
    pa, wa = _select_points_with_weights(a)
    pb, wb = _select_points_with_weights(b)
 
    d = _w_procrustes_dist(pa, pb, np.minimum(wa, wb))
    assert d < 1e-4, f"Expected near-zero Procrustes distance, got {d}"
 
    score = _dist_to_score(d)
    assert score >= 99.9, f"Expected near-100 score for near-zero distance, got {score}"
 
 
def test_low_visibility_points_are_ignored_via_min_weights():
    a = _base_pose_landmarks()
    b = _base_pose_landmarks()
 
    a[15]["v"] = 0.1
    b[15]["x"] = 0.95
    b[15]["y"] = 0.05
 
    pa, wa = _select_points_with_weights(a)
    pb, wb = _select_points_with_weights(b)
 
    d = _w_procrustes_dist(pa, pb, np.minimum(wa, wb))
    assert d < 1e-4, f"Expected low distance when low-visibility point is ignored, got {d}"
 
 
def test_procrustes_returns_large_distance_when_too_few_points():
    vis = {i: 0.0 for i in range(33)}
    vis[12] = 1.0
    vis[24] = 1.0
 
    a = _base_pose_landmarks()
    b = _base_pose_landmarks()
    for i in range(33):
        a[i]["v"] = vis.get(i, 0.0)
        b[i]["v"] = vis.get(i, 0.0)
 
    pa, wa = _select_points_with_weights(a)
    pb, wb = _select_points_with_weights(b)
 
    d = _w_procrustes_dist(pa, pb, np.minimum(wa, wb))
    assert d >= 1e8, f"Expected very large distance when valid points < 3, got {d}"
 
 
if __name__ == "__main__":
    test_weighted_procrustes_invariant_to_similarity()
    test_low_visibility_points_are_ignored_via_min_weights()
    test_procrustes_returns_large_distance_when_too_few_points()
    print("ALL TESTS PASSED")
