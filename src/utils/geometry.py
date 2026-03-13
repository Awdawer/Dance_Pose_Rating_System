import math
import numpy as np

# MediaPipe Pose connections
POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32)
]

# Landmark indices
LSH, LEL, LIH, LEK, LAW, LAN = 11, 13, 23, 25, 15, 27
RSH, REL, RIH, REK, RAW, RAN = 12, 14, 24, 26, 16, 28

def angle_between(a, b, c):
    """Calculate the angle between three points (b is the vertex)."""
    ab = np.array([a[0]-b[0], a[1]-b[1]], dtype=np.float32)
    cb = np.array([c[0]-b[0], c[1]-b[1]], dtype=np.float32)
    mab = np.linalg.norm(ab)
    mcb = np.linalg.norm(cb)
    if mab == 0 or mcb == 0:
        return 0.0
    cosv = float(np.dot(ab, cb) / (mab * mcb))
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def pick_xy(lms, i):
    """Helper to pick x, y from landmark list/dict."""
    if i >= len(lms): return (0.0, 0.0)
    p = lms[i]
    if isinstance(p, dict):
        return (p["x"], p["y"])
    return (p["x"], p["y"])

def pick_vis(lms, i):
    """Helper to pick visibility from landmark list/dict."""
    if i >= len(lms): return 0.0
    p = lms[i]
    if isinstance(p, dict):
        return p.get("v", 1.0)
    if hasattr(p, "visibility"):
        return float(p.visibility)
    return 1.0

def compute_angles(lms):
    """Compute critical joint angles from landmarks."""
    # Uses global constants LSH, LEL, etc.
    
    leftShoulder = angle_between(pick_xy(lms, LEL), pick_xy(lms, LSH), pick_xy(lms, LIH))
    rightShoulder = angle_between(pick_xy(lms, REL), pick_xy(lms, RSH), pick_xy(lms, RIH))
    leftElbow = angle_between(pick_xy(lms, LAW), pick_xy(lms, LEL), pick_xy(lms, LSH))
    rightElbow = angle_between(pick_xy(lms, RAW), pick_xy(lms, REL), pick_xy(lms, RSH))
    leftHip = angle_between(pick_xy(lms, LEK), pick_xy(lms, LIH), pick_xy(lms, LSH))
    rightHip = angle_between(pick_xy(lms, REK), pick_xy(lms, RIH), pick_xy(lms, RSH))
    leftKnee = angle_between(pick_xy(lms, LAN), pick_xy(lms, LEK), pick_xy(lms, LIH))
    rightKnee = angle_between(pick_xy(lms, RAN), pick_xy(lms, REK), pick_xy(lms, RIH))
    
    return {
        "leftShoulder": leftShoulder,
        "rightShoulder": rightShoulder,
        "leftElbow": leftElbow,
        "rightElbow": rightElbow,
        "leftHip": leftHip,
        "rightHip": rightHip,
        "leftKnee": leftKnee,
        "rightKnee": rightKnee,
    }

def compute_angle_confidence(lms):
    """Compute confidence for each angle based on landmark visibility."""
    
    def min_vis(idx_list):
        return min([pick_vis(lms, i) for i in idx_list])

    return {
        "leftShoulder": min_vis([LEL, LSH, LIH]),
        "rightShoulder": min_vis([REL, RSH, RIH]),
        "leftElbow": min_vis([LAW, LEL, LSH]),
        "rightElbow": min_vis([RAW, REL, RSH]),
        "leftHip": min_vis([LEK, LIH, LSH]),
        "rightHip": min_vis([REK, RIH, RSH]),
        "leftKnee": min_vis([LAN, LEK, LIH]),
        "rightKnee": min_vis([RAN, REK, RIH]),
    }
