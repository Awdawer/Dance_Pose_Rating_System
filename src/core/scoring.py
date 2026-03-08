import numpy as np

def score_diff(d):
    """Convert an angular difference (in degrees) to a score (0-100)."""
    # 0 deg diff -> 100 score
    # 20 deg diff -> 70 score (100 - 1.5*20)
    # 66 deg diff -> 0 score
    s = 100.0 - 1.5 * float(d)
    if s < 0.0:
        s = 0.0
    if s > 100.0:
        s = 100.0
    return s

def _dist_to_score(d, k=5.0, gamma=0.7):
    """Convert a distance metric (Procrustes) to a score (0-100)."""
    if d <= 0.0:
        return 100.0
    # Exponential decay
    p = float(np.exp(-k * d))
    # Gamma correction to adjust curve
    p = p ** gamma
    s = 100.0 * p
    if s < 0.0:
        s = 0.0
    if s > 100.0:
        s = 100.0
    return s

def _w_center_scale(arr, w):
    """Weighted center and scale normalization."""
    sw = np.sum(w)
    if sw <= 0.0:
        return arr*0.0, 0.0, np.array([0.0, 0.0], dtype=np.float32)
    c = np.sum(arr * w[:, None], axis=0) / sw
    x = arr - c
    s = np.sqrt(np.sum(w * np.sum(x*x, axis=1)) / sw)
    if s <= 1e-12:
        return x*0.0, 0.0, c
    return x / s, s, c

def _w_procrustes_dist(a, b, w):
    """Weighted Procrustes distance between two point sets."""
    # a,b: Nx2, w: N (>=0); ignore points with near-zero weight or zeros in either set
    mask = (w > 1e-6)
    mask &= ~(np.isclose(a[:,0],0.0)&np.isclose(a[:,1],0.0))
    mask &= ~(np.isclose(b[:,0],0.0)&np.isclose(b[:,1],0.0))
    if np.count_nonzero(mask) < 3:
        return 1e9
    A = a[mask]; B = b[mask]; W = w[mask]
    A, sa, ca = _w_center_scale(A, W)
    B, sb, cb = _w_center_scale(B, W)
    if sa == 0.0 or sb == 0.0:
        return 1e9
    # Weighted covariance
    sw = np.sum(W)
    M = (A * W[:, None]).T @ B / sw
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    AR = A @ R
    resid = AR - B
    d2 = np.sum(W * np.sum(resid*resid, axis=1)) / sw
    return float(np.sqrt(d2))

def _select_points_with_weights(lms, include_hands=False):
    """Select key landmarks and assign weights for Procrustes analysis."""
    pts = []
    ws = []
    def add_idx(i, w):
        if i is None:
            pts.append([0.0, 0.0]); ws.append(0.0); return
        if i < len(lms):
            pts.append([float(lms[i]["x"]), float(lms[i]["y"])]); ws.append(float(w))
        else:
            pts.append([0.0, 0.0]); ws.append(0.0)
            
    # Weights adjusted to emphasize limbs over torso
    # Base body points (no face)
    # Shoulders, elbows, wrists
    add_idx(11, 1.0); add_idx(12, 1.0) # Shoulders (reduced from 1.3)
    add_idx(13, 1.5); add_idx(14, 1.5) # Elbows (increased from 1.0)
    add_idx(15, 2.0); add_idx(16, 2.0) # Wrists (increased from 0.9, critical for arm pose)
    # Hips, knees, ankles
    add_idx(23, 1.0); add_idx(24, 1.0) # Hips (reduced from 1.3)
    add_idx(25, 1.5); add_idx(26, 1.5) # Knees (increased from 1.2)
    add_idx(27, 2.0); add_idx(28, 2.0) # Ankles (increased from 1.2, critical for leg pose)
    # Heels + foot index
    add_idx(29, 1.5); add_idx(30, 1.5) # Heels (increased from 1.1)
    add_idx(31, 1.5); add_idx(32, 1.5) # Foot index (increased from 1.1)
    # Optional hands (pinky, index, thumb) with small weight
    if include_hands:
        add_idx(17, 1.0); add_idx(18, 1.0)
        add_idx(19, 1.0); add_idx(20, 1.0)
        add_idx(21, 1.0); add_idx(22, 1.0)
    # Derived midpoints for torso orientation (heavier)
    def get_xy(idx):
        if idx is None or idx >= len(lms): return None
        p = lms[idx]; return float(p["x"]), float(p["y"])
    ls = get_xy(11); rs = get_xy(12)
    lh = get_xy(23); rh = get_xy(24)
    if ls and rs:
        add_idx(None, 0.0)  # placeholder to keep list alignment
        pts[-1] = [ (ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0 ]; ws[-1] = 1.0 # Mid-Shoulder (reduced from 1.5)
    else:
        add_idx(None, 0.0)
    if lh and rh:
        add_idx(None, 0.0)
        pts[-1] = [ (lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0 ]; ws[-1] = 1.0 # Mid-Hip (reduced from 1.5)
    else:
        add_idx(None, 0.0)
    return np.array(pts, dtype=np.float32), np.array(ws, dtype=np.float32)

def score_angles(user_angles, ref_angles):
    """Score the pose based on angular differences."""
    diffs = {}
    weights = {
        "leftShoulder": 1.0,
        "rightShoulder": 1.0,
        "leftElbow": 1.0,
        "rightElbow": 1.0,
        "leftHip": 1.2,
        "rightHip": 1.2,
        "leftKnee": 1.2,
        "rightKnee": 1.2,
    }
    total = 0.0
    sum_w = 0.0
    for k, w in weights.items():
        d = abs(user_angles.get(k, 0.0) - ref_angles.get(k, 0.0))
        diffs[k] = d
        pj = score_diff(d)
        total += pj * w
        sum_w += w
    percent = 0.0
    if sum_w > 0:
        percent = total / sum_w
    if percent < 0.0:
        percent = 0.0
    if percent > 100.0:
        percent = 100.0
    return percent, diffs
