import numpy as np


def score_diff(d, tolerance=3.0):
    """
    将角度差异（度）转换为评分（0-100）。
    添加了容差参数以处理小的检测误差。
    
    评分公式: score = 100 - 1.5 * (diff - tolerance)
    例如:
        - 0度差异 -> 100分
        - 20度差异 -> 70分 (100 - 1.5*20)
        - 66度差异 -> 0分
    
    参数:
        d: 角度差异（度）
        tolerance: 容差阈值，低于此值的差异视为完美匹配，得满分
    
    返回:
        float: 0-100之间的评分
    """
    # 应用容差：小差异视为完美匹配
    effective_diff = max(0.0, float(d) - tolerance)
    
    # 线性衰减公式
    s = 100.0 - 1.5 * effective_diff
    
    # 限制在0-100范围内
    if s < 0.0:
        s = 0.0
    if s > 100.0:
        s = 100.0
    return s


def _dist_to_score(d, k=3.0, gamma=0.5, tolerance=0.02):
    """
    将Procrustes距离转换为评分（0-100）。
    使用指数衰减映射，对微小形状差异敏感。
    
    评分公式: score = 100 * exp(-k * d) ^ gamma
    
    参数:
        d: Procrustes距离
        k: 衰减系数（值越大，对差异越敏感）
        gamma: Gamma校正参数（调整曲线形状）
        tolerance: 距离容差，低于此值视为完美匹配
    
    返回:
        float: 0-100之间的评分
    """
    # 如果距离在容差范围内，返回满分
    if d <= tolerance:
        return 100.0
    
    # 应用容差
    effective_d = d - tolerance
    
    # 指数衰减
    p = float(np.exp(-k * effective_d))
    
    # Gamma校正调整曲线
    p = p ** gamma
    
    s = 100.0 * p
    
    # 限制在0-100范围内
    if s < 0.0:
        s = 0.0
    if s > 100.0:
        s = 100.0
    return s


def _w_center_scale(arr, w):
    """
    加权质心归一化。
    将点集平移到质心原点，并缩放到单位尺度。
    
    参数:
        arr: Nx2的坐标数组
        w: N的权重数组
    
    返回:
        x: 归一化后的坐标 (Nx2)
        s: 尺度因子
        c: 质心坐标 (2,)
    """
    sw = np.sum(w)
    if sw <= 0.0:
        return arr*0.0, 0.0, np.array([0.0, 0.0], dtype=np.float32)
    
    # 计算加权质心
    c = np.sum(arr * w[:, None], axis=0) / sw
    
    # 平移到原点
    x = arr - c
    
    # 计算加权尺度
    s = np.sqrt(np.sum(w * np.sum(x*x, axis=1)) / sw)
    
    if s <= 1e-12:
        return x*0.0, 0.0, c
    
    # 归一化
    return x / s, s, c


def _w_procrustes_dist(a, b, w):
    """
    计算两组点之间的加权Procrustes距离。
    
    Procrustes分析步骤:
    1. 分别对两组点进行质心归一化
    2. 通过SVD找到最佳旋转矩阵
    3. 计算旋转后的欧氏距离
    
    参数:
        a: 第一组点，Nx2数组
        b: 第二组点，Nx2数组
        w: 权重数组，N
    
    返回:
        float: Procrustes距离（越小表示形状越相似）
    """
    # 过滤掉权重接近0或坐标为0的点
    mask = (w > 1e-6)
    mask &= ~(np.isclose(a[:,0],0.0)&np.isclose(a[:,1],0.0))
    mask &= ~(np.isclose(b[:,0],0.0)&np.isclose(b[:,1],0.0))
    
    # 如果有效点少于3个，返回大距离
    if np.count_nonzero(mask) < 3:
        return 1e9
    
    A = a[mask]; B = b[mask]; W = w[mask]
    
    # 分别归一化
    A, sa, ca = _w_center_scale(A, W)
    B, sb, cb = _w_center_scale(B, W)
    
    if sa == 0.0 or sb == 0.0:
        return 1e9
    
    # 计算加权协方差矩阵
    sw = np.sum(W)
    M = (A * W[:, None]).T @ B / sw
    
    # SVD分解求最佳旋转
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # 应用旋转
    AR = A @ R
    
    # 计算加权残差距离
    resid = AR - B
    d2 = np.sum(W * np.sum(resid*resid, axis=1)) / sw
    
    return float(np.sqrt(d2))


def _select_points_with_weights(lms, include_hands=False):
    """
    选择关键 landmarks 并分配权重，用于Procrustes分析。
    
    权重设计原则:
    - 四肢末端（手腕、脚踝）权重最高(2.0)，因为这些部位动作幅度大、区分度高
    - 肘部、膝盖权重较高(1.5)，是动作的关键节点
    - 肩部、胯部权重中等(1.0)，提供躯干方向信息
    - 可见度低的点权重设为0
    
    参数:
        lms: MediaPipe输出的33个关键点列表
        include_hands: 是否包含手指关键点
    
    返回:
        pts: 选中的关键点坐标数组 (Nx2)
        ws: 对应的权重数组 (N,)
    """
    pts = []
    ws = []
    
    def get_v(i):
        """获取第i个关键点的可见度"""
        if i is None or i >= len(lms): return 0.0
        p = lms[i]
        if isinstance(p, dict):
            return p.get("v", 1.0)
        return 1.0

    def add_idx(i, w):
        """
        添加一个关键点到列表。
        如果关键点不存在或可见度低，则添加(0,0)并设置权重为0。
        """
        if i is None:
            pts.append([0.0, 0.0]); ws.append(0.0); return
            
        vis = get_v(i)
        if vis < 0.5:
            w = 0.0
            
        if i < len(lms):
            pts.append([float(lms[i]["x"]), float(lms[i]["y"])]); ws.append(float(w))
        else:
            pts.append([0.0, 0.0]); ws.append(0.0)
            
    # 权重调整：强调四肢末端而非躯干
    # 基础身体点（不包含面部）
    
    # 肩部
    add_idx(11, 1.0); add_idx(12, 1.0)  # 左肩、右肩
    
    # 肘部（权重增加，手臂动作的关键节点）
    add_idx(13, 1.5); add_idx(14, 1.5)  # 左肘、右肘
    
    # 手腕（权重最高，手臂姿态的末端指示器）
    add_idx(15, 2.0); add_idx(16, 2.0)  # 左手腕、右手腕
    
    # 胯部
    add_idx(23, 1.0); add_idx(24, 1.0)  # 左胯、右胯
    
    # 膝盖（权重增加，腿部动作的关键节点）
    add_idx(25, 1.5); add_idx(26, 1.5)  # 左膝、右膝
    
    # 脚踝（权重最高，腿部姿态的末端指示器）
    add_idx(27, 2.0); add_idx(28, 2.0)  # 左踝、右踝
    
    # 脚跟和脚尖
    add_idx(29, 1.5); add_idx(30, 1.5)  # 左脚跟、右脚跟
    add_idx(31, 1.5); add_idx(32, 1.5)  # 左脚尖、右脚尖
    
    # 可选：手指关键点（权重较小）
    if include_hands:
        add_idx(17, 1.0); add_idx(18, 1.0)
        add_idx(19, 1.0); add_idx(20, 1.0)
        add_idx(21, 1.0); add_idx(22, 1.0)
    
    # 计算躯干中点（肩中点和胯中点）
    def get_xy(idx):
        """获取第idx个关键点的坐标"""
        if idx is None or idx >= len(lms): return None
        p = lms[idx]; return float(p["x"]), float(p["y"])
    
    ls = get_xy(11); rs = get_xy(12)  # 左右肩
    lh = get_xy(23); rh = get_xy(24)  # 左右胯
    lsv = get_v(11); rsv = get_v(12)  # 肩部可见度
    lhv = get_v(23); rhv = get_v(24)  # 胯部可见度
    
    # 添加肩中点
    if ls and rs:
        add_idx(None, 0.0)  # 占位符
        if lsv < 0.5 or rsv < 0.5:
             pts[-1] = [ (ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0 ]; ws[-1] = 0.0
        else:
             pts[-1] = [ (ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0 ]; ws[-1] = 1.0
    else:
        add_idx(None, 0.0)
    
    # 添加胯中点
    if lh and rh:
        add_idx(None, 0.0)
        if lhv < 0.5 or rhv < 0.5:
            pts[-1] = [ (lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0 ]; ws[-1] = 0.0
        else:
            pts[-1] = [ (lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0 ]; ws[-1] = 1.0
    else:
        add_idx(None, 0.0)
    
    return np.array(pts, dtype=np.float32), np.array(ws, dtype=np.float32)


def score_angles(user_angles, ref_angles, angle_weights=None):
    """
    基于关节角度差异计算姿态评分。
    
    这是双重加权评分模型的第一维度（占60%权重）。
    计算8个核心关节（肩、肘、胯、膝）的角度差异，加权平均得到最终分数。
    
    参数:
        user_angles: 用户姿态的角度字典，如 {"leftShoulder": 45.0, ...}
        ref_angles: 参考姿态的角度字典
        angle_weights: 角度置信度字典，用于处理遮挡情况
    
    返回:
        percent: 0-100之间的评分
        diffs: 各关节的角度差异字典
    """
    diffs = {}
    
    # 基础权重配置
    weights = {
        "leftShoulder": 1.0,   # 左肩
        "rightShoulder": 1.0,  # 右肩
        "leftElbow": 1.0,      # 左肘
        "rightElbow": 1.0,     # 右肘
        "leftHip": 1.2,        # 左胯（权重略高）
        "rightHip": 1.2,       # 右胯
        "leftKnee": 1.2,       # 左膝
        "rightKnee": 1.2,      # 右膝
    }
    
    total = 0.0
    sum_w = 0.0
    
    for k, w in weights.items():
        # 根据置信度调整权重（处理遮挡）
        if angle_weights and k in angle_weights:
            conf = angle_weights[k]
            if conf < 0.5:
                w = 0.0  # 低置信度关节不参与评分
        
        # 计算角度差异
        d = abs(user_angles.get(k, 0.0) - ref_angles.get(k, 0.0))
        diffs[k] = d
        
        # 转换为评分并累加
        pj = score_diff(d)
        total += pj * w
        sum_w += w
    
    # 加权平均
    percent = 0.0
    if sum_w > 0:
        percent = total / sum_w
    
    # 限制在0-100范围内
    if percent < 0.0:
        percent = 0.0
    if percent > 100.0:
        percent = 100.0
    
    return percent, diffs
