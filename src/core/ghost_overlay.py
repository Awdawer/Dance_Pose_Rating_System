import numpy as np
import cv2
from src.core.scoring import _w_center_scale, _select_points_with_weights


def align_skeleton_to_user(user_lms, ref_lms):
    """
    将参考骨架对齐到用户体型空间，实现幽灵图叠加。
    
    使用Procrustes分析的核心思想：
    1. 分别对用户和参考骨架进行质心归一化（消除位移和尺度差异）
    2. 计算最佳旋转矩阵，使参考骨架与用户骨架对齐
    3. 将参考骨架变换回用户的空间（应用用户的尺度和位移）
    
    参数:
        user_lms: 用户关键点列表 (MediaPipe格式)
        ref_lms: 参考关键点列表 (MediaPipe格式)
    
    返回:
        aligned_ref_pts: 对齐后的参考骨架点列表 [(x, y), ...]，坐标是归一化的(0-1)
        transform_info: 变换信息字典，用于调试
    """
    if not user_lms or not ref_lms:
        return None, None
    
    # 提取关键点和权重
    user_pts, user_w = _select_points_with_weights(user_lms)
    ref_pts, ref_w = _select_points_with_weights(ref_lms)
    
    # 过滤掉权重为0的点
    mask = (user_w > 1e-6) & (ref_w > 1e-6)
    mask &= ~(np.isclose(user_pts[:, 0], 0.0) & np.isclose(user_pts[:, 1], 0.0))
    mask &= ~(np.isclose(ref_pts[:, 0], 0.0) & np.isclose(ref_pts[:, 1], 0.0))
    
    if np.count_nonzero(mask) < 3:
        # 有效点太少，无法进行对齐
        return None, None
    
    user_pts_valid = user_pts[mask]
    ref_pts_valid = ref_pts[mask]
    weights_valid = user_w[mask]
    
    # 分别计算质心和尺度
    user_norm, user_scale, user_center = _w_center_scale(user_pts_valid, weights_valid)
    ref_norm, ref_scale, ref_center = _w_center_scale(ref_pts_valid, weights_valid)
    
    if user_scale == 0.0 or ref_scale == 0.0:
        return None, None
    
    # 计算加权协方差矩阵，求解最佳旋转
    sw = np.sum(weights_valid)
    M = (user_norm * weights_valid[:, None]).T @ ref_norm / sw
    
    # SVD分解求最佳旋转矩阵
    try:
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        # 确保是正常旋转（行列式为1），而非反射
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    except np.linalg.LinAlgError:
        return None, None
    
    # 对完整的参考骨架点进行变换（包括原始33个MediaPipe点）
    # 首先将参考点转换到归一化空间
    ref_full_pts = np.array([[p["x"], p["y"]] for p in ref_lms], dtype=np.float32)
    
    # 应用变换: 
    # 1. 减参考质心
    # 2. 除以参考尺度（归一化）
    # 3. 应用旋转矩阵
    # 4. 乘以用户尺度
    # 5. 加用户质心
    ref_centered = ref_full_pts - ref_center
    ref_normalized = ref_centered / ref_scale
    ref_rotated = ref_normalized @ R
    ref_scaled = ref_rotated * user_scale
    ref_aligned = ref_scaled + user_center
    
    # 转换回列表格式
    aligned_ref_pts = [(float(p[0]), float(p[1])) for p in ref_aligned]
    
    transform_info = {
        "user_scale": float(user_scale),
        "ref_scale": float(ref_scale),
        "user_center": user_center.tolist(),
        "ref_center": ref_center.tolist(),
        "scale_ratio": float(user_scale / ref_scale) if ref_scale > 0 else 1.0
    }
    
    return aligned_ref_pts, transform_info


def draw_ghost_skeleton(canvas, aligned_pts, color=(0, 255, 0), alpha=0.5, thickness=3):
    """
    在画布上绘制幽灵骨架（半透明绿色）。
    
    参数:
        canvas: OpenCV图像 (BGR格式)
        aligned_pts: 对齐后的骨架点列表 [(x, y), ...]，坐标是像素坐标
        color: 骨架颜色 (B, G, R)
        alpha: 透明度 (0-1)
        thickness: 线条粗细
    """
    from src.utils.geometry import POSE_CONNECTIONS
    
    h, w = canvas.shape[:2]
    
    # 创建透明覆盖层
    overlay = canvas.copy()
    
    # 绘制骨架连线
    for a, b in POSE_CONNECTIONS:
        if a < len(aligned_pts) and b < len(aligned_pts):
            pt1 = (int(aligned_pts[a][0] * w), int(aligned_pts[a][1] * h))
            pt2 = (int(aligned_pts[b][0] * w), int(aligned_pts[b][1] * h))
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # 绘制关键点
    body_idxs = list(range(11, 33))  # 身体关键点（排除面部）
    for i in body_idxs:
        if i < len(aligned_pts):
            x = int(aligned_pts[i][0] * w)
            y = int(aligned_pts[i][1] * h)
            cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)
    
    # 混合覆盖层和原图
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    
    return canvas


def draw_comparison_skeleton(canvas, user_lms, aligned_ref_pts, diffs=None):
    """
    绘制用户骨架和参考骨架的对比图。
    
    参数:
        canvas: OpenCV图像 (BGR格式)
        user_lms: 用户关键点列表
        aligned_ref_pts: 对齐后的参考骨架点列表 [(x, y), ...]，归一化坐标
        diffs: 关节角度差异字典（可选）
    """
    from src.utils.geometry import POSE_CONNECTIONS
    
    h, w = canvas.shape[:2]
    
    # 绘制参考骨架（幽灵图 - 半透明绿色）
    if aligned_ref_pts:
        overlay = canvas.copy()
        ghost_color = (0, 255, 0)  # 绿色
        
        for a, b in POSE_CONNECTIONS:
            if a < len(aligned_ref_pts) and b < len(aligned_ref_pts):
                pt1 = (int(aligned_ref_pts[a][0] * w), int(aligned_ref_pts[a][1] * h))
                pt2 = (int(aligned_ref_pts[b][0] * w), int(aligned_ref_pts[b][1] * h))
                cv2.line(overlay, pt1, pt2, ghost_color, 4, cv2.LINE_AA)
        
        # 绘制参考骨架的关键点
        body_idxs = list(range(11, 33))
        for i in body_idxs:
            if i < len(aligned_ref_pts):
                x = int(aligned_ref_pts[i][0] * w)
                y = int(aligned_ref_pts[i][1] * h)
                cv2.circle(overlay, (x, y), 6, ghost_color, -1, cv2.LINE_AA)
        
        # 混合
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    
    # 绘制用户骨架（实色蓝色）
    if user_lms:
        user_pts = [(p["x"], p["y"]) for p in user_lms]
        user_color = (255, 128, 0)  # 蓝色
        
        for a, b in POSE_CONNECTIONS:
            if a < len(user_pts) and b < len(user_pts):
                pt1 = (int(user_pts[a][0] * w), int(user_pts[a][1] * h))
                pt2 = (int(user_pts[b][0] * w), int(user_pts[b][1] * h))
                cv2.line(canvas, pt1, pt2, user_color, 2, cv2.LINE_AA)
        
        # 绘制用户关键点
        body_idxs = list(range(11, 33))
        for i in body_idxs:
            if i < len(user_pts):
                x = int(user_pts[i][0] * w)
                y = int(user_pts[i][1] * h)
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        
        # 如果有差异信息，在关节处显示
        if diffs:
            idxs = {
                "leftShoulder": 11, "rightShoulder": 12,
                "leftElbow": 13, "rightElbow": 14,
                "leftHip": 23, "rightHip": 24,
                "leftKnee": 25, "rightKnee": 26
            }
            for k, i in idxs.items():
                if i < len(user_pts):
                    d = diffs.get(k, 0.0)
                    if d >= 40:
                        color = (0, 0, 255)  # 红色 - 差异大
                    elif d >= 20:
                        color = (0, 165, 255)  # 橙色 - 差异中等
                    else:
                        color = (0, 255, 0)  # 绿色 - 差异小
                    
                    x = int(user_pts[i][0] * w)
                    y = int(user_pts[i][1] * h)
                    cv2.putText(canvas, f"{int(round(d))}", (x + 8, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    return canvas
