import time
import cv2
import numpy as np
from queue import Queue, Empty
from PyQt5 import QtCore

from src.utils.geometry import compute_angles, compute_angle_confidence
from src.core.scoring import score_angles, _select_points_with_weights, _w_procrustes_dist, _dist_to_score

class PoseWorker(QtCore.QObject):
    """
    Worker class for heavy MediaPipe processing.
    Runs in a separate thread to keep the GUI responsive.
    """
    results_ready = QtCore.pyqtSignal(list, list, dict, float, float, str) # u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.detector = None
        self.running = True
        self.queue = Queue(maxsize=1)
        self.roi_user = None
        self.roi_ref = None
        self.k_decay = 5.0
        self.gamma = 0.7
        
        # 历史缓冲区 - 只存储参考视频的历史，用于DTW查找
        self.ref_history = [] # List of (landmarks, angles)
        self.HISTORY_LEN = 15 # 存储约0.5秒的参考帧 (30fps * 0.5s)

    def start(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Worker init failed: {e}")
            return

        while self.running:
            try:
                # Wait for a new frame pair to process
                item = self.queue.get(timeout=0.1)
                u_frame, r_frame, k_decay, gamma = item
                # Update scoring params from GUI
                self.k_decay = float(k_decay)
                self.gamma = float(gamma)

                u_lms = []
                r_lms = []
                if u_frame is not None:
                    u_lms, self.roi_user = self.detect_landmarks(u_frame, self.roi_user)
                if r_frame is not None:
                    r_lms, self.roi_ref = self.detect_landmarks(r_frame, self.roi_ref)

                real_time_percent = 0.0
                dtw_percent = 0.0
                diffs = {}
                timing_hint = ""
                
                if len(u_lms) and len(r_lms):
                    # 计算两个评分
                    real_time_percent, dtw_percent, diffs = self.compute_dual_scores(u_lms, r_lms)
                    timing_hint = self.check_timing(u_lms)

                self.set_analysis_data(u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint)
                self.results_ready.emit(u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint)
            except Empty:
                continue
            except Exception as e:
                print(f"Worker loop error: {e}")

    def stop(self):
        self.running = False

    def detect_landmarks(self, frame_bgr, roi, max_long_side=360):
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        if roi is None:
            x0, y0, cw, ch = 0, 0, w, h
        else:
            x0, y0, cw, ch = roi

        crop = frame_bgr[y0:y0+ch, x0:x0+cw] if ch > 0 and cw > 0 else frame_bgr
        ih, iw = crop.shape[:2]
        if max(ih, iw) > max_long_side:
            s = max_long_side / float(max(ih, iw))
            crop = cv2.resize(crop, (int(iw*s), int(ih*s)), interpolation=cv2.INTER_AREA)
            ih, iw = crop.shape[:2]

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = self.detector.detect(mp_image)

        if not res or not res.pose_landmarks or len(res.pose_landmarks) == 0:
            return [], (0, 0, w, h)

        out = []
        for lm in res.pose_landmarks[0]:
            fx = (x0 + float(lm.x) * cw) / float(w)
            fy = (y0 + float(lm.y) * ch) / float(h)
            out.append({"x": fx, "y": fy, "v": float(lm.visibility)})

        # Update ROI for next frame
        xs = [p["x"] for p in out]
        ys = [p["y"] for p in out]
        minx = max(0, int(min(xs) * w) - int(0.1 * w))
        maxx = min(w, int(max(xs) * w) + int(0.1 * w))
        miny = max(0, int(min(ys) * h) - int(0.1 * h))
        maxy = min(h, int(max(ys) * h) + int(0.1 * h))
        return out, (minx, miny, max(64, maxx - minx), max(64, maxy - miny))

    def compute_dual_scores(self, u_lm, r_lm):
        """
        计算两个评分：
        1. 实时评分：用户当前帧 vs 参考当前帧
        2. DTW评分：用户当前帧 vs 参考历史中最相似帧
        
        Returns:
            (real_time_percent, dtw_percent, diffs)
        """
        # 计算角度
        u_angs = compute_angles(u_lm)
        r_angs = compute_angles(r_lm)
        
        # 更新参考历史（只存参考帧）
        self.ref_history.append((r_lm, r_angs))
        if len(self.ref_history) > self.HISTORY_LEN:
            self.ref_history.pop(0)
        
        # ========== 1. 实时评分 ==========
        real_time_percent = self._compute_score(u_lm, u_angs, r_lm, r_angs)
        
        # ========== 2. DTW评分 ==========
        # 在历史参考帧中找到与用户当前帧最相似的一帧
        best_r_lm, best_r_angs = self._find_best_match_in_history(u_lm, u_angs)
        
        if best_r_lm is not None:
            dtw_percent = self._compute_score(u_lm, u_angs, best_r_lm, best_r_angs)
        else:
            # 历史不足，使用当前帧
            dtw_percent = real_time_percent
            best_r_lm = r_lm
            best_r_angs = r_angs
        
        # 计算差异（用于可视化）- 使用实时评分的差异
        u_conf = compute_angle_confidence(u_lm)
        r_conf = compute_angle_confidence(r_lm)
        combined_conf = {k: min(u_conf.get(k, 0.0), r_conf.get(k, 0.0)) for k in u_conf}
        _, diffs = score_angles(u_angs, r_angs, angle_weights=combined_conf)
        
        return real_time_percent, dtw_percent, diffs
    
    def _compute_score(self, u_lm, u_angs, r_lm, r_angs):
        """计算单个评分"""
        # 角度置信度
        u_conf = compute_angle_confidence(u_lm)
        r_conf = compute_angle_confidence(r_lm)
        combined_conf = {k: min(u_conf.get(k, 0.0), r_conf.get(k, 0.0)) for k in u_conf}
        
        # 角度评分
        angle_percent, angle_diffs = score_angles(u_angs, r_angs, angle_weights=combined_conf)
        
        # Procrustes形状评分
        pu, wu = _select_points_with_weights(u_lm)
        pr, wr = _select_points_with_weights(r_lm)
        d = _w_procrustes_dist(pu, pr, np.minimum(wu, wr))
        procrustes_percent = _dist_to_score(d, k=self.k_decay, gamma=self.gamma)
        
        # 融合 - 只使用角度评分
        final_score = angle_percent
        
        # Debug output (every 30 frames)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 30 == 0:
            avg_diff = sum(angle_diffs.values()) / len(angle_diffs) if angle_diffs else 0
            print(f"[Score] Angle: {angle_percent:.1f}%, Avg angle diff: {avg_diff:.1f}°")
        
        return final_score
    
    def _find_best_match_in_history(self, u_lm, u_angs):
        """
        在参考历史中找到与用户当前帧最相似的一帧
        
        Returns:
            (best_r_lm, best_r_angs) 或 (None, None)
        """
        if len(self.ref_history) < 10:
            return None, None
        
        # 构建用户特征向量（8个关键角度）
        u_vec = np.array([
            u_angs.get("leftShoulder", 0),
            u_angs.get("rightShoulder", 0),
            u_angs.get("leftElbow", 0),
            u_angs.get("rightElbow", 0),
            u_angs.get("leftHip", 0),
            u_angs.get("rightHip", 0),
            u_angs.get("leftKnee", 0),
            u_angs.get("rightKnee", 0)
        ], dtype=np.float32)
        
        # 在历史中找到欧氏距离最小的帧
        best_idx = None
        best_dist = float('inf')
        
        for i, (_, r_angs) in enumerate(self.ref_history):
            r_vec = np.array([
                r_angs.get("leftShoulder", 0),
                r_angs.get("rightShoulder", 0),
                r_angs.get("leftElbow", 0),
                r_angs.get("rightElbow", 0),
                r_angs.get("leftHip", 0),
                r_angs.get("rightHip", 0),
                r_angs.get("leftKnee", 0),
                r_angs.get("rightKnee", 0)
            ], dtype=np.float32)
            
            dist = np.linalg.norm(u_vec - r_vec)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx is not None:
            best_r_lm, best_r_angs = self.ref_history[best_idx]
            return best_r_lm, best_r_angs
        
        return None, None

    def check_timing(self, current_u_lms):
        """
        节奏分析功能已禁用
        """
        return ""

    def get_analysis_data(self):
        """
        获取当前分析数据，供AI教练使用
        Returns:
            dict: 包含当前分析数据的字典，如果无可用数据返回None
        """
        if not hasattr(self, '_last_analysis_data'):
            return None

        return self._last_analysis_data

    def set_analysis_data(self, u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint):
        """
        设置当前分析数据
        """
        self._last_analysis_data = {
            'user_landmarks': u_lms,
            'ref_landmarks': r_lms,
            'diffs': diffs,
            'real_time_percent': real_time_percent,
            'dtw_percent': dtw_percent,
            'timing_hint': timing_hint
        }
