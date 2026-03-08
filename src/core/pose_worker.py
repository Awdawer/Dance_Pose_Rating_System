import time
import cv2
import numpy as np
from queue import Queue, Empty
from PyQt5 import QtCore

from src.utils.geometry import compute_angles
from src.core.scoring import score_angles, _select_points_with_weights, _w_procrustes_dist, _dist_to_score

class PoseWorker(QtCore.QObject):
    """
    Worker class for heavy MediaPipe processing.
    Runs in a separate thread to keep the GUI responsive.
    """
    results_ready = QtCore.pyqtSignal(list, list, dict, float, str) # u_lms, r_lms, diffs, percent, timing_hint

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
        
        # Timing analysis buffer
        self.ref_history = [] # List of (timestamp, landmarks)
        self.user_history = [] # List of (timestamp, landmarks)
        self.HISTORY_LEN = 30 # Store ~1 second of poses

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

                percent = 0.0
                diffs = {}
                timing_hint = ""
                
                if len(u_lms) and len(r_lms):
                    percent, diffs = self.compute_and_score(u_lms, r_lms)
                    
                    # Update histories for timing check
                    now = time.time()
                    self.user_history.append((now, u_lms))
                    self.ref_history.append((now, r_lms))
                    
                    if len(self.user_history) > self.HISTORY_LEN: self.user_history.pop(0)
                    if len(self.ref_history) > self.HISTORY_LEN: self.ref_history.pop(0)
                    
                    timing_hint = self.check_timing(u_lms)

                self.results_ready.emit(u_lms, r_lms, diffs, percent, timing_hint)
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
            out.append({"x": fx, "y": fy})

        # Update ROI for next frame
        xs = [p["x"] for p in out]
        ys = [p["y"] for p in out]
        minx = max(0, int(min(xs) * w) - int(0.1 * w))
        maxx = min(w, int(max(xs) * w) + int(0.1 * w))
        miny = max(0, int(min(ys) * h) - int(0.1 * h))
        maxy = min(h, int(max(ys) * h) + int(0.1 * h))
        return out, (minx, miny, max(64, maxx - minx), max(64, maxy - miny))

    def compute_and_score(self, u_lm, r_lm):
        # 1. Angle-based scoring
        au = compute_angles(u_lm)
        ar = compute_angles(r_lm)
        
        # Calculate angle score percent
        angle_percent, diffs = score_angles(au, ar)
        
        # 2. Procrustes-based scoring (positional)
        pu, wu = _select_points_with_weights(u_lm)
        pr, wr = _select_points_with_weights(r_lm)
        d = _w_procrustes_dist(pu, pr, np.minimum(wu, wr))
        procrustes_percent = _dist_to_score(d, k=self.k_decay, gamma=self.gamma)
        
        # 3. Combine scores
        # Procrustes is good for overall shape, Angles are good for limb correctness.
        # Let's weight them. If pose is totally wrong, both should be low.
        # If we just average, a 0 in one and 100 in other gives 50.
        # But if angles are wrong, pose is wrong.
        # Let's use a weighted average but maybe bias towards the lower one?
        # Simple weighted average for now: 40% Procrustes, 60% Angles (angles are stricter)
        final_percent = 0.4 * procrustes_percent + 0.6 * angle_percent
        
        return final_percent, diffs

    def check_timing(self, current_u_lms):
        # We need at least some history to compare
        if len(self.ref_history) < 10: return ""
        
        # Compare current user pose against past reference poses
        best_score = -1
        best_idx = -1
        
        # Check last 15 frames of reference
        search_range = self.ref_history[-15:] 
        current_ref_idx = len(search_range) - 1 # The latest frame is "Now"
        
        for i, (ts, r_lms) in enumerate(search_range):
            au = compute_angles(current_u_lms)
            ar = compute_angles(r_lms)
            score = 0
            for k in au:
                score += abs(au[k] - ar.get(k, 0))
            
            if best_score == -1 or score < best_score:
                best_score = score
                best_idx = i
                
        offset = current_ref_idx - best_idx
        
        if offset > 4: # ~130ms behind
            return "快点! (太慢)"
        elif offset < -2:
            pass
            
        # To detect "Too Fast", we need to see if User's PAST pose matches Ref's CURRENT pose.
        if len(self.user_history) >= 10:
            curr_r_lms = self.ref_history[-1][1]
            best_u_score = -1
            best_u_idx = -1
            u_search = self.user_history[-15:]
            curr_u_idx = len(u_search) - 1
            
            au_curr_ref = compute_angles(curr_r_lms)
            
            for i, (ts, u_lms) in enumerate(u_search):
                au = compute_angles(u_lms)
                score = sum(abs(au[k] - au_curr_ref.get(k, 0)) for k in au)
                if best_u_score == -1 or score < best_u_score:
                    best_u_score = score
                    best_u_idx = i
            
            u_offset = curr_u_idx - best_u_idx
            if u_offset > 4:
                return "慢点! (太快)"

        return "Perfect" if abs(offset) <= 2 else ""
