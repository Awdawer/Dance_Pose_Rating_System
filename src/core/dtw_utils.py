import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _standard_dtw(x, y):
    """
    标准DTW实现（用于小序列）
    时间复杂度: O(n*m)
    """
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


@jit(nopython=True, cache=True)
def _dtw_sakoe_chiba(x, y, window_size=5):
    """
    带Sakoe-Chiba约束的DTW
    限制搜索路径在对角线附近的带状区域，显著减少计算量
    
    Args:
        x: 序列1，shape (n, features)
        y: 序列2，shape (m, features)
        window_size: 约束窗口大小
    
    Returns:
        DTW距离
    """
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        start_j = max(1, i - window_size)
        end_j = min(m + 1, i + window_size + 1)
        
        for j in range(start_j, end_j):
            cost = np.linalg.norm(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


@jit(nopython=True, cache=True)
def _dtw_path_sakoe_chiba(x, y, window_size=5):
    """
    带Sakoe-Chiba约束的DTW，返回对齐路径
    
    Returns:
        path: 对齐路径列表，每个元素为 (i, j)
    """
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # 填充DTW矩阵
    for i in range(1, n + 1):
        start_j = max(1, i - window_size)
        end_j = min(m + 1, i + window_size + 1)
        
        for j in range(start_j, end_j):
            cost = np.linalg.norm(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # 回溯路径
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        v_diag = dtw[i-1, j-1]
        v_up = dtw[i-1, j]
        v_left = dtw[i, j-1]
        
        best = min(v_diag, v_up, v_left)
        
        if best == v_diag:
            i -= 1
            j -= 1
        elif best == v_up:
            i -= 1
        else:
            j -= 1
    
    return path


class TimingAnalyzer:
    """
    专门用于舞蹈节奏分析的类
    提供平滑处理、置信度计算和人性化提示
    """
    
    def __init__(self, history_len=60, smoothing_window=5, lag_threshold=3.0):
        """
        Args:
            history_len: 历史缓冲区长度
            smoothing_window: 平滑处理窗口大小
            lag_threshold: 判定快慢的延迟阈值（帧数）
        """
        self.history_len = history_len
        self.smoothing_window = smoothing_window
        self.lag_threshold = lag_threshold
        self.lag_history = []
        self.confidence_history = []
        
    def analyze_timing(self, user_seq, ref_seq, window_size=5):
        """
        分析用户与参考序列的时间对齐情况
        
        Args:
            user_seq: 用户动作序列，list of numpy arrays
            ref_seq: 参考动作序列，list of numpy arrays
            window_size: DTW约束窗口大小
        
        Returns:
            timing_hint: 节奏提示文本
            confidence: 置信度 (0-1)
            avg_lag: 平均延迟（帧数）
        """
        if len(user_seq) < 10 or len(ref_seq) < 10:
            return "Analyzing...", 0.0, 0.0
        
        # 转换为numpy数组
        user_array = np.array(user_seq, dtype=np.float32)
        ref_array = np.array(ref_seq, dtype=np.float32)
        
        # 使用Sakoe-Chiba约束DTW计算对齐路径
        path = _dtw_path_sakoe_chiba(user_array, ref_array, window_size)
        
        # 计算每帧的延迟
        lags = [j - i for i, j in path]
        
        # 平滑处理：移动平均
        if len(lags) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed_lags = np.convolve(lags, kernel, mode='valid')
            avg_lag = float(np.mean(smoothed_lags))
        else:
            avg_lag = float(np.mean(lags))
        
        # 存储历史用于趋势分析
        self.lag_history.append(avg_lag)
        if len(self.lag_history) > 10:
            self.lag_history.pop(0)
        
        # 计算置信度（基于历史稳定性）
        if len(self.lag_history) >= 3:
            lag_std = np.std(self.lag_history)
            confidence = max(0.0, 1.0 - min(1.0, lag_std / self.lag_threshold))
        else:
            confidence = 0.5
        
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        # 使用平均置信度提高稳定性
        avg_confidence = float(np.mean(self.confidence_history)) if self.confidence_history else confidence
        
        # 生成提示
        timing_hint = self._generate_hint(avg_lag, avg_confidence)
        
        return timing_hint, avg_confidence, avg_lag
    
    def _generate_hint(self, lag, confidence):
        """
        根据延迟生成人性化的提示
        
        Args:
            lag: 平均延迟（正数表示落后，负数表示超前）
            confidence: 置信度
        
        Returns:
            hint: 提示文本
        """
        # 置信度低时不显示具体提示
        if confidence < 0.4:
            return ""
        
        threshold = self.lag_threshold
        
        if lag > threshold + 2:
            return "Speed up! You're behind"
        elif lag > threshold:
            return "Slightly behind"
        elif lag < -(threshold + 2):
            return "Slow down! You're ahead"
        elif lag < -threshold:
            return "Slightly ahead"
        else:
            return "Perfect"
    
    def reset(self):
        """重置分析器状态"""
        self.lag_history.clear()
        self.confidence_history.clear()


def compute_weighted_features(landmarks_list, weights=None):
    """
    从关键点列表提取加权特征向量
    
    Args:
        landmarks_list: 关键点列表，每个元素为包含关节角度的字典
        weights: 各关节的权重字典，默认使用预设权重
    
    Returns:
        features: 特征向量列表
    """
    if weights is None:
        weights = {
            "leftShoulder": 1.2,
            "rightShoulder": 1.2,
            "leftElbow": 1.5,
            "rightElbow": 1.5,
            "leftHip": 1.2,
            "rightHip": 1.2,
            "leftKnee": 1.5,
            "rightKnee": 1.5
        }
    
    features = []
    for angs in landmarks_list:
        vec = [
            angs.get("leftShoulder", 0) * weights["leftShoulder"],
            angs.get("rightShoulder", 0) * weights["rightShoulder"],
            angs.get("leftElbow", 0) * weights["leftElbow"],
            angs.get("rightElbow", 0) * weights["rightElbow"],
            angs.get("leftHip", 0) * weights["leftHip"],
            angs.get("rightHip", 0) * weights["rightHip"],
            angs.get("leftKnee", 0) * weights["leftKnee"],
            angs.get("rightKnee", 0) * weights["rightKnee"]
        ]
        features.append(np.array(vec, dtype=np.float32))
    
    return features


class DTWAlignmentScorer:
    """
    使用DTW对齐的评分器
    找到参考视频中最相似的帧进行对比评分
    """
    
    def __init__(self, window_size=15, min_window=3):
        """
        Args:
            window_size: DTW对齐窗口大小
            min_window: 最小历史长度要求（降低为3帧）
        """
        self.window_size = window_size
        self.min_window = min_window
        self.user_history = []  # List of (landmarks, angles)
        self.ref_history = []   # List of (landmarks, angles)
        self.last_aligned_idx = None  # 上次对齐的参考帧索引
        
    def update_histories(self, user_lm, ref_lm, user_angs, ref_angs):
        """更新历史缓冲区"""
        self.user_history.append((user_lm, user_angs))
        self.ref_history.append((ref_lm, ref_angs))
        
        # 保持窗口大小
        if len(self.user_history) > self.window_size:
            self.user_history.pop(0)
        if len(self.ref_history) > self.window_size:
            self.ref_history.pop(0)
    
    def find_best_match(self, current_user_idx=None):
        """
        使用DTW找到当前用户帧对应的最佳参考帧
        
        Returns:
            best_ref_lm: 最佳匹配的参考关键点
            best_ref_angs: 最佳匹配的参考角度
            alignment_info: 对齐信息字典
        """
        if len(self.user_history) < self.min_window or len(self.ref_history) < self.min_window:
            return None, None, {"aligned": False, "reason": "insufficient_history"}
        
        # 提取特征序列用于DTW
        user_features = []
        for _, angs in self.user_history:
            vec = [
                angs.get("leftShoulder", 0),
                angs.get("rightShoulder", 0),
                angs.get("leftElbow", 0),
                angs.get("rightElbow", 0),
                angs.get("leftHip", 0),
                angs.get("rightHip", 0),
                angs.get("leftKnee", 0),
                angs.get("rightKnee", 0)
            ]
            user_features.append(np.array(vec, dtype=np.float32))
        
        ref_features = []
        for _, angs in self.ref_history:
            vec = [
                angs.get("leftShoulder", 0),
                angs.get("rightShoulder", 0),
                angs.get("leftElbow", 0),
                angs.get("rightElbow", 0),
                angs.get("leftHip", 0),
                angs.get("rightHip", 0),
                angs.get("leftKnee", 0),
                angs.get("rightKnee", 0)
            ]
            ref_features.append(np.array(vec, dtype=np.float32))
        
        # 使用DTW找到对齐路径
        # 窗口大小设为历史长度的一半，确保能找到较大的延迟
        dtw_window = max(10, len(user_features) // 2)
        path = _dtw_path_sakoe_chiba(user_features, ref_features, window_size=dtw_window)
        
        if not path:
            return None, None, {"aligned": False, "reason": "dtw_failed"}
        
        # 找到当前最新用户帧对应的最佳参考帧
        # path是倒序的，从后往前找
        current_user_pos = len(self.user_history) - 1
        best_ref_idx = None
        
        for user_idx, ref_idx in path:
            if user_idx == current_user_pos:
                best_ref_idx = ref_idx
                break
        
        # 如果没找到精确匹配，使用路径中最后一个点
        if best_ref_idx is None and path:
            best_ref_idx = path[0][1]
        
        # 平滑处理：限制参考帧跳变幅度（但允许更大的跳变以适应节奏变化）
        if self.last_aligned_idx is not None:
            max_jump = 8  # 增加最大允许跳变到8帧（约0.25秒）
            if abs(best_ref_idx - self.last_aligned_idx) > max_jump:
                # 限制在合理范围内
                if best_ref_idx > self.last_aligned_idx:
                    best_ref_idx = min(best_ref_idx, self.last_aligned_idx + max_jump)
                else:
                    best_ref_idx = max(best_ref_idx, self.last_aligned_idx - max_jump)
        
        self.last_aligned_idx = best_ref_idx
        
        # 获取最佳匹配的参考帧数据
        best_ref_lm, best_ref_angs = self.ref_history[best_ref_idx]
        
        # 计算对齐质量（路径的平均距离）
        total_distance = 0.0
        for user_idx, ref_idx in path:
            dist = np.linalg.norm(user_features[user_idx] - ref_features[ref_idx])
            total_distance += dist
        avg_distance = total_distance / len(path) if path else 0.0
        
        alignment_info = {
            "aligned": True,
            "user_idx": current_user_pos,
            "ref_idx": best_ref_idx,
            "path_length": len(path),
            "lag": best_ref_idx - current_user_pos,  # 正数表示用户落后
            "avg_distance": avg_distance,
            "history_len": len(self.user_history)
        }
        
        # 调试输出
        if alignment_info["lag"] != 0:
            print(f"DTW: User frame {current_user_pos} -> Ref frame {best_ref_idx} (lag: {alignment_info['lag']}, avg_dist: {avg_distance:.2f})")
        
        return best_ref_lm, best_ref_angs, alignment_info
    
    def reset(self):
        """重置评分器状态"""
        self.user_history.clear()
        self.ref_history.clear()
        self.last_aligned_idx = 0
    
    def get_alignment_hint(self, alignment_info):
        """根据对齐信息生成提示"""
        if not alignment_info.get("aligned", False):
            return ""
        
        lag = alignment_info.get("lag", 0)
        
        if lag > 2:
            return "↓"  # 用户落后，需要加速
        elif lag < -2:
            return "↑"  # 用户超前，需要减速
        else:
            return "●"  # 同步良好
