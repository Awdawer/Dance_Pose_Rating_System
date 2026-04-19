import os
import cv2
import time
import threading
from PyQt5 import QtCore, QtGui, QtWidgets

from src.utils.model_loader import ensure_model
from src.core.pose_worker import PoseWorker
from src.core.video_reader import VideoReader
from src.core.ghost_overlay import align_skeleton_to_user, draw_comparison_skeleton
from src.ui.components import VideoPanel, ScoreChartWidget


class GhostModeWindow(QtWidgets.QWidget):
    """
    幽灵图叠加模式窗口。
    
    功能特点：
    - 将参考视频的骨架实时叠加到用户视频上
    - 使用Procrustes分析自动适配不同体型
    - 提供透明度调节和镜像模式
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ghost Overlay Mode - 幽灵图叠加模式")
        self.resize(1400, 900)
        
        # 应用暗黑主题
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 16px;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 12px 24px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 16px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00C6FF;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QPushButton#backButton {
                background-color: #4B5563;
                border-color: #6B7280;
            }
            QPushButton#backButton:hover {
                background-color: #6B7280;
                border-color: #9CA3AF;
            }
            QComboBox {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                min-height: 40px;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 16px;
            }
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #333333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #10B981;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #10B981;
                border-radius: 4px;
            }
            QCheckBox {
                font-size: 16px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #555555;
                background-color: #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #10B981;
                border-color: #10B981;
            }
        """)
        
        # 视频读取器
        self.userReader = None
        self.refReader = None
        self.useCam = False
        self.playing = False
        
        # 幽灵图设置
        self.ghost_alpha = 0.4  # 透明度
        self.mirror_mode = False  # 镜像模式
        self.show_user_skeleton = True  # 显示用户骨架
        
        # 最新检测结果
        self.lastUserLandmarks = []
        self.lastRefLandmarks = []
        self.lastDiffs = None
        self.alignedRefPts = None  # 对齐后的参考骨架点
        
        # 计数器
        self._tick_count = 0
        self.detect_stride_user = 2
        self.detect_stride_ref = 4
        
        # 工作线程（延迟初始化）
        self.worker_thread = None
        self.worker = None
        
        self._setup_ui()
        
        # 定时器
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()
        
        # 延迟初始化工作线程和摄像头
        QtCore.QTimer.singleShot(100, self._init_worker)
        QtCore.QTimer.singleShot(200, self.enumerate_cams)
    
    def _init_worker(self):
        """延迟初始化工作线程"""
        try:
            model_path = ensure_model()
            if not model_path or not os.path.exists(model_path):
                QtWidgets.QMessageBox.critical(self, "Error", "Model file missing")
                return
            
            self.worker_thread = QtCore.QThread(self)
            self.worker = PoseWorker(model_path)
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.start)
            self.worker.results_ready.connect(self.on_results_ready)
            self.worker_thread.start()
        except Exception as e:
            print(f"[GhostMode] Failed to init worker: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to initialize pose detector: {e}")
    
    def _setup_ui(self):
        """设置UI布局"""
        # 主视频面板（显示叠加效果）
        self.mainPanel = VideoPanel()
        self.mainPanel.setMinimumSize(960, 540)
        
        # 参考视频小面板（可选显示）
        self.refPanel = VideoPanel()
        self.refPanel.setMinimumSize(320, 180)
        self.refPanel.setMaximumSize(480, 270)
        
        # 返回按钮
        self.btnBack = QtWidgets.QPushButton("← Back to Main")
        self.btnBack.setObjectName("backButton")
        self.btnBack.setMinimumWidth(150)
        self.btnBack.clicked.connect(self.go_back)
        
        # 视频加载按钮
        self.btnLoadUser = QtWidgets.QPushButton("Load User Video")
        self.btnLoadRef = QtWidgets.QPushButton("Load Reference Video")
        self.btnCamToggle = QtWidgets.QPushButton("Start Camera")
        self.camCombo = QtWidgets.QComboBox()
        
        # 播放控制
        self.btnPlayReset = QtWidgets.QPushButton("Start")
        self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;")
        self.btnPauseResume = QtWidgets.QPushButton("Pause")
        self.btnPauseResume.setStyleSheet("background-color: #F59E0B; font-size: 18px;")
        
        # 幽灵图控制
        ghostControlGroup = QtWidgets.QGroupBox("Ghost Overlay Settings")
        ghostControlGroup.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # 透明度滑块
        alphaLayout = QtWidgets.QHBoxLayout()
        alphaLayout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.alphaSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alphaSlider.setRange(10, 100)
        self.alphaSlider.setValue(40)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)
        alphaLayout.addWidget(self.alphaSlider)
        self.alphaLabel = QtWidgets.QLabel("40%")
        alphaLayout.addWidget(self.alphaLabel)
        
        # 镜像模式复选框
        self.chkMirror = QtWidgets.QCheckBox("Mirror Reference")
        self.chkMirror.stateChanged.connect(self.on_mirror_changed)
        
        # 显示用户骨架复选框
        self.chkShowUser = QtWidgets.QCheckBox("Show User Skeleton")
        self.chkShowUser.setChecked(True)
        self.chkShowUser.stateChanged.connect(self.on_show_user_changed)
        
        ghostControlLayout = QtWidgets.QVBoxLayout()
        ghostControlLayout.addLayout(alphaLayout)
        ghostControlLayout.addWidget(self.chkMirror)
        ghostControlLayout.addWidget(self.chkShowUser)
        ghostControlGroup.setLayout(ghostControlLayout)
        
        # 评分显示
        self.scoreLabel = QtWidgets.QLabel("Score: -- %")
        self.scoreLabel.setFont(QtGui.QFont("Segoe UI", 24, QtGui.QFont.Bold))
        self.scoreLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.scoreLabel.setStyleSheet("""
            color: #10B981;
            background-color: #000000;
            border-radius: 15px;
            padding: 15px;
        """)
        
        # 布局
        # 顶部栏
        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(self.btnBack)
        topLayout.addStretch(1)
        topLayout.addWidget(self.scoreLabel)
        topLayout.addStretch(1)
        
        # 视频区域
        videoLayout = QtWidgets.QHBoxLayout()
        videoLayout.addWidget(self.mainPanel, 3)
        
        # 右侧控制面板
        rightPanel = QtWidgets.QVBoxLayout()
        rightPanel.addWidget(QtWidgets.QLabel("Reference Preview:"))
        rightPanel.addWidget(self.refPanel)
        rightPanel.addSpacing(20)
        rightPanel.addWidget(ghostControlGroup)
        rightPanel.addStretch(1)
        
        videoLayout.addLayout(rightPanel, 1)
        
        # 控制按钮区域
        sourceLayout = QtWidgets.QHBoxLayout()
        sourceLayout.addWidget(self.btnLoadUser)
        sourceLayout.addWidget(self.btnLoadRef)
        sourceLayout.addWidget(self.camCombo)
        sourceLayout.addWidget(self.btnCamToggle)
        sourceLayout.addStretch(1)
        
        playLayout = QtWidgets.QHBoxLayout()
        playLayout.addStretch(1)
        self.btnPlayReset.setMinimumWidth(120)
        self.btnPauseResume.setMinimumWidth(120)
        playLayout.addWidget(self.btnPlayReset)
        playLayout.addWidget(self.btnPauseResume)
        playLayout.addStretch(1)
        
        ctrlLayout = QtWidgets.QVBoxLayout()
        ctrlLayout.addLayout(sourceLayout)
        ctrlLayout.addLayout(playLayout)
        
        # 主布局
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        layout.addLayout(topLayout)
        layout.addLayout(videoLayout, 5)
        layout.addLayout(ctrlLayout)
        
        # 信号连接
        self.btnLoadUser.clicked.connect(self.load_user_video)
        self.btnLoadRef.clicked.connect(self.load_ref_video)
        self.btnCamToggle.clicked.connect(self.toggle_cam)
        self.btnPlayReset.clicked.connect(self.play_reset)
        self.btnPauseResume.clicked.connect(self.pause_resume)
    
    def on_alpha_changed(self, value):
        """透明度滑块变化"""
        self.ghost_alpha = value / 100.0
        self.alphaLabel.setText(f"{value}%")
    
    def on_mirror_changed(self, state):
        """镜像模式切换"""
        self.mirror_mode = (state == QtCore.Qt.Checked)
    
    def on_show_user_changed(self, state):
        """显示用户骨架切换"""
        self.show_user_skeleton = (state == QtCore.Qt.Checked)
    
    def go_back(self):
        """返回主窗口"""
        print("[GhostMode] go_back called")
        self.pause()
        # 不直接关闭窗口，让主窗口的回调来处理
    
    def enumerate_cams(self):
        """枚举摄像头"""
        self.camCombo.clear()
        found = False
        for i in range(0, 8):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            if ok:
                ret, frame = cap.read()
                if ret:
                    found = True
                    self.camCombo.addItem(f"Camera {i}", i)
            if cap:
                cap.release()
        if not found:
            self.camCombo.addItem("No Camera Found", None)
    
    def load_user_video(self):
        """加载用户视频"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select User Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        self.userReader = VideoReader(path)
        self.userReader.start()
        self.userReader.pause()
        self.playing = False
        self.show_video_preview(path, self.mainPanel, "User")
    
    def load_ref_video(self):
        """加载参考视频"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Reference Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        if self.refReader:
            self.refReader.stop()
        self.refReader = VideoReader(path)
        self.refReader.start()
        self.refReader.pause()
        self.playing = False
        self.show_video_preview(path, self.refPanel, "Reference")
    
    def toggle_cam(self):
        """切换摄像头"""
        if self.useCam:
            self.stop_cam()
            self.btnCamToggle.setText("Start Camera")
            self.btnCamToggle.setStyleSheet("")
        else:
            self.start_cam()
            self.btnCamToggle.setText("Stop Camera")
            self.btnCamToggle.setStyleSheet("background-color: #EF4444; border-color: #EF4444;")
    
    def start_cam(self):
        """启动摄像头"""
        self.useCam = True
        if self.userReader:
            self.userReader.stop()
        idx = self.camCombo.currentData()
        if idx is None:
            idx = 0
        self.userReader = VideoReader(int(idx))
        self.userReader.start()
        self.playing = True
    
    def stop_cam(self):
        """停止摄像头"""
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        self.userReader = None
        self.playing = False
    
    def play_reset(self):
        """播放/重置"""
        if self.playing:
            self.pause()
            self.btnPlayReset.setText("Start")
            self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;")
        else:
            # 重置视频
            if self.userReader:
                self.userReader.reset()
                self.userReader.resume()
            if self.refReader:
                self.refReader.reset()
                self.refReader.resume()
            self.playing = True
            self.btnPlayReset.setText("Stop")
            self.btnPlayReset.setStyleSheet("background-color: #EF4444; font-size: 18px;")
            self.btnPauseResume.setText("Pause")
    
    def pause_resume(self):
        """暂停/恢复"""
        if self.playing:
            self.pause()
            self.btnPauseResume.setText("Resume")
        else:
            self.resume()
            self.btnPauseResume.setText("Pause")
    
    def pause(self):
        """暂停"""
        self.playing = False
        if self.userReader:
            self.userReader.pause()
        if self.refReader:
            self.refReader.pause()
    
    def resume(self):
        """恢复"""
        self.playing = True
        if self.userReader:
            self.userReader.resume()
        if self.refReader:
            self.refReader.resume()
    
    def on_tick(self):
        """定时器回调"""
        if self.playing:
            self.read_and_process()
    
    def on_results_ready(self, u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint):
        """处理姿态检测结果"""
        if len(u_lms):
            self.lastUserLandmarks = u_lms
        if len(r_lms):
            self.lastRefLandmarks = r_lms
        if diffs:
            self.lastDiffs = diffs
        
        # 计算对齐的参考骨架
        if self.lastUserLandmarks and self.lastRefLandmarks:
            # 如果需要镜像参考骨架
            ref_lms = self.lastRefLandmarks
            if self.mirror_mode:
                ref_lms = self._mirror_landmarks(ref_lms)
            
            aligned_pts, _ = align_skeleton_to_user(self.lastUserLandmarks, ref_lms)
            if aligned_pts:
                self.alignedRefPts = aligned_pts
        
        # 更新评分显示
        if real_time_percent > 0:
            self.scoreLabel.setText(f"Score: {int(round(real_time_percent))} %")
            # 根据评分改变颜色
            if real_time_percent >= 80:
                self.scoreLabel.setStyleSheet("""
                    color: #10B981;
                    background-color: #000000;
                    border-radius: 15px;
                    padding: 15px;
                """)
            elif real_time_percent >= 60:
                self.scoreLabel.setStyleSheet("""
                    color: #F59E0B;
                    background-color: #000000;
                    border-radius: 15px;
                    padding: 15px;
                """)
            else:
                self.scoreLabel.setStyleSheet("""
                    color: #EF4444;
                    background-color: #000000;
                    border-radius: 15px;
                    padding: 15px;
                """)
    
    def _mirror_landmarks(self, landmarks):
        """镜像关键点（左右互换）"""
        mirrored = []
        for p in landmarks:
            mirrored.append({
                "x": 1.0 - p["x"],
                "y": p["y"],
                "z": p.get("z", 0),
                "v": p.get("v", 1.0)
            })
        
        # 交换左右关键点索引
        swap_pairs = [
            (11, 12),  # 肩
            (13, 14),  # 肘
            (15, 16),  # 手腕
            (23, 24),  # 胯
            (25, 26),  # 膝
            (27, 28),  # 踝
            (29, 30),  # 脚跟
            (31, 32),  # 脚尖
        ]
        
        result = mirrored.copy()
        for left, right in swap_pairs:
            if left < len(result) and right < len(result):
                result[left], result[right] = result[right], result[left]
        
        return result
    
    def read_and_process(self):
        """读取视频帧并处理"""
        user_frame = None
        ref_frame = None
        
        if self.userReader:
            ok, frame = self.userReader.get_frame()
            if ok:
                user_frame = frame
        
        if self.refReader:
            ok, frame = self.refReader.get_frame()
            if ok:
                ref_frame = frame
        
        # 提交检测任务
        self._tick_count += 1
        need_detect_user = (user_frame is not None) and (
            self._tick_count % (self.detect_stride_user if self.useCam else 3) == 0
            or not self.lastUserLandmarks
        )
        need_detect_ref = (ref_frame is not None) and (
            self._tick_count % self.detect_stride_ref == 0
            or not self.lastRefLandmarks
        )
        
        if (need_detect_user or need_detect_ref) and self.worker:
            payload = (
                user_frame if need_detect_user else None,
                ref_frame if need_detect_ref else None,
                5.0,  # k_decay
                0.7   # gamma
            )
            try:
                self.worker.queue.put_nowait(payload)
            except Exception:
                pass
        
        # 绘制主面板（叠加效果）
        if user_frame is not None:
            self.draw_ghost_overlay(self.mainPanel, user_frame)
        
        # 绘制参考面板
        if ref_frame is not None:
            self.draw_frame_on_panel(self.refPanel, ref_frame)
    
    def draw_ghost_overlay(self, panel, user_frame):
        """绘制幽灵图叠加效果"""
        # 调整大小
        MAX_DISP_W = 1280
        h_orig, w_orig = user_frame.shape[:2]
        if w_orig > MAX_DISP_W:
            scale = MAX_DISP_W / float(w_orig)
            canvas = cv2.resize(user_frame, (MAX_DISP_W, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            canvas = user_frame.copy()
        
        # 绘制叠加效果
        if self.alignedRefPts:
            # 准备用户骨架（如果需要显示）
            user_lms = self.lastUserLandmarks if self.show_user_skeleton else None
            
            # 使用自定义绘制函数
            canvas = self._draw_ghost_on_canvas(canvas, user_lms, self.alignedRefPts, self.lastDiffs)
        elif self.lastUserLandmarks:
            # 只绘制用户骨架
            canvas = self._draw_user_only(canvas, self.lastUserLandmarks)
        
        # 显示
        qimg = self.frame_to_qimage(canvas)
        panel_w = panel.width()
        panel_h = panel.height()
        if panel_w > 10 and panel_h > 10:
            scaled_pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                panel_w, panel_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            panel.setPixmap(scaled_pixmap)
        else:
            panel.setPixmap(QtGui.QPixmap.fromImage(qimg))
    
    def _draw_ghost_on_canvas(self, canvas, user_lms, aligned_ref_pts, diffs):
        """在画布上绘制幽灵图叠加效果"""
        from src.utils.geometry import POSE_CONNECTIONS
        
        h, w = canvas.shape[:2]
        
        # 绘制参考骨架（幽灵图）
        if aligned_ref_pts:
            overlay = canvas.copy()
            ghost_color = (0, 255, 0)  # 绿色
            
            for a, b in POSE_CONNECTIONS:
                if a < len(aligned_ref_pts) and b < len(aligned_ref_pts):
                    pt1 = (int(aligned_ref_pts[a][0] * w), int(aligned_ref_pts[a][1] * h))
                    pt2 = (int(aligned_ref_pts[b][0] * w), int(aligned_ref_pts[b][1] * h))
                    cv2.line(overlay, pt1, pt2, ghost_color, 5, cv2.LINE_AA)
            
            # 绘制参考骨架关键点
            body_idxs = list(range(11, 33))
            for i in body_idxs:
                if i < len(aligned_ref_pts):
                    x = int(aligned_ref_pts[i][0] * w)
                    y = int(aligned_ref_pts[i][1] * h)
                    cv2.circle(overlay, (x, y), 7, ghost_color, -1, cv2.LINE_AA)
            
            # 混合
            alpha = self.ghost_alpha
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        
        # 绘制用户骨架
        if user_lms and self.show_user_skeleton:
            user_pts = [(p["x"], p["y"]) for p in user_lms]
            user_color = (255, 128, 0)  # 蓝色
            
            for a, b in POSE_CONNECTIONS:
                if a < len(user_pts) and b < len(user_pts):
                    pt1 = (int(user_pts[a][0] * w), int(user_pts[a][1] * h))
                    pt2 = (int(user_pts[b][0] * w), int(user_pts[b][1] * h))
                    cv2.line(canvas, pt1, pt2, user_color, 2, cv2.LINE_AA)
            
            body_idxs = list(range(11, 33))
            for i in body_idxs:
                if i < len(user_pts):
                    x = int(user_pts[i][0] * w)
                    y = int(user_pts[i][1] * h)
                    cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            
            # 显示差异
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
                            color = (0, 0, 255)
                        elif d >= 20:
                            color = (0, 165, 255)
                        else:
                            color = (0, 255, 0)
                        
                        x = int(user_pts[i][0] * w)
                        y = int(user_pts[i][1] * h)
                        cv2.putText(canvas, f"{int(round(d))}", (x + 8, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        return canvas
    
    def _draw_user_only(self, canvas, user_lms):
        """只绘制用户骨架"""
        from src.utils.geometry import POSE_CONNECTIONS
        
        h, w = canvas.shape[:2]
        user_pts = [(p["x"], p["y"]) for p in user_lms]
        user_color = (255, 128, 0)
        
        for a, b in POSE_CONNECTIONS:
            if a < len(user_pts) and b < len(user_pts):
                pt1 = (int(user_pts[a][0] * w), int(user_pts[a][1] * h))
                pt2 = (int(user_pts[b][0] * w), int(user_pts[b][1] * h))
                cv2.line(canvas, pt1, pt2, user_color, 2, cv2.LINE_AA)
        
        body_idxs = list(range(11, 33))
        for i in body_idxs:
            if i < len(user_pts):
                x = int(user_pts[i][0] * w)
                y = int(user_pts[i][1] * h)
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        
        return canvas
    
    def draw_frame_on_panel(self, panel, frame_bgr):
        """在面板上显示普通帧"""
        MAX_DISP_W = 640
        h_orig, w_orig = frame_bgr.shape[:2]
        if w_orig > MAX_DISP_W:
            scale = MAX_DISP_W / float(w_orig)
            canvas = cv2.resize(frame_bgr, (MAX_DISP_W, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            canvas = frame_bgr.copy()
        
        qimg = self.frame_to_qimage(canvas)
        panel_w = panel.width()
        panel_h = panel.height()
        if panel_w > 10 and panel_h > 10:
            scaled_pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                panel_w, panel_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            panel.setPixmap(scaled_pixmap)
        else:
            panel.setPixmap(QtGui.QPixmap.fromImage(qimg))
    
    def frame_to_qimage(self, frame_bgr):
        """将OpenCV帧转换为QImage"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
    
    def show_video_preview(self, video_path, panel, label):
        """显示视频预览"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.draw_frame_on_panel(panel, frame)
            cap.release()
        except Exception as e:
            print(f"[Preview] Error loading {label} video: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        print("[GhostMode] Closing window...")
        try:
            # 停止定时器
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
            
            # 停止工作线程
            if self.worker:
                self.worker.stop()
            if self.worker_thread:
                self.worker_thread.quit()
                # 等待线程结束，但设置超时避免阻塞
                if not self.worker_thread.wait(500):  # 等待500ms
                    print("[GhostMode] Worker thread did not finish in time, terminating...")
                    self.worker_thread.terminate()
                    self.worker_thread.wait(100)
            
            # 停止视频读取器
            if self.userReader:
                self.userReader.stop()
            if self.refReader:
                self.refReader.stop()
        except Exception as e:
            print(f"[GhostMode] Error during close: {e}")
        
        event.accept()
        print("[GhostMode] Window closed")
