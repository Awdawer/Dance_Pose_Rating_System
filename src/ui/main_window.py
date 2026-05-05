import os
import cv2
import time
import threading
from PyQt5 import QtCore, QtGui, QtWidgets

from src.utils.model_loader import ensure_model
from src.utils.geometry import POSE_CONNECTIONS
from src.core.pose_worker import PoseWorker
from src.core.video_reader import VideoReader
from src.ui.components import VideoPanel, ScoreChartWidget
from src.core.audio_aligner import align_videos, LIBROSA_AVAILABLE, MOVIEPY_AVAILABLE
from src.core.ai_coach import AICoach, AICoachConfig, CoachingHistory

# 尝试导入音频播放库
try:
    from ffpyplayer.player import MediaPlayer
    FFPYPLAYER_AVAILABLE = True
    print("[Audio] Using ffpyplayer for audio playback")
except ImportError:
    FFPYPLAYER_AVAILABLE = False
    print("[Audio] ffpyplayer not available, audio will be disabled")
    print("[Audio] Install with: pip install ffpyplayer")

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dance Pose Scoring System")
        self.resize(1400, 900)
        
        # Apply Modern Dark Theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 16px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a3a5c, stop:1 #2a2a45);
                border: 1px solid #4a4a7a;
                border-radius: 10px;
                padding: 12px 28px;
                color: #FFFFFF;
                font-weight: 600;
                font-size: 16px;
                min-height: 44px;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a7a, stop:1 #3a3a65);
                border-color: #00d4ff;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a45, stop:1 #1a1a35);
                border-color: #6a6aaa;
            }
            QComboBox {
                background-color: #252545;
                border: 1px solid #4a4a7a;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 16px;
                min-height: 44px;
                color: #FFFFFF;
            }
            QComboBox::drop-down {
                border-radius: 0 10px 10px 0;
                background-color: #3a3a65;
            }
            QComboBox QAbstractItemView {
                background-color: #252545;
                border: 1px solid #4a4a7a;
                border-radius: 10px;
                selection-background-color: #3a3a65;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 16px;
            }
            QListWidget {
                background-color: #1e1e3e;
                border: 1px solid #3a3a65;
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 8px;
                margin: 4px 0;
                background-color: #252545;
            }
            QListWidget::item:selected {
                background-color: #3a3a65;
                border: 1px solid #00d4ff;
            }
            QProgressDialog {
                background-color: #1a1a2e;
                border-radius: 12px;
            }
            QProgressDialog QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QMessageBox {
                background-color: #1a1a2e;
                border-radius: 12px;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)

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

        self.userReader = None
        self.refReader = None
        self.useCam = False
        self.playing = False
        self.badFrames = []  # list of (score, time_str, QImage)
        
        # Audio alignment
        self.user_video_path = None
        self.ref_video_path = None
        self.alignment_offset = 0.0  # Time offset in seconds
        self.alignment_frame_offset = 0  # Frame offset
        self.is_aligned = False
        
        # 音频播放器（使用ffpyplayer）
        self.audioPlayer = None
        self.audio_path = None
        self.audio_thread = None
        self.audio_playing = False
        self._audio_paused = False  # 音频是否处于暂停状态

        # AI教练 - 默认启用
        self.ai_coach = AICoach()
        self.coaching_history = CoachingHistory()
        self.ai_feedback_enabled = True
        self.ai_feedback_countdown = 0
        self.ai_feedback_interval = 90  # 每90帧（约3秒）请求一次AI反馈
        self._session_start_time = None
        self._total_frames = 0

        self.userPanel = VideoPanel()
        self.refPanel = VideoPanel()
        
        # 实时评分标签
        self.scoreLabel = QtWidgets.QLabel("-- %")
        f = self.scoreLabel.font()
        f.setPointSize(24)
        f.setBold(True)
        self.scoreLabel.setFont(f)
        self.scoreLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.scoreLabel.setStyleSheet("""
            color: #10B981; 
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #16213e, stop:1 #0f3460);
            border: 2px solid #10B981;
            border-radius: 25px; 
            padding: 16px 32px; 
            font-size: 24pt;
            min-width: 140px;
        """)
        
        # DTW对齐评分标签
        self.dtwScoreLabel = QtWidgets.QLabel("DTW: -- %")
        f_dtw = self.dtwScoreLabel.font()
        f_dtw.setPointSize(20)
        f_dtw.setBold(True)
        self.dtwScoreLabel.setFont(f_dtw)
        self.dtwScoreLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.dtwScoreLabel.setStyleSheet("""
            color: #60A5FA; 
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a5f, stop:1 #16213e);
            border: 2px solid #60A5FA;
            border-radius: 20px; 
            padding: 12px 24px; 
            font-size: 20pt;
        """)

        self.hintLabel = QtWidgets.QLabel("")
        f2 = self.hintLabel.font()
        f2.setPointSize(22)
        f2.setBold(True)
        self.hintLabel.setFont(f2)
        self.hintLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.hintLabel.setStyleSheet("""
            color: #FBBF24; 
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d2914, stop:1 #2d1f0f);
            border: 2px solid #FBBF24;
            border-radius: 18px; 
            padding: 12px 30px; 
            font-size: 22pt;
        """)

        # Alignment status label
        self.alignStatusLabel = QtWidgets.QLabel("Not Aligned")
        self.alignStatusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.alignStatusLabel.setStyleSheet("""
            color: #888888; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a4a, stop:1 #1e1e3e);
            border: 1px solid #4a4a7a;
            border-radius: 12px; 
            padding: 8px 20px; 
            font-size: 13pt;
        """)

        # Frame counter label for debugging alignment
        self.frameLabel = QtWidgets.QLabel("User: 0 | Ref: 0")
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.frameLabel.setStyleSheet("""
            color: #a0a0c0; 
            background: #1e1e3e;
            border: 1px solid #3a3a65;
            border-radius: 8px; 
            padding: 6px 16px; 
            font-size: 11pt;
        """)

        self._miss_count = 0
        self._ema_score = None
        self._ema_dtw_score = None
        self._ema_alpha = 0.3
        self._k_decay = 2.5  # 降低敏感度，对位置偏移更宽容
        self._gamma = 0.7
        self.detect_stride = 3
        self._tick_count = 0
        self.ts_ms = 0
        self.lastPercent = 0.0
        self.lastDtwPercent = 0.0
        self.detect_stride_user = 2
        self.detect_stride_ref = 4
        self.lastUserLandmarks = []
        self.lastRefLandmarks = []
        self.lastDiffs = None

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(QtWidgets.QLabel("Score:"))
        topLayout.addWidget(self.scoreLabel)
        topLayout.addSpacing(20)
        topLayout.addWidget(self.dtwScoreLabel)  # DTW标签默认隐藏，根据模式显示
        self.dtwScoreLabel.hide()  # 默认隐藏
        topLayout.addSpacing(20)
        topLayout.addWidget(self.alignStatusLabel)
        topLayout.addSpacing(10)
        topLayout.addWidget(self.frameLabel)
        topLayout.addStretch(1)
        topLayout.addWidget(self.hintLabel)
        topLayout.addStretch(1)

        # AI教练建议面板
        self.aiCoachLabel = QtWidgets.QLabel("")
        self.aiCoachLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.aiCoachLabel.setWordWrap(True)
        self.aiCoachLabel.setStyleSheet("""
            color: #A78BFA;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d1f4a, stop:1 #1a1040);
            border: 2px solid #A78BFA;
            border-radius: 18px;
            padding: 12px 24px;
            font-size: 14pt;
            min-width: 300px;
            max-width: 500px;
        """)
        self.aiCoachLabel.setMinimumHeight(80)
        self.aiCoachLabel.hide()  # 默认隐藏

        # AI设置按钮
        self.btnAISettings = QtWidgets.QPushButton("🤖 AI Settings")
        self.btnAISettings.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7C3AED, stop:1 #5B21B6);
            border: 2px solid #7C3AED;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            font-size: 14px;
            padding: 10px 20px;
        """)
        self.btnAISettings.setMinimumHeight(44)

        # AI布局
        aiLayout = QtWidgets.QHBoxLayout()
        aiLayout.addWidget(self.aiCoachLabel, 1)
        aiLayout.addWidget(self.btnAISettings)

        self.stageLayout = QtWidgets.QHBoxLayout()
        self.stageLayout.setSpacing(20)
        self.stageLayout.addWidget(self.userPanel, 1)
        self.stageLayout.addWidget(self.refPanel, 1)
        
        self._countdown_val = 0
        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self.on_countdown_tick)

        self.btnLoadUser = QtWidgets.QPushButton("Load User Video")
        self.btnLoadRef = QtWidgets.QPushButton("Load Reference Video")
        self.btnCamToggle = QtWidgets.QPushButton("Start Camera") # Merged
        self.camCombo = QtWidgets.QComboBox()
        # Removed Refresh
        
        self.btnPlayReset = QtWidgets.QPushButton("Start") # Merged Play/Reset
        self.btnPauseResume = QtWidgets.QPushButton("Pause") # Merged Pause/Resume
        self.btnExtract = QtWidgets.QPushButton("Extract Bad Frames")
        self.btnClear = QtWidgets.QPushButton("Clear Cache")
        # Removed Step, Export

        # Layout Logic:
        # Group 1: Source Selection (Top row)
        # Group 2: Playback Control (Bottom row, centered, larger)

        ctrlLayout = QtWidgets.QVBoxLayout()
        ctrlLayout.setSpacing(18)

        # Source Controls
        sourceLayout = QtWidgets.QHBoxLayout()
        sourceLayout.setSpacing(12)
        sourceLayout.addWidget(self.btnLoadUser)
        sourceLayout.addWidget(self.btnLoadRef)
        sourceLayout.addWidget(self.camCombo)
        sourceLayout.addWidget(self.btnCamToggle)
        sourceLayout.addStretch(1) # Push to left
        
        # Playback Controls - Make them BIG
        playLayout = QtWidgets.QHBoxLayout()
        playLayout.setSpacing(15)
        playLayout.addStretch(1)
        
        # Style Play/Pause buttons to be prominent
        self.btnPlayReset.setMinimumWidth(140)
        self.btnPlayReset.setMinimumHeight(50)
        self.btnPlayReset.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669);
            border: 2px solid #10B981;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 12px 30px;
        """) # Green
        
        self.btnPauseResume.setMinimumWidth(140)
        self.btnPauseResume.setMinimumHeight(50)
        self.btnPauseResume.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F59E0B, stop:1 #D97706);
            border: 2px solid #F59E0B;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 12px 30px;
        """) # Orange
        
        self.btnExtract.setMinimumHeight(50)
        self.btnClear.setMinimumHeight(50)
        
        playLayout.addWidget(self.btnPlayReset)
        playLayout.addWidget(self.btnPauseResume)
        playLayout.addWidget(self.btnExtract)
        playLayout.addWidget(self.btnClear)
        playLayout.addStretch(1)
        
        ctrlLayout.addLayout(sourceLayout)
        ctrlLayout.addLayout(playLayout)

        self.badList = QtWidgets.QListWidget()
        self.badList.setViewMode(QtWidgets.QListView.IconMode)
        self.badList.setIconSize(QtCore.QSize(200, 112)) # Larger thumbnails
        self.badList.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.badList.setUniformItemSizes(True)
        self.badList.itemClicked.connect(self.on_bad_frame_clicked)
        # Remove fixed max height, let layout manage
        # self.badList.setMaximumHeight(160) 

        self.scoreChart = ScoreChartWidget()
        self.scoreChart.setMinimumHeight(120) # Ensure it has some height

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20) # Add margin around window
        layout.setSpacing(15) # Increase spacing between elements
        
        # 1. Top Bar (Score) - Fixed height ~100px
        layout.addLayout(topLayout)

        # 1.5. AI Coach Layout
        layout.addLayout(aiLayout)

        # 2. Video Stage - 60% of space, try to keep 16:9
        # To enforce ratio, we might need to be careful.
        # But simple stretch factor is easiest.
        layout.addLayout(self.stageLayout, 6) 
        
        # 3. Controls - Fixed height
        layout.addLayout(ctrlLayout)
        
        # 4. Bottom Area (Bad Frames + Chart) - Remaining space (~30-40%)
        # We can put them in a HBox or VBox depending on design.
        # User said "Bottom chart", so maybe side-by-side or stacked?
        # Original was stacked. Let's keep stacked but allocate space.
        
        bottomLayout = QtWidgets.QHBoxLayout()
        bottomLayout.setSpacing(20)
        
        # Left: Bad Frames List (Fixed width or proportional?)
        # Right: Chart
        
        # Let's group Bad Frames and Chart side-by-side to save vertical space for video?
        # Or keep vertical. If video is 60%, we have 40% left.
        # Top bar ~10%, Controls ~10%, Bottom ~20%.
        
        # Let's try side-by-side for bottom area to give more height to video.
        badGroup = QtWidgets.QVBoxLayout()
        badGroup.addWidget(QtWidgets.QLabel("Bad Frames:"))
        badGroup.addWidget(self.badList)
        
        chartGroup = QtWidgets.QVBoxLayout()
        chartGroup.addWidget(QtWidgets.QLabel("Score Trend:"))
        chartGroup.addWidget(self.scoreChart)
        
        bottomLayout.addLayout(badGroup, 1)
        bottomLayout.addLayout(chartGroup, 2) # Chart wider
        
        layout.addLayout(bottomLayout, 3) # Allocate weight 3 to bottom (Video is 6)
        
        # Adjust stretches:
        # Top: 0 (fixed)
        # Video: 6
        # Controls: 0 (fixed)
        # Bottom: 3
        # Total flexible = 9. Video gets 6/9 = 66%. Close to 60%.


        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()
        self._timer_interval_ms = 33  # cache for jitter control
        self.update_timer_interval()

        self.btnLoadUser.clicked.connect(self.load_user_video)
        self.btnLoadRef.clicked.connect(self.load_ref_video)
        self.btnCamToggle.clicked.connect(self.toggle_cam)
        # self.btnRefreshCams.clicked.connect(self.enumerate_cams)
        self.btnPlayReset.clicked.connect(self.play_reset)
        self.btnPauseResume.clicked.connect(self.pause_resume)
        self.btnExtract.clicked.connect(self.extract_bad)
        self.btnClear.clicked.connect(self.clear_cache)
        self.btnAISettings.clicked.connect(self.show_ai_settings)
        QtCore.QTimer.singleShot(0, self.enumerate_cams)

    def enumerate_cams(self):
        self.camCombo.clear()
        found = False
        for i in range(0, 8):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            name = None
            if ok:
                ret, frame = cap.read()
                if ret:
                    name = f"Camera {i}"
                    found = True
            if cap:
                cap.release()
            if name:
                self.camCombo.addItem(name, i)
        if not found:
            self.camCombo.addItem("No Camera Found", None)

    def load_user_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select User Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        
        # 恢复视频面板大小：等比例
        self._set_panel_ratio(1, 1)
        
        self.userReader = VideoReader(path)
        self.userReader.start()
        # Pause initially so it doesn't auto-play
        self.userReader.pause()
        self.playing = False
        self.update_timer_interval()
        # 直接显示视频第一帧预览
        self.show_video_preview(path, self.userPanel, "User")
        
        # Save path and try alignment
        self.user_video_path = path
        self.try_audio_alignment()

    def load_ref_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        if self.refReader:
            self.refReader.stop()
        
        # 设置音频源
        if FFPYPLAYER_AVAILABLE:
            self.audio_path = path
            print(f"[Audio] Audio path set: {path}")
        else:
            print("[Audio] Audio playback disabled (ffpyplayer not installed)")
        
        self.refReader = VideoReader(path)
        self.refReader.finished.connect(self.on_playback_finished) # Connect signal
        self.refReader.start()
        # Pause initially
        self.refReader.pause()
        self.playing = False
        self.update_timer_interval()
        # 直接显示视频第一帧预览
        self.show_video_preview(path, self.refPanel, "Reference")
        
        # Save path and try alignment
        self.ref_video_path = path
        self.try_audio_alignment()

    def try_audio_alignment(self):
        if not self.user_video_path or not self.ref_video_path:
            return
        
        if not LIBROSA_AVAILABLE or not MOVIEPY_AVAILABLE:
            self.alignStatusLabel.setText("Alignment Unavailable")
            self.alignStatusLabel.setStyleSheet("""
                color: #888888; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a4a, stop:1 #1e1e3e);
                border: 1px solid #4a4a7a;
                border-radius: 12px; 
                padding: 8px 20px; 
                font-size: 13pt;
            """)
            print("[Alignment] Required libraries not available")
            return
        
        self.alignStatusLabel.setText("Aligning...")
        self.alignStatusLabel.setStyleSheet("""
            color: #FBBF24; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d2914, stop:1 #2d1f0f);
            border: 1px solid #FBBF24;
            border-radius: 12px; 
            padding: 8px 20px; 
            font-size: 13pt;
        """)
        
        # 禁用开始按钮
        self.btnPlayReset.setEnabled(False)
        
        # 显示进度对话框
        self.alignmentProgressDialog = QtWidgets.QProgressDialog(
            "Calculating audio alignment, please wait...",
            None,  # 不显示取消按钮
            0, 0,  # 不确定的进度
            self
        )
        self.alignmentProgressDialog.setWindowTitle("Audio Alignment")
        self.alignmentProgressDialog.setWindowModality(QtCore.Qt.WindowModal)
        self.alignmentProgressDialog.setMinimumDuration(0)  # 立即显示
        self.alignmentProgressDialog.setCancelButton(None)  # 禁止取消
        self.alignmentProgressDialog.setStyleSheet("""
            QProgressDialog {
                background-color: #1a1a2e;
            }
            QProgressDialog QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)
        
        # Run alignment in background thread
        def do_alignment():
            try:
                time_offset, frame_offset = align_videos(
                    self.user_video_path, 
                    self.ref_video_path, 
                    method='chroma'
                )
                QtCore.QMetaObject.invokeMethod(
                    self, 
                    "on_alignment_finished", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(float, time_offset),
                    QtCore.Q_ARG(int, frame_offset)
                )
            except Exception as e:
                print(f"[Alignment] Error: {e}")
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "on_alignment_failed",
                    QtCore.Qt.QueuedConnection
                )
        
        threading.Thread(target=do_alignment, daemon=True).start()

    @QtCore.pyqtSlot(float, int)
    def on_alignment_finished(self, time_offset, frame_offset):
        # 关闭进度对话框
        if hasattr(self, 'alignmentProgressDialog') and self.alignmentProgressDialog:
            self.alignmentProgressDialog.close()
            self.alignmentProgressDialog = None
        
        # 启用开始按钮
        self.btnPlayReset.setEnabled(True)
        
        self.alignment_offset = time_offset
        self.alignment_frame_offset = frame_offset
        self.is_aligned = True
        
        # Update UI
        if abs(time_offset) < 0.1:
            self.alignStatusLabel.setText("Aligned: In Sync")
            self.alignStatusLabel.setStyleSheet("""
                color: #10B981; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a3a2e, stop:1 #0f2a1f);
                border: 1px solid #10B981;
                border-radius: 12px; 
                padding: 8px 20px; 
                font-size: 13pt;
            """)
            msg = "Videos are already in sync!"
        else:
            direction = "User ahead" if time_offset < 0 else "User behind"
            self.alignStatusLabel.setText(f"Aligned: {direction} {abs(time_offset):.1f}s")
            self.alignStatusLabel.setStyleSheet("""
                color: #60A5FA; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e3a5f, stop:1 #16213e);
                border: 1px solid #60A5FA;
                border-radius: 12px; 
                padding: 8px 20px; 
                font-size: 13pt;
            """)
            msg = f"Alignment complete!\n\nTime offset: {abs(time_offset):.2f}s\nFrame offset: {abs(frame_offset)} frames\n\n{'User video starts earlier' if time_offset < 0 else 'User video starts later'}"
        
        # 显示美观的完成提示
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setWindowTitle("Audio Alignment Complete")
        msgBox.setText("✓ Audio Alignment Successful")
        msgBox.setInformativeText(msg)
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        msgBox.exec()
        
        # Apply alignment to video readers
        self.apply_alignment()
    
    @QtCore.pyqtSlot()
    def on_alignment_failed(self):
        # 关闭进度对话框
        if hasattr(self, 'alignmentProgressDialog') and self.alignmentProgressDialog:
            self.alignmentProgressDialog.close()
            self.alignmentProgressDialog = None
        
        # 启用开始按钮
        self.btnPlayReset.setEnabled(True)
        
        self.alignStatusLabel.setText("Alignment Failed")
        self.alignStatusLabel.setStyleSheet("""
            color: #EF4444; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a1a1a, stop:1 #2a0f0f);
            border: 1px solid #EF4444;
            border-radius: 12px; 
            padding: 8px 20px; 
            font-size: 13pt;
        """)
        self.is_aligned = False
        
        # 显示美观的失败提示
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setWindowTitle("Audio Alignment Failed")
        msgBox.setText("✗ Alignment Failed")
        msgBox.setInformativeText("Could not align videos automatically.\n\nPlease check if both videos have audio tracks.\nYou can still play videos without alignment.")
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
        msgBox.exec()

    def apply_alignment(self):
        if not self.is_aligned:
            print("[Alignment] apply_alignment called but not aligned")
            return
        
        print(f"[Alignment] apply_alignment: frame_offset={self.alignment_frame_offset}")
        
        # alignment_frame_offset 的含义：
        # 正数：参考视频比用户视频晚开始 → 用户视频需要跳过帧
        # 负数：用户视频比参考视频晚开始 → 参考视频需要跳过帧
        
        user_start = 0
        ref_start = 0
        
        if self.alignment_frame_offset > 0:
            # 参考视频晚开始，用户视频需要跳过前面的帧来等待参考视频
            user_start = self.alignment_frame_offset
            print(f"[Alignment] Ref video starts later: user_start={user_start}")
        elif self.alignment_frame_offset < 0:
            # 用户视频晚开始，参考视频需要跳过前面的帧来等待用户视频
            ref_start = -self.alignment_frame_offset
            print(f"[Alignment] User video starts later: ref_start={ref_start}")
        else:
            print("[Alignment] No frame offset needed (videos in sync)")
        
        # Recreate video readers with correct start frames
        if self.userReader:
            self.userReader.stop()
            self.userReader = VideoReader(self.user_video_path, start_frame=user_start)
            self.userReader.start()
            self.userReader.pause()
            print(f"[Alignment] User video recreated with start_frame={user_start}")
        
        if self.refReader:
            self.refReader.stop()
            self.refReader = VideoReader(self.ref_video_path, start_frame=ref_start)
            self.refReader.finished.connect(self.on_playback_finished)
            self.refReader.start()
            self.refReader.pause()
            print(f"[Alignment] Reference video recreated with start_frame={ref_start}")
        
        self.update_timer_interval()
        
        # Print FPS info for debugging
        if self.userReader and self.refReader:
            print(f"[Alignment] User FPS: {self.userReader.fps}, Ref FPS: {self.refReader.fps}")

    def on_playback_finished(self):
        self.playing = False
        self.countdown_timer.stop()
        
        if self.userReader: self.userReader.pause()
        if self.refReader: self.refReader.pause()
        
        # 停止音频（完全销毁，不是暂停）
        self.stop_audio_playback()
        
        # Calculate final score
        avg_score = 0
        if self.scoreChart.scores:
            avg_score = sum(self.scoreChart.scores) / len(self.scoreChart.scores)

        # 生成AI课后总结（同步调用，确保在对话框中显示）
        ai_summary = ""
        if self.ai_feedback_enabled and self.ai_coach.config.is_configured():
            ai_summary = self.generate_session_summary_sync()
        
        # Show Summary Dialog with AI feedback
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Session Ended")
        msg.setText(f"<h3>Final Score: {int(avg_score)} pts</h3>")
        
        if ai_summary:
            msg.setInformativeText(f"Recorded {len(self.scoreChart.scores)} score points.\n\n🤖 AI教练建议:\n{ai_summary}")
        else:
            msg.setInformativeText(f"Recorded {len(self.scoreChart.scores)} score points.\n\nGreat Job!")
        
        msg.setStyleSheet("QLabel{min-width: 400px; font-size: 14px;} QPushButton{ font-size: 14px; }")
        
        # Force OK button text
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        ok_btn = msg.button(QtWidgets.QMessageBox.Ok)
        if ok_btn:
            ok_btn.setText("OK")

        msg.exec_()

        # Reset UI state to "Ready to Start"
        self.btnPlayReset.setText("Start")
        self.btnPlayReset.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669);
            border: 2px solid #10B981;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 12px 30px;
        """) # Green
        self.btnPauseResume.setText("Pause")
        self.btnPauseResume.setEnabled(False) # Disable pause when stopped

    def toggle_cam(self):
        if self.useCam:
            self.stop_cam()
            self.btnCamToggle.setText("Start Camera")
            self.btnCamToggle.setStyleSheet("") # Reset style
        else:
            self.start_cam()
            self.btnCamToggle.setText("Stop Camera")
            self.btnCamToggle.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EF4444, stop:1 #DC2626);
                border: 2px solid #EF4444;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                font-size: 16px;
            """) # Red

    def start_cam(self):
        self.useCam = True
        if self.userReader:
            self.userReader.stop()
        idx = self.camCombo.currentData()
        if idx is None:
            idx = 0
        self.userReader = VideoReader(int(idx))
        self.userReader.start()
        
        # Camera mode doesn't support audio alignment
        self.user_video_path = None
        self.is_aligned = False
        self.alignStatusLabel.setText("Camera Mode")
        self.alignStatusLabel.setStyleSheet("""
            color: #888888; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a4a, stop:1 #1e1e3e);
            border: 1px solid #4a4a7a;
            border-radius: 12px; 
            padding: 8px 20px; 
            font-size: 13pt;
        """)
        
        # 调整视频面板大小：摄像头模式时用户面板缩小，参考面板变大
        self._set_panel_ratio(1, 2)  # 用户:参考 = 1:2
        
        # 显示DTW评分标签
        self.dtwScoreLabel.show()
        
        # 注意：不要设置 self.playing = True，等待用户点击开始按钮
        self.playing = False
        self.update_timer_interval()
        
    def stop_cam(self):
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        self.userReader = None
        
        # 恢复视频面板大小：等比例
        self._set_panel_ratio(1, 1)  # 用户:参考 = 1:1
        
        # 显示实时评分标签
        self.dtwScoreLabel.show()
        
        self.playing = False
        self.update_timer_interval()
    
    def _set_panel_ratio(self, user_ratio, ref_ratio):
        """设置视频面板的大小比例"""
        # 移除所有widget
        self.stageLayout.removeWidget(self.userPanel)
        self.stageLayout.removeWidget(self.refPanel)
        
        # 重新添加并设置新的比例
        self.stageLayout.addWidget(self.userPanel, user_ratio)
        self.stageLayout.addWidget(self.refPanel, ref_ratio)

    def play_reset(self):
        # This button now acts as Start / Stop
        if self.playing or (self.countdown_timer.isActive()):
            # User clicked "Stop"
            self.on_playback_finished() # Show score and stop
            self.btnPlayReset.setText("Start")
            self.btnPlayReset.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669);
                border: 2px solid #10B981;
                border-radius: 12px;
                color: white;
                font-weight: bold;
                font-size: 18px;
                padding: 12px 30px;
            """) # Green
        else:
            # User clicked "Start" -> Reset and Start
            self.scoreChart.reset()
            self._session_start_time = time.time()
            self._total_frames = 0
            self.coaching_history.clear()
            self.aiCoachLabel.hide()
            self.btnPlayReset.setText("Stop")
            self.btnPlayReset.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EF4444, stop:1 #DC2626);
                border: 2px solid #EF4444;
                border-radius: 12px;
                color: white;
                font-weight: bold;
                font-size: 18px;
                padding: 12px 30px;
            """) # Red
            
            # Reset readers
            if self.userReader: 
                self.userReader.pause()
                self.userReader.reset()
            if self.refReader: 
                self.refReader.pause()
                self.refReader.reset()
            
            # 停止并重置音频
            self.stop_audio_playback()
            
            # Start countdown
            self._countdown_val = 3
            self.scoreLabel.setText(str(self._countdown_val))
            self.countdown_timer.start()
            
            # Update Pause/Resume state
            self.btnPauseResume.setText("Pause")
            self.btnPauseResume.setEnabled(True)
        
    def pause_resume(self):
        if self.playing:
            self.pause()
            self.btnPauseResume.setText("Resume")
        else:
            self.resume()
            self.btnPauseResume.setText("Pause")

    def resume(self):
        self.playing = True
        # Resume threads
        if self.userReader: self.userReader.resume()
        if self.refReader: self.refReader.resume()
        # 恢复音频（使用ffpyplayer的暂停功能）
        self.resume_audio()

    def on_countdown_tick(self):
        self._countdown_val -= 1
        if self._countdown_val > 0:
            self.scoreLabel.setText(str(self._countdown_val))
        else:
            self.countdown_timer.stop()
            self.scoreLabel.setText("GO!")
            QtCore.QTimer.singleShot(500, self.start_playback)

    def start_playback(self):
        self.playing = True
        self.scoreLabel.setText("-- %")
        # Resume threads
        if self.userReader: self.userReader.resume()
        if self.refReader: self.refReader.resume()
        # 播放音频
        self.start_audio_playback()

    def pause(self):
        self.playing = False
        self.countdown_timer.stop()
        
        if self.userReader: self.userReader.pause()
        if self.refReader: self.refReader.pause()
        # 暂停音频（使用ffpyplayer的暂停功能）
        self.pause_audio()

    # Removed step()

    def on_tick(self):
        if self.playing:
            self.read_and_process(step=False)

    def on_results_ready(self, u_lms, r_lms, diffs, real_time_percent, dtw_percent, timing_hint):
        if len(u_lms):
            self.lastUserLandmarks = u_lms
        if len(r_lms):
            self.lastRefLandmarks = r_lms
        if len(u_lms) and len(r_lms):
            self.lastDiffs = diffs
            
            # 1. 实时评分EMA平滑
            if self._ema_score is None:
                self._ema_score = real_time_percent
            else:
                self._ema_score = self._ema_alpha * real_time_percent + (1.0 - self._ema_alpha) * self._ema_score
            self.lastPercent = self._ema_score
            
            # 2. DTW评分EMA平滑
            if self._ema_dtw_score is None:
                self._ema_dtw_score = dtw_percent
            else:
                self._ema_dtw_score = self._ema_alpha * dtw_percent + (1.0 - self._ema_alpha) * self._ema_dtw_score
            self.lastDtwPercent = self._ema_dtw_score
            
            # 3. 根据模式显示不同的评分
            if self.useCam:
                # 摄像头模式：显示DTW评分
                self.scoreLabel.setText(f"{int(round(self.lastDtwPercent))} %")
                self.dtwScoreLabel.hide()
            else:
                # 视频比较模式：只显示实时评分
                self.scoreLabel.setText(f"{int(round(self.lastPercent))} %")
                self.dtwScoreLabel.hide()
            
            # 4. 更新折线图
            self.scoreChart.add_scores(self.lastPercent, self.lastDtwPercent)

            # Timing hint disabled
            self.hintLabel.setText("")

            # 5. AI教练反馈（低分时主动提供建议）
            self._total_frames += 1
            if self.ai_feedback_enabled and self.ai_coach.config.is_configured():
                current_score = self.lastDtwPercent if self.useCam else self.lastPercent

                # 当分数低于70时，主动请求AI建议
                if current_score < 70 and self._total_frames % 30 == 0:
                    self.request_ai_feedback(diffs, current_score, timing_hint)

                # 定时AI反馈（每3秒一次，不管分数如何）
                if self._total_frames % self.ai_feedback_interval == 0:
                    self.request_ai_feedback(diffs, current_score, timing_hint)

            self._miss_count = 0
        else:
            self._miss_count += 1
            # Keep last skeleton/diffs to avoid flicker; only clear after sustained misses
            if self._miss_count > 30:
                self.lastPercent = 0.0
                self.lastDtwPercent = 0.0
                self.lastDiffs = None
                self.scoreLabel.setText("0 %")
                self.dtwScoreLabel.setText("0 %")
                self.hintLabel.setText("")
                self._ema_score = None
                self._ema_dtw_score = None

    def read_and_process(self, step=False):
        user_frame = None
        ref_frame = None
        
        # Update frame counter display
        user_frame_num = 0
        ref_frame_num = 0
        if self.userReader:
            user_frame_num = self.userReader.get_current_frame()
        if self.refReader:
            ref_frame_num = self.refReader.get_current_frame()
        self.frameLabel.setText(f"User: {user_frame_num} | Ref: {ref_frame_num}")
        
        if self.userReader:
            ok, frame = self.userReader.get_frame()
            if ok:
                user_frame = frame
            # If stepping or paused but frame available, we might want to see it?
            # But get_frame() clears new_frame_available.
            # If we just want current frame for refresh:
            if not ok and self.userReader.frame is not None:
                # If we are not playing, we might still want to redraw if resize happen etc.
                # But here we only redraw if we have a *new* frame or forcing update.
                pass
            if not ok and not self.useCam and self.playing:
                # If file playback and no new frame, maybe end of file or just waiting
                pass

        if self.refReader:
            ok, frame = self.refReader.get_frame()
            if ok:
                ref_frame = frame

        # Push to worker queue if it's not full
        self._tick_count += 1
        
        # If we have a new user frame, we should detect.
        # If we have a new ref frame, we should detect.
        # But to save CPU, we might still stride.
        
        # Actually, since reader is threaded, we can just check if user_frame is not None
        need_detect_user = (user_frame is not None) and (self._tick_count % (self.detect_stride_user if self.useCam else self.detect_stride) == 0 or not self.lastUserLandmarks)
        need_detect_ref = (ref_frame is not None) and (self._tick_count % self.detect_stride_ref == 0 or not self.lastRefLandmarks)
        
        if need_detect_user or need_detect_ref:
            payload = (
                user_frame if need_detect_user else None,
                ref_frame if need_detect_ref else None,
                self._k_decay,
                self._gamma
            )
            try:
                self.worker.queue.put_nowait(payload)
            except Exception:
                # Queue full, skip detection for this frame, but we still draw it below
                pass

        # Always draw the latest video frame and overlay the latest known skeleton
        # We need to draw even if no *new* frame arrived? 
        # Typically yes if we want to keep 30fps UI refresh, but if no new frame, image is same.
        # But let's just draw when we get a frame to save UI paint.
        
        if user_frame is not None:
            self.draw_skeleton_on_panel(self.userPanel, user_frame, self.lastUserLandmarks, diffs=self.lastDiffs)
        elif self.userReader and self.userReader.frame is not None and not self.playing:
             # Redraw current frame if paused (e.g. window resize or just stopped)
             # But this runs every 33ms, redrawing same image is wasteful.
             # So only draw if user_frame (new) is not None.
             pass

        if ref_frame is not None:
            self.draw_skeleton_on_panel(self.refPanel, ref_frame, self.lastRefLandmarks, diffs=None)

        # Collect bad frames if score is low
        if self.playing and self.lastPercent > 0 and self.lastPercent < 60 and user_frame is not None:
            if self._tick_count % 10 == 0: # Don't collect too many
                qimg = self.frame_to_qimage(user_frame)
                ts = self.userReader.get_time_str()
                self.badFrames.append((self.lastPercent, ts, qimg))
                if len(self.badFrames) > 200:
                    self.badFrames.pop(0)

    def closeEvent(self, event):
        self.worker.stop()
        if hasattr(self, "worker_thread"):
            self.worker_thread.quit()
            self.worker_thread.wait()
        if self.userReader:
            self.userReader.stop()
        if self.refReader:
            self.refReader.stop()
        event.accept()

    def update_timer_interval(self):
        targets = []
        if self.userReader:
            targets.append(self.userReader.fps)
        if self.refReader:
            targets.append(self.refReader.fps)
        
        if not targets:
            target_fps = 30.0
        else:
            target_fps = max(targets)
            if target_fps > 60.0: target_fps = 60.0
            if target_fps < 15.0: target_fps = 15.0
            
        new_interval = int(round(1000.0 / target_fps))
        if abs(new_interval - self._timer_interval_ms) >= 2:
            self._timer_interval_ms = new_interval
            self.timer.setInterval(new_interval)

    def draw_skeleton_on_panel(self, panel: QtWidgets.QLabel, frame_bgr, landmarks, diffs=None):
        # Resize for display performance first
        # But wait, if we resize to 16:9 fixed, we might stretch.
        # Let's keep original aspect ratio but fit into panel size?
        # Actually panel.setScaledContents(False) + setPixmap with KeepAspectRatio is better.
        # But we need to draw lines first on the original frame (or a resized copy).
        
        # 1. Resize copy to a reasonable size to speed up drawing (e.g. max 1280 width)
        # Don't force 640x360 unless we want low res.
        # Let's just use the frame as is if it's small enough, or downscale if huge (4K).
        
        MAX_DISP_W = 1280
        h_orig, w_orig = frame_bgr.shape[:2]
        if w_orig > MAX_DISP_W:
            scale = MAX_DISP_W / float(w_orig)
            canvas = cv2.resize(frame_bgr, (MAX_DISP_W, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            canvas = frame_bgr.copy()

        h, w = canvas.shape[:2]
        if len(landmarks):
            pts = [(int(p["x"] * w), int(p["y"] * h)) for p in landmarks]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(canvas, pts[a], pts[b], (16, 185, 129), 2)
            body_idxs = list(range(11, 33))
            for i in body_idxs:
                if i < len(pts):
                    x, y = pts[i]
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
            if diffs:
                idxs = {"leftShoulder":11,"rightShoulder":12,"leftElbow":13,"rightElbow":14,"leftHip":23,"rightHip":24,"leftKnee":25,"rightKnee":26}
                for k, i in idxs.items():
                    if i < len(pts):
                        d = diffs.get(k, 0.0)
                        color = (34, 197, 94)
                        if d >= 40: color = (239, 68, 68)
                        elif d >= 20: color = (245, 158, 11)
                        # OpenCV putText doesn't support unicode like ° well on Windows/default font
                        # Use ASCII or PIL. For speed, just use "d" or nothing
                        cv2.putText(canvas, f"{int(round(d))}", (pts[i][0]+6, pts[i][1]-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        qimg = self.frame_to_qimage(canvas)
        
        # Scale to fit panel size while maintaining aspect ratio
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

    def extract_bad(self):
        self.badList.clear()
        items = sorted(self.badFrames, key=lambda x: x[0])[:12]
        for score, tstr, qimg in items:
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimg).scaled(160, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            item = QtWidgets.QListWidgetItem(icon, f"{int(round(score))}% @ {tstr}")
            # 存储原始图像数据
            item.setData(QtCore.Qt.UserRole, qimg)
            item.setData(QtCore.Qt.UserRole + 1, score)
            item.setData(QtCore.Qt.UserRole + 2, tstr)
            self.badList.addItem(item)
    
    def on_bad_frame_clicked(self, item):
        """点击坏帧缩略图时显示大图"""
        # 获取存储的数据
        qimg = item.data(QtCore.Qt.UserRole)
        score = item.data(QtCore.Qt.UserRole + 1)
        tstr = item.data(QtCore.Qt.UserRole + 2)
        
        if qimg is None:
            return
        
        # 创建对话框显示大图
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Bad Frame - Score: {int(round(score))}% @ {tstr}")
        dialog.setMinimumSize(800, 600)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 创建标签显示图片
        imgLabel = QtWidgets.QLabel()
        imgLabel.setAlignment(QtCore.Qt.AlignCenter)
        
        # 缩放图片以适应窗口，但保持比例
        pixmap = QtGui.QPixmap.fromImage(qimg)
        if pixmap.width() > 1200 or pixmap.height() > 800:
            pixmap = pixmap.scaled(1200, 800, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        imgLabel.setPixmap(pixmap)
        layout.addWidget(imgLabel)
        
        # 添加信息标签
        infoLabel = QtWidgets.QLabel(f"Score: {int(round(score))}% | Time: {tstr}")
        infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        infoLabel.setStyleSheet("font-size: 16px; color: #EF4444; padding: 10px;")
        layout.addWidget(infoLabel)
        
        # 添加关闭按钮
        closeBtn = QtWidgets.QPushButton("Close")
        closeBtn.clicked.connect(dialog.close)
        closeBtn.setStyleSheet("""
            QPushButton {
                background-color: #4B5563;
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6B7280;
            }
        """)
        
        btnLayout = QtWidgets.QHBoxLayout()
        btnLayout.addStretch()
        btnLayout.addWidget(closeBtn)
        btnLayout.addStretch()
        layout.addLayout(btnLayout)
        
        # 设置对话框样式
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
            }
        """)
        
        dialog.exec()

    def clear_cache(self):
        """清除所有缓存数据，恢复到程序刚启动的状态"""
        # 1. 暂停播放
        self.pause()
        
        # 2. 重置视频读取器
        if self.userReader:
            self.userReader.pause()
            self.userReader.reset()
        if self.refReader:
            self.refReader.pause()
            self.refReader.reset()
        
        # 3. 停止并清除音频
        self.stop_audio_playback()
        self.audio_path = None
        
        # 4. 清除工作线程的历史缓存
        self.worker.ref_history.clear()
        
        # 5. 重置音频对齐状态
        self.user_video_path = None
        self.ref_video_path = None
        self.alignment_offset = 0.0
        self.alignment_frame_offset = 0
        self.is_aligned = False
        self.alignStatusLabel.setText("Not Aligned")
        self.alignStatusLabel.setStyleSheet("""
            color: #888888; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a4a, stop:1 #1e1e3e);
            border: 1px solid #4a4a7a;
            border-radius: 12px; 
            padding: 8px 20px; 
            font-size: 13pt;
        """)
        
        # 6. 重置评分数据
        self._ema_score = None
        self._ema_dtw_score = None
        self.lastPercent = 0.0
        self.lastDtwPercent = 0.0
        self.scoreLabel.setText("-- %")
        self.dtwScoreLabel.setText("DTW: -- %")
        
        # 7. 清除坏帧记录
        self.badFrames.clear()
        self.badList.clear()
        
        # 8. 清除折线图
        self.scoreChart.reset()
        
        # 9. 重置骨架数据
        self.lastUserLandmarks = []
        self.lastRefLandmarks = []
        self.lastDiffs = None
        
        # 10. 重置UI状态
        self.playing = False
        self.btnPlayReset.setText("Start")
        self.btnPlayReset.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669);
            border: 2px solid #10B981;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 12px 30px;
        """)
        self.btnPauseResume.setText("Pause")
        self.btnPauseResume.setEnabled(False)
        
        # 11. 清空视频面板
        self.userPanel.clear()
        self.refPanel.clear()
        
        print("[Clear] All cache data cleared. System reset to initial state.")

    def start_audio_playback(self):
        """使用ffpyplayer开始播放音频
        
        新逻辑：播放不需要跳过帧的视频的音频，保证音画同步
        - 如果用户视频跳过帧（用户视频晚开始），播放参考视频的音频
        - 如果参考视频跳过帧（参考视频晚开始），播放用户视频的音频
        """
        if not FFPYPLAYER_AVAILABLE:
            print("[Audio] ffpyplayer not available")
            return
        
        # 先停止之前的播放
        self.stop_audio_playback()
        
        # 决定播放哪个视频的音频
        audio_source = None
        
        if self.is_aligned:
            if self.alignment_frame_offset > 0:
                # 用户视频跳过帧，参考视频从头播放 → 播放参考视频的音频
                audio_source = self.ref_video_path
                print(f"[Audio] User video skips frames, playing ref video audio")
            elif self.alignment_frame_offset < 0:
                # 参考视频跳过帧，用户视频从头播放 → 播放用户视频的音频
                audio_source = self.user_video_path
                print(f"[Audio] Ref video skips frames, playing user video audio")
            else:
                # 两个视频同步，播放参考视频的音频
                audio_source = self.ref_video_path
                print(f"[Audio] Videos in sync, playing ref video audio")
        else:
            # 未对齐，播放参考视频的音频
            audio_source = self.ref_video_path
            print(f"[Audio] Not aligned, playing ref video audio")
        
        if not audio_source:
            print("[Audio] No audio source available")
            return
        
        print(f"[Audio] Playing audio from: {audio_source}")
        
        def play_audio():
            try:
                # 创建播放器，从头开始播放
                ff_opts = {}
                self.audioPlayer = MediaPlayer(audio_source, ff_opts=ff_opts)
                self.audio_playing = True
                self._audio_paused = False
                print(f"[Audio] Started playback")
                
                while self.audio_playing:
                    frame, val = self.audioPlayer.get_frame()
                    if val == 'eof':
                        break
                    if frame is None:
                        time.sleep(0.01)
                        continue
                
                if self.audioPlayer:
                    self.audioPlayer.close_player()
                    self.audioPlayer = None
                print("[Audio] Playback finished")
            except Exception as e:
                import traceback
                print(f"[Audio] Playback error: {e}")
                traceback.print_exc()
        
        self.audio_thread = threading.Thread(target=play_audio, daemon=True)
        self.audio_thread.start()
        print(f"[Audio] Thread started")
    
    def pause_audio(self):
        """暂停音频播放（不销毁播放器）"""
        if self.audioPlayer and not self._audio_paused:
            try:
                self.audioPlayer.set_pause(True)
                self._audio_paused = True
                print("[Audio] Paused")
            except Exception as e:
                print(f"[Audio] Pause error: {e}")
    
    def resume_audio(self):
        """恢复音频播放（不重新创建播放器）"""
        if self.audioPlayer and self._audio_paused:
            try:
                self.audioPlayer.set_pause(False)
                self._audio_paused = False
                print("[Audio] Resumed")
            except Exception as e:
                print(f"[Audio] Resume error: {e}")
    
    def _sync_audio_video(self):
        """同步音频和视频播放位置 - 已禁用"""
        pass

    def stop_audio_playback(self):
        """停止音频播放（销毁播放器）"""
        self.audio_playing = False
        self._audio_paused = False
        
        # 关闭音频播放器
        if self.audioPlayer:
            try:
                self.audioPlayer.close_player()
            except:
                pass
            self.audioPlayer = None
        
        # 等待音频线程结束
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.5)
        
        print("[Audio] Stopped")

    def on_audio_error(self, error):
        """音频错误处理"""
        error_messages = {
            0: "无错误",
            1: "资源错误",
            2: "格式错误",
            3: "网络错误",
            4: "访问被拒绝",
            5: "服务缺失"
        }
        print(f"[Audio Error] {error_messages.get(error, f'未知错误: {error}')}")
        print(f"[Audio Error String] {self.audioPlayer.errorString()}")

    def on_audio_state_changed(self, state):
        """音频状态变化监听"""
        states = {
            0: "停止",
            1: "播放中",
            2: "暂停"
        }
        print(f"[Audio State] {states.get(state, f'未知状态: {state}')}")

    def show_video_preview(self, video_path, panel, label):
        """直接读取视频第一帧并显示预览"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.draw_frame_on_panel(panel, frame)
                    print(f"[Preview] {label} video loaded: {video_path}")
            cap.release()
        except Exception as e:
            print(f"[Preview] Error loading {label} video: {e}")

    def draw_frame_on_panel(self, panel, frame_bgr):
        """在视频面板上显示一帧图像（无骨架）"""
        # 调整大小以适应显示
        MAX_DISP_W = 1280
        h_orig, w_orig = frame_bgr.shape[:2]
        if w_orig > MAX_DISP_W:
            scale = MAX_DISP_W / float(w_orig)
            canvas = cv2.resize(frame_bgr, (MAX_DISP_W, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            canvas = frame_bgr.copy()

        # 转换为QImage并显示
        qimg = self.frame_to_qimage(canvas)
        
        # 缩放以适应面板
        panel_w = panel.width()
        panel_h = panel.height()
        if panel_w > 10 and panel_h > 10:
             scaled_pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                 panel_w, panel_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
             )
             panel.setPixmap(scaled_pixmap)
        else:
             panel.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def export_pdf(self):
        try:
            from reportlab.pdfgen import canvas as pdfcanvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Warning", "Reportlab not installed. Please pip install reportlab")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Report", "pose_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        c = pdfcanvas.Canvas(path, pagesize=A4)
        w, h = A4
        y = h - 50
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "Dance Pose Scoring Report")
        y -= 30
        c.setFont("Helvetica", 12)
        items = sorted(self.badFrames, key=lambda x: x[0])[:8]
        c.drawString(40, y, f"Top {len(items)} bad frames:")
        y -= 20
        for score, tstr, qimg in items:
            c.drawString(40, y, f"{int(round(score))}% at {tstr}")
            y -= 14
            # Convert QImage to bytes for reportlab
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.ReadWrite)
            qimg.save(buffer, "PNG")
            data = buffer.data()
            buffer.close()
            img = ImageReader(bytes(data))
            iw, ih = 480, 270
            c.drawImage(img, 40, y-ih, width=iw, height=ih)
            y -= (ih + 30)
            if y < 120:
                c.showPage()
                y = h - 50
        c.save()
        QtWidgets.QMessageBox.information(self, "Done", "PDF Report Exported")

    def request_ai_feedback(self, diffs, score, timing_hint=""):
        """异步请求AI教练反馈"""
        def do_request():
            try:
                feedback = self.ai_coach.analyze_realtime_feedback(diffs, score, timing_hint)
                if feedback:
                    timestamp = time.strftime("%H:%M:%S")
                    self.coaching_history.add_feedback(timestamp, feedback, score)
                    QtCore.QMetaObject.invokeMethod(
                        self, "_update_ai_feedback",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, feedback)
                    )
            except Exception as e:
                print(f"[AI Coach] Feedback request failed: {e}")

        threading.Thread(target=do_request, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _update_ai_feedback(self, feedback):
        """在UI线程中更新AI反馈显示"""
        self.aiCoachLabel.setText(f"🤖 {feedback}")
        if not self.aiCoachLabel.isVisible():
            self.aiCoachLabel.show()

    def show_ai_settings(self):
        """显示AI设置对话框"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("🤖 AI Coach Settings")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
            QCheckBox {
                color: #E0E0E0;
                font-size: 16px;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(20)

        title = QtWidgets.QLabel("🤖 AI Dance Coach")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #A78BFA;")
        layout.addWidget(title)

        enable_check = QtWidgets.QCheckBox("启用 AI 教练")
        enable_check.setChecked(self.ai_coach.config.enabled)
        layout.addWidget(enable_check)

        desc = QtWidgets.QLabel("""
            AI教练会在训练过程中提供专业的舞蹈动作和节奏建议。
            练习结束后会生成详细的课后总结和改进建议。
        """)
        desc.setWordWrap(True)
        desc.setStyleSheet("""
            background-color: #252545;
            border: 1px solid #4a4a7a;
            border-radius: 12px;
            padding: 20px;
            font-size: 14px;
            color: #B0B0D0;
            line-height: 1.6;
        """)
        layout.addWidget(desc)

        status_label = QtWidgets.QLabel()
        if self.ai_coach.config.enabled:
            status_label.setText("✓ AI教练 当前已启用")
            status_label.setStyleSheet("color: #10B981; font-size: 15px; font-weight: bold;")
        else:
            status_label.setText("✗ AI教练 当前已禁用")
            status_label.setStyleSheet("color: #EF4444; font-size: 15px; font-weight: bold;")
        layout.addWidget(status_label)

        info = QtWidgets.QLabel("""
            💡 使用提示：
            • 确保摄像头清晰可见
            • 专注于标准动作的模仿
            • 注意AI教练的实时建议
            • 练习结束后会收到综合总结
        """)
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 13px; color: #888888; line-height: 1.8;")
        layout.addWidget(info)

        layout.addStretch()

        btn_layout = QtWidgets.QHBoxLayout()

        def on_enable_changed(state):
            if state == 2:
                status_label.setText("✓ AI教练 当前已启用")
                status_label.setStyleSheet("color: #10B981; font-size: 15px; font-weight: bold;")
            else:
                status_label.setText("✗ AI教练 当前已禁用")
                status_label.setStyleSheet("color: #EF4444; font-size: 15px; font-weight: bold;")

        enable_check.stateChanged.connect(on_enable_changed)

        save_btn = QtWidgets.QPushButton("保存设置")
        close_btn = QtWidgets.QPushButton("关闭")

        def save_settings():
            self.ai_coach.config.enabled = enable_check.isChecked()
            self.ai_feedback_enabled = enable_check.isChecked()
            dialog.accept()

        save_btn.clicked.connect(save_settings)
        close_btn.clicked.connect(dialog.reject)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

    def generate_session_summary_sync(self):
        """生成课后AI总结（同步版本，用于在对话框中显示）"""
        if not self.ai_coach.config.is_configured():
            return ""

        duration = 0
        if self._session_start_time:
            duration = time.time() - self._session_start_time

        session_data = {
            "total_frames": len(self.scoreChart.scores) if hasattr(self.scoreChart, 'scores') else 0,
            "avg_score": sum(self.scoreChart.scores) / len(self.scoreChart.scores) if self.scoreChart.scores else 0,
            "bad_frames_count": len(self.badFrames),
            "duration": duration
        }

        try:
            summary = self.ai_coach.generate_session_summary(session_data)
            if summary:
                self.coaching_history.set_summary(summary)
                return summary
        except Exception as e:
            print(f"[AI Coach] Summary generation failed: {e}")
        
        return ""

    def generate_session_summary(self):
        """生成课后AI总结（异步版本）"""
        if not self.ai_coach.config.is_configured():
            return

        duration = 0
        if self._session_start_time:
            duration = time.time() - self._session_start_time

        session_data = {
            "total_frames": len(self.scoreChart.scores) if hasattr(self.scoreChart, 'scores') else 0,
            "avg_score": sum(self.scoreChart.scores) / len(self.scoreChart.scores) if self.scoreChart.scores else 0,
            "bad_frames_count": len(self.badFrames),
            "duration": duration
        }

        def do_generate():
            try:
                summary = self.ai_coach.generate_session_summary(session_data)
                if summary:
                    self.coaching_history.set_summary(summary)
                    QtCore.QMetaObject.invokeMethod(
                        self, "_show_session_summary",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, summary)
                    )
            except Exception as e:
                print(f"[AI Coach] Summary generation failed: {e}")

        threading.Thread(target=do_generate, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _show_session_summary(self, summary):
        """显示课后总结对话框"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("🤖 AI Coach - Session Summary")
        dialog.setMinimumSize(600, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
        """)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(20)

        title = QtWidgets.QLabel("📊 Practice Session Summary")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #A78BFA;")
        layout.addWidget(title)

        summary_label = QtWidgets.QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("""
            background-color: #252545;
            border: 1px solid #4a4a7a;
            border-radius: 12px;
            padding: 20px;
            font-size: 15px;
            color: #E0E0E0;
            line-height: 1.6;
        """)
        layout.addWidget(summary_label)

        layout.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()
