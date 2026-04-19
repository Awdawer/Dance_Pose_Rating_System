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
        
        # Apply Dark Theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 16px; /* Base font size increased */
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 8px; /* Slightly rounder */
                padding: 12px 24px; /* Much larger padding */
                color: #FFFFFF;
                font-weight: bold;
                font-size: 16px;
                min-height: 40px; /* Ensure button height */
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00C6FF;
            }
            QPushButton:pressed {
                background-color: #222222;
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
            QListWidget {
                background-color: #252525;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QListWidget::item {
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #3A3A3A;
                border: 1px solid #00C6FF;
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

        self.userPanel = VideoPanel()
        self.refPanel = VideoPanel()
        
        # 实时评分标签
        self.scoreLabel = QtWidgets.QLabel("-- %")
        f = self.scoreLabel.font()
        f.setPointSize(18)
        f.setBold(True)
        self.scoreLabel.setFont(f)
        self.scoreLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.scoreLabel.setStyleSheet("color: #10B981; background-color: #000000; border-radius: 20px; padding: 20px; font-size: 18pt;")
        
        # DTW对齐评分标签
        self.dtwScoreLabel = QtWidgets.QLabel("DTW: -- %")
        f_dtw = self.dtwScoreLabel.font()
        f_dtw.setPointSize(18)
        f_dtw.setBold(True)
        self.dtwScoreLabel.setFont(f_dtw)
        self.dtwScoreLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.dtwScoreLabel.setStyleSheet("color: #60A5FA; background-color: #000000; border-radius: 20px; padding: 20px; font-size: 18pt;")

        self.hintLabel = QtWidgets.QLabel("")
        f2 = self.hintLabel.font()
        f2.setPointSize(24)
        f2.setBold(True)
        self.hintLabel.setFont(f2)
        self.hintLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.hintLabel.setStyleSheet("color: #FBBF24; background-color: #222; border-radius: 15px; padding: 10px; font-size: 24pt;")

        # Alignment status label
        self.alignStatusLabel = QtWidgets.QLabel("Not Aligned")
        self.alignStatusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.alignStatusLabel.setStyleSheet("color: #888888; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")

        # Frame counter label for debugging alignment
        self.frameLabel = QtWidgets.QLabel("User: 0 | Ref: 0")
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.frameLabel.setStyleSheet("color: #888888; background-color: #222; border-radius: 5px; padding: 3px; font-size: 10pt;")

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
        topLayout.addWidget(QtWidgets.QLabel("Real-time:"))
        topLayout.addWidget(self.scoreLabel)
        topLayout.addSpacing(20)
        topLayout.addWidget(QtWidgets.QLabel("DTW:"))
        topLayout.addWidget(self.dtwScoreLabel)
        topLayout.addSpacing(20)
        topLayout.addWidget(self.alignStatusLabel)
        topLayout.addSpacing(10)
        topLayout.addWidget(self.frameLabel)
        topLayout.addStretch(1)
        topLayout.addWidget(self.hintLabel)
        topLayout.addStretch(1)

        stageLayout = QtWidgets.QHBoxLayout()
        stageLayout.setSpacing(20)
        stageLayout.addWidget(self.userPanel, 1)
        stageLayout.addWidget(self.refPanel, 1)
        
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
        ctrlLayout.setSpacing(15)

        # Source Controls
        sourceLayout = QtWidgets.QHBoxLayout()
        sourceLayout.addWidget(self.btnLoadUser)
        sourceLayout.addWidget(self.btnLoadRef)
        sourceLayout.addWidget(self.camCombo)
        sourceLayout.addWidget(self.btnCamToggle)
        sourceLayout.addStretch(1) # Push to left
        
        # Playback Controls - Make them BIG
        playLayout = QtWidgets.QHBoxLayout()
        playLayout.addStretch(1)
        
        # Style Play/Pause buttons to be prominent
        self.btnPlayReset.setMinimumWidth(120)
        self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;") # Green
        
        self.btnPauseResume.setMinimumWidth(120)
        self.btnPauseResume.setStyleSheet("background-color: #F59E0B; font-size: 18px;") # Orange
        
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
        # Remove fixed max height, let layout manage
        # self.badList.setMaximumHeight(160) 

        self.scoreChart = ScoreChartWidget()
        self.scoreChart.setMinimumHeight(120) # Ensure it has some height

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20) # Add margin around window
        layout.setSpacing(15) # Increase spacing between elements
        
        # 1. Top Bar (Score) - Fixed height ~100px
        layout.addLayout(topLayout)
        
        # 2. Video Stage - 60% of space, try to keep 16:9
        # To enforce ratio, we might need to be careful.
        # But simple stretch factor is easiest.
        layout.addLayout(stageLayout, 6) 
        
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
            self.alignStatusLabel.setStyleSheet("color: #888888; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
            print("[Alignment] Required libraries not available")
            return
        
        self.alignStatusLabel.setText("Aligning...")
        self.alignStatusLabel.setStyleSheet("color: #FBBF24; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        
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
        self.alignment_offset = time_offset
        self.alignment_frame_offset = frame_offset
        self.is_aligned = True
        
        # Update UI
        if abs(time_offset) < 0.1:
            self.alignStatusLabel.setText("Aligned: In Sync")
            self.alignStatusLabel.setStyleSheet("color: #10B981; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        else:
            direction = "User ahead" if time_offset < 0 else "User behind"
            self.alignStatusLabel.setText(f"Aligned: {direction} {abs(time_offset):.1f}s")
            self.alignStatusLabel.setStyleSheet("color: #60A5FA; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        
        # Apply alignment to video readers
        self.apply_alignment()
    
    @QtCore.pyqtSlot()
    def on_alignment_failed(self):
        self.alignStatusLabel.setText("Alignment Failed")
        self.alignStatusLabel.setStyleSheet("color: #EF4444; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        self.is_aligned = False

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
        self.pause() # Stop threads
        
        # 停止音频
        self.stop_audio_playback()
        
        # Calculate final score
        avg_score = 0
        if self.scoreChart.scores:
            avg_score = sum(self.scoreChart.scores) / len(self.scoreChart.scores)
            
        # Show Summary Dialog
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Session Ended")
        msg.setText(f"<h3>Final Score: {int(avg_score)} pts</h3>")
        msg.setInformativeText(f"Recorded {len(self.scoreChart.scores)} score points.\n\nGreat Job!")
        msg.setStyleSheet("QLabel{min-width: 300px; font-size: 16px;} QPushButton{ font-size: 14px; }")
        
        # Force OK button text
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        ok_btn = msg.button(QtWidgets.QMessageBox.Ok)
        if ok_btn:
            ok_btn.setText("OK")

        msg.exec_()
        
        # Reset UI state to "Ready to Start"
        self.btnPlayReset.setText("Start")
        self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;") # Green
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
            self.btnCamToggle.setStyleSheet("background-color: #EF4444; border-color: #EF4444;") # Red

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
        self.alignStatusLabel.setStyleSheet("color: #888888; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        
        self.playing = True
        self.update_timer_interval()
        # Ensure chart is ready?
        
    def stop_cam(self):
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        self.userReader = None
        self.playing = False
        self.update_timer_interval()

    def play_reset(self):
        # This button now acts as Start / Stop
        if self.playing or (self.countdown_timer.isActive()):
            # User clicked "Stop"
            self.on_playback_finished() # Show score and stop
            self.btnPlayReset.setText("Start")
            self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;") # Green
        else:
            # User clicked "Start" -> Reset and Start
            self.scoreChart.reset()
            self.btnPlayReset.setText("Stop")
            self.btnPlayReset.setStyleSheet("background-color: #EF4444; font-size: 18px;") # Red
            
            # Reset readers
            if self.userReader: 
                self.userReader.pause()
                self.userReader.reset()
            if self.refReader: 
                self.refReader.pause()
                self.refReader.reset()
            
            # 重置音频
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
        # 播放音频
        self.start_audio_playback()

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
        # 暂停音频
        self.stop_audio_playback()

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
            self.scoreLabel.setText(f"{int(round(self.lastPercent))} %")
            
            # 2. DTW评分EMA平滑
            if self._ema_dtw_score is None:
                self._ema_dtw_score = dtw_percent
            else:
                self._ema_dtw_score = self._ema_alpha * dtw_percent + (1.0 - self._ema_alpha) * self._ema_dtw_score
            self.lastDtwPercent = self._ema_dtw_score
            self.dtwScoreLabel.setText(f"{int(round(self.lastDtwPercent))} %")
            
            # 3. 更新双折线图
            self.scoreChart.add_scores(self.lastPercent, self.lastDtwPercent)
            
            # Timing hint disabled
            self.hintLabel.setText("")
                
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
            self.badList.addItem(item)

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
        self.alignStatusLabel.setStyleSheet("color: #888888; background-color: #333; border-radius: 10px; padding: 5px; font-size: 12pt;")
        
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
        self.btnPlayReset.setStyleSheet("background-color: #10B981; font-size: 18px;")
        self.btnPauseResume.setText("Pause")
        self.btnPauseResume.setEnabled(False)
        
        # 11. 清空视频面板
        self.userPanel.clear()
        self.refPanel.clear()
        
        print("[Clear] All cache data cleared. System reset to initial state.")

    def start_audio_playback(self):
        """使用ffpyplayer开始播放音频"""
        if not FFPYPLAYER_AVAILABLE or not self.audio_path:
            return
        
        # 如果正在播放，先停止
        self.stop_audio_playback()
        
        # 计算音频起始时间（基于对齐偏移）
        # 如果用户视频跳过了帧，音频也需要从对应时间开始
        audio_start_time = 0.0
        if self.is_aligned:
            if self.alignment_frame_offset > 0:
                # 用户视频跳过了帧，音频从对应时间开始
                audio_start_time = self.alignment_offset
                print(f"[Audio] Starting from {audio_start_time:.2f}s (user video skipped frames)")
            # 如果参考视频跳过了帧，音频从头播放（因为音频来自参考视频）
        
        def play_audio():
            try:
                print(f"[Audio] Starting audio playback: {self.audio_path}")
                
                # 使用 ff_opts 设置起始时间
                ff_opts = {}
                if audio_start_time > 0:
                    ff_opts['ss'] = audio_start_time  # 起始时间（秒）
                
                self.audioPlayer = MediaPlayer(self.audio_path, ff_opts=ff_opts)
                self.audio_playing = True
                
                # 同步计数器
                sync_counter = 0
                
                while self.audio_playing:
                    frame, val = self.audioPlayer.get_frame()
                    if val == 'eof':
                        break
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # 每50帧同步一次音频和视频（约每1.5秒）
                    sync_counter += 1
                    if sync_counter >= 50:
                        sync_counter = 0
                        self._sync_audio_video()
                
                self.audioPlayer.close_player()
                print("[Audio] Playback finished")
            except Exception as e:
                print(f"[Audio] Playback error: {e}")
        
        # 在新线程中播放音频
        self.audio_thread = threading.Thread(target=play_audio, daemon=True)
        self.audio_thread.start()
    
    def _sync_audio_video(self):
        """同步音频和视频播放位置"""
        if not self.refReader or not self.audioPlayer:
            return
        
        try:
            # 获取参考视频当前播放时间
            video_time = self.refReader.get_current_frame() / self.refReader.fps
            
            # 获取音频当前播放时间（ffpyplayer的pts）
            # ffpyplayer 没有直接获取当前时间的方法，使用近似计算
            # 这里我们假设音频和视频起始时间相同，不做调整
            # 如果偏差太大，可以考虑重新seek音频
            
            # 简单方案：打印同步信息用于调试
            pass
            
        except Exception as e:
            print(f"[Audio Sync] Error: {e}")

    def stop_audio_playback(self):
        """停止音频播放"""
        self.audio_playing = False
        if self.audioPlayer:
            try:
                self.audioPlayer.close_player()
                self.audioPlayer = None
                print("[Audio] Stopped")
            except:
                pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)

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
