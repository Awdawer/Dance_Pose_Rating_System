import os
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets

from src.utils.model_loader import ensure_model
from src.utils.geometry import POSE_CONNECTIONS
from src.core.pose_worker import PoseWorker
from src.core.video_reader import VideoReader
from src.ui.components import VideoPanel, ScoreChartWidget

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dance Pose Scoring (Python GUI)")
        self.resize(1200, 700)

        model_path = ensure_model()
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "错误", "模型文件缺失")
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

        self.userPanel = VideoPanel()
        self.refPanel = VideoPanel()
        self.scoreLabel = QtWidgets.QLabel("-- %")
        f = self.scoreLabel.font()
        f.setPointSize(28)
        f.setBold(True)
        self.scoreLabel.setFont(f)
        self.hintLabel = QtWidgets.QLabel("")
        f2 = self.hintLabel.font()
        f2.setPointSize(24)
        f2.setBold(True)
        self.hintLabel.setFont(f2)
        self.hintLabel.setStyleSheet("color: #FBBF24;")

        self._miss_count = 0
        self._ema_score = None
        self._ema_alpha = 0.3
        self._k_decay = 5.0
        self._gamma = 0.7
        self.detect_stride = 3
        self._tick_count = 0
        self.ts_ms = 0
        self.lastPercent = 0.0
        self.detect_stride_user = 2
        self.detect_stride_ref = 4
        self.lastUserLandmarks = []
        self.lastRefLandmarks = []
        self.lastDiffs = None
        
        self._countdown_val = 0
        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self.on_countdown_tick)

        self.btnLoadUser = QtWidgets.QPushButton("加载用户视频")
        self.btnLoadRef = QtWidgets.QPushButton("加载参考视频")
        self.btnStartCam = QtWidgets.QPushButton("开启摄像头")
        self.btnStopCam = QtWidgets.QPushButton("关闭摄像头")
        self.camCombo = QtWidgets.QComboBox()
        self.btnRefreshCams = QtWidgets.QPushButton("刷新摄像头")
        self.btnPlay = QtWidgets.QPushButton("播放")
        self.btnPause = QtWidgets.QPushButton("暂停")
        self.btnStep = QtWidgets.QPushButton("逐帧")
        self.btnExtract = QtWidgets.QPushButton("提取坏帧")
        self.btnExport = QtWidgets.QPushButton("导出PDF报告")

        self.lastUserLandmarks = []
        self.lastRefLandmarks = []
        self.lastDiffs = None
        self.lastPercent = 0.0

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(QtWidgets.QLabel("分数："))
        topLayout.addWidget(self.scoreLabel)
        topLayout.addStretch(1)
        topLayout.addWidget(self.hintLabel)
        topLayout.addStretch(1)

        stageLayout = QtWidgets.QHBoxLayout()
        stageLayout.addWidget(self.userPanel, 1)
        stageLayout.addWidget(self.refPanel, 1)

        ctrlLayout = QtWidgets.QGridLayout()
        ctrlLayout.addWidget(self.btnLoadUser, 0, 0)
        ctrlLayout.addWidget(self.btnLoadRef, 0, 1)
        ctrlLayout.addWidget(self.btnStartCam, 0, 2)
        ctrlLayout.addWidget(self.btnStopCam, 0, 3)
        ctrlLayout.addWidget(self.camCombo, 0, 4)
        ctrlLayout.addWidget(self.btnRefreshCams, 0, 5)
        ctrlLayout.addWidget(self.btnPlay, 1, 0)
        ctrlLayout.addWidget(self.btnPause, 1, 1)
        ctrlLayout.addWidget(self.btnStep, 1, 2)
        ctrlLayout.addWidget(self.btnExtract, 1, 3)
        ctrlLayout.addWidget(self.btnExport, 1, 4)

        self.badList = QtWidgets.QListWidget()
        self.badList.setViewMode(QtWidgets.QListView.IconMode)
        self.badList.setIconSize(QtCore.QSize(160, 90))
        self.badList.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.badList.setUniformItemSizes(True)
        self.badList.setMaximumHeight(130) # Limit height to make room for chart

        self.scoreChart = ScoreChartWidget()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(topLayout)
        layout.addLayout(stageLayout)
        layout.addLayout(ctrlLayout)
        layout.addWidget(QtWidgets.QLabel("坏帧预览："))
        layout.addWidget(self.badList)
        layout.addWidget(QtWidgets.QLabel("分数趋势："))
        layout.addWidget(self.scoreChart, 1) # Give chart remaining space

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()
        self._timer_interval_ms = 33  # cache for jitter control
        self.update_timer_interval()

        self.btnLoadUser.clicked.connect(self.load_user_video)
        self.btnLoadRef.clicked.connect(self.load_ref_video)
        self.btnStartCam.clicked.connect(self.start_cam)
        self.btnStopCam.clicked.connect(self.stop_cam)
        self.btnRefreshCams.clicked.connect(self.enumerate_cams)
        self.btnPlay.clicked.connect(self.play)
        self.btnPause.clicked.connect(self.pause)
        self.btnStep.clicked.connect(self.step)
        self.btnExtract.clicked.connect(self.extract_bad)
        self.btnExport.clicked.connect(self.export_pdf)
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
            self.camCombo.addItem("无可用摄像头", None)

    def load_user_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择用户视频", "", "Video Files (*.mp4 *.avi *.mov)")
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

    def load_ref_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择参考视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        if self.refReader:
            self.refReader.stop()
        self.refReader = VideoReader(path)
        self.refReader.start()
        # Pause initially
        self.refReader.pause()
        self.playing = False
        self.update_timer_interval()

    def start_cam(self):
        self.useCam = True
        if self.userReader:
            self.userReader.stop()
        idx = self.camCombo.currentData()
        if idx is None:
            idx = 0
        self.userReader = VideoReader(int(idx))
        self.userReader.start()
        
        self.playing = True
        self.update_timer_interval()

    def stop_cam(self):
        self.useCam = False
        if self.userReader:
            self.userReader.stop()
        self.userReader = None
        self.playing = False
        self.update_timer_interval()

    def play(self):
        if self.playing: return
        
        # Reset chart
        self.scoreChart.reset()
        
        # Ensure paused while resetting/counting down
        if self.userReader: 
            self.userReader.pause()
            self.userReader.reset()
        if self.refReader: 
            self.refReader.pause()
            self.refReader.reset()
        
        # Start countdown
        self._countdown_val = 3
        self.scoreLabel.setText(str(self._countdown_val))
        self.countdown_timer.start()
        
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

    def pause(self):
        self.playing = False
        self.countdown_timer.stop()
        if self.userReader: self.userReader.pause()
        if self.refReader: self.refReader.pause()

    def step(self):
        self.playing = False
        self.read_and_process(step=True)

    def on_tick(self):
        if self.playing:
            self.read_and_process(step=False)

    def on_results_ready(self, u_lms, r_lms, diffs, percent, timing_hint):
        if len(u_lms):
            self.lastUserLandmarks = u_lms
        if len(r_lms):
            self.lastRefLandmarks = r_lms
        if len(u_lms) and len(r_lms):
            self.lastDiffs = diffs
            if self._ema_score is None:
                self._ema_score = percent
            else:
                self._ema_score = self._ema_alpha * percent + (1.0 - self._ema_alpha) * self._ema_score
            self.lastPercent = self._ema_score
            self.scoreLabel.setText(f"{int(round(self.lastPercent))} %")
            
            # Update chart
            self.scoreChart.add_score(self.lastPercent)
            
            # Show timing hint
            if timing_hint:
                self.hintLabel.setText(timing_hint)
                if "快点" in timing_hint:
                    self.hintLabel.setStyleSheet("color: #F87171;") # Red
                elif "慢点" in timing_hint:
                    self.hintLabel.setStyleSheet("color: #60A5FA;") # Blue
                else:
                    self.hintLabel.setStyleSheet("color: #34D399;") # Green
            else:
                self.hintLabel.setText("")
                
            self._miss_count = 0
        else:
            self._miss_count += 1
            # Keep last skeleton/diffs to avoid flicker; only clear after sustained misses
            if self._miss_count > 30:
                self.lastPercent = 0.0
                self.lastDiffs = None
                self.scoreLabel.setText("0 %")
                self.hintLabel.setText("")
                self._ema_score = None

    def read_and_process(self, step=False):
        user_frame = None
        ref_frame = None
        
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
        MAX_DISP_W = 640
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
                        cv2.putText(canvas, f"{int(round(d))}°", (pts[i][0]+6, pts[i][1]-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        qimg = self.frame_to_qimage(canvas)
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

    def export_pdf(self):
        try:
            from reportlab.pdfgen import canvas as pdfcanvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
        except Exception:
            QtWidgets.QMessageBox.warning(self, "提示", "未安装 reportlab，无法导出PDF。请先 pip install reportlab")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存报告", "pose_report.pdf", "PDF (*.pdf)")
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
        QtWidgets.QMessageBox.information(self, "完成", "PDF报告已导出")
