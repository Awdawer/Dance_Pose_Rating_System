import threading
import time
import cv2
from PyQt5 import QtCore

class VideoReader(QtCore.QThread):
    """
    A dedicated thread for reading video frames asynchronously.
    Supports file playback (with FPS throttle) and camera capture (low latency).
    """
    def __init__(self, source, start_frame=0):
        super().__init__()
        self.source = source
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.new_frame_available = False
        self.fps = 30.0
        self.is_cam = False
        self._seek_req = -1 # -1 means no seek
        self._is_paused = False # For file playback control
        self._start_frame = start_frame # Frame to start playback from (for audio alignment)

    finished = QtCore.pyqtSignal() # Signal when video ends

    def run(self):
        try:
            if isinstance(self.source, int):
                # Camera init might be slow, do it in thread
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 640x480 is usually fastest
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    # Try to disable buffering
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.is_cam = True
            else:
                self.cap = cv2.VideoCapture(self.source)
                self.is_cam = False
            
            if not self.cap or not self.cap.isOpened():
                print(f"Failed to open source: {self.source}")
                return

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self.fps <= 0: self.fps = 30.0
            frame_interval = 1.0 / self.fps
            
            # Apply start frame offset for audio alignment
            if self._start_frame > 0 and not self.is_cam:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
                actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(f"[VideoReader] Requested start frame {self._start_frame}, actual position: {actual_pos}")
            
            # Read first frame before entering main loop
            ret, frame = self.cap.read()
            if ret:
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(f"[VideoReader] First frame read, current position: {current_pos}")
                with self.lock:
                    self.frame = frame
                    self.new_frame_available = True
            
            self.running = True
            start_time = time.time()  # 记录开始时间
            frame_count = 0  # 已播放帧数（相对于起始帧）
            paused_time = 0.0  # 累计暂停时间

            while self.running:
                # Handle seek request safely in thread
                if self._seek_req >= 0:
                    if not self.is_cam and self.cap:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_req)
                        print(f"[VideoReader] Seeked to frame {self._seek_req}")
                        start_time = time.time()  # 重置开始时间
                        frame_count = self._seek_req - self._start_frame  # 重置帧计数
                        paused_time = 0.0
                    self._seek_req = -1

                # Handle Pause (only for files)
                if self._is_paused and not self.is_cam:
                    pause_start = time.time()
                    time.sleep(0.01)
                    # 累加暂停时间
                    paused_time += time.time() - pause_start
                    continue

                # Timing control for file playback - 使用绝对时间控制
                if not self.is_cam:
                    # 计算应该播放到的时间点（减去暂停时间）
                    expected_time = start_time + (frame_count * frame_interval) + paused_time
                    now = time.time()
                    wait = expected_time - now
                    if wait > 0:
                        time.sleep(wait)
                    # 如果落后太多，跳过一些帧来追赶
                    elif wait < -0.5:  # 落后超过0.5秒
                        skip_frames = int(-wait / frame_interval)
                        for _ in range(min(skip_frames, 10)):  # 最多跳过10帧
                            ret, _ = self.cap.read()
                            if not ret:
                                break
                            frame_count += 1
                        print(f"[VideoReader] Skipped {min(skip_frames, 10)} frames to catch up")

                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    if self.is_cam:
                        frame = cv2.flip(frame, 1) # Flip horizontally for mirror effect
                    with self.lock:
                        self.frame = frame
                        self.new_frame_available = True
                    
                    # If camera, try to clear buffer by reading more if available
                    # But doing this aggressively might reduce FPS. 
                    # Usually setting BUFFERSIZE=1 is enough.
                    pass
                else:
                    if not self.is_cam: # End of video
                        # Stop playback, don't loop
                        self.running = False
                        self.finished.emit()
                        break
                    else:
                        # Camera disconnect?
                        time.sleep(0.1)
                
        except Exception as e:
            print(f"VideoReader error: {e}")
        finally:
            if self.cap:
                self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def get_frame(self):
        with self.lock:
            if self.new_frame_available:
                self.new_frame_available = False
                return True, self.frame
            return False, self.frame

    def get_time_str(self):
        if not self.cap: return "0.00s"
        try:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return f"{pos/self.fps:.2f}s"
        except:
            return "0.00s"

    def get_current_frame(self):
        """Get current frame number (accounting for start_frame offset)"""
        if not self.cap:
            return 0
        try:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return int(pos)
        except:
            return 0
    
    def get_playback_time(self):
        """Get current playback time in seconds"""
        if not self.cap or self.fps <= 0:
            return 0.0
        try:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return pos / self.fps
        except:
            return 0.0

    def reset(self):
        # Thread-safe seek request - reset to start_frame for audio alignment
        print(f"[VideoReader] reset() called, seeking to frame {self._start_frame}")
        self._seek_req = self._start_frame

    def set_start_frame(self, frame):
        self._start_frame = frame

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False
