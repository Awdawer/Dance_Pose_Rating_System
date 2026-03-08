import threading
import time
import cv2
from PyQt5 import QtCore

class VideoReader(QtCore.QThread):
    """
    A dedicated thread for reading video frames asynchronously.
    Supports file playback (with FPS throttle) and camera capture (low latency).
    """
    def __init__(self, source):
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
            
            self.running = True
            last_time = time.time()

            while self.running:
                # Handle seek request safely in thread
                if self._seek_req >= 0:
                    if not self.is_cam and self.cap:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_req)
                        last_time = time.time() # Reset timer
                    self._seek_req = -1

                # Handle Pause (only for files)
                if self._is_paused and not self.is_cam:
                    time.sleep(0.01)
                    last_time = time.time() # Keep resetting last_time so resume is smooth
                    continue

                # Timing control for file playback
                if not self.is_cam:
                    now = time.time()
                    elapsed = now - last_time
                    wait = frame_interval - elapsed
                    if wait > 0:
                        time.sleep(wait)
                    last_time = time.time()

                ret, frame = self.cap.read()
                if ret:
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
                    if not self.is_cam: # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        last_time = time.time() # Reset timer on loop
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

    def reset(self):
        # Thread-safe seek request
        self._seek_req = 0

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False
