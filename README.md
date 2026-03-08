# Dance Pose Scoring System (舞蹈姿态评分系统)

A real-time dance pose analysis and scoring application using Computer Vision. This system compares a user's movements against a reference video, providing instant feedback on pose accuracy, timing, and limb positioning.

基于计算机视觉的实时舞蹈姿态评分系统。该系统通过对比用户动作与参考视频，提供关于姿态准确度、节奏和肢体位置的实时反馈。

![Demo](https://via.placeholder.com/800x450?text=Dance+Pose+Scoring+System+Demo)

## ✨ Key Features (核心功能)

*   **Real-time Pose Estimation**: Uses MediaPipe to detect 33 body landmarks in real-time.
    *   **实时姿态估计**：利用 MediaPipe 实时检测 33 个身体关键点。
*   **Dual-Criteria Scoring** (双重评分机制):
    *   **Angle Score (60%)**: Evaluates the accuracy of joint angles (shoulders, elbows, hips, knees).
    *   **Procrustes Shape Score (40%)**: Evaluates the overall body shape similarity.
*   **Timing Feedback** (节奏反馈):
    *   Real-time "Too Fast" / "Too Slow" hints to help you stay on beat.
    *   实时提示“太快”或“太慢”，辅助卡点。
*   **Smart Playback Control** (智能播放控制):
    *   **3-Second Countdown**: Synchronized start for both user and reference videos.
    *   **Auto-Mirroring**: Camera feed is horizontally flipped for a natural "mirror" experience.
*   **Performance Analysis** (表现分析):
    *   **Real-time Score Chart**: Live plotting of your score trend.
    *   **Bad Frame Extraction**: Automatically captures and lists moments with low scores.
    *   **PDF Report**: Export a detailed report with screenshots of errors.

## 🛠️ Project Structure (项目结构)

The project has been refactored for maintainability:

```text
dancepose_project/
├── gui_app.py              # Entry point for the Desktop GUI application
├── backend_app.py          # FastAPI backend (for web integration)
├── src/
│   ├── core/               # Core logic
│   │   ├── scoring.py      # Scoring algorithms (Angle + Procrustes)
│   │   ├── pose_worker.py  # Background worker for AI processing
│   │   └── video_reader.py # Threaded video/camera reading
│   ├── ui/                 # UI Components
│   │   ├── main_window.py  # Main GUI window
│   │   └── components.py   # Custom widgets (Charts, Video Panels)
│   └── utils/              # Utilities
│       ├── geometry.py     # Geometric calculations
│       └── model_loader.py # Model management
├── reports/                # Project progress reports
└── models/                 # MediaPipe model files
```

## 🚀 Getting Started (快速开始)

### Prerequisites (前置要求)

*   Python 3.8+
*   Webcam (for real-time scoring)

### Installation (安装)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Awdawer/Dance_Pose_Rating_System.git
    cd Dance_Pose_Rating_System
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the GUI Application**:
    ```bash
    python gui_app.py
    ```

## 📖 Usage Guide (使用指南)

1.  **Load Videos**:
    *   Click **"加载参考视频" (Load Ref)** to select the standard dance video.
    *   Click **"开启摄像头" (Start Cam)** to use your webcam, OR **"加载用户视频" (Load User)** to compare two video files.
2.  **Start Practice**:
    *   Click **"播放" (Play)**.
    *   Wait for the **3-second countdown**.
    *   Start dancing when you see **"GO!"**.
3.  **View Feedback**:
    *   Watch the **Score** and **Timing Hints** (Too Fast/Slow) in real-time.
    *   Check the **Score Chart** at the bottom for your performance trend.
    *   Review **Bad Frames** in the list to see where you made mistakes.
4.  **Export**:
    *   Click **"导出PDF报告"** to save your session summary.

## ⚙️ Tech Stack (技术栈)

*   **Core**: Python, OpenCV, NumPy
*   **AI/ML**: Google MediaPipe
*   **GUI**: PyQt5
*   **Backend**: FastAPI (Optional)

## 📄 License

This project is licensed under the MIT License.
