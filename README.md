# Dance Pose Scoring System (舞蹈姿态评分系统)

A real-time dance pose analysis and scoring application using Computer Vision. This system compares a user's movements against a reference video, providing instant feedback on pose accuracy, timing, and limb positioning.

基于计算机视觉的实时舞蹈姿态评分系统。该系统通过对比用户动作与参考视频，提供关于姿态准确度、节奏和肢体位置的实时反馈。

![Demo](https://via.placeholder.com/800x450?text=Dance+Pose+Scoring+System+Demo)

## ✨ Key Features (核心功能)

*   **Real-time Pose Estimation**: Uses MediaPipe to detect 33 body landmarks in real-time with high accuracy.
    *   **实时姿态估计**：利用 MediaPipe 实时检测 33 个身体关键点。
*   **Joint Angle Scoring**: Evaluates the accuracy of 8 key joint angles (shoulders, elbows, hips, knees) with confidence-weighted scoring.
    *   **关节角度评分**：评估 8 个核心关节（肩、肘、胯、膝）的角度准确度，支持置信度加权。
*   **Dynamic Time Warping (DTW)**: Advanced timing alignment that tolerates rhythm variations between user and reference.
    *   **动态时间规整 (DTW)**：高级时间对齐算法，容忍用户与参考视频之间的节奏差异。
*   **Audio Alignment**: Automatic synchronization using Chroma feature cross-correlation for precise audio-visual matching.
    *   **音频对齐**：基于 Chroma 特征互相关的自动同步，实现精确的音视频匹配。
*   **AI Coaching**: Intelligent feedback generation powered by DeepSeek API, providing personalized improvement suggestions.
    *   **AI 智能教练**：基于 DeepSeek API 的智能反馈生成，提供个性化改进建议。
*   **Ghost Mode**: Overlay skeleton visualization for direct visual comparison with reference pose.
    *   **幽灵模式**：骨架叠加显示，实现与参考姿态的直接视觉对比。
*   **Real-time Feedback**:
    *   Live score display (0-100)
    *   Timing hints ("Speed up!", "Slow down!", "Perfect")
    *   Score trend chart
    *   **实时反馈**：实时分数显示、节奏提示、评分趋势图表
*   **Performance Analysis**:
    *   Bad frame extraction with screenshots
    *   Session summary generation
    *   **表现分析**：低分帧提取、会话总结报告

## 🛠️ Project Structure (项目结构)

```text
dancepose_project/
├── gui_app.py                 # Entry point for Desktop GUI application
├── backend_app.py             # FastAPI backend (for web integration)
├── requirements.txt           # Dependencies
├── src/
│   ├── core/                  # Core logic modules
│   │   ├── scoring.py         # Joint angle scoring algorithm
│   │   ├── pose_worker.py     # Background worker for pose estimation
│   │   ├── video_reader.py    # Threaded video/camera reading
│   │   ├── dtw_utils.py       # DTW timing alignment
│   │   ├── audio_aligner.py   # Audio-based video synchronization
│   │   ├── ai_coach.py        # AI-powered feedback generation
│   │   └── ghost_overlay.py   # Ghost mode skeleton rendering
│   ├── ui/                    # UI Components
│   │   ├── main_window.py     # Main GUI window
│   │   ├── components.py      # Custom widgets (Charts, Video Panels)
│   │   └── ghost_mode_window.py # Ghost mode window
│   └── utils/                 # Utilities
│       ├── geometry.py        # Geometric calculations (angle computation)
│       └── model_loader.py    # MediaPipe model management
├── reports/                   # Project documentation
│   ├── Algorithm_Implementation_Report.md    # Algorithm documentation
│   ├── Algorithm_Implementation_Report_EN.md # English version
│   ├── Functional_Module_Design.md           # Module design docs
│   ├── Project_Progress.md                   # Progress reports
│   └── Optimization_Plan.md                  # Optimization strategies
├── tests/                     # Unit tests
│   ├── test_confidence.py     # Confidence filtering tests
│   ├── test_dtw_improved.py   # DTW algorithm tests
│   └── test_dtw_scoring.py    # DTW scoring tests
└── example/                   # Example application (DanceBattle)
    └── DanceBattle/           # Alternative implementation
```

## 🚀 Getting Started (快速开始)

### Prerequisites (前置要求)

*   Python 3.8+
*   Webcam (for real-time scoring)
*   Optional: GPU acceleration recommended for optimal performance

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

### Basic Workflow

1.  **Load Videos**:
    *   Click **"加载参考视频" (Load Ref)** to select the standard dance video.
    *   Click **"开启摄像头" (Start Cam)** for real-time webcam input, OR **"加载用户视频" (Load User)** to compare two video files.
    *   **加载视频**：选择参考视频，开启摄像头或加载用户视频。

2.  **Start Practice**:
    *   Click **"播放" (Play)**.
    *   Wait for the **3-second countdown**.
    *   Start dancing when you see **"GO!"**.
    *   **开始练习**：点击播放，等待3秒倒计时，看到"GO!"时开始跳舞。

3.  **View Real-time Feedback**:
    *   Watch the **Score** (0-100) and **Timing Hints** in real-time.
    *   Check the **Score Chart** for your performance trend.
    *   Review **Bad Frames** list to see where you need improvement.
    *   **查看反馈**：实时查看分数、节奏提示和评分趋势图表。

4.  **AI Coaching**:
    *   The AI coach automatically provides personalized feedback based on your performance.
    *   View suggestions for body angles, movement amplitude, and rhythm.
    *   **AI 教练**：自动提供个性化改进建议。

5.  **Export Report**:
    *   Click **"导出PDF报告"** to save your session summary with detailed analysis.
    *   **导出报告**：保存会话总结报告。

### Advanced Features

**Audio Alignment**:
- Automatically synchronizes user and reference videos based on audio content
- Ensures precise timing matching even if videos start at different points

**Ghost Mode**:
- Overlays reference skeleton on user video for direct visual comparison
- Helps identify exact positioning differences

## ⚙️ Tech Stack (技术栈)

| Category | Technology | Description |
|----------|------------|-------------|
| Core | Python | Main programming language |
| Computer Vision | OpenCV | Video processing and image manipulation |
| AI/ML | Google MediaPipe | Real-time pose estimation |
| Audio Processing | Librosa | Audio feature extraction and alignment |
| GUI | PyQt5 | Desktop application framework |
| Backend | FastAPI | Web API integration (optional) |
| Numerical Computing | NumPy | Matrix operations and calculations |
| AI API | DeepSeek | Intelligent feedback generation |

## 🔧 Configuration (配置)

### AI Coach Settings

The AI coach can be configured via `src/core/ai_coach.py`:

```python
class AICoachConfig:
    def __init__(self):
        self.api_key = "your_api_key_here"  # DeepSeek API key
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-v4-pro"
        self.max_tokens = 1000
        self.temperature = 0.3  # Lower = more deterministic responses
```

### Performance Settings

Adjust these parameters in `src/core/pose_worker.py` for optimal performance:

- `history_buffer_size`: Size of DTW history window (default: 15 frames)
- `confidence_threshold`: Minimum confidence for landmark usage (default: 0.5)
- `frame_skip_interval`: Frame skipping for performance (default: 1)

## 📊 Algorithm Overview (算法概述)

### Joint Angle Scoring

The scoring algorithm evaluates 8 key joint angles:
- Shoulders (left/right)
- Elbows (left/right)
- Hips (left/right)
- Knees (left/right)

**Scoring Formula**:
```
Score = Σ(w_i × score_diff(angle_difference)) / Σ(w_i)
```

Where:
- `w_i`: Weight for each joint (hips/knees: 1.2, shoulders/elbows: 1.0)
- `score_diff(d)`: Linear decay from 100 to 0 over 66° difference

### Dynamic Time Warping (DTW)

DTW aligns user and reference sequences by finding the optimal path that minimizes distance between frames, allowing for rhythm variations.

### Audio Alignment

Uses Chroma feature cross-correlation to detect temporal offsets between videos, ensuring precise synchronization.

## 🧪 Testing (测试)

Run the test suite:

```bash
python -m pytest tests/ -v
```

Test coverage includes:
- Confidence filtering mechanisms
- DTW algorithm correctness
- Scoring accuracy validation

## 📈 Performance Metrics

| Metric | Result |
|--------|--------|
| Real-time FPS | 30 FPS (stable) |
| Latency | < 80ms |
| Scoring Accuracy | ± 3% error margin |
| Supported Input | Webcam, MP4, AVI, MOV |



## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.



---

**Version**: 2.0  
**Last Updated**: May 2026  
**Project Status**: Active Development
