# Computer Vision-Based Dance Pose Scoring and Auxiliary Training System - Graduation Project Progress Report

**Project Name**: Dance Pose Scoring and Training Assistant System
**Date**: 2026-03-08
**Author**: Zheng Jiayong

---

## Abstract

This project aims to design and implement an intelligent dance auxiliary training system based on computer vision. The system utilizes the MediaPipe framework for real-time human pose estimation and performs quantitative comparison between user actions and standard reference videos through a multi-dimensional scoring algorithm (combining weighted Euclidean distance with cosine similarity). The project achieves a complete closed loop from video capture, pose analysis, real-time scoring feedback to data visualization, addressing the pain point of lacking instant quantitative feedback in traditional dance teaching. This document details the system's design philosophy, key technical implementations, and iterative optimization processes.

---

## 1. Introduction

### 1.1 Research Background
With the popularization of national fitness, the demand for dance learning continues to grow. However, traditional one-on-one teaching is costly and difficult to provide all-weather motion correction. Deep learning-based human pose estimation technology makes low-cost, high-precision automated motion evaluation possible.

### 1.2 Project Objectives
*   **Real-time Performance**: Achieve no less than 30 FPS real-time pose detection and scoring on standard PC platforms.
*   **Accuracy**: Construct a dual scoring model incorporating angle and position features to accurately identify motion differences.
*   **Interactivity**: Design an ergonomic user interface providing intuitive visual and rhythm feedback.

---

## 2. Core Technology and Implementation (Methodology & Implementation)

### 2.1 System Architecture
The system adopts a modular design, mainly containing the following core modules:
*   **Data Acquisition Layer**: The `VideoReader` module employs multi-threaded asynchronous reading technology, supporting efficient acquisition from cameras (low-latency mode) and video files (frame-rate synchronization mode).
*   **Core Logic Layer**: 
    *   `PoseWorker`: Extracts 33 body keypoints based on the MediaPipe BlazePose model.
    *   `Scoring Engine`: A hybrid scoring engine combining Procrustes analysis (shape similarity) with joint angle calculation (geometric similarity).
*   **User Interface Layer**: Modern GUI built on PyQt5, integrating real-time video stream rendering, dynamic chart plotting, and interactive controls.

### 2.2 Key Algorithm Implementations

#### 2.2.1 Dual-Weighted Scoring Model

This system adopts a hybrid scoring strategy combining local geometric features with global shape features, ensuring scoring robustness through multi-dimensional weighted fusion.

**1. Joint Angle Scoring**

Let the set of human keypoints be $\mathcal{P} = \{ \mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_{33} \}$, where $\mathbf{p}_i \in \mathbb{R}^2$ represents the 2D coordinates of the $i$-th keypoint. For a joint composed of keypoints $A, B, C$ (where $B$ is the vertex), the angle $\theta$ can be calculated using the vector dot product formula:

$$
\theta = \arccos \left( \frac{(\mathbf{p}_A - \mathbf{p}_B) \cdot (\mathbf{p}_C - \mathbf{p}_B)}{\| \mathbf{p}_A - \mathbf{p}_B \|_2 \cdot \| \mathbf{p}_C - \mathbf{p}_B \|_2} \right) \tag{1}
$$

The system selects $N=8$ groups of core joints (left/right shoulders, elbows, hips, knees), defining the angle score for the $i$-th joint as $S_{angle}^{(i)}$. The total angle score $S_{angle}$ adopts a weighted sum form:

$$
S_{angle} = \frac{\sum_{i=1}^{N} w_i \cdot \max\left(0, 100 - \alpha \cdot |\theta_{u,i} - \theta_{r,i}|\right)}{\sum_{i=1}^{N} w_i} \tag{2}
$$

Where:
*   $\theta_{u,i}, \theta_{r,i}$ represent the angles (in degrees) of the $i$-th joint for the user and reference video, respectively.
*   $w_i$ is the weight coefficient for the $i$-th joint, with higher weights assigned to limb extremities ($w_{elbow}=1.5, w_{knee}=1.5$).
*   $\alpha$ is the error penalty factor, taken as $\alpha = 1.5$ in this system.

**2. Weighted Procrustes Distance**

To evaluate overall pose shape similarity and eliminate errors caused by displacement and scaling, weighted Procrustes analysis is employed. Let the keypoint coordinate matrices for user and reference skeletons be $\mathbf{U}, \mathbf{R} \in \mathbb{R}^{M \times 2}$, respectively.

First, define the weighted centroid $\bar{\mathbf{p}}$ and weighted scale $s$ for normalization:

$$
\bar{\mathbf{p}} = \frac{\sum_{j=1}^{M} \omega_j \mathbf{p}_j}{\sum_{j=1}^{M} \omega_j}, \quad s = \sqrt{\frac{\sum_{j=1}^{M} \omega_j \|\mathbf{p}_j - \bar{\mathbf{p}}\|_2^2}{\sum_{j=1}^{M} \omega_j}} \tag{3}
$$

Normalized coordinates are $\tilde{\mathbf{p}}_j = (\mathbf{p}_j - \bar{\mathbf{p}}) / s$. The weighted Procrustes distance $D_{wp}$ is defined as finding the optimal rotation matrix $\mathbf{Q}$ that minimizes the weighted Euclidean distance:

$$
D_{wp} = \min_{\mathbf{Q}^T \mathbf{Q} = \mathbf{I}} \sqrt{ \sum_{j=1}^{M} \omega_j \| \tilde{\mathbf{p}}_{u,j} \mathbf{Q} - \tilde{\mathbf{p}}_{r,j} \|_2^2 } \tag{4}
$$

Where $\omega_j$ is the weight for the $j$-th keypoint, with torso weights set to 1.0 and limb extremities set to 2.0. The final shape score $S_{shape}$ adopts exponential decay mapping:

$$
S_{shape} = 100 \cdot \exp(-k \cdot D_{wp}) \tag{5}
$$

The decay coefficient is taken as $k=5.0$ in this system to enhance sensitivity to subtle shape differences.

**3. Final Fusion Score**

The total score $S_{total}$ is obtained through weighted linear fusion of the two components above:

$$
S_{total} = \lambda_1 S_{angle} + \lambda_2 S_{shape} \tag{6}
$$

This project sets $\lambda_1 = 0.6, \lambda_2 = 0.4$, indicating that joint angle accuracy has higher priority in dance movement evaluation.

#### 2.2.2 Real-time Rhythm Detection
*   **Sliding Time Window**: Maintains a historical action buffer of 30 frames (approximately 1 second).
*   **Simplified DTW Strategy**: Calculates temporal offset by comparing current user actions with historical reference action sequences.
    *   Offset > +4 frames: Judged as "Too Slow".
    *   Offset < -4 frames: Judged as "Too Fast".

---

## 3. Development Log

### 3.1 Phase I: Core Functionality
*   **Infrastructure Setup**: Completed MediaPipe model integration and PyQt5 basic interface construction.
*   **Preliminary Scoring Implementation**: Implemented an initial scoring algorithm based on Euclidean distance, but found it insensitive to limb extremity movements and significantly affected by camera distance.

### 3.2 Phase II: Algorithm Optimization
*   **Scoring Model Refactoring**: Introduced the aforementioned "Dual-Weighted Scoring Model", resolving robustness issues where "random movements could still achieve high scores".
*   **Multi-threading Refactoring**: Separated video reading logic from the main thread to a `VideoReader` sub-thread, completely resolving interface lag and "Not Responding" issues, achieving decoupling of UI rendering and data processing.

### 3.3 Phase III: UI/UX Enhancement
*   **Synchronized Playback Mechanism**: Implemented thread-safe control for `play/reset` and a **3-second Countdown** logic, ensuring millisecond-level synchronized startup of dual video streams.
*   **Visual Feedback Enhancement**:
    *   **Dark Theme**: Fully applied QSS stylesheets to create a professional-grade visual experience.
    *   **Adaptive Layout**: Designed a 16:9 golden ratio video container supporting Letterboxing, resolving stretching distortion issues for non-standard aspect ratio videos.
    *   **Dynamic Charts**: Developed a high-performance real-time line chart (`ScoreChartWidget`) based on `QPainter`, supporting neon light effect rendering.

### 3.4 Phase IV: Robustness & Engineering Improvements
*   **Confidence Filtering Mechanism**:
    *   **Problem**: During dance movements, limbs are often occluded by the body (e.g., hands-behind-back movements), causing MediaPipe to output erroneous coordinates, leading to score drops.
    *   **Solution**: 
        *   Modified underlying data structure to introduce keypoint visibility property.
        *   Developed dynamic weight allocation algorithm: In `Procrustes` shape analysis, automatically zero out weights for keypoints with low confidence (<0.5); in angle scoring, ignore the joint score if any constituent point is invisible.
    *   **Result**: Verified through unit testing, under severe occlusion conditions, scoring accuracy improved from 88.6% to 100% (after excluding interference items), significantly enhancing system stability under complex movements.
*   **Automated Testing System**:
    *   Established `tests/` directory, authored unit tests targeting geometric calculations and scoring logic, ensuring safety of algorithm iterations.

### 3.5 Phase V: DTW Algorithm Implementation & Optimization

This phase completed the core implementation of advanced rhythm analysis functionality, introducing the Dynamic Time Warping (DTW) algorithm to achieve intelligent matching for variable-speed movements and a dual-score system.

*   **DTW Dual-Score System Design and Implementation**:
    *   **Objective**: Address scoring accuracy issues when user and reference videos have motion delays, providing both real-time and DTW-aligned scoring perspectives.
    *   **Technical Solution Evolution**:
        *   **Initial Approach**: Attempted traditional DTW path analysis, maintaining two buffers (user history and reference history), calculating optimal alignment path through Sakoe-Chiba constraints.
        *   **Problem Discovery**: Since user history and reference history are updated synchronously (appending new frames simultaneously), both sequences always have the same length, causing DTW path analysis to always return `lag=0`, failing to detect actual motion delays.
        *   **Final Solution**: Adopted a simplified strategy—only maintain the reference video's history buffer (`ref_history`), and for each user frame, find the most similar frame in historical reference frames through Euclidean distance for scoring comparison.
    *   **Core Algorithm Logic**:
        ```
        Per-frame Processing Flow:
        1. Get current user frame (u_lm, u_angs) and current reference frame (r_lm, r_angs)
        2. Store current reference frame into history buffer ref_history
        3. Calculate real-time score: score(user current frame, reference current frame)
        4. Find frame in reference history most similar to user frame (best_r_lm, best_r_angs)
        5. Calculate DTW score: score(user current frame, historical best match frame)
        ```
    *   **Feature Vector Design**: Uses 8 key joint angles to construct feature vectors (left/right shoulders, elbows, hips, knees), measuring frame similarity through Euclidean distance.
    *   **UI Display**: Main interface simultaneously displays both scores (real-time and DTW), with dual line charts at the bottom (blue-real-time, orange-DTW) intuitively showing score differences.

*   **Iteration Optimization Record**:
    *   **Font Size Adjustment**: Based on user feedback, adjusted main interface score text size from 5x to 2.5x, balancing visual prominence with interface harmony.
    *   **Rhythm Hint Removal**: Removed "Too Fast/Too Slow" text prompts to avoid user distraction, expressing rhythm analysis results only through dual-score differences.
    *   **Scoring Tolerance Optimization**: Introduced tolerance parameters for scoring functions in same-video matching scenarios, ensuring stable high scores for identical movements.

*   **Implementation Results**:
    *   When user video has approximately 1-second motion delay relative to reference video, system correctly detects and outputs non-zero `lag` values in console (e.g., `lag=-15`).
    *   Real-time and DTW scores show significant differences: real-time score lower (comparing unsynchronized frames), DTW score higher (comparing most similar historical frames).
    *   The dual-score system provides users with a more comprehensive movement quality evaluation perspective.

## 4. Future Work

### 4.1 Planned Intelligence Expansion
This phase aims to break through existing rule-based scoring limitations through introduction of advanced algorithms and generative AI:
*   **~~Advanced Rhythm Analysis (DTW Algorithm)~~** ✅ **Completed**:
    *   ~~**Objective**: Address the inability of traditional sliding windows to handle variable-speed movements.~~
    *   ~~**Approach**: Introduce Dynamic Time Warping algorithm to calculate the optimal nonlinear alignment path between user and reference action sequences, precisely judging "rushing" or "dragging" through path slope analysis.~~
    *   **Implementation Status**: Basic DTW dual-score system completed in Phase 3.5, supporting simultaneous display of real-time and DTW-aligned scores.
*   **AR Ghost Overlay**:
    *   **Objective**: Provide "what you see is what you get" correction experience.
    *   **Approach**: Develop skeleton projection mode, directly overlaying the reference video's standard skeleton (green semi-transparent) onto the user's view, allowing users to complete motion correction simply by aligning lines.
*   **Generative AI Coach (LLM Integration)**:
    *   **Objective**: Provide humanized natural language feedback.
    *   **Approach**: Integrate Large Language Model APIs to transform structured bad-frame data into "personal trainer-level" comments (e.g., "Raise your left leg higher, watch your center of gravity"), and generate intelligent summaries in PDF reports.
