# Status Report

**Project Title:** Dance Pose Scoring and Training Assistant System  
**Team Name:** [Your Team Name]  
**Date:** March 9, 2026  
**Version:** 1.0  
**Team Members:** Zheng Jiayong  
**Mentor:** [Mentor Name]  

---

## Executive Summary

The "Dance Pose Scoring and Training Assistant System" project has successfully completed its core development phase (Phase I through Phase III). The primary objective was to develop a real-time computer vision application capable of analyzing human dance movements and providing quantitative feedback. To date, the project has delivered a robust desktop application featuring a dual-weighted scoring algorithm (combining joint angles and Procrustes shape analysis), a multi-threaded video processing architecture, and a modern, user-centric graphical interface. While minor deviations from the original proposal occurred—specifically regarding the initial scope of web-based deployment versus the current desktop focus—the project remains on schedule for its major milestones. The system demonstrates high accuracy in pose estimation and provides actionable feedback through real-time score charts and rhythm analysis. The remaining work focuses on advanced intelligence features, such as Dynamic Time Warping (DTW) for rhythm precision and LLM-based feedback generation, which are planned for the final phase.

## Project Description

This project aims to democratize professional dance training by leveraging artificial intelligence. The system utilizes a standard webcam to capture a user's movements and compares them in real-time against a reference video of a professional instructor. By employing the MediaPipe framework for pose estimation and custom geometric algorithms for scoring, the application provides instant, objective feedback on posture accuracy and timing. This solution addresses the high cost and limited availability of personal dance instruction, offering an accessible tool for self-improvement.

## Overview of Project Objectives, Scope, Approach, Major Milestones, and Deliverables

### Objectives
1.  **Real-Time Performance**: Achieve a processing speed of at least 30 FPS on standard consumer hardware.
2.  **Scoring Accuracy**: Develop a scoring model that differentiates between correct and incorrect postures with high precision, specifically targeting limb angles and overall body shape.
3.  **User Experience**: Create an intuitive interface that offers immediate visual and numerical feedback without distracting the user.

### Scope
The project scope encompasses the development of a Python-based desktop application. Key features include video file and camera input handling, pose estimation, synchronous playback, a dual-criteria scoring engine, and a post-session performance report generator. Advanced features like AR ghost overlays and audio-based beat detection are considered stretch goals for the final phase.

### Approach
The development follows an iterative, agile methodology.
*   **Phase I**: Core infrastructure setup (MediaPipe integration, basic UI).
*   **Phase II**: Algorithm refinement (Dual-Weighted Scoring Model).
*   **Phase III**: UI/UX enhancement (Dark mode, real-time charts) and performance optimization (Multi-threading).
*   **Phase IV**: Intelligence expansion (DTW, LLM integration).

### Major Milestones & Deliverables
*   **Milestone 1 (Completed)**: Functional prototype with basic pose detection and Euclidean distance scoring.
    *   *Deliverable*: Initial application entry point and repository setup.
*   **Milestone 2 (Completed)**: Implementation of the Dual-Weighted Scoring Model and multi-threaded video reader.
    *   *Deliverable*: Core scoring module and video processing module.
*   **Milestone 3 (Completed)**: UI overhaul and feature completion (Synchronized playback, Session Summary).
    *   *Deliverable*: Main GUI window module and Progress Report document.
*   **Milestone 4 (Pending)**: Advanced rhythm analysis and AI coaching feedback.

## Response to Feedback/Recommendations

**Feedback from Proposal:** "The initial proposal focused heavily on web deployment, which might introduce latency issues critical for real-time feedback. It is recommended to prioritize local processing performance."

**Response:**
We have pivoted the primary deployment target from a web-based application to a native desktop application using PyQt5. This shift allows for direct access to hardware resources, significantly reducing latency caused by network transmission. While a backend service module was retained for potential future hybrid models, the core development effort (90%) has been dedicated to optimizing the local video reading and pose processing threads. This decision has proven effective, as current tests show stable 30 FPS performance with less than 100ms end-to-end latency.

**Feedback from Proposal:** "Simple Euclidean distance may not capture the nuances of dance movements, such as joint angles."

**Response:**
We implemented a **Dual-Weighted Scoring Model** (Section 2.2.1 of the Progress Report). This model combines Weighted Procrustes Distance (for global shape) with Cosine Similarity (for local joint angles). We assigned higher weights (1.5x - 2.0x) to limb extremities (wrists, ankles) to specifically address the feedback regarding movement nuances.

## Variations or Deviations from Original Proposal

1.  **Shift from Web-First to Desktop-First Architecture**
    *   *Original*: A React/Vue frontend with a Python backend.
    *   *Current*: PyQt5 Desktop Application.
    *   *Rationale*: Real-time synchronization of two video streams (user and reference) over HTTP introduced unacceptable jitter and latency. A desktop approach ensures frame-perfect synchronization, which is critical for dance scoring.

2.  **Deferral of Audio Beat Detection**
    *   *Original*: Included in Phase II.
    *   *Current*: Moved to Phase IV (Future Work).
    *   *Rationale*: Visual rhythm detection (frame-based timing analysis) was prioritized to ensure the core visual scoring system was robust. Audio processing requires complex signal analysis libraries that could jeopardize the stability of the main thread if not implemented carefully.

3.  **Introduction of "Ghost Overlay" Concept**
    *   *Original*: Not present.
    *   *Current*: Added to roadmap.
    *   *Rationale*: User testing suggested that side-by-side comparison is sometimes insufficient for self-correction. An AR-style overlay was identified as a high-value feature for future iterations.

## Summary of Current Project Status

The project is currently **on schedule** according to the revised timeline.

**Work Completed:**
*   **Core Engine**: The pose processing and video reading modules are fully functional and optimized for multi-threading.
*   **Scoring Algorithm**: The dual-weighted algorithm is implemented and tuned.
*   **User Interface**: The "UI/UX 2.0" update is complete, featuring a dark theme, real-time neon charts, and adaptive video layouts.
*   **Reporting**: The session summary and PDF export features are operational.

**Work Yet to be Done (vs. Schedule):**
*   **Dynamic Time Warping (DTW)**: Originally scheduled for late Phase III, this has been pushed to Phase IV.
    *   *Explanation*: The complexity of the UI overhaul (Phase III) was underestimated. We prioritized a polished, usable interface over the advanced rhythm algorithm to ensure a demonstrable prototype for the mid-term review.
*   **LLM Integration**: Scheduled for Phase IV as planned.

**Significant Issues & Mitigation:**
*   **Issue**: Video Aspect Ratio Distortion. Non-16:9 videos were initially stretched.
    *   *Mitigation*: We implemented a letterboxing strategy in the video display component to maintain aspect ratios while centering content.
*   **Issue**: Thread Safety. Initial versions crashed when toggling cameras rapidly.
    *   *Mitigation*: We introduced strict mutex locking in the video reader and signal-slot mechanisms for all cross-thread communication.

## Summary of Individual Contributions

**Team Member: Zheng Jiayong**
*   **Role**: Full Stack Developer & Algorithm Engineer.
*   **Hours Contributed**: ~120 hours.
*   **Key Activities**:
    *   **Algorithm Design (40h)**: Researched and implemented the Weighted Procrustes and Cosine Similarity algorithms. Tuned weight parameters for optimal scoring.
    *   **System Architecture (30h)**: Designed the multi-threaded pipeline using `QThread` to decouple video capture from AI inference.
    *   **UI/UX Development (30h)**: Developed the PyQt5 interface, including custom widgets (`ScoreChartWidget`) and the dark theme QSS.
    *   **Testing & Documentation (20h)**: Conducted unit tests, fixed bugs (e.g., aspect ratio, unicode rendering), and authored the comprehensive Progress Report.
*   **Learning Outcomes**:
    *   Deepened understanding of **Computer Vision pipelines** and the MediaPipe framework.
    *   Mastered **Concurrent Programming** in Python (GUI vs. Worker threads).
    *   Gained proficiency in **Software Architecture** refactoring (separating Core, UI, and Utils).

## Recommendations for Improving Project Team Performance

1.  **Automated Testing**: While unit tests exist, integrating an automated testing framework (e.g., pytest) into the CI/CD pipeline would reduce regression bugs during major refactors.
2.  **Code Review Rigor**: Adopting a stricter code review process for algorithm changes, specifically focusing on edge cases (e.g., occlusion handling), would improve system robustness.
3.  **User Feedback Loop**: Establishing a more formal mechanism for gathering user feedback (e.g., a beta testing group) earlier in the development cycle would help prioritize features more effectively, such as the demand for the "Ghost Overlay" feature.
