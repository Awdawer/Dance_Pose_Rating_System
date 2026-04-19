import os
import tempfile
import numpy as np
from scipy.signal import correlate

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[AudioAligner] librosa not available. Install with: pip install librosa")

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("[AudioAligner] moviepy not available. Install with: pip install moviepy")


def extract_audio_from_video(video_path, output_path=None, sample_rate=22050):
    if not MOVIEPY_AVAILABLE:
        print("[AudioAligner] moviepy not available, cannot extract audio")
        return None
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"audio_{os.path.basename(video_path)}.wav")
    
    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                print(f"[AudioAligner] No audio track in {video_path}")
                return None
            clip.audio.write_audiofile(output_path, fps=sample_rate, logger=None)
            return output_path
    except Exception as e:
        print(f"[AudioAligner] Error extracting audio: {e}")
        return None


def _find_offset_onset(y1, y2, sr):
    hop_length = 512
    onset_env1 = librosa.onset.onset_strength(y=y1, sr=sr, hop_length=hop_length)
    onset_env2 = librosa.onset.onset_strength(y=y2, sr=sr, hop_length=hop_length)
    
    correlation = correlate(onset_env1, onset_env2, mode='full')
    lag_index = np.argmax(correlation)
    frame_offset = lag_index - (len(onset_env2) - 1)
    
    return librosa.frames_to_time(frame_offset, sr=sr, hop_length=hop_length)


def _find_offset_chroma(y1, y2, sr):
    hop_length = 512
    
    print("[AudioAligner] Computing chromagram for video A...")
    chroma_a = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=hop_length)
    
    print("[AudioAligner] Computing chromagram for video B...")
    chroma_b = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=hop_length)

    print("[AudioAligner] Computing cross-correlation...")
    total_correlation = np.zeros(chroma_a.shape[1] + chroma_b.shape[1] - 1)
    
    for i in range(chroma_a.shape[0]):
        band_corr = correlate(chroma_a[i], chroma_b[i], mode='full', method='fft')
        total_correlation += band_corr

    lag_index = np.argmax(total_correlation)
    frame_offset = lag_index - (chroma_b.shape[1] - 1)
    
    return librosa.frames_to_time(frame_offset, sr=sr, hop_length=hop_length)


def find_time_offset(audio_path1, audio_path2, sample_rate=22050, method='chroma'):
    if not LIBROSA_AVAILABLE:
        print("[AudioAligner] librosa not available, returning 0 offset")
        return 0.0
    
    print(f"[AudioAligner] Loading audio files (method: {method})...")
    y1, sr = librosa.load(audio_path1, sr=sample_rate)
    y2, _ = librosa.load(audio_path2, sr=sample_rate)

    if method == 'chroma':
        offset = _find_offset_chroma(y1, y2, sr)
    elif method == 'onset':
        offset = _find_offset_onset(y1, y2, sr)
    else:
        raise ValueError(f"Unknown alignment method: {method}. Use 'chroma' or 'onset'.")

    print(f"[AudioAligner] Found time offset: {offset:.4f} seconds")
    if offset > 0:
        print(f"[AudioAligner] Video B starts {abs(offset):.2f}s after Video A")
    else:
        print(f"[AudioAligner] Video A starts {abs(offset):.2f}s after Video B")
    
    return offset


def align_videos(video_path_a, video_path_b, method='chroma', sample_rate=22050):
    if not LIBROSA_AVAILABLE or not MOVIEPY_AVAILABLE:
        print("[AudioAligner] Required libraries not available, skipping alignment")
        return 0.0, 0
    
    audio_a = extract_audio_from_video(video_path_a, sample_rate=sample_rate)
    audio_b = extract_audio_from_video(video_path_b, sample_rate=sample_rate)
    
    if audio_a is None or audio_b is None:
        print("[AudioAligner] One or both videos have no audio, skipping alignment")
        return 0.0, 0
    
    try:
        time_offset = find_time_offset(audio_a, audio_b, sample_rate, method)
        
        cap = cv2.VideoCapture(video_path_a if video_path_a else video_path_b)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        frame_offset = int(time_offset * fps)
        
        return time_offset, frame_offset
    finally:
        if audio_a and os.path.exists(audio_a):
            try:
                os.remove(audio_a)
            except:
                pass
        if audio_b and os.path.exists(audio_b):
            try:
                os.remove(audio_b)
            except:
                pass


import cv2
