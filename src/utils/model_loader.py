import os
import shutil
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

def _ascii_cache_dir():
    """Get a cache directory path that is ASCII-safe (avoids non-ASCII chars in path)."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or os.getcwd()
    d = os.path.join(base, "dancepose_models")
    os.makedirs(d, exist_ok=True)
    return d

def _copy_to_ascii_path(src_path: str) -> str:
    """Copy model to an ASCII-safe path if needed."""
    dst_dir = _ascii_cache_dir()
    dst_path = os.path.join(dst_dir, "pose_landmarker_full.task")
    try:
        if os.path.exists(src_path):
            # Always overwrite to avoid partial/corrupt file issues
            shutil.copyfile(src_path, dst_path)
            return dst_path
    except Exception:
        pass
    return src_path

def ensure_model() -> str | None:
    """Ensure the model exists locally, downloading if necessary."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Model download failed: {e}")
            return None
    return _copy_to_ascii_path(MODEL_PATH)
