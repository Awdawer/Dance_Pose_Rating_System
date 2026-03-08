import os
import sys
import base64
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current directory to path to allow imports from src
sys.path.append(os.path.dirname(__file__))

from src.utils.model_loader import ensure_model, MODEL_PATH
from src.utils.geometry import compute_angles
from src.core.scoring import score_angles

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_model()

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
pl_options = mp_vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
detector = mp_vision.PoseLandmarker.create_from_options(pl_options)

class ScoreRequest(BaseModel):
    user_frame: str
    ref_frame: str

def b64_to_image_uri(data_uri: str) -> np.ndarray:
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def to_list_landmarks(res):
    if not res or not res.pose_landmarks:
        return []
    if len(res.pose_landmarks) == 0:
        return []
    out = []
    for lm in res.pose_landmarks[0]:
        out.append({"x": float(lm.x), "y": float(lm.y)})
    return out

def detect_landmarks_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_image)
    return res

@app.post("/api/score")
def api_score(req: ScoreRequest):
    try:
        img_user = b64_to_image_uri(req.user_frame)
        img_ref = b64_to_image_uri(req.ref_frame)
    except Exception:
        return {"ok": False, "error": "bad_image"}
    res_u = detect_landmarks_bgr(img_user)
    res_r = detect_landmarks_bgr(img_ref)
    l_u = to_list_landmarks(res_u)
    l_r = to_list_landmarks(res_r)
    if len(l_u) == 0 or len(l_r) == 0:
        return {"ok": True, "percent": 0.0, "diffs": {}, "user_landmarks": l_u, "ref_landmarks": l_r}
    au = compute_angles(l_u)
    ar = compute_angles(l_r)
    _, percent, _, diffs = score_angles(au, ar)
    return {"ok": True, "percent": percent, "diffs": diffs, "user_landmarks": l_u, "ref_landmarks": l_r}
