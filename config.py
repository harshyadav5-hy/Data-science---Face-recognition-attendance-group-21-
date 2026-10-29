"""Central configuration. Change values here-no other file needs editing."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "Face Attendance Pro"
APP_VERSION = "2.0"

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FACES_DIR = DATA_DIR / "faces"          # aligned face crops kept for audit / re-embedding
EXPORT_DIR = DATA_DIR / "exports"
BACKUP_DIR = DATA_DIR / "backups"
DB_PATH = DATA_DIR / "attendance.db"
LOG_PATH = DATA_DIR / "app.log"

# ---------------------------------------------------------------- models (OpenCV Zoo, Apache-2.0)
_GITHUB = "https://github.com/opencv/opencv_zoo/raw/main/models"
_HF = "https://huggingface.co/opencv"
MODELS = {
    "detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "min_bytes": 200_000,
        "urls": [
            f"{_GITHUB}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            f"{_HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        ],
    },
    "recognizer": {
        "file": "face_recognition_sface_2021dec.onnx",
        "min_bytes": 30_000_000,
        "urls": [
            f"{_GITHUB}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            f"{_HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        ],
    },
}
EMBEDDING_MODEL_TAG = "sface_2021dec"   # stored with every embedding; lets you migrate models later

# ---------------------------------------------------------------- camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DETECT_MAX_SIDE = 640        # detector runs on a downscaled copy for speed

# ---------------------------------------------------------------- detection / matching
DETECT_SCORE = 0.85
DETECT_NMS = 0.30
DETECT_TOP_K = 50
MIN_FACE_PX = 90             # smaller faces are ignored (too unreliable)

# Cosine similarity between SFace embeddings. OpenCV's documented "balanced" value is 0.363;
# attendance cares more about false accepts (proxy attendance), so we start stricter.
# TUNE THESE on your own students: see README "Calibrating".
MATCH_THRESHOLD = 0.50
MATCH_MARGIN = 0.05          # best identity must beat the runner-up by this much
DUPLICATE_THRESHOLD = 0.50   # at enrolment: refuse if the face already belongs to another Reg No

CONFIRM_WINDOW = 6           # look at the last N frames of a tracked face ...
CONFIRM_FRAMES = 4           # ... and need this many agreeing frames before marking
TRACK_TTL = 1.5              # seconds a track survives without a detection

# Optional head-turn liveness (stops held-up photos, NOT video replays - see README)
LIVENESS_YAW_RANGE = 0.30
LIVENESS_TIMEOUT = 8.0

# ---------------------------------------------------------------- enrolment
ENROLL_PROMPTS = [
    ("Look straight at the camera", "front"),
    ("Turn your head slightly to your LEFT", "left"),
    ("Turn your head slightly to your RIGHT", "right"),
    ("Look straight and smile", "front"),
    ("Tilt your chin slightly UP", "any"),
    ("Tilt your chin slightly DOWN", "any"),
    ("Look straight again", "front"),
    ("Look straight, move a little closer", "front"),
]
ENROLL_MIN_SCORE = 0.90
ENROLL_MIN_FACE = 110
ENROLL_STABLE_FRAMES = 4
ENROLL_COOLDOWN = 0.8
ENROLL_MIN_SHARPNESS = 25.0   # variance of Laplacian on the 112x112 aligned crop
ENROLL_MIN_BRIGHT = 50
ENROLL_MAX_BRIGHT = 215
ENROLL_MAX_SIMILARITY = 0.96  # reject near-identical consecutive samples
YAW_FRONT = 0.12
YAW_TURN_MIN = 0.15
YAW_TURN_MAX = 0.60
YAW_ANY = 0.25
SAVE_FACE_IMAGES = True       # False = store embeddings only (more private, no re-embedding later)

# ---------------------------------------------------------------- security / data
ADMIN_UNLOCK_SECONDS = 600
ADMIN_MIN_PASSWORD_LEN = 8
PBKDF2_ITERATIONS = 200_000
REG_NO_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9/_\-]{2,31}$"
BACKUP_KEEP = 14
