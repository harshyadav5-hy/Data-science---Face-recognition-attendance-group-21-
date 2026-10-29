"""Central configuration. Change values here -no other file needs editing."""
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

CONFIRM_WINDOW = 5           # look at the last N frames of a tracked face ...
CONFIRM_FRAMES = 3           # ... and need this many agreeing frames to name / mark it
TRACK_TTL = 1.5              # seconds a track survives without a detection

# Face-swap detection: if the face inside one tracked box suddenly stops resembling the previous
# frames (a different person stepped in, or a photo was swapped in), the track is reset at once
# instead of keeping the old identity. Same person frame-to-frame is normally > 0.6 similarity.
SWAP_SIMILARITY = 0.25
SWAP_FRAMES = 2              # consecutive dissimilar frames needed (ignores one bad/blurred frame)

# ---------------------------------------------------------------- live-person checks (anti-spoofing)
# LIVE mode (default in the app) accepts only a physically present person, using three layers:
#   1. passive photo/screen detector (two MiniFASNet models, Apache-2.0, minivision-ai)
#   2. random head-turn challenge measured with a photo-proof 3-D cue (see face_engine.face_turn)
#   3. identity + face-continuity checks, so the person who turns is the person who is marked
ANTISPOOF_REQUIRED = True        # True = no attendance is marked unless the photo detector is loaded
ANTISPOOF_MODELS = [
    {"file": "MiniFASNetV2.onnx", "scale": 2.7, "min_bytes": 1_000_000,
     "sha256": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
     "urls": ["https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"]},
    {"file": "MiniFASNetV1SE.onnx", "scale": 4.0, "min_bytes": 1_000_000,
     "sha256": "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
     "urls": ["https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx"]},
]
ANTISPOOF_REAL_THRESHOLD = 0.60  # mean "real" probability needed to count as a live face
ANTISPOOF_HYSTERESIS = 0.10      # a live face is only called a photo when it drops below THRESHOLD - this
ANTISPOOF_WINDOW = 8             # frames averaged per tracked face
ANTISPOOF_MIN_FRAMES = 4         # frames needed before a verdict
ANTISPOOF_ALERT_COOLDOWN = 15.0  # seconds between repeated warnings/audit entries for the same person
SHOW_DEBUG_SCORES = False        # True = print live "real" score and match similarity on the video (tuning)

LIVENESS_STEPS = 2               # random head-turn steps: 2 = turn left AND right (random order), 1 = one side
LIVENESS_FRONT_MAX = 0.14        # |turn| below this counts as "looking straight" (must be seen first)
LIVENESS_TURN_MIN = 0.18         # turn needed (~19 degrees); a flat photo stays near 0 even when tilted
LIVENESS_TURN_MAX = 0.80         # beyond this the landmarks are unreliable
LIVENESS_HOLD_FRAMES = 3         # consecutive frames the turn must be held
LIVENESS_TIMEOUT = 10.0          # seconds to finish the challenge before a new random one starts

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
