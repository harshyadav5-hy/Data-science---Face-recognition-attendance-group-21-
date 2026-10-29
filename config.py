"""Central configuration. Change values here - no other file needs editing."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "Face Attendance Pro"
APP_VERSION = "2.1"

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FACES_DIR = DATA_DIR / "faces"          # aligned face crops kept for audit / re-embedding
EXPORT_DIR = DATA_DIR / "exports"
BACKUP_DIR = DATA_DIR / "backups"
SPOOF_DIR = DATA_DIR / "spoof_alerts"    # snapshots of photo / replay attempts (evidence)
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
MATCH_THRESHOLD = 0.40
MATCH_MARGIN = 0.05          # best identity must beat the runner-up by this much
DUPLICATE_THRESHOLD = 0.50   # at enrolment: refuse if the face already belongs to another Reg No

CONFIRM_WINDOW = 6           # look at the last N frames of a tracked face ...
CONFIRM_FRAMES = 4           # ... and need this many agreeing frames before marking
TRACK_TTL = 1.5              # seconds a track survives without a detection

# ---------------------------------------------------------------- recognition quality
FUSE_FRAMES = 5             # embeddings of the last N frames are averaged into one more stable template
FUSE_MIN = 3
MATCH_TOPK = 2              # identity score = mean of its best K samples (one lucky sample can't match)
RECOG_MAX_YAW = 0.20        # frames turned further than this are not used to identify (poor accuracy)
RECOG_MIN_SHARPNESS = 12.0  # blurry / dark / glaring frames are skipped instead of being matched
RECOG_MIN_BRIGHT = 35
RECOG_MAX_BRIGHT = 225
LOSE_FRAMES = 4             # this many consecutive frames NOT matching the shown identity -> drop it
RECHECK_INTERVAL = 0.5      # already-marked faces are re-verified this often (detects a person swap)
RECHECK_MISSES = 3          # consecutive failed re-checks before the label is withdrawn
NEAR_MISS_FRAMES = 5        # consistent "almost matches" frames before the 'face changed?' hint appears
NEAR_MISS_MIN = 0.28

# ---------------------------------------------------------------- adaptive templates (face changed)
# After a LIVE, confirmed match, a new view of the face (glasses, beard, hair, lighting) is added to
# that student's templates immediately, so the next frames already recognise it.
ADAPT_ENABLED = True
ADAPT_MIN_SCORE = 0.45       # only confident matches are ever learned from
ADAPT_MAX_SIMILARITY = 0.80  # skip if an existing template already looks like this (nothing new)
ADAPT_MAX_OTHER = 0.30       # skip if the face is also similar to a different student (ambiguous)
MAX_AUTO_TEMPLATES = 8       # automatic samples kept per student (oldest dropped first)
MAX_UPDATE_TEMPLATES = 12    # samples added by the admin "Update look" action
UPDATE_PROMPTS = [           # quick admin-confirmed update (Students tab -> Update look)
    ("Look straight at the camera", "front"),
    ("Turn your head slightly to your LEFT", "left"),
    ("Turn your head slightly to your RIGHT", "right"),
    ("Look straight and smile", "front"),
]

# ---------------------------------------------------------------- liveness (anti-photo) - ALWAYS ON
# Every student must pass a random 3-D head-turn challenge (turn to one side, then the other, then
# look straight) before attendance is marked. Nose position is measured in the face's own
# coordinate frame (eyes + mouth), which does not change when a flat photo is moved, rolled or
# tilted - only a real 3-D head changes it.
LIVENESS_TURN = 0.16          # nose shift needed to count as a head turn (real 20 deg turn ~ 0.2)
LIVENESS_NEUTRAL_MAX = 0.12   # |nose offset| allowed while measuring the straight-ahead pose
LIVENESS_NEUTRAL_FRAMES = 4
LIVENESS_NEUTRAL_SPREAD = 0.05  # ... and steady (nose offset varies less than this)
LIVENESS_HOLD_FRAMES = 3      # frames the turn must be held (median-filtered)
LIVENESS_VERIFY_FRAMES = 3    # frames looking straight again to finish
LIVENESS_TIMEOUT = 12.0       # seconds to finish; otherwise "no live movement - possible photo"
LIVENESS_JUMP = 0.45          # an instant nose jump this big per frame = swapped picture -> restart
LIVENESS_FLAT_RATIO = 0.86    # face got >=14% narrower (tilted away) ...
LIVENESS_FLAT_MAX_SHIFT = 0.08  # ... while the nose did not move in the face = flat picture tilted
LIVENESS_FLAT_FRAMES = 5
VERIFY_TIMEOUT = 6.0          # seconds allowed to re-confirm identity after the challenge
SPOOF_RED_SECONDS = 4.0       # how long the red PHOTO warning stays on the face
SAVE_SPOOF_SNAPSHOTS = True   # keep a cropped photo of each spoof attempt in data/spoof_alerts
SPOOF_KEEP = 200
LIVENESS_DEBUG = False        # True: show nose shift (dU) and face-width ratio (f) on the label

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
