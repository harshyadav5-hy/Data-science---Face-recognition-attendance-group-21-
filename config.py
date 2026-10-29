"""Central configuration. Change values here - no other file needs editing."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "Face Attendance Pro"
APP_VERSION = "2.2"

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

# Recognition works on an AVERAGE of the last few good frames of a tracked face (much less noisy than
# one frame), and keeps checking afterwards, so a different person stepping in is noticed at once.
FUSE_FRAMES = 6              # good frames averaged into one embedding ...
FUSE_MIN = 3                 # ... but never decide on fewer than this many
CONFIRM_FRAMES = 2           # consecutive agreeing decisions before an identity is locked
RECOG_MAX_YAW = 0.30         # frames turned further than this are not used for recognition
RECOG_MIN_SHARPNESS = 12.0   # variance of Laplacian on the 112x112 aligned crop (enrolment: 25)
RECOG_MIN_BRIGHT = 35
RECOG_MAX_BRIGHT = 230
CHANGE_SIM = 0.28            # locked person scores below this against their own samples ...
CHANGE_FRAMES = 2            # ... in this many checks in a row (or someone else matches) = face changed
MARKED_VERIFY_INTERVAL = 0.2 # seconds between identity re-checks of an already marked face
TRACK_TTL = 1.5              # seconds a track survives without a detection

# ---------------------------------------------------------------- attendance modes
# Two explicit modes (chosen per session): "live" = physical presence, photos are rejected (default);
# "photo" = a photo is accepted as proof (e.g. class photo files) - needs the admin password.
ALLOW_PHOTO_MODE = True          # False removes photo mode and photo-file import completely
PHOTO_MAX_SIDE = 2400            # detection resolution for photo files (webcam preview uses DETECT_MAX_SIDE)
PHOTO_MIN_FACE_PX = 48           # faces smaller than this in a photo file are ignored
PHOTO_MATCH_THRESHOLD = 0.45     # stricter than MATCH_THRESHOLD: small faces in group photos are less reliable

# ---------------------------------------------------------------- live-person check (anti-photo)
# In LIVE mode attendance is NEVER marked on recognition alone. Every student must also pass a short,
# randomised head-turn challenge that a flat photo / phone screen cannot pass (see liveness.py, README).
LIVENESS_STEPS = 2               # 2 = turn LEFT and RIGHT (random order); 1 = one random side
LIVENESS_MIN_FACE_PX = 130       # smaller faces are asked to come closer: landmark jitter matters more there
LIVENESS_TURN_YAW = 0.22         # nose offset / eye distance needed on the requested side (~25 deg)
LIVENESS_MIN_PARALLAX = 0.12     # ... AND the nose must slide this far vs eyes+mouth (3-D proof)
LIVENESS_BASELINE_YAW = 0.10     # how straight "look straight" must be before the challenge starts
LIVENESS_BASELINE_FRAMES = 3
LIVENESS_HOLD_FRAMES = 3         # frames each turn must be held
LIVENESS_RAMP_FROM = 0.06        # on the way in, a turn must spend at least LIVENESS_RAMP_FRAMES frames between
LIVENESS_RAMP_FRAMES = 3         # this yaw and 'fully turned' AND keep rising by LIVENESS_RAMP_SPAN from the first
LIVENESS_RAMP_SPAN = 0.12        # of them. A real head sweeps through these angles; a phone slideshow that
                                 # hard-cuts straight -> turned does not (and one parked below the limit does not rise)
LIVENESS_VERIFY_YAW = 0.15       # final "look straight" tolerance
LIVENESS_VERIFY_FRAMES = 3       # frames in which the SAME person must be recognised again
LIVENESS_SMOOTH = 5              # frames averaged to suppress landmark jitter
LIVENESS_TIMEOUT = 15.0          # whole challenge, seconds
LIVENESS_STILL_SECONDS = 6.0     # asked to turn, but zero 3-D movement this long -> "NOT LIVE"
LIVENESS_MAX_GAP = 0.5           # face lost longer than this (photo swapped?) restarts the challenge
LIVENESS_FLAT_PARALLAX = 0.08    # below this the face counts as flat
LIVENESS_PHOTO_FORESHORTEN = 0.08  # flat AND this much narrower/shorter = photo being rotated/tilted
LIVENESS_PHOTO_FRAMES = 4        # consecutive frames of that evidence before "PHOTO DETECTED"
LIVENESS_FAIL_DISPLAY = 4.0      # seconds a verdict stays on screen before a fresh challenge starts
SPOOF_BANNER_SECONDS = 6.0       # how long the warning stays in the camera view
SPOOF_ALERT_COOLDOWN = 30.0      # audit-log entry + beep at most once per person and kind per this time
LIVENESS_DEBUG = False           # True: show yaw / parallax / narrowing numbers on screen, for tuning

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
