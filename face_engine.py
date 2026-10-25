"""Face detection (YuNet) + recognition (SFace) on OpenCV's built-in DNN module.

There is no training step: enrolling a person means storing a few 128-d embeddings,
and recognising means a nearest-neighbour search against them.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """Raised when the ONNX models are missing or cannot be downloaded."""


# ---------------------------------------------------------------- model files
def _download(spec: dict, dest: Path, progress: Optional[Callable[[str], None]]) -> None:
    last_error: Optional[Exception] = None
    for url in spec["urls"]:
        tmp_path: Optional[Path] = None
        try:
            if progress:
                progress(f"Downloading {dest.name} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "FaceAttendancePro/2.0"})
            with urllib.request.urlopen(request, timeout=60) as response, \
                    tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(response, tmp)
            if tmp_path.stat().st_size < spec["min_bytes"]:
                raise ModelError("download incomplete (file too small)")
            os.replace(tmp_path, dest)
            return
        except Exception as exc:  # try the next mirror
            last_error = exc
            log.warning("Download from %s failed: %s", url, exc)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    manual = "\n".join(spec["urls"])
    raise ModelError(
        f"Could not download {dest.name} ({last_error}).\n"
        f"Download it manually from one of:\n{manual}\n"
        f"and place it in: {dest.parent}"
    )


def ensure_models(progress: Optional[Callable[[str], None]] = None) -> tuple[Path, Path]:
    """Return (detector_path, recognizer_path), downloading them on first run."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in ("detector", "recognizer"):
        spec = config.MODELS[key]
        path = config.MODELS_DIR / spec["file"]
        if not (path.exists() and path.stat().st_size >= spec["min_bytes"]):
            _download(spec, path, progress)
        paths.append(path)
    return paths[0], paths[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_ok(path: Path, spec: dict) -> bool:
    """File exists, is big enough and (when a hash is pinned) is byte-for-byte the tested model."""
    if not (path.exists() and path.stat().st_size >= spec["min_bytes"]):
        return False
    expected = spec.get("sha256")
    return not expected or _sha256(path) == expected


def ensure_antispoof_models(progress: Optional[Callable[[str], None]] = None) -> list[tuple[Path, float]]:
    """[(model_path, crop_scale), ...] for the photo/screen detector, downloading on first use.

    The files are verified against the SHA-256 pinned in config.py: this is a security component,
    so a corrupted or swapped model is refused instead of silently trusted.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for spec in config.ANTISPOOF_MODELS:
        path = config.MODELS_DIR / spec["file"]
        if not _model_ok(path, spec):
            path.unlink(missing_ok=True)
            _download(spec, path, progress)
            if not _model_ok(path, spec):
                path.unlink(missing_ok=True)
                raise ModelError(
                    f"{spec['file']} does not match the checksum pinned in config.py, so it was rejected.\n"
                    f"Download the original from:\n" + "\n".join(spec["urls"]))
        result.append((path, float(spec["scale"])))
    return result


# ---------------------------------------------------------------- geometry helpers
def head_yaw(face: np.ndarray) -> float:
    """Horizontal nose offset divided by inter-eye distance, in raw camera coordinates.

    ~0 when facing the camera. Positive = nose towards the right of the *camera* image,
    which is the person's LEFT. Sorting the eyes by x makes this independent of the
    detector's left/right landmark labelling.
    """
    pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    eyes = pts[:2][np.argsort(pts[:2, 0])]
    eye_dist = float(eyes[1, 0] - eyes[0, 0])
    if eye_dist < 1e-3:
        return 0.0
    mid_x = float(eyes[0, 0] + eyes[1, 0]) / 2.0
    return (float(pts[2, 0]) - mid_x) / eye_dist


def face_turn(face: np.ndarray) -> float:
    """Sideways offset of the nose from the face's own centre line, in eye-distance units.

    Same scale and sign as head_yaw (0 = facing the camera, positive = the person's LEFT), but
    built so a FLAT PHOTO cannot fake it. The nose position is written as
        nose - eye_mid = beta * (eye_vector) + alpha * (eye_mid -> mouth_mid vector)
    and beta is returned. beta is unchanged by ANY affine transform of the picture - tilting,
    rotating, sliding, shearing, or turning a flat photo about any axis - because those transforms
    keep the same coefficients. Only genuine 3-D depth (the nose sticking out of the face) changes
    it, which is exactly what a real head turn does. (head_yaw projects on the camera x axis, so
    simply rolling a photo sideways changed it.)
    """
    pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    eyes = pts[:2][np.argsort(pts[:2, 0])]          # camera-left eye first, independent of labelling
    eye_vec = eyes[1] - eyes[0]
    eye_mid = eyes.mean(axis=0)
    down = pts[3:5].mean(axis=0) - eye_mid           # eye centre -> mouth centre
    nose = pts[2] - eye_mid
    det = float(eye_vec[0] * down[1] - eye_vec[1] * down[0])
    if abs(det) < 0.2 * float(np.linalg.norm(eye_vec) * np.linalg.norm(down)) or abs(det) < 1e-3:
        return 0.0                                   # degenerate landmarks: never counts as a turn
    return float(nose[0] * down[1] - nose[1] * down[0]) / det


def image_quality(aligned: np.ndarray) -> tuple[float, float]:
    """(sharpness, brightness) of an aligned face crop."""
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())


# ---------------------------------------------------------------- engine
class FaceEngine:
    """Owns the two DNNs. Use from ONE thread only (OpenCV DNN objects are not thread-safe)."""

    def __init__(self, detector_path: Path, recognizer_path: Path):
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            config.DETECT_SCORE, config.DETECT_NMS, config.DETECT_TOP_K)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._input_size = (320, 320)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Faces as an (N, 15) float32 array in full-frame coordinates:
        x, y, w, h, 5 landmarks (x, y), score."""
        h, w = frame.shape[:2]
        scale = min(1.0, config.DETECT_MAX_SIDE / max(h, w))
        small = frame if scale == 1.0 else cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size
        _, faces = self._detector.detect(small)
        if faces is None:
            return np.empty((0, 15), np.float32)
        faces = faces.astype(np.float32)
        faces[:, :14] /= scale
        return faces

    def align(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        """112x112 aligned face crop (BGR)."""
        return self._recognizer.alignCrop(frame, np.asarray(face, dtype=np.float32))

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """L2-normalised 128-d embedding of an aligned crop."""
        vec = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------- passive photo / screen detector
def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - float(np.max(logits)))
    return e / float(e.sum())


def _scaled_crop(frame: np.ndarray, box: tuple, scale: float, size: int = 80) -> Optional[np.ndarray]:
    """Crop the face box enlarged by `scale` around its centre (shifted, never squashed, to stay inside
    the frame) and resize to size x size. The anti-spoof models were trained on this exact context crop:
    they look at the surroundings (paper edge, phone bezel, screen glare), not just the skin."""
    fh, fw = frame.shape[:2]
    x, y, w, h = (float(v) for v in box)
    x0, y0 = max(0.0, x), max(0.0, y)
    w, h = min(fw - 1.0, x + w) - x0, min(fh - 1.0, y + h) - y0
    if w < 8 or h < 8:
        return None
    scale = min((fh - 1) / h, (fw - 1) / w, scale)
    new_w, new_h = w * scale, h * scale
    cx, cy = x0 + w / 2, y0 + h / 2
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = left + new_w, top + new_h
    if left < 0:
        right, left = right - left, 0.0
    if top < 0:
        bottom, top = bottom - top, 0.0
    if right > fw - 1:
        left, right = left - (right - (fw - 1)), fw - 1.0
    if bottom > fh - 1:
        top, bottom = top - (bottom - (fh - 1)), fh - 1.0
    l, t, r, b = (int(max(v, 0.0)) for v in (left, top, right, bottom))
    crop = frame[t:b + 1, l:r + 1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


class AntiSpoof:
    """Photo / screen / print detector: the average of two MiniFASNet networks (different crop scales).

    Output is the probability that the face is a LIVE person. A photo, a phone/tablet screen or a
    printout normally scores low. Like every single-camera detector it is not infallible (see README);
    that is why the app also demands a random head-turn. Use from ONE thread only.
    """
    REAL_CLASS = 1       # model outputs [spoof-A, REAL, spoof-B]

    def __init__(self, models: Sequence[tuple[Path, float]]):
        if not models:
            raise ModelError("No anti-spoof models given.")
        self._nets = [(cv2.dnn.readNetFromONNX(str(path)), float(scale)) for path, scale in models]

    def real_probability(self, frame: np.ndarray, face: np.ndarray) -> float:
        """P(live) in [0, 1] for one detected face in a full-resolution BGR frame."""
        probs = []
        for net, scale in self._nets:
            crop = _scaled_crop(frame, face[:4], scale)
            if crop is None:
                return 0.0                           # unusable crop is never evidence of a live face
            # trained on raw 0-255 BGR values (no mean/std, no /255)
            net.setInput(np.ascontiguousarray(crop.astype(np.float32).transpose(2, 0, 1)[None]))
            probs.append(float(_softmax(net.forward().reshape(-1))[self.REAL_CLASS]))
        return float(np.mean(probs))


# ---------------------------------------------------------------- gallery / matching
@dataclass(frozen=True)
class MatchResult:
    reg_no: Optional[str]    # best identity (even when rejected, useful for diagnostics)
    score: float             # cosine similarity of the best identity
    runner_up: float         # best score among the *other* identities
    accepted: bool


class Gallery:
    """All enrolled embeddings in one matrix; identity score = best sample of that identity."""

    def __init__(self, ids: list[str], labels: np.ndarray, matrix: np.ndarray, names: dict[str, str]):
        self.ids = ids
        self._labels = labels
        self._matrix = matrix
        self.names = names

    @classmethod
    def empty(cls) -> "Gallery":
        return cls([], np.zeros(0, np.int64), np.zeros((0, 128), np.float32), {})

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, str, np.ndarray]]) -> "Gallery":
        ids: list[str] = []
        index: dict[str, int] = {}
        names: dict[str, str] = {}
        labels: list[int] = []
        vectors: list[np.ndarray] = []
        for reg_no, name, vec in records:
            if reg_no not in index:
                index[reg_no] = len(ids)
                ids.append(reg_no)
                names[reg_no] = name
            labels.append(index[reg_no])
            vectors.append(np.asarray(vec, np.float32))
        if not vectors:
            return cls.empty()
        return cls(ids, np.asarray(labels, np.int64), np.vstack(vectors), names)

    def __len__(self) -> int:
        return len(self.ids)

    def _per_identity(self, query: np.ndarray) -> np.ndarray:
        sims = self._matrix @ np.asarray(query, np.float32)
        best = np.full(len(self.ids), -1.0, np.float32)
        np.maximum.at(best, self._labels, sims)
        return best

    def match(self, query: np.ndarray) -> MatchResult:
        if not self.ids:
            return MatchResult(None, -1.0, -1.0, False)
        best = self._per_identity(query)
        order = np.argsort(best)[::-1]
        top = int(order[0])
        score = float(best[top])
        runner = float(best[order[1]]) if len(order) > 1 else -1.0
        accepted = score >= config.MATCH_THRESHOLD and (score - runner) >= config.MATCH_MARGIN
        return MatchResult(self.ids[top], score, runner, accepted)

    def best_other(self, vectors: Sequence[np.ndarray],
                   exclude: Optional[str] = None) -> Optional[tuple[str, float]]:
        """Highest-scoring *other* identity for any of the given vectors (duplicate check)."""
        best: Optional[tuple[str, float]] = None
        for vec in vectors:
            scores = self._per_identity(vec) if self.ids else np.zeros(0)
            for reg_no, score in zip(self.ids, scores):
                if reg_no == exclude:
                    continue
                if best is None or score > best[1]:
                    best = (reg_no, float(score))
        return best
