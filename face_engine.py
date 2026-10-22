"""Face detection (YuNet) + recognition (SFace) on OpenCV's built-in DNN module.

There is no training step: enrolling a person means storing a few 128-d embeddings,
and recognising means a nearest-neighbour search against them.
"""
from __future__ import annotations

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

    def detect(self, frame: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
        """Faces as an (N, 15) float32 array in full-frame coordinates:
        x, y, w, h, 5 landmarks (x, y), score.  max_side: detection resolution (default
        config.DETECT_MAX_SIDE; class photos use a larger value so small faces survive)."""
        h, w = frame.shape[:2]
        scale = min(1.0, (max_side or config.DETECT_MAX_SIDE) / max(h, w))
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
        self._index = {reg_no: i for i, reg_no in enumerate(ids)}

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

    def score_of(self, query: np.ndarray, reg_no: str) -> float:
        """Similarity of `query` to ONE identity's enrolled samples (-1 if not in the gallery).
        Used to notice that the face in front of the camera is no longer the person we locked on to."""
        i = self._index.get(reg_no)
        if i is None:
            return -1.0
        return float(self._per_identity(query)[i])

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
