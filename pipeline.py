"""Per-frame logic with no Qt and no camera, so it can be unit-tested.

* AttendanceProcessor - tracks faces across frames, confirms an identity only after several
  agreeing frames, optionally demands a head-turn liveness check, then reports it once.
* EnrollmentSession   - guided capture of varied, sharp, well-lit samples for one person.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from face_engine import Gallery, head_yaw, image_quality

# Overlay = (x, y, w, h, text, "#rrggbb") in raw camera coordinates.
COL_OK = "#22c55e"
COL_WAIT = "#eab308"
COL_BAD = "#ef4444"
COL_CHALLENGE = "#f97316"
COL_DIM = "#9ca3af"


def _box(face: np.ndarray) -> tuple[float, float, float, float]:
    return float(face[0]), float(face[1]), float(face[2]), float(face[3])


# ======================================================================= attendance
class Track:
    _counter = 0

    def __init__(self, face: np.ndarray, now: float):
        Track._counter += 1
        self.id = Track._counter
        self.history: deque = deque(maxlen=config.CONFIRM_WINDOW)   # (reg_no | None, score)
        self.identity: Optional[str] = None
        self.score = 0.0
        self.marked = False
        self.live = False
        self.challenge_t0: Optional[float] = None
        self.yaw_min = 9.0
        self.yaw_max = -9.0
        self.update_geometry(face, now)

    def update_geometry(self, face: np.ndarray, now: float) -> None:
        x, y, w, h = _box(face)
        self.cx, self.cy, self.w = x + w / 2, y + h / 2, w
        self.last_seen = now

    def reset_challenge(self, now: Optional[float]) -> None:
        self.challenge_t0 = now
        self.yaw_min, self.yaw_max = 9.0, -9.0


class AttendanceProcessor:
    def __init__(self, engine, gallery: Gallery, already_marked: set[str], require_liveness: bool):
        self.engine = engine
        self.gallery = gallery
        self.marked = set(already_marked)
        self.require_liveness = require_liveness
        self.tracks: list[Track] = []

    def _label(self, reg_no: str) -> str:
        name = self.gallery.names.get(reg_no)
        return f"{name} ({reg_no})" if name else reg_no

    def _assign(self, faces: np.ndarray, now: float) -> dict[int, Track]:
        """Greedy nearest-centre matching of detections to existing tracks."""
        free = list(self.tracks)
        assigned: dict[int, Track] = {}
        for i in sorted(range(len(faces)), key=lambda k: -float(faces[k][2] * faces[k][3])):
            x, y, w, h = _box(faces[i])
            cx, cy = x + w / 2, y + h / 2
            best, best_d = None, None
            for track in free:
                d = ((track.cx - cx) ** 2 + (track.cy - cy) ** 2) ** 0.5
                if d < 0.75 * max(w, track.w) and (best_d is None or d < best_d):
                    best, best_d = track, d
            if best is not None:
                free.remove(best)
                track = best
            else:
                track = Track(faces[i], now)
                self.tracks.append(track)
            track.update_geometry(faces[i], now)
            assigned[i] = track
        self.tracks = [t for t in self.tracks if now - t.last_seen <= config.TRACK_TTL]
        return assigned

    def _update_identity(self, track: Track) -> None:
        votes = Counter(reg for reg, _ in track.history if reg)
        if not votes:
            return
        reg, count = votes.most_common(1)[0]
        if count < config.CONFIRM_FRAMES:
            return
        scores = [s for r, s in track.history if r == reg]
        if reg != track.identity:
            track.identity = reg
            track.live = not self.require_liveness
            track.reset_challenge(None)
        track.score = float(np.mean(scores))

    def process(self, frame: np.ndarray, faces: np.ndarray, now: Optional[float] = None):
        """Returns (overlays, events, hud). events = [(reg_no, score)] to record - once each."""
        now = time.monotonic() if now is None else now
        height, width = frame.shape[:2]
        assigned = self._assign(faces, now)
        overlays: list[tuple] = []
        events: list[tuple[str, float]] = []

        for i, face in enumerate(faces):
            track = assigned[i]
            x, y, fw, fh = _box(face)
            box = (x, y, fw, fh)

            cut_off = (x < -0.05 * fw or y < -0.05 * fh
                       or x + fw > width + 0.05 * fw or y + fh > height + 0.05 * fh)
            if fw < config.MIN_FACE_PX or cut_off:
                overlays.append((*box, "Come closer / centre your face", COL_DIM))
                continue

            if track.marked:
                overlays.append((*box, f"\u2713 {self._label(track.identity)}", COL_OK))
                continue

            aligned = self.engine.align(frame, face)
            result = self.gallery.match(self.engine.embed_aligned(aligned))
            track.history.append((result.reg_no if result.accepted else None, result.score))
            self._update_identity(track)

            if track.identity is None:
                if any(reg for reg, _ in track.history):
                    overlays.append((*box, "Identifying...", COL_WAIT))
                elif len(track.history) >= 3:
                    overlays.append((*box, "Unknown", COL_BAD))
                else:
                    overlays.append((*box, "Scanning...", COL_WAIT))
                continue

            reg_no = track.identity
            if reg_no in self.marked:                       # marked via another track / earlier
                track.marked = True
                overlays.append((*box, f"\u2713 {self._label(reg_no)} (already marked)", COL_OK))
                continue

            if not track.live:
                yaw = head_yaw(face)
                if track.challenge_t0 is None:
                    track.reset_challenge(now)
                track.yaw_min = min(track.yaw_min, yaw)
                track.yaw_max = max(track.yaw_max, yaw)
                if track.yaw_max - track.yaw_min >= config.LIVENESS_YAW_RANGE:
                    track.live = True
                elif now - track.challenge_t0 > config.LIVENESS_TIMEOUT:
                    track.reset_challenge(now)

            if track.live:
                self.marked.add(reg_no)
                track.marked = True
                events.append((reg_no, track.score))
                overlays.append((*box, f"\u2713 {self._label(reg_no)}", COL_OK))
            else:
                overlays.append((*box, f"{self._label(reg_no)}: slowly turn head left and right", COL_CHALLENGE))

        hud = [f"Marked this session: {len(self.marked)}"]
        return overlays, events, hud


# ======================================================================= enrolment
@dataclass
class EnrollResult:
    vectors: list
    crops: list


def _pose_check(pose: str, yaw: float) -> tuple[bool, str]:
    if pose == "front":
        return (abs(yaw) <= config.YAW_FRONT), "Face the camera directly"
    if pose == "any":
        return (abs(yaw) <= config.YAW_ANY), "Face roughly towards the camera"
    sign = 1.0 if pose == "left" else -1.0          # camera-right (+) is the person's left
    side = "LEFT" if pose == "left" else "RIGHT"
    signed = yaw * sign
    if signed < config.YAW_TURN_MIN:
        return False, f"Turn a little more to your {side}"
    if signed > config.YAW_TURN_MAX:
        return False, "Not that far - turn back slightly"
    return True, ""


class EnrollmentSession:
    """Captures one sample per prompt, but only when the frame is good: one face, large,
    inside the frame, sharp, evenly lit, the requested pose, held steady, and different
    enough from the previous samples."""

    def __init__(self, engine, prompts=None):
        self.engine = engine
        self.prompts = list(prompts or config.ENROLL_PROMPTS)
        self.vectors: list[np.ndarray] = []
        self.crops: list[np.ndarray] = []
        self._stable = 0
        self._last_capture = -1e9

    def _check(self, frame: np.ndarray, face: np.ndarray, pose: str):
        height, width = frame.shape[:2]
        x, y, fw, fh = _box(face)
        if float(face[14]) < config.ENROLL_MIN_SCORE:
            return False, "Face not clear - hold still", None
        if fw < config.ENROLL_MIN_FACE:
            return False, "Move closer to the camera", None
        if x < 0 or y < 0 or x + fw > width or y + fh > height:
            return False, "Keep your whole face inside the picture", None
        aligned = self.engine.align(frame, face)
        sharpness, brightness = image_quality(aligned)
        if brightness < config.ENROLL_MIN_BRIGHT:
            return False, "Too dark - face a light source", None
        if brightness > config.ENROLL_MAX_BRIGHT:
            return False, "Too bright / glare - avoid direct light", None
        if sharpness < config.ENROLL_MIN_SHARPNESS:
            return False, "Image blurry - hold still / clean the lens", None
        ok, message = _pose_check(pose, head_yaw(face))
        if not ok:
            return False, message, None
        return True, "Hold still...", aligned

    def process(self, frame: np.ndarray, faces: np.ndarray, now: Optional[float] = None):
        """Returns (overlays, hud, result). result is an EnrollResult once all samples exist."""
        now = time.monotonic() if now is None else now
        total = len(self.prompts)
        if len(self.vectors) >= total:
            return [], [], EnrollResult(self.vectors, self.crops)
        prompt, pose = self.prompts[len(self.vectors)]
        hud = [f"Sample {len(self.vectors) + 1} of {total}", prompt]

        if len(faces) != 1:
            self._stable = 0
            note = "No face detected" if len(faces) == 0 else "Only one person may be in view"
            return [(*_box(f), "", COL_BAD) for f in faces], hud + [note], None

        face = faces[0]
        ok, message, aligned = self._check(frame, face, pose)
        if not ok:
            self._stable = 0
        else:
            self._stable += 1
            if self._stable >= config.ENROLL_STABLE_FRAMES and now - self._last_capture >= config.ENROLL_COOLDOWN:
                vector = self.engine.embed_aligned(aligned)
                self._stable = 0
                if any(float(v @ vector) > config.ENROLL_MAX_SIMILARITY for v in self.vectors):
                    message = "Change your pose or expression slightly"
                else:
                    self.vectors.append(vector)
                    self.crops.append(aligned.copy())
                    self._last_capture = now
                    message = "Captured!"

        overlays = [(*_box(face), "", COL_OK if ok else COL_WAIT)]
        result = EnrollResult(self.vectors, self.crops) if len(self.vectors) == total else None
        return overlays, hud + [message], result
