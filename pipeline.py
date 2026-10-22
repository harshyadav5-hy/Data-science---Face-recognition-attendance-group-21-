"""Per-frame logic with no Qt and no camera, so it can be unit-tested.

* AttendanceProcessor - tracks faces across frames and recognises each one from an AVERAGE of several
  good frames (quality-gated), keeps re-checking the identity so a different person stepping in is
  noticed within a fraction of a second, and then marks attendance in one of two modes:
    LIVE  (default)  physical presence: a recognised face must also pass the live head-turn challenge
                     (liveness.py). A photo / phone screen is recognised but never marked, and a
                     warning ("photo of NAME is being shown") is raised instead.
    PHOTO            a recognised face is marked without a live check (admin-only, method "photo").
* EnrollmentSession   - guided capture of varied, sharp, well-lit samples for one person.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
import liveness
from face_engine import Gallery, head_yaw, image_quality
from liveness import LivenessChallenge

# Overlay = (x, y, w, h, text, "#rrggbb") in raw camera coordinates.
COL_OK = "#22c55e"
COL_WAIT = "#eab308"
COL_BAD = "#ef4444"
COL_CHALLENGE = "#f97316"
COL_DIM = "#9ca3af"


def _box(face: np.ndarray) -> tuple[float, float, float, float]:
    return float(face[0]), float(face[1]), float(face[2]), float(face[3])


# ======================================================================= attendance
@dataclass
class SpoofAlert:
    """A blocked spoofing attempt, for the GUI banner and the audit log (see pop_alerts)."""
    reg_no: str
    name: str
    kind: str        # liveness.PHOTO | NOT_LIVE | SWAP
    message: str     # full sentence for the banner
    detail: str      # measured numbers, for the audit log
    at: float


def spoof_message(verdict: str, label: str) -> str:
    if verdict == liveness.PHOTO:
        return f"PHOTO DETECTED: a photo of {label} is being shown - attendance NOT marked"
    if verdict == liveness.NOT_LIVE:
        return f"NOT LIVE: possibly a photo of {label} (no head movement) - attendance NOT marked"
    return f"FACE CHANGED during the live check ({label}) - attendance NOT marked"


_SHORT = {liveness.PHOTO: "PHOTO DETECTED", liveness.NOT_LIVE: "NOT LIVE - photo?",
          liveness.SWAP: "FACE CHANGED"}


MODE_LIVE, MODE_PHOTO = "live", "photo"


class Track:
    _counter = 0

    def __init__(self, face: np.ndarray, now: float):
        Track._counter += 1
        self.id = Track._counter
        self.last_check = -1e9
        self.reset_identity()
        self.update_geometry(face, now)

    def reset_identity(self) -> None:
        """Forget who this is: a different face stepped in, or recognition has to start over."""
        self.embs: deque = deque(maxlen=config.FUSE_FRAMES)    # embeddings of recent GOOD frames
        self.cand: Optional[str] = None                        # identity the averaged embedding points to
        self.cand_run = 0                                      # ... for this many decisions in a row
        self.unknown_run = 0
        self.identity: Optional[str] = None                    # locked identity
        self.score = 0.0
        self.marked = False
        self.live = False
        self.challenge: Optional[LivenessChallenge] = None
        self.drift = 0                                         # consecutive checks that disagree with `identity`

    def fused(self) -> np.ndarray:
        vec = np.mean(np.asarray(self.embs, np.float32), axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def update_geometry(self, face: np.ndarray, now: float) -> None:
        x, y, w, h = _box(face)
        self.cx, self.cy, self.w = x + w / 2, y + h / 2, w
        self.last_seen = now


class AttendanceProcessor:
    def __init__(self, engine, gallery: Gallery, already_marked: set[str],
                 require_liveness: bool = True, rng: Optional[random.Random] = None,
                 mode: Optional[str] = None):
        """mode: MODE_LIVE (default) or MODE_PHOTO.  `require_liveness` is kept only so older callers
        still work: passing False does NOT switch the live check off - choose MODE_PHOTO explicitly."""
        mode = MODE_LIVE if mode is None else mode
        if mode not in (MODE_LIVE, MODE_PHOTO):
            raise ValueError(f"Unknown attendance mode: {mode!r}")
        if mode == MODE_PHOTO and not config.ALLOW_PHOTO_MODE:
            raise ValueError("Photo mode is disabled (ALLOW_PHOTO_MODE = False in config.py).")
        self.engine = engine
        self.gallery = gallery
        self.marked = set(already_marked)
        self.mode = mode
        self.require_liveness = mode == MODE_LIVE
        self.tracks: list[Track] = []
        self._rng = rng or random.Random()
        self.alerts: list[SpoofAlert] = []
        self._last_alert: dict[tuple[str, str], float] = {}
        self._warning: Optional[tuple[str, float]] = None      # (text, show until)
        self._notice: Optional[tuple[str, float]] = None       # e.g. "New face - re-identifying"

    def pop_alerts(self) -> list[SpoofAlert]:
        """Blocked spoofing attempts since the last call (rate-limited per person and kind)."""
        alerts, self.alerts = self.alerts, []
        return alerts

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

    # ------------------------------------------------------------ recognition
    @staticmethod
    def _quality_problem(aligned: np.ndarray, face: np.ndarray) -> Optional[str]:
        """Why this frame is not good enough to recognise from (None = good)."""
        if abs(head_yaw(face)) > config.RECOG_MAX_YAW:
            return "Face the camera"
        sharpness, brightness = image_quality(aligned)
        if brightness < config.RECOG_MIN_BRIGHT:
            return "Too dark - need more light"
        if brightness > config.RECOG_MAX_BRIGHT:
            return "Too bright / glare"
        if sharpness < config.RECOG_MIN_SHARPNESS:
            return "Blurry - hold still"
        return None

    def _face_changed(self, track: Track, now: float) -> None:
        """The face on this track is no longer the locked person: start over immediately."""
        old = track.identity
        track.reset_identity()
        self._notice = (f"Different face - re-identifying (was {self._label(old)})" if old
                        else "Different face - re-identifying", now + 2.0)

    def _maintain_identity(self, track: Track, emb: np.ndarray, result, now: float) -> None:
        """Called with the embedding of a GOOD frame."""
        if track.identity is not None:
            own = self.gallery.score_of(emb, track.identity)
            other = result.accepted and result.reg_no != track.identity
            track.drift = track.drift + 1 if (own < config.CHANGE_SIM or other) else 0
            if track.drift >= config.CHANGE_FRAMES:
                self._face_changed(track, now)
        if track.identity is not None:
            return
        # identify from the average of the recent good frames
        track.embs.append(emb)
        if len(track.embs) < config.FUSE_MIN:
            return
        fused = self.gallery.match(track.fused())
        if fused.accepted:
            if fused.reg_no == track.cand:
                track.cand_run += 1
            else:
                track.cand, track.cand_run = fused.reg_no, 1
            track.unknown_run = 0
            track.score = fused.score
            if track.cand_run >= config.CONFIRM_FRAMES:
                track.identity = track.cand
                track.live = not self.require_liveness
                track.challenge = None
                track.drift = 0
        else:
            track.cand, track.cand_run = None, 0
            track.unknown_run += 1

    @staticmethod
    def _match_flag(result, reg_no: str) -> Optional[bool]:
        """True: this frame recognised the tracked identity. False: recognised someone else."""
        if result is not None and result.accepted:
            return result.reg_no == reg_no
        return None

    # ------------------------------------------------------------ live check
    def _run_liveness(self, track: Track, face: np.ndarray, result, reg_no: str, now: float):
        challenge = track.challenge
        if (challenge is not None and challenge.state == liveness.FAILED
                and now - challenge.failed_at >= config.LIVENESS_FAIL_DISPLAY):
            challenge = track.challenge = None            # verdict shown long enough: try again
        if challenge is None:
            challenge = track.challenge = LivenessChallenge(now, self._rng)
        if challenge.state == liveness.FAILED:
            return challenge.status()
        status = challenge.update(face, now, self._match_flag(result, reg_no))
        if status.state == liveness.FAILED and status.verdict in liveness.SPOOF_VERDICTS:
            self._raise_alert(reg_no, status, now)
        return status

    def _raise_alert(self, reg_no: str, status, now: float) -> None:
        message = spoof_message(status.verdict, self._label(reg_no))
        self._warning = (message, now + config.SPOOF_BANNER_SECONDS)
        key = (reg_no, status.verdict)
        if now - self._last_alert.get(key, -1e9) >= config.SPOOF_ALERT_COOLDOWN:
            self._last_alert[key] = now
            self.alerts.append(SpoofAlert(reg_no, self.gallery.names.get(reg_no, ""),
                                          status.verdict, message, status.detail, now))

    def _liveness_overlay(self, box, reg_no: str, status):
        if status.state == liveness.FAILED:
            if status.verdict in liveness.SPOOF_VERDICTS:
                return (*box, _SHORT[status.verdict], COL_BAD)
            return (*box, f"{self._label(reg_no)}: live check not completed - try again", COL_WAIT)
        return (*box, f"{self._label(reg_no)}: {status.prompt}", COL_CHALLENGE)

    # ------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, faces: np.ndarray, now: Optional[float] = None):
        """Returns (overlays, events, hud). events = [(reg_no, score)] to record - once each."""
        now = time.monotonic() if now is None else now
        height, width = frame.shape[:2]
        assigned = self._assign(faces, now)
        overlays: list[tuple] = []
        events: list[tuple[str, float]] = []
        prompts: list[str] = []
        debug: list[str] = []

        for i, face in enumerate(faces):
            track = assigned[i]
            x, y, fw, fh = _box(face)
            box = (x, y, fw, fh)

            cut_off = (x < -0.05 * fw or y < -0.05 * fh
                       or x + fw > width + 0.05 * fw or y + fh > height + 0.05 * fh)
            if fw < config.MIN_FACE_PX or cut_off:
                overlays.append((*box, "Come closer / centre your face", COL_DIM))
                continue

            # ---- look at the face (every frame while unmarked; every few tenths of a second once marked,
            #      so that a different person stepping into the same spot is noticed)
            emb = result = reason = None
            if (not track.marked) or (now - track.last_check >= config.MARKED_VERIFY_INTERVAL):
                aligned = self.engine.align(frame, face)
                reason = self._quality_problem(aligned, face)
                if reason is None:
                    emb = self.engine.embed_aligned(aligned)
                    result = self.gallery.match(emb)
                track.last_check = now
            if emb is not None:
                self._maintain_identity(track, emb, result, now)

            # ---- not (yet) identified
            if track.identity is None:
                if reason is not None:
                    overlays.append((*box, reason, COL_WAIT))
                elif track.cand is not None:
                    overlays.append((*box, "Identifying...", COL_WAIT))
                elif track.unknown_run >= 2:
                    overlays.append((*box, "Unknown", COL_BAD))
                else:
                    overlays.append((*box, "Scanning...", COL_WAIT))
                continue

            reg_no = track.identity
            if reg_no in self.marked:                       # marked via another track / earlier
                track.marked = True
            if track.marked:
                overlays.append((*box, f"\u2713 {self._label(reg_no)}", COL_OK))
                continue

            status = None
            if self.require_liveness:
                if fw < config.LIVENESS_MIN_FACE_PX:
                    # the live check measures small facial movements: it needs enough pixels to do so
                    overlays.append((*box, f"{self._label(reg_no)}: come closer for the live check", COL_WAIT))
                    continue
                status = self._run_liveness(track, face, result, reg_no, now)
                track.live = status.state == liveness.PASSED

            if track.live:
                self.marked.add(reg_no)
                track.marked = True
                events.append((reg_no, track.score))
                overlays.append((*box, f"\u2713 {self._label(reg_no)}", COL_OK))
            else:
                overlays.append(self._liveness_overlay(box, reg_no, status))
                if status.prompt:
                    prompts.append(f"{self.gallery.names.get(reg_no, reg_no)}: {status.prompt}")
                if config.LIVENESS_DEBUG and status.debug:
                    debug.append(status.debug)

        hud = []
        if self._warning is not None:
            if now < self._warning[1]:
                hud.append(self._warning[0])
            else:
                self._warning = None
        if self._notice is not None:
            if now < self._notice[1]:
                hud.append(self._notice[0])
            else:
                self._notice = None
        hud.extend(prompts[:3])
        hud.extend(debug[:2])
        hud.append(f"Marked this session: {len(self.marked)}")
        hud.append("Mode: LIVE - photos are rejected" if self.require_liveness
                   else "Mode: PHOTO - photos are accepted (no live check)")
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
