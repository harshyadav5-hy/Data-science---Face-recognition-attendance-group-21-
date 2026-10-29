"""Per-frame logic with no Qt and no camera, so it can be unit-tested.

* AttendanceProcessor - tracks faces across frames, identifies them from several agreeing frames
  (quality-gated, multi-frame fused), then REQUIRES a live 3-D head-turn challenge before the
  attendance is reported. A photo / flat picture never passes: it is flagged and the person it
  shows is named in a warning. Already-marked faces are re-checked, so a swap is noticed fast.
* EnrollmentSession   - guided capture of varied, sharp, well-lit samples for one person.
"""
from __future__ import annotations

import time
from collections import Counter, deque, namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from face_engine import Gallery, face_frame, head_yaw, image_quality
from liveness import LivenessChallenge

# Overlay = (x, y, w, h, text, "#rrggbb") in raw camera coordinates.
COL_OK = "#22c55e"
COL_WAIT = "#eab308"
COL_BAD = "#ef4444"
COL_CHALLENGE = "#f97316"
COL_DIM = "#9ca3af"

# kind: "photo" (strong evidence) | "timeout" (no live movement -> possible photo)
SpoofAlert = namedtuple("SpoofAlert", "kind reg_no text box")


def _box(face: np.ndarray) -> tuple[float, float, float, float]:
    return float(face[0]), float(face[1]), float(face[2]), float(face[3])


# ======================================================================= attendance
class Track:
    _counter = 0

    def __init__(self, face: np.ndarray, now: float):
        Track._counter += 1
        self.id = Track._counter
        self.history: deque = deque(maxlen=config.CONFIRM_WINDOW)   # (reg_no | None, score)
        self.embs: deque = deque(maxlen=config.FUSE_FRAMES)
        self.near: deque = deque(maxlen=config.NEAR_MISS_FRAMES)    # (reg_no | None, score)
        self.identity: Optional[str] = None
        self.score = 0.0
        self.fused: Optional[np.ndarray] = None
        self.marked = False
        self.challenge: Optional[LivenessChallenge] = None
        self.verifying = False
        self.verify_deadline = 0.0
        self.spoof_until = 0.0
        self.spoof_text = ""
        self.last_check = 0.0
        self.recheck_misses = 0
        self.update_geometry(face, now)

    def update_geometry(self, face: np.ndarray, now: float) -> None:
        x, y, w, h = _box(face)
        self.cx, self.cy, self.w = x + w / 2, y + h / 2, w
        self.last_seen = now

    def reset_liveness(self) -> None:
        self.challenge = None
        self.verifying = False

    def reset_identity(self) -> None:
        self.history.clear()
        self.embs.clear()
        self.near.clear()
        self.identity = None
        self.score = 0.0
        self.fused = None
        self.marked = False
        self.recheck_misses = 0
        self.reset_liveness()


class AttendanceProcessor:
    def __init__(self, engine, gallery: Gallery, already_marked: set[str]):
        self.engine = engine
        self.gallery = gallery
        self.marked = set(already_marked)
        self.tracks: list[Track] = []
        self._alerts: list[SpoofAlert] = []

    def pop_alerts(self) -> list[SpoofAlert]:
        alerts, self._alerts = self._alerts, []
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

    # ------------------------------------------------------------------ recognition
    @staticmethod
    def _good_quality(aligned: np.ndarray) -> bool:
        sharpness, brightness = image_quality(aligned)
        return (sharpness >= config.RECOG_MIN_SHARPNESS
                and config.RECOG_MIN_BRIGHT <= brightness <= config.RECOG_MAX_BRIGHT)

    @staticmethod
    def _fused(track: Track) -> Optional[np.ndarray]:
        if len(track.embs) < config.FUSE_MIN:
            return None
        vec = np.mean(np.asarray(track.embs, np.float32), axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    def _identify(self, track: Track, frame: np.ndarray, face: np.ndarray, frontal: bool) -> str:
        """Adds one vote for this frame. Returns 'ok' | 'turned' | 'quality'."""
        if not frontal:
            return "turned"                      # too far turned to identify reliably
        aligned = self.engine.align(frame, face)
        if not self._good_quality(aligned):
            return "quality"
        emb = self.engine.embed_aligned(aligned)
        result = self.gallery.match(emb)
        track.history.append((result.reg_no if result.accepted else None, result.score))
        track.embs.append(emb)
        track.near.append((None, 0.0) if result.accepted else (result.reg_no, result.score))
        self._update_identity(track)
        return "ok"

    def _update_identity(self, track: Track) -> None:
        votes = Counter(reg for reg, _ in track.history if reg)
        if votes:
            reg, count = votes.most_common(1)[0]
            if count >= config.CONFIRM_FRAMES:
                fused = self._fused(track)
                fres = self.gallery.match(fused) if fused is not None else None
                if fres is None or (fres.accepted and fres.reg_no == reg):
                    if reg != track.identity:        # a different person: start from scratch
                        track.identity = reg
                        track.reset_liveness()
                    track.score = float(np.mean([s for r, s in track.history if r == reg]))
                    track.fused = fused
                    return
        recent = list(track.history)[-config.LOSE_FRAMES:]
        if (track.identity and len(recent) >= config.LOSE_FRAMES
                and all(r != track.identity for r, _ in recent)):
            track.identity = None                    # face changed / person left: withdraw label
            track.fused = None
            track.reset_liveness()

    def _recheck_marked(self, track: Track, frame: np.ndarray, face: np.ndarray,
                        frontal: bool, now: float) -> None:
        """Already-marked faces are verified again every RECHECK_INTERVAL seconds."""
        if not frontal or now - track.last_check < config.RECHECK_INTERVAL:
            return
        track.last_check = now
        aligned = self.engine.align(frame, face)
        if not self._good_quality(aligned):
            return
        result = self.gallery.match(self.engine.embed_aligned(aligned))
        if result.accepted and result.reg_no == track.identity:
            track.recheck_misses = 0
            return
        track.recheck_misses += 1
        other_person = result.accepted and result.reg_no != track.identity
        if other_person or track.recheck_misses >= config.RECHECK_MISSES:
            track.reset_identity()                   # someone else sat down / face changed a lot

    @staticmethod
    def _near_miss(track: Track) -> bool:
        if len(track.near) < config.NEAR_MISS_FRAMES:
            return False
        regs = {r for r, _ in track.near}
        return None not in regs and len(regs) == 1 and min(s for _, s in track.near) >= config.NEAR_MISS_MIN

    # ------------------------------------------------------------------ per-frame
    def process(self, frame: np.ndarray, faces: np.ndarray, now: Optional[float] = None):
        """Returns (overlays, events, hud).
        events = [(reg_no, score, fused_embedding)] - reported once per person, only after liveness.
        Spoof alerts are collected separately: see pop_alerts()."""
        now = time.monotonic() if now is None else now
        height, width = frame.shape[:2]
        assigned = self._assign(faces, now)
        overlays: list[tuple] = []
        events: list[tuple] = []

        for i, face in enumerate(faces):
            track = assigned[i]
            box = _box(face)
            x, y, fw, fh = box

            cut_off = (x < -0.05 * fw or y < -0.05 * fh
                       or x + fw > width + 0.05 * fw or y + fh > height + 0.05 * fh)
            if fw < config.MIN_FACE_PX or cut_off:
                overlays.append((*box, "Come closer / centre your face", COL_DIM))
                continue

            geom = face_frame(face)
            frontal = geom is not None and abs(geom[0]) <= config.RECOG_MAX_YAW

            if track.marked:
                self._recheck_marked(track, frame, face, frontal, now)
                if track.marked:
                    overlays.append((*box, f"\u2713 {self._label(track.identity)}", COL_OK))
                    continue

            if now < track.spoof_until:
                overlays.append((*box, track.spoof_text, COL_BAD))
                continue

            status = self._identify(track, frame, face, frontal)

            if track.identity is None:
                if any(reg for reg, _ in track.history):
                    overlays.append((*box, "Identifying...", COL_WAIT))
                elif self._near_miss(track):
                    overlays.append((*box, "Not recognised - face changed? Ask admin: Update look", COL_BAD))
                elif len(track.history) >= 3:
                    overlays.append((*box, "Unknown", COL_BAD))
                elif status == "turned":
                    overlays.append((*box, "Face the camera", COL_WAIT))
                elif status == "quality":
                    overlays.append((*box, "Hold still / improve lighting", COL_WAIT))
                else:
                    overlays.append((*box, "Scanning...", COL_WAIT))
                continue

            reg_no = track.identity
            label = self._label(reg_no)
            if reg_no in self.marked:                       # marked via another track / earlier
                track.marked = True
                track.last_check = now
                overlays.append((*box, f"\u2713 {label} (already marked)", COL_OK))
                continue

            # ---- mandatory liveness challenge (never skipped)
            if track.challenge is None:
                track.challenge = LivenessChallenge(now)
                track.verifying = False
            verdict = track.challenge.update(geom, now)
            debug = f"  [{verdict.debug}]" if verdict.debug else ""

            if verdict.state == "running":
                if verdict.stage == "verify" and not track.verifying:
                    track.verifying = True               # identity must be re-confirmed afterwards
                    track.history.clear()
                    track.embs.clear()
                    track.verify_deadline = now + config.VERIFY_TIMEOUT
                overlays.append((*box, f"{label}: {verdict.prompt}{debug}", COL_CHALLENGE))

            elif verdict.state == "passed":
                confirmed = Counter(r for r, _ in track.history if r).get(reg_no, 0) >= config.CONFIRM_FRAMES
                if confirmed and track.identity == reg_no:
                    self.marked.add(reg_no)
                    track.marked = True
                    track.last_check = now
                    track.recheck_misses = 0
                    events.append((reg_no, track.score, track.fused))
                    overlays.append((*box, f"\u2713 {label}", COL_OK))
                else:
                    if now > track.verify_deadline:
                        track.reset_liveness()           # could not re-confirm: run a fresh challenge
                    overlays.append((*box, f"{label}: verifying...", COL_WAIT))

            else:                                        # "photo" or "timeout"
                if verdict.state == "photo":
                    text = f"PHOTO of {label} shown - NOT MARKED"
                else:
                    text = f"No live movement - possible PHOTO of {label} - NOT MARKED"
                track.spoof_text = text
                track.spoof_until = now + config.SPOOF_RED_SECONDS
                track.reset_liveness()
                self._alerts.append(SpoofAlert(verdict.state, reg_no, text, box))
                overlays.append((*box, text, COL_BAD))

        hud = [f"Marked this session: {len(self.marked)}", "Anti-photo liveness check: ON"]
        return overlays, events, hud


def adaptive_template_ok(gallery: Gallery, reg_no: str, embedding, score: float) -> bool:
    """Should this live-verified match be learned as a new template for `reg_no`?
    Only confident matches that show a genuinely NEW view (glasses, beard, light...) and are not
    ambiguous with another student."""
    if not config.ADAPT_ENABLED or embedding is None or reg_no not in gallery.ids:
        return False
    if score < config.ADAPT_MIN_SCORE:
        return False
    if gallery.identity_score(reg_no, embedding, topk=1) >= config.ADAPT_MAX_SIMILARITY:
        return False
    other = gallery.best_other([embedding], exclude=reg_no)
    return other is None or other[1] < config.ADAPT_MAX_OTHER


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
