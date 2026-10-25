"""Per-frame logic with no Qt and no camera, so it can be unit-tested.

* AttendanceProcessor - tracks faces across frames and marks a student only when
    1. the face is recognised (several agreeing frames),
    2. LIVE mode: it is a real person - not a photo/screen (passive detector) - who also passes a
       random head-turn challenge measured with a photo-proof cue (face_engine.face_turn),
    3. it is still the same face that was recognised (face-swap detection resets the track at once).
  PHOTO mode (admin option) skips step 2.
* EnrollmentSession   - guided capture of varied, sharp, well-lit samples for one person.
"""
from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from face_engine import Gallery, face_turn, head_yaw, image_quality

# Overlay = (x, y, w, h, text, "#rrggbb") in raw camera coordinates.
COL_OK = "#22c55e"
COL_WAIT = "#eab308"
COL_BAD = "#ef4444"
COL_CHALLENGE = "#f97316"
COL_DIM = "#9ca3af"
COL_PHOTO = "#ff1f6e"
BANNER_SECONDS = 6.0


def _box(face: np.ndarray) -> tuple[float, float, float, float]:
    return float(face[0]), float(face[1]), float(face[2]), float(face[3])


@dataclass(frozen=True)
class SpoofAlert:
    reg_no: Optional[str]    # whose photo it is, when the face was recognised
    label: str               # "Name (REG)" or "" when unknown
    score: float             # mean live-probability that triggered the alert


# ======================================================================= attendance
class Track:
    _counter = 0

    def __init__(self, face: np.ndarray, now: float):
        Track._counter += 1
        self.id = Track._counter
        self.reset_identity()
        self.update_geometry(face, now)

    def reset_identity(self) -> None:
        """Forget everything learned about the face in this box (a different face appeared)."""
        self.history: deque = deque(maxlen=config.CONFIRM_WINDOW)    # (reg_no | None, score)
        self.embeddings: deque = deque(maxlen=4)                      # recent embeddings: continuity check
        self.swap_strikes = 0
        self.real_hist: deque = deque(maxlen=config.ANTISPOOF_WINDOW)  # live-probabilities
        self.verdict = "checking"                                     # checking | real | photo
        self.photo_since: Optional[float] = None
        self.named: Optional[str] = None                              # last confirmed identity
        self.marked = False
        self.marked_reg: Optional[str] = None
        self.reset_challenge()

    def reset_challenge(self) -> None:
        self.steps: Optional[list] = None
        self.step_idx = 0
        self.front_run = 0
        self.hold = 0
        self.armed = False
        self.challenge_t0: Optional[float] = None
        self.challenge_for: Optional[str] = None
        self.done = False

    def update_geometry(self, face: np.ndarray, now: float) -> None:
        x, y, w, h = _box(face)
        self.cx, self.cy, self.w = x + w / 2, y + h / 2, w
        self.last_seen = now


class AttendanceProcessor:
    """require_liveness=True  -> LIVE mode (photos/screens rejected, head-turn challenge).
    require_liveness=False -> PHOTO mode (anything recognised is marked; admin option).
    antispoof_required=True makes LIVE mode refuse to mark anything while the photo detector is missing."""

    def __init__(self, engine, gallery: Gallery, already_marked: set[str], require_liveness: bool = True,
                 antispoof=None, antispoof_required: bool = False, rng: Optional[random.Random] = None):
        self.engine = engine
        self.gallery = gallery
        self.marked = set(already_marked)
        self.require_liveness = require_liveness
        self.antispoof = antispoof
        self.antispoof_required = antispoof_required
        self.rng = rng or random.Random()
        self.tracks: list[Track] = []
        self.alerts: list[SpoofAlert] = []
        self.photo_attempts = 0
        self._alert_last: dict[str, float] = {}
        self._banner: Optional[tuple[str, float]] = None

    # ------------------------------------------------------------------ helpers
    def _label(self, reg_no: Optional[str]) -> str:
        if not reg_no:
            return ""
        name = self.gallery.names.get(reg_no)
        return f"{name} ({reg_no})" if name else reg_no

    def pop_alerts(self) -> list[SpoofAlert]:
        alerts, self.alerts = self.alerts, []
        return alerts

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

    # ------------------------------------------------------------------ face-swap detection
    def _check_swap(self, track: Track, emb: np.ndarray) -> str:
        """Compare this frame's face with the recent frames of the same box.
        'ok'    - same face;  'skip' - suspicious frame (ignored, could be blur);
        'reset' - the face really changed: the track was wiped and starts again from this frame."""
        state = "ok"
        if track.embeddings:
            ref = np.mean(track.embeddings, axis=0)
            norm = float(np.linalg.norm(ref))
            sim = float(emb @ ref) / norm if norm > 0 else 1.0
            if sim < config.SWAP_SIMILARITY:
                track.swap_strikes += 1
                if track.swap_strikes < config.SWAP_FRAMES:
                    return "skip"
                track.reset_identity()
                state = "reset"
            else:
                track.swap_strikes = 0
        track.embeddings.append(emb)
        return state

    def _current_identity(self, track: Track) -> tuple[Optional[str], float]:
        """Identity backed by enough of the recent frames (recomputed every frame, never sticky)."""
        votes = Counter(reg for reg, _ in track.history if reg)
        if not votes:
            return None, 0.0
        reg, count = votes.most_common(1)[0]
        if count < config.CONFIRM_FRAMES:
            return None, 0.0
        return reg, float(np.mean([s for r, s in track.history if r == reg]))

    # ------------------------------------------------------------------ passive photo / screen verdict
    def _update_verdict(self, track: Track, frame: np.ndarray, face: np.ndarray) -> Optional[float]:
        if self.antispoof is None:
            track.verdict = "real"          # only reachable when ANTISPOOF_REQUIRED is off (challenge-only)
            return None
        track.real_hist.append(float(self.antispoof.real_probability(frame, face)))
        mean = float(np.mean(track.real_hist))
        if len(track.real_hist) >= config.ANTISPOOF_MIN_FRAMES:
            if mean >= config.ANTISPOOF_REAL_THRESHOLD:
                track.verdict = "real"
            elif mean < config.ANTISPOOF_REAL_THRESHOLD - config.ANTISPOOF_HYSTERESIS:
                track.verdict = "photo"
            # in between: keep the previous verdict (hysteresis stops flicker)
        return mean

    def _flag_photo(self, track: Track, now: float, mean: Optional[float]) -> str:
        """Block the face, restart its challenge, and raise one rate-limited warning."""
        if track.photo_since is None:
            track.photo_since = now
        track.reset_challenge()
        who = track.named
        key = who or "?"
        ready = who is not None or now - track.photo_since >= 1.0   # wait briefly so the name is known
        if ready and now - self._alert_last.get(key, -1e9) >= config.ANTISPOOF_ALERT_COOLDOWN:
            self._alert_last[key] = now
            self.photo_attempts += 1
            self.alerts.append(SpoofAlert(who, self._label(who), float(mean if mean is not None else 0.0)))
            text = (f"WARNING: photo of {self._label(who)} shown - NOT marked" if who
                    else "WARNING: photo / screen shown - NOT marked")
            self._banner = (text, now + BANNER_SECONDS)
        if who:
            return f"PHOTO of {self._label(who)} - NOT MARKED"
        return "PHOTO / SCREEN DETECTED - NOT MARKED"

    # ------------------------------------------------------------------ head-turn challenge
    def _challenge(self, track: Track, face: np.ndarray, reg: Optional[str], now: float) -> tuple[bool, str]:
        """Random left/right head-turn. Uses face_turn(), which a flat photo cannot change however
        it is tilted or moved. Returns (finished, prompt)."""
        if reg is not None and track.challenge_for not in (None, reg):
            track.reset_challenge()                                  # a different person: start over
        if track.steps is None:
            if reg is None:
                return False, "Look at the camera"
            order = ["left", "right"]
            self.rng.shuffle(order)
            track.steps = order[:max(1, min(2, config.LIVENESS_STEPS))]
            track.challenge_for = reg
            track.challenge_t0 = now
        elif now - track.challenge_t0 > config.LIVENESS_TIMEOUT:
            track.reset_challenge()
            return False, "Time is up - look at the camera, then try again"

        turn = face_turn(face)
        if not track.armed:                                          # must be seen facing the camera first
            track.front_run = track.front_run + 1 if abs(turn) <= config.LIVENESS_FRONT_MAX else 0
            if track.front_run < 2:
                return False, "Look straight at the camera"
            track.armed = True

        sign = 1.0 if track.steps[track.step_idx] == "left" else -1.0   # camera-right (+) = person's left
        if config.LIVENESS_TURN_MIN <= turn * sign <= config.LIVENESS_TURN_MAX:
            track.hold += 1
        else:
            track.hold = 0
        if track.hold >= config.LIVENESS_HOLD_FRAMES:
            track.step_idx += 1
            track.hold = 0
            if track.step_idx >= len(track.steps):
                track.done = True
                return True, ""
        side = track.steps[track.step_idx].upper()
        return False, (f"Slowly turn your head to your {side}" if track.step_idx == 0
                       else f"Good - now turn to your {side}")

    # ------------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, faces: np.ndarray, now: Optional[float] = None):
        """Returns (overlays, events, hud). events = [(reg_no, score)] to record - once each."""
        now = time.monotonic() if now is None else now
        live = self.require_liveness
        if live and self.antispoof_required and self.antispoof is None:
            return ([(*_box(f), "Photo check unavailable", COL_BAD) for f in faces], [],
                    ["Photo detector not loaded - attendance PAUSED"])

        height, width = frame.shape[:2]
        assigned = self._assign(faces, now)
        overlays: list[tuple] = []
        events: list[tuple[str, float]] = []

        for i, face in enumerate(faces):
            track = assigned[i]
            x, y, fw, fh = box = _box(face)
            cut_off = (x < -0.05 * fw or y < -0.05 * fh
                       or x + fw > width + 0.05 * fw or y + fh > height + 0.05 * fh)
            if fw < config.MIN_FACE_PX or cut_off:
                overlays.append((*box, "Come closer / centre your face", COL_DIM))
                continue

            emb = self.engine.embed_aligned(self.engine.align(frame, face))
            if self._check_swap(track, emb) == "skip":
                overlays.append((*box, "Verifying...", COL_WAIT))
                continue

            result = self.gallery.match(emb)
            track.history.append((result.reg_no if result.accepted else None, result.score))
            reg, score = self._current_identity(track)
            if reg:
                track.named = reg
            if track.marked and reg and reg != track.marked_reg:      # someone else is now in this box
                track.reset_identity()
                track.history.append((result.reg_no if result.accepted else None, result.score))
                track.embeddings.append(emb)
                reg, score = None, 0.0
            debug = f"  [sim {result.score:.2f}]" if config.SHOW_DEBUG_SCORES else ""

            if track.marked:
                overlays.append((*box, f"\u2713 {self._label(track.marked_reg)}{debug}", COL_OK))
                continue

            # ---- LIVE mode: is this a real, present person?
            mean_real = None
            if live:
                mean_real = self._update_verdict(track, frame, face)
                if config.SHOW_DEBUG_SCORES and mean_real is not None:
                    debug += f" [live {mean_real:.2f}]"
                if track.verdict == "photo":
                    overlays.append((*box, self._flag_photo(track, now, mean_real) + debug, COL_PHOTO))
                    continue
                track.photo_since = None

            target = reg or track.challenge_for
            if target is None:
                if any(r for r, _ in track.history):
                    overlays.append((*box, f"Identifying...{debug}", COL_WAIT))
                elif len(track.history) >= 3:
                    overlays.append((*box, f"Unknown{debug}", COL_BAD))
                else:
                    overlays.append((*box, f"Scanning...{debug}", COL_WAIT))
                continue

            if reg and reg in self.marked:                            # marked via another track / earlier
                track.marked, track.marked_reg = True, reg
                overlays.append((*box, f"\u2713 {self._label(reg)} (already marked)", COL_OK))
                continue

            if live:
                if track.verdict != "real":
                    overlays.append((*box, f"Checking you are live...{debug}", COL_WAIT))
                    continue
                if not track.done:
                    _, prompt = self._challenge(track, face, reg, now)
                    overlays.append((*box, f"{self._label(target)}: {prompt}{debug}", COL_CHALLENGE))
                    continue
                if reg != track.challenge_for:                        # finished turning - re-confirm identity
                    overlays.append((*box, f"{self._label(target)}: look at the camera{debug}", COL_WAIT))
                    continue

            # ---- all checks passed
            self.marked.add(reg)
            track.marked, track.marked_reg = True, reg
            events.append((reg, score))
            overlays.append((*box, f"\u2713 {self._label(reg)}{debug}", COL_OK))

        hud = ["LIVE mode - photos and screens rejected" if live else "PHOTO mode - photos are accepted",
               f"Marked this session: {len(self.marked)}"]
        if self.photo_attempts:
            hud.append(f"Photo attempts blocked: {self.photo_attempts}")
        if self._banner and now < self._banner[1]:
            hud.insert(0, self._banner[0])
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
