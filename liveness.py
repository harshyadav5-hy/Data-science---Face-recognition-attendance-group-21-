"""Live-person check (anti-photo) for attendance.  Pure NumPy: no camera, no Qt, no OpenCV.

Why a photo cannot pass
-----------------------
A photo or phone screen is a *flat plane*.  Move it, tilt it, rotate it, hold it closer: the five
face landmarks (two eyes, nose, two mouth corners) always stay related by a flat (affine) transform.
A real head is 3-D: the nose tip sticks out ~3 cm in front of the eye/mouth plane, so when the head
turns the nose slides sideways relative to the eyes and mouth ("parallax").  No flat transform can
fake that slide.  Two numbers are measured against a "look straight" reference:

* parallax      - how far the nose is from where a FLAT face would put it, in eye-distances
* foreshortening- how much narrower the face became (eye distance / eye-to-mouth distance)

A real turn raises BOTH.  A rotated photo raises only foreshortening, which is exactly the
signature used to say "PHOTO DETECTED".

The challenge (randomised every time, so nothing can be pre-recorded)
---------------------------------------------------------------------
  1. baseline - look straight; reference landmarks are stored
  2. turn     - turn to YOUR LEFT and to YOUR RIGHT, in random order
  3. verify   - look straight again; the SAME person must be recognised again
                (closes the "show a photo, then swap in another face" hole)
Attendance is marked only when all three succeed.  Face lost for a moment or a sudden jump in
position/size restarts the challenge, so photos cannot be swapped mid-way, and a turn must sweep
through intermediate angles, so a slideshow of photos that hard-cuts between poses is rejected too.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config

BASELINE, TURN, VERIFY, PASSED, FAILED = "baseline", "turn", "verify", "passed", "failed"

# Verdicts (set only when state == FAILED)
PHOTO = "photo"            # a flat picture was rotated/tilted: face got narrower, nose did not move
NOT_LIVE = "not_live"      # asked to turn, but no 3-D movement at all (still photo / frozen person)
SWAP = "swap"              # the face in front of the camera changed during the check
INCOMPLETE = "incomplete"  # a moving, real-looking face that simply did not finish: try again

SPOOF_VERDICTS = (PHOTO, NOT_LIVE, SWAP)
SIDE_SIGN = {"left": 1.0, "right": -1.0}   # camera-right (+) is the person's LEFT (as in head_yaw)


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Landmarks:
    eyes: np.ndarray    # 2x2, ordered by image x
    mouth: np.ndarray   # 2x2, ordered by image x
    nose: np.ndarray    # (2,)

    @property
    def eye_dist(self) -> float:
        return float(np.linalg.norm(self.eyes[1] - self.eyes[0]))

    @property
    def yaw(self) -> float:
        """Same definition as face_engine.head_yaw: nose offset / inter-eye distance."""
        dx = float(self.eyes[1, 0] - self.eyes[0, 0])
        return (float(self.nose[0]) - float(self.eyes[:, 0].mean())) / dx

    @property
    def width_ratio(self) -> float:
        """Eye distance / eye-to-mouth distance.  Independent of how far the face is from the
        camera; shrinks when the face (real or printed) is turned, changes when tilted up/down."""
        gap = float(np.linalg.norm(self.mouth.mean(axis=0) - self.eyes.mean(axis=0)))
        return self.eye_dist / max(gap, 1e-6)

    @property
    def eye_mid(self) -> np.ndarray:
        return self.eyes.mean(axis=0)


def landmarks(face: np.ndarray) -> Optional[Landmarks]:
    """Landmarks from a YuNet row (x, y, w, h, 5 x (x, y), score), or None if unusable."""
    pts = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
    if not np.all(np.isfinite(pts)):
        return None
    eyes = pts[:2][np.argsort(pts[:2, 0])]
    mouth = pts[3:5][np.argsort(pts[3:5, 0])]
    lm = Landmarks(eyes, mouth, pts[2])
    if eyes[1, 0] - eyes[0, 0] < 2.0 or lm.eye_dist < 2.0:
        return None
    if float(np.linalg.norm(mouth.mean(axis=0) - eyes.mean(axis=0))) < 2.0:
        return None
    return lm


def _mean_landmarks(items: list[Landmarks]) -> Landmarks:
    return Landmarks(np.mean([i.eyes for i in items], axis=0),
                     np.mean([i.mouth for i in items], axis=0),
                     np.mean([i.nose for i in items], axis=0))


def planar_residual(ref: Landmarks, cur: Landmarks) -> np.ndarray:
    """Nose displacement (x, y), in eye-distances, that a FLAT face could not explain.

    Fit the best affine map from the reference eyes+mouth to the current eyes+mouth, use it to
    predict where the nose would be if the face were a flat picture, return actual - predicted.
    ~0 for any photo (however moved/tilted/rotated); grows with a real head turn.
    x > 0 means the nose slid towards camera-right (the person's LEFT)."""
    src = np.hstack([np.vstack([ref.eyes, ref.mouth]), np.ones((4, 1))])
    dst = np.vstack([cur.eyes, cur.mouth])
    coef, *_ = np.linalg.lstsq(src, dst, rcond=None)
    predicted = np.append(ref.nose, 1.0) @ coef
    return (cur.nose - predicted) / cur.eye_dist


# ---------------------------------------------------------------- challenge
@dataclass(frozen=True)
class Status:
    state: str
    prompt: str = ""
    verdict: Optional[str] = None
    detail: str = ""     # numbers for the audit log
    debug: str = ""      # numbers for the on-screen tuning overlay


class LivenessChallenge:
    """One randomised live check for ONE tracked face.  Feed it every frame via update()."""

    def __init__(self, now: float, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()
        self.verdict: Optional[str] = None
        self.failed_at: Optional[float] = None
        self.detail = ""
        self.restarts = 0
        self.metrics: Optional[tuple] = None      # (yaw, parallax_x, parallax_y, narrowing) of the last frame
        self.last_t = now
        self.last_lm: Optional[Landmarks] = None
        self._reset(now)

    # ------------------------------------------------------------ internals
    def _reset(self, now: float) -> None:
        order = ["left", "right"]
        self._rng.shuffle(order)
        self.steps = order[:max(1, min(2, int(config.LIVENESS_STEPS)))]
        self.step = 0
        self.state = BASELINE
        self.t_start = now
        self.t_turn = now
        self._base: list[Landmarks] = []
        self._ref: Optional[Landmarks] = None
        self._ref_ratio = 1.0
        self._buf: deque = deque(maxlen=max(1, int(config.LIVENESS_SMOOTH)))
        self._hold = 0
        self._yaw3: deque = deque(maxlen=3)
        self._approach = 0
        self._approach_first: Optional[float] = None
        self._arrived = False
        self._too_fast = False
        self._photo_run = 0
        self._verify_run = 0
        self.best_par = 0.0
        self.max_fs = 0.0
        self._dbg = ""

    def _discontinuous(self, lm: Landmarks, now: float) -> bool:
        prev = self.last_lm
        if prev is None:
            return False
        if now - self.last_t > config.LIVENESS_MAX_GAP:          # face vanished: photo swapped?
            return True
        size = lm.eye_dist / prev.eye_dist
        jump = float(np.linalg.norm(lm.eye_mid - prev.eye_mid)) / prev.eye_dist
        return not (1 / 1.5 <= size <= 1.5) or jump > 1.5          # teleport: another face/photo

    def _fail(self, now: float, verdict: str) -> Status:
        self.state = FAILED
        self.verdict = verdict
        self.failed_at = now
        self.detail = (f"{verdict}: foreshortening={self.max_fs:.2f} parallax={self.best_par:.2f} "
                       f"turns_done={self.step}/{len(self.steps)}")
        return self.status()

    def _timeout_verdict(self) -> str:
        return SWAP if self.state == VERIFY else INCOMPLETE

    # ------------------------------------------------------------ public
    def status(self) -> Status:
        prompt = ""
        if self.state == BASELINE:
            prompt = "Look straight at the camera"
        elif self.state == TURN:
            side, n = self.steps[self.step].upper(), f"({self.step + 1}/{len(self.steps)})"
            prompt = (f"Too fast - look straight, then turn SLOWLY to YOUR {side} {n}" if self._too_fast
                      else f"Slowly turn your head to YOUR {side} {n}")
        elif self.state == VERIFY:
            prompt = "Look straight at the camera - verifying"
        return Status(self.state, prompt, self.verdict, self.detail, self._dbg)

    def update(self, face: np.ndarray, now: float, match: Optional[bool] = None) -> Status:
        """face: YuNet row.  match: did THIS frame's recognition accept the same identity as the
        track (True), accept a different person (False), or was it inconclusive (None)?"""
        if self.state in (PASSED, FAILED):
            return self.status()
        lm = landmarks(face)
        if lm is None:
            if now - self.t_start > config.LIVENESS_TIMEOUT:
                return self._fail(now, self._timeout_verdict())
            return self.status()

        if self._discontinuous(lm, now):
            self.restarts += 1
            self._reset(now)
        self.last_t, self.last_lm = now, lm
        if now - self.t_start > config.LIVENESS_TIMEOUT:
            return self._fail(now, self._timeout_verdict())

        if self.state == BASELINE:
            if abs(lm.yaw) <= config.LIVENESS_BASELINE_YAW:
                self._base.append(lm)
            else:
                self._base.clear()
            if len(self._base) >= config.LIVENESS_BASELINE_FRAMES:
                self._ref = _mean_landmarks(self._base)
                self._ref_ratio = float(np.mean([b.width_ratio for b in self._base]))
                self.state = TURN
                self.t_turn = now
                self._buf.clear()
                self._hold = self._photo_run = 0
            return self.status()

        # ---- TURN / VERIFY: compare against the straight-ahead reference
        residual = planar_residual(self._ref, lm)
        self._buf.append((lm.yaw, residual[0], residual[1], lm.width_ratio))
        yaw_s, rx_s, ry_s, ratio_s = (float(v) for v in np.mean(self._buf, axis=0))
        ready = len(self._buf) >= min(3, self._buf.maxlen)
        par_mag = math.hypot(rx_s, ry_s)
        fs = abs(ratio_s / self._ref_ratio - 1.0)
        self.metrics = (yaw_s, rx_s, ry_s, fs)
        if ready:
            self.best_par = max(self.best_par, par_mag)
            self.max_fs = max(self.max_fs, fs)
        self._dbg = f"live-check {self.state}: yaw={yaw_s:+.2f} parallax={par_mag:.2f} narrowing={fs:.2f}"

        # Flat face that got narrower/shorter = a picture being rotated or tilted.
        if ready and fs >= config.LIVENESS_PHOTO_FORESHORTEN and par_mag < config.LIVENESS_FLAT_PARALLAX:
            self._photo_run += 1
        else:
            self._photo_run = 0
        if self._photo_run >= config.LIVENESS_PHOTO_FRAMES:
            return self._fail(now, PHOTO)

        if self.state == TURN:
            sign = SIDE_SIGN[self.steps[self.step]]
            # Continuity: a real head sweeps THROUGH intermediate angles on the way to the turned pose;
            # a slideshow of photos (straight -> turned, hard cut) jumps over them.  A 3-frame MEDIAN is
            # used (not the 5-frame mean above): it removes single-frame landmark spikes but keeps hard
            # edges, whereas a mean would smear one cut into a fake ramp.  Only the approach is counted.
            self._yaw3.append(lm.yaw)
            raw = float(np.median(self._yaw3)) * sign
            if raw < config.LIVENESS_RAMP_FROM - 0.02:             # back at (or past) straight
                self._approach, self._approach_first = 0, None
                self._arrived = self._too_fast = False
            elif not self._arrived:
                if raw >= config.LIVENESS_TURN_YAW:
                    self._arrived = True
                elif raw >= config.LIVENESS_RAMP_FROM:
                    self._approach += 1
                    if self._approach_first is None:
                        self._approach_first = raw
            if (ready and yaw_s * sign >= config.LIVENESS_TURN_YAW
                    and rx_s * sign >= config.LIVENESS_MIN_PARALLAX):
                self._hold += 1
            else:
                self._hold = 0
            swept = (self._approach >= config.LIVENESS_RAMP_FRAMES and self._approach_first is not None
                     and yaw_s * sign - self._approach_first >= config.LIVENESS_RAMP_SPAN)
            if self._hold >= config.LIVENESS_HOLD_FRAMES and not swept:
                self._too_fast = True                              # got there without passing through
            if self._hold >= config.LIVENESS_HOLD_FRAMES and swept:
                self.step += 1
                self._hold = self._photo_run = 0
                self._approach, self._approach_first = 0, None
                self._arrived = self._too_fast = False
                if self.step >= len(self.steps):
                    self.state = VERIFY
                    self._verify_run = 0
            elif now - self.t_turn >= config.LIVENESS_STILL_SECONDS and self.best_par < config.LIVENESS_FLAT_PARALLAX:
                return self._fail(now, NOT_LIVE)
            return self.status()

        # ---- VERIFY: looking straight, same person recognised again
        frontal = abs(lm.yaw) <= config.LIVENESS_VERIFY_YAW
        if frontal and match is False:
            return self._fail(now, SWAP)
        self._verify_run = self._verify_run + 1 if (frontal and match is True) else 0
        if self._verify_run >= config.LIVENESS_VERIFY_FRAMES:
            self.state = PASSED
        return self.status()
