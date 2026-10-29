"""Anti-photo liveness: a random head-turn challenge judged on 3-D geometry. No Qt, no camera.

Why a photo cannot pass
-----------------------
`face_frame()` (face_engine.py) measures the nose position inside the face's own eye/mouth frame.
Moving, rolling, scaling or tilting a FLAT picture never changes that position; only a real head
turning does. The challenge therefore asks for:

    1. look straight (neutral pose is measured),
    2. turn to one side (random side), 3. turn to the other side,
    4. look straight again.

Two kinds of failure are reported, so the operator can tell them apart:
    "photo"   - strong evidence: the face got narrower (picture tilted away) but the nose did not
                move inside it. A flat picture is being tilted.
    "timeout" - the challenge was not completed in time (picture held still, or the person did
                not follow the prompts). Reported as "possible photo".
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config


@dataclass(frozen=True)
class Verdict:
    state: str      # "running" | "passed" | "photo" | "timeout"
    stage: str      # "neutral" | "turn" | "verify" | "done"
    prompt: str
    debug: str = ""


class LivenessChallenge:
    def __init__(self, now: float, rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        first = rng.choice((1, -1))             # +1 = person's LEFT (nose to camera-right)
        self.steps = [first, -first]
        self.t0 = now
        self.state = "running"
        self._reset_progress()

    def _reset_progress(self) -> None:
        self.stage = "neutral"
        self.step = 0
        self.u0: Optional[float] = None
        self.r0: Optional[float] = None
        self._neutral: list[tuple[float, float]] = []
        self._hold = self._flat = self._verify = 0
        self._recent_u: deque = deque(maxlen=3)
        self._recent_r: deque = deque(maxlen=3)
        self._last_u: Optional[float] = None
        self._debug = ""

    # ------------------------------------------------------------------ helpers
    def _prompt(self) -> str:
        if self.stage == "neutral":
            return "look straight at the camera"
        if self.stage == "turn":
            side = "LEFT" if self.steps[self.step] > 0 else "RIGHT"
            return f"turn your head slowly to your {side} ({self.step + 1}/{len(self.steps)})"
        if self.stage == "verify":
            return "now look straight again"
        return ""

    def _verdict(self) -> Verdict:
        return Verdict(self.state, self.stage, self._prompt(), self._debug if config.LIVENESS_DEBUG else "")

    # ------------------------------------------------------------------ main entry
    def update(self, geom: Optional[tuple[float, float, float]], now: float) -> Verdict:
        """geom = face_frame(face) of the current frame (or None). Returns the current verdict."""
        if self.state != "running":
            return self._verdict()
        if now - self.t0 > config.LIVENESS_TIMEOUT:
            self.state = "timeout"
            return self._verdict()
        if geom is None:
            return self._verdict()
        u, _v, r = geom

        # A nose that teleports between two frames means the picture was swapped: start over.
        # (The overall time limit is NOT reset, so this cannot be used to gain extra time.)
        if self._last_u is not None and abs(u - self._last_u) > config.LIVENESS_JUMP:
            self._reset_progress()
        self._last_u = u
        self._recent_u.append(u)
        self._recent_r.append(r)
        u_s = float(np.median(self._recent_u))          # median-of-3: one glitchy frame cannot pass
        r_s = float(np.median(self._recent_r))

        if self.stage == "neutral":
            # the straight-ahead baseline must be STEADY, otherwise a head that is still moving
            # would give a biased baseline and make the later turns look smaller than they are
            self._neutral.append((u, r))
            del self._neutral[:-config.LIVENESS_NEUTRAL_FRAMES]
            us = [a for a, _ in self._neutral]
            if (len(us) == config.LIVENESS_NEUTRAL_FRAMES
                    and max(abs(a) for a in us) <= config.LIVENESS_NEUTRAL_MAX
                    and max(us) - min(us) <= config.LIVENESS_NEUTRAL_SPREAD):
                self.u0 = float(np.mean(us))
                self.r0 = float(np.median([b for _, b in self._neutral]))
                self.stage = "turn"
            return self._verdict()

        shift = u_s - self.u0
        flat = r_s / self.r0
        self._debug = f"dU={shift:+.2f} f={flat:.2f}"

        # Flat picture tilted away: face narrows, nose does not move inside it.
        if flat <= config.LIVENESS_FLAT_RATIO and abs(shift) <= config.LIVENESS_FLAT_MAX_SHIFT:
            self._flat += 1
        else:
            self._flat = 0
        if self._flat >= config.LIVENESS_FLAT_FRAMES:
            self.state = "photo"
            self.stage = "done"
            return self._verdict()

        if self.stage == "turn":
            if shift * self.steps[self.step] >= config.LIVENESS_TURN:
                self._hold += 1
            else:
                self._hold = 0
            if self._hold >= config.LIVENESS_HOLD_FRAMES:
                self.step += 1
                self._hold = 0
                if self.step >= len(self.steps):
                    self.stage = "verify"
        elif self.stage == "verify":
            if abs(shift) <= config.LIVENESS_NEUTRAL_MAX + 0.03:
                self._verify += 1
            else:
                self._verify = 0
            if self._verify >= config.LIVENESS_VERIFY_FRAMES:
                self.state = "passed"
                self.stage = "done"
        return self._verdict()
