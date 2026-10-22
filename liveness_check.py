"""Camera test bench for the live check.  No database, no students, no admin password needed.

    python liveness_check.py                 # default camera
    python liveness_check.py --camera 1 --csv numbers.csv

It runs the same live challenge as the attendance screen on the biggest face in view and shows the
verdict, so you can try it for real:
  * your own face      -> follow the prompts, it must end with  PASSED
  * a photo / phone    -> hold it up, move and tilt it: it must NEVER pass; you should see
                          "PHOTO DETECTED" or "NOT LIVE" (still photo)
Keys: R = restart the challenge, Q or Esc = quit.  --csv saves yaw / parallax / narrowing for every
frame, which is what you need to tune the thresholds in config.py (see README, "Calibrating the live check").
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Optional

import cv2
import numpy as np

import config
import liveness
from liveness import LivenessChallenge

HOLD_RESULT_SECONDS = 3.0
COLOURS = {"ok": (80, 200, 60), "bad": (60, 60, 240), "wait": (40, 200, 240), "info": (255, 255, 255)}


def biggest_face(faces: np.ndarray) -> Optional[np.ndarray]:
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))] if len(faces) else None


def verdict_text(status: liveness.Status) -> tuple[str, str]:
    """(text, colour) describing a finished challenge."""
    if status.state == liveness.PASSED:
        return "PASSED - live person", "ok"
    return {liveness.PHOTO: ("PHOTO DETECTED - a flat picture was moved/rotated", "bad"),
            liveness.NOT_LIVE: ("NOT LIVE - no head movement (still photo?)", "bad"),
            liveness.SWAP: ("FACE CHANGED during the check", "bad"),
            }.get(status.verdict, ("Not completed - try again", "wait"))


def step(engine, frame: np.ndarray, challenge: LivenessChallenge, now: float):
    """One frame.  Returns (status, face or None)."""
    face = biggest_face(engine.detect(frame))
    if face is None or float(face[2]) < config.LIVENESS_MIN_FACE_PX:
        return challenge.status(), face
    return challenge.update(face, now, True), face       # no recognition here: identity is not the point


def run(camera: int, csv_path: Optional[str], engine=None, capture=None, show=cv2.imshow,
        wait_key=cv2.waitKey, max_frames: Optional[int] = None) -> int:
    if engine is None:
        from face_engine import FaceEngine, ensure_models
        engine = FaceEngine(*ensure_models(print))
    cap = capture or cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Camera could not be opened.")
        return 1
    log = writer = None
    if csv_path:
        log = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log)
        writer.writerow(["time_s", "state", "step", "yaw", "parallax_x", "parallax_y", "narrowing", "verdict"])
    t0 = time.monotonic()
    challenge = LivenessChallenge(0.0)
    finished_at: Optional[float] = None
    frames = 0
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            now = time.monotonic() - t0
            if finished_at is not None and now - finished_at > HOLD_RESULT_SECONDS:
                challenge, finished_at = LivenessChallenge(now), None
            status, face = (challenge.status(), None) if finished_at is not None else step(engine, frame, challenge, now)
            if finished_at is None and status.state in (liveness.PASSED, liveness.FAILED):
                finished_at = now
                print(f"[{now:6.1f}s] {verdict_text(status)[0]}   {status.detail}")
            if writer and challenge.metrics and face is not None:
                writer.writerow([f"{now:.3f}", status.state, challenge.step, *(f"{m:.4f}" for m in challenge.metrics),
                                 status.verdict or ""])

            view = cv2.flip(frame, 1)
            if face is not None:
                x, y, w, h = (int(v) for v in face[:4])
                colour = COLOURS["ok"] if status.state == liveness.PASSED else COLOURS["wait"]
                if status.state == liveness.FAILED:
                    colour = COLOURS["bad"]
                cv2.rectangle(view, (view.shape[1] - x - w, y), (view.shape[1] - x, y + h), colour, 2)
            lines = []
            if status.state in (liveness.PASSED, liveness.FAILED):
                text, key = verdict_text(status)
                lines.append((text, COLOURS[key]))
            else:
                lines.append((status.prompt or "Show your face to the camera", COLOURS["wait"]))
            if challenge.metrics:
                yaw, rx, ry, fs = challenge.metrics
                lines.append((f"yaw {yaw:+.2f}   parallax {np.hypot(rx, ry):.2f}   narrowing {fs:.2f}", COLOURS["info"]))
            lines.append(("R = restart   Q = quit", COLOURS["info"]))
            for n, (text, colour) in enumerate(lines):
                cv2.putText(view, text, (12, 32 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(view, text, (12, 32 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            show("Live check test bench", view)
            key = wait_key(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                challenge, finished_at = LivenessChallenge(now), None
    finally:
        cap.release()
        if log:
            log.close()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--csv", help="write per-frame numbers to this CSV file")
    args = parser.parse_args()
    return run(args.camera, args.csv)


if __name__ == "__main__":
    sys.exit(main())
