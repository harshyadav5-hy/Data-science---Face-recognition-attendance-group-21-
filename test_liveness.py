import random
import unittest

import numpy as np

import config
from face_engine import face_frame, head_yaw
from liveness import LivenessChallenge
from tests.synth import HEAD, PHOTO, make_face, yaw_sequence


def run(face_at, seed=1, fps=15, seconds=14):
    ch = LivenessChallenge(0.0, random.Random(seed))
    for t in np.arange(0, seconds, 1.0 / fps):
        verdict = ch.update(face_frame(face_at(t)), t)
        if verdict.state != "running":
            return verdict.state, t
    return "running", seconds


class FaceFrameTests(unittest.TestCase):
    def test_frontal_is_zero_and_sign_convention(self):
        self.assertAlmostEqual(face_frame(make_face(HEAD))[0], 0.0, places=3)
        # turning the head: nose moves to camera-right for positive yaw in the simulator
        self.assertGreater(abs(face_frame(make_face(HEAD, yaw=25))[0]), 0.15)

    def test_flat_photo_is_invariant_to_roll_tilt_scale_move(self):
        for roll in (-30, -15, 0, 15, 30):
            for yaw in (0, 20, 40):
                for scale in (40, 70):
                    f = make_face(PHOTO, yaw=yaw, roll=roll, scale=scale, centre=(500 + roll, 300))
                    self.assertAlmostEqual(face_frame(f)[0], 0.0, places=3)
                    self.assertAlmostEqual(head_yaw(f), 0.0, places=3)


class ChallengeTests(unittest.TestCase):
    def test_real_head_passes_for_both_random_orders(self):
        ts, yaws = yaw_sequence()
        for seed in range(12):                       # both challenge orders appear
            first = random.Random(seed).choice((1, -1))
            sign = 1 if first > 0 else -1             # simulator +yaw -> camera-right -> person's LEFT
            def face(t):
                y = float(np.interp(t, ts, yaws)) * sign
                return make_face(HEAD, yaw=y)
            state, t = run(face, seed=seed)
            self.assertEqual(state, "passed", f"seed {seed}")
            self.assertLess(t, 9)

    def test_real_head_with_detector_noise_passes(self):
        rng = np.random.default_rng(0)
        ts, yaws = yaw_sequence()
        first = random.Random(3).choice((1, -1))
        state, _ = run(lambda t: make_face(HEAD, yaw=float(np.interp(t, ts, yaws)) * first, noise=0.8, rng=rng), seed=3)
        self.assertEqual(state, "passed")

    def test_wrong_direction_first_does_not_pass(self):
        ts, yaws = yaw_sequence()
        first = random.Random(3).choice((1, -1))
        # does only the OPPOSITE of what is asked, never the requested side
        state, _ = run(lambda t: make_face(HEAD, yaw=-first * abs(float(np.interp(t, ts, yaws)))), seed=3)
        self.assertNotEqual(state, "passed")

    def test_still_photo_never_passes(self):
        state, _ = run(lambda t: make_face(PHOTO))
        self.assertEqual(state, "timeout")

    def test_photo_rolled_waved_and_moved_never_passes(self):
        rng = np.random.default_rng(1)
        for noise in (0.0, 1.0):
            def face(t):
                roll = 25 * np.sin(t * 3)                 # waving the picture side to side
                return make_face(PHOTO, roll=roll, scale=50 + 8 * np.sin(t * 2),
                                 centre=(640 + 80 * np.sin(t * 1.7), 360 + 40 * np.cos(t)), noise=noise, rng=rng)
            state, _ = run(face)
            self.assertIn(state, ("timeout", "photo"))
            self.assertNotEqual(state, "passed")

    def test_photo_tilted_away_is_flagged_as_photo(self):
        ts, yaws = yaw_sequence()
        state, t = run(lambda t: make_face(PHOTO, yaw=float(np.interp(t, ts, yaws)) * 1.8))   # up to 45 deg
        self.assertEqual(state, "photo")
        self.assertLess(t, 6)

    def test_swapped_pictures_restart_the_challenge(self):
        # left-turned picture then right-turned picture swapped in: a teleporting nose never counts
        def face(t):
            return make_face(HEAD, yaw=30 if int(t) % 2 == 0 else -30)
        state, _ = run(face)
        self.assertNotEqual(state, "passed")

    def test_old_method_was_fooled_new_one_is_not(self):
        def old_yaw(face):                      # the v2.0 measure: nose offset / eye distance
            pts = np.asarray(face[4:14], np.float32).reshape(5, 2)
            eyes = pts[:2][np.argsort(pts[:2, 0])]
            return (pts[2, 0] - eyes[:, 0].mean()) / (eyes[1, 0] - eyes[0, 0])
        rolled = [make_face(PHOTO, roll=r) for r in (-15, 15)]
        old_range = abs(old_yaw(rolled[0]) - old_yaw(rolled[1]))
        self.assertGreaterEqual(old_range, 0.30)          # old LIVENESS_YAW_RANGE: a rolled photo passed
        new_range = abs(head_yaw(rolled[0]) - head_yaw(rolled[1]))
        self.assertLess(new_range, 0.01)                  # new measure: no movement at all


if __name__ == "__main__":
    unittest.main()
