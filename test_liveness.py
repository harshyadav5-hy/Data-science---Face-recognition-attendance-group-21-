"""Live-person / anti-photo / face swap tests. No camera, no GUI, no model downloads needed.

Faces are synthetic: a 3-D head (eyes, nose tip that sticks out, mouth corners) projected to YuNet-style
landmarks, and a FLAT PHOTO of that head that can be tilted, rolled and rotated about any axis.

    python -m unittest discover -s tests -v
"""
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from face_engine import Gallery, _scaled_crop, face_turn, head_yaw  # noqa: E402
from pipeline import AttendanceProcessor  # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)
# 3-D landmark model in cm relative to the eye centre: (x right, y down, z towards camera)
EYE_R, EYE_L = (-3.15, 0.0, 0.0), (3.15, 0.0, 0.0)
NOSE = (0.1, 4.5, 3.0)                    # nose tip sticks out ~3 cm
MOUTH_R, MOUTH_L = (-2.5, 8.5, -0.5), (2.5, 8.5, -0.5)


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def make_face(cx=600.0, cy=360.0, width=160.0, yaw=0.0, pitch=0.0, roll=0.0, flat=False):
    """YuNet-style row (15,). flat=True -> the same face printed on a flat sheet, then moved by the
    same rotation (depth of the nose is frozen at 0), which is what a held-up photo does."""
    s = width / 16.0
    pts = np.array([EYE_R, EYE_L, NOSE, MOUTH_R, MOUTH_L], float)
    if flat:
        pts[:, 2] = 0.0
    xy = (pts @ rot(yaw, pitch, roll).T)[:, :2] * s
    xy += np.array([cx, cy - 0.1 * width])
    return np.array([cx - width / 2, cy - 0.6 * width, width, 1.2 * width, *xy.reshape(-1), 0.99], np.float32)


class FakeEngine:
    """align() tags the crop with the face index (0 left half of frame, 1 right half); embed returns vecs[idx]."""
    def __init__(self):
        self.vecs = [None, None]

    def align(self, frame, face):
        return np.full((112, 112, 3), 0 if face[0] < 640 else 1, np.uint8)

    def embed_aligned(self, aligned):
        return self.vecs[int(aligned[0, 0, 0])]


class FakeSpoof:
    """p = probability of 'live'; either a number or a function of the face row."""
    def __init__(self, p=0.95):
        self.p = p

    def real_probability(self, frame, face):
        return float(self.p(face) if callable(self.p) else self.p)


E = np.eye(128, dtype=np.float32)
ALI, SARA, STRANGER = E[0], E[1], E[2]


def blend(base, other_axis, sim):
    """Unit vector whose cosine similarity with `base` is exactly `sim`."""
    return (sim * base + math.sqrt(1 - sim * sim) * E[other_axis]).astype(np.float32)


def gallery():
    return Gallery.from_records([("R1", "Ali", ALI), ("R2", "Sara", SARA)])


class Rig:
    def __init__(self, live=True, p=0.95, required=True, with_spoof=True, seed=1, already=(), gal=None):
        self.engine = FakeEngine()
        self.spoof = FakeSpoof(p)
        self.proc = AttendanceProcessor(
            self.engine, gal or gallery(), set(already), live,
            antispoof=self.spoof if (live and with_spoof) else None,
            antispoof_required=required, rng=random.Random(seed))
        self.t = 100.0
        self.events, self.last_overlays, self.last_hud = [], [], []
        self.texts = []

    def step(self, *faces, dt=0.05):
        arr = np.array(faces, np.float32) if faces else np.empty((0, 15), np.float32)
        self.t += dt
        overlays, events, hud = self.proc.process(FRAME, arr, now=self.t)
        self.events += events
        self.last_overlays, self.last_hud = overlays, hud
        self.texts += [o[4] for o in overlays]
        return overlays

    def frames(self, n, who, **pose):
        self.engine.vecs[0] = who
        for _ in range(n):
            self.step(make_face(**pose))

    def do_challenge(self, who, amount=26.0):
        """Behave like a cooperative real person: look straight, then turn as the prompts ask."""
        self.frames(8, who)
        track = self.proc.tracks[0]
        for _ in range(60):
            if track.done or track.steps is None:
                break
            direction = track.steps[track.step_idx]
            self.frames(5, who, yaw=amount if direction == "left" else -amount)
        self.frames(6, who)                      # return to the camera


class TestFaceTurn(unittest.TestCase):
    def test_straight_is_near_zero(self):
        self.assertLess(abs(face_turn(make_face())), 0.06)

    def test_real_turn_has_sign_and_size(self):
        left, right = face_turn(make_face(yaw=25)), face_turn(make_face(yaw=-25))
        self.assertGreater(left, config.LIVENESS_TURN_MIN)          # person's LEFT = positive
        self.assertLess(right, -config.LIVENESS_TURN_MIN)

    def test_matches_head_yaw_scale_for_real_heads(self):
        for yaw in (-30, -15, 15, 30):
            face = make_face(yaw=yaw)
            self.assertAlmostEqual(face_turn(face), head_yaw(face), delta=0.08)

    def test_flat_photo_cannot_fake_a_turn_by_any_motion(self):
        worst = 0.0
        for yaw in (-50, -30, 0, 30, 50):
            for pitch in (-30, 0, 30):
                for roll in (-35, 0, 35):
                    worst = max(worst, abs(face_turn(make_face(yaw=yaw, pitch=pitch, roll=roll, flat=True))))
        self.assertLess(worst, config.LIVENESS_TURN_MIN * 0.6, f"flat photo reached {worst:.3f}")

    def test_old_metric_was_fooled_by_rolling_a_photo(self):
        """Documents the bug that let photos through: head_yaw read a rolled flat photo as a head turn."""
        photo = make_face(roll=25, flat=True)
        self.assertGreater(abs(head_yaw(photo)), config.LIVENESS_TURN_MIN)   # old metric: fooled
        self.assertLess(abs(face_turn(photo)), 0.06)                          # new metric: not fooled

    def test_degenerate_landmarks_never_count_as_a_turn(self):
        face = make_face()
        face[4:14] = 0
        self.assertEqual(face_turn(face), 0.0)


class TestLiveMode(unittest.TestCase):
    def test_real_person_is_marked_once_after_the_random_challenge(self):
        rig = Rig()
        rig.frames(10, ALI)
        self.assertEqual(rig.events, [], "must not mark before the head-turn")
        rig.do_challenge(ALI)
        self.assertEqual([r for r, _ in rig.events], ["R1"])
        rig.frames(10, ALI)
        self.assertEqual(len(rig.events), 1, "marked only once")
        self.assertTrue(any("\u2713" in t for t in rig.texts))

    def test_challenge_order_is_random_but_always_covers_both_sides(self):
        orders = set()
        for seed in range(12):
            rig = Rig(seed=seed)
            rig.frames(10, ALI)
            orders.add(tuple(rig.proc.tracks[0].steps))
        self.assertEqual(orders, {("left", "right"), ("right", "left")})

    def test_photo_is_blocked_and_names_the_person(self):
        rig = Rig(p=0.04)
        rig.frames(14, ALI)
        self.assertEqual(rig.events, [])
        self.assertTrue(any("PHOTO of Ali (R1) - NOT MARKED" in t for t in rig.texts), set(rig.texts))
        self.assertTrue(rig.last_hud[0].startswith("WARNING: photo of Ali (R1)"))
        alerts = rig.proc.pop_alerts()
        self.assertEqual([(a.reg_no, a.label) for a in alerts], [("R1", "Ali (R1)")])
        self.assertEqual(rig.proc.pop_alerts(), [])

    def test_photo_warning_is_rate_limited_then_repeats(self):
        rig = Rig(p=0.04)
        rig.frames(60, ALI)                                          # ~3 s of the same photo
        self.assertEqual(len(rig.proc.pop_alerts()), 1)
        rig.t += config.ANTISPOOF_ALERT_COOLDOWN + 1
        rig.frames(8, ALI)
        self.assertEqual(len(rig.proc.pop_alerts()), 1)

    def test_photo_of_unknown_person_still_warns(self):
        rig = Rig(p=0.04)
        rig.frames(40, STRANGER)
        self.assertTrue(any("PHOTO / SCREEN DETECTED" in t for t in rig.texts))
        self.assertEqual(rig.events, [])
        alerts = rig.proc.pop_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertIsNone(alerts[0].reg_no)

    def test_photo_that_fools_the_detector_still_fails_the_head_turn(self):
        """Worst case: the passive detector says 'live' for a photo. Waving/tilting/rolling it must not pass."""
        rig = Rig(p=0.99)
        rig.engine.vecs[0] = ALI
        for k in range(400):                                          # 20 s of vigorous photo motion
            rig.step(make_face(yaw=45 * math.sin(k / 6), pitch=25 * math.sin(k / 9),
                               roll=35 * math.sin(k / 4), flat=True))
        self.assertEqual(rig.events, [])

    def test_wrong_direction_does_not_advance(self):
        rig = Rig()
        rig.frames(10, ALI)
        track = rig.proc.tracks[0]
        wrong = -26 if track.steps[0] == "left" else 26
        rig.frames(30, ALI, yaw=wrong)
        self.assertEqual(track.step_idx, 0)
        self.assertEqual(rig.events, [])

    def test_must_look_straight_before_turning(self):
        """A photo already posed at an angle cannot start the challenge."""
        rig = Rig()
        rig.frames(10, ALI, yaw=26)
        rig.frames(10, ALI, yaw=-26)
        track = rig.proc.tracks[0]
        self.assertFalse(track.armed)
        self.assertEqual(rig.events, [])

    def test_live_then_photo_midway_resets_progress(self):
        rig = Rig()
        rig.frames(10, ALI)
        track = rig.proc.tracks[0]
        first = track.steps[0]
        rig.frames(5, ALI, yaw=26 if first == "left" else -26)
        self.assertGreaterEqual(track.step_idx, 1)
        rig.spoof.p = 0.03
        rig.frames(12, ALI)
        self.assertIsNone(track.steps)
        self.assertEqual(track.step_idx, 0)

    def test_challenge_times_out_and_restarts(self):
        rig = Rig()
        rig.frames(10, ALI)
        track = rig.proc.tracks[0]
        started = track.challenge_t0
        rig.t += config.LIVENESS_TIMEOUT + 1
        rig.frames(2, ALI)                                   # timeout -> challenge wiped
        self.assertTrue(any("Time is up" in t for t in rig.texts))
        self.assertEqual(rig.events, [])
        rig.frames(8, ALI)                                   # ... and a fresh random one begins
        self.assertIsNotNone(track.steps)
        self.assertGreater(track.challenge_t0, started + config.LIVENESS_TIMEOUT)
        self.assertEqual(sorted(track.steps), ["left", "right"])

    def test_borderline_score_does_not_flip_a_live_face_to_photo(self):
        rig = Rig(p=0.95)
        rig.frames(10, ALI)
        rig.spoof.p = 0.55                    # between THRESHOLD-HYSTERESIS and THRESHOLD
        rig.frames(12, ALI)
        self.assertEqual(rig.proc.tracks[0].verdict, "real")

    def test_missing_detector_pauses_attendance_when_required(self):
        rig = Rig(with_spoof=False, required=True)
        rig.frames(40, ALI)
        rig.frames(10, ALI, yaw=26)
        rig.frames(10, ALI, yaw=-26)
        self.assertEqual(rig.events, [])
        self.assertIn("Photo check unavailable", rig.texts)
        self.assertIn("PAUSED", rig.last_hud[0])

    def test_challenge_only_fallback_when_detector_not_required(self):
        rig = Rig(with_spoof=False, required=False)
        rig.do_challenge(ALI)
        self.assertEqual([r for r, _ in rig.events], ["R1"])

    def test_already_marked_student_is_not_asked_again(self):
        rig = Rig(already={"R1"})
        rig.frames(10, ALI)
        self.assertTrue(any("already marked" in t for t in rig.texts))
        self.assertEqual(rig.events, [])


class TestFaceChangeIsFast(unittest.TestCase):
    def test_new_identity_shows_within_a_few_frames(self):
        rig = Rig()
        rig.frames(8, ALI)
        self.assertTrue(any("Ali" in o[4] for o in rig.last_overlays))
        rig.engine.vecs[0] = SARA
        seen = None
        for n in range(1, 12):
            rig.step(make_face())
            if any("Sara" in o[4] for o in rig.last_overlays):
                seen = n
                break
        self.assertIsNotNone(seen)
        self.assertLessEqual(seen, config.SWAP_FRAMES + config.CONFIRM_FRAMES, f"took {seen} frames")

    def test_challenge_started_for_one_person_cannot_be_finished_by_another(self):
        rig = Rig()
        rig.frames(10, ALI)                                  # Ali recognised, challenge running
        rig.do_challenge(SARA)                               # Sara steps in and does everything right
        self.assertNotIn("R1", [r for r, _ in rig.events])
        self.assertEqual([r for r, _ in rig.events], ["R2"])

    def test_unrecognised_stranger_cannot_finish_someone_elses_challenge(self):
        rig = Rig()
        rig.frames(10, ALI)
        rig.do_challenge(STRANGER)
        self.assertEqual(rig.events, [])

    def test_photo_swapped_in_after_a_real_face_is_caught(self):
        rig = Rig(p=lambda f: 0.95)
        rig.frames(10, ALI)
        rig.spoof.p = 0.03                                   # attacker holds a photo of Sara over the box
        rig.engine.vecs[0] = SARA
        rig.frames(14, SARA)
        self.assertEqual(rig.events, [])
        self.assertTrue(any("PHOTO of Sara (R2)" in t for t in rig.texts))

    def test_marked_box_is_rechecked_when_someone_else_steps_in(self):
        rig = Rig()
        rig.do_challenge(ALI)
        self.assertEqual([r for r, _ in rig.events], ["R1"])
        rig.engine.vecs[0] = SARA
        rig.frames(8, SARA)
        self.assertFalse(any("Ali" in o[4] for o in rig.last_overlays), "stale name still shown")
        rig.do_challenge(SARA)
        self.assertEqual([r for r, _ in rig.events], ["R1", "R2"])

    def test_one_blurry_frame_does_not_reset_a_real_person(self):
        rig = Rig()
        rig.frames(10, ALI)
        track = rig.proc.tracks[0]
        steps = list(track.steps)
        rig.engine.vecs[0] = STRANGER                        # a single garbage frame
        rig.step(make_face())
        rig.engine.vecs[0] = ALI
        rig.frames(3, ALI)
        self.assertEqual(track.steps, steps)

    def test_stale_name_disappears_the_moment_a_stranger_steps_in(self):
        rig = Rig()
        rig.frames(10, ALI)
        self.assertTrue(any("Ali" in o[4] for o in rig.last_overlays))
        rig.engine.vecs[0] = STRANGER
        for _ in range(config.SWAP_FRAMES + 1):
            rig.step(make_face())
        self.assertFalse(any("Ali" in o[4] for o in rig.last_overlays), rig.last_overlays)
        self.assertIsNone(rig.proc.tracks[0].challenge_for)

    def test_lookalike_below_match_threshold_cannot_finish_challenge(self):
        """Similar enough to NOT trip swap detection, different enough to NOT be recognised as Ali.
        The final identity re-check is the only thing standing in the way."""
        lookalike = blend(ALI, 7, (config.SWAP_SIMILARITY + config.MATCH_THRESHOLD) / 2)
        self.assertGreater(float(lookalike @ ALI), config.SWAP_SIMILARITY)
        self.assertLess(float(lookalike @ ALI), config.MATCH_THRESHOLD)
        rig = Rig()
        rig.frames(10, ALI)
        rig.do_challenge(lookalike)
        self.assertEqual(rig.events, [])

    def test_marked_box_is_rechecked_for_a_similar_looking_enrolled_student(self):
        """Two enrolled students who resemble each other (sim 0.3 > swap threshold): the marked box must
        still be released when the second one steps in, by identity votes alone."""
        sam = blend(ALI, 3, 0.30)
        gal = Gallery.from_records([("R1", "Ali", ALI), ("R3", "Sam", sam)])
        rig = Rig(gal=gal)
        rig.do_challenge(ALI)
        self.assertEqual([r for r, _ in rig.events], ["R1"])
        rig.engine.vecs[0] = sam
        rig.frames(8, sam)
        self.assertFalse(any("Ali" in o[4] for o in rig.last_overlays), "stale name still shown")
        rig.do_challenge(sam)
        self.assertEqual([r for r, _ in rig.events], ["R1", "R3"])

    def test_lookalike_student_stepping_in_mid_challenge_gets_their_own_challenge(self):
        sam = blend(ALI, 3, 0.30)                            # enrolled, resembles Ali, no swap alarm
        gal = Gallery.from_records([("R1", "Ali", ALI), ("R3", "Sam", sam)])
        rig = Rig(gal=gal)
        rig.frames(10, ALI)                                  # Ali's challenge is running
        rig.do_challenge(sam)                                # Sam takes over and does it properly
        self.assertEqual([r for r, _ in rig.events], ["R3"])


class TestTwoPeople(unittest.TestCase):
    def test_real_person_marked_while_a_photo_beside_them_is_blocked(self):
        rig = Rig(p=lambda f: 0.95 if f[0] < 640 else 0.03)
        rig.engine.vecs = [ALI, SARA]
        left = lambda **k: make_face(cx=300, **k)            # noqa: E731
        right = lambda **k: make_face(cx=900, **k)           # noqa: E731
        for _ in range(8):
            rig.step(left(), right())
        track = next(t for t in rig.proc.tracks if t.cx < 640)
        for _ in range(60):
            if track.done:
                break
            yaw = 26 if track.steps[track.step_idx] == "left" else -26
            for _ in range(5):
                rig.step(left(yaw=yaw), right())
        for _ in range(8):
            rig.step(left(), right())
        self.assertEqual([r for r, _ in rig.events], ["R1"])
        self.assertTrue(any("PHOTO of Sara" in t for t in rig.texts))


class TestPhotoMode(unittest.TestCase):
    def test_photo_mode_marks_without_liveness_checks(self):
        rig = Rig(live=False)
        rig.frames(8, ALI)
        self.assertEqual([r for r, _ in rig.events], ["R1"])
        self.assertTrue(rig.last_hud[0].startswith("PHOTO mode"))

    def test_live_mode_hud_says_so(self):
        rig = Rig()
        rig.frames(2, ALI)
        self.assertTrue(rig.last_hud[0].startswith("LIVE mode"))


class TestCrop(unittest.TestCase):
    def test_crop_is_80x80_everywhere(self):
        for box in [(600, 300, 150, 180), (0, 0, 120, 140), (1160, 580, 120, 140), (-20, 100, 120, 140),
                    (100, 100, 1500, 1500), (400, 100, 9, 9)]:
            for scale in (2.7, 4.0):
                crop = _scaled_crop(FRAME, box, scale)
                self.assertTrue(crop is None or crop.shape == (80, 80, 3), (box, scale))

    def test_crop_never_leaves_the_frame_or_squashes(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[:, :, 0] = np.arange(1280) % 256                  # horizontal ramp: detects shifts/squash
        crop = _scaled_crop(img, (1100, 300, 150, 180), 4.0)
        self.assertEqual(crop.shape, (80, 80, 3))


@unittest.skipUnless((config.MODELS_DIR / config.ANTISPOOF_MODELS[0]["file"]).exists()
                     and (config.MODELS_DIR / config.ANTISPOOF_MODELS[1]["file"]).exists(),
                     "anti-spoof models not downloaded yet")
class TestRealModels(unittest.TestCase):
    def test_models_load_hashes_match_and_return_probabilities(self):
        from face_engine import AntiSpoof, ensure_antispoof_models
        models = ensure_antispoof_models()                    # also verifies the pinned SHA-256
        spoof = AntiSpoof(models)
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        p = spoof.real_probability(frame, make_face())
        self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()
