import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from database import Database
from face_engine import Gallery
from pipeline import AttendanceProcessor, adaptive_template_ok
from tests.synth import HEAD, PHOTO, make_face, yaw_sequence

DIM = 128


def unit(seed):
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def noisy(base, seed, amount=0.15):
    v = base + np.random.default_rng(seed).normal(size=DIM).astype(np.float32) * amount / np.sqrt(DIM) * 3
    return v / np.linalg.norm(v)


class FakeEngine:
    """Returns whatever 'who' is currently set to; crops have texture so quality gating passes."""
    def __init__(self):
        self.who = None
        self.counter = 0
        self.rng = np.random.default_rng(0)

    def align(self, frame, face):
        return self.rng.integers(60, 190, (112, 112, 3), dtype=np.uint8)

    def embed_aligned(self, aligned):
        self.counter += 1
        return noisy(self.who, self.counter)


A, B = unit(1), unit(2)
GALLERY = Gallery.from_records(
    [("A001", "Asha", noisy(A, 100 + i)) for i in range(4)] +
    [("B002", "Bilal", noisy(B, 200 + i)) for i in range(4)])
FRAME = np.zeros((720, 1280, 3), np.uint8)


def simulate(proc, engine, who, face_at, seconds, t0=0.0, fps=15):
    engine.who = who
    events, alerts, overlays = [], [], []
    for t in np.arange(0, seconds, 1.0 / fps):
        ov, ev, _ = proc.process(FRAME, np.array([face_at(t)]), now=t0 + t)
        events += ev
        alerts += proc.pop_alerts()
        overlays.append(ov[0][4])
    return events, alerts, overlays


def follow_prompts(proc, engine, who, seconds, t0=0.0, fps=15, speed=100.0, obey=True, noise=0.6, seed=0):
    """A real person: reads the on-screen prompt and turns the head towards it (speed deg/s)."""
    rng = np.random.default_rng(seed)
    engine.who = who
    yaw, events, alerts, overlays = 0.0, [], [], []
    for t in np.arange(0, seconds, 1.0 / fps):
        face = make_face(HEAD, yaw=yaw, noise=noise, rng=rng)
        ov, ev, _ = proc.process(FRAME, np.array([face]), now=t0 + t)
        events += ev
        alerts += proc.pop_alerts()
        text = ov[0][4]
        overlays.append(text)
        target = 25.0 if ("LEFT" in text and obey) else -25.0 if ("RIGHT" in text and obey) else 0.0
        yaw += float(np.clip(target - yaw, -speed / fps, speed / fps))
    return events, alerts, overlays


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.proc = AttendanceProcessor(self.engine, GALLERY, set())

    def test_live_person_is_marked_once_after_challenge(self):
        for seed in range(6):                                   # random challenge order each time
            proc = AttendanceProcessor(self.engine, GALLERY, set())
            events, alerts, overlays = follow_prompts(proc, self.engine, A, 14, seed=seed)
            self.assertEqual([e[0] for e in events], ["A001"], f"seed {seed}")   # exactly once
            self.assertEqual(alerts, [])
            self.assertIsNotNone(events[0][2])                  # fused embedding for adaptation
            first_mark = next(i for i, o in enumerate(overlays) if o.startswith("\u2713"))
            self.assertTrue(any("turn your head" in o for o in overlays[:first_mark]))   # challenge came first

    def test_person_who_ignores_the_prompts_is_not_marked(self):
        events, alerts, _ = follow_prompts(self.proc, self.engine, A, 20, obey=False)
        self.assertEqual(events, [])
        self.assertTrue(alerts)                                 # reported as "possible photo"
        self.assertEqual(alerts[0].kind, "timeout")

    def test_photo_of_enrolled_student_is_never_marked_and_named(self):
        def photo(t):                                   # waved around, rolled, moved
            return make_face(PHOTO, roll=20 * np.sin(t * 3), centre=(640 + 60 * np.sin(t), 360))
        events, alerts, overlays = simulate(self.proc, self.engine, A, photo, 30)
        self.assertEqual(events, [])
        self.assertTrue(alerts)
        self.assertEqual(alerts[0].reg_no, "A001")
        self.assertIn("PHOTO", alerts[0].text)
        self.assertIn("Asha", alerts[0].text)
        self.assertTrue(any("PHOTO of Asha" in o or "possible PHOTO of Asha" in o for o in overlays))

    def test_tilted_photo_gives_strong_photo_alert(self):
        ts, yaws = yaw_sequence()
        events, alerts, _ = simulate(self.proc, self.engine, A,
                                     lambda t: make_face(PHOTO, yaw=float(np.interp(t, ts, yaws)) * 1.8), 12)
        self.assertEqual(events, [])
        self.assertEqual(alerts[0].kind, "photo")

    def test_unknown_person_gets_no_challenge_no_mark(self):
        events, alerts, overlays = simulate(self.proc, self.engine, unit(99), lambda t: make_face(HEAD), 5)
        self.assertEqual(events, [])
        self.assertEqual(alerts, [])
        self.assertIn("Unknown", overlays[-1])

    def test_person_swap_is_noticed_quickly(self):
        follow_prompts(self.proc, self.engine, A, 14)                 # A marked
        self.assertIn("A001", self.proc.marked)
        _, _, overlays = simulate(self.proc, self.engine, B, lambda t: make_face(HEAD), 3, t0=14.0)
        self.assertFalse(any("Asha" in o and "\u2713" in o for o in overlays[-8:]))   # label no longer A
        self.assertTrue(any("Bilal" in o for o in overlays))                          # now B is being handled

    def test_face_that_stops_matching_loses_its_label(self):
        follow_prompts(self.proc, self.engine, A, 14)
        _, _, overlays = simulate(self.proc, self.engine, unit(77), lambda t: make_face(HEAD), 4, t0=14.0)
        self.assertNotIn("\u2713 Asha (A001)", overlays[-1])
        self.assertTrue(overlays[-1].startswith(("Unknown", "Identifying")))


class RecognitionTests(unittest.TestCase):
    def test_topk_needs_two_good_samples(self):
        g = Gallery.from_records([("X", "x", A), ("X", "x", unit(5)), ("Y", "y", unit(6))])
        q = A                                      # only ONE X sample equals the query
        topk1 = g._per_identity(q, topk=1)[0]
        topk2 = g._per_identity(q, topk=2)[0]
        self.assertAlmostEqual(float(topk1), 1.0, places=3)
        self.assertLess(float(topk2), 0.7)         # one lucky sample is no longer enough

    def test_adaptive_template_rules(self):
        new_look = noisy(A, 7, amount=0.9)         # fairly different view of A
        sim_own = GALLERY.identity_score("A001", new_look, 1)
        self.assertLess(sim_own, config.ADAPT_MAX_SIMILARITY)
        self.assertTrue(adaptive_template_ok(GALLERY, "A001", new_look, 0.50))
        self.assertFalse(adaptive_template_ok(GALLERY, "A001", new_look, 0.40))          # not confident
        self.assertFalse(adaptive_template_ok(GALLERY, "A001", noisy(A, 100), 0.9))       # nothing new
        self.assertFalse(adaptive_template_ok(GALLERY, "A001", None, 0.9))


class DatabaseTests(unittest.TestCase):
    def test_add_templates_cap_and_migration(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.db"
            con = sqlite3.connect(path)                    # a v2.0 database without `source`
            con.executescript("""
                CREATE TABLE students (reg_no TEXT PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL DEFAULT '',
                    section TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
                CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, reg_no TEXT NOT NULL, model TEXT NOT NULL,
                    vector BLOB NOT NULL, created_at TEXT NOT NULL);
                INSERT INTO students VALUES ('A001','Asha','','',1,'2024-01-01 00:00:00');
                INSERT INTO embeddings(reg_no, model, vector, created_at) VALUES ('A001','m', x'00', '2024-01-01 00:00:00');
            """)
            con.commit(); con.close()
            db = Database(path)                            # must migrate in place
            with db._conn() as c:
                self.assertEqual(c.execute("SELECT source FROM embeddings").fetchone()[0], "enroll")
            for i in range(config.MAX_AUTO_TEMPLATES + 5):
                db.add_templates("A001", [unit(i)], "m", "auto")
            with db._conn() as c:
                n_auto = c.execute("SELECT COUNT(*) FROM embeddings WHERE source='auto'").fetchone()[0]
                n_enroll = c.execute("SELECT COUNT(*) FROM embeddings WHERE source='enroll'").fetchone()[0]
            self.assertEqual(n_auto, config.MAX_AUTO_TEMPLATES)
            self.assertEqual(n_enroll, 1)                  # original enrollment is never evicted
            with self.assertRaises(ValueError):
                db.add_templates("NOPE", [unit(1)], "m", "update")


class EnrollmentTests(unittest.TestCase):
    """Enrolling from a photo must fail: the left/right prompts need a real 3-D head turn."""
    def run_enroll(self, make_pts, seconds=40, fps=15):
        from pipeline import EnrollmentSession
        engine = FakeEngine(); engine.who = A
        sess = EnrollmentSession(engine)
        rng = np.random.default_rng(5)
        yaw, result = 0.0, None
        for t in np.arange(0, seconds, 1.0 / fps):
            face = make_face(make_pts, yaw=yaw if make_pts is HEAD else 0.0,
                             roll=0.0 if make_pts is HEAD else 15 * np.sin(t * 2.5), noise=0.4, rng=rng)
            _, hud, result = sess.process(FRAME, np.array([face]), now=t)
            if result is not None:
                break
            prompt = hud[1] if len(hud) > 1 else ""
            target = 25.0 if "LEFT" in prompt else -25.0 if "RIGHT" in prompt else 0.0
            yaw += float(np.clip(target - yaw, -100 / fps, 100 / fps))
        return result, len(sess.vectors)

    def test_real_person_completes_enrollment(self):
        result, n = self.run_enroll(HEAD)
        self.assertIsNotNone(result)
        self.assertEqual(n, len(config.ENROLL_PROMPTS))

    def test_rolled_photo_cannot_complete_enrollment(self):
        result, n = self.run_enroll(PHOTO)
        self.assertIsNone(result)
        self.assertLess(n, len(config.ENROLL_PROMPTS))   # stuck at the first left/right prompt


if __name__ == "__main__":
    unittest.main()
