# Face Attendance Pro 2.1

Desktop attendance system for classes: enroll students by **Reg No + name**, mark attendance by face,
report/export to CSV. No TensorFlow, no training step.

## Why this replaces the CNN
The old trainer learned to *classify* 44 people from ~5 images each, which is far too little data
(it scored 1.89% - chance level - on the test split). Adding a student also meant retraining.
This version uses pretrained face **embeddings** (OpenCV YuNet detector + SFace recognizer):
enrolling a person stores a few 128-number templates, and recognizing is a nearest-neighbour
search. A new student takes about 30 seconds and needs no retraining.

## Install (Windows / Linux / macOS, Python 3.9+)
    pip install -r requirements.txt
    python main.py

First launch downloads two small ONNX models (~37 MB) into `models/`. If you are offline,
put these files in `models/` yourself: `face_detection_yunet_2023mar.onnx` and
`face_recognition_sface_2021dec.onnx` (OpenCV Zoo; mirrors are listed in `config.py`).
First launch also asks for consent acknowledgement and an admin password.

## Daily use
1. **Enroll student** - enter Reg No and name, press *Start face capture*, follow the prompts.
   Eight samples are taken automatically (only when sharp, well-lit, correct pose, one face).
   A face already enrolled under another Reg No is refused.
2. **Attendance** - enter subject, set the "on time until" time, *Start session*. Each student is
   recognised, then must pass a **live head-turn challenge** (turn to the side shown on screen, then the
   other side, then look straight). Only then is attendance marked, once per session; later arrivals are
   marked *Late*. The liveness check is always on - there is no switch to turn it off.
   **Photos, phone screens and printouts are never marked.** The face gets a red box
   *"PHOTO of <name> shown - NOT MARKED"*, a red banner appears, the attempt is written to the audit log
   and a cropped snapshot is saved in `data/spoof_alerts/`.
3. **Reports** - date range + subject: attendance or absentees, export to CSV (opens in Excel).
4. **Students** - search, **Update look** (add samples when someone changed glasses/beard/hair),
   re-enroll, or remove (deletes face data, keeps attendance history).

Enrollment, removal and manual attendance need the admin password. Manual marks require a reason
and are written to the audit log. A daily database backup goes to `data/backups/`.

## Bulk import from existing photos
    python bulk_import.py --csv students.csv --images dataset
CSV header: `reg_no,name,department,section,folder`. Tight 100x100 Haar crops are a weak source;
re-enroll from the camera if many are skipped.

## How photo / spoof detection works
The nose is measured inside the face's *own* coordinate frame (built from eyes and mouth). If you move,
roll, scale or tilt a **flat** picture, that position never changes; only a real 3-D head changes it when
it turns. So:
* **"PHOTO of <name> shown"** (strong evidence) - the face got narrower as if tilted away, but the nose
  did not move inside it: a flat picture is being tilted.
* **"No live movement - possible PHOTO of <name>"** - the challenge was not completed in 12 s (a still
  photo, or a real person who did not follow the prompts; they can simply try again).
* The challenge order is random, the neutral pose must be steady, the median of 3 frames is used, a
  "teleporting" nose (picture swap) restarts it, and the identity is re-confirmed after the turn.
* Enrollment uses the same measure, so a photo cannot be used to enroll either.

*Why v2.0 let photos through:* the check was optional (off by default), and its yaw measure could be
fooled - rolling a flat photo about 15 degrees in the hand produced a fake "head turn". Both are fixed
(see `tests/test_liveness.py::test_old_method_was_fooled_new_one_is_not`).

## Recognition accuracy and changing faces
* Frames that are blurry, too dark/bright or turned too far are skipped instead of guessed.
* An identity needs several agreeing frames **and** a match of the last frames' averaged embedding.
  Score = mean of the best 2 samples (`MATCH_TOPK`), so one lucky sample is not enough.
* Already-marked faces are re-checked twice a second: if someone else sits down, or the face stops
  matching, the label is withdrawn within about a second and the new person is identified.
* **Face changed (glasses, beard, hair, lighting)?**
  1. After a confident, live-verified match that shows a genuinely *new* view, the system learns it
     immediately (`ADAPT_*` in `config.py`; max 8 automatic samples per student, oldest dropped first).
     It never learns from rejected, photo or ambiguous frames.
  2. If a student is no longer recognised at all, the label says *"face changed? Ask admin: Update
     look"*. In **Students > Update look** (admin password, 4 prompts, ~20 s) new samples are added
     without deleting the original enrollment. End the session first; starting the same subject again
     the same day continues it.

## Calibrating (do this once, with your real students)
`MATCH_THRESHOLD` (0.40) and `MATCH_MARGIN` (0.05) in `config.py` are sensible starting values, not
measured ones for your camera and lighting. Enroll ~10 students, then have each one and a few
non-enrolled people stand in front of the camera. Also test it with a printed photo and a photo on a phone: set `LIVENESS_DEBUG = True` to see the nose shift (`dU`) and face-width ratio (`f`) on the label. Too many *Unknown* for enrolled students: lower the
threshold slightly (not below ~0.36). Any wrong person marked: raise it or enable the liveness option.

## Limits you should know
* **What liveness stops / does not stop.** It stops printed photos, photos on phone/tablet screens, and
  photos held, waved, rolled or tilted. It does **not** stop a *video* of the person turning their head
  played on a screen, a 3-D mask, or a deepfake: a single webcam cannot reliably tell those apart without
  a trained anti-spoofing model or a depth/IR camera. For unsupervised high-stakes use add one, or keep
  a teacher present. If a real student keeps failing, use manual attendance (admin, logged).
* Thresholds were verified on simulated geometry, not on your camera. Do the calibration above once;
  `LIVENESS_TURN` (default 0.16) is the main knob (lower = easier turn, higher = stricter).
* Accuracy drops with strong backlight, masks, very low resolution or faces under ~90 px wide.
* Twins/lookalikes: the margin rule will refuse ambiguous matches rather than guess.
* Biometric data: keep the PC and `data/` folder access-controlled, get consent, and delete data on
  request. Set `SAVE_FACE_IMAGES = False` to keep only numeric templates, and `SAVE_SPOOF_SNAPSHOTS = False` to stop saving photo-attempt snapshots.

## Layout
    main.py          GUI + camera thread          database.py   SQLite, audit log, backup
    pipeline.py      tracking, confirmation,      face_engine.py models, detection, embeddings, matching
                     liveness, guided enrollment  config.py     every tunable value
    liveness.py      anti-photo challenge         bulk_import.py photo import
                     tests/  python -m unittest discover -s tests -t . -v
