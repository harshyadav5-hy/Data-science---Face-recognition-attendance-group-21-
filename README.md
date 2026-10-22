# Face Attendance Pro

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

First launch downloads four small ONNX models (~41 MB) into `models/`. If you are offline, put these
files in `models/` yourself (mirrors are listed in `config.py`): `face_detection_yunet_2023mar.onnx`,
`face_recognition_sface_2021dec.onnx`, `MiniFASNetV2.onnx`, `MiniFASNetV1SE.onnx`.
The two anti-spoof files are checked against a SHA-256 pinned in `config.py`; a changed file is refused.
First launch also asks for consent acknowledgement and an admin password.

## Daily use
1. **Enroll student** - enter Reg No and name, press *Start face capture*, follow the prompts.
   Eight samples are taken automatically (only when sharp, well-lit, correct pose, one face).
   A face already enrolled under another Reg No is refused.
2. **Attendance** - enter subject, set the "on time until" time, choose the **mode**, *Start session*.
   Faces are marked once per session; later arrivals are marked *Late*. Same subject + same day
   resumes the same session.
3. **Reports** - date range + subject: attendance or absentees, export to CSV (opens in Excel).
   The *Method* column shows `face` (live), `face-photo` (photo mode) or `manual`.
4. **Students** - search, re-enroll, or remove (deletes face data, keeps attendance history).

Enrollment, removal, manual attendance and **photo mode** need the admin password. Manual marks require
a reason and are written to the audit log. A daily database backup goes to `data/backups/`.

## Two attendance modes
| | **Live person only** (default) | **Photo attendance** |
|---|---|---|
| Real person in front of the camera | marked after a short head-turn | marked |
| Photo / printout / phone screen | **rejected with a red warning** naming the person | marked |
| Admin password to start | no | yes (+ confirmation) |
| Recorded as | `face` | `face-photo` |

### What happens in Live mode
Every face goes through three independent layers; all must pass before anyone is marked:
1. **Photo / screen detector** - two MiniFASNet networks (Silent-Face-Anti-Spoofing, Apache-2.0) look at the
   face *and its surroundings* (paper edge, phone bezel, glare). A face that scores as a photo gets a red box
   `PHOTO of <Name> (<Reg No>) - NOT MARKED`, a red banner, a beep, and an `spoof-attempt` line in the audit log
   (once per 15 s per person). A photo of someone who is not enrolled shows `PHOTO / SCREEN DETECTED`.
2. **Random head-turn** - the student must look straight, then turn left *and* right (random order every time).
   It is measured with `face_turn()`, which a flat picture cannot change however it is tilted, rolled or waved
   (the old check could be fooled just by rolling a photo sideways - see `tests/test_liveness.py`).
3. **Same face throughout** - if the face in a box suddenly changes (a different person steps in, or a photo is
   swapped over a real face) the box is reset within ~2 frames, the old name disappears, the new person is named
   after ~3 frames, and a challenge started by one person can never be finished by another.

If the photo detector cannot be loaded, live attendance **pauses** (it never silently falls back to accepting
photos). Set `ANTISPOOF_REQUIRED = False` in `config.py` only if you accept a head-turn-only check.

## Testing it on YOUR camera (please do this once)
The logic is covered by automated tests, but photo-detector accuracy depends on your camera and lighting, so
try each of these in Live mode and watch the overlay:
1. A real student: should turn left/right as prompted and be marked.
2. A **printed photo** (steady, then tilted, rolled, waved): must show `PHOTO of ...` and never mark.
3. A **photo on a phone/tablet screen**: same.
4. A **video of the student on a phone**: the weakest case (see Limits); note the result.
5. Two people swapping places quickly: names must switch within a fraction of a second.

Tuning (all in `config.py`; set `SHOW_DEBUG_SCORES = True` to print the live score on the video):
* Real students wrongly called *photo*: lower `ANTISPOOF_REAL_THRESHOLD` (0.60) in steps of 0.05, improve the
  light (face a window, no backlight), and keep 50-100 cm from the camera.
* A photo/screen is accepted: raise `ANTISPOOF_REAL_THRESHOLD` (e.g. 0.70). Compare the `live` score you see for
  real students with the score for a photo on your own camera, and put the threshold between the two.
* Too many *Unknown* for enrolled students: lower `MATCH_THRESHOLD` slightly (not below ~0.36); any wrong person
  marked: raise it.

## Bulk import from existing photos
    python bulk_import.py --csv students.csv --images dataset
CSV header: `reg_no,name,department,section,folder`. Tight 100x100 Haar crops are a weak source;
re-enroll from the camera if many are skipped. (This only enrolls faces; it never marks attendance.)

## Limits you should know
* **No single-camera system is spoof-proof.** Live mode is designed to stop printed photos and phone/tablet
  screens (check this on your camera - see above), and it fails closed. It does **not** guarantee protection against a good video replay on a screen
  that also fools the photo detector, a 3-D mask or a deepfake. For exams or other high-stakes use, keep a
  teacher present or add a depth/IR camera.
* A student wrongly flagged as a photo can be marked by an admin with *Manual attendance* (reason is logged).
* Accuracy drops with strong backlight, masks, very low resolution or faces under ~90 px wide.
* Twins/lookalikes: the margin rule will refuse ambiguous matches rather than guess.
* Biometric data: keep the PC and `data/` folder access-controlled, get consent, and delete data on
  request. Set `SAVE_FACE_IMAGES = False` to keep only numeric templates.

## Layout
    main.py          GUI + camera thread          database.py   SQLite, audit log, backup
    pipeline.py      tracking, face-swap check,   face_engine.py models, detection, embeddings, matching,
                     live checks, enrollment                    photo detector, face_turn
    bulk_import.py   photo import                 config.py     every tunable value
    tests/           python -m unittest discover -s tests -v
