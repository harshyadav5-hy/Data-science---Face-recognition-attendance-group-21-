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

First launch downloads two small ONNX models (~37 MB) into `models/`. If you are offline,
put these files in `models/` yourself: `face_detection_yunet_2023mar.onnx` and
`face_recognition_sface_2021dec.onnx` (OpenCV Zoo; mirrors are listed in `config.py`).
First launch also asks for consent acknowledgement and an admin password.

## Daily use
1. **Enroll student** - enter Reg No and name, press *Start face capture*, follow the prompts.
   Eight samples are taken automatically (only when sharp, well-lit, correct pose, one face).
   A face already enrolled under another Reg No is refused.
2. **Attendance** - enter subject, set the "on time until" time, *Start session*. Faces are marked
   once per session; later arrivals are marked *Late*. Same subject + same day resumes the same session.
3. **Reports** - date range + subject: attendance or absentees, export to CSV (opens in Excel).
4. **Students** - search, re-enroll, or remove (deletes face data, keeps attendance history).

Enrollment, removal and manual attendance need the admin password. Manual marks require a reason
and are written to the audit log. A daily database backup goes to `data/backups/`.

## Bulk import from existing photos
    python bulk_import.py --csv students.csv --images dataset
CSV header: `reg_no,name,department,section,folder`. Tight 100x100 Haar crops are a weak source;
re-enroll from the camera if many are skipped.

## Calibrating (do this once, with your real students)
`MATCH_THRESHOLD` (0.40) and `MATCH_MARGIN` (0.05) in `config.py` are sensible starting values, not
measured ones for your camera and lighting. Enroll ~10 students, then have each one and a few
non-enrolled people stand in front of the camera. Too many *Unknown* for enrolled students: lower the
threshold slightly (not below ~0.36). Any wrong person marked: raise it or enable the liveness option.

## Limits you should know
* **Liveness is basic.** The optional head-turn check defeats a held-up photo (a flat photo's
  nose/eye geometry doesn't change as it moves). It does **not** stop a video replay, a mask or
  a deepfake. For unsupervised, high-stakes use add a trained anti-spoofing model or depth/IR camera,
  or keep a teacher present.
* Accuracy drops with strong backlight, masks, very low resolution or faces under ~90 px wide.
* Twins/lookalikes: the margin rule will refuse ambiguous matches rather than guess.
* Biometric data: keep the PC and `data/` folder access-controlled, get consent, and delete data on
  request. Set `SAVE_FACE_IMAGES = False` to keep only numeric templates.

## Layout
    main.py          GUI + camera thread          database.py   SQLite, audit log, backup
    pipeline.py      tracking, confirmation,      face_engine.py models, detection, embeddings, matching
                     liveness, guided enrollment  config.py     every tunable value
    bulk_import.py   photo import                 tests/        python -m unittest discover -s tests -v
