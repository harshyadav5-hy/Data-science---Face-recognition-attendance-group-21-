"""Enroll many students from existing photos [e.g. your old `dataset/` folder].

CSV columns (header required):  reg_no,name,department,section,folder
`folder` is the sub-folder of --images that holds that student's photos.

    python bulk_import.py --csv students.csv --images dataset

Tight 100x100 face crops (like the old Haar-cascade dataset) are padded and enlarged before
detection, but they are a poor source. If many students are skipped, re-enroll them from
the camera in the app instead - that path checks sharpness, lighting and pose for you.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from database import Database, normalize_reg_no, student_dir
from face_engine import FaceEngine, Gallery, ensure_models

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def embed_photo(engine: FaceEngine, image: np.ndarray):
    """(embedding, aligned crop) of the largest face, or None."""
    for padding in (0.0, 0.5):
        img = image
        if padding:
            pad = int(max(img.shape[:2]) * padding)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        upscale = max(1.0, 480 / max(img.shape[:2]))
        if upscale > 1.0:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        faces = engine.detect(img)
        if len(faces):
            face = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
            aligned = engine.align(img, face)
            return engine.embed_aligned(aligned), aligned
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--min-images", type=int, default=3, help="usable photos needed per student")
    parser.add_argument("--replace", action="store_true", help="overwrite students that already exist")
    args = parser.parse_args()

    engine = FaceEngine(*ensure_models(print))
    db = Database()
    gallery = Gallery.from_records(db.load_embeddings(config.EMBEDDING_MODEL_TAG))
    done = skipped = 0

    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                reg_no = normalize_reg_no(row["reg_no"])
            except (ValueError, KeyError) as exc:
                print(f"SKIP row {row}: {exc}")
                skipped += 1
                continue
            folder = args.images / (row.get("folder") or "").strip()
            files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_TYPES) if folder.is_dir() else []
            vectors, crops = [], []
            for path in files:
                image = cv2.imread(str(path))
                found = embed_photo(engine, image) if image is not None else None
                if found:
                    vectors.append(found[0])
                    crops.append(found[1])
            if len(vectors) < args.min_images:
                print(f"SKIP {reg_no}: only {len(vectors)}/{len(files)} usable photos in {folder}")
                skipped += 1
                continue
            other = gallery.best_other(vectors, exclude=reg_no)
            if other and other[1] >= config.DUPLICATE_THRESHOLD:
                print(f"SKIP {reg_no}: face matches existing {other[0]} (similarity {other[1]:.2f})")
                skipped += 1
                continue
            try:
                db.save_enrollment(reg_no, " ".join(row["name"].split()), row.get("department", "").strip(),
                                   row.get("section", "").strip(), vectors, config.EMBEDDING_MODEL_TAG,
                                   replace=args.replace)
            except ValueError as exc:
                print(f"SKIP {reg_no}: {exc} (use --replace to overwrite)")
                skipped += 1
                continue
            if config.SAVE_FACE_IMAGES:
                target = student_dir(reg_no)
                shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True, exist_ok=True)
                for i, crop in enumerate(crops, 1):
                    cv2.imwrite(str(target / f"{i:02d}.png"), crop)
            gallery = Gallery.from_records(db.load_embeddings(config.EMBEDDING_MODEL_TAG))
            print(f"OK   {reg_no} {row['name']}: {len(vectors)}/{len(files)} photos")
            done += 1

    print(f"\nEnrolled {done}, skipped {skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
