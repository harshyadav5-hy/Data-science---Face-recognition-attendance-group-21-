"""Photo attendance from image FILES (e.g. a class photo).  This is the explicit PHOTO option.

A photo proves that a face was photographed, not that the person is present now - the app therefore
records these marks with method "photo" and asks for the admin password.  Pure logic (no Qt, no camera),
so it can be unit-tested; the GUI only picks the files and shows the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

import config
from face_engine import Gallery


@dataclass
class FileReport:
    path: str
    faces: int = 0               # faces detected
    matched: int = 0             # faces recognised as an enrolled student
    unknown: int = 0             # faces not recognised (or claimed twice)
    too_small: int = 0           # faces below PHOTO_MIN_FACE_PX, ignored
    error: str = ""


@dataclass
class PhotoReport:
    files: list[FileReport] = field(default_factory=list)
    matches: dict[str, float] = field(default_factory=dict)    # reg_no -> best score over all photos

    @property
    def unknown_faces(self) -> int:
        return sum(f.unknown for f in self.files)

    @property
    def too_small_faces(self) -> int:
        return sum(f.too_small for f in self.files)


def recognise_image(engine, gallery: Gallery, image: np.ndarray) -> tuple[list[tuple[str, float]], int, int]:
    """-> ([(reg_no, score)], unknown_faces, too_small_faces) for ONE image.

    Stricter than live recognition (PHOTO_MATCH_THRESHOLD): faces in group photos are small.
    One student can appear only once in a photo: if two faces claim the same student, the weaker
    claim is treated as unknown."""
    faces = engine.detect(image, max_side=config.PHOTO_MAX_SIDE)
    claims: list[tuple[float, str]] = []
    unknown = small = 0
    for face in faces:
        if float(face[2]) < config.PHOTO_MIN_FACE_PX:
            small += 1
            continue
        result = gallery.match(engine.embed_aligned(engine.align(image, face)))
        if result.accepted and result.score >= config.PHOTO_MATCH_THRESHOLD:
            claims.append((result.score, result.reg_no))
        else:
            unknown += 1
    best: dict[str, float] = {}
    for score, reg_no in sorted(claims, reverse=True):
        if reg_no in best:
            unknown += 1
        else:
            best[reg_no] = score
    return list(best.items()), unknown, small


def recognise_files(engine, gallery: Gallery, paths: Iterable[str],
                    imread: Callable[[str], Optional[np.ndarray]] = cv2.imread) -> PhotoReport:
    report = PhotoReport()
    for path in paths:
        entry = FileReport(str(path))
        report.files.append(entry)
        image = imread(str(path))
        if image is None:
            entry.error = "could not be read as an image"
            continue
        matches, entry.unknown, entry.too_small = recognise_image(engine, gallery, image)
        entry.matched = len(matches)
        entry.faces = entry.matched + entry.unknown + entry.too_small
        for reg_no, score in matches:
            report.matches[reg_no] = max(score, report.matches.get(reg_no, -1.0))
    return report
