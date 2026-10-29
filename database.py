"""SQLite persistence: students, face embeddings, sessions, attendance, audit-log, settings.

Every call opens its own short-lived connection, so the GUI thread and worker threads
can never trip over each other (SQLite connections must not be shared across threads).
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    reg_no      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL DEFAULT '',
    section     TEXT NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reg_no      TEXT NOT NULL REFERENCES students(reg_no) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_reg ON embeddings(reg_no);
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT NOT NULL,
    date        TEXT NOT NULL,
    late_after  TEXT NOT NULL,
    closed      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    UNIQUE(subject, date)
);
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    reg_no      TEXT NOT NULL REFERENCES students(reg_no),
    marked_at   TEXT NOT NULL,
    status      TEXT NOT NULL,
    method      TEXT NOT NULL,
    confidence  REAL,
    UNIQUE(session_id, reg_no)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_reg_no(raw: str) -> str:
    """Upper-case, trimmed, validated registration number."""
    reg_no = (raw or "").strip().upper()
    if not re.fullmatch(config.REG_NO_PATTERN, reg_no):
        raise ValueError("Reg No must be 3-32 characters: letters, digits, '-', '_' or '/'.")
    return reg_no


def student_dir(reg_no: str) -> Path:
    """Folder for a student's saved face crops (filesystem-safe)."""
    return config.FACES_DIR / re.sub(r"[^A-Za-z0-9_-]", "_", reg_no)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    """UTF-8 with BOM so Excel opens non-ASCII names correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class Database:
    def __init__(self, path: Path = config.DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------ settings & admin password
    def get_setting(self, key: str) -> Optional[str]:
        with self._conn() as con:
            row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as con:
            con.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def has_admin_password(self) -> bool:
        return self.get_setting("admin_pw") is not None

    def set_admin_password(self, password: str) -> None:
        if len(password) < config.ADMIN_MIN_PASSWORD_LEN:
            raise ValueError(f"Password must be at least {config.ADMIN_MIN_PASSWORD_LEN} characters.")
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, config.PBKDF2_ITERATIONS)
        self.set_setting("admin_pw", f"{config.PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}")
        self.audit("admin-password-set")

    def verify_admin_password(self, password: str) -> bool:
        stored = self.get_setting("admin_pw")
        if not stored:
            return False
        iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)

    # ------------------------------------------------------------ audit
    def audit(self, action: str, target: str = "", detail: str = "") -> None:
        with self._conn() as con:
            con.execute("INSERT INTO audit_log(at, action, target, detail) VALUES(?,?,?,?)",
                        (now_str(), action, target, detail))

    # ------------------------------------------------------------ students
    def get_student(self, reg_no: str) -> Optional[sqlite3.Row]:
        with self._conn() as con:
            return con.execute("SELECT * FROM students WHERE reg_no=?", (reg_no,)).fetchone()

    def list_students(self, active_only: bool = True) -> list[sqlite3.Row]:
        with self._conn() as con:
            return con.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM embeddings e WHERE e.reg_no = s.reg_no) AS samples "
                "FROM students s WHERE (? = 0 OR s.active = 1) ORDER BY s.reg_no",
                (1 if active_only else 0,)).fetchall()

    def save_enrollment(self, reg_no: str, name: str, department: str, section: str,
                        vectors: Sequence[np.ndarray], model: str, replace: bool = False) -> None:
        """Create a student (or replace an existing student's face samples)."""
        if not vectors:
            raise ValueError("No face samples to save.")
        stamp = now_str()
        with self._conn() as con:
            row = con.execute("SELECT active FROM students WHERE reg_no=?", (reg_no,)).fetchone()
            if row is None:
                con.execute("INSERT INTO students(reg_no, name, department, section, active, created_at) "
                            "VALUES(?,?,?,?,1,?)", (reg_no, name, department, section, stamp))
            elif row["active"] and not replace:
                raise ValueError(f"Reg No {reg_no} is already enrolled.")
            else:
                con.execute("UPDATE students SET name=?, department=?, section=?, active=1 WHERE reg_no=?",
                            (name, department, section, reg_no))
                con.execute("DELETE FROM embeddings WHERE reg_no=?", (reg_no,))
            con.executemany(
                "INSERT INTO embeddings(reg_no, model, vector, created_at) VALUES(?,?,?,?)",
                [(reg_no, model, np.asarray(v, "<f4").tobytes(), stamp) for v in vectors])
            con.execute("INSERT INTO audit_log(at, action, target, detail) VALUES(?,?,?,?)",
                        (stamp, "enroll" if row is None else "re-enroll", reg_no, f"{len(vectors)} samples"))

    def deactivate_student(self, reg_no: str) -> None:
        """Delete biometric data and deactivate; attendance history is kept."""
        with self._conn() as con:
            con.execute("DELETE FROM embeddings WHERE reg_no=?", (reg_no,))
            con.execute("UPDATE students SET active=0 WHERE reg_no=?", (reg_no,))
            con.execute("INSERT INTO audit_log(at, action, target, detail) VALUES(?,?,?,?)",
                        (now_str(), "remove-student", reg_no, "biometrics deleted"))

    def load_embeddings(self, model: str) -> list[tuple[str, str, np.ndarray]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT e.reg_no, s.name, e.vector FROM embeddings e "
                "JOIN students s ON s.reg_no = e.reg_no "
                "WHERE s.active = 1 AND e.model = ? ORDER BY e.reg_no, e.id", (model,)).fetchall()
        return [(r["reg_no"], r["name"], np.frombuffer(r["vector"], "<f4").copy()) for r in rows]

    # ------------------------------------------------------------ sessions & attendance
    def get_or_create_session(self, subject: str, day: str, late_after: str) -> int:
        """Re-opening the same subject on the same day continues the same session."""
        with self._conn() as con:
            row = con.execute("SELECT id FROM sessions WHERE subject=? AND date=?", (subject, day)).fetchone()
            if row:
                con.execute("UPDATE sessions SET late_after=?, closed=0 WHERE id=?", (late_after, row["id"]))
                return int(row["id"])
            cur = con.execute("INSERT INTO sessions(subject, date, late_after, created_at) VALUES(?,?,?,?)",
                              (subject, day, late_after, now_str()))
            return int(cur.lastrowid)

    def close_session(self, session_id: int) -> None:
        with self._conn() as con:
            con.execute("UPDATE sessions SET closed=1 WHERE id=?", (session_id,))

    def subjects(self) -> list[str]:
        with self._conn() as con:
            return [r["subject"] for r in con.execute(
                "SELECT DISTINCT subject FROM sessions ORDER BY subject").fetchall()]

    def mark_attendance(self, session_id: int, reg_no: str, status: str, method: str,
                        confidence: Optional[float]) -> bool:
        """True if newly recorded, False if the student was already marked in this session."""
        with self._conn() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO attendance(session_id, reg_no, marked_at, status, method, confidence) "
                "VALUES(?,?,?,?,?,?)", (session_id, reg_no, now_str(), status, method, confidence))
            return cur.rowcount == 1

    def marked_set(self, session_id: int) -> set[str]:
        with self._conn() as con:
            return {r["reg_no"] for r in con.execute(
                "SELECT reg_no FROM attendance WHERE session_id=?", (session_id,)).fetchall()}

    def session_rows(self, session_id: int) -> list[sqlite3.Row]:
        with self._conn() as con:
            return con.execute(
                "SELECT a.reg_no, s.name, a.marked_at, a.status, a.method FROM attendance a "
                "JOIN students s ON s.reg_no = a.reg_no WHERE a.session_id=? ORDER BY a.marked_at",
                (session_id,)).fetchall()

    def session_counts(self, session_id: int) -> tuple[int, int]:
        """(marked, active students enrolled)."""
        with self._conn() as con:
            marked = con.execute("SELECT COUNT(*) FROM attendance WHERE session_id=?", (session_id,)).fetchone()[0]
            enrolled = con.execute("SELECT COUNT(*) FROM students WHERE active=1").fetchone()[0]
        return int(marked), int(enrolled)

    def report(self, date_from: str, date_to: str, subject: Optional[str] = None) -> list[sqlite3.Row]:
        sql = ("SELECT se.date, se.subject, a.reg_no, s.name, s.department, s.section, "
               "substr(a.marked_at, 12) AS time, a.status, a.method "
               "FROM attendance a JOIN sessions se ON se.id = a.session_id "
               "JOIN students s ON s.reg_no = a.reg_no WHERE se.date BETWEEN ? AND ?")
        args: list = [date_from, date_to]
        if subject:
            sql += " AND se.subject = ?"
            args.append(subject)
        with self._conn() as con:
            return con.execute(sql + " ORDER BY se.date, se.subject, a.reg_no", args).fetchall()

    def absentees(self, date_from: str, date_to: str, subject: Optional[str] = None) -> list[sqlite3.Row]:
        """Active students with no record in a session. Students enrolled after the
        session's day are not counted as absent for it."""
        sql = ("SELECT se.date, se.subject, st.reg_no, st.name, st.department, st.section "
               "FROM sessions se JOIN students st ON st.active = 1 AND st.created_at <= se.date || ' 23:59:59' "
               "LEFT JOIN attendance a ON a.session_id = se.id AND a.reg_no = st.reg_no "
               "WHERE a.id IS NULL AND se.date BETWEEN ? AND ?")
        args: list = [date_from, date_to]
        if subject:
            sql += " AND se.subject = ?"
            args.append(subject)
        with self._conn() as con:
            return con.execute(sql + " ORDER BY se.date, se.subject, st.reg_no", args).fetchall()

    # ------------------------------------------------------------ backup
    def backup(self, dest_dir: Path, keep: int) -> Path:
        """One consistent snapshot per day; older snapshots beyond `keep` are pruned."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"attendance-{datetime.now():%Y%m%d}.db"
        if target.exists():
            return target
        src = sqlite3.connect(self.path)
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        for old in sorted(dest_dir.glob("attendance-*.db"))[:-keep]:
            old.unlink(missing_ok=True)
        return target
