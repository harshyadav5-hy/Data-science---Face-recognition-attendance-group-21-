"""Face Attendance Pro - PyQt5 desktop application.

Run: python main.py
"""
from __future__ import annotations

import logging
import shutil
import sys
import threading
import time
from datetime import date, datetime
from logging.handlers import RotatingFileHandler

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import config
from database import Database, normalize_reg_no, now_str, student_dir, write_csv
from face_engine import AntiSpoof, FaceEngine, Gallery, ensure_antispoof_models, ensure_models
from pipeline import AttendanceProcessor, EnrollmentSession

log = logging.getLogger("app")


# =============================================================================== camera worker
def open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraWorker(QThread):
    """Owns the camera and the DNNs. All heavy work happens here, never in the GUI thread."""

    frame_ready = pyqtSignal(QtGui.QImage, list, list)   # image, overlays, hud lines
    identity_confirmed = pyqtSignal(str, float)           # reg_no, score
    spoof_detected = pyqtSignal(str, str, float)          # reg_no ("" if unknown), "Name (REG)", live-score
    antispoof_problem = pyqtSignal(str)                   # photo detector could not be loaded
    enroll_finished = pyqtSignal(object)                  # EnrollResult or None (cancelled)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, camera_index: int):
        super().__init__()
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._mode = "preview"            # preview | attendance | enroll
        self._gallery = Gallery.empty()
        self._session_args = (set(), True)     # (already marked, live_only)
        self._antispoof = None
        self._attendance = None
        self._enroll = None
        self._cancel_enroll = False

    # ---- control (called from the GUI thread)
    def set_gallery(self, gallery: Gallery) -> None:
        self._gallery = gallery

    def begin_session(self, marked: set, live_only: bool = True) -> None:
        with self._lock:
            self._session_args = (set(marked), bool(live_only))
            self._attendance = None
            self._mode = "attendance"

    def end_session(self) -> None:
        with self._lock:
            self._attendance = None
            self._mode = "preview"

    def begin_enrollment(self) -> None:
        with self._lock:
            self._enroll = None
            self._cancel_enroll = False
            self._mode = "enroll"

    def cancel_enrollment(self) -> None:
        with self._lock:
            self._cancel_enroll = True

    def stop(self) -> None:
        self._stop.set()

    # ---- worker thread
    def run(self) -> None:
        try:
            detector_path, recognizer_path = ensure_models(self.status.emit)
            engine = FaceEngine(detector_path, recognizer_path)
        except Exception as exc:
            log.exception("Model initialisation failed")
            self.failed.emit(str(exc))
            return

        try:
            self._antispoof = AntiSpoof(ensure_antispoof_models(self.status.emit))
        except Exception as exc:                     # live attendance stays paused (ANTISPOOF_REQUIRED)
            log.exception("Photo detector unavailable")
            self._antispoof = None
            self.antispoof_problem.emit(str(exc))

        cap = None
        failures = 0
        fps, last = 0.0, time.perf_counter()
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = open_camera(self._camera_index)
                    if cap is None:
                        self.status.emit("Camera not available - check the connection/permissions. Retrying...")
                        self._stop.wait(2.0)
                        continue
                    failures = 0
                    self.status.emit("Camera running")
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        cap.release()
                        cap = None
                        self.status.emit("Camera disconnected - reconnecting...")
                    self.msleep(30)
                    continue
                failures = 0

                try:
                    overlays, hud = self._handle_frame(engine, frame)
                except Exception:
                    log.exception("Frame processing failed")
                    overlays, hud = [], ["Processing error - see data/app.log"]

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                self._emit_frame(frame, overlays, hud + [f"{fps:.0f} FPS"])
        finally:
            if cap is not None:
                cap.release()

    def _handle_frame(self, engine: FaceEngine, frame):
        faces = engine.detect(frame)
        with self._lock:
            mode = self._mode
            if mode == "attendance" and self._attendance is None:
                marked, live_only = self._session_args
                self._attendance = AttendanceProcessor(
                    engine, self._gallery, marked, live_only,
                    antispoof=self._antispoof if live_only else None,
                    antispoof_required=config.ANTISPOOF_REQUIRED and live_only)
            if mode == "enroll" and self._enroll is None:
                self._enroll = EnrollmentSession(engine)
            attendance, enroll = self._attendance, self._enroll
            cancel, self._cancel_enroll = self._cancel_enroll, False

        if mode == "attendance" and attendance is not None:
            attendance.gallery = self._gallery
            overlays, events, hud = attendance.process(frame, faces)
            for reg_no, score in events:
                self.identity_confirmed.emit(reg_no, score)
            for alert in attendance.pop_alerts():
                self.spoof_detected.emit(alert.reg_no or "", alert.label, alert.score)
            return overlays, hud
        if mode == "enroll" and enroll is not None:
            if cancel:
                self._finish_enroll(None)
                return [], ["Enrollment cancelled"]
            overlays, hud, result = enroll.process(frame, faces)
            if result is not None:
                self._finish_enroll(result)
            return overlays, hud
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), "", "#38bdf8") for f in faces], []

    def _finish_enroll(self, result) -> None:
        with self._lock:
            self._enroll = None
            self._mode = "preview"
        self.enroll_finished.emit(result)

    def _emit_frame(self, frame, overlays, hud) -> None:
        mirrored = cv2.flip(frame, 1)                   # selfie view; labels are drawn after the flip
        height, width = mirrored.shape[:2]
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888).copy()
        flipped = [(width - x - w, y, w, h, text, color) for (x, y, w, h, text, color) in overlays]
        self.frame_ready.emit(image, flipped, hud)


# =============================================================================== widgets
class CameraView(QtWidgets.QWidget):
    """Paints the frame plus overlays with QPainter (so non-ASCII names render correctly)."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._image = None
        self._overlays: list = []
        self._hud: list = []
        self._placeholder = "Camera off"

    def set_frame(self, image, overlays, hud) -> None:
        self._image, self._overlays, self._hud = image, overlays, hud
        self.update()

    def clear(self, text: str = "Camera off") -> None:
        self._image, self._overlays, self._hud, self._placeholder = None, [], [], text
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#111827"))
        if self._image is None:
            p.setPen(QtGui.QColor("#9ca3af"))
            p.setFont(QtGui.QFont("Arial", 16))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        p.drawImage(QtCore.QRectF(ox, oy, dw, dh), self._image)

        p.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        fm = p.fontMetrics()
        for x, y, w, h, text, color in self._overlays:
            rect = QtCore.QRectF(ox + x * scale, oy + y * scale, w * scale, h * scale)
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidth(3)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)
            if text:
                tw, th = fm.horizontalAdvance(text) + 12, fm.height() + 6
                label = QtCore.QRectF(rect.left(), max(rect.top() - th, oy), tw, th)
                p.fillRect(label, QtGui.QColor(color))
                p.setPen(QtGui.QColor("#000000"))
                p.drawText(label, Qt.AlignCenter, text)

        p.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        fm = p.fontMetrics()
        y = oy + 8
        for line in self._hud:
            tw, th = fm.horizontalAdvance(line) + 14, fm.height() + 6
            box = QtCore.QRectF(ox + 8, y, tw, th)
            p.fillRect(box, QtGui.QColor(0, 0, 0, 150))
            p.setPen(QtGui.QColor("#ffffff"))
            p.drawText(box, Qt.AlignCenter, line)
            y += th + 3


class AdminGate:
    """Password prompt for sensitive actions; stays unlocked for a few minutes."""

    def __init__(self, db: Database):
        self.db = db
        self._until = 0.0

    def ensure(self, parent) -> bool:
        if time.monotonic() < self._until:
            return True
        password, ok = QtWidgets.QInputDialog.getText(
            parent, "Admin password", "Enter the admin password:", QtWidgets.QLineEdit.Password)
        if not ok:
            return False
        if self.db.verify_admin_password(password):
            self._until = time.monotonic() + config.ADMIN_UNLOCK_SECONDS
            return True
        self.db.audit("admin-login-failed")
        QtWidgets.QMessageBox.warning(parent, "Access denied", "Wrong password.")
        return False


def fill_table(table: QtWidgets.QTableWidget, header, rows) -> None:
    table.setSortingEnabled(False)
    table.clear()
    table.setColumnCount(len(header))
    table.setHorizontalHeaderLabels(list(header))
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, item)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    table.setSortingEnabled(True)


def make_table() -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget()
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    return table


# =============================================================================== main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.gate = AdminGate(db)
        self.worker = None
        self.gallery = Gallery.empty()
        self.session_id = None
        self.session_deadline = "00:00:00"
        self._enroll_ctx = None
        self.session_method = "face"
        self._spoof_count = 0
        self._report_header: list = []
        self._report_rows: list = []

        self.setWindowTitle(f"{config.APP_NAME} {config.APP_VERSION}")
        self.resize(1320, 780)
        self._build_ui()
        self.reload_gallery()
        self.refresh_students()
        self.refresh_subjects()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        self.btn_camera = QtWidgets.QPushButton("Start camera")
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.lbl_status = QtWidgets.QLabel("Camera off")
        top.addWidget(self.btn_camera)
        top.addWidget(self.lbl_status, 1)
        root.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._tab_attendance(), "Attendance")
        self.tabs.addTab(self._tab_enroll(), "Enroll student")
        self.tabs.addTab(self._tab_students(), "Students")
        self.tabs.addTab(self._tab_reports(), "Reports")

    def _tab_attendance(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.view_att = CameraView()
        left = QtWidgets.QVBoxLayout()
        self.lbl_alert = QtWidgets.QLabel("")
        self.lbl_alert.setWordWrap(True)
        self.lbl_alert.setAlignment(Qt.AlignCenter)
        self.lbl_alert.setStyleSheet("background:#b91c1c; color:white; font-size:17px; font-weight:bold;"
                                     "padding:12px; border-radius:6px;")
        self.lbl_alert.hide()
        self._alert_timer = QtCore.QTimer(self)
        self._alert_timer.setSingleShot(True)
        self._alert_timer.timeout.connect(self.lbl_alert.hide)
        left.addWidget(self.lbl_alert)
        left.addWidget(self.view_att, 1)
        layout.addLayout(left, 3)

        side = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("Class session")
        form = QtWidgets.QFormLayout(box)
        self.cmb_subject = QtWidgets.QComboBox()
        self.cmb_subject.setEditable(True)
        self.cmb_subject.lineEdit().setPlaceholderText("Subject / course, e.g. PHY402")
        self.time_late = QtWidgets.QTimeEdit(QtCore.QTime.currentTime().addSecs(600))
        self.time_late.setDisplayFormat("HH:mm")
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItem("Live person only (recommended)", "live")
        self.cmb_mode.addItem("Photo attendance (photos accepted)", "photo")
        self.lbl_mode_hint = QtWidgets.QLabel()
        self.lbl_mode_hint.setWordWrap(True)
        self.cmb_mode.currentIndexChanged.connect(self._update_mode_hint)
        form.addRow("Subject", self.cmb_subject)
        form.addRow("On time until", self.time_late)
        form.addRow("Attendance mode", self.cmb_mode)
        form.addRow(self.lbl_mode_hint)
        self._update_mode_hint()
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start session")
        self.btn_end = QtWidgets.QPushButton("End session")
        self.btn_end.setEnabled(False)
        self.btn_start.clicked.connect(self.start_session)
        self.btn_end.clicked.connect(self.end_session)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_end)
        form.addRow(row)
        side.addWidget(box)

        self.lbl_count = QtWidgets.QLabel("No session running")
        self.lbl_count.setStyleSheet("font-size: 15px; font-weight: bold;")
        side.addWidget(self.lbl_count)
        self.tbl_live = make_table()
        fill_table(self.tbl_live, ["Reg No", "Name", "Time", "Status", "Method"], [])
        side.addWidget(self.tbl_live, 1)

        manual = QtWidgets.QGroupBox("Manual attendance (admin)")
        mrow = QtWidgets.QHBoxLayout(manual)
        self.ed_manual = QtWidgets.QLineEdit()
        self.ed_manual.setPlaceholderText("Reg No")
        btn = QtWidgets.QPushButton("Mark present")
        btn.clicked.connect(self.manual_mark)
        mrow.addWidget(self.ed_manual, 1)
        mrow.addWidget(btn)
        side.addWidget(manual)

        wrap = QtWidgets.QWidget()
        wrap.setLayout(side)
        wrap.setFixedWidth(400)
        layout.addWidget(wrap)
        return page

    def _tab_enroll(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.view_enroll = CameraView()
        layout.addWidget(self.view_enroll, 3)

        side = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("Student details")
        form = QtWidgets.QFormLayout(box)
        self.ed_reg = QtWidgets.QLineEdit()
        self.ed_reg.setPlaceholderText("e.g. MS23045")
        self.ed_name = QtWidgets.QLineEdit()
        self.ed_dept = QtWidgets.QLineEdit()
        self.ed_section = QtWidgets.QLineEdit()
        form.addRow("Reg No *", self.ed_reg)
        form.addRow("Full name *", self.ed_name)
        form.addRow("Department", self.ed_dept)
        form.addRow("Section / batch", self.ed_section)
        side.addWidget(box)

        row = QtWidgets.QHBoxLayout()
        self.btn_enroll_start = QtWidgets.QPushButton("Start face capture")
        self.btn_enroll_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_enroll_cancel.setEnabled(False)
        self.btn_enroll_start.clicked.connect(self.start_enrollment)
        self.btn_enroll_cancel.clicked.connect(self.cancel_enrollment)
        row.addWidget(self.btn_enroll_start)
        row.addWidget(self.btn_enroll_cancel)
        side.addLayout(row)

        tips = QtWidgets.QLabel(
            "Tips for a reliable enrollment:\n"
            "\u2022 Student alone in view, 50-100 cm from the camera\n"
            "\u2022 Light in front of the face, not behind\n"
            "\u2022 Follow the on-screen prompts; each sample is taken automatically\n"
            "\u2022 Enroll with the same kind of camera used for attendance\n"
            "\u2022 Glasses: capture once with and once without if the student switches")
        tips.setWordWrap(True)
        side.addWidget(tips)
        side.addStretch(1)

        wrap = QtWidgets.QWidget()
        wrap.setLayout(side)
        wrap.setFixedWidth(400)
        layout.addWidget(wrap)
        return page

    def _tab_students(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.ed_search = QtWidgets.QLineEdit()
        self.ed_search.setPlaceholderText("Search by Reg No / name / department ...")
        self.ed_search.textChanged.connect(self._filter_students)
        btn_re = QtWidgets.QPushButton("Re-enroll selected")
        btn_rm = QtWidgets.QPushButton("Remove selected")
        btn_re.clicked.connect(self.reenroll_selected)
        btn_rm.clicked.connect(self.remove_selected)
        row.addWidget(self.ed_search, 1)
        row.addWidget(btn_re)
        row.addWidget(btn_rm)
        layout.addLayout(row)
        self.tbl_students = make_table()
        layout.addWidget(self.tbl_students, 1)
        self.lbl_students = QtWidgets.QLabel("")
        layout.addWidget(self.lbl_students)
        return page

    def _tab_reports(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.date_from = QtWidgets.QDateEdit(QtCore.QDate.currentDate().addDays(-30))
        self.date_to = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        for edit in (self.date_from, self.date_to):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
        self.cmb_report_subject = QtWidgets.QComboBox()
        btn_load = QtWidgets.QPushButton("Show attendance")
        btn_abs = QtWidgets.QPushButton("Show absentees")
        btn_exp = QtWidgets.QPushButton("Export table to CSV")
        btn_load.clicked.connect(lambda: self.load_report(False))
        btn_abs.clicked.connect(lambda: self.load_report(True))
        btn_exp.clicked.connect(self.export_report)
        for widget in (QtWidgets.QLabel("From"), self.date_from, QtWidgets.QLabel("To"), self.date_to,
                       self.cmb_report_subject, btn_load, btn_abs, btn_exp):
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.tbl_report = make_table()
        layout.addWidget(self.tbl_report, 1)
        return page

    # ------------------------------------------------------------------ camera control
    def toggle_camera(self) -> None:
        if self.worker is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        if self.worker is not None:
            return
        worker = CameraWorker(config.CAMERA_INDEX)
        worker.frame_ready.connect(self._on_frame)
        worker.identity_confirmed.connect(self._on_identity)
        worker.spoof_detected.connect(self._on_spoof)
        worker.antispoof_problem.connect(self._on_antispoof_problem)
        worker.enroll_finished.connect(self._on_enroll_finished)
        worker.status.connect(self.lbl_status.setText)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        worker.set_gallery(self.gallery)
        self.worker = worker
        self.lbl_status.setText("Starting camera ...")
        self.btn_camera.setText("Stop camera")
        worker.start()

    def stop_camera(self) -> None:
        worker = self.worker
        if worker is None:
            return
        if self.session_id is not None:
            self.end_session(ask_export=False)
        if self._enroll_ctx is not None:
            self._enroll_ctx = None
            self._reset_enroll_buttons()
        self.worker = None
        worker.stop()
        worker.wait(5000)
        self._camera_stopped_ui()

    def _camera_stopped_ui(self) -> None:
        self.btn_camera.setText("Start camera")
        self.lbl_status.setText("Camera off")
        self.view_att.clear()
        self.view_enroll.clear()

    def _on_worker_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Camera / model error", message)

    def _on_worker_finished(self) -> None:
        if self.worker is not None and not self.worker.isRunning():     # ended by itself (error)
            self.worker = None
            self._camera_stopped_ui()

    def _on_frame(self, image, overlays, hud) -> None:
        for view in (self.view_att, self.view_enroll):
            if view.isVisible():
                view.set_frame(image, overlays, hud)

    # ------------------------------------------------------------------ gallery / data refresh
    def reload_gallery(self) -> None:
        self.gallery = Gallery.from_records(self.db.load_embeddings(config.EMBEDDING_MODEL_TAG))
        if self.worker is not None:
            self.worker.set_gallery(self.gallery)

    def refresh_subjects(self) -> None:
        subjects = self.db.subjects()
        current = self.cmb_subject.currentText()
        self.cmb_subject.clear()
        self.cmb_subject.addItems(subjects)
        self.cmb_subject.setEditText(current)
        self.cmb_report_subject.clear()
        self.cmb_report_subject.addItem("All subjects")
        self.cmb_report_subject.addItems(subjects)

    # ------------------------------------------------------------------ attendance session
    def start_session(self) -> None:
        subject = self.cmb_subject.currentText().strip()
        if not subject:
            QtWidgets.QMessageBox.warning(self, "Subject needed", "Enter the subject / course name first.")
            return
        if len(self.gallery) == 0:
            QtWidgets.QMessageBox.warning(self, "No students", "Enroll at least one student first.")
            return
        if self._enroll_ctx is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Finish or cancel the enrollment first.")
            return
        live_only = self.cmb_mode.currentData() == "live"
        if not live_only:
            if not self.gate.ensure(self):
                return
            answer = QtWidgets.QMessageBox.question(
                self, "Photo attendance",
                "In PHOTO mode a photo, printout or phone screen shown to the camera WILL mark that "
                "person present. Attendance is recorded as 'face-photo' so it can be told apart in reports.\n\n"
                "Use LIVE mode for real classes. Start photo mode anyway?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.start_camera()
        self.session_deadline = self.time_late.time().toString("HH:mm") + ":00"
        self.session_id = self.db.get_or_create_session(subject, date.today().isoformat(), self.session_deadline)
        marked = self.db.marked_set(self.session_id)
        self.session_method = "face" if live_only else "face-photo"
        self._spoof_count = 0
        if not live_only:
            self.db.audit("photo-mode-session", subject, f"session {self.session_id} started in photo mode")
        self.worker.begin_session(marked, live_only)
        rows = self.db.session_rows(self.session_id)
        fill_table(self.tbl_live, ["Reg No", "Name", "Time", "Status", "Method"],
                   [[r["reg_no"], r["name"], r["marked_at"][11:], r["status"], r["method"]] for r in rows])
        self._update_count()
        self.btn_start.setEnabled(False)
        self.btn_end.setEnabled(True)
        self.cmb_subject.setEnabled(False)
        self.cmb_mode.setEnabled(False)
        self.lbl_alert.hide()
        self.refresh_subjects()
        self.cmb_subject.setEditText(subject)

    def end_session(self, ask_export: bool = True) -> None:
        if self.session_id is None:
            return
        session_id, self.session_id = self.session_id, None
        if self.worker is not None:
            self.worker.end_session()
        self.db.close_session(session_id)
        marked, enrolled = self.db.session_counts(session_id)
        self.btn_start.setEnabled(True)
        self.btn_end.setEnabled(False)
        self.cmb_subject.setEnabled(True)
        self.cmb_mode.setEnabled(True)
        self.lbl_alert.hide()
        self.lbl_count.setText("No session running")
        if ask_export:
            answer = QtWidgets.QMessageBox.question(
                self, "Session ended",
                f"Marked: {marked}\nNot marked: {max(enrolled - marked, 0)}\n\nExport this session to CSV now?")
            if answer == QtWidgets.QMessageBox.Yes:
                self._export_session(session_id)

    def _export_session(self, session_id: int) -> None:
        rows = [[r["reg_no"], r["name"], r["marked_at"], r["status"], r["method"]]
                for r in self.db.session_rows(session_id)]
        default = config.EXPORT_DIR / f"session_{session_id}_{datetime.now():%Y%m%d_%H%M}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export session", str(default), "CSV (*.csv)")
        if path:
            write_csv(path, ["Reg No", "Name", "Marked at", "Status", "Method"], rows)
            self.statusBar().showMessage(f"Saved {path}", 6000)

    def _status_for_now(self) -> str:
        return "Present" if datetime.now().strftime("%H:%M:%S") <= self.session_deadline else "Late"

    def _on_identity(self, reg_no: str, score: float) -> None:
        if self.session_id is None:
            return
        status = self._status_for_now()
        if self.db.mark_attendance(self.session_id, reg_no, status, self.session_method, score):
            self._append_live_row(reg_no, status, self.session_method)
            QtWidgets.QApplication.beep()
            name = self.gallery.names.get(reg_no, "")
            self.statusBar().showMessage(f"Marked {name} ({reg_no}) - {status}", 6000)

    def _on_spoof(self, reg_no: str, label: str, score: float) -> None:
        """A photo/screen was held up: show a red warning naming the person, beep, and keep an audit trail."""
        if self.session_id is None:
            return
        self._spoof_count += 1
        if label:
            text = f"WARNING - PHOTO DETECTED\nA photo of {label} is being shown to the camera.\nAttendance NOT marked."
        else:
            text = "WARNING - PHOTO / SCREEN DETECTED\nA picture is being shown to the camera.\nAttendance NOT marked."
        self.lbl_alert.setText(text)
        self.lbl_alert.show()
        self._alert_timer.start(8000)
        QtWidgets.QApplication.beep()
        self.statusBar().showMessage(text.replace("\n", " "), 8000)
        self.db.audit("spoof-attempt", reg_no,
                      f"photo/screen shown in session {self.session_id} (live score {score:.2f})")
        self._update_count()

    def _on_antispoof_problem(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(
            self, "Photo detector unavailable",
            f"{message}\n\nLive attendance is PAUSED until the photo detector is available "
            f"(config.ANTISPOOF_REQUIRED). Enrollment and photo mode still work.")

    def _update_mode_hint(self) -> None:
        if self.cmb_mode.currentData() == "live":
            self.lbl_mode_hint.setText(
                "Only a physically present person counts. Photos, printouts and phone screens are rejected "
                "with a warning, and each student is asked to turn their head.")
            self.lbl_mode_hint.setStyleSheet("color:#15803d;")
        else:
            self.lbl_mode_hint.setText(
                "Anything recognised is marked, INCLUDING a photo. Needs the admin password. "
                "Recorded as 'face-photo'.")
            self.lbl_mode_hint.setStyleSheet("color:#b45309;")

    def _append_live_row(self, reg_no: str, status: str, method: str) -> None:
        name = self.gallery.names.get(reg_no) or (self.db.get_student(reg_no) or {"name": ""})["name"]
        table = self.tbl_live
        table.setSortingEnabled(False)
        row = table.rowCount()
        table.insertRow(row)
        for col, value in enumerate([reg_no, name, datetime.now().strftime("%H:%M:%S"), status, method]):
            item = QtWidgets.QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, col, item)
        table.scrollToBottom()
        table.setSortingEnabled(True)
        self._update_count()

    def _update_count(self) -> None:
        if self.session_id is not None:
            marked, enrolled = self.db.session_counts(self.session_id)
            text = f"Marked {marked} of {enrolled} enrolled"
            if self._spoof_count:
                text += f"  |  photo attempts blocked: {self._spoof_count}"
            self.lbl_count.setText(text)

    def manual_mark(self) -> None:
        if self.session_id is None:
            QtWidgets.QMessageBox.warning(self, "No session", "Start a session first.")
            return
        try:
            reg_no = normalize_reg_no(self.ed_manual.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Reg No", str(exc))
            return
        student = self.db.get_student(reg_no)
        if not student or not student["active"]:
            QtWidgets.QMessageBox.warning(self, "Not found", f"{reg_no} is not an enrolled student.")
            return
        if not self.gate.ensure(self):
            return
        reason, ok = QtWidgets.QInputDialog.getText(
            self, "Manual attendance", f"Reason for marking {student['name']} manually (saved in audit log):")
        if not ok or not reason.strip():
            return
        status = self._status_for_now()
        if self.db.mark_attendance(self.session_id, reg_no, status, "manual", None):
            self.db.audit("manual-attendance", reg_no, reason.strip())
            self._append_live_row(reg_no, status, "manual")
            self.ed_manual.clear()
        else:
            QtWidgets.QMessageBox.information(self, "Already marked", f"{reg_no} is already marked.")

    # ------------------------------------------------------------------ enrollment
    def start_enrollment(self) -> None:
        try:
            reg_no = normalize_reg_no(self.ed_reg.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Reg No", str(exc))
            return
        name = " ".join(self.ed_name.text().split())
        if len(name) < 2:
            QtWidgets.QMessageBox.warning(self, "Name needed", "Enter the student's full name.")
            return
        if self.session_id is not None:
            QtWidgets.QMessageBox.warning(self, "Session running", "End the attendance session first.")
            return
        existing = self.db.get_student(reg_no)
        replace = False
        if existing and existing["active"]:
            answer = QtWidgets.QMessageBox.question(
                self, "Already enrolled",
                f"{reg_no} is enrolled as {existing['name']}.\nReplace this student's face samples?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
            replace = True
        if not self.gate.ensure(self):
            return
        self.start_camera()
        self._enroll_ctx = {"reg_no": reg_no, "name": name, "dept": self.ed_dept.text().strip(),
                            "section": self.ed_section.text().strip(), "replace": replace}
        self.worker.begin_enrollment()
        self.btn_enroll_start.setEnabled(False)
        self.btn_enroll_cancel.setEnabled(True)

    def cancel_enrollment(self) -> None:
        if self.worker is not None and self._enroll_ctx is not None:
            self.worker.cancel_enrollment()

    def _reset_enroll_buttons(self) -> None:
        self.btn_enroll_start.setEnabled(True)
        self.btn_enroll_cancel.setEnabled(False)

    def _on_enroll_finished(self, result) -> None:
        ctx, self._enroll_ctx = self._enroll_ctx, None
        self._reset_enroll_buttons()
        if ctx is None or result is None:
            self.statusBar().showMessage("Enrollment cancelled", 5000)
            return
        other = self.gallery.best_other(result.vectors, exclude=ctx["reg_no"])
        if other and other[1] >= config.DUPLICATE_THRESHOLD:
            who = self.gallery.names.get(other[0], "")
            QtWidgets.QMessageBox.critical(
                self, "Face already enrolled",
                f"This face closely matches {who} ({other[0]}), who is already enrolled under another Reg No "
                f"(similarity {other[1]:.2f}).\n\nNothing was saved. If this is a different person "
                f"(e.g. a twin), an administrator can raise DUPLICATE_THRESHOLD in config.py.")
            return
        try:
            self.db.save_enrollment(ctx["reg_no"], ctx["name"], ctx["dept"], ctx["section"],
                                    result.vectors, config.EMBEDDING_MODEL_TAG, replace=ctx["replace"])
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not save", str(exc))
            return
        if config.SAVE_FACE_IMAGES:
            folder = student_dir(ctx["reg_no"])
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(result.crops, 1):
                cv2.imwrite(str(folder / f"{i:02d}.png"), crop)
        self.reload_gallery()
        self.refresh_students()
        for edit in (self.ed_reg, self.ed_name, self.ed_dept, self.ed_section):
            edit.clear()
        QtWidgets.QMessageBox.information(
            self, "Enrolled", f"{ctx['name']} ({ctx['reg_no']}) enrolled with {len(result.vectors)} face samples.")

    # ------------------------------------------------------------------ students tab
    def refresh_students(self) -> None:
        rows = self.db.list_students(active_only=True)
        fill_table(self.tbl_students, ["Reg No", "Name", "Department", "Section", "Samples"],
                   [[r["reg_no"], r["name"], r["department"], r["section"], r["samples"]] for r in rows])
        self.lbl_students.setText(f"{len(rows)} students enrolled")
        self._filter_students()

    def _filter_students(self) -> None:
        needle = self.ed_search.text().strip().lower()
        for r in range(self.tbl_students.rowCount()):
            text = " ".join((self.tbl_students.item(r, c).text() if self.tbl_students.item(r, c) else "")
                            for c in range(self.tbl_students.columnCount())).lower()
            self.tbl_students.setRowHidden(r, bool(needle) and needle not in text)

    def _selected_reg(self):
        row = self.tbl_students.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Select a student", "Click a student in the table first.")
            return None
        return self.tbl_students.item(row, 0).text()

    def reenroll_selected(self) -> None:
        reg_no = self._selected_reg()
        student = self.db.get_student(reg_no) if reg_no else None
        if student:
            self.ed_reg.setText(student["reg_no"])
            self.ed_name.setText(student["name"])
            self.ed_dept.setText(student["department"])
            self.ed_section.setText(student["section"])
            self.tabs.setCurrentIndex(1)

    def remove_selected(self) -> None:
        reg_no = self._selected_reg()
        if not reg_no or not self.gate.ensure(self):
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Remove student",
            f"Remove {reg_no}?\n\nTheir face data is permanently deleted and they will no longer be "
            f"recognised. Past attendance records are kept.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.db.deactivate_student(reg_no)
        shutil.rmtree(student_dir(reg_no), ignore_errors=True)
        self.reload_gallery()
        self.refresh_students()

    # ------------------------------------------------------------------ reports
    def load_report(self, absentees: bool) -> None:
        start = self.date_from.date().toString("yyyy-MM-dd")
        end = self.date_to.date().toString("yyyy-MM-dd")
        subject = None if self.cmb_report_subject.currentIndex() <= 0 else self.cmb_report_subject.currentText()
        if absentees:
            self._report_header = ["Date", "Subject", "Reg No", "Name", "Department", "Section"]
            rows = [list(r) for r in self.db.absentees(start, end, subject)]
        else:
            self._report_header = ["Date", "Subject", "Reg No", "Name", "Department", "Section",
                                   "Time", "Status", "Method"]
            rows = [list(r) for r in self.db.report(start, end, subject)]
        self._report_rows = rows
        fill_table(self.tbl_report, self._report_header, rows)
        self.statusBar().showMessage(f"{len(rows)} rows", 5000)

    def export_report(self) -> None:
        if not self._report_rows:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Load a report first.")
            return
        default = config.EXPORT_DIR / f"report_{datetime.now():%Y%m%d_%H%M}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export report", str(default), "CSV (*.csv)")
        if path:
            write_csv(path, self._report_header, self._report_rows)
            self.statusBar().showMessage(f"Saved {path}", 6000)

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event) -> None:
        if self.session_id is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Session running", "A session is still running. End it and quit?")
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self.stop_camera()
        event.accept()


# =============================================================================== start-up
def setup_logging() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(config.LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run_first_time_setup(db: Database) -> bool:
    """Consent notice (once) and admin password (once). Returns False if the user backs out."""
    if not db.get_setting("consent_at"):
        box = QtWidgets.QMessageBox()
        box.setWindowTitle("Privacy & consent")
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setText(
            "This system captures facial images and stores numeric face templates (embeddings) "
            "on this computer to record attendance.\n\n"
            "By continuing you confirm that every person enrolled has been informed and has given "
            "consent, that the data is used only for attendance, and that you will delete a "
            "person's data when they leave or withdraw consent.")
        accept = box.addButton("I understand and accept", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Quit", QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() != accept:
            return False
        db.set_setting("consent_at", now_str())
        db.audit("consent-accepted")

    while not db.has_admin_password():
        password, ok = QtWidgets.QInputDialog.getText(
            None, "Create admin password",
            f"Choose an admin password (min {config.ADMIN_MIN_PASSWORD_LEN} characters).\n"
            "It protects enrollment, student removal and manual attendance:",
            QtWidgets.QLineEdit.Password)
        if not ok:
            return False
        again, ok = QtWidgets.QInputDialog.getText(
            None, "Confirm admin password", "Type the password again:", QtWidgets.QLineEdit.Password)
        if not ok:
            return False
        if password != again:
            QtWidgets.QMessageBox.warning(None, "Mismatch", "The two passwords differ. Try again.")
            continue
        try:
            db.set_admin_password(password)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(None, "Too weak", str(exc))
    return True


def main() -> int:
    setup_logging()
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    db = Database()
    try:
        db.backup(config.BACKUP_DIR, config.BACKUP_KEEP)
    except Exception:
        log.exception("Daily backup failed")
    if not run_first_time_setup(db):
        return 0
    window = MainWindow(db)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
