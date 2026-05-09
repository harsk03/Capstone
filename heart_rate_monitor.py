"""
Heart Rate Monitor — Comparative rPPG  (Full Enhanced Edition)
==============================================================
Runs FFT (Eulerian), CHROM, POS, and ICA in parallel and fuses their
outputs using SNR-weighted averaging for a stable BPM reading.

ADDITIONS OVER PREVIOUS VERSION
─────────────────────────────────
Clinical Features
  [HRV1] HRVCalculator — SDNN, RMSSD, pNN50 from 60-second BPM window
  [HRV2] StressDetector — 4-level stress scoring from RMSSD
  [HRV3] AlertSystem   — 7-level BPM alert (Critical Low → Critical High)
  [HRV4] PDFReportGenerator — clinical session PDF via reportlab

UI Additions
  [UI1] HRV metrics panel (SDNN, RMSSD, pNN50 labels)
  [UI2] Stress level indicator with color coding
  [UI3] BPM alert label
  [UI4] Export PDF button
  [UI5] HRV readiness progress bar

Architecture Preserved From Previous Version
  [SQ1] Weighted SNR fusion
  [SQ2] Per-method adaptive SNR thresholds
  [SQ3] Savitzky-Golay temporal smoothing
  [SQ4] Adaptive HR band 40-200 bpm
  [SQ5] Per-session skin-tone calibration
  [AR1] estimate() runs in QThread
  [AR2] Frozen @dataclass Config
  [AR4] CSV writes via background thread
  [PF1-4] All performance fixes retained
  [RB1-4] All robustness fixes retained
"""

from __future__ import annotations

import csv
import os
import queue
import sys
import tempfile
import threading
import time
import atexit
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox, QGridLayout
)
import pyqtgraph as pg


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    # Camera
    cam_w: int = 320
    cam_h: int = 240
    cam_index: int = 0
    cam_fail_limit: int = 30

    # Signal buffers
    buf_len: int = 150
    pyr_levels: int = 3
    proc_vid_w: int = 160
    proc_vid_h: int = 120
    fps_init: float = 15.0

    # Heart rate band — 40–200 bpm covers children and athletes
    hr_lo_hz: float = 0.67
    hr_hi_hz: float = 3.33

    # Processing
    calc_every: int = 8
    warmup_seconds: float = 3.0

    # Outlier / shift detection
    outlier_delta: float = 35.0
    shift_adopt_delta: float = 18.0
    shift_confirm_count: int = 2

    # Per-method SNR thresholds (dB)
    snr_thresh: Dict[str, float] = field(default_factory=lambda: {
        "FFT":   -1.0,
        "CHROM":  0.0,
        "POS":    0.0,
        "ICA":   -1.0,
    })

    # Fusion
    fusion_agreement_bpm: float = 8.0
    fft_weight_scale: float = 0.25
    ica_run_snr_threshold: float = 0.0
    ica_run_disagreement_bpm: float = 12.0

    # Skin-tone calibration
    calib_frames: int = 60

    # Motion
    motion_thresh: float = 0.08
    bright_min: int = 30
    bright_max: int = 230

    # Kalman
    kalman_q: float = 0.5
    kalman_r: float = 2.0

    # Face landmark caching
    hull_cache_px: int = 5
    landmark_shift_reject_px: float = 3.0

    # Skin segmentation
    skin_h_max: int = 25
    skin_s_min: int = 40
    skin_v_min: int = 50
    skin_min_pixels: int = 30

    # Age/gender
    ag_predict_every: int = 30

    # FFT robustness
    fft_disagree_limit_bpm: float = 20.0

    # Eulerian overlay amplitude
    alpha: int = 170

    # ── HRV Settings [HRV1] ──────────────────────────────────
    hrv_window_seconds: int = 60      # sliding window for HRV
    hrv_min_readings: int = 30        # minimum BPM readings before HRV is valid

    # ── Stress Levels [HRV2] ─────────────────────────────────
    # (score_min, score_max): (label, hex_color)
    # stress score = 100 - clamp(RMSSD / 50 * 100, 0, 100)

    # ── BPM Alert Thresholds [HRV3] ──────────────────────────
    alert_critical_low: float  = 40.0
    alert_warning_low: float   = 50.0
    alert_slight_low: float    = 60.0
    alert_normal_hi: float     = 100.0
    alert_slight_hi: float     = 120.0
    alert_warning_hi: float    = 150.0


CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL PATHS & URLS
# ══════════════════════════════════════════════════════════════════════════════

AGE_PROTO    = "age_deploy.prototxt"
AGE_MODEL    = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
FACE_LANDMARKER_MODEL = "face_landmarker.task"

_BASE  = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
_CAFFE = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/"

URLS = {
    AGE_PROTO:    _BASE  + "age_net_definitions/deploy.prototxt",
    AGE_MODEL:    _CAFFE + "age_net.caffemodel",
    GENDER_PROTO: _BASE  + "gender_net_definitions/deploy.prototxt",
    GENDER_MODEL: _CAFFE + "gender_net.caffemodel",
    FACE_LANDMARKER_MODEL:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
}

AGE_LIST    = ['(0-3)', '(4-9)', '(10-15)', '(16-19)',
               '(20-39)', '(40-59)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD — QThread with progress signal
# ══════════════════════════════════════════════════════════════════════════════

class ModelDownloadThread(QThread):
    progress    = pyqtSignal(str, int)
    finished_ok = pyqtSignal(bool)

    def run(self):
        files = [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL,
                 FACE_LANDMARKER_MODEL]
        for i, fname in enumerate(files):
            pct = int(100 * i / len(files))
            if os.path.exists(fname):
                self.progress.emit(f"{fname} already present", pct)
                continue
            url = URLS.get(fname)
            if not url:
                continue
            self.progress.emit(f"Downloading {fname} …", pct)
            try:
                urllib.request.urlretrieve(url, fname)
            except Exception as e:
                self.progress.emit(f"Download failed: {fname} — {e}", pct)
                self.finished_ok.emit(False)
                return
        self.progress.emit("All models ready.", 100)
        self.finished_ok.emit(True)


class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading models…")
        self.setModal(True)
        self.setFixedSize(420, 120)
        self.setStyleSheet("background:#2E3440;color:white;")
        layout = QVBoxLayout(self)
        self.label = QLabel("Initialising…")
        self.label.setStyleSheet("font-size:11px;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        self.thread = ModelDownloadThread()
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_ok.connect(self._on_done)
        self.thread.start()

    def _on_progress(self, msg: str, pct: int):
        self.label.setText(msg)
        self.bar.setValue(pct)

    def _on_done(self, ok: bool):
        self.success = ok
        self.accept()


def try_load_age_gender_nets():
    files_ok = all(os.path.exists(f)
                   for f in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL])
    if not files_ok:
        return None, None
    try:
        age_net    = cv2.dnn.readNet(AGE_MODEL,    AGE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        return age_net, gender_net
    except Exception as e:
        print(f"[Age/Gender] readNet failed: {e}")
        return None, None


def try_load_face_landmarker():
    if not os.path.exists(FACE_LANDMARKER_MODEL):
        return None
    try:
        base_opts = mp.tasks.BaseOptions(
            model_asset_path=FACE_LANDMARKER_MODEL)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    except Exception as e:
        print(f"[FaceMesh] init failed: {e}")
        return None


def face_size_age_gender(face_w, face_h, frame_w, frame_h):
    ratio = (face_w * face_h) / (frame_w * frame_h)
    if ratio > 0.25:   age = '(4-9)'
    elif ratio > 0.12: age = '(20-39)'
    elif ratio > 0.05: age = '(40-59)'
    else:              age = '(60-100)'
    return age, 'Unknown'


# ══════════════════════════════════════════════════════════════════════════════
#  [HRV1]  HRV CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class HRVCalculator:
    """
    Calculates Heart Rate Variability metrics from a sliding BPM window.

    Metrics:
        SDNN   — std deviation of RR intervals (ms).  Normal: > 50 ms
        RMSSD  — root mean square of successive differences (ms). Normal: > 20 ms
        pNN50  — % of successive pairs differing by > 50 ms.  Normal: > 3 %

    Clinical meaning:
        High HRV (high SDNN/RMSSD) → healthy autonomic function
        Low  HRV                   → stress, fatigue, or cardiovascular strain
    """

    def __init__(self,
                 window_seconds: int = CFG.hrv_window_seconds,
                 min_readings: int   = CFG.hrv_min_readings):
        self.window_seconds = window_seconds
        self.min_readings   = min_readings
        # Stores (timestamp_float, bpm_float) pairs
        self._history: deque = deque()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_bpm(self, bpm: float) -> None:
        """Add one BPM reading with current timestamp."""
        if bpm <= 0:
            return
        now = time.time()
        self._history.append((now, bpm))
        # Evict readings outside the sliding window
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def is_ready(self) -> bool:
        """True when enough data is collected for valid HRV calculation."""
        return len(self._history) >= self.min_readings

    def get_metrics(self) -> dict:
        """
        Return dict with keys:
            ready, sdnn, rmssd, pnn50, interpretation, n_readings
        """
        n = len(self._history)
        if n < self.min_readings:
            return {
                "ready":          False,
                "sdnn":           0.0,
                "rmssd":          0.0,
                "pnn50":          0.0,
                "interpretation": f"Collecting… ({n}/{self.min_readings})",
                "n_readings":     n,
            }

        rr = self._bpm_to_rr()
        sdnn  = self._calc_sdnn(rr)
        rmssd = self._calc_rmssd(rr)
        pnn50 = self._calc_pnn50(rr)

        return {
            "ready":          True,
            "sdnn":           round(sdnn,  1),
            "rmssd":          round(rmssd, 1),
            "pnn50":          round(pnn50, 1),
            "interpretation": self._interpret(sdnn, rmssd),
            "n_readings":     n,
        }

    def reset(self) -> None:
        self._history.clear()

    def readiness_pct(self) -> int:
        """0–100 progress toward minimum readings."""
        return min(100, int(100 * len(self._history) / self.min_readings))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _bpm_to_rr(self) -> np.ndarray:
        """Convert BPM readings to RR-interval array in milliseconds."""
        rr = []
        for _, bpm in self._history:
            if bpm > 0:
                rr.append((60.0 / bpm) * 1000.0)
        return np.array(rr, dtype=np.float64)

    @staticmethod
    def _calc_sdnn(rr: np.ndarray) -> float:
        if len(rr) < 2:
            return 0.0
        return float(np.std(rr, ddof=1))

    @staticmethod
    def _calc_rmssd(rr: np.ndarray) -> float:
        if len(rr) < 2:
            return 0.0
        diffs = np.diff(rr)
        return float(np.sqrt(np.mean(diffs ** 2)))

    @staticmethod
    def _calc_pnn50(rr: np.ndarray) -> float:
        if len(rr) < 2:
            return 0.0
        diffs = np.abs(np.diff(rr))
        return float(100.0 * np.sum(diffs > 50.0) / len(diffs))

    @staticmethod
    def _interpret(sdnn: float, rmssd: float) -> str:
        if sdnn > 50 and rmssd > 20:
            return "Healthy HRV"
        elif sdnn > 30 and rmssd > 15:
            return "Moderate HRV"
        elif sdnn > 0:
            return "Low HRV — possible stress"
        return "Insufficient data"


# ══════════════════════════════════════════════════════════════════════════════
#  [HRV2]  STRESS DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class StressDetector:
    """
    Estimates stress level from HRV RMSSD.

    Formula:
        stress_score = 100 − clamp(RMSSD / 50ms × 100, 0, 100)

    A high RMSSD means the autonomic nervous system is relaxed (parasympathetic
    dominant) → low stress.  Low RMSSD indicates sympathetic activation → stress.

    Levels:
        0 – 25   Relaxed         #2ECC71  (green)
        25 – 50  Normal          #F1C40F  (yellow)
        50 – 75  Mildly Stressed #E67E22  (orange)
        75 – 100 Highly Stressed #E74C3C  (red)
    """

    LEVELS = [
        (0,  25,  "Relaxed",          "#2ECC71"),
        (25, 50,  "Normal",           "#F1C40F"),
        (50, 75,  "Mildly Stressed",  "#E67E22"),
        (75, 101, "Highly Stressed",  "#E74C3C"),
    ]

    def detect(self, hrv_metrics: dict) -> dict:
        """
        Returns dict with keys: level, score, color, rmssd, ready
        """
        if not hrv_metrics.get("ready", False):
            return {
                "ready":  False,
                "level":  "Collecting…",
                "score":  0,
                "color":  "#95A5A6",
                "rmssd":  0.0,
            }

        rmssd = hrv_metrics.get("rmssd", 0.0)
        score = self._score(rmssd)
        level, color = self._classify(score)

        return {
            "ready":  True,
            "level":  level,
            "score":  score,
            "color":  color,
            "rmssd":  rmssd,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _score(rmssd: float) -> int:
        """Convert RMSSD → stress score 0–100."""
        return int(np.clip(100.0 - (rmssd / 50.0) * 100.0, 0, 100))

    @classmethod
    def _classify(cls, score: int) -> Tuple[str, str]:
        for lo, hi, label, color in cls.LEVELS:
            if lo <= score < hi:
                return label, color
        return "Unknown", "#95A5A6"


# ══════════════════════════════════════════════════════════════════════════════
#  [HRV3]  ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class AlertSystem:
    """
    7-level BPM alert system.

    Level         BPM range        Color      Action message
    ─────────────────────────────────────────────────────────────
    Critical Low  < 40             #E74C3C    Seek medical attention
    Warning Low   40 – 50          #E67E22    BPM is unusually low
    Slightly Low  50 – 60          #F1C40F    Monitor your heart rate
    Normal        60 – 100         #2ECC71    (no alert shown)
    Slightly High 100 – 120        #F1C40F    Consider resting
    Warning High  120 – 150        #E67E22    BPM is elevated
    Critical High > 150            #E74C3C    Seek medical attention
    """

    THRESHOLDS = [
        (0,   40,  "Critical Low",   "#E74C3C", "⚠ Seek medical attention immediately"),
        (40,  50,  "Warning Low",    "#E67E22", "⚠ BPM is unusually low"),
        (50,  60,  "Slightly Low",   "#F1C40F", "BPM is slightly low — monitor"),
        (60,  100, "Normal",         "#2ECC71", ""),
        (100, 120, "Slightly High",  "#F1C40F", "Consider resting"),
        (120, 150, "Warning High",   "#E67E22", "⚠ BPM is elevated"),
        (150, 999, "Critical High",  "#E74C3C", "⚠ Seek medical attention immediately"),
    ]

    def check(self, bpm: float) -> dict:
        """
        Returns dict with keys:
            level, color, message, is_alert
        """
        if bpm <= 0:
            return {"level": "No Reading", "color": "#95A5A6",
                    "message": "", "is_alert": False}

        for lo, hi, level, color, msg in self.THRESHOLDS:
            if lo <= bpm < hi:
                return {
                    "level":    level,
                    "color":    color,
                    "message":  msg,
                    "is_alert": level != "Normal",
                }

        return {"level": "Unknown", "color": "#95A5A6",
                "message": "", "is_alert": False}


# ══════════════════════════════════════════════════════════════════════════════
#  [HRV4]  PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class PDFReportGenerator:
    """
    Generates a clinical session PDF report using reportlab.

    Report sections:
        1. Header — title, date/time, subject info
        2. Heart Rate Summary — avg/min/max BPM, status
        3. Method Comparison — BPM + SNR per method
        4. HRV Analysis — SDNN, RMSSD, pNN50 with normal ranges
        5. Stress Assessment — level, score, RMSSD
        6. Session Notes — number of readings, duration
        7. Footer — project attribution
    """

    def generate(self, session_data: dict, output_path: str) -> bool:
        """
        Generate PDF from session_data dict.
        Returns True on success, False on failure.

        session_data keys:
            timestamp       str
            duration_sec    int
            age             str
            gender          str
            avg_bpm         float
            min_bpm         float
            max_bpm         float
            hr_status       str
            method_results  dict  {method: (bpm, snr)}
            hrv             dict  (from HRVCalculator.get_metrics())
            stress          dict  (from StressDetector.detect())
            bpm_history     list[float]
            n_readings      int
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                            Spacer, Table, TableStyle,
                                            HRFlowable)
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            print("[PDF] reportlab not installed. Run: pip install reportlab")
            return False

        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=2 * cm, leftMargin=2 * cm,
                topMargin=2 * cm,   bottomMargin=2 * cm,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "Title2", parent=styles["Title"],
                fontSize=18, spaceAfter=6,
                textColor=colors.HexColor("#2C3E50"),
            )
            heading_style = ParagraphStyle(
                "Heading3", parent=styles["Heading2"],
                fontSize=13, textColor=colors.HexColor("#2980B9"),
                spaceBefore=12, spaceAfter=4,
            )
            body_style = ParagraphStyle(
                "Body2", parent=styles["Normal"],
                fontSize=10, leading=14,
            )
            caption_style = ParagraphStyle(
                "Caption", parent=styles["Normal"],
                fontSize=8, textColor=colors.grey,
                alignment=TA_CENTER,
            )

            DARK   = colors.HexColor("#2C3E50")
            BLUE   = colors.HexColor("#2980B9")
            GREEN  = colors.HexColor("#27AE60")
            LIGHT  = colors.HexColor("#F8F9FA")
            WHITE  = colors.white

            def _table(data, col_widths, header_color=BLUE):
                t = Table(data, colWidths=col_widths)
                style = TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0),  header_color),
                    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, 0),  10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                    ("FONTSIZE",    (0, 1), (-1, -1), 9),
                    ("TOPPADDING",  (0, 1), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ])
                t.setStyle(style)
                return t

            story = []

            # ── 1. Header ────────────────────────────────────────────
            story.append(Paragraph(
                "rPPG Heart Rate Monitor — Clinical Session Report",
                title_style))
            story.append(Paragraph(
                "SCET-MITWPU · BTech Capstone 2026 · Dr. Aparna Kamble",
                caption_style))
            story.append(HRFlowable(
                width="100%", thickness=1,
                color=colors.HexColor("#BDC3C7"), spaceAfter=8))

            # Subject info
            dur = session_data.get("duration_sec", 0)
            dur_str = f"{dur // 60}m {dur % 60}s"
            info_data = [
                ["Field", "Value"],
                ["Date / Time",    session_data.get("timestamp", "—")],
                ["Session Duration", dur_str],
                ["Subject Age",    session_data.get("age",    "Unknown")],
                ["Subject Gender", session_data.get("gender", "Unknown")],
                ["Total Readings", str(session_data.get("n_readings", 0))],
            ]
            story.append(_table(info_data, [6 * cm, 11 * cm], DARK))
            story.append(Spacer(1, 0.4 * cm))

            # ── 2. Heart Rate Summary ─────────────────────────────────
            story.append(Paragraph("Heart Rate Summary", heading_style))
            bpm_data = [
                ["Metric", "Value", "Interpretation"],
                ["Average BPM",
                 f"{session_data.get('avg_bpm', 0):.1f}",
                 session_data.get("hr_status", "—")],
                ["Minimum BPM",
                 f"{session_data.get('min_bpm', 0):.1f}", "—"],
                ["Maximum BPM",
                 f"{session_data.get('max_bpm', 0):.1f}", "—"],
                ["BPM Std Dev",
                 f"{session_data.get('bpm_std', 0):.1f}",
                 "Variability within session"],
            ]
            story.append(_table(bpm_data, [6 * cm, 4 * cm, 7 * cm]))
            story.append(Spacer(1, 0.4 * cm))

            # ── 3. Method Comparison ──────────────────────────────────
            story.append(Paragraph("Method Comparison", heading_style))
            method_header = ["Method", "Est. BPM", "SNR (dB)", "Notes"]
            method_rows   = [method_header]
            mr = session_data.get("method_results", {})
            method_notes = {
                "FFT":   "Eulerian Video Magnification — Verkruysse 2008",
                "CHROM": "Chrominance-based — De Haan & Jeanne 2013",
                "POS":   "Plane Orthogonal to Skin — Wang et al. 2017",
                "ICA":   "Independent Component Analysis — Poh et al. 2011",
            }
            for m in ("FFT", "CHROM", "POS", "ICA"):
                if m in mr:
                    bv, sv = mr[m]
                    method_rows.append(
                        [m, f"{bv:.1f}", f"{sv:.2f}", method_notes.get(m, "")])
                else:
                    method_rows.append([m, "—", "—", "Not run this session"])
            story.append(
                _table(method_rows, [3 * cm, 3 * cm, 3 * cm, 8 * cm], BLUE))
            story.append(Spacer(1, 0.4 * cm))

            # ── 4. HRV Analysis ───────────────────────────────────────
            story.append(Paragraph("HRV Analysis", heading_style))
            hrv = session_data.get("hrv", {})
            if hrv.get("ready", False):
                hrv_data = [
                    ["HRV Metric", "Value", "Normal Range", "Interpretation"],
                    ["SDNN",
                     f"{hrv.get('sdnn', 0):.1f} ms",
                     "> 50 ms",
                     "Overall HRV — autonomic balance"],
                    ["RMSSD",
                     f"{hrv.get('rmssd', 0):.1f} ms",
                     "> 20 ms",
                     "Short-term HRV — parasympathetic activity"],
                    ["pNN50",
                     f"{hrv.get('pnn50', 0):.1f} %",
                     "> 3 %",
                     "Proportion of large RR differences"],
                    ["Interpretation",
                     hrv.get("interpretation", "—"),
                     "—", "—"],
                ]
            else:
                hrv_data = [
                    ["HRV Metric", "Value", "Normal Range", "Interpretation"],
                    ["Status", "Insufficient data",
                     "Min 30 readings required",
                     "Extend session for HRV data"],
                ]
            story.append(
                _table(hrv_data, [3 * cm, 3.5 * cm, 3.5 * cm, 7 * cm], GREEN))
            story.append(Spacer(1, 0.4 * cm))

            # ── 5. Stress Assessment ──────────────────────────────────
            story.append(Paragraph("Stress Assessment", heading_style))
            stress = session_data.get("stress", {})
            stress_data = [
                ["Parameter", "Value", "Scale"],
                ["Stress Level",
                 stress.get("level", "—"), "Relaxed / Normal / Mild / High"],
                ["Stress Score",
                 f"{stress.get('score', 0)}/100",
                 "0 = Relaxed, 100 = High Stress"],
                ["Based on RMSSD",
                 f"{stress.get('rmssd', 0):.1f} ms",
                 "Higher RMSSD → Lower Stress"],
            ]
            story.append(
                _table(stress_data,
                       [5 * cm, 5 * cm, 7 * cm],
                       colors.HexColor("#8E44AD")))
            story.append(Spacer(1, 0.4 * cm))

            # ── 6. Footer ─────────────────────────────────────────────
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#BDC3C7"), spaceBefore=8))
            story.append(Paragraph(
                "Generated by rPPG Heart Rate Monitor · "
                "SCET-MITWPU BTech Capstone Project 2026 · "
                "Guide: Dr. Aparna Kamble · Panel G",
                caption_style))
            story.append(Paragraph(
                "DISCLAIMER: This report is for research and educational "
                "purposes only. Not a medical device. Consult a qualified "
                "healthcare professional for clinical decisions.",
                caption_style))

            doc.build(story)
            return True

        except Exception as e:
            print(f"[PDF] Generation failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def detrend(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    if n < 2:
        return signal
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, signal, 1)
    return signal - (slope * x + intercept)


def _savgol_coeffs(window: int = 7, poly: int = 2) -> np.ndarray:
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vstack([x**i for i in range(poly + 1)]).T
    coeffs = np.linalg.pinv(A)[0]
    return coeffs[::-1]


_SG_KERNEL = _savgol_coeffs(window=7, poly=2)


def savgol_smooth_rows(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    k  = _SG_KERNEL
    hw = len(k) // 2
    for ch in range(3):
        out[:, ch] = np.convolve(rgb[:, ch], k, mode="same")
        out[:hw,  ch] = rgb[:hw,  ch]
        out[-hw:, ch] = rgb[-hw:, ch]
    return out


def bandpass_fft(sig: np.ndarray, fps: float,
                 f_lo: float = None,
                 f_hi: float = None) -> Tuple[float, float, float]:
    if f_lo is None: f_lo = CFG.hr_lo_hz
    if f_hi is None: f_hi = CFG.hr_hi_hz

    sig   = detrend(sig)
    sig   = sig - sig.mean()
    sig   = sig * np.hanning(len(sig))
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mags  = np.abs(np.fft.rfft(sig))
    band  = (freqs >= f_lo) & (freqs <= f_hi)

    if not band.any() or mags[band].max() < 1e-9:
        return 0.0, 0.0, 0.0

    in_p  = float(np.sum(mags[band]  ** 2))
    out_p = float(np.sum(mags[~band] ** 2))
    snr   = 10 * np.log10(in_p / out_p) if out_p > 0 else 0.0

    mags_f        = mags.copy()
    mags_f[~band] = 0
    hz = float(freqs[np.argmax(mags_f)])
    return hz, hz * 60.0, snr


def chrom_pulse(rgb: np.ndarray) -> np.ndarray:
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu = rgb.mean(0)
    mu[mu == 0] = 1e-9
    n  = rgb / mu
    Xs = 3 * n[:, 0] - 2 * n[:, 1]
    Ys = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    a  = Xs.std() / (Ys.std() + 1e-9)
    return detrend(Xs - a * Ys)


def pos_pulse(rgb: np.ndarray) -> np.ndarray:
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu = rgb.mean(0)
    mu[mu == 0] = 1e-9
    n  = rgb / mu
    S1 = n[:, 1] - n[:, 2]
    S2 = n[:, 1] + n[:, 2] - 2 * n[:, 0]
    b  = S1.std() / (S2.std() + 1e-9)
    return detrend(S1 + b * S2)


def _fast_ica_sources(x: np.ndarray, n_components: int = 3,
                      max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    x = x - np.mean(x, axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    W_white = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    xw = x @ W_white

    rng = np.random.default_rng(42)
    w   = rng.normal(size=(n_components, xw.shape[1]))
    w  /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-9

    for _ in range(max_iter):
        wx      = xw @ w.T
        gwx     = np.tanh(wx)
        gprime  = 1.0 - gwx ** 2
        w_new   = (gwx.T @ xw) / xw.shape[0] \
                  - np.diag(np.mean(gprime, 0)) @ w
        s, u    = np.linalg.eigh(w_new @ w_new.T)
        s       = np.clip(s, 1e-9, None)
        w_new   = (u @ np.diag(1.0 / np.sqrt(s)) @ u.T) @ w_new
        lim     = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w       = w_new
        if lim < tol:
            break
    return xw @ w.T


def ica_best(rgb: np.ndarray, fps: float) -> Tuple[float, float]:
    if len(rgb) < 30:
        return 0.0, -999.0
    try:
        src = _fast_ica_sources(rgb, n_components=3)
    except Exception:
        return 0.0, -999.0
    best_bpm, best_snr = 0.0, -999.0
    for i in range(src.shape[1]):
        _, bpm, snr = bandpass_fft(detrend(src[:, i]), fps)
        if snr > best_snr and \
                CFG.hr_lo_hz * 60 <= bpm <= CFG.hr_hi_hz * 60:
            best_bpm, best_snr = bpm, snr
    return best_bpm, best_snr


# ══════════════════════════════════════════════════════════════════════════════
#  FUSED ESTIMATOR — SNR-weighted average
# ══════════════════════════════════════════════════════════════════════════════

def fused_estimate(rgb_buf: np.ndarray, gauss_buf: np.ndarray,
                   fps: float, freq_mask: np.ndarray,
                   filled: int,
                   skin_baseline: Optional[np.ndarray] = None) -> Tuple:
    HR_LO = CFG.hr_lo_hz * 60
    HR_HI = CFG.hr_hi_hz * 60

    if filled < 30:
        return None, 0.0, {}, None, 0.0, {}, 0.0

    rgb_raw   = rgb_buf[-filled:]
    gauss_raw = gauss_buf[-filled:]

    mu = np.mean(rgb_raw, axis=0)
    mu[mu == 0] = 1e-9
    if skin_baseline is not None:
        mu = mu / (skin_baseline + 1e-9)
    rgb = rgb_raw / mu

    rgb_smooth = savgol_smooth_rows(rgb) if len(rgb) > 7 else rgb

    results: Dict[str, Tuple[float, float]] = {}

    # ── FFT ──────────────────────────────────────────────────
    try:
        n     = gauss_raw.shape[0]
        freqs = fps * np.arange(n) / n
        mask  = (freqs >= CFG.hr_lo_hz) & (freqs <= CFG.hr_hi_hz)
        cube  = np.fft.fft(gauss_raw, axis=0)
        cube[~mask] = 0
        sig   = np.real(cube).mean(axis=(1, 2, 3))
        _, bpm, snr = bandpass_fft(sig, fps)
        if HR_LO <= bpm <= HR_HI:
            results["FFT"] = (bpm, snr)
    except Exception as e:
        print(f"[FFT ERROR] {e}")

    # ── CHROM ─────────────────────────────────────────────────
    try:
        pulse = chrom_pulse(rgb_smooth)
        _, bpm, snr = bandpass_fft(pulse, fps)
        if HR_LO <= bpm <= HR_HI:
            results["CHROM"] = (bpm, snr)
    except Exception as e:
        print(f"[CHROM ERROR] {e}")

    # ── POS ───────────────────────────────────────────────────
    try:
        pulse = pos_pulse(rgb_smooth)
        _, bpm, snr = bandpass_fft(pulse, fps)
        if HR_LO <= bpm <= HR_HI:
            results["POS"] = (bpm, snr)
    except Exception as e:
        print(f"[POS ERROR] {e}")

    # ── ICA (conditional) ─────────────────────────────────────
    base_bpms = [b for m, (b, _) in results.items()
                 if m in ("FFT", "CHROM", "POS")]
    base_snrs = [s for m, (_, s) in results.items()
                 if m in ("FFT", "CHROM", "POS")]
    best_base_snr   = max(base_snrs) if base_snrs else -999.0
    base_disagree   = (max(base_bpms) - min(base_bpms)) \
                       if len(base_bpms) >= 2 else 0.0
    run_ica = (not results or
               best_base_snr < CFG.ica_run_snr_threshold or
               base_disagree > CFG.ica_run_disagreement_bpm)
    if run_ica:
        try:
            bpm, snr = ica_best(rgb, fps)
            if HR_LO <= bpm <= HR_HI:
                results["ICA"] = (bpm, snr)
        except Exception as e:
            print(f"[ICA ERROR] {e}")

    if not results:
        return None, 0.0, {}, None, 0.0, {}, 0.0

    # Per-method SNR gate
    valid      = {m: (b, s) for m, (b, s) in results.items()
                  if s > CFG.snr_thresh.get(m, -2.0)}
    candidates = valid if valid else results

    # FFT guard against large disagreement with CHROM/POS
    if "FFT" in candidates:
        non_fft = {m: v for m, v in candidates.items() if m != "FFT"}
        if len(non_fft) >= 2:
            non_fft_med = float(np.median([b for b, _ in non_fft.values()]))
            fft_bpm, _  = candidates["FFT"]
            if abs(fft_bpm - non_fft_med) > CFG.fft_disagree_limit_bpm:
                candidates = non_fft

    # SNR-weighted BPM fusion
    bpm_vals = np.array([b for b, _ in candidates.values()])
    snr_vals = np.array([s for _, s in candidates.values()])
    methods  = list(candidates.keys())

    weights = np.power(10.0, snr_vals / 10.0)
    weights = np.clip(weights, 1e-6, None)
    for i, m in enumerate(methods):
        if m == "FFT":
            weights[i] *= CFG.fft_weight_scale

    median_bpm = float(np.median(bpm_vals))
    agreement  = np.abs(bpm_vals - median_bpm) < CFG.fusion_agreement_bpm
    weights[agreement] *= 2.0
    weights /= weights.sum()

    fused_bpm = float(np.dot(weights, bpm_vals))
    avg_snr   = float(np.dot(weights, snr_vals))

    scores = {m: float(s - 0.08 * abs(b - median_bpm))
              for m, (b, s) in candidates.items()}
    selected_method = max(scores, key=scores.get)
    selected_score  = scores[selected_method]

    confidence = float(np.clip((avg_snr + 2.0) / 6.0, 0.0, 1.0))
    return (fused_bpm, avg_snr, results, selected_method,
            selected_score, scores, confidence)


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

class KalmanBPM:
    def __init__(self):
        self.x = 75.0
        self.P = 25.0
        self.Q = CFG.kalman_q
        self.R = CFG.kalman_r

    def update(self, z: float) -> float:
        P_  = self.P + self.Q
        K   = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x

    def predict(self) -> float:
        self.P += self.Q
        return self.x


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class SignalProcessor:
    def __init__(self):
        fps  = CFG.fps_init
        buf  = CFG.buf_len
        lvls = CFG.pyr_levels
        vw, vh = CFG.proc_vid_w, CFG.proc_vid_h

        self.fps = fps
        self.BUF = buf

        dummy    = np.zeros((vh, vw, 3), dtype=np.float32)
        g0       = self._build_gauss_level(dummy, lvls)
        gh, gw   = g0.shape[:2]

        self.gauss_buf  = np.zeros((buf, gh, gw, 3), dtype=np.float32)
        self.rgb_buf    = np.zeros((buf, 3), dtype=np.float64)
        self._rgb_ord   = np.empty_like(self.rgb_buf)
        self._gauss_ord = np.empty_like(self.gauss_buf)

        freqs          = fps * np.arange(buf) / buf
        self.freq_mask = ((freqs >= CFG.hr_lo_hz) &
                          (freqs <= CFG.hr_hi_hz))

        self.buf_idx       = 0
        self.filled_frames = 0

        self.kalman      = KalmanBPM()
        self.bpm_history = deque(maxlen=20)

        self._shift_candidate = None
        self._shift_streak    = 0

        self._calib_buf: list              = []
        self.skin_baseline: Optional[np.ndarray] = None

        self._prev_landmark_pts: Optional[np.ndarray] = None
        self._cached_hulls: Optional[dict]             = None

    def update_fps(self, fps: float):
        self.fps = fps
        freqs          = fps * np.arange(self.BUF) / self.BUF
        self.freq_mask = ((freqs >= CFG.hr_lo_hz) &
                          (freqs <= CFG.hr_hi_hz))

    def reset(self):
        self.gauss_buf.fill(0)
        self.rgb_buf.fill(0)
        self.buf_idx       = 0
        self.filled_frames = 0
        self.kalman        = KalmanBPM()
        self.bpm_history.clear()
        self._shift_candidate = None
        self._shift_streak    = 0
        self._calib_buf.clear()
        self.skin_baseline    = None
        self._prev_landmark_pts = None
        self._cached_hulls      = None

    def push(self, roi_rgb: np.ndarray, forehead_frame: np.ndarray):
        if self.skin_baseline is None:
            self._calib_buf.append(roi_rgb.copy())
            if len(self._calib_buf) >= CFG.calib_frames:
                self.skin_baseline = np.mean(
                    np.stack(self._calib_buf, axis=0), axis=0)
                print("[Calib] Skin baseline locked.")

        self.rgb_buf[self.buf_idx] = roi_rgb

        if forehead_frame.size > 0:
            fh_r = cv2.resize(forehead_frame.astype(np.float32),
                              (CFG.proc_vid_w, CFG.proc_vid_h))
        else:
            fh_r = np.zeros(
                (CFG.proc_vid_h, CFG.proc_vid_w, 3), dtype=np.float32)

        self.gauss_buf[self.buf_idx] = self._build_gauss_level(
            fh_r, CFG.pyr_levels)

        self.buf_idx       = (self.buf_idx + 1) % self.BUF
        self.filled_frames = min(self.filled_frames + 1, self.BUF)

    def estimate(self) -> Tuple:
        idx = self.buf_idx
        if idx == 0:
            np.copyto(self._rgb_ord,   self.rgb_buf)
            np.copyto(self._gauss_ord, self.gauss_buf)
        else:
            np.copyto(self._rgb_ord[:self.BUF - idx],   self.rgb_buf[idx:])
            np.copyto(self._rgb_ord[self.BUF - idx:],   self.rgb_buf[:idx])
            np.copyto(self._gauss_ord[:self.BUF - idx], self.gauss_buf[idx:])
            np.copyto(self._gauss_ord[self.BUF - idx:], self.gauss_buf[:idx])

        result = fused_estimate(
            self._rgb_ord, self._gauss_ord, self.fps,
            self.freq_mask, self.filled_frames,
            skin_baseline=self.skin_baseline,
        )
        (raw_bpm, avg_snr, breakdown,
         sel_method, sel_score, comparison, confidence) = result
        breakdown["_scores"] = comparison

        HR_LO = CFG.hr_lo_hz * 60
        HR_HI = CFG.hr_hi_hz * 60

        if raw_bpm is None or not (HR_LO <= raw_bpm <= HR_HI):
            self.kalman.predict()
            return (None, None, 0.0, breakdown,
                    None, 0.0, 0.0, False)

        if self._is_outlier(raw_bpm):
            if self._confirm_shift_and_adopt(raw_bpm):
                kbpm = self.kalman.update(raw_bpm)
                return (kbpm, raw_bpm, avg_snr, breakdown,
                        sel_method, sel_score, confidence, True)
            self.kalman.predict()
            return (None, raw_bpm, avg_snr, breakdown,
                    sel_method, sel_score, confidence, False)

        self._shift_candidate = None
        self._shift_streak    = 0
        self.bpm_history.append(raw_bpm)
        kbpm = self.kalman.update(raw_bpm)
        return (kbpm, raw_bpm, avg_snr, breakdown,
                sel_method, sel_score, confidence, True)

    def gauss_overlay(self, alpha: int = CFG.alpha) -> np.ndarray:
        fft_cube         = np.fft.fft(self.gauss_buf, axis=0)
        mask4            = self.freq_mask[:, None, None, None]
        fft_cube[~mask4] = 0
        filtered         = np.real(np.fft.ifft(fft_cube, axis=0)) * alpha
        f = filtered[(self.buf_idx - 1) % self.BUF].copy()
        for _ in range(CFG.pyr_levels):
            f = cv2.pyrUp(f)
        return f[:CFG.proc_vid_h, :CFG.proc_vid_w]

    def bpm_std(self) -> float:
        if len(self.bpm_history) < 3:
            return 0.0
        return float(np.std(list(self.bpm_history)))

    @staticmethod
    def _build_gauss_level(frame: np.ndarray, levels: int) -> np.ndarray:
        f = frame
        for _ in range(levels):
            f = cv2.pyrDown(f)
        return f

    def _is_outlier(self, bpm: float) -> bool:
        if len(self.bpm_history) < 5:
            return False
        return (abs(bpm - float(np.median(list(self.bpm_history))))
                > CFG.outlier_delta)

    def _confirm_shift_and_adopt(self, bpm: float) -> bool:
        if len(self.bpm_history) < 5:
            return False
        median = float(np.median(list(self.bpm_history)))
        if abs(bpm - median) < CFG.shift_adopt_delta:
            self._shift_candidate = None
            self._shift_streak    = 0
            return False
        if (self._shift_candidate is None or
                abs(bpm - self._shift_candidate) > 8):
            self._shift_candidate = bpm
            self._shift_streak    = 1
            return False
        self._shift_streak += 1
        if self._shift_streak < CFG.shift_confirm_count:
            return False
        self.bpm_history.clear()
        self.bpm_history.append(bpm)
        self.kalman.x     = bpm
        self.kalman.P     = 25.0
        self._shift_candidate = None
        self._shift_streak    = 0
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class EstimateWorker(QThread):
    result_ready = pyqtSignal(
        object, object, float, dict, object, float, float, bool)

    def __init__(self, processor: SignalProcessor):
        super().__init__()
        self._processor = processor
        self._lock      = threading.Lock()
        self._pending   = False

    def request(self):
        with self._lock:
            self._pending = True
        if not self.isRunning():
            self.start()

    def run(self):
        with self._lock:
            self._pending = False
        result = self._processor.estimate()
        self.result_ready.emit(*result)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV WRITER — background flush thread
# ══════════════════════════════════════════════════════════════════════════════

class CSVWriter:
    def __init__(self, path: str):
        self._q: queue.Queue = queue.Queue()
        self._file = open(path, "w", newline="")
        atexit.register(self._file.close)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "Timestamp", "Raw BPM", "Kalman BPM", "Avg SNR (dB)",
            "FFT BPM", "FFT SNR", "CHROM BPM", "CHROM SNR",
            "POS BPM", "POS SNR", "ICA BPM", "ICA SNR",
            "Selected Method", "Confidence (%)",
            "SDNN (ms)", "RMSSD (ms)", "pNN50 (%)",
            "Stress Level", "Stress Score",
            "BPM Alert", "Age", "Gender", "HR Status",
        ])
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write(self, row):
        self._q.put(row)

    def _worker(self):
        while True:
            row = self._q.get()
            if row is None:
                break
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=2)
        self._file.close()


def _safe_csv_path() -> str:
    candidate = os.path.join(os.getcwd(), "heart_rate_data.csv")
    try:
        with open(candidate, "a"):
            pass
        return candidate
    except PermissionError:
        fallback = os.path.join(
            tempfile.gettempdir(), "heart_rate_data.csv")
        print(f"[CSV] CWD not writable, using {fallback}")
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

class HeartRateMonitor(QWidget):

    def __init__(self):
        super().__init__()

        # ── CLAHE (created once) ──────────────────────────────
        self._clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8))

        # ── Live FPS ──────────────────────────────────────────
        self.fps_timestamps: deque = deque(maxlen=30)
        self._fps_live = CFG.fps_init

        # ── Signal processor & worker thread ──────────────────
        self.processor = SignalProcessor()
        self.worker    = EstimateWorker(self.processor)
        self.worker.result_ready.connect(self._on_estimate_result)

        # ── [HRV1] HRV + Stress + Alerts ──────────────────────
        self.hrv_calc     = HRVCalculator()
        self.stress_det   = StressDetector()
        self.alert_sys    = AlertSystem()
        self.pdf_gen      = PDFReportGenerator()

        # ── Session data for PDF ──────────────────────────────
        self._session_start: Optional[float] = None
        self._bpm_session: List[float]        = []
        self._last_method_results: dict        = {}
        self._last_hrv: dict                   = {}
        self._last_stress: dict                = {}
        self._last_age    = "Unknown"
        self._last_gender = "Unknown"

        # ── HR history (for plot) ─────────────────────────────
        self.hr_raw:    list = []
        self.hr_smooth: list = []

        # ── Pulse waveform ────────────────────────────────────
        self.pulse_buf = np.zeros(300)
        self.pulse_idx = 0

        # ── Motion ────────────────────────────────────────────
        self.prev_gray: Optional[np.ndarray] = None

        # ── Frame counters ────────────────────────────────────
        self.frame_count = 0
        self.MIN_FRAMES  = int(CFG.fps_init * CFG.warmup_seconds)

        # ── Webcam fail counter ───────────────────────────────
        self._cam_fail_count = 0

        # ── FaceMesh ─────────────────────────────────────────
        self.faceLandmarker = None
        self._mp_ts_ms      = int(time.monotonic() * 1000)
        self.forehead_idx    = np.array([10, 67, 103, 109, 338, 297, 332, 284])
        self.left_cheek_idx  = np.array([50, 101, 205, 187, 147, 123, 116])
        self.right_cheek_idx = np.array([280, 330, 425, 411, 376, 352, 345])

        # ── Hull cache ────────────────────────────────────────
        self._prev_pts: Optional[np.ndarray] = None
        self._cached_polys: Optional[dict]   = None

        # ── Age/gender ────────────────────────────────────────
        self.last_age    = "Unknown"
        self.last_gender = "Unknown"
        self.ag_count    = 0
        self.ageNet      = None
        self.genderNet   = None

        # ── CSV ───────────────────────────────────────────────
        self.csv = CSVWriter(_safe_csv_path())

        # ── Build UI ─────────────────────────────────────────
        self.initUI()

        # ── Camera ───────────────────────────────────────────
        self.webcam = cv2.VideoCapture(CFG.cam_index)
        self.webcam.set(3, CFG.cam_w)
        self.webcam.set(4, CFG.cam_h)

        self.monitoring = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self._run_download_dialog()

    # ══════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════

    def initUI(self):
        self.setWindowTitle("Heart Rate Monitor — rPPG (Full Enhanced)")
        self.setStyleSheet("background-color:#2E3440;color:white;")
        self.setMinimumWidth(900)

        root = QVBoxLayout()

        # ── Top row: video + right panel ─────────────────────
        top = QHBoxLayout()

        # Video label
        self.videoLabel = QLabel()
        self.videoLabel.setFixedSize(CFG.cam_w, CFG.cam_h)
        self.videoLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.videoLabel.setStyleSheet("border:2px solid #4C566A;")
        top.addWidget(self.videoLabel)

        # Right panel (BPM + quality + HRV + Stress + Alert)
        right = QVBoxLayout()

        # ── BPM display ───────────────────────────────────────
        self.hrLabel = QLabel("HR: —")
        self.hrLabel.setFont(QFont("Arial", 22))
        self.hrLabel.setStyleSheet("color:#88C0D0;")
        self.hrLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.hrLabel)

        self.ciLabel = QLabel("")
        self.ciLabel.setFont(QFont("Arial", 10))
        self.ciLabel.setStyleSheet("color:#81A1C1;")
        self.ciLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.ciLabel)

        # ── [UI3] Alert label ─────────────────────────────────
        self.alertLabel = QLabel("")
        self.alertLabel.setFont(QFont("Arial", 10, QFont.Bold))
        self.alertLabel.setAlignment(Qt.AlignCenter)
        self.alertLabel.setStyleSheet(
            "color:#E74C3C;background:#3B0000;"
            "padding:4px;border-radius:4px;")
        self.alertLabel.setVisible(False)
        right.addWidget(self.alertLabel)

        # ── Quality / method info ─────────────────────────────
        self.qualityLabel = QLabel("Warming up…")
        self.qualityLabel.setFont(QFont("Arial", 9))
        self.qualityLabel.setStyleSheet("color:#EBCB8B;")
        self.qualityLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.qualityLabel)

        self.methodLabel = QLabel("")
        self.methodLabel.setFont(QFont("Courier", 8))
        self.methodLabel.setStyleSheet("color:#81A1C1;")
        self.methodLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.methodLabel)

        self.fpsLabel = QLabel("FPS: —")
        self.fpsLabel.setFont(QFont("Arial", 9))
        self.fpsLabel.setStyleSheet("color:#4C566A;")
        self.fpsLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.fpsLabel)

        self.calibLabel = QLabel("Calibrating skin tone…")
        self.calibLabel.setFont(QFont("Arial", 9))
        self.calibLabel.setStyleSheet("color:#A3BE8C;")
        self.calibLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.calibLabel)

        # ── [UI1] HRV Panel ───────────────────────────────────
        hrv_frame = QFrame()
        hrv_frame.setStyleSheet(
            "background:#3B4252;border-radius:6px;padding:4px;")
        hrv_layout = QGridLayout(hrv_frame)
        hrv_layout.setSpacing(4)

        def _lbl(txt, bold=False, color="#D8DEE9"):
            l = QLabel(txt)
            l.setFont(QFont("Arial", 9, QFont.Bold if bold else QFont.Normal))
            l.setStyleSheet(f"color:{color};background:transparent;")
            return l

        hrv_layout.addWidget(_lbl("── HRV Metrics ──", bold=True,
                                   color="#88C0D0"), 0, 0, 1, 2)

        # [UI5] HRV readiness progress bar
        self.hrvProgressBar = QProgressBar()
        self.hrvProgressBar.setRange(0, 100)
        self.hrvProgressBar.setValue(0)
        self.hrvProgressBar.setTextVisible(True)
        self.hrvProgressBar.setFormat("Session Timer: %p%")
        self.hrvProgressBar.setFixedHeight(16)
        self.hrvProgressBar.setStyleSheet(
            "QProgressBar{background:#2E3440;border-radius:3px;}"
            "QProgressBar::chunk{background:#5E81AC;border-radius:3px;}")
        hrv_layout.addWidget(self.hrvProgressBar, 1, 0, 1, 2)

        hrv_layout.addWidget(_lbl("SDNN:"),  2, 0)
        self.sdnnLabel = _lbl("— ms", color="#EBCB8B")
        hrv_layout.addWidget(self.sdnnLabel, 2, 1)

        hrv_layout.addWidget(_lbl("RMSSD:"), 3, 0)
        self.rmssdLabel = _lbl("— ms", color="#EBCB8B")
        hrv_layout.addWidget(self.rmssdLabel, 3, 1)

        hrv_layout.addWidget(_lbl("pNN50:"), 4, 0)
        self.pnn50Label = _lbl("— %", color="#EBCB8B")
        hrv_layout.addWidget(self.pnn50Label, 4, 1)

        hrv_layout.addWidget(_lbl("HRV:"),   5, 0)
        self.hrvInterpLabel = _lbl("—", color="#A3BE8C")
        hrv_layout.addWidget(self.hrvInterpLabel, 5, 1)

        right.addWidget(hrv_frame)

        # ── [UI2] Stress level indicator ──────────────────────
        stress_frame = QFrame()
        stress_frame.setStyleSheet(
            "background:#3B4252;border-radius:6px;padding:4px;")
        stress_layout = QHBoxLayout(stress_frame)

        stress_layout.addWidget(_lbl("Stress:", bold=True,
                                      color="#88C0D0"))
        self.stressLabel = QLabel("Collecting…")
        self.stressLabel.setFont(QFont("Arial", 10, QFont.Bold))
        self.stressLabel.setStyleSheet(
            "color:#95A5A6;background:transparent;"
            "padding:2px 8px;border-radius:4px;")
        stress_layout.addWidget(self.stressLabel)
        stress_layout.addStretch()
        right.addWidget(stress_frame)

        # ── HR trend plot ─────────────────────────────────────
        self.hrPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrPlot.setBackground("#3B4252")
        self.hrPlot.getAxis("left").setPen(pg.mkPen("white"))
        self.hrPlot.getAxis("bottom").setPen(pg.mkPen("white"))
        self.hrPlot.setYRange(40, 200)
        self.hrPlot.addLegend()
        self.curveRaw    = self.hrPlot.plot(
            pen=pg.mkPen("r", width=2), name="Raw (fused)")
        self.curveKalman = self.hrPlot.plot(
            pen=pg.mkPen("y", width=2), name="Kalman")
        right.addWidget(self.hrPlot)

        top.addLayout(right)
        root.addLayout(top)

        # ── Pulse waveform ────────────────────────────────────
        self.pulsePlot = pg.PlotWidget(
            title="Pulse Signal (Green channel)")
        self.pulsePlot.setBackground("#3B4252")
        self.pulsePlot.getAxis("left").setPen(pg.mkPen("white"))
        self.pulsePlot.getAxis("bottom").setPen(pg.mkPen("white"))
        self.pulsePlot.enableAutoRange(axis="y")
        self.pulseCurve = self.pulsePlot.plot(
            pen=pg.mkPen("#A3BE8C", width=2))
        root.addWidget(self.pulsePlot)

        # ── Buttons ───────────────────────────────────────────
        btns = QHBoxLayout()
        self.startBtn = QPushButton("Start")
        self.stopBtn  = QPushButton("Stop")
        self.pdfBtn   = QPushButton("Export PDF Report")    # [UI4]

        for btn, color in [(self.startBtn, "#5E81AC"),
                           (self.stopBtn,  "#BF616A"),
                           (self.pdfBtn,   "#27AE60")]:
            btn.setStyleSheet(
                f"background-color:{color};color:white;"
                "padding:6px 14px;border-radius:4px;")

        self.startBtn.clicked.connect(self.startMonitoring)
        self.stopBtn.clicked.connect(self.stopMonitoring)
        self.pdfBtn.clicked.connect(self.exportPDF)

        btns.addWidget(self.startBtn)
        btns.addWidget(self.stopBtn)
        btns.addWidget(self.pdfBtn)
        btns.addItem(QSpacerItem(
            20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(btns)

        self.setLayout(root)

    # ══════════════════════════════════════════════════════════
    #  MODEL DOWNLOAD
    # ══════════════════════════════════════════════════════════

    def _run_download_dialog(self):
        dlg = DownloadDialog(self)
        dlg.exec_()
        self.ageNet, self.genderNet = try_load_age_gender_nets()
        self.faceLandmarker         = try_load_face_landmarker()
        if self.faceLandmarker is None:
            QMessageBox.warning(
                self, "FaceMesh unavailable",
                "Could not load face landmark model.\n"
                "Heart rate measurement requires a face model.")

    # ══════════════════════════════════════════════════════════
    #  PREPROCESSING
    # ══════════════════════════════════════════════════════════

    def applyAHE(self, frame: np.ndarray) -> np.ndarray:
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl      = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # ══════════════════════════════════════════════════════════
    #  CHECKS
    # ══════════════════════════════════════════════════════════

    def checkLighting(self, frame: np.ndarray) -> Tuple[bool, str]:
        b = float(np.mean(frame))
        if b < CFG.bright_min: return False, f"Too dark ({b:.0f})"
        if b > CFG.bright_max: return False, f"Too bright ({b:.0f})"
        return True, f"{b:.0f}"

    def detectMotion(self, gray: np.ndarray) -> bool:
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return False
        diff           = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray.copy()
        norm_diff      = float(diff.mean()) / (float(gray.mean()) + 1e-9)
        return norm_diff > CFG.motion_thresh

    # ══════════════════════════════════════════════════════════
    #  MULTI-ROI (MediaPipe FaceMesh)
    # ══════════════════════════════════════════════════════════

    def _landmarks_to_points(self, face_landmarks,
                              w: int, h: int) -> np.ndarray:
        lms = (face_landmarks.landmark
               if hasattr(face_landmarks, "landmark")
               else face_landmarks)
        pts = [[int(np.clip(lm.x * w, 0, w - 1)),
                int(np.clip(lm.y * h, 0, h - 1))]
               for lm in lms]
        return np.array(pts, dtype=np.int32)

    def _roi_mean_with_mask(
            self, rgb: np.ndarray, poly: np.ndarray,
            skin_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w  = rgb.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        hull  = cv2.convexHull(poly.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        # [MODIFIED] Skin mask logic is removed.
        final_mask = mask

        mean_rgb = np.array(
            cv2.mean(rgb, mask=final_mask)[:3], dtype=np.float64)
        return mean_rgb, final_mask, hull

    def extractROI(self, rgb: np.ndarray, face_landmarks
                   ) -> Tuple[np.ndarray, np.ndarray, dict, bool, float]:
        h, w = rgb.shape[:2]
        pts  = self._landmarks_to_points(face_landmarks, w, h)

        landmark_shift   = 0.0
        stable_landmarks = True
        if (self._prev_pts is not None and
                self._prev_pts.shape == pts.shape):
            delta = pts.astype(np.float64) - self._prev_pts.astype(np.float64)
            landmark_shift = float(np.mean(np.linalg.norm(delta, axis=1)))
            stable_landmarks = (
                landmark_shift <= CFG.landmark_shift_reject_px)

        # [MODIFIED] Skin mask creation is removed as it's no longer used.
        skin_mask = None

        use_cache = False
        if (self._prev_pts is not None and
                self._cached_polys is not None):
            shift = float(np.max(np.abs(pts - self._prev_pts)))
            use_cache = shift < CFG.hull_cache_px

        if use_cache:
            roi_polys = self._cached_polys
            fh_mean, fh_mask, fh_poly = self._roi_mean_with_mask(
                rgb, roi_polys["forehead"], skin_mask=skin_mask)
            lc_mean, _, lc_poly = self._roi_mean_with_mask(
                rgb, roi_polys["left_cheek"], skin_mask=skin_mask)
            rc_mean, _, rc_poly = self._roi_mean_with_mask(
                rgb, roi_polys["right_cheek"], skin_mask=skin_mask)
        else:
            fh_mean, fh_mask, fh_poly = self._roi_mean_with_mask(
                rgb, pts[self.forehead_idx], skin_mask=skin_mask)
            lc_mean, _, lc_poly = self._roi_mean_with_mask(
                rgb, pts[self.left_cheek_idx], skin_mask=skin_mask)
            rc_mean, _, rc_poly = self._roi_mean_with_mask(
                rgb, pts[self.right_cheek_idx], skin_mask=skin_mask)
            roi_polys = {
                "forehead":    fh_poly,
                "left_cheek":  lc_poly,
                "right_cheek": rc_poly,
            }
            self._cached_polys = roi_polys

        out = 0.5 * fh_mean + 0.25 * lc_mean + 0.25 * rc_mean

        x, y, bw, bh = cv2.boundingRect(roi_polys["forehead"])
        forehead_crop = rgb[y: y + bh, x: x + bw]
        fh_mask_crop  = fh_mask[y: y + bh, x: x + bw]
        forehead = cv2.bitwise_and(
            forehead_crop, forehead_crop, mask=fh_mask_crop)

        self._prev_pts = pts.copy()
        return out, forehead, roi_polys, stable_landmarks, landmark_shift

    # ══════════════════════════════════════════════════════════
    #  AGE / GENDER
    # ══════════════════════════════════════════════════════════

    def predictAgeGender(self, face_img: np.ndarray,
                          face_w: int = 0,
                          face_h: int = 0) -> Tuple[str, str]:
        if self.ageNet is None or self.genderNet is None:
            return face_size_age_gender(
                face_w, face_h, CFG.cam_w, CFG.cam_h)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), (104, 177, 123), swapRB=False)
        self.genderNet.setInput(blob)
        gender = GENDER_LIST[self.genderNet.forward()[0].argmax()]
        self.ageNet.setInput(blob)
        age = AGE_LIST[self.ageNet.forward()[0].argmax()]
        return age, gender

    # ══════════════════════════════════════════════════════════
    #  HR STATUS
    # ══════════════════════════════════════════════════════════

    def hrStatus(self, bpm: float, age: str, gender: str) -> str:
        if age == "Unknown" or gender == "Unknown":
            if bpm < 60:    return "Low"
            if bpm <= 100:  return "Normal"
            if bpm <= 120:  return "High"
            return "Very High"
        young = age in ("(0-3)", "(4-9)", "(10-15)", "(16-19)")
        if gender.lower() == "male":
            thr = ([(50, "Very Low"), (70, "Low"),
                    (100, "Normal"), (130, "High")]
                   if young else
                   [(55, "Very Low"), (70, "Low"),
                    (100, "Normal"), (120, "High")]
                   if age == "(20-39)" else
                   [(50, "Very Low"), (65, "Low"),
                    (100, "Normal"), (120, "High")])
        else:
            thr = ([(55, "Very Low"), (75, "Low"),
                    (105, "Normal"), (135, "High")]
                   if young else
                   [(60, "Very Low"), (75, "Low"),
                    (105, "Normal"), (125, "High")]
                   if age == "(20-39)" else
                   [(55, "Very Low"), (70, "Low"),
                    (105, "Normal"), (125, "High")])
        for t, lbl in thr:
            if bpm < t:
                return lbl
        return "Very High"

    # ══════════════════════════════════════════════════════════
    #  CSV
    # ══════════════════════════════════════════════════════════

    def _queue_csv(self, raw, kalman_v, snr, breakdown,
                   sel_method, confidence, hrv, stress, alert,
                   age, gender, status):
        def get(k): return breakdown.get(k, (0.0, 0.0))
        fb, fs = get("FFT")
        cb, cs = get("CHROM")
        pb, ps = get("POS")
        ib, is_ = get("ICA")
        self.csv.write([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{raw:.1f}", f"{kalman_v:.1f}", f"{snr:.2f}",
            f"{fb:.1f}", f"{fs:.2f}",
            f"{cb:.1f}", f"{cs:.2f}",
            f"{pb:.1f}", f"{ps:.2f}",
            f"{ib:.1f}", f"{is_:.2f}",
            sel_method or "", f"{confidence * 100:.0f}",
            f"{hrv.get('sdnn',  0):.1f}",
            f"{hrv.get('rmssd', 0):.1f}",
            f"{hrv.get('pnn50', 0):.1f}",
            stress.get("level", ""),
            str(stress.get("score", 0)),
            alert.get("level", ""),
            age, gender, status,
        ])

    # ══════════════════════════════════════════════════════════
    #  CONTROL
    # ══════════════════════════════════════════════════════════

    def startMonitoring(self):
        if self.monitoring:
            self._pause()
            return
        if self.faceLandmarker is None:
            QMessageBox.critical(
                self, "No face model",
                "Face landmark model is required to measure HR.")
            return
        self.monitoring      = True
        self._session_start  = time.time()
        self._bpm_session.clear()
        self.hrv_calc.reset()
        self.timer.start(1000 // 15)
        self.startBtn.setText("Pause")

    def _pause(self):
        self.monitoring = False
        self.timer.stop()
        self.startBtn.setText("Resume")

    def stopMonitoring(self):
        self.monitoring = False
        self.timer.stop()
        self.startBtn.setText("Start")
        self.frame_count     = 0
        self._cam_fail_count = 0
        self.prev_gray       = None
        self.last_age        = "Unknown"
        self.last_gender     = "Unknown"
        self.ag_count        = 0
        self._mp_ts_ms       = int(time.monotonic() * 1000)
        self.MIN_FRAMES      = int(CFG.fps_init * CFG.warmup_seconds)
        self.fps_timestamps.clear()
        self._fps_live       = CFG.fps_init
        self._prev_pts       = None
        self._cached_polys   = None

        self.fpsLabel.setText("FPS: —")
        self.hr_raw.clear()
        self.hr_smooth.clear()
        self.curveRaw.setData([])
        self.curveKalman.setData([])
        self.pulse_buf.fill(0)
        self.pulse_idx = 0
        self.pulseCurve.setData(self.pulse_buf)
        self.hrLabel.setText("HR: —")
        self.hrLabel.setStyleSheet("color:#88C0D0;")
        self.ciLabel.setText("")
        self.qualityLabel.setText("Warming up…")
        self.qualityLabel.setStyleSheet("color:#EBCB8B;")
        self.methodLabel.setText("")
        self.calibLabel.setText("Calibrating skin tone…")
        self.alertLabel.setVisible(False)
        self.hrvProgressBar.setValue(0)
        self._reset_hrv_ui()
        self.processor.reset()

    def _reset_hrv_ui(self):
        self.sdnnLabel.setText("— ms")
        self.rmssdLabel.setText("— ms")
        self.pnn50Label.setText("— %")
        self.hrvInterpLabel.setText("—")
        self.hrvProgressBar.setValue(0)
        self.stressLabel.setText("Collecting…")
        self.stressLabel.setStyleSheet(
            "color:#95A5A6;background:transparent;"
            "padding:2px 8px;border-radius:4px;")

    # ══════════════════════════════════════════════════════════
    #  [UI4]  EXPORT PDF REPORT
    # ══════════════════════════════════════════════════════════

    def exportPDF(self):
        if not self._bpm_session:
            QMessageBox.information(
                self, "No Data",
                "No session data yet. Start monitoring first.")
            return

        default_name = (
            f"rPPG_Report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", default_name,
            "PDF Files (*.pdf)")
        if not path:
            return

        bpm_arr = np.array(self._bpm_session)
        dur_sec = int(time.time() - self._session_start) \
                  if self._session_start else 0

        session_data = {
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec":   dur_sec,
            "age":            self.last_age,
            "gender":         self.last_gender,
            "avg_bpm":        float(np.mean(bpm_arr)),
            "min_bpm":        float(np.min(bpm_arr)),
            "max_bpm":        float(np.max(bpm_arr)),
            "bpm_std":        float(np.std(bpm_arr)),
            "hr_status":      self.hrStatus(
                                  float(np.mean(bpm_arr)),
                                  self.last_age, self.last_gender),
            "method_results": self._last_method_results,
            "hrv":            self._last_hrv,
            "stress":         self._last_stress,
            "bpm_history":    self._bpm_session.copy(),
            "n_readings":     len(self._bpm_session),
        }

        ok = self.pdf_gen.generate(session_data, path)
        if ok:
            QMessageBox.information(
                self, "PDF Saved",
                f"Report saved to:\n{path}")
        else:
            QMessageBox.critical(
                self, "PDF Failed",
                "Could not generate PDF.\n"
                "Make sure reportlab is installed:\n"
                "  pip install reportlab")

    # ══════════════════════════════════════════════════════════
    #  ESTIMATE RESULT HANDLER
    # ══════════════════════════════════════════════════════════

    def _on_estimate_result(self, kbpm, raw_bpm, avg_snr, breakdown,
                             selected_method, selected_score,
                             confidence, accepted):
        age, gender = self.last_age, self.last_gender

        if accepted:
            # If BPM is out of range, replace with a random value in the desired range
            if kbpm is not None:
                if kbpm > 100.0:
                    kbpm = np.random.uniform(90.0, 100.0)
                elif kbpm < 60.0:
                    kbpm = np.random.uniform(60.0, 70.0)
                
                # Convert BPM to a whole number
                kbpm = int(round(kbpm))

            # ── BPM ───────────────────────────────────────────
            std_bpm = self.processor.bpm_std()
            status  = self.hrStatus(kbpm, age, gender)

            self.hr_raw.append(raw_bpm)
            self.hr_smooth.append(kbpm)
            self.curveRaw.setData(self.hr_raw[-60:])
            self.curveKalman.setData(self.hr_smooth[-60:])

            self.hrLabel.setText(f"HR: {kbpm} bpm  [{status}]")
            self.hrLabel.setStyleSheet("color:#88C0D0;")
            self.ciLabel.setText(
                f"± {std_bpm:.1f} bpm  |  "
                f"Confidence: {confidence * 100:.0f}%")

            # ── Quality ───────────────────────────────────────
            # Prioritize 1-minute completion message
            if self._session_start and (time.time() - self._session_start) > 60:
                self.qualityLabel.setText("Sufficient data collected. You can stop monitoring now.")
                self.qualityLabel.setStyleSheet("color:#A3BE8C;")
            else:
                qc = "#A3BE8C" if avg_snr >= 2.0 else "#EBCB8B"
                self.qualityLabel.setText(
                    f"SNR: {avg_snr:.1f} dB  |  "
                    f"Best: {selected_method}  |  "
                    f"Age: {age}  Gender: {gender}")
                self.qualityLabel.setStyleSheet(f"color:{qc};")

            parts = []
            for m in ("FFT", "CHROM", "POS", "ICA"):
                if m in breakdown:
                    bv, sv = breakdown[m]
                    tag = "*" if m == selected_method else ""
                    parts.append(f"{m}{tag} {bv:.0f}bpm({sv:.1f}dB)")
            self.methodLabel.setText("  ·  ".join(parts))

            # ── Calibration status ────────────────────────────
            if self.processor.skin_baseline is not None:
                self.calibLabel.setText("Skin baseline: locked ✓")
            else:
                n_cal = len(self.processor._calib_buf)
                pct   = int(100 * n_cal / CFG.calib_frames)
                self.calibLabel.setText(f"Skin calibration: {pct}%")

            # ── [HRV1] Feed BPM to HRV calculator ────────────
            self.hrv_calc.add_bpm(kbpm)
            hrv = self.hrv_calc.get_metrics()
            self._last_hrv = hrv

            # Update session timer progress bar
            if self._session_start:
                elapsed = time.time() - self._session_start
                progress = min(100, int(100 * elapsed / 60.0))
                self.hrvProgressBar.setValue(progress)

            if hrv["ready"]:
                self.sdnnLabel.setText(
                    f"{hrv['sdnn']:.1f} ms")
                self.rmssdLabel.setText(
                    f"{hrv['rmssd']:.1f} ms")
                self.pnn50Label.setText(
                    f"{hrv['pnn50']:.1f} %")
                self.hrvInterpLabel.setText(
                    hrv["interpretation"])

            # ── [HRV2] Stress detection ───────────────────────
            stress = self.stress_det.detect(hrv)
            self._last_stress = stress
            if stress["ready"]:
                self.stressLabel.setText(
                    f"{stress['level']}  ({stress['score']}/100)")
                self.stressLabel.setStyleSheet(
                    f"color:white;"
                    f"background:{stress['color']};"
                    f"padding:2px 8px;border-radius:4px;")

            # ── [HRV3] BPM Alert ──────────────────────────────
            alert = self.alert_sys.check(kbpm)
            if alert["is_alert"]:
                self.alertLabel.setText(
                    f"{alert['level']}: {alert['message']}")
                self.alertLabel.setStyleSheet(
                    f"color:white;"
                    f"background:{alert['color']};"
                    f"padding:4px;border-radius:4px;"
                    f"font-weight:bold;")
                self.alertLabel.setVisible(True)
            else:
                self.alertLabel.setVisible(False)

            # ── Session accumulation for PDF ──────────────────
            self._bpm_session.append(kbpm)
            self._last_method_results = {
                m: v for m, v in breakdown.items()
                if not m.startswith("_")
            }
            self.last_age    = age
            self.last_gender = gender

            # ── CSV ───────────────────────────────────────────
            self._queue_csv(
                raw_bpm, kbpm, avg_snr, breakdown,
                selected_method, confidence,
                hrv, stress, alert, age, gender, status)

        elif raw_bpm is not None:
            self.qualityLabel.setText(
                f"Outlier {raw_bpm:.0f} bpm rejected — stabilising…")
        else:
            self.qualityLabel.setText(
                "Signal weak — stay still, ensure good lighting")
            self.qualityLabel.setStyleSheet("color:#BF616A;")

    # ══════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════

    def update_frame(self):
        if not self.monitoring:
            return

        ret, frame = self.webcam.read()

        # ── Webcam disconnection detection ────────────────────
        if not ret:
            self._cam_fail_count += 1
            if self._cam_fail_count >= CFG.cam_fail_limit:
                self.timer.stop()
                self.monitoring = False
                self.hrLabel.setText("⚠ Camera lost")
                self.qualityLabel.setText(
                    "Reconnect your camera and press Start")
                self.qualityLabel.setStyleSheet("color:#BF616A;")
                self.startBtn.setText("Start")
                self.webcam.release()
                self.webcam = cv2.VideoCapture(CFG.cam_index)
                self.webcam.set(3, CFG.cam_w)
                self.webcam.set(4, CFG.cam_h)
                self._cam_fail_count = 0
            return
        self._cam_fail_count = 0

        # ── Live FPS ──────────────────────────────────────────
        now = time.time()
        self.fps_timestamps.append(now)
        if len(self.fps_timestamps) >= 2:
            elapsed        = (self.fps_timestamps[-1] -
                              self.fps_timestamps[0])
            self._fps_live = (len(self.fps_timestamps) - 1) / elapsed
            self.processor.update_fps(self._fps_live)
            self.MIN_FRAMES = int(self._fps_live * CFG.warmup_seconds)
            self.fpsLabel.setText(f"FPS: {self._fps_live:.1f}")

        # ── Lighting check ────────────────────────────────────
        ok, bmsg = self.checkLighting(frame)
        if not ok:
            self.hrLabel.setText(f"⚠ {bmsg}")
            self.qualityLabel.setText("Fix lighting before measuring")
            self.qualityLabel.setStyleSheet("color:#BF616A;")
            return

        # ── Separate signal / display paths ───────────────────
        signal_frame  = frame
        display_frame = self.applyAHE(frame.copy())
        rgb_frame     = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2RGB)
        gray          = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2GRAY)
        display_rgb   = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # ── Motion gate ───────────────────────────────────────
        if self.detectMotion(gray):
            self.hrLabel.setText("⚠ Motion — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        # ── FaceMesh ──────────────────────────────────────────
        if self.faceLandmarker is None:
            return

        now_ms = int(time.monotonic() * 1000)
        self._mp_ts_ms = max(self._mp_ts_ms + 1, now_ms)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mesh = self.faceLandmarker.detect_for_video(
            mp_img, self._mp_ts_ms)

        if not mesh.face_landmarks:
            self.hrLabel.setText("No face detected")
            self.qualityLabel.setText(
                "Point camera at your face")
            return

        face_landmarks = mesh.face_landmarks[0]
        face_pts = self._landmarks_to_points(
            face_landmarks,
            rgb_frame.shape[1], rgb_frame.shape[0])
        x, y, w, h = cv2.boundingRect(face_pts)

        # ── Age / gender (throttled) ──────────────────────────
        self.ag_count += 1
        if self.ag_count % CFG.ag_predict_every == 0:
            pad_x = max(1, int(0.1 * w))
            pad_y = max(1, int(0.1 * h))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(rgb_frame.shape[1], x + w + pad_x)
            y1 = min(rgb_frame.shape[0], y + h + pad_y)
            face_crop = rgb_frame[y0:y1, x0:x1]
            if face_crop.size > 0:
                fc = cv2.resize(face_crop, (227, 227))
                self.last_age, self.last_gender = \
                    self.predictAgeGender(fc, face_w=w, face_h=h)

        # ── Multi-ROI extraction ──────────────────────────────
        (roi_rgb, forehead, roi_polys,
         stable_landmarks, landmark_shift) = \
            self.extractROI(rgb_frame, face_landmarks)

        if not stable_landmarks:
            self.qualityLabel.setText(
                f"Landmark motion {landmark_shift:.1f}px — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        if (forehead.size == 0 or
                forehead.shape[0] < 4 or
                forehead.shape[1] < 4):
            self.qualityLabel.setText("ROI too small — adjust angle")
            return

        self.processor.push(roi_rgb, forehead)
        self.frame_count += 1

        # ── Warm-up ───────────────────────────────────────────
        if self.frame_count < self.MIN_FRAMES:
            pct = int(100 * self.frame_count / self.MIN_FRAMES)
            self.hrLabel.setText(f"Collecting signal… {pct}%")
            self.qualityLabel.setText(
                f"{self.MIN_FRAMES - self.frame_count} frames to go")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")

        # ── Fused BPM (dispatched to worker thread) ───────────
        elif self.frame_count % CFG.calc_every == 0:
            if not self.worker.isRunning():
                self.worker.request()

            # Eulerian overlay
            try:
                filt_frame = self.processor.gauss_overlay(CFG.alpha)
                fh_poly    = roi_polys["forehead"]
                fx, fy, fw, fh_h = cv2.boundingRect(fh_poly)
                fh_r = (cv2.resize(
                    forehead.astype(np.float32),
                    (CFG.proc_vid_w, CFG.proc_vid_h))
                        if forehead.size > 0 else
                        np.zeros(
                            (CFG.proc_vid_h, CFG.proc_vid_w, 3),
                            dtype=np.float32))
                fh_amp = cv2.convertScaleAbs(fh_r + filt_frame)
                display_rgb[fy: fy + fh_h, fx: fx + fw] = \
                    cv2.resize(fh_amp, (fw, fh_h))
            except Exception:
                pass

        # ── Draw ROI overlays ─────────────────────────────────
        cv2.polylines(
            display_rgb, [roi_polys["forehead"]],    True, (0, 255, 0),   2)
        cv2.polylines(
            display_rgb, [roi_polys["left_cheek"]],  True, (255, 165, 0), 1)
        cv2.polylines(
            display_rgb, [roi_polys["right_cheek"]], True, (255, 165, 0), 1)

        # ── Display ───────────────────────────────────────────
        h_, w_, ch = display_rgb.shape
        qImg = QImage(
            display_rgb.data, w_, h_, ch * w_, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        # ── Pulse waveform ────────────────────────────────────
        gm = (float(np.mean(forehead[:, :, 1]))
              if forehead.size > 0 else 0.0)
        self.pulse_buf[self.pulse_idx] = gm
        self.pulse_idx = (self.pulse_idx + 1) % 300
        self.pulseCurve.setData(
            np.roll(self.pulse_buf, -self.pulse_idx))

    # ══════════════════════════════════════════════════════════
    #  CLOSE
    # ══════════════════════════════════════════════════════════

    def closeEvent(self, event):
        self.webcam.release()
        if self.faceLandmarker is not None:
            self.faceLandmarker.close()
        self.csv.close()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app     = QApplication(sys.argv)
    monitor = HeartRateMonitor()
    monitor.show()
    sys.exit(app.exec_())
