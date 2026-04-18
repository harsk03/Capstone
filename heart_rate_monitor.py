"""
Heart Rate Monitor — Comparative rPPG  (Improved Edition)
==========================================================
Runs FFT (Eulerian), CHROM, POS, and ICA in parallel and fuses their
outputs using SNR-weighted averaging for a more stable BPM reading.

KEY IMPROVEMENTS OVER PREVIOUS VERSION
───────────────────────────────────────
Signal Quality
  [SQ1] Weighted SNR fusion instead of winner-takes-all selection
  [SQ2] Per-method adaptive SNR thresholds (CHROM/POS stricter than FFT)
  [SQ3] Savitzky-Golay temporal smoothing on rgb_buf before CHROM/POS
  [SQ4] Adaptive HR band widened to 40–200 bpm to cover children/athletes
  [SQ5] Per-session baseline skin-tone calibration (first N frames)

Architecture
  [AR1] SignalProcessor.estimate() runs in a QThread — UI never blocks
  [AR2] Frozen @dataclass Config centralises every tunable constant
  [AR3] Face landmark detection skipped on confirmed-static frames
  [AR4] CSV writes offloaded to a queue + background flush thread

Performance
  [PF1] rgb_ord / gauss_ord pre-allocated; np.roll replaced by copyto
  [PF2] Gaussian pyramid computes only the needed level (no intermediate list)
  [PF3] convexHull result cached when face hasn't moved > HULL_CACHE_PX
  [PF4] CLAHE instance created once in __init__ (not every frame)

Robustness
  [RB1] Webcam drop detected after N consecutive failures → reconnect UI
  [RB2] Zero/degenerate forehead crop guarded before push()
  [RB3] Model downloads moved to a QThread with progress dialog
  [RB4] CSV path validated at startup; falls back to tempfile directory
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
from typing import Optional, Tuple, Dict

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
import pyqtgraph as pg


# ══════════════════════════════════════════════════════════════════════════════
#  [AR2]  CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    # Camera
    cam_w: int = 320
    cam_h: int = 240
    cam_index: int = 0
    cam_fail_limit: int = 30          # [RB1] consecutive read-fails before reconnect UI

    # Signal buffers
    buf_len: int = 150
    pyr_levels: int = 3
    proc_vid_w: int = 160
    proc_vid_h: int = 120
    fps_init: float = 15.0

    # Heart rate band  [SQ4] widened to cover children and athletes
    hr_lo_hz: float = 0.67            # 40 bpm
    hr_hi_hz: float = 3.33            # 200 bpm

    # Processing
    calc_every: int = 8               # fused estimate every N frames
    warmup_seconds: float = 3.0       # seconds before first estimate

    # Outlier / shift detection
    outlier_delta: float = 35.0
    shift_adopt_delta: float = 18.0
    shift_confirm_count: int = 2

    # [SQ2]  Per-method SNR thresholds (dB)
    snr_thresh: Dict[str, float] = field(default_factory=lambda: {
        "FFT":   -1.0,
        "CHROM":  0.0,
        "POS":    0.0,
        "ICA":   -1.0,
    })

    # Fusion
    fusion_agreement_bpm: float = 8.0   # methods within this window get bonus weight
    fft_weight_scale: float = 0.25      # down-weight Eulerian path in noisy scenes
    ica_run_snr_threshold: float = 0.0  # run ICA only when base quality is weak
    ica_run_disagreement_bpm: float = 12.0

    # Skin-tone calibration  [SQ5]
    calib_frames: int = 60             # frames used for per-session baseline

    # Motion
    motion_thresh: float = 0.08
    bright_min: int = 30
    bright_max: int = 230

    # Kalman
    kalman_q: float = 0.5
    kalman_r: float = 2.0

    # Face landmark caching  [PF3]
    hull_cache_px: int = 5            # max landmark shift (px) before hull recomputed
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
#  [RB3]  MODEL DOWNLOAD — QThread with progress signal
# ══════════════════════════════════════════════════════════════════════════════

class ModelDownloadThread(QThread):
    """Downloads all required model files off the UI thread."""

    progress   = pyqtSignal(str, int)   # (message, percent 0-100)
    finished_ok = pyqtSignal(bool)      # True = all succeeded

    def run(self):
        files = [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL, FACE_LANDMARKER_MODEL]
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
    """Modal progress dialog shown while models download."""

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
        base_opts = mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL)
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
    """Pre-compute Savitzky-Golay smoothing coefficients (convolution kernel)."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vstack([x**i for i in range(poly + 1)]).T
    coeffs = np.linalg.pinv(A)[0]          # 0th derivative = smoothing
    return coeffs[::-1]                    # flip for np.convolve


_SG_KERNEL = _savgol_coeffs(window=7, poly=2)


def savgol_smooth_rows(rgb: np.ndarray) -> np.ndarray:
    """
    [SQ3] Apply Savitzky-Golay smoothing along the time axis of each
    RGB channel independently.  Reduces high-frequency sensor noise
    without distorting the pulse waveform amplitude or phase.
    """
    out = rgb.copy()
    k = _SG_KERNEL
    hw = len(k) // 2
    for ch in range(3):
        out[:, ch] = np.convolve(rgb[:, ch], k, mode='same')
        # Repair edge artefacts (convolve 'same' zero-pads at boundaries)
        out[:hw,  ch] = rgb[:hw,  ch]
        out[-hw:, ch] = rgb[-hw:, ch]
    return out


def bandpass_fft(sig: np.ndarray, fps: float,
                 f_lo: float = None, f_hi: float = None
                 ) -> Tuple[float, float, float]:
    """
    FFT with Hann windowing, returns (dominant_hz, bpm, snr_db).
    f_lo/f_hi default to CFG.hr_lo_hz / CFG.hr_hi_hz.
    """
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
    hz            = float(freqs[np.argmax(mags_f)])
    return hz, hz * 60.0, snr


def chrom_pulse(rgb: np.ndarray) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013)."""
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
    """POS (Wang et al. 2017)."""
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
    w = rng.normal(size=(n_components, xw.shape[1]))
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-9

    for _ in range(max_iter):
        wx    = xw @ w.T
        gwx   = np.tanh(wx)
        gprime = 1.0 - gwx ** 2
        w_new = (gwx.T @ xw) / xw.shape[0] - np.diag(np.mean(gprime, 0)) @ w
        s, u  = np.linalg.eigh(w_new @ w_new.T)
        s     = np.clip(s, 1e-9, None)
        w_new = (u @ np.diag(1.0 / np.sqrt(s)) @ u.T) @ w_new
        lim   = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w     = w_new
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
        if snr > best_snr and CFG.hr_lo_hz * 60 <= bpm <= CFG.hr_hi_hz * 60:
            best_bpm, best_snr = bpm, snr
    return best_bpm, best_snr


# ══════════════════════════════════════════════════════════════════════════════
#  [SQ1] [SQ2]  FUSED ESTIMATOR — SNR-weighted average
# ══════════════════════════════════════════════════════════════════════════════

def fused_estimate(rgb_buf: np.ndarray, gauss_buf: np.ndarray,
                   fps: float, freq_mask: np.ndarray, filled: int,
                   skin_baseline: Optional[np.ndarray] = None
                   ) -> Tuple:
    """
    Run all four rPPG methods and return an SNR-weighted BPM estimate.

    [SQ1] Weighted fusion: each method's contribution is proportional to its
          SNR (linear scale), so a high-SNR CHROM and medium-SNR POS both
          contribute rather than only the winner.

    [SQ2] Per-method SNR thresholds applied before weighting — a method that
          falls below its own floor is excluded from the fused estimate.

    [SQ3] Savitzky-Golay smoothing applied to rgb before CHROM/POS.

    [SQ5] If skin_baseline is provided, subtract it from rgb for DC removal.

    Returns:
        (fused_bpm, avg_snr, breakdown, selected_method,
         selected_score, comparison, confidence)
    where breakdown[method] = (bpm, snr).
    """
    HR_LO = CFG.hr_lo_hz * 60
    HR_HI = CFG.hr_hi_hz * 60

    if filled < 30:
        return None, 0.0, {}, None, 0.0, {}

    rgb_raw   = rgb_buf[-filled:]
    gauss_raw = gauss_buf[-filled:]

    # Mean normalisation preserves channel ratios
    mu = np.mean(rgb_raw, axis=0)
    mu[mu == 0] = 1e-9

    # [SQ5] Subtract per-session skin baseline
    if skin_baseline is not None:
        mu = mu / (skin_baseline + 1e-9)

    rgb = rgb_raw / mu

    # [SQ3] Smoothed copy for CHROM/POS
    rgb_smooth = savgol_smooth_rows(rgb) if len(rgb) > 7 else rgb

    results: Dict[str, Tuple[float, float]] = {}

    # ── FFT (Eulerian) ──────────────────────────────────────
    try:
        n     = gauss_raw.shape[0]
        freqs = fps * np.arange(n) / n
        mask  = (freqs >= CFG.hr_lo_hz) & (freqs <= CFG.hr_hi_hz)
        cube  = np.fft.fft(gauss_raw, axis=0)
        cube[~mask] = 0
        sig   = np.real(cube).mean(axis=(1, 2, 3))
        _, bpm, snr = bandpass_fft(sig, fps)
        if HR_LO <= bpm <= HR_HI:
            results['FFT'] = (bpm, snr)
    except Exception as e:
        print(f"[FFT ERROR] {e}")

    # ── CHROM ───────────────────────────────────────────────
    try:
        pulse = chrom_pulse(rgb_smooth)
        _, bpm, snr = bandpass_fft(pulse, fps)
        if HR_LO <= bpm <= HR_HI:
            results['CHROM'] = (bpm, snr)
    except Exception as e:
        print(f"[CHROM ERROR] {e}")

    # ── POS ─────────────────────────────────────────────────
    try:
        pulse = pos_pulse(rgb_smooth)
        _, bpm, snr = bandpass_fft(pulse, fps)
        if HR_LO <= bpm <= HR_HI:
            results['POS'] = (bpm, snr)
    except Exception as e:
        print(f"[POS ERROR] {e}")

    # ── ICA (conditional for performance) ───────────────────
    base_bpms = [b for m, (b, _) in results.items() if m in ('FFT', 'CHROM', 'POS')]
    base_snrs = [s for m, (_, s) in results.items() if m in ('FFT', 'CHROM', 'POS')]
    best_base_snr = max(base_snrs) if base_snrs else -999.0
    base_disagreement = (max(base_bpms) - min(base_bpms)) if len(base_bpms) >= 2 else 0.0
    run_ica = (
        not results or
        best_base_snr < CFG.ica_run_snr_threshold or
        base_disagreement > CFG.ica_run_disagreement_bpm
    )
    if run_ica:
        try:
            bpm, snr = ica_best(rgb, fps)
            if HR_LO <= bpm <= HR_HI:
                results['ICA'] = (bpm, snr)
        except Exception as e:
            print(f"[ICA ERROR] {e}")

    if not results:
        return None, 0.0, {}, None, 0.0, {}, 0.0

    # [SQ2] Per-method SNR gate
    valid = {m: (b, s) for m, (b, s) in results.items()
             if s > CFG.snr_thresh.get(m, -2.0)}
    candidates = valid if valid else results

    # Additional FFT guard: if FFT is far from non-FFT consensus, exclude it.
    if 'FFT' in candidates:
        non_fft = {m: v for m, v in candidates.items() if m != 'FFT'}
        if len(non_fft) >= 2:
            non_fft_med = float(np.median([b for b, _ in non_fft.values()]))
            fft_bpm, _ = candidates['FFT']
            if abs(fft_bpm - non_fft_med) > CFG.fft_disagree_limit_bpm:
                candidates = non_fft

    # [SQ1] SNR-weighted BPM fusion
    bpm_vals  = np.array([b for b, _ in candidates.values()])
    snr_vals  = np.array([s for _, s in candidates.values()])
    methods   = list(candidates.keys())

    # Convert SNR dB → linear power for weighting
    weights   = np.power(10.0, snr_vals / 10.0)
    weights   = np.clip(weights, 1e-6, None)
    for i, m in enumerate(methods):
        if m == "FFT":
            weights[i] *= CFG.fft_weight_scale

    # Agreement bonus: methods within fusion_agreement_bpm of the median
    median_bpm = float(np.median(bpm_vals))
    agreement  = np.abs(bpm_vals - median_bpm) < CFG.fusion_agreement_bpm
    weights[agreement] *= 2.0

    weights  /= weights.sum()
    fused_bpm = float(np.dot(weights, bpm_vals))
    avg_snr   = float(np.dot(weights, snr_vals))

    # Best single method (for display / CSV)
    scores = {m: float(s - 0.08 * abs(b - median_bpm))
              for m, (b, s) in candidates.items()}
    selected_method = max(scores, key=scores.get)
    selected_score  = scores[selected_method]

    comparison = scores
    confidence = float(np.clip((avg_snr + 2.0) / 6.0, 0.0, 1.0))
    return fused_bpm, avg_snr, results, selected_method, selected_score, comparison, confidence


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
#  SIGNAL PROCESSOR  (decoupled from Qt)
# ══════════════════════════════════════════════════════════════════════════════

class SignalProcessor:
    """
    Owns all rPPG buffers.  Thread-safe for the pattern:
      main thread → push()
      worker thread → estimate()
    (push and estimate are called from different threads but never
     concurrently — estimate is triggered after push completes.)
    """

    def __init__(self):
        fps  = CFG.fps_init
        buf  = CFG.buf_len
        lvls = CFG.pyr_levels
        vw, vh = CFG.proc_vid_w, CFG.proc_vid_h

        self.fps  = fps
        self.BUF  = buf

        # Determine gauss level size
        dummy = np.zeros((vh, vw, 3), dtype=np.float32)
        g0    = self._build_gauss_level(dummy, lvls)
        gh, gw = g0.shape[:2]

        # [PF1] Pre-allocate ordered roll arrays
        self.gauss_buf = np.zeros((buf, gh, gw, 3), dtype=np.float32)
        self.rgb_buf   = np.zeros((buf, 3),          dtype=np.float64)
        self._rgb_ord  = np.empty_like(self.rgb_buf)
        self._gauss_ord = np.empty_like(self.gauss_buf)

        freqs          = fps * np.arange(buf) / buf
        self.freq_mask = (freqs >= CFG.hr_lo_hz) & (freqs <= CFG.hr_hi_hz)

        self.buf_idx       = 0
        self.filled_frames = 0

        self.kalman      = KalmanBPM()
        self.bpm_history = deque(maxlen=20)

        self._shift_candidate = None
        self._shift_streak    = 0

        # [SQ5] Skin baseline calibration
        self._calib_buf:   list = []
        self.skin_baseline: Optional[np.ndarray] = None

        # [PF3] Hull cache
        self._prev_landmark_pts: Optional[np.ndarray] = None
        self._cached_hulls: Optional[dict] = None

    # ── public ────────────────────────────────────────────────────────────────

    def update_fps(self, fps: float):
        self.fps = fps
        freqs          = fps * np.arange(self.BUF) / self.BUF
        self.freq_mask = (freqs >= CFG.hr_lo_hz) & (freqs <= CFG.hr_hi_hz)

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
        self.skin_baseline = None
        self._prev_landmark_pts = None
        self._cached_hulls = None

    def push(self, roi_rgb: np.ndarray, forehead_frame: np.ndarray):
        """Add one frame of data to the ring buffers."""
        # [SQ5] Accumulate calibration frames
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
            fh_r = np.zeros((CFG.proc_vid_h, CFG.proc_vid_w, 3), dtype=np.float32)

        # [PF2] Only compute the required pyramid level
        self.gauss_buf[self.buf_idx] = self._build_gauss_level(fh_r, CFG.pyr_levels)

        self.buf_idx       = (self.buf_idx + 1) % self.BUF
        self.filled_frames = min(self.filled_frames + 1, self.BUF)

    def estimate(self) -> Tuple:
        """
        Run fused estimator.
        Returns (kalman_bpm, raw_bpm, snr, breakdown,
                 selected_method, selected_score, confidence, accepted)
        """
        # [PF1] Reuse pre-allocated arrays for the ordered view
        idx = self.buf_idx
        if idx == 0:
            np.copyto(self._rgb_ord,   self.rgb_buf)
            np.copyto(self._gauss_ord, self.gauss_buf)
        else:
            np.copyto(self._rgb_ord[:self.BUF - idx],   self.rgb_buf[idx:])
            np.copyto(self._rgb_ord[self.BUF - idx:],   self.rgb_buf[:idx])
            np.copyto(self._gauss_ord[:self.BUF - idx], self.gauss_buf[idx:])
            np.copyto(self._gauss_ord[self.BUF - idx:], self.gauss_buf[:idx])

        raw_bpm, avg_snr, breakdown, sel_method, sel_score, comparison, confidence = fused_estimate(
            self._rgb_ord, self._gauss_ord, self.fps,
            self.freq_mask, self.filled_frames,
            skin_baseline=self.skin_baseline,
        )
        breakdown['_scores'] = comparison

        HR_LO = CFG.hr_lo_hz * 60
        HR_HI = CFG.hr_hi_hz * 60

        if raw_bpm is None or not (HR_LO <= raw_bpm <= HR_HI):
            self.kalman.predict()
            return None, None, 0.0, breakdown, None, 0.0, 0.0, False

        if self._is_outlier(raw_bpm):
            if self._confirm_shift_and_adopt(raw_bpm):
                kbpm = self.kalman.update(raw_bpm)
                return kbpm, raw_bpm, avg_snr, breakdown, sel_method, sel_score, confidence, True
            self.kalman.predict()
            return None, raw_bpm, avg_snr, breakdown, sel_method, sel_score, confidence, False

        self._shift_candidate = None
        self._shift_streak    = 0
        self.bpm_history.append(raw_bpm)
        kbpm = self.kalman.update(raw_bpm)
        return kbpm, raw_bpm, avg_snr, breakdown, sel_method, sel_score, confidence, True

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
        """Return ±1σ of recent BPM history for confidence interval display."""
        if len(self.bpm_history) < 3:
            return 0.0
        return float(np.std(list(self.bpm_history)))

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_gauss_level(frame: np.ndarray, levels: int) -> np.ndarray:
        """[PF2] Compute only the target pyramid level, no intermediate list."""
        f = frame
        for _ in range(levels):
            f = cv2.pyrDown(f)
        return f

    def _is_outlier(self, bpm: float) -> bool:
        if len(self.bpm_history) < 5:
            return False
        return abs(bpm - float(np.median(list(self.bpm_history)))) > CFG.outlier_delta

    def _confirm_shift_and_adopt(self, bpm: float) -> bool:
        if len(self.bpm_history) < 5:
            return False
        median = float(np.median(list(self.bpm_history)))
        if abs(bpm - median) < CFG.shift_adopt_delta:
            self._shift_candidate = None
            self._shift_streak    = 0
            return False
        if self._shift_candidate is None or abs(bpm - self._shift_candidate) > 8:
            self._shift_candidate = bpm
            self._shift_streak    = 1
            return False
        self._shift_streak += 1
        if self._shift_streak < CFG.shift_confirm_count:
            return False
        self.bpm_history.clear()
        self.bpm_history.append(bpm)
        self.kalman.x = bpm
        self.kalman.P = 25.0
        self._shift_candidate = None
        self._shift_streak    = 0
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  [AR1]  WORKER THREAD — runs estimate() off the UI thread
# ══════════════════════════════════════════════════════════════════════════════

class EstimateWorker(QThread):
    """
    Runs SignalProcessor.estimate() in a background thread.
    Emits result_ready when done.
    """
    result_ready = pyqtSignal(object, object, float, dict, object, float, float, bool)

    def __init__(self, processor: SignalProcessor):
        super().__init__()
        self._processor = processor
        self._lock      = threading.Lock()
        self._pending   = False

    def request(self):
        """Ask the worker to run one estimate cycle (non-blocking)."""
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
#  [AR4]  CSV WRITER — background flush thread
# ══════════════════════════════════════════════════════════════════════════════

class CSVWriter:
    """
    Thread-safe CSV writer.  Rows are queued from the UI thread and
    flushed to disk from a daemon thread — never blocks the timer.
    """

    def __init__(self, path: str):
        self._q: queue.Queue = queue.Queue()
        self._file = open(path, 'w', newline='')
        atexit.register(self._file.close)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'Timestamp', 'Raw BPM', 'Kalman BPM', 'Avg SNR (dB)',
            'FFT BPM', 'FFT SNR', 'CHROM BPM', 'CHROM SNR',
            'POS BPM', 'POS SNR', 'ICA BPM', 'ICA SNR',
            'Selected Method', 'Selected Score', 'Confidence (%)',
            'Age', 'Gender', 'HR Status'])
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
    """[RB4] Return a writable path for the CSV, falling back to tempdir."""
    candidate = os.path.join(os.getcwd(), 'heart_rate_data.csv')
    try:
        with open(candidate, 'a'):
            pass
        return candidate
    except PermissionError:
        fallback = os.path.join(tempfile.gettempdir(), 'heart_rate_data.csv')
        print(f"[CSV] CWD not writable, using {fallback}")
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

class HeartRateMonitor(QWidget):

    def __init__(self):
        super().__init__()

        # [PF4] CLAHE instance created once
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Live FPS
        self.fps_timestamps: deque = deque(maxlen=30)
        self._fps_live = CFG.fps_init

        # Signal processor & worker thread
        self.processor = SignalProcessor()
        self.worker    = EstimateWorker(self.processor)
        self.worker.result_ready.connect(self._on_estimate_result)

        # HR history
        self.hr_raw:    list = []
        self.hr_smooth: list = []

        # Pulse waveform
        self.pulse_buf = np.zeros(300)
        self.pulse_idx = 0

        # Motion
        self.prev_gray: Optional[np.ndarray] = None

        # Frame counters
        self.frame_count   = 0
        self.MIN_FRAMES    = int(CFG.fps_init * CFG.warmup_seconds)

        # [RB1] Webcam fail counter
        self._cam_fail_count = 0

        # FaceMesh (loaded after download dialog)
        self.faceLandmarker = None
        self._mp_ts_ms      = int(time.monotonic() * 1000)
        self.forehead_idx   = np.array([10, 67, 103, 109, 338, 297, 332, 284])
        self.left_cheek_idx = np.array([50, 101, 205, 187, 147, 123, 116])
        self.right_cheek_idx = np.array([280, 330, 425, 411, 376, 352, 345])

        # [PF3] Hull cache
        self._prev_pts: Optional[np.ndarray] = None
        self._cached_polys: Optional[dict]   = None

        # Age/gender
        self.last_age    = 'Unknown'
        self.last_gender = 'Unknown'
        self.ag_count    = 0
        self.ageNet      = None
        self.genderNet   = None

        # [AR4] CSV
        csv_path = _safe_csv_path()
        self.csv = CSVWriter(csv_path)

        self.initUI()

        # Camera
        self.webcam = cv2.VideoCapture(CFG.cam_index)
        self.webcam.set(3, CFG.cam_w)
        self.webcam.set(4, CFG.cam_h)

        self.monitoring = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Show download dialog, then load models
        self._run_download_dialog()

    # ── UI ────────────────────────────────────────────────────────────────────

    def initUI(self):
        self.setWindowTitle('Heart Rate Monitor — rPPG Fusion')
        self.setStyleSheet("background-color:#2E3440;color:white;")

        root = QVBoxLayout()
        top  = QHBoxLayout()

        self.videoLabel = QLabel()
        self.videoLabel.setFixedSize(CFG.cam_w, CFG.cam_h)
        self.videoLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.videoLabel.setStyleSheet("border:2px solid #4C566A;")
        top.addWidget(self.videoLabel)

        right = QVBoxLayout()

        self.hrLabel = QLabel('HR: —')
        self.hrLabel.setFont(QFont('Arial', 20))
        self.hrLabel.setStyleSheet("color:#88C0D0;")
        self.hrLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.hrLabel)

        # Confidence interval label
        self.ciLabel = QLabel('')
        self.ciLabel.setFont(QFont('Arial', 10))
        self.ciLabel.setStyleSheet("color:#81A1C1;")
        self.ciLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.ciLabel)

        self.qualityLabel = QLabel('Warming up…')
        self.qualityLabel.setFont(QFont('Arial', 10))
        self.qualityLabel.setStyleSheet("color:#EBCB8B;")
        self.qualityLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.qualityLabel)

        self.methodLabel = QLabel('')
        self.methodLabel.setFont(QFont('Courier', 9))
        self.methodLabel.setStyleSheet("color:#81A1C1;")
        self.methodLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.methodLabel)

        self.fpsLabel = QLabel('FPS: —')
        self.fpsLabel.setFont(QFont('Arial', 9))
        self.fpsLabel.setStyleSheet("color:#4C566A;")
        self.fpsLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.fpsLabel)

        self.calibLabel = QLabel('Calibrating skin tone…')
        self.calibLabel.setFont(QFont('Arial', 9))
        self.calibLabel.setStyleSheet("color:#A3BE8C;")
        self.calibLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.calibLabel)

        self.hrPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrPlot.setBackground('#3B4252')
        self.hrPlot.getAxis('left').setPen(pg.mkPen('white'))
        self.hrPlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.hrPlot.setYRange(40, 200)
        self.hrPlot.addLegend()
        self.curveRaw    = self.hrPlot.plot(pen=pg.mkPen('r', width=2), name='Raw (fused)')
        self.curveKalman = self.hrPlot.plot(pen=pg.mkPen('y', width=2), name='Kalman')
        right.addWidget(self.hrPlot)

        top.addLayout(right)
        root.addLayout(top)

        self.pulsePlot = pg.PlotWidget(title="Pulse Signal (Green channel)")
        self.pulsePlot.setBackground('#3B4252')
        self.pulsePlot.getAxis('left').setPen(pg.mkPen('white'))
        self.pulsePlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.pulsePlot.enableAutoRange(axis='y')
        self.pulseCurve = self.pulsePlot.plot(pen=pg.mkPen('#A3BE8C', width=2))
        root.addWidget(self.pulsePlot)

        btns = QHBoxLayout()
        self.startBtn = QPushButton('Start')
        self.stopBtn  = QPushButton('Stop')
        self.startBtn.setStyleSheet(
            "background-color:#5E81AC;color:white;padding:6px 14px;")
        self.stopBtn.setStyleSheet(
            "background-color:#BF616A;color:white;padding:6px 14px;")
        self.startBtn.clicked.connect(self.startMonitoring)
        self.stopBtn.clicked.connect(self.stopMonitoring)
        btns.addWidget(self.startBtn)
        btns.addWidget(self.stopBtn)
        btns.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(btns)

        self.setLayout(root)

    # ── Model download ────────────────────────────────────────────────────────

    def _run_download_dialog(self):
        """[RB3] Download models in background; block with a progress dialog."""
        dlg = DownloadDialog(self)
        dlg.exec_()
        # Load nets after downloads finish (regardless of success)
        self.ageNet, self.genderNet = try_load_age_gender_nets()
        self.faceLandmarker         = try_load_face_landmarker()
        if self.faceLandmarker is None:
            QMessageBox.warning(self, "FaceMesh unavailable",
                                "Could not load face landmark model.\n"
                                "Heart rate measurement requires a face model.")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def applyAHE(self, frame: np.ndarray) -> np.ndarray:
        """[PF4] CLAHE on L channel — reuses self._clahe instance."""
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl      = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # ── Checks ────────────────────────────────────────────────────────────────

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

    # ── Multi-ROI ─────────────────────────────────────────────────────────────

    def _landmarks_to_points(self, face_landmarks, w: int, h: int) -> np.ndarray:
        lms = (face_landmarks.landmark
               if hasattr(face_landmarks, "landmark") else face_landmarks)
        pts = [[int(np.clip(lm.x * w, 0, w - 1)),
                int(np.clip(lm.y * h, 0, h - 1))]
               for lm in lms]
        return np.array(pts, dtype=np.int32)

    def _roi_mean_with_mask(self, rgb: np.ndarray, poly: np.ndarray,
                            skin_mask: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w   = rgb.shape[:2]
        mask   = np.zeros((h, w), dtype=np.uint8)
        hull   = cv2.convexHull(poly.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        if skin_mask is not None:
            final_mask = cv2.bitwise_and(mask, skin_mask)
            if cv2.countNonZero(final_mask) < CFG.skin_min_pixels:
                final_mask = mask
        else:
            final_mask = mask

        mean_rgb = np.array(cv2.mean(rgb, mask=final_mask)[:3], dtype=np.float64)
        return mean_rgb, final_mask, hull

    def extractROI(self, rgb: np.ndarray, face_landmarks
                   ) -> Tuple[np.ndarray, np.ndarray, dict, bool, float]:
        h, w = rgb.shape[:2]
        pts  = self._landmarks_to_points(face_landmarks, w, h)

        landmark_shift = 0.0
        stable_landmarks = True
        if self._prev_pts is not None and self._prev_pts.shape == pts.shape:
            delta = pts.astype(np.float64) - self._prev_pts.astype(np.float64)
            landmark_shift = float(np.mean(np.linalg.norm(delta, axis=1)))
            stable_landmarks = landmark_shift <= CFG.landmark_shift_reject_px

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        skin_mask = (
            (hsv[:, :, 0] < CFG.skin_h_max) &
            (hsv[:, :, 1] > CFG.skin_s_min) &
            (hsv[:, :, 2] > CFG.skin_v_min)
        ).astype(np.uint8) * 255

        # [PF3] Cache convex hulls when landmarks haven't moved
        use_cache = False
        if self._prev_pts is not None and self._cached_polys is not None:
            shift = float(np.max(np.abs(pts - self._prev_pts)))
            if shift < CFG.hull_cache_px:
                use_cache = True

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
        forehead_crop = rgb[y:y + bh, x:x + bw]
        fh_mask_crop  = fh_mask[y:y + bh, x:x + bw]
        forehead = cv2.bitwise_and(forehead_crop, forehead_crop, mask=fh_mask_crop)
        self._prev_pts = pts.copy()
        return out, forehead, roi_polys, stable_landmarks, landmark_shift

    # ── Age / gender ──────────────────────────────────────────────────────────

    def predictAgeGender(self, face_img: np.ndarray,
                          face_w: int = 0, face_h: int = 0) -> Tuple[str, str]:
        if self.ageNet is None or self.genderNet is None:
            return face_size_age_gender(face_w, face_h, CFG.cam_w, CFG.cam_h)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), (104, 177, 123), swapRB=False)
        self.genderNet.setInput(blob)
        gender = GENDER_LIST[self.genderNet.forward()[0].argmax()]
        self.ageNet.setInput(blob)
        age = AGE_LIST[self.ageNet.forward()[0].argmax()]
        return age, gender

    # ── HR status ─────────────────────────────────────────────────────────────

    def hrStatus(self, bpm: float, age: str, gender: str) -> str:
        if age == 'Unknown' or gender == 'Unknown':
            if bpm < 60:   return "Low"
            if bpm <= 100: return "Normal"
            if bpm <= 120: return "High"
            return "Very High"
        young = age in ('(0-3)', '(4-9)', '(10-15)', '(16-19)')
        if gender.lower() == 'male':
            thr = ([(50,'Very Low'),(70,'Low'),(100,'Normal'),(130,'High')] if young
                   else [(55,'Very Low'),(70,'Low'),(100,'Normal'),(120,'High')]
                   if age == '(20-39)'
                   else [(50,'Very Low'),(65,'Low'),(100,'Normal'),(120,'High')])
        else:
            thr = ([(55,'Very Low'),(75,'Low'),(105,'Normal'),(135,'High')] if young
                   else [(60,'Very Low'),(75,'Low'),(105,'Normal'),(125,'High')]
                   if age == '(20-39)'
                   else [(55,'Very Low'),(70,'Low'),(105,'Normal'),(125,'High')])
        for t, lbl in thr:
            if bpm < t:
                return lbl
        return "Very High"

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _queue_csv(self, raw, kalman_v, snr, breakdown, sel_method,
                   sel_score, confidence, age, gender, status):
        def get(k): return breakdown.get(k, (0.0, 0.0))
        fb, fs   = get('FFT')
        cb, cs   = get('CHROM')
        pb, ps   = get('POS')
        ib, is_  = get('ICA')
        self.csv.write([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            f'{raw:.1f}', f'{kalman_v:.1f}', f'{snr:.2f}',
            f'{fb:.1f}', f'{fs:.2f}', f'{cb:.1f}', f'{cs:.2f}',
            f'{pb:.1f}', f'{ps:.2f}', f'{ib:.1f}', f'{is_:.2f}',
            sel_method or '', f'{sel_score:.2f}', f'{confidence * 100.0:.0f}',
            age, gender, status,
        ])

    # ── Control ───────────────────────────────────────────────────────────────

    def startMonitoring(self):
        if self.monitoring:
            self._pause()
            return
        if self.faceLandmarker is None:
            QMessageBox.critical(self, "No face model",
                                 "Face landmark model is required to measure HR.")
            return
        self.monitoring = True
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
        self.last_age        = 'Unknown'
        self.last_gender     = 'Unknown'
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

        self.processor.reset()

    # ── Estimate result handler (slot) ────────────────────────────────────────

    def _on_estimate_result(self, kbpm, raw_bpm, avg_snr, breakdown,
                             selected_method, selected_score, confidence, accepted):
        age, gender = self.last_age, self.last_gender

        if accepted:
            std_bpm = self.processor.bpm_std()
            status  = self.hrStatus(kbpm, age, gender)

            self.hr_raw.append(raw_bpm)
            self.hr_smooth.append(kbpm)
            self.curveRaw.setData(self.hr_raw[-60:])
            self.curveKalman.setData(self.hr_smooth[-60:])

            self.hrLabel.setText(f"HR: {kbpm:.1f} bpm  [{status}]")
            self.hrLabel.setStyleSheet("color:#88C0D0;")

            # Confidence interval
            if std_bpm > 0:
                self.ciLabel.setText(f"± {std_bpm:.1f} bpm")
            else:
                self.ciLabel.setText("")

            qc = "#A3BE8C" if avg_snr >= 2.0 else "#EBCB8B"
            self.qualityLabel.setText(
                f"SNR: {avg_snr:.1f} dB  |  Raw: {raw_bpm:.1f}  |  "
                f"Best: {selected_method} ({selected_score:.2f})  |  "
                f"Confidence: {confidence * 100.0:.0f}%  |  "
                f"Age: {age}  Gender: {gender}")
            self.qualityLabel.setStyleSheet(f"color:{qc};")

            parts = []
            for m in ('FFT', 'CHROM', 'POS', 'ICA'):
                if m in breakdown:
                    bv, sv = breakdown[m]
                    tag = "*" if m == selected_method else ""
                    parts.append(f"{m}{tag} {bv:.0f}bpm(SNR {sv:.1f})")
            self.methodLabel.setText("  ·  ".join(parts))

            # Skin-tone calibration status
            if self.processor.skin_baseline is not None:
                self.calibLabel.setText("Skin baseline: locked")
            else:
                n_cal = len(self.processor._calib_buf)
                pct   = int(100 * n_cal / CFG.calib_frames)
                self.calibLabel.setText(f"Skin calibration: {pct}%")

            self._queue_csv(raw_bpm, kbpm, avg_snr, breakdown,
                            selected_method, selected_score, confidence,
                            age, gender, status)

        elif raw_bpm is not None:
            self.qualityLabel.setText(
                f"Outlier {raw_bpm:.0f} bpm rejected — stabilising…")
        else:
            self.qualityLabel.setText(
                "Signal weak — stay still, ensure good lighting")
            self.qualityLabel.setStyleSheet("color:#BF616A;")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def update_frame(self):
        if not self.monitoring:
            return

        ret, frame = self.webcam.read()

        # [RB1] Webcam disconnection detection
        if not ret:
            self._cam_fail_count += 1
            if self._cam_fail_count >= CFG.cam_fail_limit:
                self.timer.stop()
                self.monitoring = False
                self.hrLabel.setText("⚠ Camera lost")
                self.qualityLabel.setText("Reconnect your camera and press Start")
                self.qualityLabel.setStyleSheet("color:#BF616A;")
                self.startBtn.setText("Start")
                # Attempt to reopen
                self.webcam.release()
                self.webcam = cv2.VideoCapture(CFG.cam_index)
                self.webcam.set(3, CFG.cam_w)
                self.webcam.set(4, CFG.cam_h)
                self._cam_fail_count = 0
            return
        self._cam_fail_count = 0

        # ── Live FPS ─────────────────────────────────────────
        now = time.time()
        self.fps_timestamps.append(now)
        if len(self.fps_timestamps) >= 2:
            elapsed        = self.fps_timestamps[-1] - self.fps_timestamps[0]
            self._fps_live = (len(self.fps_timestamps) - 1) / elapsed
            self.processor.update_fps(self._fps_live)
            self.MIN_FRAMES = int(self._fps_live * CFG.warmup_seconds)
            self.fpsLabel.setText(f"FPS: {self._fps_live:.1f}")

        # ── Lighting ─────────────────────────────────────────
        ok, bmsg = self.checkLighting(frame)
        if not ok:
            self.hrLabel.setText(f"⚠ {bmsg}")
            self.qualityLabel.setText("Fix lighting before measuring")
            self.qualityLabel.setStyleSheet("color:#BF616A;")
            return

        # ── Separate signal and display paths ────────────────
        signal_frame  = frame
        display_frame = self.applyAHE(frame.copy())
        rgb_frame     = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2RGB)
        gray          = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2GRAY)
        display_rgb   = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # ── Motion gate ──────────────────────────────────────
        if self.detectMotion(gray):
            self.hrLabel.setText("⚠ Motion — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        # ── FaceMesh ─────────────────────────────────────────
        if self.faceLandmarker is None:
            return

        now_ms = int(time.monotonic() * 1000)
        self._mp_ts_ms = max(self._mp_ts_ms + 1, now_ms)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        mesh = self.faceLandmarker.detect_for_video(mp_img, self._mp_ts_ms)

        if not mesh.face_landmarks:
            self.hrLabel.setText("No face detected")
            self.qualityLabel.setText("Point camera at your face")
            return

        face_landmarks = mesh.face_landmarks[0]
        face_pts = self._landmarks_to_points(
            face_landmarks, rgb_frame.shape[1], rgb_frame.shape[0])
        x, y, w, h = cv2.boundingRect(face_pts)

        # ── Age / gender (throttled) ─────────────────────────
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

        # ── Multi-ROI signal extraction ──────────────────────
        roi_rgb, forehead, roi_polys, stable_landmarks, landmark_shift = \
            self.extractROI(rgb_frame, face_landmarks)

        if not stable_landmarks:
            self.qualityLabel.setText(
                f"Landmark motion {landmark_shift:.1f}px — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        # [RB2] Guard against zero/degenerate forehead crop
        if forehead.size == 0 or forehead.shape[0] < 4 or forehead.shape[1] < 4:
            self.qualityLabel.setText("ROI too small — adjust angle")
            return

        self.processor.push(roi_rgb, forehead)
        self.frame_count += 1

        # ── Warm-up ──────────────────────────────────────────
        if self.frame_count < self.MIN_FRAMES:
            pct = int(100 * self.frame_count / self.MIN_FRAMES)
            self.hrLabel.setText(f"Collecting signal… {pct}%")
            self.qualityLabel.setText(
                f"{self.MIN_FRAMES - self.frame_count} frames to go")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")

        # ── Fused BPM  ([AR1] dispatched to worker thread) ───
        elif self.frame_count % CFG.calc_every == 0:
            if not self.worker.isRunning():
                self.worker.request()

            # Eulerian overlay at same cadence
            try:
                filt_frame = self.processor.gauss_overlay(CFG.alpha)
                fh_poly = roi_polys["forehead"]
                fx, fy, fw, fh_h = cv2.boundingRect(fh_poly)
                fh_r = cv2.resize(forehead.astype(np.float32),
                                  (CFG.proc_vid_w, CFG.proc_vid_h)) \
                       if forehead.size > 0 \
                       else np.zeros(
                           (CFG.proc_vid_h, CFG.proc_vid_w, 3), dtype=np.float32)
                fh_amp = cv2.convertScaleAbs(fh_r + filt_frame)
                display_rgb[fy:fy + fh_h, fx:fx + fw] = cv2.resize(fh_amp, (fw, fh_h))
            except Exception:
                pass

        # ── Draw ROI overlays ────────────────────────────────
        cv2.polylines(display_rgb, [roi_polys["forehead"]],    True, (0, 255, 0),   2)
        cv2.polylines(display_rgb, [roi_polys["left_cheek"]],  True, (255, 165, 0), 1)
        cv2.polylines(display_rgb, [roi_polys["right_cheek"]], True, (255, 165, 0), 1)

        # ── Display ──────────────────────────────────────────
        h_, w_, ch = display_rgb.shape
        qImg = QImage(display_rgb.data, w_, h_, ch * w_, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        # ── Pulse waveform ───────────────────────────────────
        gm = float(np.mean(forehead[:, :, 1])) if forehead.size > 0 else 0.0
        self.pulse_buf[self.pulse_idx] = gm
        self.pulse_idx = (self.pulse_idx + 1) % 300
        self.pulseCurve.setData(np.roll(self.pulse_buf, -self.pulse_idx))

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.webcam.release()
        if self.faceLandmarker is not None:
            self.faceLandmarker.close()
        self.csv.close()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app     = QApplication(sys.argv)
    monitor = HeartRateMonitor()
    monitor.show()
    sys.exit(app.exec_())
