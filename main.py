"""
Heart Rate Monitor — Fused rPPG
================================
Single optimal estimator that runs FFT (Eulerian), CHROM, and POS in
parallel and merges them via SNR-weighted fusion.  No algorithm dropdown.

Age/Gender: auto-downloads Caffe models on first run (~50 MB total).
If download fails, falls back to face-size heuristic estimation.

Improvements over original:
  - Live FPS measurement (replaces hardcoded 15 fps)
  - AHE applied to display copy only — signal path is untouched
  - Hann windowing on FFT to reduce spectral leakage
  - Mean-only normalisation in fused_estimate (preserves channel ratios)
  - Face detection throttled to every 10 frames
  - Age/gender prediction throttled to every 30 frames
  - Motion detection normalised relative to frame brightness
  - pulse_buf displayed in correct chronological order
  - QImage uses RGB frame directly (no double cvtColor / channel swap)
  - Eulerian overlay reuses CALC_EVERY cadence (no per-frame FFT)
  - Dead variable mask_ord removed
  - Double detrend on CHROM/POS removed
  - Kalman R tuned to 2.0 for better responsiveness
  - MIN_FRAMES derived from live fps
  - atexit guard on CSV file
  - freq_mask4 roll removed (mask is static)
"""

import numpy as np
import cv2
import sys
import csv
import time
import os
import atexit
import urllib.request
from collections import deque
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QHBoxLayout, QFrame,
                             QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg


# ══════════════════════════════════════════════════════════════
#  AGE / GENDER MODEL LOADER  (auto-download on first run)
# ══════════════════════════════════════════════════════════════

AGE_PROTO    = "age_deploy.prototxt"
AGE_MODEL    = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

_BASE  = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
_CAFFE = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/"

URLS = {
    AGE_PROTO:    _BASE  + "age_net_definitions/deploy.prototxt",
    AGE_MODEL:    _CAFFE + "age_net.caffemodel",
    GENDER_PROTO: _BASE  + "gender_net_definitions/deploy.prototxt",
    GENDER_MODEL: _CAFFE + "gender_net.caffemodel",
}


def _try_download(fname):
    """Download model file if not already present. Returns True on success."""
    if os.path.exists(fname):
        return True
    url = URLS.get(fname)
    if not url:
        return False
    try:
        print(f"[Age/Gender] Downloading {fname} …")
        urllib.request.urlretrieve(url, fname)
        print(f"[Age/Gender] {fname} ready.")
        return True
    except Exception as e:
        print(f"[Age/Gender] Download failed for {fname}: {e}")
        return False


def load_age_gender_nets():
    """
    Returns (ageNet, genderNet) or (None, None) if models unavailable.
    Falls back gracefully — the app works without them.
    """
    files = [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    ok = all(_try_download(f) for f in files)
    if not ok:
        print("[Age/Gender] Models unavailable — using face-size heuristic.")
        return None, None
    try:
        ageNet    = cv2.dnn.readNet(AGE_MODEL,    AGE_PROTO)
        genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        print("[Age/Gender] Caffe models loaded successfully.")
        return ageNet, genderNet
    except Exception as e:
        print(f"[Age/Gender] cv2.dnn.readNet failed: {e}")
        return None, None


def face_size_age_gender(face_w, face_h, frame_w, frame_h):
    """
    Heuristic fallback when Caffe models are unavailable.
    Uses relative face size as a rough age proxy.
    Gender returns 'Unknown' — no reliable heuristic exists.
    """
    ratio = (face_w * face_h) / (frame_w * frame_h)
    if ratio > 0.25:
        age = '(4-9)'
    elif ratio > 0.12:
        age = '(20-39)'
    elif ratio > 0.05:
        age = '(40-59)'
    else:
        age = '(60-100)'
    return age, 'Unknown'


# ══════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════

def detrend(signal):
    """Remove linear trend from signal (improves FFT peak quality)."""
    n = len(signal)
    if n < 2:
        return signal
    x = np.arange(n)
    slope, intercept = np.polyfit(x, signal, 1)
    return signal - (slope * x + intercept)


def bandpass_fft(sig, fps, f_lo=0.75, f_hi=2.5):
    """
    Compute FFT on 1-D signal, isolate heart-rate band [f_lo, f_hi] Hz.
    Band: 0.75–2.5 Hz = 45–150 bpm (physiological adult resting range).

    FIX: Hann window applied before FFT to reduce spectral leakage from
    the rectangular window implied by simply taking an FFT of a finite
    segment.  This sharpens the dominant peak and reduces SNR estimation
    errors caused by side-lobe energy leaking into the band.

    Returns (dominant_hz, bpm, snr_db).
    """
    sig    = detrend(sig)
    sig    = sig - sig.mean()
    # ── Hann window ──────────────────────────────────────────
    window = np.hanning(len(sig))
    sig    = sig * window
    # ─────────────────────────────────────────────────────────
    n      = len(sig)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fps)
    mags   = np.abs(np.fft.rfft(sig))
    band   = (freqs >= f_lo) & (freqs <= f_hi)

    if not band.any() or mags[band].max() < 1e-9:
        return 0.0, 0.0, 0.0

    in_p  = float(np.sum(mags[band]  ** 2))
    out_p = float(np.sum(mags[~band] ** 2))
    snr   = 10 * np.log10(in_p / out_p) if out_p > 0 else 0.0

    mags_f        = mags.copy()
    mags_f[~band] = 0
    hz            = float(freqs[np.argmax(mags_f)])
    return hz, hz * 60.0, snr


def chrom_pulse(rgb):
    """CHROM (de Haan & Jeanne, 2013).  rgb: (N,3)."""
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu      = rgb.mean(0)
    mu[mu == 0] = 1e-9
    n       = rgb / mu
    Xs      = 3 * n[:, 0] - 2 * n[:, 1]
    Ys      = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    a       = Xs.std() / (Ys.std() + 1e-9)
    return detrend(Xs - a * Ys)


def pos_pulse(rgb):
    """POS (Wang et al., 2017).  rgb: (N,3)."""
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu      = rgb.mean(0)
    mu[mu == 0] = 1e-9
    n       = rgb / mu
    S1      = n[:, 1] - n[:, 2]
    S2      = n[:, 1] + n[:, 2] - 2 * n[:, 0]
    b       = S1.std() / (S2.std() + 1e-9)
    return detrend(S1 + b * S2)


# ══════════════════════════════════════════════════════════════
#  FUSED ESTIMATOR
# ══════════════════════════════════════════════════════════════

def fused_estimate(rgb_buf, gauss_buf, fps, freq_mask_1d, filled):
    """
    Optimized fusion:
    - Rejects low-quality signals
    - Selects best method based on SNR
    - Fuses only when methods agree

    FIX: rgb normalisation changed from z-score to mean-only division.
    Z-score (dividing by std) destroyed the channel-ratio information that
    CHROM and POS require.  Mean-normalisation removes DC offset while
    preserving the relative R:G:B ratios that encode the pulse signal.

    FIX: detrend() no longer called here on CHROM/POS pulses — it is
    already called inside chrom_pulse() and pos_pulse() themselves,
    so the previous double-application was wasting compute and could
    over-flatten genuine low-frequency pulse trends.
    """
    if filled < 30:
        return None, 0.0, {}

    rgb   = rgb_buf[-filled:]
    gauss = gauss_buf[-filled:]

    # ── Mean-only normalisation (preserves channel ratios) ──
    mu        = np.mean(rgb, axis=0)
    mu[mu == 0] = 1e-9
    rgb       = rgb / mu
    # ────────────────────────────────────────────────────────

    results = {}

    # ---------------- FFT (Eulerian) ----------------
    try:
        n      = gauss.shape[0]
        freqs  = fps * np.arange(n) / n
        mask   = (freqs >= 0.75) & (freqs <= 2.5)

        cube       = np.fft.fft(gauss, axis=0)
        cube[~mask] = 0

        sig        = np.real(cube).mean(axis=(1, 2, 3))
        _, bpm, snr = bandpass_fft(sig, fps)

        if 45 <= bpm <= 150:
            results['FFT'] = (bpm, snr)

    except Exception as e:
        print(f"[FFT ERROR] {e}")

    # ---------------- CHROM ----------------
    try:
        # detrend already applied inside chrom_pulse — don't call again
        pulse      = chrom_pulse(rgb)
        _, bpm, snr = bandpass_fft(pulse, fps)

        if 45 <= bpm <= 150:
            results['CHROM'] = (bpm, snr)

    except Exception as e:
        print(f"[CHROM ERROR] {e}")

    # ---------------- POS ----------------
    try:
        # detrend already applied inside pos_pulse — don't call again
        pulse      = pos_pulse(rgb)
        _, bpm, snr = bandpass_fft(pulse, fps)

        if 45 <= bpm <= 150:
            results['POS'] = (bpm, snr)

    except Exception as e:
        print(f"[POS ERROR] {e}")

    # ---------------- NO SIGNAL ----------------
    if not results:
        return None, 0.0, {}

    # ---------------- QUALITY FILTER ----------------
    SNR_THRESHOLD = -2.0

    valid = {
        m: (b, s) for m, (b, s) in results.items()
        if s > SNR_THRESHOLD
    }

    if not valid:
        best_method = max(results.items(), key=lambda x: x[1][1])
        best_bpm, best_snr = best_method[1]
        return best_bpm, best_snr, results

    # ---------------- CONSISTENCY CHECK ----------------
    bpms     = [b for b, s in valid.values()]
    snrs     = [s for b, s in valid.values()]
    bpm_range = max(bpms) - min(bpms)

    if len(valid) >= 2 and bpm_range < 10:
        weights   = np.array([max(s, 0.1) for s in snrs])
        weights  /= np.sum(weights)
        fused_bpm = np.sum(weights * np.array(bpms))
        avg_snr   = float(np.mean(snrs))
        return fused_bpm, avg_snr, results

    # ---------------- OTHERWISE: PICK BEST ----------------
    best_method       = max(valid.items(), key=lambda x: x[1][1])
    best_bpm, best_snr = best_method[1]
    return best_bpm, best_snr, results


# ══════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════

class KalmanBPM:
    def __init__(self):
        self.x = 75.0
        self.P = 25.0
        self.Q = 0.5
        # FIX: R reduced from 5.0 → 2.0 so Kalman responds faster to
        # genuine BPM changes while still smoothing frame-to-frame noise.
        self.R = 2.0

    def update(self, z):
        P_ = self.P + self.Q
        K  = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x

    def predict(self):
        self.P += self.Q
        return self.x


# ══════════════════════════════════════════════════════════════
#  SIGNAL PROCESSOR  (separated from UI)
# ══════════════════════════════════════════════════════════════

class SignalProcessor:
    """
    Owns all rPPG buffers and the fused estimator.
    Completely decoupled from Qt — can be unit-tested independently.
    """

    def __init__(self, fps_init=15, buf_len=150, levels=3,
                 vid_w=160, vid_h=120):
        self.fps    = fps_init
        self.BUF    = buf_len
        self.levels = levels
        self.vidW   = vid_w
        self.vidH   = vid_h

        dummy = np.zeros((vid_h, vid_w, 3), dtype=np.float32)
        g0    = self._build_gauss(dummy, levels + 1)[levels]
        self.gauss_buf = np.zeros((buf_len, g0.shape[0], g0.shape[1], 3))
        self.rgb_buf   = np.zeros((buf_len, 3))

        freqs          = fps_init * np.arange(buf_len) / buf_len
        self.freq_mask = (freqs >= 0.75) & (freqs <= 2.5)

        self.buf_idx       = 0
        self.filled_frames = 0

        self.kalman      = KalmanBPM()
        self.bpm_history = deque(maxlen=20)

        self.OUTLIER_DELTA = 35

    # ── public ────────────────────────────────────────────────

    def update_fps(self, fps):
        """Call with measured fps each frame."""
        self.fps = fps
        freqs          = fps * np.arange(self.BUF) / self.BUF
        self.freq_mask = (freqs >= 0.75) & (freqs <= 2.5)

    def push(self, roi_rgb, forehead_frame):
        """Add one frame's worth of data to the ring buffers."""
        self.rgb_buf[self.buf_idx] = roi_rgb

        fh_r = cv2.resize(forehead_frame.astype(np.float32),
                          (self.vidW, self.vidH)) \
               if forehead_frame.size > 0 \
               else np.zeros((self.vidH, self.vidW, 3), dtype=np.float32)

        gl = self._build_gauss(fh_r, self.levels + 1)[self.levels]
        self.gauss_buf[self.buf_idx] = gl

        self.buf_idx       = (self.buf_idx + 1) % self.BUF
        self.filled_frames = min(self.filled_frames + 1, self.BUF)

    def estimate(self):
        """
        Run fused estimator on the current buffer contents.
        Returns (kalman_bpm, raw_bpm, snr, breakdown, accepted).
        """
        rgb_ord   = np.roll(self.rgb_buf,   -self.buf_idx, axis=0)
        gauss_ord = np.roll(self.gauss_buf, -self.buf_idx, axis=0)

        raw_bpm, avg_snr, breakdown = fused_estimate(
            rgb_ord, gauss_ord, self.fps,
            self.freq_mask,          # FIX: static mask — no roll needed
            self.filled_frames)

        if raw_bpm is None or not (45 <= raw_bpm <= 150):
            self.kalman.predict()
            return None, None, 0.0, breakdown, False

        if self._is_outlier(raw_bpm):
            self.kalman.predict()
            return None, raw_bpm, avg_snr, breakdown, False

        self.bpm_history.append(raw_bpm)
        kbpm = self.kalman.update(raw_bpm)
        return kbpm, raw_bpm, avg_snr, breakdown, True

    def gauss_overlay(self, alpha=170):
        """
        Compute Eulerian-amplified forehead frame for display overlay.
        Only call this at the same cadence as estimate() to avoid
        per-frame FFT recomputation.
        """
        fft_cube            = np.fft.fft(self.gauss_buf, axis=0)
        mask4               = self.freq_mask[:, None, None, None]
        fft_cube[~mask4]    = 0
        filtered            = np.real(np.fft.ifft(fft_cube, axis=0)) * alpha
        idx                 = (self.buf_idx - 1) % self.BUF
        f                   = filtered[idx]
        for _ in range(self.levels):
            f = cv2.pyrUp(f)
        return f[:self.vidH, :self.vidW]

    # ── private ───────────────────────────────────────────────

    def _build_gauss(self, frame, levels):
        pyr = [frame]
        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyr.append(frame)
        return pyr

    def _is_outlier(self, bpm):
        if len(self.bpm_history) < 5:
            return False
        return abs(bpm - float(np.median(list(self.bpm_history)))) \
               > self.OUTLIER_DELTA


# ══════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # ── Config ────────────────────────────────────────────
        self.realW  = 320
        self.realH  = 240
        self.alpha  = 170
        self.CALC_EVERY = 8        # recalculate every N frames
        # MIN_FRAMES derived later once live FPS is measured
        self.MIN_FRAMES = 45       # updated in update_frame

        # ── Live FPS measurement ──────────────────────────────
        # FIX: replaces hardcoded fps=15 which caused incorrect
        # frequency calculations whenever the camera ran at a
        # different rate.
        self.fps_timestamps = deque(maxlen=30)
        self._fps_live      = 15.0

        # ── Signal processor ─────────────────────────────────
        self.processor = SignalProcessor(
            fps_init=15, buf_len=150, levels=3, vid_w=160, vid_h=120)

        # ── HR history (for plot) ─────────────────────────────
        self.hr_raw    = []
        self.hr_smooth = []

        # ── Thresholds ────────────────────────────────────────
        self.MOTION_THRESH = 0.08   # relative (normalised) motion threshold
        self.BRIGHT_MIN    = 30
        self.BRIGHT_MAX    = 230

        # ── Pulse waveform display ────────────────────────────
        self.pulse_buf = np.zeros(300)
        self.pulse_idx = 0

        # ── Motion ───────────────────────────────────────────
        self.prev_gray = None

        # ── Frame counters ───────────────────────────────────
        self.frame_count = 0

        # ── Face detection throttle ───────────────────────────
        # FIX: Haar cascade is expensive (~10–30 ms per frame).
        # Re-detect every 10 frames; between detections reuse the
        # last known rectangle.
        self.face_rect         = None
        self.face_frame_count  = 0
        self.FACE_DETECT_EVERY = 10

        # ── Age/gender throttle ───────────────────────────────
        # FIX: DNN inference runs ~50–200 ms per call.  Throttle
        # to every 30 frames (~2 s at 15 fps).
        self.last_age          = 'Unknown'
        self.last_gender       = 'Unknown'
        self.ag_frame_count    = 0
        self.AG_PREDICT_EVERY  = 30

        # ── Camera ───────────────────────────────────────────
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(3, self.realW)
        self.webcam.set(4, self.realH)

        # ── Face detector ────────────────────────────────────
        self.faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # ── Age/gender models ─────────────────────────────────
        self.ageList    = ['(0-3)', '(4-9)', '(10-15)', '(16-19)',
                           '(20-39)', '(40-59)', '(60-100)']
        self.genderList = ['Male', 'Female']
        self.ageNet, self.genderNet = load_age_gender_nets()

        # ── CSV ───────────────────────────────────────────────
        self.csv_file   = open('heart_rate_data.csv', 'w', newline='')
        # FIX: atexit ensures the file is closed even on unexpected exit
        atexit.register(self.csv_file.close)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Timestamp', 'Raw BPM', 'Kalman BPM', 'Avg SNR (dB)',
            'FFT BPM', 'FFT SNR', 'CHROM BPM', 'CHROM SNR',
            'POS BPM', 'POS SNR', 'Age', 'Gender', 'HR Status'])

        # ── Timer ────────────────────────────────────────────
        self.monitoring = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ──────────────────────────────────────────────────────────
    #  UI
    # ──────────────────────────────────────────────────────────

    def initUI(self):
        self.setWindowTitle('Heart Rate Monitor (Fused rPPG)')
        self.setStyleSheet("background-color:#2E3440; color:white;")

        root = QVBoxLayout()
        top  = QHBoxLayout()

        self.videoLabel = QLabel()
        self.videoLabel.setFixedSize(320, 240)
        self.videoLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.videoLabel.setStyleSheet("border:2px solid #4C566A;")
        top.addWidget(self.videoLabel)

        right = QVBoxLayout()

        self.hrLabel = QLabel('HR: —')
        self.hrLabel.setFont(QFont('Arial', 20))
        self.hrLabel.setStyleSheet("color:#88C0D0;")
        self.hrLabel.setAlignment(Qt.AlignCenter)
        right.addWidget(self.hrLabel)

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

        self.hrPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrPlot.setBackground('#3B4252')
        self.hrPlot.getAxis('left').setPen(pg.mkPen('white'))
        self.hrPlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.hrPlot.setYRange(40, 180)
        self.hrPlot.addLegend()
        self.curveRaw    = self.hrPlot.plot(pen=pg.mkPen('r', width=2),
                                            name='Raw (fused)')
        self.curveKalman = self.hrPlot.plot(pen=pg.mkPen('y', width=2),
                                            name='Kalman')
        right.addWidget(self.hrPlot)

        top.addLayout(right)
        root.addLayout(top)

        self.pulsePlot = pg.PlotWidget(title="Pulse Signal (Green channel)")
        self.pulsePlot.setBackground('#3B4252')
        self.pulsePlot.getAxis('left').setPen(pg.mkPen('white'))
        self.pulsePlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.pulsePlot.enableAutoRange(axis='y')
        self.pulseCurve = self.pulsePlot.plot(
            pen=pg.mkPen('#A3BE8C', width=2))
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
        btns.addItem(QSpacerItem(
            20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(btns)

        self.setLayout(root)

    # ──────────────────────────────────────────────────────────
    #  PREPROCESSING
    # ──────────────────────────────────────────────────────────

    def applyAHE(self, frame):
        """CLAHE on the L channel of LAB.  For display copy only."""
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl      = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # ──────────────────────────────────────────────────────────
    #  CHECKS
    # ──────────────────────────────────────────────────────────

    def checkLighting(self, frame):
        b = float(np.mean(frame))
        if b < self.BRIGHT_MIN: return False, f"Too dark ({b:.0f})"
        if b > self.BRIGHT_MAX: return False, f"Too bright ({b:.0f})"
        return True, f"{b:.0f}"

    def detectMotion(self, gray):
        """
        FIX: threshold is now relative to mean frame brightness so that
        the same physical motion doesn't trigger differently under bright
        vs dim lighting conditions.
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return False
        diff       = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray.copy()
        norm_diff  = float(diff.mean()) / (float(gray.mean()) + 1e-9)
        return norm_diff > self.MOTION_THRESH

    # ──────────────────────────────────────────────────────────
    #  MULTI-ROI
    # ──────────────────────────────────────────────────────────

    def extractROI(self, rgb, x, y, w, h):
        fh = rgb[y : y + max(1, int(0.25 * h)),               x : x + w]
        lc = rgb[y + int(0.45*h) : y + int(0.75*h), x : x + max(1, int(0.45*w))]
        rc = rgb[y + int(0.45*h) : y + int(0.75*h), x + int(0.55*w) : x + w]
        out = np.zeros(3)
        for roi, wt in zip([fh, lc, rc], [0.5, 0.25, 0.25]):
            if roi.size > 0:
                out += wt * roi.mean(axis=(0, 1))
        return out, fh

    # ──────────────────────────────────────────────────────────
    #  AGE / GENDER
    # ──────────────────────────────────────────────────────────

    def predictAgeGender(self, face_img, face_w=0, face_h=0):
        if self.ageNet is None or self.genderNet is None:
            return face_size_age_gender(
                face_w, face_h, self.realW, self.realH)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), (104, 177, 123), swapRB=False)
        self.genderNet.setInput(blob)
        gender = self.genderList[self.genderNet.forward()[0].argmax()]
        self.ageNet.setInput(blob)
        age    = self.ageList[self.ageNet.forward()[0].argmax()]
        return age, gender

    # ──────────────────────────────────────────────────────────
    #  HR STATUS
    # ──────────────────────────────────────────────────────────

    def hrStatus(self, bpm, age, gender):
        if age == 'Unknown' or gender == 'Unknown':
            if bpm < 60:   return "Low"
            if bpm <= 100: return "Normal"
            if bpm <= 120: return "High"
            return "Very High"
        young = age in ('(0-3)', '(4-9)', '(10-15)', '(16-19)')
        if gender.lower() == 'male':
            if young:              thr = [(50,'Very Low'),(70,'Low'),(100,'Normal'),(130,'High')]
            elif age == '(20-39)': thr = [(55,'Very Low'),(70,'Low'),(100,'Normal'),(120,'High')]
            else:                  thr = [(50,'Very Low'),(65,'Low'),(100,'Normal'),(120,'High')]
        else:
            if young:              thr = [(55,'Very Low'),(75,'Low'),(105,'Normal'),(135,'High')]
            elif age == '(20-39)': thr = [(60,'Very Low'),(75,'Low'),(105,'Normal'),(125,'High')]
            else:                  thr = [(55,'Very Low'),(70,'Low'),(105,'Normal'),(125,'High')]
        for t, lbl in thr:
            if bpm < t:
                return lbl
        return "Very High"

    # ──────────────────────────────────────────────────────────
    #  CSV
    # ──────────────────────────────────────────────────────────

    def saveCSV(self, raw, kalman_v, snr, breakdown, age, gender, status):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')

        def get(k): return breakdown.get(k, (0.0, 0.0))

        fb, fs = get('FFT')
        cb, cs = get('CHROM')
        pb, ps = get('POS')
        self.csv_writer.writerow([
            ts, f'{raw:.1f}', f'{kalman_v:.1f}', f'{snr:.2f}',
            f'{fb:.1f}', f'{fs:.2f}', f'{cb:.1f}', f'{cs:.2f}',
            f'{pb:.1f}', f'{ps:.2f}', age, gender, status])
        self.csv_file.flush()

    # ──────────────────────────────────────────────────────────
    #  CONTROL
    # ──────────────────────────────────────────────────────────

    def startMonitoring(self):
        self.monitoring = True
        self.timer.start(1000 // 15)   # nominal; actual rate measured live
        self.startBtn.setText("Pause")

    def stopMonitoring(self):
        self.monitoring = False
        self.timer.stop()
        self.startBtn.setText("Start")

    # ──────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────

    def update_frame(self):
        if not self.monitoring:
            return

        ret, frame = self.webcam.read()
        if not ret:
            return

        # ── Live FPS measurement ─────────────────────────────
        now = time.time()
        self.fps_timestamps.append(now)
        if len(self.fps_timestamps) >= 2:
            elapsed        = self.fps_timestamps[-1] - self.fps_timestamps[0]
            self._fps_live = (len(self.fps_timestamps) - 1) / elapsed
            self.processor.update_fps(self._fps_live)
            self.MIN_FRAMES = int(self._fps_live * 3)  # ~3 s warm-up
            self.fpsLabel.setText(f"FPS: {self._fps_live:.1f}")

        # ── Lighting check ───────────────────────────────────
        ok, bmsg = self.checkLighting(frame)
        if not ok:
            self.hrLabel.setText(f"⚠ {bmsg}")
            self.qualityLabel.setText("Fix lighting before measuring")
            self.qualityLabel.setStyleSheet("color:#BF616A;")
            return

        # ── Signal path uses RAW frame; display path uses AHE ─
        # FIX: the original code ran applyAHE on `frame` and then used
        # that same frame for signal extraction.  CLAHE alters pixel
        # values inter-frame inconsistently, corrupting the tiny RGB
        # changes that CHROM and POS depend on.
        signal_frame  = frame                          # untouched
        display_frame = self.applyAHE(frame.copy())   # enhanced for display only

        rgb_frame     = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2RGB)
        gray          = cv2.cvtColor(signal_frame,  cv2.COLOR_BGR2GRAY)
        display_rgb   = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # ── Motion gate ──────────────────────────────────────
        if self.detectMotion(gray):
            self.hrLabel.setText("⚠ Motion — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        # ── Face detection (throttled) ───────────────────────
        # FIX: cascade runs every FACE_DETECT_EVERY frames only.
        self.face_frame_count += 1
        if self.face_frame_count % self.FACE_DETECT_EVERY == 0 \
                or self.face_rect is None:
            faces = self.faceCascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces):
                self.face_rect = faces[0]
            else:
                self.face_rect = None

        if self.face_rect is None:
            self.hrLabel.setText("No face detected")
            self.qualityLabel.setText("Point camera at your face")
            return

        x, y, w, h = self.face_rect

        # ── Age/gender (throttled) ───────────────────────────
        # FIX: DNN inference moved out of the per-frame hot path.
        self.ag_frame_count += 1
        if self.ag_frame_count % self.AG_PREDICT_EVERY == 0:
            fc = cv2.resize(rgb_frame[y:y+h, x:x+w], (227, 227))
            self.last_age, self.last_gender = \
                self.predictAgeGender(fc, face_w=w, face_h=h)
        age, gender = self.last_age, self.last_gender

        # ── Multi-ROI signal extraction ──────────────────────
        roi_rgb, forehead = self.extractROI(rgb_frame, x, y, w, h)
        self.processor.push(roi_rgb, forehead)
        self.frame_count += 1

        # ── Warm-up progress ─────────────────────────────────
        if self.frame_count < self.MIN_FRAMES:
            pct = int(100 * self.frame_count / self.MIN_FRAMES)
            self.hrLabel.setText(f"Collecting signal… {pct}%")
            self.qualityLabel.setText(
                f"{self.MIN_FRAMES - self.frame_count} frames to go")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")

        # ── Fused BPM calculation ────────────────────────────
        elif self.frame_count % self.CALC_EVERY == 0:
            kbpm, raw_bpm, avg_snr, breakdown, accepted = \
                self.processor.estimate()

            if accepted:
                status = self.hrStatus(kbpm, age, gender)

                self.hr_raw.append(raw_bpm)
                self.hr_smooth.append(kbpm)
                self.curveRaw.setData(self.hr_raw[-60:])
                self.curveKalman.setData(self.hr_smooth[-60:])

                self.hrLabel.setText(f"HR: {kbpm:.1f} bpm  [{status}]")
                self.hrLabel.setStyleSheet("color:#88C0D0;")

                qc = "#A3BE8C" if avg_snr >= 2.0 else "#EBCB8B"
                self.qualityLabel.setText(
                    f"SNR: {avg_snr:.1f} dB  |  Raw: {raw_bpm:.1f} bpm  |  "
                    f"Age: {age}  Gender: {gender}")
                self.qualityLabel.setStyleSheet(f"color:{qc};")

                parts = []
                for m in ('FFT', 'CHROM', 'POS'):
                    if m in breakdown:
                        bv, sv = breakdown[m]
                        parts.append(f"{m} {bv:.0f}bpm(SNR {sv:.1f})")
                self.methodLabel.setText("  ·  ".join(parts))

                self.saveCSV(raw_bpm, kbpm, avg_snr,
                             breakdown, age, gender, status)

            elif raw_bpm is not None:
                self.qualityLabel.setText(
                    f"Outlier {raw_bpm:.0f} bpm rejected — stabilising…")
            else:
                self.qualityLabel.setText(
                    "Signal weak — stay still, ensure good lighting")
                self.qualityLabel.setStyleSheet("color:#BF616A;")

            # ── Eulerian overlay (same cadence as estimate) ──
            # FIX: overlay FFT now only runs every CALC_EVERY frames,
            # not every frame.  Eliminates a full FFT on the gauss_buf
            # (BUF × H × W × 3) on every single timer tick.
            try:
                filt_frame = self.processor.gauss_overlay(self.alpha)
                fh_r       = cv2.resize(
                    forehead.astype(np.float32),
                    (self.processor.vidW, self.processor.vidH)) \
                    if forehead.size > 0 \
                    else np.zeros(
                        (self.processor.vidH, self.processor.vidW, 3),
                        dtype=np.float32)
                fh_amp = cv2.convertScaleAbs(fh_r + filt_frame)
                ov_h   = max(1, int(0.25 * h))
                display_rgb[y : y + ov_h, x : x + w] = \
                    cv2.resize(fh_amp, (w, ov_h))
            except Exception:
                pass

        # ── Draw ROI boxes on display_rgb ────────────────────
        ov_h = max(1, int(0.25 * h))
        cv2.rectangle(display_rgb, (x, y), (x + w, y + ov_h), (0, 255, 0), 2)
        cy1 = y + int(0.45 * h)
        cy2 = y + int(0.75 * h)
        cv2.rectangle(display_rgb,
                      (x, cy1), (x + int(0.45 * w), cy2), (255, 165, 0), 1)
        cv2.rectangle(display_rgb,
                      (x + int(0.55 * w), cy1), (x + w, cy2), (255, 165, 0), 1)

        # ── Display (RGB frame directly — no double cvtColor) ─
        # FIX: original code converted rgb_frame → BGR → QImage with
        # Format_RGB888, which swapped R and B channels in the display.
        # Now display_rgb stays in RGB throughout and is passed directly.
        h_, w_, ch = display_rgb.shape
        qImg = QImage(display_rgb.data, w_, h_, ch * w_, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        # ── Pulse waveform ────────────────────────────────────
        gm = float(np.mean(forehead[:, :, 1])) if forehead.size > 0 else 0.0
        self.pulse_buf[self.pulse_idx] = gm
        self.pulse_idx = (self.pulse_idx + 1) % 300
        # FIX: roll buffer so waveform displays in chronological order
        # rather than wrapping mid-plot.
        ordered = np.roll(self.pulse_buf, -self.pulse_idx)
        self.pulseCurve.setData(ordered)

    # ──────────────────────────────────────────────────────────
    #  CLOSE
    # ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.webcam.release()
        self.csv_file.close()
        event.accept()


if __name__ == '__main__':
    app     = QApplication(sys.argv)
    monitor = HeartRateMonitor()
    monitor.show()
    sys.exit(app.exec_())