"""
Heart Rate Monitor — Fused rPPG
================================
Single optimal estimator that runs FFT (Eulerian), CHROM, and POS in
parallel and merges them via SNR-weighted fusion.  No algorithm dropdown.

Age/Gender: auto-downloads Caffe models on first run (~50 MB total).
If download fails, falls back to face-size heuristic estimation.
"""

import numpy as np
import cv2
import sys
import csv
import time
import os
import urllib.request
from collections import deque
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QHBoxLayout, QFrame, QMenuBar,
                             QAction, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg


# ══════════════════════════════════════════════════════════════
#  AGE / GENDER MODEL LOADER  (auto-download on first run)
# ══════════════════════════════════════════════════════════════

AGE_PROTO   = "age_deploy.prototxt"
AGE_MODEL   = "age_net.caffemodel"
GENDER_PROTO= "gender_deploy.prototxt"
GENDER_MODEL= "gender_net.caffemodel"

_BASE = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
_CAFFE= "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/"

URLS = {
    AGE_PROTO:    _BASE + "age_net_definitions/deploy.prototxt",
    AGE_MODEL:    _CAFFE + "age_net.caffemodel",
    GENDER_PROTO: _BASE + "gender_net_definitions/deploy.prototxt",
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
    # Children typically have larger face-to-frame ratios when close to camera
    # Adults vary widely — so we just estimate broad buckets
    if ratio > 0.25:
        age = '(4-9)'      # very close / small face (child-like framing)
    elif ratio > 0.12:
        age = '(20-39)'    # typical adult selfie distance
    elif ratio > 0.05:
        age = '(40-59)'    # farther away / smaller face
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
    Signals outside this are almost certainly noise or harmonics.
    Returns (dominant_hz, bpm, snr_db).
    """
    sig   = detrend(sig)                         # remove DC + linear drift
    sig   = sig - sig.mean()                     # zero-mean
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mags  = np.abs(np.fft.rfft(sig))
    band  = (freqs >= f_lo) & (freqs <= f_hi)

    if not band.any() or mags[band].max() < 1e-9:
        return 0.0, 0.0, 0.0

    in_p  = float(np.sum(mags[band]  ** 2))
    out_p = float(np.sum(mags[~band] ** 2))
    snr   = 10 * np.log10(in_p / out_p) if out_p > 0 else 0.0

    mags_f = mags.copy(); mags_f[~band] = 0
    hz     = float(freqs[np.argmax(mags_f)])
    return hz, hz * 60.0, snr


def chrom_pulse(rgb):
    """CHROM (de Haan & Jeanne, 2013).  rgb: (N,3)."""
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu = rgb.mean(0); mu[mu == 0] = 1e-9
    n  = rgb / mu
    Xs = 3*n[:,0] - 2*n[:,1]
    Ys = 1.5*n[:,0] + n[:,1] - 1.5*n[:,2]
    a  = Xs.std() / (Ys.std() + 1e-9)
    return detrend(Xs - a * Ys)


def pos_pulse(rgb):
    """POS (Wang et al., 2017).  rgb: (N,3)."""
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mu = rgb.mean(0); mu[mu == 0] = 1e-9
    n  = rgb / mu
    S1 = n[:,1] - n[:,2]
    S2 = n[:,1] + n[:,2] - 2*n[:,0]
    b  = S1.std() / (S2.std() + 1e-9)
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
    """

    if filled < 30:
        return None, 0.0, {}

    rgb   = rgb_buf[-filled:]
    gauss = gauss_buf[-filled:]

    # Normalize RGB (critical for CHROM/POS)
    rgb = (rgb - np.mean(rgb, axis=0)) / (np.std(rgb, axis=0) + 1e-9)

    results = {}

    # ---------------- FFT (Eulerian) ----------------
    try:
        n      = gauss.shape[0]
        freqs  = fps * np.arange(n) / n
        mask   = (freqs >= 0.75) & (freqs <= 2.5)

        cube   = np.fft.fft(gauss, axis=0)
        cube[~mask] = 0

        sig    = np.real(cube).mean(axis=(1, 2, 3))
        _, bpm, snr = bandpass_fft(sig, fps)

        if 45 <= bpm <= 150:
            results['FFT'] = (bpm, snr)

    except Exception as e:
        print(f"[FFT ERROR] {e}")

    # ---------------- CHROM ----------------
    try:
        pulse = chrom_pulse(rgb)
        pulse = detrend(pulse)

        _, bpm, snr = bandpass_fft(pulse, fps)

        if 45 <= bpm <= 150:
            results['CHROM'] = (bpm, snr)

    except Exception as e:
        print(f"[CHROM ERROR] {e}")

    # ---------------- POS ----------------
    try:
        pulse = pos_pulse(rgb)
        pulse = detrend(pulse)

        _, bpm, snr = bandpass_fft(pulse, fps)

        if 45 <= bpm <= 150:
            results['POS'] = (bpm, snr)

    except Exception as e:
        print(f"[POS ERROR] {e}")

    # ---------------- NO SIGNAL ----------------
    if not results:
        return None, 0.0, {}

    # ---------------- QUALITY FILTER ----------------
    # Keep only methods with acceptable SNR
    SNR_THRESHOLD = -2.0   # tune this

    valid = {
        m: (b, s) for m, (b, s) in results.items()
        if s > SNR_THRESHOLD
    }

    # If nothing passes threshold → fallback to best available
    if not valid:
        best_method = max(results.items(), key=lambda x: x[1][1])
        best_bpm, best_snr = best_method[1]
        return best_bpm, best_snr, results

    # ---------------- CONSISTENCY CHECK ----------------
    bpms = [b for b, s in valid.values()]
    snrs = [s for b, s in valid.values()]

    bpm_range = max(bpms) - min(bpms)

    # If methods agree → fuse
    if len(valid) >= 2 and bpm_range < 10:
        weights = np.array([max(s, 0.1) for s in snrs])
        weights /= np.sum(weights)

        fused_bpm = np.sum(weights * np.array(bpms))
        avg_snr   = float(np.mean(snrs))

        return fused_bpm, avg_snr, results

    # ---------------- OTHERWISE: PICK BEST ----------------
    best_method = max(valid.items(), key=lambda x: x[1][1])
    best_bpm, best_snr = best_method[1]

    return best_bpm, best_snr, results


# ══════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════

class KalmanBPM:
    def __init__(self):
        self.x = 75.0
        self.P = 25.0
        self.Q = 0.5   # lower → smoother
        self.R = 5.0

    def update(self, z):
        P_  = self.P + self.Q
        K   = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x

    def predict(self):
        self.P += self.Q
        return self.x


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # ── Config ────────────────────────────────────────────
        self.fps    = 15
        self.realW  = 320
        self.realH  = 240
        self.vidW   = 160
        self.vidH   = 120
        self.levels = 3
        self.alpha  = 170
        self.BUF    = 150          # rolling buffer length
        self.CALC_EVERY  = 8       # recalculate every N frames
        self.MIN_FRAMES  = 45      # warm-up: ~3 s at 15 fps

        # ── Gauss pyramid buffer (Eulerian / FFT path) ────────
        dummy = np.zeros((self.vidH, self.vidW, 3), dtype=np.float32)
        g0    = self.buildGauss(dummy, self.levels + 1)[self.levels]
        self.gauss_buf = np.zeros((self.BUF, g0.shape[0], g0.shape[1], 3))

        # 1-D and 4-D frequency masks for Euler FFT  (0.75–2.5 Hz = 45–150 bpm)
        freqs           = self.fps * np.arange(self.BUF) / self.BUF
        self.freq_mask  = (freqs >= 0.75) & (freqs <= 2.5)   # 1-D
        self.freq_mask4 = self.freq_mask[:, None, None, None]

        # ── Multi-ROI RGB buffer (CHROM / POS) ───────────────
        self.rgb_buf = np.zeros((self.BUF, 3))

        # ── Index / frame counters ────────────────────────────
        self.buf_idx     = 0
        self.frame_count = 0

        # ── BPM smoothing & history ───────────────────────────
        self.kalman      = KalmanBPM()
        self.bpm_history = deque(maxlen=20)
        self.hr_raw      = []
        self.hr_smooth   = []

        # ── Thresholds ────────────────────────────────────────
        self.MOTION_THRESH  = 20.0
        self.BRIGHT_MIN     = 30
        self.BRIGHT_MAX     = 230
        self.OUTLIER_DELTA  = 35

        # ── Pulse waveform display ────────────────────────────
        self.pulse_buf = np.zeros(300)
        self.pulse_idx = 0

        # ── Motion ───────────────────────────────────────────
        self.prev_gray = None

        # ── Camera ───────────────────────────────────────────
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(3, self.realW)
        self.webcam.set(4, self.realH)

        # ── Face detector ────────────────────────────────────
        self.faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # ── Age/gender models (auto-downloaded on first run) ──
        self.ageList    = ['(0-3)','(4-9)','(10-15)','(16-19)',
                           '(20-39)','(40-59)','(60-100)']
        self.genderList = ['Male', 'Female']
        self.ageNet, self.genderNet = load_age_gender_nets()

        # filled_frames: how many unique frames are in the ring buffer
        # (never exceeds BUF; prevents zero-padded rows reaching CHROM/POS)
        self.filled_frames = 0

        # ── CSV ───────────────────────────────────────────────
        self.csv_file   = open('heart_rate_data.csv', 'w', newline='')
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

        self.hrPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrPlot.setBackground('#3B4252')
        self.hrPlot.getAxis('left').setPen(pg.mkPen('white'))
        self.hrPlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.hrPlot.setYRange(40, 180)
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
        btns.addItem(QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        root.addLayout(btns)

        self.setLayout(root)

    # ──────────────────────────────────────────────────────────
    #  PYRAMID
    # ──────────────────────────────────────────────────────────

    def buildGauss(self, frame, levels):
        pyr = [frame]
        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyr.append(frame)
        return pyr

    def reconstructFrame(self, cube, idx, levels):
        f = cube[idx]
        for _ in range(levels):
            f = cv2.pyrUp(f)
        return f[:self.vidH, :self.vidW]

    # ──────────────────────────────────────────────────────────
    #  PREPROCESSING
    # ──────────────────────────────────────────────────────────

    def applyAHE(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def normSkin(self, frame):
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    # ──────────────────────────────────────────────────────────
    #  CHECKS
    # ──────────────────────────────────────────────────────────

    def checkLighting(self, frame):
        b = float(np.mean(frame))
        if b < self.BRIGHT_MIN: return False, f"Too dark ({b:.0f})"
        if b > self.BRIGHT_MAX: return False, f"Too bright ({b:.0f})"
        return True, f"{b:.0f}"

    def detectMotion(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy(); return False
        d = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
        self.prev_gray = gray.copy()
        return d > self.MOTION_THRESH

    def isOutlier(self, bpm):
        if len(self.bpm_history) < 5: return False
        return abs(bpm - float(np.median(list(self.bpm_history)))) > self.OUTLIER_DELTA

    # ──────────────────────────────────────────────────────────
    #  MULTI-ROI
    # ──────────────────────────────────────────────────────────

    def extractROI(self, rgb, x, y, w, h):
        fh = rgb[y : y+max(1,int(0.25*h)),              x : x+w]
        lc = rgb[y+int(0.45*h) : y+int(0.75*h), x : x+max(1,int(0.45*w))]
        rc = rgb[y+int(0.45*h) : y+int(0.75*h), x+int(0.55*w) : x+w]
        out = np.zeros(3)
        for roi, wt in zip([fh, lc, rc], [0.5, 0.25, 0.25]):
            if roi.size > 0:
                out += wt * roi.mean(axis=(0, 1))
        return out, fh

    # ──────────────────────────────────────────────────────────
    #  AGE / GENDER
    # ──────────────────────────────────────────────────────────

    def predictAgeGender(self, face_img, face_w=0, face_h=0):
        """
        Predict age & gender.  Uses Caffe DNN if models loaded,
        otherwise falls back to face-size heuristic.
        """
        if self.ageNet is None or self.genderNet is None:
            # Heuristic: face size relative to frame
            return face_size_age_gender(
                face_w, face_h, self.realW, self.realH)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227,227), (104,177,123), swapRB=False)
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
            if bpm < 60: return "Low"
            if bpm <= 100: return "Normal"
            if bpm <= 120: return "High"
            return "Very High"
        young = age in ('(0-3)','(4-9)','(10-15)','(16-19)')
        if gender.lower() == 'male':
            if young:            thr = [(50,'Very Low'),(70,'Low'),(100,'Normal'),(130,'High')]
            elif age=='(20-39)': thr = [(55,'Very Low'),(70,'Low'),(100,'Normal'),(120,'High')]
            else:                thr = [(50,'Very Low'),(65,'Low'),(100,'Normal'),(120,'High')]
        else:
            if young:            thr = [(55,'Very Low'),(75,'Low'),(105,'Normal'),(135,'High')]
            elif age=='(20-39)': thr = [(60,'Very Low'),(75,'Low'),(105,'Normal'),(125,'High')]
            else:                thr = [(55,'Very Low'),(70,'Low'),(105,'Normal'),(125,'High')]
        for t, lbl in thr:
            if bpm < t: return lbl
        return "Very High"

    # ──────────────────────────────────────────────────────────
    #  CSV
    # ──────────────────────────────────────────────────────────

    def saveCSV(self, raw, kalman_v, snr, breakdown, age, gender, status):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        def get(k): return breakdown.get(k, (0.0, 0.0))
        fb, fs = get('FFT'); cb, cs = get('CHROM'); pb, ps = get('POS')
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
        self.timer.start(1000 // self.fps)
        self.startBtn.setText("Pause")

    def stopMonitoring(self):
        self.monitoring = False
        self.timer.stop()
        self.startBtn.setText("Start")

    # ──────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────

    def update_frame(self):
        if not self.monitoring: return

        ret, frame = self.webcam.read()
        if not ret: return

        # Lighting
        ok, bmsg = self.checkLighting(frame)
        if not ok:
            self.hrLabel.setText(f"⚠ {bmsg}")
            self.qualityLabel.setText("Fix lighting before measuring")
            self.qualityLabel.setStyleSheet("color:#BF616A;")
            return

        # Preprocess — AHE only; per-frame normSkin is intentionally skipped
        # because NORM_MINMAX equalises every frame independently, destroying
        # the tiny inter-frame RGB changes that CHROM and POS depend on.
        frame     = self.applyAHE(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion gate
        if self.detectMotion(gray):
            self.hrLabel.setText("⚠ Motion — hold still")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")
            return

        # Face detection
        faces = self.faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(faces) == 0:
            self.hrLabel.setText("No face detected")
            self.qualityLabel.setText("Point camera at your face")
            return

        x, y, w, h = faces[0]

        # Age/gender — pass face dimensions for heuristic fallback
        fc = cv2.resize(rgb_frame[y:y+h, x:x+w], (227,227))
        age, gender = self.predictAgeGender(fc, face_w=w, face_h=h)

        # Multi-ROI
        roi_rgb, forehead = self.extractROI(rgb_frame, x, y, w, h)
        self.rgb_buf[self.buf_idx] = roi_rgb

        # Gauss level for Eulerian
        fh_r = cv2.resize(forehead, (self.vidW, self.vidH)) \
               if forehead.size > 0 else np.zeros((self.vidH, self.vidW, 3))
        gl   = self.buildGauss(fh_r.astype(np.float32), self.levels+1)[self.levels]
        self.gauss_buf[self.buf_idx] = gl

        self.buf_idx       = (self.buf_idx + 1) % self.BUF
        self.frame_count  += 1
        self.filled_frames = min(self.filled_frames + 1, self.BUF)

        # ── SHOW WARM-UP PROGRESS ────────────────────────────
        if self.frame_count < self.MIN_FRAMES:
            pct = int(100 * self.frame_count / self.MIN_FRAMES)
            self.hrLabel.setText(f"Collecting signal… {pct}%")
            self.qualityLabel.setText(
                f"{self.MIN_FRAMES - self.frame_count} frames to go")
            self.qualityLabel.setStyleSheet("color:#EBCB8B;")

        # ── FUSED CALCULATION ────────────────────────────────
        elif self.frame_count % self.CALC_EVERY == 0:
            # Roll buffers so oldest frame is first
            rgb_ord   = np.roll(self.rgb_buf,   -self.buf_idx, axis=0)
            gauss_ord = np.roll(self.gauss_buf,  -self.buf_idx, axis=0)
            mask_ord  = np.roll(self.freq_mask4, -self.buf_idx, axis=0)

            raw_bpm, avg_snr, breakdown = fused_estimate(
                rgb_ord, gauss_ord, self.fps,
                np.roll(self.freq_mask, -self.buf_idx),
                self.filled_frames)

            if raw_bpm is not None and 45 <= raw_bpm <= 150:
                if not self.isOutlier(raw_bpm):
                    self.bpm_history.append(raw_bpm)
                    kbpm   = self.kalman.update(raw_bpm)
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
                else:
                    self.qualityLabel.setText(
                        f"Outlier {raw_bpm:.0f} bpm rejected — stabilising…")
                    self.kalman.predict()
            else:
                self.qualityLabel.setText(
                    "Signal weak — stay still, ensure good lighting")
                self.qualityLabel.setStyleSheet("color:#BF616A;")
                self.kalman.predict()

        # ── EULERIAN AMPLIFICATION OVERLAY ──────────────────
        try:
            fft_cube = np.fft.fft(self.gauss_buf, axis=0)
            fft_cube[~self.freq_mask] = 0
            filtered   = np.real(np.fft.ifft(fft_cube, axis=0)) * self.alpha
            filt_frame = self.reconstructFrame(filtered, self.buf_idx - 1, self.levels)
            fh_amp     = cv2.convertScaleAbs(fh_r + filt_frame)
            ov_h = max(1, int(0.25 * h))
            rgb_frame[y : y+ov_h, x : x+w] = cv2.resize(fh_amp, (w, ov_h))
        except Exception:
            pass

        # Draw ROI boxes
        ov_h = max(1, int(0.25 * h))
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+ov_h), (0,255,0), 2)
        cy1, cy2 = y+int(0.45*h), y+int(0.75*h)
        cv2.rectangle(rgb_frame, (x, cy1),            (x+int(0.45*w), cy2), (255,165,0), 1)
        cv2.rectangle(rgb_frame, (x+int(0.55*w), cy1),(x+w, cy2),           (255,165,0), 1)

        # Display
        disp = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h_, w_, ch = disp.shape
        qImg = QImage(disp.data, w_, h_, ch*w_, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        # Pulse waveform
        gm = float(np.mean(forehead[:,:,1])) if forehead.size > 0 else 0.0
        self.pulse_buf[self.pulse_idx] = gm
        self.pulse_idx = (self.pulse_idx + 1) % 300
        self.pulseCurve.setData(self.pulse_buf)

    # ──────────────────────────────────────────────────────────
    #  CLOSE
    # ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.webcam.release()
        self.csv_file.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    monitor = HeartRateMonitor()
    monitor.show()
    sys.exit(app.exec_())
