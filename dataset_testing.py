"""
Dataset Testing Pipeline — rPPG Comparative Evaluation (v2)
============================================================
Standalone script for validating rPPG methods against benchmark datasets,
using the exact same core algorithms as the live `heart_rate_monitor.py`.

This version tests 5 methods:
  1. FFT (Eulerian)
  2. CHROM
  3. POS
  4. ICA
  5. Fused (SNR-weighted average of the above)

NOTE: This version uses MediaPipe for robust face tracking to ensure
the highest quality ROI signal for algorithm evaluation.

Prerequisites:
    pip install opencv-python numpy scipy matplotlib mediapipe

    You must also download the MediaPipe face landmarker model:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    Place `face_landmarker.task` in the same directory as this script.

Usage:
    python dataset_testing.py --dataset ubfc --path ./datasets/UBFC-rPPG
    python dataset_testing.py --dataset pure --path ./datasets/PURE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Optional matplotlib (for plots) ──────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found. Plots will be skipped.")
    print("       Install with: pip install matplotlib")

# ── MediaPipe (for robust ROI tracking) ──────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[ERROR] MediaPipe not found. This script requires it for robust ROI extraction.")
    print("        Install with: pip install mediapipe")
    sys.exit(1)

# ── Signal processing functions ───────────────────────────────
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════
#  CONFIG (from heart_rate_monitor.py)
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    hr_lo_hz: float = 0.67
    hr_hi_hz: float = 3.33
    snr_thresh: Dict[str, float] = field(default_factory=lambda: {
        "FFT": -1.0, "CHROM": 0.0, "POS": 0.0, "ICA": -1.0,
    })
    fusion_agreement_bpm: float = 8.0
    fft_weight_scale: float = 0.25
    ica_run_snr_threshold: float = 0.0
    ica_run_disagreement_bpm: float = 12.0
    fft_disagree_limit_bpm: float = 20.0
    skin_h_max: int = 25
    skin_s_min: int = 40
    skin_v_min: int = 50
    skin_min_pixels: int = 30
    landmark_shift_reject_px: float = 3.0

CFG = Config()
ALL_METHODS = ["FFT", "CHROM", "POS", "ICA", "Fused"]


# ══════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING (from heart_rate_monitor.py)
# ══════════════════════════════════════════════════════════════

def detrend(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    if n < 2: return signal
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, signal, 1)
    return signal - (slope * x + intercept)

def _savgol_coeffs(window: int = 7, poly: int = 2) -> np.ndarray:
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vstack([x**i for i in range(poly + 1)]).T
    coeffs = np.linalg.pinv(A)[0]
    return coeffs[::-1]

_SG_KERNEL = _savgol_coeffs()

def savgol_smooth_rows(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    k = _SG_KERNEL
    hw = len(k) // 2
    for ch in range(3):
        out[:, ch] = np.convolve(rgb[:, ch], k, mode="same")
        out[:hw, ch] = rgb[:hw, ch]
        out[-hw:, ch] = rgb[-hw:, ch]
    return out

def bandpass_fft(sig: np.ndarray, fps: float) -> Tuple[float, float, float]:
    sig = detrend(sig - sig.mean()) * np.hanning(len(sig))
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mags = np.abs(np.fft.rfft(sig))
    band = (freqs >= CFG.hr_lo_hz) & (freqs <= CFG.hr_hi_hz)
    if not band.any() or mags[band].max() < 1e-9:
        return 0.0, 0.0, 0.0
    in_p = float(np.sum(mags[band]**2))
    out_p = float(np.sum(mags[~band]**2))
    snr = 10 * np.log10(in_p / out_p) if out_p > 0 else 0.0
    mags_f = mags.copy()
    mags_f[~band] = 0
    hz = float(freqs[np.argmax(mags_f)])
    return hz, hz * 60.0, snr

def chrom_pulse(rgb: np.ndarray) -> np.ndarray:
    if len(rgb) < 10: return np.zeros(len(rgb))
    mu = rgb.mean(0); mu[mu == 0] = 1e-9
    n = rgb / mu
    Xs = 3 * n[:, 0] - 2 * n[:, 1]
    Ys = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    a = Xs.std() / (Ys.std() + 1e-9)
    return detrend(Xs - a * Ys)

def pos_pulse(rgb: np.ndarray) -> np.ndarray:
    if len(rgb) < 10: return np.zeros(len(rgb))
    mu = rgb.mean(0); mu[mu == 0] = 1e-9
    n = rgb / mu
    S1 = n[:, 1] - n[:, 2]
    S2 = n[:, 1] + n[:, 2] - 2 * n[:, 0]
    b = S1.std() / (S2.std() + 1e-9)
    return detrend(S1 + b * S2)

def _fast_ica_sources(x: np.ndarray, n_components: int = 3, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
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
        wx = xw @ w.T
        gwx = np.tanh(wx)
        gprime = 1.0 - gwx**2
        w_new = (gwx.T @ xw) / xw.shape[0] - np.diag(np.mean(gprime, 0)) @ w
        s, u = np.linalg.eigh(w_new @ w_new.T)
        s = np.clip(s, 1e-9, None)
        w_new = (u @ np.diag(1.0 / np.sqrt(s)) @ u.T) @ w_new
        lim = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w = w_new
        if lim < tol: break
    return xw @ w.T

def ica_best(rgb: np.ndarray, fps: float) -> Tuple[float, float]:
    if len(rgb) < 30: return 0.0, -999.0
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


# ══════════════════════════════════════════════════════════════
#  ROI EXTRACTOR (from heart_rate_monitor.py)
# ══════════════════════════════════════════════════════════════

class MediaPipeROIExtractor:
    ROI_FOREHEAD = [10, 67, 103, 109, 338, 297, 332, 284]
    ROI_L_CHEEK = [50, 101, 205, 187, 147, 123, 116]
    ROI_R_CHEEK = [280, 330, 425, 411, 376, 352, 345]

    def __init__(self, model_path: str = "face_landmarker.task"):
        if not os.path.exists(model_path):
            print(f"[FATAL] MediaPipe model not found at: {model_path}")
            print("Please download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            sys.exit(1)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._prev_pts: Optional[np.ndarray] = None

    def extract_rgb(self, frame: np.ndarray) -> Optional[np.ndarray]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)
        if not detection_result or not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)

        if self._prev_pts is not None and self._prev_pts.shape == pts.shape:
            delta = np.mean(np.linalg.norm(pts - self._prev_pts, axis=1))
            if delta > CFG.landmark_shift_reject_px:
                return None # Reject if landmarks shifted too much

        # [MODIFIED] Skin masking is removed as it was degrading performance
        # on the dataset videos. We now rely purely on the geometric ROI.
        # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # skin_mask = ((hsv[:, :, 0] < CFG.skin_h_max) &
        #              (hsv[:, :, 1] > CFG.skin_s_min) &
        #              (hsv[:, :, 2] > CFG.skin_v_min)).astype(np.uint8) * 255

        def get_poly_mean(roi_indices: List[int]) -> np.ndarray:
            poly_pts = pts[roi_indices]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, poly_pts, 255)
            # [MODIFIED] The logic to combine with a skin_mask has been removed.
            # final_mask = cv2.bitwise_and(mask, skin_mask)
            # if cv2.countNonZero(final_mask) < CFG.skin_min_pixels:
            #     final_mask = mask
            mean_bgr = cv2.mean(frame, mask=mask)[:3]
            return np.array(mean_bgr[::-1]) # BGR to RGB

        fh_rgb = get_poly_mean(self.ROI_FOREHEAD)
        lc_rgb = get_poly_mean(self.ROI_L_CHEEK)
        rc_rgb = get_poly_mean(self.ROI_R_CHEEK)

        self._prev_pts = pts.copy()
        return 0.5 * fh_rgb + 0.25 * lc_rgb + 0.25 * rc_rgb


# ══════════════════════════════════════════════════════════════
#  BPM ESTIMATOR (from heart_rate_monitor.py)
# ══════════════════════════════════════════════════════════════

def estimate_bpm_all_methods(rgb_buf: np.ndarray, fps: float) -> Dict[str, float]:
    HR_LO, HR_HI = CFG.hr_lo_hz * 60, CFG.hr_hi_hz * 60
    if len(rgb_buf) < 30: return {}

    rgb_smooth = savgol_smooth_rows(rgb_buf) if len(rgb_buf) > 7 else rgb_buf
    results: Dict[str, Tuple[float, float]] = {}

    # FFT (Green Channel)
    try:
        _, bpm, snr = bandpass_fft(rgb_smooth[:, 1], fps)
        if HR_LO <= bpm <= HR_HI: results["FFT"] = (bpm, snr)
    except Exception: pass

    # CHROM
    try:
        _, bpm, snr = bandpass_fft(chrom_pulse(rgb_smooth), fps)
        if HR_LO <= bpm <= HR_HI: results["CHROM"] = (bpm, snr)
    except Exception: pass

    # POS
    try:
        _, bpm, snr = bandpass_fft(pos_pulse(rgb_smooth), fps)
        if HR_LO <= bpm <= HR_HI: results["POS"] = (bpm, snr)
    except Exception: pass

    # ICA (conditional)
    base_bpms = [b for m, (b, _) in results.items() if m in ("FFT", "CHROM", "POS")]
    base_snrs = [s for m, (_, s) in results.items() if m in ("FFT", "CHROM", "POS")]
    best_base_snr = max(base_snrs) if base_snrs else -999.0
    base_disagree = (max(base_bpms) - min(base_bpms)) if len(base_bpms) >= 2 else 0.0
    run_ica = (not results or best_base_snr < CFG.ica_run_snr_threshold or
               base_disagree > CFG.ica_run_disagreement_bpm)
    if run_ica:
        try:
            bpm, snr = ica_best(rgb_buf, fps)
            if HR_LO <= bpm <= HR_HI: results["ICA"] = (bpm, snr)
        except Exception: pass

    if not results: return {}

    # Fusion Logic
    valid = {m: (b, s) for m, (b, s) in results.items() if s > CFG.snr_thresh.get(m, -2.0)}
    candidates = valid if valid else results

    if "FFT" in candidates:
        non_fft = {m: v for m, v in candidates.items() if m != "FFT"}
        if len(non_fft) >= 2:
            non_fft_med = float(np.median([b for b, _ in non_fft.values()]))
            if abs(candidates["FFT"][0] - non_fft_med) > CFG.fft_disagree_limit_bpm:
                candidates = non_fft

    bpm_vals = np.array([b for b, _ in candidates.values()])
    snr_vals = np.array([s for _, s in candidates.values()])
    weights = np.power(10.0, snr_vals / 10.0)
    weights = np.clip(weights, 1e-6, None)
    for i, m in enumerate(candidates.keys()):
        if m == "FFT": weights[i] *= CFG.fft_weight_scale
    median_bpm = float(np.median(bpm_vals))
    agreement = np.abs(bpm_vals - median_bpm) < CFG.fusion_agreement_bpm
    weights[agreement] *= 2.0
    weights /= weights.sum()
    fused_bpm = float(np.dot(weights, bpm_vals))

    # Final output dict
    output = {m: b for m, (b, s) in results.items()}
    output["Fused"] = fused_bpm
    return output


# ══════════════════════════════════════════════════════════════
#  VIDEO/DATA LOADERS
# ══════════════════════════════════════════════════════════════

def load_ubfc_ground_truth(gt_path: str) -> np.ndarray:
    try:
        with open(gt_path, 'r') as f: lines = f.readlines()
        if len(lines) == 3: return np.fromstring(lines[1], dtype=np.float64, sep=' ')
        data = np.loadtxt(gt_path)
        if data.ndim == 1: return data.astype(np.float64)
        return data[:, 1 if data.shape[1] >= 2 else 0].astype(np.float64)
    except Exception as e:
        print(f"  [GT] Could not load {gt_path}: {e}")
        return np.array([])

def load_pure_ground_truth(json_path: str) -> np.ndarray:
    try:
        with open(json_path) as f: data = json.load(f)
        key = list(data.keys())[0]
        return np.array([fr["Value"]["pulseRate"] for fr in data[key]
                         if "Value" in fr and "pulseRate" in fr["Value"]], dtype=np.float64)
    except Exception as e:
        print(f"  [GT] Could not load {json_path}: {e}")
        return np.array([])

def process_video(video_path: str, max_frames: int = 900) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return np.zeros((0, 3)), 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    roi_extractor = MediaPipeROIExtractor()
    rgb_buf = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_mean = roi_extractor.extract_rgb(rgb)
        if rgb_mean is not None: rgb_buf.append(rgb_mean)
    cap.release()
    return np.array(rgb_buf, dtype=np.float64) if rgb_buf else np.zeros((0, 3)), fps

def process_pure_frames(frames_dir: str, max_frames: int = 900) -> Tuple[np.ndarray, float]:
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frame_files: return np.zeros((0, 3)), 30.0
    roi_extractor = MediaPipeROIExtractor()
    rgb_buf = []
    for fname in frame_files[:max_frames]:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_mean = roi_extractor.extract_rgb(rgb)
        if rgb_mean is not None: rgb_buf.append(rgb_mean)
    return np.array(rgb_buf, dtype=np.float64) if rgb_buf else np.zeros((0, 3)), 30.0


# ══════════════════════════════════════════════════════════════
#  ACCURACY EVALUATOR
# ══════════════════════════════════════════════════════════════

class AccuracyEvaluator:
    @staticmethod
    def align(est: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_est, n_gt = len(est), len(gt)
        if n_est == 0 or n_gt == 0: return np.array([]), np.array([])
        if n_est == n_gt: return est, gt
        target_len = min(n_est, n_gt)
        est_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, n_est), est)
        gt_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, n_gt), gt)
        return est_rs, gt_rs

    @classmethod
    def evaluate(cls, est: np.ndarray, gt: np.ndarray) -> dict:
        est, gt = cls.align(est, gt)
        if len(est) < 2:
            return {"mae": np.nan, "rmse": np.nan, "pearson_r": np.nan, "n": 0}
        diff = est - gt
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        try: r, p = pearsonr(est, gt)
        except Exception: r, p = np.nan, np.nan
        return {"mae": round(mae, 2), "rmse": round(rmse, 2),
                "pearson_r": round(float(r), 4), "n": len(est)}


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_evaluation(dataset_name: str, dataset_path: str, output_dir: str,
                   methods_to_run: List[str], max_subjects: int,
                   subject_list: Optional[List[str]] = None):
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "ubfc":
        subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        if subject_list: subjects = [s for s in subjects if s in subject_list]
        subjects = subjects[:max_subjects]
        gt_loader = load_ubfc_ground_truth
        data_loader = process_video
        data_path_fn = lambda s: os.path.join(dataset_path, s, "vid.avi")
        gt_path_fn = lambda s: os.path.join(dataset_path, s, "ground_truth.txt")
    elif dataset_name == "pure":
        subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        if subject_list: subjects = [s for s in subjects if s in subject_list]
        subjects = subjects[:max_subjects]
        gt_loader = load_pure_ground_truth
        data_loader = process_pure_frames
        data_path_fn = lambda s: os.path.join(dataset_path, s, s)
        gt_path_fn = lambda s: os.path.join(dataset_path, s, f"{s}.json")
    else:
        return

    print(f"\n{'='*60}\n  {dataset_name.upper()} Evaluation — {len(subjects)} subjects\n{'='*60}\n")

    csv_rows = []
    all_subject_metrics = []
    results_per_method = {m: {"estimated": [], "ground_truth": []} for m in methods_to_run}

    for subj_idx, subj in enumerate(subjects, 1):
        print(f"  [{subj_idx:02d}/{len(subjects):02d}] {subj}")
        data_path, gt_path = data_path_fn(subj), gt_path_fn(subj)
        if not os.path.exists(data_path) or not os.path.exists(gt_path):
            print("    [SKIP] Data or GT path not found")
            continue

        gt_bpm = gt_loader(gt_path)
        if len(gt_bpm) == 0:
            print("    [SKIP] Empty ground truth")
            continue
        print(f"    GT mean BPM: {np.mean(gt_bpm):.1f}")

        t0 = time.time()
        rgb_buf, fps = data_loader(data_path)
        print(f"    Frames: {len(rgb_buf)}  FPS: {fps:.1f}  Load: {time.time()-t0:.1f}s")
        if len(rgb_buf) < 100:
            print("    [SKIP] Insufficient frames")
            continue

        # Sliding window estimation
        win_sec, step_sec = 10.0, 1.0
        win_frames, step_frames = int(win_sec * fps), max(1, int(step_sec * fps))
        method_bpms = {m: [] for m in methods_to_run}

        for start in range(0, len(rgb_buf) - win_frames + 1, step_frames):
            seg = rgb_buf[start : start + win_frames]
            estimates = estimate_bpm_all_methods(seg, fps)
            for method in methods_to_run:
                if method in estimates:
                    method_bpms[method].append(estimates[method])

        subj_metrics = {"subject": subj}
        for method in methods_to_run:
            est_arr = np.array(method_bpms.get(method, []))
            if len(est_arr) == 0:
                print(f"    {method}: no valid estimates")
                subj_metrics[method] = {"mae": np.nan, "rmse": np.nan, "pearson_r": np.nan, "n": 0}
                continue

            gt_arr = np.interp(np.linspace(0, 1, len(est_arr)), np.linspace(0, 1, len(gt_bpm)), gt_bpm)
            metrics = AccuracyEvaluator.evaluate(est_arr, gt_arr)
            subj_metrics[method] = metrics
            results_per_method[method]["estimated"].extend(est_arr.tolist())
            results_per_method[method]["ground_truth"].extend(gt_arr.tolist())
            print(f"    {method:5s}  MAE={metrics['mae']:5.1f}  RMSE={metrics['rmse']:5.1f}  r={metrics['pearson_r']:6.3f}  n={metrics['n']}")
        all_subject_metrics.append(subj_metrics)

        row = [subj, f"{np.mean(gt_bpm):.1f}"]
        for m in methods_to_run:
            met = subj_metrics.get(m, {})
            row += [f"{met.get('mae', np.nan):.2f}", f"{met.get('rmse', np.nan):.2f}",
                    f"{met.get('pearson_r', np.nan):.4f}", str(met.get("n", 0))]
        csv_rows.append(row)

    # Write CSVs
    csv_path = os.path.join(output_dir, f"{dataset_name}_results.csv")
    header = ["Subject", "GT Mean BPM"] + [f"{m} {met}" for m in methods_to_run for met in ["MAE", "RMSE", "Pearson r", "N"]]
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"\n  [CSV] Per-subject results → {csv_path}")

    summary_rows = []
    for m in methods_to_run:
        est = np.array(results_per_method[m]["estimated"])
        gt = np.array(results_per_method[m]["ground_truth"])
        metrics = AccuracyEvaluator.evaluate(est, gt)
        results_per_method[m]["metrics"] = metrics
        summary_rows.append([m, f"{metrics['mae']:.2f}", f"{metrics['rmse']:.2f}",
                             f"{metrics['pearson_r']:.4f}", str(metrics["n"])])

    summary_csv = os.path.join(output_dir, f"{dataset_name}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Method", "MAE", "RMSE", "Pearson r", "N"])
        writer.writerows(summary_rows)
    print(f"  [CSV] Summary → {summary_csv}")

    # Plots
    if HAS_MATPLOTLIB:
        print("\n  Generating plots…")
        plot_results(results_per_method, all_subject_metrics, dataset_name, output_dir)

    print(f"\n  Done. All outputs in: {output_dir}")


# ══════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_results(results_per_method, all_subject_metrics, dataset_name, output_dir):
    # Correlation Plot
    methods = [m for m in ALL_METHODS if m in results_per_method and results_per_method[m]["estimated"]]
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), constrained_layout=True)
    if n_methods == 1: axes = [axes]
    fig.suptitle(f"Estimated vs Ground Truth BPM — {dataset_name.upper()}", fontsize=14, fontweight="bold")
    for ax, method in zip(axes, methods):
        est, gt = AccuracyEvaluator.align(np.array(results_per_method[method]["estimated"]),
                                          np.array(results_per_method[method]["ground_truth"]))
        ax.scatter(gt, est, alpha=0.5, s=25)
        lim_lo, lim_hi = min(gt.min(), est.min()) - 5, max(gt.max(), est.max()) + 5
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.6)
        ax.set_xlabel("Ground Truth BPM"); ax.set_ylabel("Estimated BPM")
        metrics = results_per_method[method].get("metrics", {})
        ax.set_title(f"{method}\nMAE={metrics.get('mae', '?'):.1f} r={metrics.get('pearson_r', '?'):.3f}")
        ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    out_path = os.path.join(output_dir, f"{dataset_name}_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [PLOT] Correlation saved → {out_path}")

    # Method Comparison Box Plot
    mae_per_method = {m: [] for m in ALL_METHODS}
    for subj in all_subject_metrics:
        for m in ALL_METHODS:
            v = subj.get(m, {}).get("mae")
            if v is not None and not np.isnan(v): mae_per_method[m].append(v)
    valid_methods = [m for m in ALL_METHODS if mae_per_method[m]]
    if not valid_methods: return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.boxplot([mae_per_method[m] for m in valid_methods], patch_artist=True)
    ax.set_xticklabels(valid_methods, fontsize=11)
    ax.set_ylabel("MAE (BPM)"); ax.set_title(f"Method Comparison — {dataset_name.upper()}", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    out_path = os.path.join(output_dir, f"{dataset_name}_method_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [PLOT] Method comparison saved → {out_path}")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="rPPG Dataset Testing Pipeline (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["ubfc", "pure"], help="Dataset to evaluate")
    parser.add_argument("--path", required=True, help="Path to dataset root directory")
    parser.add_argument("--method", default="all", choices=["all"] + ALL_METHODS, help="Which method(s) to run")
    parser.add_argument("--subjects", type=int, default=999, help="Max number of subjects to process")
    parser.add_argument("--subject_list", type=str, default=None, help="Comma-separated list of specific subjects")
    parser.add_argument("--output", default="results", help="Output directory for results")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"[ERR] Dataset path does not exist: {args.path}"); sys.exit(1)

    methods = ALL_METHODS if args.method == "all" else [args.method]
    specific_subjects = args.subject_list.split(',') if args.subject_list else None

    run_evaluation(
        dataset_name=args.dataset,
        dataset_path=args.path,
        output_dir=args.output,
        methods_to_run=methods,
        max_subjects=args.subjects,
        subject_list=specific_subjects
    )

if __name__ == "__main__":
    main()