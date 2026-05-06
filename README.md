# Facial Video-Based Heart Rate Monitor with HRV & Stress Analysis

**B.Tech Capstone Project 2026 | MIT-WPU School of Computer Engineering and Technology**

**Authors:** Harshal Kale, Snehanshu Phalle, Vinod Ravisundaram, Piyush Phaske  

## Overview

This project presents a real-time, non-contact heart rate monitor that uses a standard webcam to measure vital signs from facial video. It implements and fuses four distinct remote photoplethysmography (rPPG) algorithms to deliver a robust and accurate heart rate estimation.

Beyond simple BPM, the application provides clinical context by calculating **Heart Rate Variability (HRV)** metrics, estimating **physiological stress levels**, and generating comprehensive **PDF session reports**. The system is built entirely in Python and is optimized to run on standard consumer hardware without requiring a dedicated GPU.

## Key Features

- **Multi-Algorithm Fusion:** Combines four rPPG methods (FFT-Eulerian, CHROM, POS, ICA) using an SNR-weighted fusion strategy for superior accuracy and stability.
- **Advanced Signal Processing:** Employs Savitzky-Golay filtering, Butterworth band-pass filters, and Kalman filtering to enhance signal quality and reduce noise.
- **Robust ROI Tracking:** Uses MediaPipe FaceMesh for precise, multi-point facial landmark tracking (forehead and cheeks), ensuring a stable signal even with minor movements.
- **Clinical Insights:**
    - **HRV Analysis:** Calculates SDNN, RMSSD, and pNN50 from a 60-second sliding window.
    - **Stress Detection:** Provides a 4-level stress assessment (Relaxed, Normal, Mildly Stressed, Highly Stressed) based on RMSSD.
    - **BPM Alerts:** A 7-level alert system warns users of abnormal heart rates (e.g., Critical Low, Warning High).
- **Automated PDF Reporting:** Generates a detailed clinical session report with one click, summarizing BPM, HRV, stress levels, and method performance.
- **Real-Time GUI:** An intuitive PyQt5 interface displays live BPM, heart rate trends, pulse waveform, and all clinical metrics.
- **CPU-Optimized:** Designed for high performance on standard CPUs, achieving >15 FPS for real-time analysis.

## System Pipeline

The processing pipeline is as follows:

1.  **Frame Capture & Preprocessing:** Captures video from the webcam and applies CLAHE for contrast enhancement.
2.  **Face & Landmark Detection:** MediaPipe FaceMesh detects 468 facial landmarks.
3.  **ROI Extraction:** A weighted average of the RGB signal is extracted from the forehead and cheek regions.
4.  **Signal Processing:** The raw signal is detrended and smoothed.
5.  **rPPG Estimation:** Four algorithms (FFT, CHROM, POS, ICA) run in parallel to estimate BPM.
6.  **Fusion & Filtering:** Estimates are fused based on their Signal-to-Noise Ratio (SNR), and the final output is stabilized with a Kalman filter.
7.  **Clinical Analysis:** The BPM stream is used to calculate HRV and stress metrics.
8.  **Display & Reporting:** All data is displayed in the GUI and can be exported to a PDF report.

## Experimental Results

The algorithms were validated against the **UBFC-rPPG dataset** (42 subjects). The SNR-weighted fusion method demonstrated superior performance compared to any single algorithm.

| Method        | MAE (BPM) | RMSE (BPM) | Pearson r |
| :------------ | :-------- | :--------- | :-------- |
| FFT (Green)   | 8.11      | 10.92      | 0.891     |
| CHROM         | 5.43      | 7.88       | 0.943     |
| POS           | 6.02      | 8.51       | 0.930     |
| ICA           | 7.50      | 9.80       | 0.912     |
| **★ SNR Fusion** | **4.15**  | **5.97**   | **0.973** |

## Getting Started

### Prerequisites

- Python 3.8+
- A standard webcam

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    The application will automatically download the necessary model files (`.caffemodel`, `.task`) on the first run. A progress dialog will be shown.
    ```bash
    python heart_rate_monitor.py
    ```

### Usage

1.  **Start:** Click the "Start" button to begin monitoring. Ensure your face is well-lit and centered in the camera view.
2.  **Stop:** Click "Stop" to end the session.
3.  **Export PDF:** After stopping, click "Export PDF Report" to save a detailed summary of the session.

## Dataset Link
https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk

## Dataset Testing

A standalone script is provided to validate the rPPG algorithms against benchmark datasets like UBFC-rPPG and PURE.

```bash
# Example for UBFC dataset
python dataset_testing.py --dataset ubfc --path ./path/to/UBFC-rPPG/dataset

# Example for PURE dataset
python dataset_testing.py --dataset pure --path ./path/to/PURE
```

Results, including CSV files and plots, will be saved to the `results/` directory.
