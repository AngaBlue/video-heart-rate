from __future__ import annotations
import os
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

import cv2 as cv
import numpy as np
import scipy.signal as sp
import mediapipe as mp

from utils.video_io import read_video


# --------------------------
# Tunables (match your script)
# --------------------------
FREQ_LOW = 0.75   # Hz  (~45 BPM)
FREQ_HIGH = 1.66  # Hz  (~100 BPM)
FILTER_ORDER = 2
WINDOW_SEC = 10.0           # rolling window length in seconds
REUSE_LANDMARKS_FOR = 15    # if detection drops, reuse last landmarks for N frames

# Cheek ROI ratios inside face bbox (x1,y1,x2,y2)
CHEEK_HR = 0.15   # horizontal margin ratio
CHEEK_TOP = 0.40
CHEEK_BOT = 0.65

# --------------------------
# MediaPipe setup
# --------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def _locate_model() -> Path:
    """
    Try common places for face_landmarker.task:
    - env MEDIAPIPE_FACE_TASK
    - same directory as this file
    - project root next to main.py (one level up)
    """
    env = os.getenv("MEDIAPIPE_FACE_TASK")
    if env and Path(env).is_file():
        return Path(env)

    here = Path(__file__).resolve().parent
    cand = [
        here / "face_landmarker.task",
        here.parent / "face_landmarker.task",
        Path.cwd() / "face_landmarker.task",
    ]
    for p in cand:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find 'face_landmarker.task'. "
        "Set MEDIAPIPE_FACE_TASK or place the model next to this module."
    )

def _setup_face_landmarker(model_path: Optional[str] = None) -> FaceLandmarker:
    model = Path(model_path) if model_path else _locate_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    return FaceLandmarker.create_from_options(options)

# --------------------------
# Geometry & Signal helpers
# --------------------------
def _bbox_from_landmarks(landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmark],
                         w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(max(0, min(xs) * w))
    y1 = int(max(0, min(ys) * h))
    x2 = int(min(w - 1, max(xs) * w))
    y2 = int(min(h - 1, max(ys) * h))
    return x1, y1, x2, y2

def _cheek_roi_from_bbox(bb: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bb
    roi_y1 = int(np.clip(y1 + CHEEK_TOP * (y2 - y1), 0, h - 1))
    roi_y2 = int(np.clip(y1 + CHEEK_BOT * (y2 - y1), 0, h))
    roi_x1 = int(np.clip(x1 + CHEEK_HR * (x2 - x1), 0, w - 1))
    roi_x2 = int(np.clip(x2 - CHEEK_HR * (x2 - x1), 0, w))
    return roi_x1, roi_y1, roi_x2, roi_y2

def _bandpass_butterworth(signal: np.ndarray, fs: float,
                          freq_lo: float, freq_hi: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(1e-6, freq_lo / nyq)
    high = min(0.999, freq_hi / nyq)
    if high <= low:
        return signal
    sos = sp.butter(order, [low, high], btype="band", output="sos")
    return sp.sosfiltfilt(sos, signal, axis=0)

def _estimate_bpm(filtered: np.ndarray, fs: float) -> Optional[float]:
    if filtered.size < 8:
        return None
    fft_vals = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(filtered.shape[0], d=1.0 / fs)
    pos = freqs > 0
    freqs = freqs[pos]
    mags = np.abs(fft_vals[pos])

    band = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if not np.any(band):
        return None
    dom_freq = freqs[band][np.argmax(mags[band])]
    return float(dom_freq * 60.0)

# --------------------------
# Public API
# --------------------------
def measure(video_path: str) -> np.ndarray:
    """
    Estimate heart rate (BPM) from cheek ROI green-channel using MediaPipe landmarks.

    Returns:
        np.ndarray of shape (N, 2):
            column 0: timestamp in seconds (per-frame, 0..(N-1)/fps)
            column 1: estimated BPM (carry-forward after first valid estimate)
    """
    frames, fps = read_video(video_path)
    n = len(frames)
    if n == 0:
        return np.empty((0, 2), dtype=float)

    timestamps = np.arange(n, dtype=float) / float(fps)
    bpm_series = np.full(n, np.nan, dtype=float)

    # Rolling window for green signal
    window_len = max(1, int(WINDOW_SEC * fps))
    green = deque(maxlen=window_len)

    last_bpm: Optional[float] = None
    last_landmarks = None
    reuse_left = 0

    landmarker = _setup_face_landmarker()
    with landmarker:
        for i, bgr in enumerate(frames):
            h, w = bgr.shape[:2]
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(timestamps[i] * 1000.0)

            result = landmarker.detect_for_video(mp_image, ts_ms)
            if result and result.face_landmarks:
                last_landmarks = result.face_landmarks[0]
                reuse_left = REUSE_LANDMARKS_FOR
            elif last_landmarks is not None and reuse_left > 0:
                reuse_left -= 1
            else:
                # No landmarks available; keep previous BPM (if any) and continue
                bpm_series[i] = last_bpm if last_bpm is not None else np.nan
                continue

            # ROI from landmarks
            bb = _bbox_from_landmarks(last_landmarks, w, h)
            cx1, cy1, cx2, cy2 = _cheek_roi_from_bbox(bb, w, h)
            if cy2 <= cy1 or cx2 <= cx1:
                bpm_series[i] = last_bpm if last_bpm is not None else np.nan
                continue

            roi = bgr[cy1:cy2, cx1:cx2]
            green_val = float(np.mean(roi[:, :, 1]))
            green.append(green_val)

            # Compute BPM when window is sufficiently filled
            if len(green) >= max(64, int(3.0 * fps)):  # at least ~3 seconds of data
                sig = np.asarray(green, dtype=np.float32)
                sig = (sig - np.mean(sig))  # detrend mean
                filt = _bandpass_butterworth(sig, fps, FREQ_LOW, FREQ_HIGH, FILTER_ORDER)
                bpm = _estimate_bpm(filt, fps)
                if bpm is not None:
                    last_bpm = bpm

            # Carry-forward last BPM (after first valid estimate)
            bpm_series[i] = last_bpm if last_bpm is not None else np.nan

    # If we never got a BPM, fall back to NaNs → optional: fill with overall median?
    if np.all(np.isnan(bpm_series)):
        return np.column_stack([timestamps, bpm_series])

    # Forward-fill NaNs at the beginning (before first estimate) with first valid BPM
    first_valid = np.argmax(~np.isnan(bpm_series))
    if not np.isnan(bpm_series[first_valid]):
        bpm_series[:first_valid] = bpm_series[first_valid]
    # Also fill any intermittent NaNs by carrying forward
    for i in range(1, n):
        if np.isnan(bpm_series[i]):
            bpm_series[i] = bpm_series[i - 1]

    return np.column_stack([timestamps, bpm_series])
