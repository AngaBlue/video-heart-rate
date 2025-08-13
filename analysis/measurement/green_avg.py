from collections import deque
from typing import Optional
import numpy as np
import scipy.signal as sp
from utils.roi import get_roi
from utils.video_io import read_video


FREQ_LOW = 40 / 60
FREQ_HIGH = 200 / 60
FILTER_ORDER = 2
WINDOW_SIZE = 10.0 # seconds
ACQUISITION_TIME = 5.0 # seconds


def _bandpass_butterworth(signal: np.ndarray, fs: float, freq_lo: float, freq_hi: float, order: int) -> np.ndarray:
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


def measure(video_path: str) -> np.ndarray:
    """
    Estimate heart rate (BPM) from cheek ROI green-channel using MediaPipe landmarks.

    Returns:
        np.ndarray of shape (N, 2):
            column 0: timestamp in seconds (per-frame, 0..(N-1)/fps)
            column 1: estimated BPM
    """
    # Read video
    frames, fps = read_video(video_path)

    # Rolling window for green signal
    window_len = int(WINDOW_SIZE * fps)
    acquisition_len = int(ACQUISITION_TIME * fps)
    green = deque(maxlen=window_len)
    
    # Results
    timestamps = []
    bpm_series = []

    for i, roi in enumerate(get_roi(frames, fps)):
        # Calculate green channel average
        green_val = float(np.mean(roi[:, :, 1]))
        green.append(green_val)

        # Compute BPM after acquisition time
        if len(green) <= acquisition_len:
            continue

        # Detrend mean, bandpass butterworth, then estimate the bpm via fft
        sig = np.asarray(green, dtype=np.float32)
        sig = (sig - np.mean(sig))
        filt = _bandpass_butterworth(sig, fps, FREQ_LOW, FREQ_HIGH, FILTER_ORDER)
        bpm = _estimate_bpm(filt, fps)

        # Append timestamp and BPM to results
        if bpm is not None:
            ts = i * (1 / fps)
            timestamps.append(ts)
            bpm_series.append(bpm)

    return np.column_stack([timestamps, bpm_series])
