from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np
import scipy.signal as sp
from utils.roi import get_roi
from utils.video_io import read_video


FREQ_LOW = 40 / 60
FREQ_HIGH = 200 / 60
FILTER_ORDER = 2
WINDOW_SEC = 10.0


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

    timestamps = np.arange(n, dtype=float) / float(fps)
    bpm_series = np.full(n, np.nan, dtype=float)

    # Rolling window for green signal
    window_len = max(1, int(WINDOW_SEC * fps))
    green = deque(maxlen=window_len)
    last_bpm: Optional[float] = None

    for i, (roi, _ts_ms) in enumerate(get_roi(frames, fps)):
        green_val = float(np.mean(roi[:, :, 1]))
        green.append(green_val)

        # Compute BPM when window is sufficiently filled
        if len(green) >= max(64, int(3.0 * fps)):  # at least ~3 seconds of data
            sig = np.asarray(green, dtype=np.float32)
            sig = (sig - np.mean(sig))  # detrend mean
            filt = _bandpass_butterworth(
                sig, fps, FREQ_LOW, FREQ_HIGH, FILTER_ORDER)
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
