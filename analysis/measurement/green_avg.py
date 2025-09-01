from collections import deque
import numpy as np
from utils.estimate_bpm import estimate_bpm
from utils.roi import get_roi
from utils.video_io import read_video

WINDOW_SIZE = 10.0  # seconds
ACQUISITION_TIME = 5.0  # seconds


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
        bpm = estimate_bpm(sig, fps)

        # Append timestamp and BPM to results
        if bpm is not None:
            ts = i * (1 / fps)
            timestamps.append(ts)
            bpm_series.append(bpm)

    return np.column_stack([timestamps, bpm_series])
