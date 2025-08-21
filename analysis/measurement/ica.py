from collections import deque
import numpy as np
from utils.roi import get_roi
from utils.video_io import read_video

WINDOW_SIZE = 10.0  # seconds
ACQUISITION_TIME = 5.0  # seconds


def measure(video_path: str) -> np.ndarray:
    """
    Estimate heart rate (BPM) from cheek ROI green-channel using Independent Component Analysis (ICA).

    Returns:
        np.ndarray of shape (N, 2):
            column 0: timestamp in seconds (per-frame, 0..(N-1)/fps)
            column 1: estimated BPM
    """
    # Read video
    frames, fps = read_video(video_path)

    # Rolling window for bgr signals
    window_len = int(WINDOW_SIZE * fps)
    acquisition_len = int(ACQUISITION_TIME * fps)

    bgr = deque(maxlen=window_len)

    for i, roi in enumerate(get_roi(frames, fps)):
        # Append BGR average
        bgr_val = np.mean(roi, axis=(0, 1))
        bgr.append(bgr_val)

        # Compute BPM after acquisition time
        if len(bgr) <= acquisition_len:
            continue

        # De-trend mean
        signal = np.asarray(bgr, dtype=np.float32)
        signal = signal / np.mean(signal, axis=0) - 1

    return np.column_stack([[], []])
