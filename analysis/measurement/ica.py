import numpy as np
from utils.roi import get_roi
from utils.video_io import read_video


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

    for i, roi in enumerate(get_roi(frames, fps)):
        pass

    return np.column_stack([[], []])
