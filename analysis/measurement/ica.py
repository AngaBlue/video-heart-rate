from collections import deque
import numpy as np
from sklearn.decomposition import FastICA
from utils.estimate_bpm import estimate_bpm
from utils.roi import get_roi
from utils.video_io import read_video
import warnings
from sklearn.exceptions import ConvergenceWarning

WINDOW_SIZE = 10.0  # seconds
ACQUISITION_TIME = 5.0  # seconds


def measure(video_path: str) -> np.ndarray:
    """
    Estimate heart rate (BPM) from cheek ROI using Independent Component Analysis (ICA).

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

    # Results
    timestamps = []
    bpm_series = []

    # Prepare a single ICA instance we reuse on each window
    ica = FastICA(
        n_components=3,
        algorithm="parallel",
        fun="logcosh",
        max_iter=300,
        tol=1e-6,
        whiten="unit-variance",
        random_state=0,
    )

    for i, roi in enumerate(get_roi(frames, fps)):
        # Append BGR spatial average
        bgr_val = np.mean(roi, axis=(0, 1))
        bgr.append(bgr_val)

        # Compute BPM after acquisition time
        if len(bgr) < acquisition_len:
            continue

        # De-trend mean
        signal = np.asarray(bgr, dtype=np.float32)

        # Standardise channels to unit variance for stability
        std_vals = np.std(signal, axis=0, ddof=1)
        std_vals[std_vals == 0] = 1.0
        signal = signal / std_vals

        # Run ICA
        with warnings.catch_warnings(record=True) as w:
            sources = ica.fit_transform(signal)

            # Skip if the segment failed to converge
            if any(issubclass(wi.category, ConvergenceWarning) for wi in w):
                continue

        # Estimate BPM from the independent components
        bpm = estimate_bpm(sources, fs=fps)
        if bpm is not None:
            ts = i * (1 / fps)
            timestamps.append(ts)
            bpm_series.append(bpm)

    return np.column_stack([timestamps, bpm_series])
