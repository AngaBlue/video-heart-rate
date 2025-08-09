import numpy as np
from utils.video_io import read_video

def measure(video_path: str) -> np.ndarray:
    """
    Dummy HR measurement: returns random heart rates aligned to video timestamps.

    Returns:
        np.ndarray of shape (N, 2): [timestamp (s), predicted_hr (BPM)]
    """
    frames, fps = read_video(video_path)
    n = len(frames)
    if n == 0:
        return np.empty((0, 2), dtype=float)

    # Timestamps for each frame (seconds)
    t = np.arange(n, dtype=float) / float(fps)

    # Random-but-plausible HR around 72 BPM
    hr = np.random.normal(loc=72.0, scale=3.0, size=n)

    return np.column_stack([t, hr.astype(float)])

