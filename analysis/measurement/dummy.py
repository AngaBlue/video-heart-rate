import numpy as np


def measure(video_path: str) -> np.ndarray:
    """
    Dummy HR measurement: returns a synthetic sine wave with noise.

    Args:
        video_path: Path to the input video (ignored in dummy).

    Returns:
        np.ndarray: Simulated heart rate signal over time.
    """
    # Simulate 30 seconds of HR data sampled at 30 Hz (900 samples)
    fs = 30
    duration = 30
    t = np.linspace(0, duration, duration * fs)

    # Base heart rate ~1.2 Hz (~72 BPM), plus noise
    hr_signal = 72 + 5 * np.sin(2 * np.pi * 1.2 * t) + \
        np.random.normal(0, 1.5, t.shape)
    return hr_signal
