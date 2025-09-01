from typing import Optional
import numpy as np
import scipy.signal as sp

FREQ_LOW = 40 / 60
FREQ_HIGH = 200 / 60
FILTER_ORDER = 2


def _bandpass_butterworth(signal: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(1e-6, FREQ_LOW / nyq)
    high = min(0.999, FREQ_HIGH / nyq)
    if high <= low:
        return signal
    sos = sp.butter(FILTER_ORDER, [low, high], btype="band", output="sos")
    return sp.sosfiltfilt(sos, signal, axis=0)


def estimate_bpm(signal: np.ndarray, fs: float) -> Optional[float]:
    filtered = _bandpass_butterworth(signal, fs)

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
