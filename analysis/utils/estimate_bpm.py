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


from typing import Optional
import numpy as np

def estimate_bpm(signal: np.ndarray, fs: float) -> Optional[float]:
    """
    Estimate BPM from a 1D or 2D signal.
      signal: (T,) or (T, C) time series (time along axis 0)
      fs: sampling rate (Hz)
    Returns:
      BPM as float, or None if not resolvable.
    """
    if signal is None:
        return None

    X = np.asarray(signal, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]  # (T, 1)
    elif X.ndim != 2:
        raise ValueError("signal must be 1D (T,) or 2D (T, C) with time along axis 0.")

    N = X.shape[0]
    if N < 8:
        return None

    # Bandpass filter (support both 2-arg and 5-arg helper signatures)
    Xf = _bandpass_butterworth(X, fs)

    if Xf.shape[0] < 8:
        return None

    # Safety against NaNs/Infs
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)

    # FFT along time
    fft_vals = np.fft.fft(Xf, axis=0)              # (N, C)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)          # (N,)

    # Positive frequencies only
    pos = freqs > 0
    if not np.any(pos):
        return None
    freqs_pos = freqs[pos]                          # (Npos,)
    mags = np.abs(fft_vals[pos, ...])               # (Npos, C)
    if mags.ndim == 1:
        mags = mags[:, None]

    # Physiological band mask
    band = (freqs_pos >= FREQ_LOW) & (freqs_pos <= FREQ_HIGH)
    if not np.any(band):
        return None

    band_mags = mags[band, :]                       # (Nband, C)
    if band_mags.size == 0:
        return None

    # Peak per channel, then choose best channel
    peak_idx_per_col = np.argmax(band_mags, axis=0)                 # (C,)
    peak_mag_per_col = band_mags[peak_idx_per_col, np.arange(band_mags.shape[1])]
    best_col = int(np.argmax(peak_mag_per_col))
    dom_freq = float(freqs_pos[band][peak_idx_per_col[best_col]])

    return dom_freq * 60.0

