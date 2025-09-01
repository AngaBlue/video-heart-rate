from collections import deque
import numpy as np
from utils.estimate_bpm import estimate_bpm
from utils.roi import get_roi
from utils.video_io import read_video

WINDOW_SIZE = 10.0  # seconds
ACQUISITION_TIME = 5.0  # seconds


def _whiten(X: np.ndarray):
    """
    Whitens X (T x D). Returns X_white (T x D).
    """
    # SVD-based whitening for numerical stability
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Avoid division by zero for tiny singular values
    eps = 1e-12
    S_inv = 1.0 / np.sqrt(S**2 / (X.shape[0] - 1) + eps)
    X_white = U * S_inv
    X_white = X_white @ Vt

    return X_white


def _sym_decorrelate(W: np.ndarray) -> np.ndarray:
    """
    Symmetric decorrelation to keep W orthogonal.
    """
    # W W^T = E De E^T  => W <- (E De^{-1/2} E^T) W
    S = W @ W.T
    E, D, Et = np.linalg.svd(S, full_matrices=False)
    W = (E @ np.diag(1.0 / np.sqrt(D + 1e-12)) @ Et) @ W
    return W


def _fastica(X: np.ndarray, n_components: int | None = None, max_iter: int = 200, tol: float = 1e-5):
    """
    FastICA (symmetric) with tanh nonlinearity.
    X: (T x D). Returns sources S: (T x K), mixing A: (D x K), unmixing W: (K x D).
    """
    T, D = X.shape
    if n_components is None:
        n_components = D
    n_components = min(n_components, D)

    Xw = _whiten(X)

    # Initialize unmixing with random orthogonal rows
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_components, D))
    W = _sym_decorrelate(W)

    for _ in range(max_iter):
        WX = Xw @ W.T  # (T x K)
        # tanh nonlinearity
        g = np.tanh(WX)
        g_prime = 1.0 - g**2  # derivative
        # Update (symmetric)
        W_new = (g.T @ Xw) / T - (np.mean(g_prime, axis=0)[:, None] * W)
        W_new = _sym_decorrelate(W_new)

        # Convergence check (max absolute cosine between old/new rows)
        lim = np.max(np.abs(np.sum(W_new * W, axis=1)))
        W = W_new
        if 1.0 - lim < tol:
            break

    # Estimated sources in whitened space
    S_white = Xw @ W.T  # (T x K)

    return S_white


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

    for i, roi in enumerate(get_roi(frames, fps)):
        # Append BGR spatial average
        bgr_val = np.mean(roi, axis=(0, 1))
        bgr.append(bgr_val)

        # Compute BPM after acquisition time
        if len(bgr) <= acquisition_len:
            continue

        # De-trend mean
        signal = np.asarray(bgr, dtype=np.float32)
        signal = signal / np.mean(signal, axis=0) - 1

        # Standardise channels to unit variance for stability
        std_vals = np.std(signal, axis=0, ddof=1)
        std_vals[std_vals == 0] = 1.0
        signal = signal / std_vals

        # Estimate BPM from current window
        signals = _fastica(signal.astype(np.float64),
                           n_components=3, max_iter=300, tol=1e-6)
        bpm = estimate_bpm(signals, fs=fps)

        # Append timestamp and BPM to results
        if bpm is not None:
            ts = i * (1 / fps)
            timestamps.append(ts)
            bpm_series.append(bpm)

    return np.column_stack([timestamps, bpm_series])
