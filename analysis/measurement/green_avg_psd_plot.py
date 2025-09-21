"""Estimate heart rate (BPM) from cheek ROI green-channel using MediaPipe. This
module uses the same pipeline as green_avg.py. The pipeline is:

    Video → Face Detection → Cheek ROI → BGR Mean → Green Channel →
    Z-score Normalization → Bandpass Butterworth Filter → FFT →
    Ideal Band Pass (40-200 BPM) → Dominant Frequency → BPM Estimate

Additional features:
    - Visualizes PSD and BPM comparison across different processing stages
    - Supports preloading and saving ROI mean data to skip MediaPipe processing
"""

from collections import deque
from typing import Optional
import numpy as np
import scipy.signal as sp
from pathlib import Path
import os

from utils.roi import get_roi
from utils.video_io import read_video
import utils.psd_plot as psd_plot


FREQ_LOW = 40 / 60
FREQ_HIGH = 200 / 60
FILTER_ORDER = 2
WINDOW_SIZE = 10.0 # seconds
ACQUISITION_TIME = 10.0 # seconds
LOAD_ROI_MEAN = False  # TODO: allow this to be set externally from CLI
ROI_DATA_DIR = "cache/roi_mean_data"


def _bandpass_butterworth(signal: np.ndarray, fs: float, freq_lo: float,
                          freq_hi: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(1e-6, freq_lo / nyq)
    high = min(0.999, freq_hi / nyq)
    if high <= low:
        return signal
    sos = sp.butter(order, [low, high], btype="band", output="sos")
    return sp.sosfiltfilt(sos, signal, axis=0)


def _estimate_bpm(filtered: np.ndarray, fs: float) -> Optional[float]:
    if filtered.size < 8:
        return np.nan
    fft_vals = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(filtered.shape[0], d=1.0 / fs)
    pos = freqs > 0
    freqs = freqs[pos]
    mags = np.abs(fft_vals[pos])**2 / (fs * len(filtered))

    band = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)

    if not np.any(band):
        return np.nan

    dom_freq = freqs[band][np.argmax(mags[band])]
    bpm = float(dom_freq * 60.0)
    psd_data = np.column_stack([freqs[band], mags[band]])

    return bpm, psd_data


def preload_signal(video_path: str, frames: list, fps: float
                   ) -> tuple[np.ndarray, str]:
    """Preload or create ROI mean signal and save it as a .npy file to
    ROI_DATA_DIR. This is used to speed up subsequent runs of measure() by
    avoiding mediapipe face landmark processing.
    Args:
        video_path: path to the input video file
        frames: list of video frames as numpy arrays
        fps: frames per second of the video
    Returns:
        signal: np.ndarray of shape (N, 3) containing BGR means per frame
        test_name: str, name of the test conducted (derived from video_path)"""
    # Create ROI mean data path
    os.makedirs(ROI_DATA_DIR, exist_ok=True)
    if Path(video_path).parent.name == "videos":
        signal_avg_path = f"{ROI_DATA_DIR}/{Path(video_path).stem}-original.npy"
    else:
        signal_avg_path = (
            f"{ROI_DATA_DIR}/{Path(video_path).parts[1]}-"
            f"{Path(video_path).parts[3]}-{Path(video_path).stem}.npy"
        )

    # Create ROI mean data if it doesn't exist
    if not os.path.exists(signal_avg_path):
        print(f"       ROI means data not yet created for {video_path}.")
        print("       Creating new data now...\n")
        signal = []
        
        # Extract signals from each frame
        for _, roi in enumerate(get_roi(frames, fps)):
            bgr_mean = np.mean(roi, axis=(0, 1))
            signal.append(bgr_mean)

        # Convert to numpy array
        signal = np.array(signal)

        # Save extracted BGR means
        np.save(signal_avg_path, signal)
        print(f"\n       BGR means saved to {signal_avg_path}")
    else:
        # Load previously saved BGR means
        signal = np.load(signal_avg_path)
        print(f"       Loaded pre-saved signal from {signal_avg_path}")
    
    # Extract the name of the test conducted
    # (eg. "subject1-temporal_resolution-25fps.npy")
    test_name = Path(signal_avg_path).stem
    return signal, test_name


def measure(video_path: str, load_roi_mean: bool = LOAD_ROI_MEAN) -> np.ndarray:
    """
    Estimate heart rate (BPM) from cheek ROI green-channel using MediaPipe 
    landmarks.

    Returns:
        np.ndarray of shape (N, 2):
            column 0: timestamp in seconds (per-frame, 0..(N-1)/fps)
            column 1: estimated BPM
    """
    # Read video
    frames, fps = read_video(video_path)

    # Rolling window for green signal
    window_len = int(round(WINDOW_SIZE * fps))
    acquisition_len = int(round(ACQUISITION_TIME * fps))
    green = deque(maxlen=window_len)
    
    # Results
    timestamps = []
    bpm_series = []
    signal_comparison_data = []

    # Conditionally preload signal or prepare for real-time extraction
    if load_roi_mean:
        # Preload or create ROI mean signal
        data_source, test_name = preload_signal(video_path, frames, fps)
    else:
        # Don't preload, use real-time ROI extraction
        data_source = get_roi(frames, fps)
        # Extract the name of the test conducted
        if Path(video_path).parent.name == "videos":
            test_name = f"{Path(video_path).stem}-original"
        else:
            test_name = (
                f"{Path(video_path).parts[1]}-{Path(video_path).parts[3]}-"
                f"{Path(video_path).stem}"
            )

    # Determine HR via a rolling window
    for i, data_val in enumerate(data_source):

        if load_roi_mean:
            # data_val is already bgr_mean from preloaded signal
            green.append(data_val[1])  # Extract green channel
        else:
            # data_val is ROI, need to compute mean
            bgr_mean = np.mean(data_val, axis=(0, 1))
            green.append(bgr_mean[1])  # Extract green channel

        # Compute BPM after acquisition time
        if len(green) < acquisition_len:
            timestamps.append(i * (1 / fps))
            bpm_series.append(np.nan)
            signal_comparison_data.append(np.nan)
            continue

        # ===================== Current bpm estimation ====================== #
        # Normalise mean, bandpass butterworth, then estimate the bpm via fft
        signal = np.asarray(green, dtype=np.float32)
        signal = (signal - np.mean(signal)) / np.std(signal)
        filtered = _bandpass_butterworth(signal, fps, FREQ_LOW, FREQ_HIGH, 
                                         FILTER_ORDER)
        bpm, _ = _estimate_bpm(filtered, fps)

        # Append timestamp and BPM to results
        timestamps.append(i * (1 / fps))
        bpm_series.append(bpm)
        # =================================================================== #

        # ============== bpm estimation for different filters =============== #
        # Create different filtering combinations
        green_np = np.asarray(green, dtype=np.float32)
        normalised = (green_np - np.mean(green_np)) / np.std(green_np)
        bp_bw = _bandpass_butterworth(green_np, fps, FREQ_LOW, 
                                    FREQ_HIGH, FILTER_ORDER)
        normalised_bw = _bandpass_butterworth(normalised, fps, FREQ_LOW, 
                                              FREQ_HIGH, FILTER_ORDER)

        # Estimate BPM from different filtering combinations
        bpm1, psd_data1 = _estimate_bpm(green_np, fps)
        bpm2, psd_data2 = _estimate_bpm(normalised, fps)
        bpm3, psd_data3 = _estimate_bpm(bp_bw, fps)
        bpm4, psd_data4 = _estimate_bpm(normalised_bw, fps)

        # Record result for the current frame and append to the list
        record = {
            'input':         {'bpm': bpm1, 'psd_data': psd_data1},
            'detrend':       {'bpm': bpm2, 'psd_data': psd_data2},
            'bp_bw':         {'bpm': bpm3, 'psd_data': psd_data3},
            'detrend-bp_bw': {'bpm': bpm4, 'psd_data': psd_data4}
        }
        signal_comparison_data.append(record)
        # =================================================================== #

    bpm_timeseries = np.column_stack([timestamps, bpm_series])

    # ====================== Display PSD and BPM plots ====================== #
    plotting_state = psd_plot.PlotState()
    for frame_idx in range(len(timestamps)):
        if not plotting_state.continue_plotting:
            break

        # Plot every 10 frames, unless skipping acquisition period plots
        if (frame_idx % 10 == 0 and not (plotting_state.skip_acquisition and 
                                         frame_idx < acquisition_len)):
            psd_plot.psd_plot(bpm_timeseries,
                              signal_comparison_data, test_name,
                              plotting_state, frame_idx, ACQUISITION_TIME)
    # ======================================================================= #

    return bpm_timeseries
