import cv2
from typing import List, Tuple
import os
import pandas as pd
import numpy as np


def read_video(video_path: str) -> Tuple[List, float]:
    """
    Reads a video file and returns its frames and frame rate.

    Returns:
        frames: List of BGR frames (as numpy arrays)
        fps: Frames per second
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def write_video(frames: List, output_path: str, fps: float) -> None:
    """
    Writes a list of BGR frames to a video file (MP4).

    Params:
        frames: List of numpy arrays (frames)
        output_path: Output file path
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to write.")

    height, width = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # pyright: ignore[reportAttributeAccessIssue] # Use MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def read_truth_for_video(truth_path: str) -> pd.DataFrame:
    df = pd.read_csv(truth_path)
    if not {'timestamp', 'heart_rate'}.issubset(df.columns):
        raise ValueError(
            "Ground truth data must have columns ['timestamp', 'heart_rate'].")

    # Clean & sort truth
    df = (
        df[['timestamp', 'heart_rate']]
        .dropna(subset=['timestamp', 'heart_rate'])
        .drop_duplicates(subset=['timestamp'])
        .sort_values('timestamp')
    )
    if df.empty:
        raise ValueError(
            "Ground truth data has no valid timestamp/heart_rate rows.")

    return df


def interpolate_hr_to_frames(truth: pd.DataFrame, measured: np.ndarray) -> np.ndarray:
    """
    Interpolate ground-truth heart rate to the timestamps of `measured`.

    Args:
        truth: DataFrame with columns ['timestamp', 'heart_rate'] in seconds.
        measured: Either a 1D array of timestamps, or a 2D array whose first column
                  contains timestamps. The return will be [timestamp, hr] with the same
                  number of rows as `measured`.

    Returns:
        np.ndarray of shape (N, 2): [timestamp, interpolated_heart_rate]
    """
    t_truth = truth['timestamp'].to_numpy(dtype=float)
    hr_truth = truth['heart_rate'].to_numpy(dtype=float)

    measured = np.asarray(measured)
    if measured.ndim != 2 or measured.shape[1] < 1:
        raise ValueError("`measured` must be 2D with timestamps in column 0.")
    t_meas = measured[:, 0].astype(float)

    # For each t_meas, pick the last truth time <= t_meas; clamp before-first to index 0
    idx = np.searchsorted(t_truth, t_meas, side='right') - 1
    idx = np.clip(idx, 0, len(t_truth) - 1)

    assigned_hr = hr_truth[idx]
    return np.column_stack([t_meas, assigned_hr])
