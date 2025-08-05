from pathlib import Path
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


def read_truth_for_video(video_path: str, fps: float, num_frames: int) -> np.ndarray:
    csv_path = Path(video_path).with_name(Path(video_path).stem + "_hr.csv")
    df = pd.read_csv(csv_path)
    if len(df) != num_frames:
        raise ValueError(
            "Mismatch between interpolated HR length and video frame count.")
    return df['heart_rate'].values


def interpolate_hr_to_frames(hr_data: pd.DataFrame, num_frames: int, fps: float) -> pd.DataFrame:
    """
    hr_data: DataFrame with columns ['timestamp', 'heart_rate'] in seconds
    num_frames: total number of frames in the video
    fps: frames per second
    Returns a new DataFrame with one row per frame.
    """
    # Target frame timestamps
    frame_times = np.arange(num_frames) / fps

    # Interpolate heart rate
    interpolated_hr = np.interp(
        frame_times, hr_data['timestamp'], hr_data['heart_rate'])

    # Build new DataFrame
    result = pd.DataFrame({
        'frame_number': np.arange(num_frames),
        'timestamp': frame_times,
        'heart_rate': interpolated_hr
    })

    return result
