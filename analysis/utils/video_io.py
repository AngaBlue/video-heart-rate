import cv2
from typing import List, Tuple
import os


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def get_video_files(directory: str) -> List[str]:
    """
    Lists video files in a given directory with specified extensions.

    Returns:
        List of filenames (not full paths)
    """
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith('.mp4') and os.path.isfile(os.path.join(directory, f))
    ]
