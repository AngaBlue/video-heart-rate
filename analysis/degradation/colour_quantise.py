import os
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video, write_video

# Levels of colour quantisation (number of bits per channel)
COLOUR_DEPTHS = [6, 4, 2]  # 6-bit, 4-bit, 2-bit

def quantise_colour(frame: np.ndarray, bits: int) -> np.ndarray:
    """
    Reduces the number of bits per channel in a frame.

    Args:
        frame: Input RGB image as a NumPy array.
        bits: Target bits per channel (1-8).

    Returns:
        Quantised image.
    """
    levels = 2 ** bits
    scale = 256 // levels
    return (frame // scale) * scale


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies colour quantisation degradation at various bit levels.

    Yields:
        Tuple of (output_path, label) for each degraded version.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "degraded" / "colour_quantisation"
    os.makedirs(output_root, exist_ok=True)

    frames, fps = read_video(input_path)

    # Include original as control
    control_out_path = output_root / "original.mp4"
    if not control_out_path.exists():
        write_video(frames, str(control_out_path), fps)
    yield str(control_out_path), "original"

    for bits in COLOUR_DEPTHS:
        label = f"{bits}-bit"
        out_path = output_root / f"{label}.mp4"

        if out_path.exists():
            yield str(out_path), label
            continue

        degraded_frames = [quantise_colour(frame, bits) for frame in frames]
        write_video(degraded_frames, str(out_path), fps)
        yield str(out_path), label
