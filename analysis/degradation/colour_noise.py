import os
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video, write_video

# Noise standard deviations to test (in pixel value units)
NOISE_LEVELS = [5, 10, 20, 40]  # Reasonable range for visible degradation


def add_gaussian_noise(frame: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Adds Gaussian noise to a colour image frame.

    Args:
        frame: Input image (uint8).
        std_dev: Standard deviation of the Gaussian noise.

    Returns:
        Noisy image (uint8).
    """
    noise = np.random.normal(0, std_dev, frame.shape).astype(np.float32)
    noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255)
    return noisy_frame.astype(np.uint8)


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies Gaussian colour noise to the video at multiple levels.

    Yields:
        Tuple of (output_path, label) for each degraded version.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "degraded" / "colour_noise"
    os.makedirs(output_root, exist_ok=True)

    frames, fps = read_video(input_path)

    # Include original as control
    control_out_path = output_root / "original.mp4"
    if not control_out_path.exists():
        write_video(frames, str(control_out_path), fps)
    yield str(control_out_path), "original"

    for std_dev in NOISE_LEVELS:
        label = f"{int(std_dev)}std"
        out_path = output_root / f"{label}.mp4"

        if out_path.exists():
            yield str(out_path), label
            continue

        noisy_frames = [add_gaussian_noise(frame, std_dev) for frame in frames]
        write_video(noisy_frames, str(out_path), fps)
        yield str(out_path), label
