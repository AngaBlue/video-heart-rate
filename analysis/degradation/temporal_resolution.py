import os
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video, write_video

# Target frame rates to downsample to
TARGET_FPS = [60, 30, 25, 15, 10, 5]


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies temporal resolution degradation by reducing frame rate via frame dropping.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / \
        "degraded" / "temporal_resolution"
    os.makedirs(output_root, exist_ok=True)

    frames, fps = read_video(input_path)

    # Include original video as control
    label = f"{int(fps)}fps"
    out_path = output_root / f"{label}.mp4"
    if not out_path.exists():
        write_video(frames, str(out_path), fps)
    yield str(out_path), label

    for target_fps in filter(lambda x: x < fps, TARGET_FPS):
        label = f"{target_fps}fps"
        out_path = output_root / f"{label}.mp4"

        # Skip if file exists
        if out_path.exists():
            yield str(out_path), label
            continue

        # Calculate frame step for downsampling
        step = int(round(fps / target_fps))
        if step < 1:
            step = 1

        # Select frames to keep
        downsampled_frames = frames[::step]

        write_video(downsampled_frames, str(out_path), target_fps)
        yield str(out_path), label
