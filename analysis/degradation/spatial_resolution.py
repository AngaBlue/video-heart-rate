import os
import subprocess
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video

TARGET_HEIGHTS = [720, 480, 360, 240]


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies spatial resolution degradation by scaling height and adjusting width
    proportionally.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / \
        "degraded" / "spatial_resolution"
    os.makedirs(output_root, exist_ok=True)

    # Read original frames and video metadata
    frames, _ = read_video(input_path)
    orig_height, orig_width = frames[0].shape[:2]
    aspect_ratio = orig_width / orig_height

    # Include original as control
    label = f"{orig_width}x{orig_height}"
    yield input_path, f"{orig_width}x{orig_height}"

    # Generate degraded resolutions
    for target_height in filter(lambda x: x < orig_height, TARGET_HEIGHTS):
        target_width = int(round(target_height * aspect_ratio))
        target_width -= target_width % 2
        label = f"{target_width}x{target_height}"
        out_path = output_root / f"{label}.mp4"

        if not out_path.exists():
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-c:v", "libx264",
                "-s", label,
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]
            subprocess.run(cmd, check=True)

        yield str(out_path), label
