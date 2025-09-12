from math import floor
import os
import subprocess
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video

CRF_LEVELS = [25, 30, 35, 40, 45, 51]
GOP = 10  # seconds


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies temporal compression to the video by varying H264 CRF.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "encoded_crf"
    os.makedirs(output_root, exist_ok=True)

    # Read original framerate
    _, fps = read_video(input_path)
    gop = floor(GOP * fps)

    yield str(input_path), "original"

    # Generate degraded videos
    for crf in CRF_LEVELS:
        out_path = output_root / f"{crf}.mp4"
        if not out_path.exists():
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(crf),
                "-g", str(gop),
                # fixed B-frames (constant across outputs)
                "-bf", "4",
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]
            subprocess.run(cmd, check=True)
        yield str(out_path), str(crf)
