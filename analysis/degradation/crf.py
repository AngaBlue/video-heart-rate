import os
import subprocess
from pathlib import Path
from typing import Generator, Tuple

CRF_LEVELS = [25, 30, 35, 40, 45, 51]


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies temporal compression to the video by varying H.264 CRF.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "degraded" / "crf"
    os.makedirs(output_root, exist_ok=True)

    yield str(input_path), "original"

    # Generate degraded videos
    for crf in CRF_LEVELS:
        out_path = output_root / f"{crf}.mp4"
        if not out_path.exists():
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]
            subprocess.run(cmd, check=True)
        yield str(out_path), str(crf)
