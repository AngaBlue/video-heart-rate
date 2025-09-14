import subprocess
import os
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video

TARGET_FPS = [60, 30, 25, 15, 10, 5]


def _format_fps_label(fps: float) -> str:
    # Use integer label if it's effectively an integer (e.g., 30.0 → "30fps")
    if abs(fps - round(fps)) < 1e-3:
        return f"{int(round(fps))}fps"
    return f"{fps:.2f}fps"


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies temporal compression to the video by lowering the framerate.
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / \
        "degraded" / "temporal_resolution"
    os.makedirs(output_root, exist_ok=True)

    _, fps = read_video(input_path)
    yield str(input_path), _format_fps_label(fps)

    # Generate degraded videos
    for target_fps in filter(lambda x: x < fps, TARGET_FPS):
        if target_fps >= fps:
            continue

        label = _format_fps_label(target_fps)
        out_path = output_root / f"{label}.mp4"
        if not out_path.exists():
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-c:v", "libx264",
                "-r", str(target_fps),
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]
            subprocess.run(cmd, check=True)
        yield str(out_path), label
