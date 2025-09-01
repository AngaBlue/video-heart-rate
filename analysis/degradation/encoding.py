import os
import subprocess
from pathlib import Path
from typing import Generator, Tuple

# Define codecs to test
CODECS = {
    "mjpeg": {
        "ext": "avi",
        "args": ["-c:v", "mjpeg", "-q:v", "31", "-pix_fmt", "yuvj444p"]
    },
    "h264": {
        "ext": "mp4",
        "args": ["-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p"]
    },
    "ffv1": {
        "ext": "mkv",
        "args": ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
    }
}







def run_ffmpeg(input_path: str, output_path: str, codec_args: list) -> None:
    """
    Runs ffmpeg to encode a video with given codec arguments.
    """
    cmd = ["ffmpeg", "-y", "-i", input_path] + codec_args + [output_path]
    subprocess.run(cmd, check=True)


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Encodes the input video with different codecs and saves results.
    Yields (output_path, label).
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "encoded"
    os.makedirs(output_root, exist_ok=True)

    for label, cfg in CODECS.items():
        out_path = output_root / f"{label}.{cfg['ext']}"
        if out_path.exists():
            yield str(out_path), label
            continue

        run_ffmpeg(input_path, str(out_path), cfg["args"])
        yield str(out_path), label



