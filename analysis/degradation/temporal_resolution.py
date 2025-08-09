import numpy as np
import os
from pathlib import Path
from typing import Generator, Tuple
from utils.video_io import read_video, write_video

TARGET_FPS = [60, 30, 25, 15, 10, 5]

def _format_fps_label(fps: float) -> str:
    # Use integer label if it's effectively an integer (e.g., 30.0 → "30fps")
    if abs(fps - round(fps)) < 1e-3:
        return f"{int(round(fps))}fps"
    return f"{fps:.2f}fps"

def _resample_by_time(frames, src_fps: float, dst_fps: float):
    """
    Resample by *time* to preserve duration:
    - n_src frames at src_fps ⇒ duration ≈ (n_src-1)/src_fps (first-to-last spacing)
    - choose n_dst so that first/last timestamps match closely
    """
    n_src = len(frames)
    if n_src == 0:
        return []

    # Use first-to-last spacing so last frame maps to last timestamp
    duration = (n_src - 1) / src_fps
    # Ensure we include both ends: first frame at t=0 and last at t=duration
    n_dst = max(1, int(round(duration * dst_fps)) + 1)

    # Desired output timestamps
    t_out = np.linspace(0.0, duration, n_dst, endpoint=True)            # shape (n_dst,)
    # Map to nearest input frame index
    idx = np.round(t_out * src_fps).astype(int)
    idx = np.clip(idx, 0, n_src - 1)

    return [frames[i] for i in idx]

def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies temporal resolution degradation by resampling frames in *time*
    (prevents duration drift and HR distortion).
    """
    base_name = Path(input_path).stem
    output_root = Path("results") / base_name / "degraded" / "temporal_resolution"
    os.makedirs(output_root, exist_ok=True)

    frames, fps = read_video(input_path)

    # Include original video as control
    label = _format_fps_label(fps)
    out_path = output_root / f"{label}.mp4"
    if not out_path.exists():
        write_video(frames, str(out_path), fps)
    yield str(out_path), label

    for target_fps in TARGET_FPS:
        if target_fps >= fps:
            continue

        label = _format_fps_label(target_fps)
        out_path = output_root / f"{label}.mp4"
        if out_path.exists():
            yield str(out_path), label
            continue

        downsampled_frames = _resample_by_time(frames, fps, float(target_fps))
        write_video(downsampled_frames, str(out_path), float(target_fps))
        yield str(out_path), label
