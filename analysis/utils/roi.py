from typing import Generator, Tuple, Sequence, Any, List, Optional
from pathlib import Path
import mediapipe as mp
import cv2
import numpy as np
REUSE_LANDMARKS_FOR = 15    # if detection drops, reuse last landmarks for N frames

# Cheek ROI ratios inside face bbox (x1,y1,x2,y2)
CHEEK_HR = 0.15   # horizontal margin ratio
CHEEK_TOP = 0.40
CHEEK_BOT = 0.65

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

Frame = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
Landmarks = List


def bbox_from_landmarks(landmarks: Landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(max(0, min(xs) * w))
    y1 = int(max(0, min(ys) * h))
    x2 = int(min(w - 1, max(xs) * w))
    y2 = int(min(h - 1, max(ys) * h))
    return x1, y1, x2, y2


def _cheek_roi_from_bbox(bb: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bb
    roi_y1 = int(np.clip(y1 + CHEEK_TOP * (y2 - y1), 0, h - 1))
    roi_y2 = int(np.clip(y1 + CHEEK_BOT * (y2 - y1), 0, h))
    roi_x1 = int(np.clip(x1 + CHEEK_HR * (x2 - x1), 0, w - 1))
    roi_x2 = int(np.clip(x2 - CHEEK_HR * (x2 - x1), 0, w))
    return roi_x1, roi_y1, roi_x2, roi_y2


def get_roi(frames: Sequence[Frame], fps: float) -> Generator[Frame, None, None]:
    """
    Find ROI for each frame using MediaPipe landmarks.

    Returns a generator yielding:
        roi: np.ndarray of shape (H, W, 3) of all BGR pixels in the ROI.
    """
    # Setup landmarker
    model = Path(__file__).resolve().parent / 'face_landmarker.task'
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # Detect
    last_landmarks: Optional[Landmarks] = None
    reuse_left = 0
    timestamps = np.arange(len(frames), dtype=float) / float(fps)
    with landmarker:
        for i, bgr in enumerate(frames):
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(timestamps[i] * 1000.0)

            result = landmarker.detect_for_video(mp_image, ts_ms)
            if result and result.face_landmarks:
                last_landmarks = result.face_landmarks[0]
                reuse_left = REUSE_LANDMARKS_FOR
            elif last_landmarks is not None and reuse_left > 0:
                reuse_left -= 1
            else:
                # No landmarks, yield empty ROI
                yield np.mat(data=[[]])

            if last_landmarks is None:
                continue

            # ROI from landmarks
            bb = bbox_from_landmarks(last_landmarks, w, h)
            cx1, cy1, cx2, cy2 = _cheek_roi_from_bbox(bb, w, h)
            roi = bgr[cy1:cy2, cx1:cx2]

            # Yield complete ROI
            yield roi
