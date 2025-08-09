# utils/rois.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Optional
import numpy as np
import cv2 as cv

Coord = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Coord:
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    return x1, y1, x2, y2


def bbox_from_normalized_landmarks(
    landmarks: Iterable, img_w: int, img_h: int
) -> Coord:
    """
    Build a pixel-space face bounding box from normalized landmarks (0..1).
    `landmarks` can be any iterable of objects/tuples with `.x` and `.y` or index [0],[1].
    """
    xs, ys = [], []
    for lm in landmarks:
        try:
            xs.append(float(lm.x))
            ys.append(float(lm.y))
        except AttributeError:
            xs.append(float(lm[0]))
            ys.append(float(lm[1]))
    x1 = int(min(xs) * img_w)
    y1 = int(min(ys) * img_h)
    x2 = int(max(xs) * img_w)
    y2 = int(max(ys) * img_h)
    return _clip_box(x1, y1, x2, y2, img_w, img_h)


def cheek_band_from_bbox(
    bbox: Coord, img_w: int, img_h: int,
    horizontal_margin: float = 0.15,
    top_ratio: float = 0.40,
    bottom_ratio: float = 0.65,
) -> Coord:
    """
    Compute the cheek *band* rectangle inside a face bbox, using the same ratios
    as your prototype (0.15 horizontal margin, 0.40–0.65 vertical band).
    """
    x1, y1, x2, y2 = bbox
    band_y1 = y1 + int(top_ratio * (y2 - y1))
    band_y2 = y1 + int(bottom_ratio * (y2 - y1))
    band_x1 = x1 + int(horizontal_margin * (x2 - x1))
    band_x2 = x2 - int(horizontal_margin * (x2 - x1))
    return _clip_box(band_x1, band_y1, band_x2, band_y2, img_w, img_h)


def split_band_into_cheeks(
    band: Coord, img_w: int, img_h: int, mid_gap_ratio: float = 0.08
) -> Tuple[Coord, Coord]:
    """
    Split the cheek *band* into left/right cheek boxes, leaving a small
    gap around the nose (mid_gap_ratio of band width).
    """
    bx1, by1, bx2, by2 = band
    bw = bx2 - bx1
    gap = int(bw * mid_gap_ratio)
    mid = bx1 + bw // 2

    left = _clip_box(bx1, by1, max(bx1, mid - gap // 2), by2, img_w, img_h)
    right = _clip_box(min(bx2, mid + gap // 2), by1, bx2, by2, img_w, img_h)
    return left, right


def get_cheek_rois(
    frame_bgr: np.ndarray,
    *,
    landmarks: Optional[Iterable] = None,
    bbox: Optional[Coord] = None,
    horizontal_margin: float = 0.15,
    top_ratio: float = 0.40,
    bottom_ratio: float = 0.65,
    mid_gap_ratio: float = 0.08,
    return_arrays: bool = False,
    draw_on: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Dict[str, Coord] | Tuple[Dict[str, Coord], Dict[str, np.ndarray]]:
    """
    Compute left/right cheek ROIs from either MediaPipe landmarks or a face bbox.

    Args:
        frame_bgr: Input frame (H,W,3).
        landmarks: Iterable of normalized landmarks (0..1). Provide either this or `bbox`.
        bbox: (x1,y1,x2,y2) face rectangle in pixels. Provide either this or `landmarks`.
        horizontal_margin: Horizontal margin inside face bbox (0..1 of width).
        top_ratio, bottom_ratio: Vertical band for cheeks inside bbox (0..1 of height).
        mid_gap_ratio: Fraction of band width to exclude around nose (0..1).
        return_arrays: If True, also return cropped BGR arrays for each cheek.
        draw_on: If provided, rectangles will be drawn on this image (in-place).
        color, thickness: Drawing parameters.

    Returns:
        If return_arrays=False (default):
            {"left": (x1,y1,x2,y2), "right": (x1,y1,x2,y2)}
        If return_arrays=True:
            (coords_dict, arrays_dict) where arrays_dict has "left", "right" crops.
    """
    h, w = frame_bgr.shape[:2]

    if bbox is None and landmarks is None:
        raise ValueError(
            "Provide either `landmarks` (normalized) or `bbox` (pixel coords).")

    if bbox is None:
        bbox = bbox_from_normalized_landmarks(landmarks, w, h)

    band = cheek_band_from_bbox(
        bbox, w, h,
        horizontal_margin=horizontal_margin,
        top_ratio=top_ratio,
        bottom_ratio=bottom_ratio,
    )
    left, right = split_band_into_cheeks(
        band, w, h, mid_gap_ratio=mid_gap_ratio)

    if draw_on is not None:
        cv.rectangle(draw_on, (left[0], left[1]),
                     (left[2], left[3]), color, thickness)
        cv.rectangle(draw_on, (right[0], right[1]),
                     (right[2], right[3]), color, thickness)

    coords = {"left": left, "right": right}

    if not return_arrays:
        return coords

    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    crops = {
        "left": frame_bgr[ly1:ly2, lx1:lx2] if ly2 > ly1 and lx2 > lx1 else np.empty((0, 0, 3), dtype=frame_bgr.dtype),
        "right": frame_bgr[ry1:ry2, rx1:rx2] if ry2 > ry1 and rx2 > rx1 else np.empty((0, 0, 3), dtype=frame_bgr.dtype),
    }
    return coords, crops
