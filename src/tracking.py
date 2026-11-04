"""Framework-independent track records.

The detector/tracker backend converts its output into ``Track`` objects so the
counting engine, annotator and pipeline never depend on Ultralytics types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ANCHOR_MODES = ("bottom_center", "center")


@dataclass(frozen=True)
class Track:
    """One tracked object in one frame. Coordinates are pixels (x1, y1, x2, y2)."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int

    def anchor(self, mode: str = "bottom_center") -> tuple[float, float]:
        """Representative point used for line-crossing tests."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        if mode == "bottom_center":
            return cx, y2
        if mode == "center":
            return cx, (y1 + y2) / 2.0
        raise ValueError(f"Unknown anchor mode {mode!r}; expected one of {ANCHOR_MODES}.")


def tracks_from_ultralytics(result: Any) -> list[Track]:
    """Convert an Ultralytics ``Results`` object (track mode) into Track records.

    Detections without an assigned track ID (unconfirmed by the tracker) are
    dropped: counting is based on tracked objects only, never raw detections.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    ids = _np(boxes.id).astype(int)
    xyxy = _np(boxes.xyxy).astype(float)
    conf = _np(boxes.conf).astype(float)
    cls = _np(boxes.cls).astype(int)
    return [
        Track(int(i), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c), int(k))
        for i, b, c, k in zip(ids, xyxy, conf, cls)
    ]


def _np(x: Any) -> np.ndarray:
    """Tensor or array-like → NumPy array."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)
