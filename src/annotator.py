"""Frame annotation: boxes, track IDs, counting line, count panel, frame info.

All sizes scale with the frame so overlays stay legible on both phone
portrait footage and 1080p/4K landscape video.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from src.tracking import Track

# BGR colours
GREEN = (94, 197, 34)
AMBER = (11, 158, 245)
WHITE = (255, 255, 255)
NAVY = (42, 23, 15)
DARK = (20, 20, 20)

_FONT = cv2.FONT_HERSHEY_DUPLEX
_PALETTE = [  # distinct, non-green colours for uncounted tracks
    (255, 178, 29), (207, 210, 49), (255, 112, 31), (189, 0, 255), (255, 149, 200),
    (199, 55, 255), (52, 147, 26), (168, 153, 44), (255, 194, 0), (147, 69, 52),
]


@dataclass(frozen=True)
class AnnotationStyle:
    show_confidence: bool = True
    show_ids: bool = True
    show_anchor: bool = True


def _scale(frame: np.ndarray) -> float:
    return max(min(frame.shape[:2]) / 720.0, 0.5)


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    counted_ids: Collection[int] = (),
    style: AnnotationStyle = AnnotationStyle(),
    anchor_mode: str = "bottom_center",
) -> None:
    s = _scale(frame)
    thick = max(int(round(2 * s)), 1)
    fs = 0.5 * s
    for t in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in t.bbox)
        color = GREEN if t.track_id in counted_ids else _color_for(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        parts = []
        if style.show_ids:
            parts.append(f"#{t.track_id}")
        if style.show_confidence:
            parts.append(f"{t.confidence:.2f}")
        if parts:
            label = " ".join(parts)
            (tw, th), base = cv2.getTextSize(label, _FONT, fs, max(thick - 1, 1))
            ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
            cv2.rectangle(frame, (x1, ty - th - base), (x1 + tw + 6, ty + base // 2), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - base // 2), _FONT, fs, DARK,
                        max(thick - 1, 1), cv2.LINE_AA)

        if style.show_anchor:
            ax, ay = (int(round(v)) for v in t.anchor(anchor_mode))
            cv2.circle(frame, (ax, ay), max(int(3 * s), 2), color, -1, cv2.LINE_AA)


def draw_counting_line(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    direction: str = "forward",
) -> None:
    """Draw the line and arrow(s) showing which crossing direction is counted.

    'forward' points to the right-hand side of the p1→p2 direction in image
    coordinates, i.e. along the normal (-dy, dx).
    """
    s = _scale(frame)
    thick = max(int(round(3 * s)), 2)
    cv2.line(frame, p1, p2, AMBER, thick, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(frame, p, max(int(6 * s), 3), AMBER, -1, cv2.LINE_AA)
        cv2.circle(frame, p, max(int(6 * s), 3), WHITE, 1, cv2.LINE_AA)

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = float(np.hypot(dx, dy))
    if length < 1:
        return
    nx, ny = -dy / length, dx / length
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    arrow = 40 * s
    signs = {"forward": (1,), "reverse": (-1,), "both": (1, -1)}.get(direction, (1,))
    for sign in signs:
        start = (int(mx - sign * nx * arrow * 0.5), int(my - sign * ny * arrow * 0.5))
        end = (int(mx + sign * nx * arrow), int(my + sign * ny * arrow))
        cv2.arrowedLine(frame, start, end, AMBER, thick, cv2.LINE_AA, tipLength=0.35)


def draw_count_panel(frame: np.ndarray, count: int | None, label: str = "BRICKS COUNTED") -> None:
    """Prominent green count box in the top-left corner (``None`` = counting disabled)."""
    s = _scale(frame)
    pad = int(14 * s)
    value = "--" if count is None else str(count)
    lfs, vfs = 0.55 * s, 1.6 * s
    lt, vt = max(int(1 * s), 1), max(int(3 * s), 2)
    (lw, lh), _ = cv2.getTextSize(label, _FONT, lfs, lt)
    (vw, vh), _ = cv2.getTextSize(value, _FONT, vfs, vt)
    w = max(lw, vw) + 2 * pad
    h = lh + vh + 3 * pad
    x0, y0 = int(12 * s), int(12 * s)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), GREEN, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), WHITE, max(int(s), 1))
    cv2.putText(frame, label, (x0 + pad, y0 + pad + lh), _FONT, lfs, NAVY, lt, cv2.LINE_AA)
    cv2.putText(frame, value, (x0 + pad, y0 + 2 * pad + lh + vh), _FONT, vfs, WHITE, vt, cv2.LINE_AA)


def draw_info_bar(frame: np.ndarray, frame_index: int, total_frames: int, fps: float, active: int) -> None:
    """Small frame/time/active-track readout in the top-right corner."""
    s = _scale(frame)
    t = frame_index / fps if fps > 0 else 0.0
    total = f"/{total_frames}" if total_frames else ""
    text = f"Frame {frame_index + 1}{total}  {int(t // 60):02d}:{t % 60:04.1f}  Tracks {active}"
    fs, th = 0.45 * s, max(int(s), 1)
    (tw, tht), base = cv2.getTextSize(text, _FONT, fs, th)
    pad = int(6 * s)
    x1 = frame.shape[1] - tw - 2 * pad - int(12 * s)
    y0 = int(12 * s)
    x1 = max(x1, 0)
    roi = frame[y0 : y0 + tht + base + 2 * pad, x1 : x1 + tw + 2 * pad]
    roi[:] = (roi * 0.4 + np.array(DARK) * 0.6).astype(frame.dtype)  # translucent backdrop
    cv2.putText(frame, text, (x1 + pad, y0 + pad + tht), _FONT, fs, WHITE, th, cv2.LINE_AA)
