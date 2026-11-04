"""Pure geometry for virtual-line counting (no OpenCV, no tracker types).

Conventions (image coordinates, y grows downward):
  * The line runs from ``p1`` to ``p2``.
  * The *positive* side is the right-hand side of the p1→p2 direction, i.e.
    the side the normal (-dy, dx) points to. For a line drawn left-to-right,
    positive is *below* the line.
  * A *forward* crossing moves from the negative side to the positive side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]

_EPS = 1e-9


@dataclass(frozen=True)
class Line:
    p1: Point
    p2: Point

    def __post_init__(self) -> None:
        if self.length < _EPS:
            raise ValueError("Counting line endpoints must be distinct.")

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    def signed_distance(self, pt: Point) -> float:
        """Perpendicular distance from the infinite line; > 0 on the positive side."""
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        return (dx * (pt[1] - self.p1[1]) - dy * (pt[0] - self.p1[0])) / self.length

    def side(self, pt: Point, band: float = 0.0) -> int:
        """+1 / -1 for points clearly on a side; 0 for points within ``band`` of the line."""
        d = self.signed_distance(pt)
        if d > band:
            return 1
        if d < -band:
            return -1
        return 0


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
            and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True if closed segments AB and CD share at least one point.

    Standard orientation test, including collinear and endpoint-touching cases.
    """
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)

    def sgn(v: float) -> int:
        return 0 if abs(v) < _EPS else (1 if v > 0 else -1)

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)
    if s1 != s2 and s3 != s4:
        return True
    if s1 == 0 and _on_segment(a, b, c):
        return True
    if s2 == 0 and _on_segment(a, b, d):
        return True
    if s3 == 0 and _on_segment(c, d, a):
        return True
    if s4 == 0 and _on_segment(c, d, b):
        return True
    return False


def intersection_point(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    """Intersection of segment AB with segment CD (None if parallel or disjoint)."""
    if not segments_intersect(a, b, c, d):
        return None
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denom = rx * sy - ry * sx
    if abs(denom) < _EPS:  # collinear overlap: report the first touching endpoint
        return c if _on_segment(a, b, c) else a
    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / denom
    return (a[0] + t * rx, a[1] + t * ry)


def crossing_direction(from_side: int, to_side: int) -> str | None:
    """'forward' for -1 → +1, 'reverse' for +1 → -1, otherwise None."""
    if from_side == -1 and to_side == 1:
        return "forward"
    if from_side == 1 and to_side == -1:
        return "reverse"
    return None


def direction_allowed(direction: str, mode: str) -> bool:
    return mode == "both" or direction == mode
