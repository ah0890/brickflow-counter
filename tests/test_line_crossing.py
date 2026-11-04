"""Geometry primitives for line-crossing detection."""

import pytest

from src.line_crossing import (
    Line,
    crossing_direction,
    direction_allowed,
    intersection_point,
    segments_intersect,
)


def test_degenerate_line_rejected():
    with pytest.raises(ValueError):
        Line((5, 5), (5, 5))


def test_sign_convention_horizontal_left_to_right():
    line = Line((0, 100), (200, 100))
    assert line.signed_distance((50, 150)) == pytest.approx(50)   # below → positive
    assert line.signed_distance((50, 40)) == pytest.approx(-60)   # above → negative


def test_sign_convention_flips_with_line_orientation():
    line = Line((200, 100), (0, 100))  # drawn right-to-left
    assert line.signed_distance((50, 150)) < 0


def test_sign_convention_vertical_top_to_bottom():
    line = Line((100, 0), (100, 200))
    # Right-hand side of a downward direction (y down) is the image's left.
    assert line.signed_distance((50, 50)) > 0
    assert line.signed_distance((150, 50)) < 0


def test_signed_distance_diagonal_magnitude():
    line = Line((0, 0), (100, 100))
    assert abs(line.signed_distance((100, 0))) == pytest.approx(70.7107, rel=1e-4)


@pytest.mark.parametrize("y,expected", [(90, -1), (94, -1), (95, 0), (100, 0), (105, 0), (106, 1)])
def test_side_with_hysteresis_band(y, expected):
    assert Line((0, 100), (200, 100)).side((50, y), band=5) == expected


@pytest.mark.parametrize(
    "a,b,c,d,expected",
    [
        ((0, 0), (10, 10), (0, 10), (10, 0), True),       # proper X crossing
        ((0, 0), (10, 0), (0, 5), (10, 5), False),        # parallel
        ((0, 0), (10, 0), (10, 0), (10, 10), True),       # touching at endpoint
        ((0, 0), (10, 0), (5, 0), (15, 0), True),         # collinear overlap
        ((0, 0), (10, 0), (11, 0), (20, 0), False),       # collinear disjoint
        ((5, -5), (5, 5), (0, 0), (10, 0), True),         # T through middle
        ((5, 1), (5, 5), (0, 0), (10, 0), False),         # stops short of the line
        ((15, -5), (15, 5), (0, 0), (10, 0), False),      # beyond the segment's end
    ],
)
def test_segments_intersect(a, b, c, d, expected):
    assert segments_intersect(a, b, c, d) is expected
    assert segments_intersect(c, d, a, b) is expected  # symmetric


def test_intersection_point():
    assert intersection_point((5, -5), (5, 5), (0, 0), (10, 0)) == pytest.approx((5, 0))
    assert intersection_point((15, -5), (15, 5), (0, 0), (10, 0)) is None


def test_direction_helpers():
    assert crossing_direction(-1, 1) == "forward"
    assert crossing_direction(1, -1) == "reverse"
    assert crossing_direction(1, 1) is None
    assert direction_allowed("forward", "both") and direction_allowed("reverse", "both")
    assert direction_allowed("forward", "forward") and not direction_allowed("reverse", "forward")
