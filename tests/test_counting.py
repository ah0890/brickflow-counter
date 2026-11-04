"""Counting engine against synthetic trajectories (independent of YOLO/tracker).

Default geometry: horizontal line y=100 from x=0 to x=200, drawn left→right,
so *forward* = moving downward. Boxes are 20×20 with the anchor at the
bottom-centre, so ``box(id, x, y)`` places the anchor exactly at (x, y).
"""

import dataclasses

import pytest

from src.config import load_config
from src.counting import CountingEngine, CountingParams, build_counter
from src.line_crossing import Line
from src.tracking import Track

LINE = Line((0, 100), (200, 100))


def box(tid, x, y, conf=0.9):
    return Track(tid, (x - 10, y - 20, x + 10, y), conf, 0)


def engine(**kw):
    return CountingEngine(LINE, CountingParams(**kw))


def run(eng, frames):
    """frames: list of per-frame track lists. Returns all events."""
    events = []
    for i, tracks in enumerate(frames):
        events += eng.update(tracks, i)
    return events


def path(tid, x, ys):
    return [[box(tid, x, y)] for y in ys]


def merge(*sequences):
    """Combine per-frame track lists from several objects (padding with empty frames)."""
    n = max(len(s) for s in sequences)
    return [sum((s[i] if i < len(s) else [] for s in sequences), []) for i in range(n)]


# --------------------------------------------------------------------------- #
# Scenario 1 – one brick crossing
# --------------------------------------------------------------------------- #

def test_single_crossing_counted_once():
    eng = engine()
    events = run(eng, path(1, 100, range(60, 141, 5)))
    assert eng.total == 1 and len(events) == 1
    e = events[0]
    assert e.track_id == 1 and e.direction == "forward" and e.count == 1
    assert e.position == pytest.approx((100, 100))
    assert eng.counted_ids == {1}


def test_crossing_frame_is_where_band_was_left():
    eng = engine(hysteresis_px=6)
    ys = [80, 90, 98, 103, 110, 120]  # leaves band (y>106) at index 4
    events = run(eng, path(1, 100, ys))
    assert events[0].crossing_frame == 4


# --------------------------------------------------------------------------- #
# Scenario 2 – multiple bricks sequentially
# --------------------------------------------------------------------------- #

def test_multiple_sequential_bricks():
    eng = engine()
    frames = path(1, 50, range(60, 141, 10)) + path(2, 100, range(60, 141, 10)) + path(3, 150, range(60, 141, 10))
    events = run(eng, frames)
    assert eng.total == 3
    assert [e.track_id for e in events] == [1, 2, 3]
    assert [e.count for e in events] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Scenario 3 – brick remaining near the line
# --------------------------------------------------------------------------- #

def test_jitter_inside_band_never_counts():
    eng = engine(hysteresis_px=6)
    ys = [80] + [96, 104, 97, 105, 95, 103] * 20
    run(eng, path(1, 100, ys))
    assert eng.total == 0


def test_object_parked_on_line_never_counts():
    eng = engine()
    run(eng, path(1, 100, [100] * 200))
    assert eng.total == 0


def test_hovering_back_and_forth_counts_at_most_once():
    eng = engine(hysteresis_px=6, direction="both")
    ys = [80] + [90, 112] * 30  # repeatedly leaves the band on both sides
    run(eng, path(1, 100, ys))
    assert eng.total == 1


# --------------------------------------------------------------------------- #
# Scenario 4 – reverse crossing and direction filtering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode,expected", [("forward", 0), ("reverse", 1), ("both", 1)])
def test_upward_crossing_by_direction_mode(mode, expected):
    eng = engine(direction=mode)
    events = run(eng, path(1, 100, range(140, 59, -5)))
    assert eng.total == expected
    if expected:
        assert events[0].direction == "reverse"


@pytest.mark.parametrize("mode,expected", [("forward", 1), ("reverse", 0), ("both", 1)])
def test_downward_crossing_by_direction_mode(mode, expected):
    eng = engine(direction=mode)
    run(eng, path(1, 100, range(60, 141, 5)))
    assert eng.total == expected


def test_wrong_direction_recorded_in_stats():
    eng = engine(direction="forward")
    run(eng, path(1, 100, range(140, 59, -5)))
    assert eng.stats.wrong_direction == 1


# --------------------------------------------------------------------------- #
# Scenario 5 – brick disappearing temporarily
# --------------------------------------------------------------------------- #

def test_brief_disappearance_across_line_still_counts():
    eng = engine()
    frames = path(1, 100, [60, 70, 80, 90]) + [[]] * 10 + path(1, 100, [115, 125, 135])
    run(eng, frames)
    assert eng.total == 1


def test_disappearance_longer_than_timeout_resets_side():
    eng = engine(track_timeout_frames=5, handover_max_gap_frames=0)
    frames = path(1, 100, [60, 70, 80]) + [[]] * 10 + path(1, 100, [120, 130, 140])
    run(eng, frames)
    assert eng.total == 0  # side history discarded; no unverifiable crossing


# --------------------------------------------------------------------------- #
# Scenario 6 – object crossing repeatedly
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["forward", "reverse", "both"])
def test_repeated_crossings_counted_once_per_id(mode):
    eng = engine(direction=mode)
    down, up = list(range(60, 141, 10)), list(range(140, 59, -10))
    run(eng, path(1, 100, down + up + down + up + down))
    assert eng.total == 1


# --------------------------------------------------------------------------- #
# Scenario 7 – two objects crossing simultaneously
# --------------------------------------------------------------------------- #

def test_two_simultaneous_crossings():
    eng = engine()
    ys = range(60, 141, 5)
    events = run(eng, merge(path(1, 60, ys), path(2, 140, ys)))
    assert eng.total == 2
    assert {e.track_id for e in events} == {1, 2}
    assert events[0].crossing_frame == events[1].crossing_frame


def test_adjacent_objects_do_not_steal_identity():
    # Two touching bricks, both continuously tracked: handover must not merge them.
    eng = engine()
    ys = range(60, 141, 5)
    run(eng, merge(path(1, 90, ys), path(2, 110, ys)))
    assert eng.total == 2 and eng.stats.handovers == 0


# --------------------------------------------------------------------------- #
# Scenario 8 – tracker ID changes
# --------------------------------------------------------------------------- #

def test_id_switch_mid_crossing_is_recovered_by_handover():
    eng = engine()
    frames = path(1, 100, [60, 70, 80, 90]) + path(2, 102, [108, 118, 128])
    events = run(eng, frames)
    assert eng.total == 1
    assert events[0].track_id == 2 and events[0].inherited_from == 1
    assert eng.stats.handovers == 1


def test_id_switch_mid_crossing_missed_without_handover():
    # Documents the limitation that handover addresses.
    eng = engine(handover_max_gap_frames=0)
    run(eng, path(1, 100, [60, 70, 80, 90]) + path(2, 102, [108, 118, 128]))
    assert eng.total == 0


def test_id_switch_after_counting_does_not_double_count():
    eng = engine(direction="both")
    frames = path(1, 100, [60, 80, 110, 120, 130]) + path(2, 101, [135, 120, 110, 90, 70, 110, 130])
    run(eng, frames)
    assert eng.total == 1
    assert eng.counted_ids == {1, 2}  # new ID inherits the counted flag


def test_distant_new_id_is_independent():
    eng = engine()
    frames = path(1, 30, [60, 70, 80, 90]) + path(2, 170, [60, 80, 110, 130])
    run(eng, frames)
    assert eng.stats.handovers == 0
    assert eng.total == 1 and eng.events[0].track_id == 2


def test_handover_gap_limit():
    eng = engine(handover_max_gap_frames=3)
    frames = path(1, 100, [60, 70, 80, 90]) + [[]] * 5 + path(2, 101, [110, 120, 130])
    run(eng, frames)
    assert eng.stats.handovers == 0 and eng.total == 0


def test_tracker_id_reappearing_after_prune_not_recounted():
    eng = engine(track_timeout_frames=3, handover_max_gap_frames=0, direction="both")
    frames = path(1, 100, range(60, 141, 10)) + [[]] * 10 + path(1, 100, range(140, 59, -10))
    run(eng, frames)
    assert eng.total == 1


# --------------------------------------------------------------------------- #
# Scenario 9 – objects appearing on (or past) the line
# --------------------------------------------------------------------------- #

def test_object_appearing_on_line_then_moving_away_not_counted():
    eng = engine()
    run(eng, path(1, 100, [100, 104, 110, 120, 140]))
    assert eng.total == 0


def test_object_appearing_past_line_not_counted():
    eng = engine()
    run(eng, path(1, 100, [110, 120, 130]))
    assert eng.total == 0


def test_object_appearing_on_line_counted_after_establishing_side():
    eng = engine()
    run(eng, path(1, 100, [100, 96, 85, 80, 90, 100, 110, 120]))
    assert eng.total == 1


# --------------------------------------------------------------------------- #
# Duplicate frames, track age, geometry limits, teleport guard
# --------------------------------------------------------------------------- #

def test_repeated_frame_updates_do_not_inflate_count():
    eng = engine()
    for i, y in enumerate(range(60, 141, 10)):
        tracks = [box(1, 100, y)]
        eng.update(tracks, i)
        eng.update(tracks, i)  # same frame delivered twice
    assert eng.total == 1


def test_duplicate_id_within_one_frame_ignored():
    eng = engine()
    for i, y in enumerate(range(60, 141, 10)):
        eng.update([box(1, 100, y), box(1, 100, y)], i)
    assert eng.total == 1


def test_frozen_video_on_far_side_does_not_count_again():
    eng = engine()
    run(eng, path(1, 100, list(range(60, 141, 10)) + [140] * 300))
    assert eng.total == 1


def test_short_lived_track_not_counted():
    eng = engine(min_track_frames=3)
    run(eng, path(1, 100, [90, 115]))  # only 2 frames
    assert eng.total == 0


def test_fast_crossing_confirmed_once_track_matures():
    eng = engine(min_track_frames=4)
    events = run(eng, path(1, 100, [80, 120, 130, 140, 150]))
    assert eng.total == 1
    assert events[0].crossing_frame == 1 and events[0].confirmed_frame == 3


def test_pending_crossing_cancelled_if_object_returns():
    eng = engine(min_track_frames=5, direction="both")
    run(eng, path(1, 100, [80, 120, 80, 75, 70, 65, 60]))
    assert eng.total == 0 and eng.stats.pending_cancelled == 1


def test_crossing_outside_finite_segment_not_counted():
    eng = CountingEngine(Line((50, 100), (150, 100)), CountingParams())
    run(eng, path(1, 180, range(60, 141, 5)))
    assert eng.total == 0 and eng.stats.outside_segment == 1


def test_crossing_at_segment_endpoint_counted():
    eng = CountingEngine(Line((50, 100), (150, 100)), CountingParams())
    run(eng, path(1, 150, range(60, 141, 5)))
    assert eng.total == 1


def test_teleport_jump_rejected():
    eng = engine(max_step_ratio=1.5)
    run(eng, path(1, 100, [60, 62, 64, 400, 402, 404]))  # scene-cut-like jump
    assert eng.total == 0 and eng.stats.jumps_rejected == 1


def test_teleport_guard_disabled():
    eng = engine(max_step_ratio=0)
    run(eng, path(1, 100, [60, 62, 64, 400, 402, 404]))
    assert eng.total == 1


def test_fast_but_plausible_motion_counts():
    eng = engine(max_step_ratio=1.5)  # box diagonal ≈ 28 px → up to 42 px/frame
    run(eng, path(1, 100, [30, 65, 100, 135, 170]))
    assert eng.total == 1


# --------------------------------------------------------------------------- #
# Line orientations
# --------------------------------------------------------------------------- #

def test_vertical_line_forward_is_right_to_left():
    eng = CountingEngine(Line((100, 0), (100, 200)), CountingParams(direction="forward"))
    run(eng, [[box(1, x, 150)] for x in range(160, 39, -10)])
    assert eng.total == 1
    eng2 = CountingEngine(Line((100, 0), (100, 200)), CountingParams(direction="forward"))
    run(eng2, [[box(1, x, 150)] for x in range(40, 161, 10)])
    assert eng2.total == 0


def test_diagonal_line_crossing():
    eng = CountingEngine(Line((0, 0), (200, 200)), CountingParams(direction="both"))
    run(eng, [[box(1, 100 + d, 100 - d)] for d in range(-50, 51, 10)])
    assert eng.total == 1
    assert eng.events[0].position == pytest.approx((100, 100), abs=1)


def test_center_anchor_mode():
    eng = engine(anchor="center")
    # Bottom-centre crosses at y=100 but the centre (10 px higher) stays above.
    run(eng, path(1, 100, [60, 80, 100, 108, 112, 114]))
    assert eng.total == 0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_invalid_direction_rejected():
    with pytest.raises(ValueError):
        engine(direction="sideways")


def test_build_counter_from_config():
    cfg = load_config().counting
    cfg = dataclasses.replace(cfg, line=dataclasses.replace(cfg.line, x1=0.0, y1=0.5, x2=1.0, y2=0.5))
    eng = build_counter(cfg, 200, 200)
    assert eng.line.p1 == (0, 100) and eng.line.p2 == (200, 100)
    assert eng.params.direction == cfg.direction
    run(eng, path(1, 100, range(60, 141, 5)))
    assert eng.total == (1 if cfg.direction in ("forward", "both") else 0)
