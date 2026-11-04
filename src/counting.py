"""Trajectory-based counting engine.

A tracked object is counted when its anchor point moves from one *confirmed*
side of the counting line to the other and the movement segment intersects
the (finite) line segment, in an allowed direction. Safeguards:

* Hysteresis band - points within ``hysteresis_px`` of the line never change
  a track's confirmed side, so jitter on the line cannot produce crossings.
* Objects first seen on the line have no side until they leave the band, so
  they are never counted for "appearing" across it.
* One count per track ID, permanently for the job (survives state pruning and
  re-appearance after long gaps).
* Minimum track age - a crossing is held as *pending* until the track has
  existed for ``min_track_frames``; if it moves back first, it is discarded.
  This filters one- or two-frame false tracks without missing fast crossings.
* Teleport guard - a jump larger than ``max_step_ratio`` box diagonals per
  elapsed frame (scene cut, identity swap) resets the track's side instead of
  being treated as a crossing.
* Identity handover - a new ID appearing next to a recently lost ID inherits
  its state (side, pending crossing, counted flag), reducing both double
  counts and misses caused by tracker ID switches.
* Lost-track state is pruned after ``track_timeout_frames``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config import CountingConfig
from src.line_crossing import Line, Point, crossing_direction, direction_allowed, intersection_point
from src.tracking import Track


@dataclass(frozen=True)
class CountingParams:
    direction: str = "forward"
    anchor: str = "bottom_center"
    hysteresis_px: float = 6.0
    min_track_frames: int = 3
    track_timeout_frames: int = 90
    max_step_ratio: float = 1.5          # 0 disables the teleport guard
    handover_max_gap_frames: int = 15    # 0 disables identity handover
    handover_max_distance: float = 0.75  # in multiples of the lost box diagonal

    @classmethod
    def from_config(cls, c: CountingConfig) -> CountingParams:
        return cls(
            direction=c.direction,
            anchor=c.anchor,
            hysteresis_px=c.hysteresis_px,
            min_track_frames=c.min_track_frames,
            track_timeout_frames=c.track_timeout_frames,
            max_step_ratio=c.max_step_ratio,
            handover_max_gap_frames=c.handover_max_gap_frames,
            handover_max_distance=c.handover_max_distance,
        )


@dataclass(frozen=True)
class CountEvent:
    count: int                  # Running total after this event.
    track_id: int
    crossing_frame: int         # Frame in which the line was crossed.
    confirmed_frame: int        # Frame in which the count was registered.
    direction: str              # forward | reverse
    position: Point             # Where the trajectory crossed the line (px).
    inherited_from: int | None  # Original track ID if state came via handover.


@dataclass
class CountingStats:
    """Diagnostics: why candidate crossings were not counted."""

    outside_segment: int = 0     # Side changed but path missed the finite line.
    wrong_direction: int = 0
    already_counted: int = 0
    pending_cancelled: int = 0   # Crossed then returned before min track age.
    jumps_rejected: int = 0      # Teleport guard resets.
    handovers: int = 0


@dataclass
class _Pending:
    frame: int
    direction: str
    position: Point


@dataclass
class _TrackState:
    track_id: int
    first_seen: int
    last_seen: int
    frames_seen: int = 0
    side: int | None = None            # Confirmed side: -1, +1 or None (unknown).
    side_point: Point | None = None    # Last anchor observed on the confirmed side.
    last_anchor: Point | None = None
    last_diag: float = 0.0
    counted: bool = False
    pending: _Pending | None = None
    handed_over: bool = False          # State already passed to a new ID.
    origin_id: int | None = None       # First ID in a handover chain.


def _diag(t: Track) -> float:
    x1, y1, x2, y2 = t.bbox
    return math.hypot(x2 - x1, y2 - y1)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class CountingEngine:
    line: Line
    params: CountingParams = field(default_factory=CountingParams)

    def __post_init__(self) -> None:
        if self.params.direction not in ("forward", "reverse", "both"):
            raise ValueError(f"Invalid counting direction {self.params.direction!r}.")
        self._states: dict[int, _TrackState] = {}
        self._counted: set[int] = set()
        self._events: list[CountEvent] = []
        self.stats = CountingStats()

    # -- FrameCounter interface ------------------------------------------- #

    @property
    def total(self) -> int:
        return len(self._events)

    @property
    def counted_ids(self) -> frozenset[int]:
        return frozenset(self._counted)

    @property
    def events(self) -> list[CountEvent]:
        return list(self._events)

    def update(self, tracks: list[Track], frame_index: int) -> list[CountEvent]:
        p = self.params
        current_ids = {t.track_id for t in tracks}
        new_events: list[CountEvent] = []

        for t in tracks:
            anchor = t.anchor(p.anchor)
            st = self._states.get(t.track_id)
            if st is None:
                st = self._new_state(t, anchor, frame_index, current_ids)
                self._states[t.track_id] = st
            elif st.last_seen == frame_index:
                continue  # duplicate ID within one frame: ignore the repeat

            st.frames_seen += 1
            self._apply_teleport_guard(st, anchor, frame_index)
            self._update_side(st, anchor, frame_index)

            if st.pending and not st.counted and st.frames_seen >= p.min_track_frames:
                new_events.append(self._register(st, frame_index))

            st.last_seen = frame_index
            st.last_anchor = anchor
            st.last_diag = _diag(t)

        self._prune(frame_index)
        return new_events

    # -- internals -------------------------------------------------------- #

    def _new_state(self, t: Track, anchor: Point, frame: int, current_ids: set[int]) -> _TrackState:
        st = _TrackState(t.track_id, first_seen=frame, last_seen=frame - 1)
        st.counted = t.track_id in self._counted  # ID seen before & already counted
        donor = self._find_handover_donor(anchor, frame, current_ids)
        if donor is not None:
            donor.handed_over = True
            self.stats.handovers += 1
            st.side, st.side_point = donor.side, donor.side_point
            st.last_anchor, st.last_seen, st.last_diag = donor.last_anchor, donor.last_seen, donor.last_diag
            st.frames_seen = donor.frames_seen
            st.pending = donor.pending
            st.origin_id = donor.origin_id or donor.track_id
            if donor.counted:
                st.counted = True
                self._counted.add(t.track_id)
        return st

    def _find_handover_donor(self, anchor: Point, frame: int, current_ids: set[int]) -> _TrackState | None:
        p = self.params
        if p.handover_max_gap_frames <= 0:
            return None
        best, best_d = None, math.inf
        for s in self._states.values():
            if (s.handed_over or s.track_id in current_ids or s.last_anchor is None
                    or not 0 < frame - s.last_seen <= p.handover_max_gap_frames):
                continue
            d = _dist(anchor, s.last_anchor)
            if d <= p.handover_max_distance * s.last_diag and d < best_d:
                best, best_d = s, d
        return best

    def _apply_teleport_guard(self, st: _TrackState, anchor: Point, frame: int) -> None:
        p = self.params
        if p.max_step_ratio <= 0 or st.last_anchor is None or st.last_diag <= 0:
            return
        gap = max(frame - st.last_seen, 1)
        if _dist(anchor, st.last_anchor) > p.max_step_ratio * st.last_diag * gap:
            self.stats.jumps_rejected += 1
            st.side, st.side_point, st.pending = None, None, None

    def _update_side(self, st: _TrackState, anchor: Point, frame: int) -> None:
        zone = self.line.side(anchor, self.params.hysteresis_px)
        if zone == 0:
            return  # inside the band: confirmed side unchanged
        if st.side is None or zone == st.side:
            st.side, st.side_point = zone, anchor
            return

        # Confirmed side flipped: candidate crossing.
        direction = crossing_direction(st.side, zone)
        crossed_at = intersection_point(st.side_point, anchor, self.line.p1, self.line.p2)
        if st.pending is not None:
            # Moved back across before the pending crossing was confirmed.
            st.pending = None
            self.stats.pending_cancelled += 1
        elif crossed_at is None:
            self.stats.outside_segment += 1
        elif st.counted:
            self.stats.already_counted += 1
        elif not direction_allowed(direction, self.params.direction):
            self.stats.wrong_direction += 1
        else:
            st.pending = _Pending(frame, direction, crossed_at)
        st.side, st.side_point = zone, anchor

    def _register(self, st: _TrackState, frame: int) -> CountEvent:
        pend = st.pending
        assert pend is not None
        st.counted, st.pending = True, None
        self._counted.add(st.track_id)
        event = CountEvent(
            count=len(self._events) + 1,
            track_id=st.track_id,
            crossing_frame=pend.frame,
            confirmed_frame=frame,
            direction=pend.direction,
            position=(round(pend.position[0], 1), round(pend.position[1], 1)),
            inherited_from=st.origin_id,
        )
        self._events.append(event)
        return event

    def _prune(self, frame: int) -> None:
        timeout = self.params.track_timeout_frames
        stale = [tid for tid, s in self._states.items() if frame - s.last_seen > timeout]
        for tid in stale:
            del self._states[tid]

    @property
    def active_states(self) -> int:
        return len(self._states)


def build_counter(cfg: CountingConfig, width: int, height: int) -> CountingEngine:
    """Create an engine with the configured line converted to pixel coordinates."""
    p1, p2 = cfg.line.to_pixels(width, height)
    return CountingEngine(Line(p1, p2), CountingParams.from_config(cfg))
