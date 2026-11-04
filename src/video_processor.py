"""Frame-by-frame processing pipeline: read → detect+track → (count) → annotate.

UI-agnostic: progress is reported through callbacks, cancellation through a
``should_stop`` callable, and annotated frames can be handed to a sink (the
video writer). Failures are returned as a result with status ``failed``
rather than raised, so partial progress is preserved and reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.annotator import (
    AnnotationStyle,
    draw_count_panel,
    draw_counting_line,
    draw_info_bar,
    draw_tracks,
)
from src.detection import Detector
from src.tracking import Track
from src.video_io import VideoInfo, VideoValidationError, iter_frames

# A file that stops decoding before this fraction of its reported frames is
# reported as possibly truncated.
_TRUNCATION_TOLERANCE = 0.98


class FrameCounter(Protocol):
    """Interface implemented by the counting engine."""

    def update(self, tracks: list[Track], frame_index: int) -> list[Any]:
        """Process one frame's tracks; return newly registered counting events."""
        ...

    @property
    def total(self) -> int: ...

    @property
    def counted_ids(self) -> set[int]: ...


@dataclass(frozen=True)
class ProcessingOptions:
    line_px: tuple[tuple[int, int], tuple[int, int]]
    direction: str = "forward"
    anchor: str = "bottom_center"
    style: AnnotationStyle = AnnotationStyle()
    max_frames: int | None = None       # Optional hard cap (safety / previews).


@dataclass
class FrameUpdate:
    frame_index: int
    total_frames: int
    annotated: np.ndarray
    tracks: list[Track]
    count: int | None
    new_events: list[Any]
    processing_fps: float
    elapsed_s: float

    @property
    def progress(self) -> float:
        return min((self.frame_index + 1) / self.total_frames, 1.0) if self.total_frames else 0.0


@dataclass
class ProcessingResult:
    status: str                          # completed | cancelled | failed
    frames_processed: int
    expected_frames: int
    elapsed_s: float
    count: int | None
    unique_track_ids: int                # Distinct tracker IDs seen (NOT a brick count).
    events: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def average_fps(self) -> float:
        return self.frames_processed / self.elapsed_s if self.elapsed_s > 0 else 0.0


def process_video(
    video: VideoInfo,
    detector: Detector,
    tracker_config: Path,
    options: ProcessingOptions,
    counter: FrameCounter | None = None,
    on_frame: Callable[[FrameUpdate], None] | None = None,
    frame_sink: Callable[[np.ndarray], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ProcessingResult:
    """Process a validated video. Never raises for per-frame failures."""
    detector.reset()
    expected = video.frame_count
    total_for_progress = min(expected, options.max_frames) if options.max_frames and expected else expected
    p1, p2 = options.line_px

    seen_ids: set[int] = set()
    events: list[Any] = []
    warnings: list[str] = []
    processed = 0
    fps_ema = 0.0
    start = time.perf_counter()
    last = start
    status, error = "completed", None

    try:
        for index, frame in iter_frames(video.path):
            if should_stop and should_stop():
                status = "cancelled"
                break
            if options.max_frames is not None and index >= options.max_frames:
                break

            try:
                tracks = detector.track(frame, tracker_config)
            except Exception as exc:  # noqa: BLE001
                status, error = "failed", f"Detection/tracking failed at frame {index + 1}: {type(exc).__name__}: {exc}"
                break

            seen_ids.update(t.track_id for t in tracks)
            new_events = counter.update(tracks, index) if counter else []
            events.extend(new_events)
            count = counter.total if counter else None
            counted = counter.counted_ids if counter else set()

            draw_counting_line(frame, p1, p2, options.direction)
            draw_tracks(frame, tracks, counted, options.style, options.anchor)
            draw_count_panel(frame, count)
            draw_info_bar(frame, index, expected, video.fps, len(tracks))

            if frame_sink is not None:
                try:
                    frame_sink(frame)
                except Exception as exc:  # noqa: BLE001
                    status, error = "failed", f"Writing output failed at frame {index + 1}: {exc}"
                    break

            processed += 1
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            fps_ema = inst if processed == 1 else 0.9 * fps_ema + 0.1 * inst
            last = now

            if on_frame is not None:
                on_frame(FrameUpdate(index, total_for_progress, frame, tracks, count,
                                     new_events, fps_ema, now - start))
    except VideoValidationError as exc:
        status, error = "failed", str(exc)
    except MemoryError:
        status, error = "failed", "Ran out of memory while processing the video."

    if status == "completed" and expected and options.max_frames is None:
        if processed < expected * _TRUNCATION_TOLERANCE:
            warnings.append(
                f"Video stopped decoding at frame {processed} of {expected} reported frames; "
                "the file may be truncated or partially corrupted."
            )
    if status == "completed" and processed == 0:
        status, error = "failed", "No frames could be processed."

    return ProcessingResult(
        status=status,
        frames_processed=processed,
        expected_frames=expected,
        elapsed_s=time.perf_counter() - start,
        count=counter.total if counter else None,
        unique_track_ids=len(seen_ids),
        events=events,
        warnings=warnings,
        error=error,
    )
