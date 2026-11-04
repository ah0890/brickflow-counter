"""Background processing job with thread-safe progress snapshots.

Frontend-agnostic: a UI starts a job, polls ``snapshot()`` and may call
``cancel()``. The worker thread never touches UI code.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.annotator import AnnotationStyle
from src.config import CountingConfig, ModelConfig, OutputConfig
from src.counting import CountEvent, build_counter
from src.detection import Detector, ModelInfo, ModelLoadError, load_detector
from src.video_io import VideoInfo
from src.video_processor import FrameUpdate, ProcessingOptions, ProcessingResult, process_video

PREVIEW_MAX_SIDE = 960
PREVIEW_INTERVAL_S = 0.15


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading model"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    video: VideoInfo
    original_filename: str
    model: ModelConfig
    tracker_config: Path
    counting: CountingConfig
    output: OutputConfig
    output_dir: Path


@dataclass
class JobSnapshot:
    status: JobStatus = JobStatus.QUEUED
    message: str = "Waiting to start"
    frame_index: int = -1
    total_frames: int = 0
    count: int = 0
    active_tracks: int = 0
    fps: float = 0.0
    elapsed_s: float = 0.0
    preview: np.ndarray | None = None          # RGB, downscaled
    events: list[CountEvent] = field(default_factory=list)
    model_info: ModelInfo | None = None
    result: ProcessingResult | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, Path] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def progress(self) -> float:
        if self.status == JobStatus.COMPLETED:
            return 1.0
        if not self.total_frames or self.frame_index < 0:
            return 0.0
        return min((self.frame_index + 1) / self.total_frames, 1.0)


# Hook run after processing (e.g. finalise video, write reports). Receives the
# spec, snapshot copy and result; returns named output files.
Finalizer = Callable[[JobSpec, JobSnapshot, ProcessingResult], dict[str, Path]]
# Optional factory for a frame sink plus its close function.
SinkFactory = Callable[[JobSpec], tuple[Callable[[np.ndarray], None], Callable[[], dict[str, Path]]]]


class ProcessingJob:
    def __init__(
        self,
        spec: JobSpec,
        detector_factory: Callable[[ModelConfig], Detector] = load_detector,
        sink_factory: SinkFactory | None = None,
        finalizer: Finalizer | None = None,
    ):
        self.spec = spec
        self._detector_factory = detector_factory
        self._sink_factory = sink_factory
        self._finalizer = finalizer
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._snap = JobSnapshot(total_frames=spec.video.frame_count)
        self._thread = threading.Thread(target=self._run, name=f"brickflow-{spec.job_id}", daemon=True)
        self._last_preview = 0.0

    # -- public API ------------------------------------------------------- #

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return replace(self._snap, events=list(self._snap.events),
                           warnings=list(self._snap.warnings), outputs=dict(self._snap.outputs))

    # -- worker ----------------------------------------------------------- #

    def _set(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._snap, k, v)

    def _run(self) -> None:
        self._set(status=JobStatus.LOADING, message="Loading detection model…", started_at=time.time())
        close_sink: Callable[[], dict[str, Path]] | None = None
        try:
            try:
                detector = self._detector_factory(self.spec.model)
            except ModelLoadError as exc:
                self._finish(JobStatus.FAILED, str(exc))
                return
            self._set(model_info=detector.info)

            if self._cancel.is_set():
                self._finish(JobStatus.CANCELLED, "Cancelled before processing started.")
                return

            v = self.spec.video
            counter = build_counter(self.spec.counting, v.width, v.height)
            options = ProcessingOptions(
                line_px=(counter.line.p1, counter.line.p2),
                direction=self.spec.counting.direction,
                anchor=self.spec.counting.anchor,
                style=AnnotationStyle(show_confidence=self.spec.output.show_confidence),
            )
            sink = None
            if self._sink_factory is not None:
                sink, close_sink = self._sink_factory(self.spec)

            self._set(status=JobStatus.PROCESSING, message="Processing video…")
            result = process_video(
                v, detector, self.spec.tracker_config, options,
                counter=counter, on_frame=self._on_frame, frame_sink=sink,
                should_stop=self._cancel.is_set,
            )
            self._set(result=result, events=list(counter.events), count=counter.total,
                      warnings=list(result.warnings))

            outputs: dict[str, Path] = {}
            if close_sink is not None:
                self._set(status=JobStatus.FINALIZING, message="Finalising annotated video…")
                fn, close_sink = close_sink, None
                try:
                    outputs.update(fn())
                except Exception as exc:  # noqa: BLE001 - export failure must not lose the count
                    with self._lock:
                        self._snap.warnings.append(f"Annotated video could not be finalised: {exc}")
            if self._finalizer is not None:
                self._set(status=JobStatus.FINALIZING, message="Writing reports…")
                outputs.update(self._finalizer(self.spec, self.snapshot(), result))
            self._set(outputs=outputs)

            status = {"completed": JobStatus.COMPLETED, "cancelled": JobStatus.CANCELLED}.get(
                result.status, JobStatus.FAILED)
            message = {
                JobStatus.COMPLETED: "Processing complete.",
                JobStatus.CANCELLED: "Processing cancelled. Partial results are shown.",
            }.get(status, result.error or "Processing failed.")
            self._finish(status, message if status != JobStatus.FAILED else None, error=result.error)
        except Exception as exc:  # noqa: BLE001 - never die silently in a thread
            self._finish(JobStatus.FAILED, None, error=f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            if close_sink is not None:  # make sure writers are released on any failure
                try:
                    outputs = close_sink()
                    with self._lock:
                        self._snap.outputs.update(outputs)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._snap.warnings.append(f"Output video could not be finalised: {exc}")

    def _finish(self, status: JobStatus, message: str | None, error: str | None = None) -> None:
        if status == JobStatus.FAILED:
            error = error or message or "Processing failed."
            message = error
        self._set(status=status, message=message or status.value, error=error, finished_at=time.time())

    def _on_frame(self, u: FrameUpdate) -> None:
        now = time.perf_counter()
        preview = None
        if now - self._last_preview >= PREVIEW_INTERVAL_S or (u.total_frames and u.frame_index + 1 >= u.total_frames):
            self._last_preview = now
            preview = _to_preview(u.annotated)
        with self._lock:
            s = self._snap
            s.frame_index, s.count = u.frame_index, u.count or 0
            s.active_tracks, s.fps, s.elapsed_s = len(u.tracks), u.processing_fps, u.elapsed_s
            if u.new_events:
                s.events.extend(u.new_events)
            if preview is not None:
                s.preview = preview


def _to_preview(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(PREVIEW_MAX_SIDE / max(h, w), 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
