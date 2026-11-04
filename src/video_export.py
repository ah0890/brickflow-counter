"""Annotated video export.

Frames are written with OpenCV (MPEG-4 Part 2, ``mp4v``), which every OpenCV
build can encode but browsers cannot play. When enabled, the file is then
re-encoded to H.264 (yuv420p, faststart) with the ffmpeg binary bundled by
imageio-ffmpeg so it plays in the browser and in common media players.

Plugs into ``ProcessingJob`` as a ``SinkFactory``: the sink receives each
annotated BGR frame; ``close()`` releases the writer, transcodes and returns
the produced files as ``{"video": ..., "video_web": ...}``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.job import JobSpec

TRANSCODE_TIMEOUT_S = 1800


class VideoExportError(RuntimeError):
    """The annotated video could not be written."""


class AnnotatedVideoWriter:
    """Write BGR frames to an MP4 file; the file only gets its final name on close."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.fps = fps if fps and fps > 0 else 30.0
        self.size = size  # (width, height)
        self._partial = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self._writer: cv2.VideoWriter | None = None
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self._partial), cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.fps, self.size)
            if not self._writer.isOpened():
                self._writer = None
                raise VideoExportError("The video writer could not be opened (codec or disk problem).")
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> Path | None:
        """Release the writer. Returns the finished file, or None if nothing was written."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.frames_written == 0 or not self._partial.is_file():
            self._partial.unlink(missing_ok=True)
            return None
        self._partial.replace(self.path)
        return self.path


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - missing package or binary
        return None


def transcode_h264(src: Path, dst: Path, timeout: float = TRANSCODE_TIMEOUT_S) -> Path:
    """Re-encode ``src`` to browser-playable H.264 MP4. Raises VideoExportError."""
    exe = ffmpeg_exe()
    if exe is None:
        raise VideoExportError("ffmpeg is not available (install imageio-ffmpeg).")
    partial = dst.with_name(dst.stem + ".partial" + dst.suffix)
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p needs even dimensions
        "-movflags", "+faststart", str(partial),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        partial.unlink(missing_ok=True)
        raise VideoExportError("Video conversion timed out.") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise VideoExportError(f"ffmpeg could not be started: {exc}") from exc
    if proc.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown error"]
        raise VideoExportError(f"Video conversion failed: {detail[0][:200]}")
    partial.replace(dst)
    return dst


def output_paths(spec: JobSpec) -> tuple[Path, Path]:
    """(raw annotated video, browser-playable H.264 video) for a job."""
    return (spec.output_dir / f"{spec.job_id}_annotated_raw.mp4",
            spec.output_dir / f"{spec.job_id}_annotated.mp4")


def make_sink_factory(transcode: bool | None = None):
    """Build a ``SinkFactory`` for ``ProcessingJob``.

    ``transcode`` overrides ``spec.output.transcode_h264`` when not None.
    If transcoding succeeds the raw file is removed; if it fails, the raw
    file is kept and returned as ``"video"`` so the result is never lost.
    """

    def factory(spec: JobSpec) -> tuple[Callable[[np.ndarray], None], Callable[[], dict[str, Path]]]:
        raw_path, web_path = output_paths(spec)
        writer = AnnotatedVideoWriter(raw_path, spec.video.fps, (spec.video.width, spec.video.height))
        closed = False

        def close() -> dict[str, Path]:
            nonlocal closed
            if closed:
                return {}
            closed = True
            raw = writer.close()
            if raw is None:
                return {}
            do_transcode = spec.output.transcode_h264 if transcode is None else transcode
            if not do_transcode:
                return {"video": raw}
            try:
                web = transcode_h264(raw, web_path)
            except VideoExportError:
                return {"video": raw}
            raw.unlink(missing_ok=True)
            return {"video_web": web}

        return writer.write, close

    return factory
