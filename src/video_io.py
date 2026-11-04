"""Video validation, probing and frame reading (OpenCV).

Validation is layered so each failure gives a specific, operator-readable
reason: existence → extension → size → container signature → decodable
frames → duration limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import VideoConfig

FALLBACK_FPS = 30.0
MAX_REASONABLE_FPS = 240.0

# ISO-BMFF (MP4/MOV) top-level box types that may legitimately start a file.
_ISO_BOX_TYPES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


class VideoValidationError(ValueError):
    """Raised when a video cannot be accepted for processing."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    filename: str
    width: int
    height: int
    fps: float
    fps_estimated: bool         # True if the file reported no usable FPS.
    frame_count: int            # Container-reported; may be approximate (0 = unknown).
    duration_s: float           # 0.0 when frame count is unknown.
    codec: str
    size_bytes: int

    @property
    def resolution(self) -> str:
        return f"{self.width}×{self.height}"

    @property
    def orientation(self) -> str:
        return "Portrait" if self.height > self.width else "Landscape"


def _sniff_container(path: Path) -> str | None:
    """Identify the container from its leading bytes: 'iso' (MP4/MOV), 'avi', or None."""
    with path.open("rb") as fh:
        head = fh.read(12)
    if len(head) < 12:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[4:8] in _ISO_BOX_TYPES:
        return "iso"
    return None


_EXPECTED_CONTAINER = {".mp4": "iso", ".mov": "iso", ".avi": "avi"}


def _fourcc_str(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() and chars.strip() else "unknown"


def probe_video(path: Path) -> VideoInfo:
    """Open a video and read its properties, verifying the first frame decodes."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoValidationError("The video could not be opened. The file may be corrupted.")
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise VideoValidationError("No frames could be decoded. The file may be corrupted or empty.")
        height, width = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_estimated = not (0.0 < fps <= MAX_REASONABLE_FPS)
        if fps_estimated:
            fps = FALLBACK_FPS
        frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0)
        codec = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return VideoInfo(
        path=path,
        filename=path.name,
        width=width,
        height=height,
        fps=fps,
        fps_estimated=fps_estimated,
        frame_count=frame_count,
        duration_s=frame_count / fps if frame_count else 0.0,
        codec=codec,
        size_bytes=path.stat().st_size,
    )


def validate_video_file(path: Path | str, cfg: VideoConfig) -> VideoInfo:
    """Fully validate a local video file against the configured limits."""
    path = Path(path)
    if not path.is_file():
        raise VideoValidationError("Video file not found.")

    ext = path.suffix.lower()
    if ext not in cfg.allowed_extensions:
        allowed = ", ".join(e.lstrip(".").upper() for e in cfg.allowed_extensions)
        raise VideoValidationError(f"Unsupported file type '{ext or 'none'}'. Supported: {allowed}.")

    size = path.stat().st_size
    if size == 0:
        raise VideoValidationError("The file is empty.")
    max_bytes = cfg.max_file_size_mb * 1024 * 1024
    if size > max_bytes:
        raise VideoValidationError(
            f"File is {size / 1048576:.0f} MB; the limit is {cfg.max_file_size_mb:.0f} MB."
        )

    container = _sniff_container(path)
    expected = _EXPECTED_CONTAINER.get(ext)
    if container is None:
        raise VideoValidationError("The file is not a recognised video (unknown file signature).")
    if expected and container != expected:
        raise VideoValidationError(
            f"File content does not match its '{ext}' extension. It may be renamed or corrupted."
        )

    info = probe_video(path)
    if info.duration_s and info.duration_s > cfg.max_duration_s:
        raise VideoValidationError(
            f"Video is {info.duration_s / 60:.1f} min long; the limit is {cfg.max_duration_s / 60:.0f} min."
        )
    return info


@contextmanager
def open_capture(path: Path | str) -> Iterator[cv2.VideoCapture]:
    """Context-managed VideoCapture that is always released."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoValidationError("The video could not be opened.")
        yield cap
    finally:
        cap.release()


def iter_frames(path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) until the stream ends or stops decoding."""
    with open_capture(path) as cap:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            yield index, frame
            index += 1


def read_frame(path: Path | str, index: int = 0) -> np.ndarray:
    """Read a single frame (for previews / line placement)."""
    with open_capture(path) as cap:
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    if not ok or frame is None:
        raise VideoValidationError(f"Frame {index} could not be read.")
    return frame
