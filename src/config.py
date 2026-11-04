"""Typed, validated application configuration loaded from YAML.

The rest of the application consumes these dataclasses rather than raw
dictionaries, so invalid values are rejected once, at load time, with a
message an operator can act on.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils import PROJECT_ROOT

DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

VALID_DIRECTIONS = ("forward", "reverse", "both")
VALID_ANCHORS = ("bottom_center", "center")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class LineConfig:
    """Counting line endpoints as fractions (0-1) of frame width/height."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixels(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (round(self.x1 * width), round(self.y1 * height)),
            (round(self.x2 * width), round(self.y2 * height)),
        )


@dataclass(frozen=True)
class PathsConfig:
    models_dir: Path
    outputs_dir: Path
    temp_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    weights: Path
    confidence: float
    iou: float
    imgsz: int
    device: str
    class_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class TrackerConfig:
    config: Path


@dataclass(frozen=True)
class CountingConfig:
    anchor: str
    direction: str
    hysteresis_px: float
    min_track_frames: int
    track_timeout_frames: int
    line: LineConfig


@dataclass(frozen=True)
class VideoConfig:
    allowed_extensions: tuple[str, ...]
    max_file_size_mb: float
    max_duration_s: float


@dataclass(frozen=True)
class OutputConfig:
    show_confidence: bool
    transcode_h264: bool


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    tracker: TrackerConfig
    counting: CountingConfig
    video: VideoConfig
    output: OutputConfig
    source: Path | None = field(default=None, compare=False)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid section '{name}' in settings.")
    return value


def _get(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing setting '{where}.{key}'.")
    return section[key]


def _number(section: dict, key: str, where: str, lo: float, hi: float, integer: bool = False):
    value = _get(section, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{where}.{key}' must be a number (got {value!r}).")
    if integer and int(value) != value:
        raise ConfigError(f"'{where}.{key}' must be a whole number (got {value!r}).")
    if not lo <= value <= hi:
        raise ConfigError(f"'{where}.{key}' must be between {lo} and {hi} (got {value}).")
    return int(value) if integer else float(value)


def _choice(section: dict, key: str, where: str, options: tuple[str, ...]) -> str:
    value = _get(section, key, where)
    if value not in options:
        raise ConfigError(f"'{where}.{key}' must be one of {', '.join(options)} (got {value!r}).")
    return value


def _bool(section: dict, key: str, where: str) -> bool:
    value = _get(section, key, where)
    if not isinstance(value, bool):
        raise ConfigError(f"'{where}.{key}' must be true or false (got {value!r}).")
    return value


def _path(section: dict, key: str, where: str, root: Path) -> Path:
    value = _get(section, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{where}.{key}' must be a non-empty path.")
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def parse_line(raw: dict[str, Any], where: str = "counting.line") -> LineConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where}' must define x1, y1, x2, y2.")
    vals = {k: _number(raw, k, where, 0.0, 1.0) for k in ("x1", "y1", "x2", "y2")}
    line = LineConfig(**vals)
    if (line.x1, line.y1) == (line.x2, line.y2):
        raise ConfigError("The counting line's two endpoints must be different points.")
    return line


def parse_config(raw: dict[str, Any], root: Path = PROJECT_ROOT) -> AppConfig:
    """Validate a raw settings dictionary and build an AppConfig."""
    if not isinstance(raw, dict):
        raise ConfigError("Settings file must contain a YAML mapping.")
    raw = copy.deepcopy(raw)

    p = _section(raw, "paths")
    paths = PathsConfig(
        models_dir=_path(p, "models_dir", "paths", root),
        outputs_dir=_path(p, "outputs_dir", "paths", root),
        temp_dir=_path(p, "temp_dir", "paths", root),
    )

    m = _section(raw, "model")
    device = str(_get(m, "device", "model")).strip().lower()
    if not (device in ("auto", "cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ConfigError(f"'model.device' must be auto, cpu, cuda or cuda:N (got {device!r}).")
    imgsz = _number(m, "imgsz", "model", 32, 4096, integer=True)
    if imgsz % 32:
        raise ConfigError(f"'model.imgsz' must be a multiple of 32 (got {imgsz}).")
    class_ids = m.get("class_ids")
    if class_ids is not None:
        if not isinstance(class_ids, list) or not all(
            isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in class_ids
        ):
            raise ConfigError("'model.class_ids' must be null or a list of non-negative integers.")
        class_ids = tuple(class_ids)
    model = ModelConfig(
        weights=_path(m, "weights", "model", root),
        confidence=_number(m, "confidence", "model", 0.01, 1.0),
        iou=_number(m, "iou", "model", 0.01, 1.0),
        imgsz=imgsz,
        device=device,
        class_ids=class_ids,
    )

    t = _section(raw, "tracker")
    tracker = TrackerConfig(config=_path(t, "config", "tracker", root))

    c = _section(raw, "counting")
    counting = CountingConfig(
        anchor=_choice(c, "anchor", "counting", VALID_ANCHORS),
        direction=_choice(c, "direction", "counting", VALID_DIRECTIONS),
        hysteresis_px=_number(c, "hysteresis_px", "counting", 0, 200),
        min_track_frames=_number(c, "min_track_frames", "counting", 1, 1000, integer=True),
        track_timeout_frames=_number(c, "track_timeout_frames", "counting", 1, 100000, integer=True),
        line=parse_line(_get(c, "line", "counting")),
    )

    v = _section(raw, "video")
    exts = _get(v, "allowed_extensions", "video")
    if not isinstance(exts, list) or not exts or not all(isinstance(e, str) for e in exts):
        raise ConfigError("'video.allowed_extensions' must be a non-empty list such as ['.mp4'].")
    video = VideoConfig(
        allowed_extensions=tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts),
        max_file_size_mb=_number(v, "max_file_size_mb", "video", 1, 100_000),
        max_duration_s=_number(v, "max_duration_s", "video", 1, 86_400),
    )

    o = _section(raw, "output")
    output = OutputConfig(
        show_confidence=_bool(o, "show_confidence", "output"),
        transcode_h264=_bool(o, "transcode_h264", "output"),
    )

    return AppConfig(paths, model, tracker, counting, video, output)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate settings from YAML (defaults to config/settings.yaml)."""
    cfg_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Settings file not found: {cfg_path.name}")
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Settings file is not valid YAML: {exc}") from exc
    cfg = parse_config(raw or {})
    if not cfg.tracker.config.is_file():
        raise ConfigError(f"Tracker configuration not found: {cfg.tracker.config.name}")
    return AppConfig(**{**cfg.__dict__, "source": cfg_path.resolve()})


def ensure_directories(cfg: AppConfig) -> None:
    """Create runtime directories if they do not exist."""
    for d in (cfg.paths.models_dir, cfg.paths.outputs_dir, cfg.paths.temp_dir):
        d.mkdir(parents=True, exist_ok=True)
