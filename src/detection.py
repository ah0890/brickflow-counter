"""Detector interface and the Ultralytics YOLO11 implementation.

The counting engine never imports this module; it only consumes plain track
data. Swapping the detector (e.g. for a permissively licensed model) means
providing another ``Detector`` implementation.

Security note: ``.pt`` weights are PyTorch pickles and can execute code when
loaded. Only load weights from sources you trust.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Disable Ultralytics network activity (usage analytics, automatic weight and
# package downloads). Must be set before ultralytics is imported anywhere.
os.environ.setdefault("YOLO_OFFLINE", "1")

from src.config import ModelConfig  # noqa: E402
from src.utils import DeviceInfo, display_path, select_device  # noqa: E402

SUPPORTED_WEIGHT_SUFFIXES = (".pt", ".onnx")

# Class names that indicate a generic COCO-pretrained model, which has no
# "brick" class and must not be presented as a brick detector.
_COCO_MARKERS = {"person", "bicycle", "car", "dog", "toothbrush"}


class ModelLoadError(RuntimeError):
    """Raised with an operator-friendly message when weights cannot be used."""


@dataclass(frozen=True)
class WeightsStatus:
    ok: bool
    message: str
    path_display: str


def check_weights_file(path: Path) -> WeightsStatus:
    """Cheap pre-flight check of a weights path without loading the model."""
    shown = display_path(path)
    if not path.exists():
        return WeightsStatus(
            False,
            "No model weights found. Train or obtain a YOLO11 model for bricks and "
            f"place the file at '{shown}' (or update model.weights in config/settings.yaml).",
            shown,
        )
    if not path.is_file():
        return WeightsStatus(False, f"'{shown}' is not a file.", shown)
    if path.suffix.lower() not in SUPPORTED_WEIGHT_SUFFIXES:
        return WeightsStatus(
            False,
            f"Unsupported weights format '{path.suffix}'. Use {', '.join(SUPPORTED_WEIGHT_SUFFIXES)}.",
            shown,
        )
    if path.stat().st_size < 1024:
        return WeightsStatus(False, f"'{shown}' is too small to be a valid model file.", shown)
    return WeightsStatus(True, "Weights file found.", shown)


@dataclass(frozen=True)
class ModelInfo:
    weights_name: str
    class_names: dict[int, str]
    task: str
    device: DeviceInfo
    looks_generic_coco: bool


@runtime_checkable
class Detector(Protocol):
    """Minimal interface the processing pipeline depends on."""

    info: ModelInfo

    def track(self, frame: Any, tracker_config: Path) -> Any:
        """Run detection + tracking on one BGR frame (persisting tracker state)."""
        ...

    def reset(self) -> None:
        """Reset tracker state before a new video."""
        ...


class UltralyticsDetector:
    """YOLO11 detector backed by Ultralytics, used with its BoT-SORT tracker."""

    def __init__(self, cfg: ModelConfig, weights: Path | None = None):
        self.cfg = cfg
        self.weights = Path(weights) if weights else cfg.weights
        status = check_weights_file(self.weights)
        if not status.ok:
            raise ModelLoadError(status.message)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("The 'ultralytics' package is not installed.") from exc

        device = select_device(cfg.device)
        try:
            self._model = YOLO(str(self.weights), task="detect")
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            raise ModelLoadError(
                f"Could not load '{self.weights.name}': {type(exc).__name__}: {exc}"
            ) from exc

        task = getattr(self._model, "task", "unknown")
        if task != "detect":
            raise ModelLoadError(
                f"'{self.weights.name}' is a '{task}' model; an object-detection model is required."
            )

        names = dict(getattr(self._model, "names", {}) or {})
        if cfg.class_ids is not None:
            unknown = [c for c in cfg.class_ids if c not in names]
            if unknown:
                raise ModelLoadError(
                    f"model.class_ids {unknown} do not exist in this model (classes: {names})."
                )

        lowered = {str(n).lower() for n in names.values()}
        self.info = ModelInfo(
            weights_name=self.weights.name,
            class_names=names,
            task=task,
            device=device,
            looks_generic_coco=len(names) >= 80 and _COCO_MARKERS <= lowered,
        )

    def track(self, frame: Any, tracker_config: Path) -> Any:
        results = self._model.track(
            frame,
            persist=True,
            tracker=str(tracker_config),
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.info.device.device,
            classes=list(self.cfg.class_ids) if self.cfg.class_ids is not None else None,
            verbose=False,
        )
        return results[0]

    def reset(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()


def load_detector(cfg: ModelConfig, weights: Path | None = None) -> UltralyticsDetector:
    """Factory used by the application; raises ModelLoadError on any problem."""
    return UltralyticsDetector(cfg, weights)
