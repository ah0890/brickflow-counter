"""Shared fixtures: synthetic videos and a fake detector (no YOLO required)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection import ModelInfo
from src.tracking import Track
from src.utils import DeviceInfo


def write_video(path: Path, frames: int = 30, size=(160, 120), fps: float = 30.0,
                fourcc: str = "mp4v") -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not create the test video"
    for i in range(frames):
        img = np.full((h, w, 3), 40, np.uint8)
        y = int(i * (h - 20) / max(frames - 1, 1))
        cv2.rectangle(img, (w // 2 - 10, y), (w // 2 + 10, y + 20), (0, 140, 255), -1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def make_video(tmp_path):
    def _make(name="clip.mp4", **kw):
        if "fourcc" not in kw and name.lower().endswith(".avi"):
            kw["fourcc"] = "MJPG"
        return write_video(tmp_path / name, **kw)
    return _make


class FakeDetector:
    """Emits one track moving top→bottom, mimicking a tracker's output."""

    def __init__(self, frames_total: int, size=(160, 120), fail_at: int | None = None):
        self.info = ModelInfo("fake.pt", {0: "brick"}, "detect", DeviceInfo("cpu", "CPU"), False)
        self.frames_total = frames_total
        self.w, self.h = size
        self.fail_at = fail_at
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.calls = 0

    def track(self, frame, tracker_config) -> list[Track]:
        i = self.calls
        self.calls += 1
        if self.fail_at is not None and i == self.fail_at:
            raise RuntimeError("simulated inference failure")
        y = i * (self.h - 20) / max(self.frames_total - 1, 1)
        return [Track(1, (self.w / 2 - 10, y, self.w / 2 + 10, y + 20), 0.9, 0)]


@pytest.fixture
def fake_detector_cls():
    return FakeDetector
