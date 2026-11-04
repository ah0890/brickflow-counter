"""Track records and conversion from Ultralytics-style results."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking import Track, tracks_from_ultralytics


def test_anchor_points():
    t = Track(3, (10, 20, 30, 60), 0.8, 0)
    assert t.anchor("bottom_center") == (20, 60)
    assert t.anchor("center") == (20, 40)
    with pytest.raises(ValueError):
        t.anchor("top")


class _Boxes:
    def __init__(self, ids, xyxy, conf, cls):
        self.id = None if ids is None else np.array(ids)
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)

    def __len__(self):
        return len(self.xyxy)


def test_conversion():
    r = SimpleNamespace(boxes=_Boxes([7, 9], [[0, 0, 10, 10], [5, 5, 20, 25]], [0.9, 0.6], [0, 0]))
    tracks = tracks_from_ultralytics(r)
    assert [t.track_id for t in tracks] == [7, 9]
    assert tracks[1].bbox == (5, 5, 20, 25)
    assert tracks[0].confidence == pytest.approx(0.9)
    assert all(isinstance(t.track_id, int) for t in tracks)


def test_untracked_detections_are_ignored():
    # Detections with no tracker ID must never reach the counter.
    r = SimpleNamespace(boxes=_Boxes(None, [[0, 0, 10, 10]], [0.9], [0]))
    assert tracks_from_ultralytics(r) == []


def test_no_boxes():
    assert tracks_from_ultralytics(SimpleNamespace(boxes=None)) == []
    assert tracks_from_ultralytics(SimpleNamespace(boxes=_Boxes([], [], [], []))) == []
