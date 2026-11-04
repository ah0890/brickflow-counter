"""Training-data frame extraction (scripts/extract_frames.py)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "extract_frames", Path(__file__).resolve().parents[1] / "scripts" / "extract_frames.py")
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)


def test_plan_splits_by_time_not_randomly():
    train, val = ef.plan_frames(fps=30, frame_count=300, every_s=0.5, val_fraction=0.2)
    assert train == list(range(0, 240, 15))
    assert val == list(range(240, 300, 15))
    assert max(train) < min(val)  # validation strictly after training frames


def test_plan_respects_range_and_no_val():
    train, val = ef.plan_frames(fps=25, frame_count=1000, every_s=1, start_s=4, end_s=10, val_fraction=0)
    assert train[0] == 100 and train[-1] == 250 and val == []


@pytest.mark.parametrize("kw", [
    dict(fps=0, frame_count=10, every_s=1),
    dict(fps=30, frame_count=100, every_s=0),
    dict(fps=30, frame_count=100, every_s=1, val_fraction=1.0),
    dict(fps=30, frame_count=100, every_s=1, start_s=5, end_s=2),
])
def test_plan_rejects_bad_input(kw):
    with pytest.raises(ValueError):
        ef.plan_frames(**kw)


def test_extract_writes_dataset(make_video, tmp_path):
    video = make_video("site clip.mp4", frames=60, size=(96, 64))
    out = tmp_path / "ds"
    n_train, n_val = ef.extract(video, out, every_s=0.2, start_s=0, end_s=0, val_fraction=0.25)
    assert (n_train, n_val) == (8, 2)
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 8
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 2
    assert (out / "labels" / "train").is_dir() and (out / "labels" / "val").is_dir()
    data = (out / "data.yaml").read_text()
    assert "0: brick" in data and "train: images/train" in data
    assert all(" " not in p.name for p in (out / "images" / "train").iterdir())
