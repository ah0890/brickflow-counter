"""Integration: real YOLO11 + BoT-SORT through the Detector interface.

Skipped automatically when weights / sample footage are not present (they are
never committed). Uses the generic COCO model to track *people* — this checks
pipeline mechanics only and says nothing about brick-detection accuracy.
"""

import dataclasses
from pathlib import Path

import pytest

from src.config import load_config
from src.detection import load_detector
from src.utils import PROJECT_ROOT
from src.video_io import validate_video_file
from src.video_processor import ProcessingOptions, process_video

WEIGHTS = PROJECT_ROOT / "models" / "yolo11n.pt"
SAMPLE = PROJECT_ROOT / "samples" / "Real_video.mp4"

needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason="models/yolo11n.pt not present")
needs_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="samples/Real_video.mp4 not present")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def person_detector(cfg):
    return load_detector(dataclasses.replace(cfg.model, class_ids=(0,)), weights=WEIGHTS)


@needs_weights
def test_generic_model_loads_and_is_flagged(person_detector):
    info = person_detector.info
    assert info.task == "detect"
    assert info.looks_generic_coco  # must be flagged as not a brick model
    assert info.device.device in ("cpu",) or info.device.device.startswith("cuda")


@needs_weights
def test_invalid_class_id_rejected(cfg):
    from src.detection import ModelLoadError

    with pytest.raises(ModelLoadError, match="class_ids"):
        load_detector(dataclasses.replace(cfg.model, class_ids=(999,)), weights=WEIGHTS)


@needs_weights
@needs_sample
def test_tracks_real_footage_and_resets_ids(cfg, person_detector):
    info = validate_video_file(SAMPLE, cfg.video)
    opts = ProcessingOptions(line_px=cfg.counting.line.to_pixels(info.width, info.height), max_frames=45)

    def run():
        ids, per_frame = set(), []
        res = process_video(info, person_detector, cfg.tracker.config, opts,
                            on_frame=lambda u: (ids.update(t.track_id for t in u.tracks),
                                                per_frame.append(len(u.tracks))))
        return res, ids, per_frame

    res1, ids1, frames1 = run()
    assert res1.status == "completed" and res1.frames_processed == 45
    assert ids1, "expected people to be tracked in the opening scene"
    assert sum(1 for n in frames1 if n > 0) > 30  # tracking persists across most frames

    res2, ids2, _ = run()
    assert min(ids2) == 1  # tracker reset between videos restarts IDs
    assert ids2 == ids1     # deterministic on identical input
