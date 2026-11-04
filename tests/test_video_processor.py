"""Pipeline mechanics with a fake detector (no YOLO): EOF, cancel, failures, callbacks."""

from pathlib import Path

import numpy as np
import pytest

from src.annotator import AMBER, GREEN
from src.config import load_config
from src.video_io import validate_video_file
from src.video_processor import ProcessingOptions, process_video

TRACKER = Path("config/botsort.yaml")
OPTS = ProcessingOptions(line_px=((0, 60), (159, 60)))


@pytest.fixture
def video(make_video):
    return validate_video_file(make_video("clip.mp4", frames=30, size=(160, 120)), load_config().video)


def test_processes_every_frame_to_eof(video, fake_detector_cls):
    det = fake_detector_cls(30)
    updates = []
    res = process_video(video, det, TRACKER, OPTS, on_frame=updates.append)
    assert res.status == "completed" and res.error is None
    assert res.frames_processed == 30 == len(updates)
    assert det.resets == 1  # tracker state reset before processing
    assert res.unique_track_ids == 1
    assert res.count is None  # no counter attached in this milestone
    assert updates[-1].progress == pytest.approx(1.0)
    assert not res.warnings


def test_frames_reach_sink_annotated(video, fake_detector_cls):
    sunk = []
    process_video(video, fake_detector_cls(30), TRACKER, OPTS, frame_sink=sunk.append)
    assert len(sunk) == 30
    f = sunk[0]
    assert f.shape == (120, 160, 3)
    # Count panel (green) in the top-left and counting line (amber) at y=60.
    assert np.any(np.all(f[5:40, 5:60] == GREEN, axis=-1))
    assert np.any(np.all(f[58:63, 80:100] == AMBER, axis=-1))


def test_cancellation(video, fake_detector_cls):
    seen = []
    res = process_video(video, fake_detector_cls(30), TRACKER, OPTS,
                        on_frame=seen.append, should_stop=lambda: len(seen) >= 5)
    assert res.status == "cancelled"
    assert res.frames_processed == 5


def test_detector_failure_reported_not_raised(video, fake_detector_cls):
    res = process_video(video, fake_detector_cls(30, fail_at=10), TRACKER, OPTS)
    assert res.status == "failed"
    assert "frame 11" in res.error and "simulated inference failure" in res.error
    assert res.frames_processed == 10


def test_sink_failure_reported(video, fake_detector_cls):
    def bad_sink(_):
        raise OSError("disk full")
    res = process_video(video, fake_detector_cls(30), TRACKER, OPTS, frame_sink=bad_sink)
    assert res.status == "failed" and "disk full" in res.error


def test_max_frames_cap(video, fake_detector_cls):
    res = process_video(video, fake_detector_cls(30), TRACKER,
                        ProcessingOptions(line_px=OPTS.line_px, max_frames=7))
    assert res.status == "completed" and res.frames_processed == 7 and not res.warnings


def test_truncated_video_warns(make_video, tmp_path, fake_detector_cls):
    path = make_video("full.avi", frames=40, size=(160, 120))
    info = validate_video_file(path, load_config().video)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # truncate after validation
    res = process_video(info, fake_detector_cls(40), TRACKER, OPTS)
    assert res.status == "completed"
    assert res.frames_processed < 40
    assert any("truncated" in w for w in res.warnings)


def test_end_to_end_with_counting_engine(video, fake_detector_cls):
    from src.counting import CountingEngine, CountingParams
    from src.line_crossing import Line

    counter = CountingEngine(Line((0, 60), (159, 60)), CountingParams(direction="forward"))
    counts = []
    res = process_video(video, fake_detector_cls(30), TRACKER, OPTS, counter=counter,
                        on_frame=lambda u: counts.append(u.count))
    assert res.status == "completed"
    assert res.count == 1 and len(res.events) == 1
    assert res.events[0].track_id == 1 and res.events[0].direction == "forward"
    # Live count is monotonic and equals the number of events registered so far.
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == 1


def test_counter_hook_receives_tracks(video, fake_detector_cls):
    class SpyCounter:
        def __init__(self):
            self.frames, self.total, self.counted_ids = [], 0, set()

        def update(self, tracks, frame_index):
            self.frames.append(frame_index)
            if frame_index == 15:
                self.total, self.counted_ids = 1, {1}
                return ["event"]
            return []

    spy = SpyCounter()
    res = process_video(video, fake_detector_cls(30), TRACKER, OPTS, counter=spy)
    assert spy.frames == list(range(30))
    assert res.count == 1 and res.events == ["event"]
