"""Background job lifecycle (fake detector, no YOLO)."""

import dataclasses
import time

import pytest

from src.config import LineConfig, load_config
from src.detection import ModelLoadError
from src.job import JobSpec, JobStatus, ProcessingJob
from src.video_io import validate_video_file


@pytest.fixture
def spec(make_video, tmp_path):
    cfg = load_config()
    info = validate_video_file(make_video("clip.mp4", frames=30, size=(160, 120)), cfg.video)
    return JobSpec(
        job_id="test_job",
        video=info,
        original_filename="clip.mp4",
        model=cfg.model,
        tracker_config=cfg.tracker.config,
        counting=dataclasses.replace(cfg.counting, direction="forward",
                                     line=LineConfig(0.0, 0.5, 1.0, 0.5)),
        output=cfg.output,
        output_dir=tmp_path / "out",
    )


def _wait(job, timeout=30):
    job.join(timeout)
    assert not job.is_alive, "job did not finish"
    return job.snapshot()


def test_job_completes_and_counts(spec, fake_detector_cls):
    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30))
    assert job.snapshot().status == JobStatus.QUEUED
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.COMPLETED and snap.error is None
    assert snap.count == 1 and len(snap.events) == 1
    assert snap.progress == 1.0
    assert snap.preview is not None and snap.preview.shape[2] == 3
    assert snap.result.frames_processed == 30
    assert snap.started_at and snap.finished_at >= snap.started_at


def test_model_load_error_fails_cleanly(spec):
    def boom(_):
        raise ModelLoadError("No model weights found.")
    job = ProcessingJob(spec, detector_factory=boom)
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.FAILED and "No model weights" in snap.error


def test_cancel_keeps_partial_results(spec, fake_detector_cls):
    class Slow(fake_detector_cls):
        def track(self, frame, cfg):
            time.sleep(0.02)
            return super().track(frame, cfg)

    job = ProcessingJob(spec, detector_factory=lambda m: Slow(30))
    job.start()
    while job.snapshot().frame_index < 3:
        time.sleep(0.01)
    job.cancel()
    snap = _wait(job)
    assert snap.status == JobStatus.CANCELLED
    assert 0 < snap.result.frames_processed < 30


def test_detector_failure_mid_video(spec, fake_detector_cls):
    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30, fail_at=20))
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.FAILED
    assert "frame 21" in snap.error
    assert snap.count == 1  # crossing before the failure is preserved


def test_sink_and_finalizer_hooks(spec, fake_detector_cls, tmp_path):
    frames, closed, final_calls = [], [], []

    def sink_factory(s):
        def close():
            closed.append(True)
            p = tmp_path / "video.mp4"
            p.write_bytes(b"x")
            return {"video": p}
        return frames.append, close

    def finalizer(s, snap, result):
        final_calls.append((snap.count, result.status))
        return {"summary_csv": tmp_path / "summary.csv"}

    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30),
                        sink_factory=sink_factory, finalizer=finalizer)
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.COMPLETED
    assert len(frames) == 30 and closed == [True]
    assert final_calls == [(1, "completed")]
    assert set(snap.outputs) == {"video", "summary_csv"}


def test_sink_closed_even_when_finalizer_crashes(spec, fake_detector_cls):
    closed = []

    def sink_factory(s):
        return (lambda f: None), (lambda: closed.append(True) or {})

    def bad_finalizer(*a):
        raise OSError("disk full")

    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30),
                        sink_factory=sink_factory, finalizer=bad_finalizer)
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.FAILED and "disk full" in snap.error
    assert closed == [True]


def test_video_finalise_error_is_warning_not_failure(spec, fake_detector_cls):
    def sink_factory(s):
        def close():
            raise RuntimeError("encoder crashed")
        return (lambda f: None), close

    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30), sink_factory=sink_factory)
    job.start()
    snap = _wait(job)
    assert snap.status == JobStatus.COMPLETED and snap.count == 1
    assert any("encoder crashed" in w for w in snap.warnings)


def test_snapshot_is_a_copy(spec, fake_detector_cls):
    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30))
    job.start()
    snap = _wait(job)
    snap.events.clear()
    assert len(job.snapshot().events) == 1
