"""Milestone 5: annotated video export, CSV reports and job metadata."""

import csv
import dataclasses
import json

import cv2
import numpy as np
import pytest

from src import video_export
from src.config import LineConfig, load_config
from src.job import JobSpec, JobStatus, ProcessingJob
from src.report_generator import csv_safe, fmt_timestamp, list_recent_jobs, write_job_reports
from src.video_export import AnnotatedVideoWriter, VideoExportError, make_sink_factory, transcode_h264
from src.video_io import validate_video_file

needs_ffmpeg = pytest.mark.skipif(video_export.ffmpeg_exe() is None, reason="ffmpeg not available")


@pytest.fixture
def cfg():
    return load_config()


def _spec(cfg, info, tmp_path, name="clip.mp4", transcode=True, job_id="job1"):
    return JobSpec(
        job_id=job_id, video=info, original_filename=name, model=cfg.model,
        tracker_config=cfg.tracker.config,
        counting=dataclasses.replace(cfg.counting, direction="forward", line=LineConfig(0.0, 0.5, 1.0, 0.5)),
        output=dataclasses.replace(cfg.output, transcode_h264=transcode),
        output_dir=tmp_path / "outputs" / job_id,
    )


@pytest.fixture
def info(make_video, cfg):
    # Odd height exercises the even-dimension scaling in the transcode.
    return validate_video_file(make_video("clip.mp4", frames=30, size=(160, 121)), cfg.video)


def _run(spec, fake_detector_cls, **kw):
    job = ProcessingJob(spec, detector_factory=lambda m: fake_detector_cls(30), **kw)
    job.start()
    job.join(120)
    assert not job.is_alive
    return job.snapshot()


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


# --------------------------------------------------------------------------- #
# Writer / transcode
# --------------------------------------------------------------------------- #

def test_writer_writes_all_frames_and_renames(tmp_path):
    out = tmp_path / "v" / "a.mp4"
    w = AnnotatedVideoWriter(out, 25.0, (64, 48))
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.write(np.zeros((100, 100, 3), np.uint8))  # mismatched size is resized, not rejected
    assert w.close() == out and out.is_file()
    assert not list(out.parent.glob("*.partial*"))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 11
    cap.release()


def test_writer_without_frames_produces_nothing(tmp_path):
    w = AnnotatedVideoWriter(tmp_path / "a.mp4", 30.0, (64, 48))
    assert w.close() is None
    assert not list(tmp_path.iterdir())


@needs_ffmpeg
def test_transcode_produces_h264(tmp_path):
    src = tmp_path / "raw.mp4"
    w = AnnotatedVideoWriter(src, 30.0, (65, 49))
    for i in range(15):
        w.write(np.full((49, 65, 3), i * 10, np.uint8))
    w.close()
    dst = transcode_h264(src, tmp_path / "web.mp4")
    head = dst.read_bytes()
    assert b"avc1" in head  # H.264 sample entry
    cap = cv2.VideoCapture(str(dst))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    cap.release()


@needs_ffmpeg
def test_transcode_failure_raises_and_cleans_up(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoExportError):
        transcode_h264(bad, tmp_path / "web.mp4")
    assert not (tmp_path / "web.mp4").exists()
    assert not list(tmp_path.glob("*.partial*"))


def test_missing_ffmpeg_keeps_raw_video(cfg, info, tmp_path, fake_detector_cls, monkeypatch):
    monkeypatch.setattr(video_export, "ffmpeg_exe", lambda: None)
    spec = _spec(cfg, info, tmp_path)
    snap = _run(spec, fake_detector_cls, sink_factory=make_sink_factory())
    assert snap.status == JobStatus.COMPLETED
    assert set(snap.outputs) == {"video"} and snap.outputs["video"].is_file()


# --------------------------------------------------------------------------- #
# Full job with export + reports
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_job_exports_video_and_reports(cfg, info, tmp_path, fake_detector_cls):
    spec = _spec(cfg, info, tmp_path)
    snap = _run(spec, fake_detector_cls, sink_factory=make_sink_factory(), finalizer=write_job_reports)
    assert snap.status == JobStatus.COMPLETED
    assert set(snap.outputs) == {"video_web", "summary_csv", "events_csv"}
    files = sorted(p.name for p in spec.output_dir.iterdir())
    assert files == ["job.json", "job1_annotated.mp4", "job1_events.csv", "job1_summary.csv"]

    cap = cv2.VideoCapture(str(snap.outputs["video_web"]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()

    summary = dict(_read_csv(snap.outputs["summary_csv"])[1:])
    assert summary["total_count"] == "1" and summary["status"] == "completed"
    assert summary["frames_processed"] == "30" and summary["count_direction"] == "forward"
    assert summary["line_start_px"] == "0;60"
    assert "diagnostic_already_counted" in summary

    events = _read_csv(snap.outputs["events_csv"])
    assert events[0][:3] == ["count", "track_id", "crossing_frame"]
    assert len(events) == 2 and events[1][0] == "1" and events[1][6] == "forward"
    assert int(events[1][2]) >= 1  # 1-based frame number

    meta = json.loads((spec.output_dir / "job.json").read_text())
    assert meta["count"] == 1 and meta["status"] == "completed"


def test_cancelled_job_still_writes_partial_outputs(cfg, info, tmp_path, fake_detector_cls):
    import time

    class Slow(fake_detector_cls):
        def track(self, frame, c):
            time.sleep(0.02)
            return super().track(frame, c)

    spec = _spec(cfg, info, tmp_path, transcode=False)
    job = ProcessingJob(spec, detector_factory=lambda m: Slow(30),
                        sink_factory=make_sink_factory(), finalizer=write_job_reports)
    job.start()
    while job.snapshot().frame_index < 3:
        time.sleep(0.01)
    job.cancel()
    job.join(60)
    snap = job.snapshot()
    assert snap.status == JobStatus.CANCELLED
    assert {"video", "summary_csv", "events_csv"} <= set(snap.outputs)
    summary = dict(_read_csv(snap.outputs["summary_csv"])[1:])
    assert summary["status"] == "cancelled"
    assert int(summary["frames_processed"]) < 30


def test_formula_injection_is_neutralised(cfg, info, tmp_path, fake_detector_cls):
    spec = _spec(cfg, info, tmp_path, name="=HYPERLINK(\"x\").mp4")
    snap = _run(spec, fake_detector_cls, finalizer=write_job_reports)
    summary = dict(_read_csv(snap.outputs["summary_csv"])[1:])
    assert summary["video_file"].startswith("'=")


@pytest.mark.parametrize("value,expected", [
    ("=1+1", "'=1+1"), ("+x", "'+x"), ("-x", "'-x"), ("@x", "'@x"), ("\tx", "'\tx"),
    ("normal.mp4", "normal.mp4"), (5, 5), (-3, -3),
])
def test_csv_safe(value, expected):
    assert csv_safe(value) == expected


def test_fmt_timestamp():
    assert fmt_timestamp(0) == "00:00:00.000"
    assert fmt_timestamp(3725.4567) == "01:02:05.457"


def test_list_recent_jobs(tmp_path):
    out = tmp_path / "outputs"
    for jid, created, count in [("a", "2026-01-01T10:00:00", 3), ("b", "2026-02-01T10:00:00", 7)]:
        (out / jid).mkdir(parents=True)
        (out / jid / "job.json").write_text(json.dumps(
            {"job_id": jid, "created_at": created, "video": "v.mp4", "status": "completed",
             "count": count, "model": "m.pt", "generic_model": jid == "a"}))
    (out / "broken").mkdir()
    (out / "broken" / "job.json").write_text("{not json")
    (out / ".tmp").mkdir()
    (out / "nometa").mkdir()
    jobs = list_recent_jobs(out)
    assert [j["Job"] for j in jobs] == ["b", "a"]
    assert jobs[1]["Model"] == "m.pt (generic)" and jobs[0]["Count"] == 7
    assert list_recent_jobs(out, limit=1)[0]["Job"] == "b"
    assert list_recent_jobs(tmp_path / "missing") == []
