"""Video validation: accepted formats, limits, and corrupted/renamed/truncated files."""

import dataclasses

import numpy as np
import pytest

from src.config import load_config
from src.video_io import VideoValidationError, iter_frames, read_frame, validate_video_file


@pytest.fixture
def vcfg():
    return load_config().video


def test_valid_mp4(make_video, vcfg):
    info = validate_video_file(make_video("a.mp4", frames=45, size=(160, 120)), vcfg)
    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(30.0)
    assert info.frame_count == 45
    assert info.duration_s == pytest.approx(1.5)
    assert not info.fps_estimated
    assert info.orientation == "Landscape"


def test_valid_avi(make_video, vcfg):
    info = validate_video_file(make_video("a.avi", frames=10), vcfg)
    assert info.frame_count == 10


def test_portrait_detected(make_video, vcfg):
    info = validate_video_file(make_video("p.mp4", size=(96, 160)), vcfg)
    assert info.orientation == "Portrait"


def test_uppercase_extension_accepted(make_video, vcfg):
    validate_video_file(make_video("UPPER.MP4"), vcfg)


def test_missing_file(tmp_path, vcfg):
    with pytest.raises(VideoValidationError, match="not found"):
        validate_video_file(tmp_path / "nope.mp4", vcfg)


def test_unsupported_extension(tmp_path, vcfg):
    f = tmp_path / "clip.mkv"
    f.write_bytes(b"\x1aE\xdf\xa3" + b"\0" * 100)
    with pytest.raises(VideoValidationError, match="Unsupported file type"):
        validate_video_file(f, vcfg)


def test_empty_file(tmp_path, vcfg):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with pytest.raises(VideoValidationError, match="empty"):
        validate_video_file(f, vcfg)


def test_random_bytes_rejected(tmp_path, vcfg):
    f = tmp_path / "corrupt.mp4"
    f.write_bytes(np.random.default_rng(0).bytes(50_000))
    with pytest.raises(VideoValidationError, match="not a recognised video"):
        validate_video_file(f, vcfg)


def test_text_renamed_to_mp4(tmp_path, vcfg):
    f = tmp_path / "notes.mp4"
    f.write_text("this is not a video" * 100)
    with pytest.raises(VideoValidationError):
        validate_video_file(f, vcfg)


def test_avi_renamed_to_mp4(make_video, tmp_path, vcfg):
    avi = make_video("real.avi")
    fake = tmp_path / "renamed.mp4"
    fake.write_bytes(avi.read_bytes())
    with pytest.raises(VideoValidationError, match="does not match"):
        validate_video_file(fake, vcfg)


def test_valid_header_garbage_body(tmp_path, vcfg):
    f = tmp_path / "header_only.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12 + b"\xff" * 5000)
    with pytest.raises(VideoValidationError, match="could not be opened|No frames"):
        validate_video_file(f, vcfg)


def test_size_limit(make_video, vcfg):
    path = make_video("big.mp4", frames=30)
    tiny = dataclasses.replace(vcfg, max_file_size_mb=path.stat().st_size / 1024 / 1024 / 2)
    with pytest.raises(VideoValidationError, match="limit"):
        validate_video_file(path, tiny)


def test_duration_limit(make_video, vcfg):
    path = make_video("long.mp4", frames=90)  # 3 s
    short = dataclasses.replace(vcfg, max_duration_s=1)
    with pytest.raises(VideoValidationError, match="min long"):
        validate_video_file(path, short)


def test_iter_frames_reads_all(make_video):
    path = make_video("n.mp4", frames=17)
    assert sum(1 for _ in iter_frames(path)) == 17


def test_iter_frames_on_truncated_file_stops_cleanly(make_video, tmp_path):
    path = make_video("full.avi", frames=40)
    data = path.read_bytes()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(data[: len(data) // 2])
    frames = sum(1 for _ in iter_frames(cut))  # must not raise
    assert frames < 40


def test_read_frame(make_video):
    path = make_video("r.mp4", frames=10, size=(160, 120))
    assert read_frame(path, 5).shape == (120, 160, 3)
    with pytest.raises(VideoValidationError):
        read_frame(path, 500)
