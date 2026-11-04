"""Path safety, filename sanitisation and device selection."""

import re

import pytest

from src.utils import (
    UnsafePathError,
    display_path,
    generate_job_id,
    resolve_within,
    safe_stem,
    select_device,
    PROJECT_ROOT,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("truck load 01.mp4", "truck_load_01"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\system32\\evil.mov", "evil"),
        ("CON.mp4", "video_CON"),
        ("", "video"),
        ("....", "video"),
        ("<script>.avi", "script"),
        ("brick.video.final.MP4", "brick.video.final"),
    ],
)
def test_safe_stem(name, expected):
    assert safe_stem(name) == expected


def test_safe_stem_length_limited():
    assert len(safe_stem("a" * 500 + ".mp4")) <= 60


def test_job_id_format_and_uniqueness():
    a, b = generate_job_id("x.mp4"), generate_job_id("x.mp4")
    assert a != b
    assert re.fullmatch(r"\d{8}_\d{6}_x_[0-9a-f]{6}", a)


def test_resolve_within_accepts_child(tmp_path):
    assert resolve_within(tmp_path, "a/b.pt") == (tmp_path / "a" / "b.pt").resolve()


@pytest.mark.parametrize("bad", ["../escape.pt", "a/../../escape.pt"])
def test_resolve_within_rejects_traversal(tmp_path, bad):
    with pytest.raises(UnsafePathError):
        resolve_within(tmp_path / "base", bad)


def test_resolve_within_rejects_foreign_absolute(tmp_path):
    other = tmp_path.parent / "elsewhere.pt"
    with pytest.raises(UnsafePathError):
        resolve_within(tmp_path, other)


def test_display_path_hides_absolute_paths(tmp_path):
    assert display_path(PROJECT_ROOT / "config" / "settings.yaml") == "config/settings.yaml"
    assert display_path(tmp_path / "secret" / "file.pt") == "file.pt"


def test_select_device_cpu_explicit():
    info = select_device("cpu")
    assert info.device == "cpu"


def test_select_device_auto_returns_valid_device():
    info = select_device("auto")
    assert info.device == "cpu" or info.device.startswith("cuda:")
    if info.device == "cpu":
        assert info.reason  # fallback must be explained
