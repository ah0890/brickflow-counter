"""Model loading: missing/invalid weights must fail with helpful messages, not crashes."""

import dataclasses

import pytest

from src.config import load_config
from src.detection import ModelLoadError, check_weights_file, load_detector


@pytest.fixture
def model_cfg():
    return load_config().model


def test_missing_weights_status(tmp_path):
    status = check_weights_file(tmp_path / "brick.pt")
    assert not status.ok
    assert "No model weights found" in status.message


def test_missing_weights_raises_model_load_error(tmp_path, model_cfg):
    cfg = dataclasses.replace(model_cfg, weights=tmp_path / "brick.pt")
    with pytest.raises(ModelLoadError, match="No model weights found"):
        load_detector(cfg)


def test_directory_instead_of_file(tmp_path, model_cfg):
    with pytest.raises(ModelLoadError, match="not a file"):
        load_detector(model_cfg, weights=tmp_path)


def test_unsupported_suffix(tmp_path, model_cfg):
    f = tmp_path / "model.txt"
    f.write_bytes(b"x" * 4096)
    with pytest.raises(ModelLoadError, match="Unsupported weights format"):
        load_detector(model_cfg, weights=f)


def test_tiny_file_rejected(tmp_path, model_cfg):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x" * 10)
    with pytest.raises(ModelLoadError, match="too small"):
        load_detector(model_cfg, weights=f)


def test_corrupt_weights_raise_model_load_error(tmp_path, model_cfg):
    f = tmp_path / "corrupt.pt"
    f.write_bytes(b"not a real pytorch checkpoint" * 200)
    with pytest.raises(ModelLoadError, match="Could not load"):
        load_detector(model_cfg, weights=f)


def test_offline_mode_enabled():
    import os

    assert os.environ.get("YOLO_OFFLINE") == "1"
