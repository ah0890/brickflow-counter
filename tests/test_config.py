"""Configuration loading and validation."""

import copy
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, LineConfig, load_config, parse_config
from src.utils import PROJECT_ROOT

SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"


@pytest.fixture
def raw():
    with SETTINGS.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_default_settings_load():
    cfg = load_config()
    assert cfg.counting.direction in ("forward", "reverse", "both")
    assert 0 < cfg.model.confidence <= 1
    assert cfg.tracker.config.is_file()
    assert cfg.model.weights.is_absolute()
    assert cfg.video.allowed_extensions == (".mp4", ".avi", ".mov")


def test_missing_settings_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_missing_tracker_config(tmp_path, raw):
    raw["tracker"]["config"] = str(tmp_path / "missing_tracker.yaml")
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="Tracker configuration not found"):
        load_config(p)


@pytest.mark.parametrize(
    "section,key,value,msg",
    [
        ("model", "confidence", 1.5, "between"),
        ("model", "confidence", "high", "must be a number"),
        ("model", "confidence", True, "must be a number"),
        ("model", "iou", 0, "between"),
        ("model", "imgsz", 650, "multiple of 32"),
        ("model", "device", "tpu", "model.device"),
        ("model", "class_ids", [-1], "class_ids"),
        ("counting", "direction", "sideways", "one of"),
        ("counting", "anchor", "top", "one of"),
        ("counting", "min_track_frames", 2.5, "whole number"),
        ("video", "allowed_extensions", [], "non-empty"),
        ("output", "show_confidence", "yes", "true or false"),
    ],
)
def test_invalid_values_rejected(raw, section, key, value, msg):
    bad = copy.deepcopy(raw)
    bad[section][key] = value
    with pytest.raises(ConfigError, match=msg):
        parse_config(bad)


def test_missing_section(raw):
    del raw["counting"]
    with pytest.raises(ConfigError, match="counting"):
        parse_config(raw)


def test_line_out_of_range(raw):
    raw["counting"]["line"]["x2"] = 1.2
    with pytest.raises(ConfigError, match="between"):
        parse_config(raw)


def test_degenerate_line_rejected(raw):
    raw["counting"]["line"] = {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5}
    with pytest.raises(ConfigError, match="different points"):
        parse_config(raw)


def test_line_to_pixels():
    line = LineConfig(0.1, 0.5, 0.9, 0.5)
    assert line.to_pixels(1000, 600) == ((100, 300), (900, 300))


def test_extensions_normalised(raw):
    raw["video"]["allowed_extensions"] = ["MP4", ".Avi"]
    cfg = parse_config(raw)
    assert cfg.video.allowed_extensions == (".mp4", ".avi")


def test_device_cuda_index_accepted(raw):
    raw["model"]["device"] = "CUDA:1"
    assert parse_config(raw).model.device == "cuda:1"


def test_relative_paths_resolve_to_root(raw):
    cfg = parse_config(raw, root=Path("C:/somewhere") if Path("C:/").exists() else Path("/somewhere"))
    assert cfg.paths.models_dir.name == "models"
    assert cfg.paths.models_dir.is_absolute()
