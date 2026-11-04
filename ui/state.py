"""Session-state helpers and cached resources for the Streamlit frontend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import AppConfig, ensure_directories, load_config
from src.detection import SUPPORTED_WEIGHT_SUFFIXES, ModelInfo, load_detector
from src.utils import cleanup_old_files, delete_if_within, select_device
from src.video_io import VideoInfo, read_frame

PAGES = ["Home", "Configure", "Process", "Results"]
PERSISTENT_WIDGET_KEYS = ("lx1", "ly1", "lx2", "ly2", "direction", "preview_frame")
PAGE_LABELS = {
    "Home": "1 · Choose video",
    "Configure": "2 · Configure",
    "Process": "3 · Process",
    "Results": "4 · Results",
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_config() -> AppConfig:
    cfg = load_config()
    ensure_directories(cfg)
    cleanup_old_files(cfg.paths.temp_dir, cfg.paths.temp_max_age_hours)
    return cfg


@st.cache_resource(show_spinner=False)
def get_device(requested: str):
    return select_device(requested)


@st.cache_resource(show_spinner=False, max_entries=4)
def get_model_info(weights: str, mtime: float) -> ModelInfo:
    """Load a model once to read its classes and device. Raises ModelLoadError."""
    cfg = get_config()
    from dataclasses import replace

    return load_detector(replace(cfg.model, class_ids=None), Path(weights)).info


@st.cache_data(show_spinner=False, max_entries=64)
def get_frame_rgb(path: str, mtime: float, index: int) -> np.ndarray:
    return cv2.cvtColor(read_frame(path, index), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Navigation and selection state
# --------------------------------------------------------------------------- #

def init_state(cfg: AppConfig) -> None:
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("video", None)
    ss.setdefault("video_name", None)
    ss.setdefault("video_is_temp", False)
    ss.setdefault("job", None)
    if "lx1" not in ss:
        reset_line(cfg)
    ss.setdefault("pending_point", None)
    ss.setdefault("last_click", None)
    # Streamlit discards widget-bound keys when their widget is not rendered
    # (e.g. on another page); re-assigning them keeps them across pages.
    for key in PERSISTENT_WIDGET_KEYS:
        if key in ss:
            ss[key] = ss[key]


def go(page: str) -> None:
    """Navigation callback (use with on_click)."""
    st.session_state.page = page


def reset_line(cfg: AppConfig) -> None:
    line = cfg.counting.line
    ss = st.session_state
    ss.lx1, ss.ly1, ss.lx2, ss.ly2 = (round(v * 100, 1) for v in (line.x1, line.y1, line.x2, line.y2))
    ss.pending_point = None
    ss.last_click = None


def set_video(cfg: AppConfig, info: VideoInfo, display_name: str, is_temp: bool) -> None:
    """Select a validated video, discarding a previous temporary upload."""
    ss = st.session_state
    previous: VideoInfo | None = ss.get("video")
    if previous is not None and ss.get("video_is_temp") and previous.path != info.path:
        delete_if_within(cfg.paths.temp_dir, previous.path)
    ss.video, ss.video_name, ss.video_is_temp = info, display_name, is_temp
    ss.preview_frame = 0
    reset_line(cfg)


def job_running() -> bool:
    job = st.session_state.get("job")
    return job is not None and not job.snapshot().status.terminal


def list_weights(cfg: AppConfig) -> list[Path]:
    d = cfg.paths.models_dir
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_WEIGHT_SUFFIXES)


def list_samples(cfg: AppConfig) -> list[Path]:
    d = cfg.paths.samples_dir
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in cfg.video.allowed_extensions)


def fmt_duration(seconds: float) -> str:
    seconds = max(seconds, 0)
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_size(nbytes: int) -> str:
    return f"{nbytes / 1048576:.1f} MB" if nbytes >= 1048576 else f"{nbytes / 1024:.0f} KB"
