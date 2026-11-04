"""Home: system status and video selection (upload or local sample)."""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from src.config import AppConfig
from src.detection import check_weights_file
from src.utils import display_path, generate_job_id, resolve_within, UnsafePathError
from src.video_io import VideoValidationError, validate_video_file
from ui import state
from ui.theme import header, key_values, status_card, step_label


def render(cfg: AppConfig) -> None:
    header("Automated brick counting from video")
    st.write(
        "BrickFlow detects bricks with a YOLO11 model, follows each one with BoT-SORT tracking, "
        "and counts it once when it crosses a virtual line you place on the video. "
        "All processing runs locally on this computer."
    )
    _system_status(cfg)
    st.divider()
    step_label("Step 1 of 4")
    st.subheader("Choose a video")
    _video_selection(cfg)
    _recent_jobs(cfg)


def _system_status(cfg: AppConfig) -> None:
    weights = state.list_weights(cfg)
    configured = check_weights_file(cfg.model.weights)
    c1, c2, c3 = st.columns(3)
    with c1:
        status_card("Configuration", "Loaded", "ok", display_path(cfg.source) if cfg.source else "")
    with c2:
        dev = state.get_device(cfg.model.device)
        status_card("Compute device", dev.label, "ok" if dev.device != "cpu" else "warn",
                    dev.reason or "GPU acceleration active.")
    with c3:
        if configured.ok:
            status_card("Detection model", configured.path_display, "ok", "Configured brick model found.")
        elif weights:
            status_card("Detection model", f"{len(weights)} model file(s) in models/", "warn",
                        f"Configured model '{configured.path_display}' not found; choose one in step 2.")
        else:
            status_card("Detection model", "Not available", "error", "")
    if not weights:
        st.warning(
            f"**Model weights required.** {configured.message}\n\n"
            "A generic pretrained YOLO model has no brick class and cannot count bricks reliably; "
            "a model trained on brick images is needed. See docs/TRAINING.md."
        )


def _video_selection(cfg: AppConfig) -> None:
    allowed = [e.lstrip(".") for e in cfg.video.allowed_extensions]
    tab_up, tab_local = st.tabs(["Upload a video", "Videos on this computer"])
    with tab_up:
        uploaded = st.file_uploader(
            f"Video file ({', '.join(a.upper() for a in allowed)}; up to {cfg.video.max_file_size_mb:.0f} MB)",
            type=allowed, accept_multiple_files=False, disabled=state.job_running(),
        )
        if uploaded is not None:
            _accept_upload(cfg, uploaded)
        if st.session_state.get("upload_error"):
            st.error(st.session_state.upload_error)
    with tab_local:
        samples = state.list_samples(cfg)
        if not samples:
            st.info(f"No videos found in '{display_path(cfg.paths.samples_dir)}'. "
                    "Copy MP4, AVI or MOV files there to select them here.")
        else:
            choice = st.selectbox("Video", samples, format_func=lambda p: p.name)
            if st.button("Use this video", disabled=state.job_running()):
                _accept_local(cfg, choice)

    info = st.session_state.get("video")
    if info is not None:
        st.success(f"Selected: **{state.md_escape(st.session_state.video_name)}**")
        key_values([
            ("Duration", state.fmt_duration(info.duration_s) if info.duration_s else "Unknown"),
            ("Resolution", f"{info.resolution} ({info.orientation.lower()})"),
            ("Frame rate", f"{info.fps:.2f} fps" + (" (estimated)" if info.fps_estimated else "")),
            ("Frames", f"{info.frame_count:,}" if info.frame_count else "Unknown"),
            ("Codec / size", f"{info.codec} · {state.fmt_size(info.size_bytes)}"),
        ])
        st.button("Continue to configuration →", type="primary", on_click=state.go, args=("Configure",))


def _accept_upload(cfg: AppConfig, uploaded) -> None:
    sig = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") == sig:
        return
    st.session_state.upload_sig = sig
    st.session_state.upload_error = None
    ext = Path(uploaded.name).suffix.lower()
    if ext not in cfg.video.allowed_extensions:
        st.session_state.upload_error = f"Unsupported file type '{ext}'."
        return
    if uploaded.size > cfg.video.max_file_size_mb * 1048576:
        st.session_state.upload_error = f"File exceeds the {cfg.video.max_file_size_mb:.0f} MB limit."
        return
    dest = cfg.paths.temp_dir / f"{generate_job_id(uploaded.name)}{ext}"
    try:
        with st.spinner("Checking video…"):
            with dest.open("wb") as fh:
                shutil.copyfileobj(uploaded, fh, 1 << 20)
            info = validate_video_file(dest, cfg.video)
    except (VideoValidationError, OSError) as exc:
        dest.unlink(missing_ok=True)
        st.session_state.upload_error = f"This video can't be used: {exc}"
        return
    state.set_video(cfg, info, uploaded.name, is_temp=True)


def _accept_local(cfg: AppConfig, path: Path) -> None:
    st.session_state.upload_error = None
    try:
        safe = resolve_within(cfg.paths.samples_dir, path)
        info = validate_video_file(safe, cfg.video)
    except UnsafePathError:
        st.session_state.upload_error = "That file is outside the samples folder."
        return
    except VideoValidationError as exc:
        st.session_state.upload_error = f"This video can't be used: {exc}"
        return
    state.set_video(cfg, info, path.name, is_temp=False)


def _recent_jobs(cfg: AppConfig) -> None:
    try:
        from src.report_generator import list_recent_jobs
    except ImportError:
        return
    jobs = list_recent_jobs(cfg.paths.outputs_dir, limit=8)
    if not jobs:
        return
    st.divider()
    st.subheader("Recent jobs")
    import pandas as pd

    st.dataframe(pd.DataFrame(jobs), hide_index=True, width="stretch")
