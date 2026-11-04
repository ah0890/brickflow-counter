"""Process: live annotated preview, large count, progress, FPS, status, cancel."""

from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from src.job import JobSnapshot, JobStatus, ProcessingJob
from ui import state
from ui.configure_labels import DIRECTION_SHORT
from ui.theme import big_counter, header, key_values, status_badge, step_label

REFRESH_S = 0.5


def render(cfg: AppConfig) -> None:
    header("Processing")
    job: ProcessingJob | None = st.session_state.get("job")
    if job is None:
        st.info("Nothing is being processed. Configure a video first.")
        st.button("← Configure a video", on_click=state.go, args=("Configure",))
        return
    step_label("Step 3 of 4")
    if job.snapshot().status.terminal:
        _render(job, job.snapshot())
    else:
        _live()


@st.fragment(run_every=REFRESH_S)
def _live() -> None:
    job: ProcessingJob = st.session_state.job
    snap = job.snapshot()
    if snap.status.terminal:
        st.rerun(scope="app")  # switch to the static final view
    _render(job, snap)


def _render(job: ProcessingJob, snap: JobSnapshot) -> None:
    spec = job.spec
    v = spec.video
    left, right = st.columns([3, 2], gap="large")

    with left:
        if snap.preview is not None:
            width = 460 if v.height > v.width else "stretch"
            st.image(snap.preview, caption="Live annotated preview", width=width)
        else:
            st.info("Preparing… the first annotated frame will appear here.")

    with right:
        done = snap.status == JobStatus.COMPLETED
        big_counter(
            snap.count,
            label="Bricks counted" if done else "Bricks counted so far",
            sub=f"{len(snap.events)} crossing event(s) registered",
            idle=snap.status in (JobStatus.QUEUED, JobStatus.LOADING),
        )
        st.write("")
        status_badge(snap.status.value)
        st.caption(snap.message)

        total = snap.total_frames
        frame_no = max(snap.frame_index + 1, 0)
        label = f"Frame {frame_no:,} of {total:,}" if total else f"Frame {frame_no:,}"
        st.progress(snap.progress, text=f"{label} · {snap.progress * 100:.0f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric("Processing speed", f"{snap.fps:.1f} fps")
        m2.metric("Active tracks", snap.active_tracks)
        remaining = (total - frame_no) / snap.fps if snap.fps > 0 and total and not snap.status.terminal else 0
        m3.metric("Time left", state.fmt_duration(remaining) if remaining else "–")

        p1, p2 = _line_px(spec)
        key_values([
            ("Model", spec.model.weights.name),
            ("Confidence", f"{spec.model.confidence:.2f}"),
            ("Direction", DIRECTION_SHORT.get(spec.counting.direction, spec.counting.direction)),
            ("Counting line", f"{p1} → {p2} px"),
            ("Elapsed", state.fmt_duration(snap.elapsed_s)),
        ])
        st.write("")

        if not snap.status.terminal:
            st.button("Cancel processing", on_click=job.cancel, width="stretch",
                      help="Stops after the current frame. Partial results are kept.")
        else:
            _final_message(snap)
            st.button("View results →", type="primary", on_click=state.go, args=("Results",), width="stretch")


def _final_message(snap: JobSnapshot) -> None:
    if snap.status == JobStatus.COMPLETED:
        st.success(f"Processing complete: **{snap.count}** brick(s) counted.")
    elif snap.status == JobStatus.CANCELLED:
        st.warning("Processing was cancelled. The count covers only the processed part of the video.")
    else:
        st.error(f"Processing failed. {snap.error or ''}")
    for w in snap.warnings:
        st.warning(w)


def _line_px(spec):
    return spec.counting.line.to_pixels(spec.video.width, spec.video.height)
