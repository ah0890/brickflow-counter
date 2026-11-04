"""Results: final count, processed video, downloads, summary and event log."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.job import JobSnapshot, JobStatus, ProcessingJob
from src.utils import resolve_within, UnsafePathError
from ui import state
from ui.configure_labels import DIRECTION_SHORT
from ui.theme import big_counter, header, key_values, status_badge, step_label


def render(cfg: AppConfig) -> None:
    header("Results")
    job: ProcessingJob | None = st.session_state.get("job")
    if job is None:
        st.info("No results yet. Process a video to see its count here.")
        st.button("← Choose a video", on_click=state.go, args=("Home",))
        return
    snap = job.snapshot()
    if not snap.status.terminal:
        st.info("The video is still being processed.")
        st.button("Go to processing →", on_click=state.go, args=("Process",))
        return

    step_label("Step 4 of 4")
    spec = job.spec
    left, right = st.columns([2, 3], gap="large")
    with left:
        partial = snap.status != JobStatus.COMPLETED
        big_counter(snap.count, label="Bricks counted" + (" (partial)" if partial else ""),
                    sub=st.session_state.get("video_name") or spec.original_filename)
        st.write("")
        status_badge(snap.status.value)
        if snap.status == JobStatus.FAILED:
            st.error(snap.error or "Processing failed.")
        elif snap.status == JobStatus.CANCELLED:
            st.warning("Processing was cancelled; this count covers only the processed frames.")
        for w in snap.warnings:
            st.warning(w)
        if snap.model_info is not None and snap.model_info.looks_generic_coco:
            st.warning("Processed with a generic COCO model (no brick class). This number is a pipeline "
                       "demonstration, not a brick count.")
        _downloads(cfg, snap)
    with right:
        _video(cfg, snap)

    st.divider()
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.subheader("Processing summary")
        key_values(_summary_rows(job, snap))
    with s2:
        st.subheader("Counting events")
        _events_table(job, snap)

    st.divider()
    st.button("Process another video", on_click=state.go, args=("Home",))


def _safe_output(cfg: AppConfig, snap: JobSnapshot, key: str):
    path = snap.outputs.get(key)
    if path is None:
        return None
    try:
        p = resolve_within(cfg.paths.outputs_dir, path)
    except UnsafePathError:
        return None
    return p if p.is_file() else None


def _video(cfg: AppConfig, snap: JobSnapshot) -> None:
    playable = _safe_output(cfg, snap, "video_web")
    if playable is not None:
        st.video(str(playable))
    elif _safe_output(cfg, snap, "video") is not None:
        st.info("The annotated video was saved but could not be converted for in-browser playback. "
                "Download it to watch it in a media player.")
    elif snap.preview is not None:
        st.image(snap.preview, caption="Last processed frame", width=460)


def _downloads(cfg: AppConfig, snap: JobSnapshot) -> None:
    items = [
        ("video_web", "Download annotated video (MP4, H.264)", "video/mp4"),
        ("video", "Download annotated video (original encoding)", "video/mp4"),
        ("summary_csv", "Download summary report (CSV)", "text/csv"),
        ("events_csv", "Download counting events (CSV)", "text/csv"),
    ]
    shown = False
    for key, label, mime in items:
        if key == "video" and _safe_output(cfg, snap, "video_web") is not None:
            continue  # offer the browser-friendly file when it exists
        path = _safe_output(cfg, snap, key)
        if path is None:
            continue
        if not shown:
            st.write("")
            st.markdown("**Downloads**")
            shown = True
        with path.open("rb") as fh:
            st.download_button(label, fh.read(), file_name=path.name, mime=mime, width="stretch",
                               key=f"dl_{key}")
    if not shown and snap.status.terminal:
        st.caption("No output files were produced for this job.")


def _summary_rows(job: ProcessingJob, snap: JobSnapshot) -> list[tuple[str, str]]:
    spec, v, r = job.spec, job.spec.video, snap.result
    p1, p2 = spec.counting.line.to_pixels(v.width, v.height)
    rows = [
        ("Video", spec.original_filename),
        ("Duration", f"{state.fmt_duration(v.duration_s)} · {v.fps:.2f} fps · {v.resolution}"),
        ("Model", spec.model.weights.name),
        ("Device", snap.model_info.device.label if snap.model_info else "–"),
        ("Confidence / IoU", f"{spec.model.confidence:.2f} / {spec.model.iou:.2f}"),
        ("Tracker", spec.tracker_config.name),
        ("Direction", DIRECTION_SHORT.get(spec.counting.direction, spec.counting.direction)),
        ("Counting line", f"{p1} → {p2} px"),
    ]
    if r is not None:
        rows += [
            ("Frames processed", f"{r.frames_processed:,} of {r.expected_frames:,}"),
            ("Processing time", f"{state.fmt_duration(r.elapsed_s)} ({r.average_fps:.1f} fps)"),
            ("Distinct track IDs", f"{r.unique_track_ids} (not a count)"),
        ]
    return rows


def _events_table(job: ProcessingJob, snap: JobSnapshot) -> None:
    if not snap.events:
        st.caption("No objects crossed the counting line.")
        return
    fps = job.spec.video.fps
    df = pd.DataFrame([{
        "#": e.count,
        "Track ID": e.track_id,
        "Time": state.fmt_duration(e.crossing_frame / fps) if fps else "",
        "Frame": e.crossing_frame + 1,
        "Direction": e.direction,
        "Inherited from": e.inherited_from if e.inherited_from is not None else "",
    } for e in snap.events])
    st.dataframe(df, hide_index=True, width="stretch", height=min(38 + 35 * len(df), 400))
