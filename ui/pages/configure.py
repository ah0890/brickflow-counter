"""Configure: counting line placement, model/thresholds, direction, then start."""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import streamlit as st
import yaml
from streamlit_image_coordinates import streamlit_image_coordinates

from src.annotator import draw_counting_line
from src.config import AppConfig, LineConfig
from src.detection import ModelLoadError, check_weights_file
from src.job import JobSpec, ProcessingJob
from src.utils import display_path, generate_job_id
from ui import state
from ui.theme import header, key_values, step_label

MIN_LINE_PX = 20
DIRECTION_LABELS = {
    "forward": "Forward – in the arrow's direction",
    "reverse": "Reverse – against the arrow",
    "both": "Both directions",
}


def render(cfg: AppConfig) -> None:
    header("Video configuration")
    info = st.session_state.get("video")
    if info is None:
        st.info("No video selected yet.")
        st.button("← Choose a video", on_click=state.go, args=("Home",))
        return
    if not info.path.is_file():
        st.error("The selected video is no longer available. Please choose it again.")
        st.session_state.video = None
        st.button("← Choose a video", on_click=state.go, args=("Home",))
        return

    step_label("Step 2 of 4")
    st.caption(f"Video: **{st.session_state.video_name}** · {info.resolution} · "
               f"{state.fmt_duration(info.duration_s)} · {info.fps:.1f} fps")

    left, right = st.columns([3, 2], gap="large")
    with left:
        _line_editor(cfg, info)
    with right:
        settings = _detection_settings(cfg)
        _start_section(cfg, info, settings)


# --------------------------------------------------------------------------- #
# Counting line
# --------------------------------------------------------------------------- #

def _line_px(info) -> tuple[tuple[int, int], tuple[int, int]]:
    ss = st.session_state
    if (ss.lx1, ss.ly1) == (ss.lx2, ss.ly2):
        return (0, 0), (0, 0)
    return LineConfig(ss.lx1 / 100, ss.ly1 / 100, ss.lx2 / 100, ss.ly2 / 100).to_pixels(info.width, info.height)


def _line_editor(cfg: AppConfig, info) -> None:
    ss = st.session_state
    st.subheader("Counting line")
    st.write("Click **two points** on the frame: the first click sets the start, the second the end. "
             "Bricks are counted when their bottom-centre point crosses the line in the arrow's direction.")

    max_frame = max(info.frame_count - 1, 0)
    if max_frame:
        st.slider("Preview frame", 0, max_frame, key="preview_frame",
                  help="Choose a representative frame for positioning the line.")
    idx = min(ss.get("preview_frame", 0), max_frame)
    try:
        frame = state.get_frame_rgb(str(info.path), info.path.stat().st_mtime, idx)
    except Exception:  # noqa: BLE001
        st.error("This frame could not be read. Try another frame.")
        return

    # Display size: fit 760×640 box.
    h, w = frame.shape[:2]
    scale = min(760 / w, 640 / h, 1.0)
    # Draw in BGR (the annotator's colour order), then convert back for display.
    disp = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2BGR)
    p1, p2 = _line_px(info)
    if p1 != p2:
        draw_counting_line(disp, (int(p1[0] * scale), int(p1[1] * scale)),
                           (int(p2[0] * scale), int(p2[1] * scale)), ss.get("direction", cfg.counting.direction))
    if ss.pending_point is not None:
        px = (int(ss.pending_point[0] * disp.shape[1]), int(ss.pending_point[1] * disp.shape[0]))
        cv2.circle(disp, px, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(disp, px, 7, (11, 158, 245), 2, cv2.LINE_AA)
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    click = streamlit_image_coordinates(disp, key="line_canvas", image_format="JPEG", jpeg_quality=85)
    if click and click != ss.last_click:
        ss.last_click = click
        fx = min(max(click["x"] / click["width"], 0.0), 1.0)
        fy = min(max(click["y"] / click["height"], 0.0), 1.0)
        if ss.pending_point is None:
            ss.pending_point = (fx, fy)
        else:
            sx, sy = ss.pending_point
            if math.hypot((fx - sx) * w, (fy - sy) * h) < MIN_LINE_PX:
                st.toast("End point is too close to the start point. Click further away.")
            else:
                ss.lx1, ss.ly1, ss.lx2, ss.ly2 = (round(v * 100, 1) for v in (sx, sy, fx, fy))
            ss.pending_point = None
        st.rerun()

    if ss.pending_point is not None:
        st.info("Start point set — now click the end point.")
        st.button("Cancel point", on_click=lambda: ss.update(pending_point=None))

    with st.expander("Fine-tune line position (% of frame)", expanded=False):
        c1, c2 = st.columns(2)
        c1.slider("Start X", 0.0, 100.0, step=0.5, key="lx1")
        c1.slider("Start Y", 0.0, 100.0, step=0.5, key="ly1")
        c2.slider("End X", 0.0, 100.0, step=0.5, key="lx2")
        c2.slider("End Y", 0.0, 100.0, step=0.5, key="ly2")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("Horizontal", on_click=_preset, args=("h",), help="Horizontal line across the middle", width="stretch")
    b2.button("Vertical", on_click=_preset, args=("v",), help="Vertical line down the middle", width="stretch")
    b3.button("Flip arrow", on_click=_preset, args=("swap",), help="Swap start and end (reverses the arrow)",
              width="stretch")
    b4.button("Reset", on_click=state.reset_line, args=(cfg,), width="stretch")
    st.caption(f"Line in pixels: {p1} → {p2}")


def _preset(kind: str) -> None:
    ss = st.session_state
    if kind == "h":
        ss.lx1, ss.ly1, ss.lx2, ss.ly2 = 5.0, 55.0, 95.0, 55.0
    elif kind == "v":
        ss.lx1, ss.ly1, ss.lx2, ss.ly2 = 50.0, 5.0, 50.0, 95.0
    elif kind == "swap":
        ss.lx1, ss.ly1, ss.lx2, ss.ly2 = ss.lx2, ss.ly2, ss.lx1, ss.ly1
    ss.pending_point = None


# --------------------------------------------------------------------------- #
# Detection / tracking / counting settings
# --------------------------------------------------------------------------- #

def _detection_settings(cfg: AppConfig) -> dict:
    ss = st.session_state
    st.subheader("Detection")
    weights = state.list_weights(cfg)
    settings: dict = {"weights": None, "class_ids": None, "model_ok": False}
    if not weights:
        st.error(f"No model files in '{display_path(cfg.paths.models_dir)}'. "
                 "Add a YOLO11 detection model trained on bricks (.pt or .onnx).")
    else:
        default = weights.index(cfg.model.weights) if cfg.model.weights in weights else 0
        chosen = st.selectbox("Model", weights, index=default, format_func=lambda p: p.name,
                              help="Weights files found in the models folder.")
        status = check_weights_file(chosen)
        if not status.ok:
            st.error(status.message)
        else:
            try:
                with st.spinner("Reading model…"):
                    minfo = state.get_model_info(str(chosen), chosen.stat().st_mtime)
                settings.update(weights=chosen, model_ok=True)
                names = minfo.class_names
                if minfo.looks_generic_coco:
                    st.warning("**This is a generic COCO model – it has no brick class.** It can demonstrate "
                               "the pipeline (e.g. select *person*), but its counts are **not brick counts**.")
                options = list(names)
                picked = st.multiselect(
                    "Classes to count", options, default=[] if minfo.looks_generic_coco else options,
                    format_func=lambda i: f"{i}: {names[i]}",
                    help="Leave empty to use all classes.",
                )
                settings["class_ids"] = tuple(picked) if picked and len(picked) < len(options) else None
                st.caption(f"Device: {minfo.device.label}")
            except ModelLoadError as exc:
                st.error(str(exc))

    settings["confidence"] = st.slider("Confidence threshold", 0.05, 0.95, float(cfg.model.confidence), 0.05,
                                       help="Detections below this confidence are ignored.")
    settings["iou"] = st.slider("Overlap (IoU) threshold", 0.10, 0.90, float(cfg.model.iou), 0.05,
                                help="Higher keeps more overlapping boxes.")

    st.subheader("Tracking")
    with st.expander(f"BoT-SORT · {display_path(cfg.tracker.config)}"):
        try:
            params = yaml.safe_load(cfg.tracker.config.read_text(encoding="utf-8")) or {}
            key_values([(k, str(v)) for k, v in params.items()])
        except (OSError, yaml.YAMLError):
            st.error("Tracker configuration could not be read.")

    st.subheader("Counting")
    ss.setdefault("direction", cfg.counting.direction)
    settings["direction"] = st.radio("Count bricks moving", list(DIRECTION_LABELS),
                                     format_func=DIRECTION_LABELS.get, key="direction")
    settings["show_confidence"] = st.checkbox("Show confidence on boxes", value=cfg.output.show_confidence)
    return settings


# --------------------------------------------------------------------------- #
# Start
# --------------------------------------------------------------------------- #

def _validate(info, settings: dict) -> list[str]:
    problems = []
    p1, p2 = _line_px(info)
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < MIN_LINE_PX:
        problems.append(f"The counting line must be at least {MIN_LINE_PX} pixels long.")
    if not settings.get("model_ok"):
        problems.append("A valid detection model must be selected.")
    if state.job_running():
        problems.append("A video is already being processed. Wait for it to finish or cancel it.")
    return problems


def _start_section(cfg: AppConfig, info, settings: dict) -> None:
    st.divider()
    problems = _validate(info, settings)
    for p in problems:
        st.warning(p)
    # Starting runs in the click callback so navigation happens before widgets render.
    st.button("Start processing", type="primary", disabled=bool(problems), width="stretch",
              on_click=_start_job, args=(cfg, info, settings))


def _start_job(cfg: AppConfig, info, settings: dict) -> None:
    ss = st.session_state
    if state.job_running():  # double-click protection
        return
    line = LineConfig(ss.lx1 / 100, ss.ly1 / 100, ss.lx2 / 100, ss.ly2 / 100)
    job_id = generate_job_id(ss.video_name or info.filename)
    spec = JobSpec(
        job_id=job_id,
        video=info,
        original_filename=ss.video_name or info.filename,
        model=replace(cfg.model, weights=settings["weights"], confidence=settings["confidence"],
                      iou=settings["iou"], class_ids=settings["class_ids"]),
        tracker_config=cfg.tracker.config,
        counting=replace(cfg.counting, direction=settings["direction"], line=line),
        output=replace(cfg.output, show_confidence=settings["show_confidence"]),
        output_dir=cfg.paths.outputs_dir / job_id,
    )
    job = _make_job(spec)
    job.start()
    ss.job = job
    ss.page = "Process"


def _make_job(spec: JobSpec) -> ProcessingJob:
    """Create the job with export/report hooks when available."""
    try:
        from src.video_export import make_sink_factory
        from src.report_generator import write_job_reports
    except ImportError:
        return ProcessingJob(spec)
    return ProcessingJob(spec, sink_factory=make_sink_factory(), finalizer=write_job_reports)
