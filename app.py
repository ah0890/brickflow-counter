"""BrickFlow Streamlit entry point.

Run with:  streamlit run app.py
The UI layer only orchestrates; detection, tracking and counting live in src/.
"""

from __future__ import annotations

import streamlit as st

from src.config import AppConfig, ConfigError, ensure_directories, load_config
from src.detection import check_weights_file
from src.utils import display_path, select_device
from ui.theme import apply_theme, header, status_card

st.set_page_config(page_title="BrickFlow", page_icon="🧱", layout="wide")
apply_theme()


@st.cache_resource(show_spinner=False)
def get_config() -> AppConfig:
    cfg = load_config()
    ensure_directories(cfg)
    return cfg


@st.cache_resource(show_spinner=False)
def get_device(requested: str):
    return select_device(requested)


def render_home(cfg: AppConfig) -> None:
    header("Automated brick counting from video")
    st.write(
        "BrickFlow detects bricks with a YOLO11 model, follows each one with BoT-SORT "
        "tracking, and counts it once when it crosses a virtual line you place on the video. "
        "All processing runs locally on this computer."
    )

    st.subheader("System status")
    c1, c2, c3 = st.columns(3)
    with c1:
        status_card("Configuration", "Loaded", "ok", display_path(cfg.source) if cfg.source else "")
    with c2:
        dev = get_device(cfg.model.device)
        status_card(
            "Compute device",
            dev.label,
            "ok" if dev.device != "cpu" else "warn",
            dev.reason or "GPU acceleration active.",
        )
    with c3:
        ws = check_weights_file(cfg.model.weights)
        status_card(
            "Detection model",
            ws.path_display if ws.ok else "Not available",
            "ok" if ws.ok else "error",
            "Weights found (loaded when processing starts)." if ws.ok else "",
        )
    if not ws.ok:
        st.warning(
            f"**Model weights required.** {ws.message}\n\n"
            "A generic pretrained YOLO model has no brick class and cannot count bricks "
            "reliably; a model trained on brick images is needed."
        )

    st.subheader("Current settings")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Confidence threshold", f"{cfg.model.confidence:.2f}")
    s2.metric("IoU threshold", f"{cfg.model.iou:.2f}")
    s3.metric("Tracker", display_path(cfg.tracker.config))
    s4.metric("Counting direction", cfg.counting.direction.capitalize())


def render_pending(title: str, milestone: int) -> None:
    header(title)
    st.info(f"This screen is not implemented yet. It is planned for Milestone {milestone}.")


def main() -> None:
    try:
        cfg = get_config()
    except ConfigError as exc:
        header("Configuration error")
        st.error(f"BrickFlow could not start because the configuration is invalid.\n\n**{exc}**")
        st.caption("Fix config/settings.yaml and reload the page.")
        st.stop()

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Screen",
            ["Home", "Configure video", "Process", "Results"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Local processing only. No video leaves this computer.")

    if page == "Home":
        render_home(cfg)
    elif page == "Configure video":
        render_pending("Video configuration", 4)
    elif page == "Process":
        render_pending("Processing", 4)
    else:
        render_pending("Results", 5)


main()
