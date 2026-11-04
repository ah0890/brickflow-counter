"""BrickFlow Streamlit entry point.

Run with:  streamlit run app.py
The UI layer only orchestrates; detection, tracking and counting live in src/.
"""

from __future__ import annotations

import streamlit as st

from src.config import ConfigError
from ui import state
from ui.pages import configure, home, process, results
from ui.theme import apply_theme, header

st.set_page_config(page_title="BrickFlow", page_icon="🧱", layout="wide")
apply_theme()

RENDERERS = {"Home": home.render, "Configure": configure.render,
             "Process": process.render, "Results": results.render}


def main() -> None:
    try:
        cfg = state.get_config()
    except ConfigError as exc:
        header("Configuration error")
        st.error(f"BrickFlow could not start because the configuration is invalid.\n\n**{exc}**")
        st.caption("Fix config/settings.yaml and reload the page.")
        st.stop()

    state.init_state(cfg)
    with st.sidebar:
        st.markdown("### BrickFlow")
        st.radio("Screen", state.PAGES, key="page", format_func=state.PAGE_LABELS.get,
                 label_visibility="collapsed")
        st.divider()
        if state.job_running():
            snap = st.session_state.job.snapshot()
            st.caption(f"Processing… {snap.progress * 100:.0f}% · {snap.count} counted")
        st.caption("Local processing only. No video leaves this computer.")

    RENDERERS[st.session_state.page](cfg)


main()
