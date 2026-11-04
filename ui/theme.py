"""Visual styling for the Streamlit frontend (kept out of the core package)."""

from __future__ import annotations

import html

import streamlit as st

NAVY = "#0F172A"
CHARCOAL = "#1E293B"
BORDER = "#334155"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
MUTED = "#94A3B8"

_CSS = f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1200px; }}
  .bf-header {{ display:flex; align-items:baseline; gap:.75rem; margin-bottom:.25rem; }}
  .bf-logo {{ font-size:2rem; font-weight:800; letter-spacing:-.02em; color:#F8FAFC; }}
  .bf-logo span {{ color:{GREEN}; }}
  .bf-tag {{ color:{MUTED}; font-size:.95rem; }}
  .bf-card {{ background:{CHARCOAL}; border:1px solid {BORDER}; border-radius:10px;
              padding:1rem 1.1rem; height:100%; }}
  .bf-card h4 {{ margin:0 0 .35rem 0; font-size:.8rem; text-transform:uppercase;
                 letter-spacing:.06em; color:{MUTED}; font-weight:600; }}
  .bf-card .bf-val {{ font-size:1.05rem; font-weight:600; color:#F1F5F9; }}
  .bf-card .bf-sub {{ font-size:.85rem; color:{MUTED}; margin-top:.3rem; }}
  .bf-dot {{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
             margin-right:.45rem; vertical-align:middle; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def header(subtitle: str) -> None:
    st.markdown(
        f'<div class="bf-header"><div class="bf-logo">Brick<span>Flow</span></div>'
        f'<div class="bf-tag">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def status_card(title: str, value: str, state: str, detail: str = "") -> None:
    """Render a status card. ``state`` is one of ok | warn | error."""
    color = {"ok": GREEN, "warn": AMBER, "error": RED}.get(state, MUTED)
    sub = f'<div class="bf-sub">{html.escape(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="bf-card"><h4>{html.escape(title)}</h4>'
        f'<div class="bf-val"><span class="bf-dot" style="background:{color}"></span>'
        f"{html.escape(value)}</div>{sub}</div>",
        unsafe_allow_html=True,
    )
