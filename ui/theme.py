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
  .bf-counter {{ background:{GREEN}; color:#052e16; border-radius:14px; padding:1.1rem 1.4rem;
                 text-align:center; box-shadow:0 6px 24px rgba(34,197,94,.25); }}
  .bf-counter.bf-idle {{ background:{CHARCOAL}; color:#F1F5F9; border:1px solid {BORDER}; box-shadow:none; }}
  .bf-counter .bf-clabel {{ font-size:.85rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
  .bf-counter .bf-cvalue {{ font-size:clamp(3rem, 8vw, 5.5rem); font-weight:800; line-height:1.05;
                            font-variant-numeric:tabular-nums; }}
  .bf-counter .bf-csub {{ font-size:.85rem; opacity:.85; }}
  .bf-badge {{ display:inline-block; padding:.25rem .7rem; border-radius:999px; font-size:.8rem;
               font-weight:700; letter-spacing:.05em; text-transform:uppercase; }}
  .bf-kv {{ display:grid; grid-template-columns:auto 1fr; gap:.3rem .9rem; font-size:.9rem; }}
  .bf-kv dt {{ color:{MUTED}; }}
  .bf-kv dd {{ margin:0; color:#F1F5F9; font-variant-numeric:tabular-nums; }}
  .bf-step {{ color:{MUTED}; font-size:.85rem; letter-spacing:.06em; text-transform:uppercase; }}
  [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; }}
  @media (max-width: 900px) {{ .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
</style>
"""

_STATE_COLORS = {
    "completed": GREEN, "processing": AMBER, "loading model": AMBER, "finalizing": AMBER,
    "queued": MUTED, "cancelled": MUTED, "failed": RED,
}


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def header(subtitle: str) -> None:
    st.markdown(
        f'<div class="bf-header"><div class="bf-logo">Brick<span>Flow</span></div>'
        f'<div class="bf-tag">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def big_counter(value: int | str, label: str = "Bricks counted", sub: str = "", idle: bool = False) -> None:
    cls = "bf-counter bf-idle" if idle else "bf-counter"
    sub_html = f'<div class="bf-csub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="{cls}" role="status" aria-live="polite"><div class="bf-clabel">{html.escape(label)}</div>'
        f'<div class="bf-cvalue">{html.escape(str(value))}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> None:
    color = _STATE_COLORS.get(status, MUTED)
    st.markdown(
        f'<span class="bf-badge" style="background:{color}22;color:{color};border:1px solid {color}">'
        f"{html.escape(status)}</span>",
        unsafe_allow_html=True,
    )


def key_values(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in rows)
    st.markdown(f'<dl class="bf-kv">{body}</dl>', unsafe_allow_html=True)


def step_label(text: str) -> None:
    st.markdown(f'<div class="bf-step">{html.escape(text)}</div>', unsafe_allow_html=True)


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
