"""UI helper components for the Streamlit portfolio app."""
from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Configure page layout and title."""
    st.set_page_config(page_title="Portfolio Analytics", layout="wide", page_icon="📊")


def kpi_card(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    """Render a KPI metric card with optional delta and tooltip."""
    col = st.container(border=True)
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


def badge(text: str, color: str = "#4B9CD3") -> None:
    """Render a simple badge for inline metadata."""
    st.markdown(
        f"<span style='background-color:{color}; padding:2px 6px; border-radius:6px; color:white; font-size:0.75rem'>{text}</span>",
        unsafe_allow_html=True,
    )
