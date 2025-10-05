"""UI helper components for the Streamlit portfolio app."""
from __future__ import annotations

from typing import Iterable, Mapping

import streamlit as st


def apply_theme() -> None:
    """Configure page layout and title."""

    st.set_page_config(page_title="Portfolio Analytics", layout="wide", page_icon="📊")


def kpi_card(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    help_text: str | None = None,
    muted: bool = False,
) -> None:
    """Render a KPI metric card with optional delta, tooltip, and disabled styling."""

    container = st.container(border=True)
    with container:
        label_text = label if not muted else f"{label} (benchmark needed)"
        metric_value = value if not muted else "–"
        st.metric(label=label_text, value=metric_value, delta=delta, help=help_text)


def render_metric_grid(
    metrics: Mapping[str, tuple[str, str | None]],
    *,
    columns: int = 4,
    muted_keys: Iterable[str] | None = None,
) -> None:
    """Display a responsive grid of KPI cards."""

    muted = set(muted_keys or [])
    items = list(metrics.items())
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for (label, (value, tooltip)), column in zip(items[start : start + columns], cols, strict=False):
            with column:
                kpi_card(label, value, help_text=tooltip, muted=label in muted)


def badge(text: str, color: str = "#4B9CD3") -> None:
    """Render a simple badge for inline metadata."""

    st.markdown(
        f"<span style='background-color:{color}; padding:2px 6px; border-radius:6px; color:white; font-size:0.75rem'>{text}</span>",
        unsafe_allow_html=True,
    )


def chart_block(
    title: str,
    figure,
    *,
    insight: str,
    tooltip: str | None = None,
) -> None:
    """Render a chart with an optional tooltip and short insight blurb."""

    if tooltip:
        st.caption(tooltip)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"<p style='font-size:0.85rem; color:#a0a0a0'><em>{insight}</em></p>", unsafe_allow_html=True)


def two_column_charts(
    charts: Mapping[str, tuple],
) -> None:
    """Arrange charts in two-column responsive layout."""

    chart_items = list(charts.items())
    for start in range(0, len(chart_items), 2):
        cols = st.columns(2)
        for (name, (figure, insight, tooltip)), column in zip(chart_items[start : start + 2], cols, strict=False):
            with column:
                with st.expander(name, expanded=True):
                    chart_block(name, figure, insight=insight, tooltip=tooltip)
