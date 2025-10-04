from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import chat, data, metrics, portfolio, ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("app")

ui.apply_theme()

st.title("Portfolio Analytics Workbench")

def _read_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""


POLYGON_KEY = _read_secret("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY", "")

if "prices" not in st.session_state:
    st.session_state["prices"] = pd.DataFrame()
    st.session_state["meta"] = {}
    st.session_state["universe"] = pd.DataFrame()
    st.session_state["portfolio_state"] = portfolio.PortfolioState()
    st.session_state["returns"] = pd.DataFrame()


def cache_key(symbols: list[str], freq: str, lookback: int) -> str:
    return f"{hash(tuple(sorted(symbols)))}-{freq}-{lookback}"


@st.cache_data(show_spinner=False)
def cached_ingest(symbols: tuple[str, ...], freq: str, lookback: int, polygon_key: str | None) -> tuple[pd.DataFrame, dict]:
    frame, meta = data.fetch_prices(list(symbols), freq, lookback, polygon_key or None)
    return frame, meta


@st.cache_data(show_spinner=False)
def cached_returns(prices: pd.DataFrame, freq: str, cache_id: str) -> pd.DataFrame:
    return metrics.price_to_returns(prices, freq)


def refresh_returns(freq: str, lookback: int) -> None:
    if st.session_state["prices"].empty:
        st.session_state["returns"] = pd.DataFrame()
        return
    cache_id = cache_key(list(st.session_state["prices"].columns), freq, lookback)
    st.session_state["returns"] = cached_returns(st.session_state["prices"], freq, cache_id)


def render_summary_tab() -> None:
    st.subheader("Performance Summary")
    returns = st.session_state.get("returns", pd.DataFrame())
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    portfolio_series = metrics.portfolio_returns(returns, weights)
    summary = metrics.compute_summary(portfolio_series, st.session_state.get("frequency", "Daily"))
    cols = st.columns(5)
    labels = ["CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max Drawdown"]
    values = [summary.cagr, summary.volatility, summary.sharpe, summary.sortino, summary.max_drawdown]
    for col, label, value in zip(cols, labels, values, strict=False):
        with col:
            ui.kpi_card(label, f"{value:.2%}" if isinstance(value, float) and pd.notna(value) else "–")

    st.markdown("### Equity Curve")
    if not portfolio_series.empty:
        spark = (1 + portfolio_series).cumprod()
        st.line_chart(spark, height=180, use_container_width=True)
    else:
        st.info("Build a portfolio in the Data Ingestion tab to see results.")

    st.markdown("### Context-aware Assistant")
    provider = st.selectbox("Provider", ["Groq", "Gemini"], key="chat_provider")
    model = st.text_input("Model", value="mixtral-8x7b-32768" if provider == "Groq" else "gemini-pro", key="chat_model")
    api_key_default = st.session_state.get("chat_api_key", "")
    api_key = st.text_input("API Key", type="password", value=api_key_default, help="Stored in memory only.")
    st.session_state["chat_api_key"] = api_key
    prompt = st.text_area("Question", placeholder="Ask about portfolio performance…")
    summary_dict = {
        "kpis": summary.to_dict(),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
    }
    system_prompt = chat.build_system_prompt(summary_dict, {"symbols": portfolio_state.symbols})
    if st.button("Send", type="primary"):
        if not api_key:
            st.warning("Provide an API key to chat.")
        elif not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                if provider == "Groq":
                    response = chat.call_groq(api_key, model, system_prompt, prompt)
                else:
                    response = chat.call_gemini(api_key, model, system_prompt, prompt)
                st.markdown("**Assistant**")
                st.write(response)
            except chat.ChatProviderError as err:
                st.error(f"Chat error: {err}")


def render_ingestion_tab() -> None:
    st.subheader("Data Ingestion & Portfolio Builder")
    universe_choice = st.selectbox("Universe", ["S&P 100", "S&P 500", "Random (N)"])
    random_n = st.slider("Random selection size", min_value=5, max_value=50, value=15) if universe_choice == "Random (N)" else None
    freq = st.selectbox("Frequency", ["Daily", "Monthly"], key="frequency")
    lookback = st.slider("Lookback (years)", min_value=1, max_value=10, value=3, key="lookback")

    universe_df = data.load_universe("assets/sample_universe.csv", universe_choice, random_n)
    st.session_state["universe"] = universe_df
    st.dataframe(universe_df.head(20), use_container_width=True)

    if st.button("Ingest Universe Data", type="primary"):
        symbols = universe_df["symbol"].tolist()
        if not symbols:
            st.warning("Universe has no symbols.")
            return
        LOGGER.info("User triggered ingestion for %d symbols", len(symbols))
        with st.status("Ingesting data…", expanded=True) as status:
            progress = st.progress(0.0)
            status.write("Starting ingestion")
            try:
                prices, meta = cached_ingest(tuple(symbols), freq, lookback, POLYGON_KEY)
                total = max(len(symbols), 1)
                for idx, sym in enumerate(symbols, start=1):
                    progress.progress(idx / total)
                    status.write(f"Loaded {sym}")
                status.update(label="Ingestion complete", state="complete")
                st.session_state["prices"] = prices
                st.session_state["meta"] = meta
                refresh_returns(freq, lookback)
                st.success(f"Loaded {prices.shape[1]} tickers via {meta.get('provider', 'unknown')}.")
                if meta.get("provider_sequence"):
                    st.caption(f"Provider sequence: {' → '.join(meta['provider_sequence'])}")
                if meta.get("errors"):
                    failed = [s for s, msg in meta["errors"].items() if msg]
                    if failed:
                        st.warning(f"Missing data for {len(failed)} tickers: {', '.join(failed[:5])}{'…' if len(failed) > 5 else ''}")
            except Exception as err:  # noqa: BLE001
                status.update(label="Ingestion failed", state="error")
                st.error(f"Data ingestion error: {err}")
            finally:
                progress.progress(1.0)

    prices = st.session_state.get("prices", pd.DataFrame())
    if prices.empty:
        st.info("No prices available yet.")
        return

    st.markdown("### Portfolio Builder")
    symbols = prices.columns.tolist()
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    default_selection = portfolio_state.symbols or symbols[:5]
    selected = st.multiselect("Select tickers", options=symbols, default=default_selection)
    weights: Dict[str, float] = {}
    for sym in selected:
        default_weight = portfolio_state.normalized_weights().get(sym, 1.0 / len(selected) if selected else 0.0)
        weights[sym] = st.number_input(f"Weight for {sym}", min_value=0.0, max_value=1.0, value=float(default_weight), step=0.01)
    if selected:
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            st.warning(f"Weights sum to {weight_sum:.2f}; they will be normalized automatically.")
    st.session_state["portfolio_state"] = portfolio.PortfolioState(symbols=selected, weights=weights)
    refresh_returns(freq, lookback)

    if st.button("Dry-run portfolio", help="Auto-build diversified basket"):
        dry = portfolio.generate_dry_run(universe_df)
        st.session_state["portfolio_state"] = dry
        st.rerun()


def render_outlook_tab() -> None:
    st.subheader("Portfolio Outlook & Fund Summary")
    returns = st.session_state.get("returns", pd.DataFrame())
    if returns.empty:
        st.info("Ingest data first to view outlook.")
        return
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    portfolio_series = metrics.portfolio_returns(returns, weights)
    benchmark_option = st.selectbox("Benchmark", options=["None"] + returns.columns.tolist())
    benchmark = None
    if benchmark_option != "None":
        benchmark = returns[benchmark_option].reindex(portfolio_series.index).dropna()
    freq = st.session_state.get("frequency", "Daily")
    bars_per_year = metrics.ANNUALIZATION.get(freq, 252)
    current_year = datetime.now(timezone.utc).year
    periods = {
        "1Y": min(len(portfolio_series), bars_per_year),
        "3Y": min(len(portfolio_series), bars_per_year * 3),
        "5Y": min(len(portfolio_series), bars_per_year * 5),
        "YTD": len(portfolio_series.loc[portfolio_series.index >= pd.Timestamp(current_year, 1, 1)]),
    }
    summary = metrics.compute_summary(portfolio_series, freq)
    cols = st.columns(5)
    for col, (label, value) in zip(cols, summary.to_dict().items(), strict=False):
        with col:
            ui.kpi_card(label, f"{value:.2%}" if pd.notna(value) else "–")
    st.plotly_chart(metrics.equity_curve(portfolio_series, benchmark), use_container_width=True)
    period_perf = metrics.period_returns(portfolio_series, periods)
    st.dataframe(period_perf.to_frame("Return"), use_container_width=True)


def render_risk_tab() -> None:
    st.subheader("Risk Metrics & Diagnostics")
    returns = st.session_state.get("returns", pd.DataFrame())
    if returns.empty:
        st.info("Ingest data first.")
        return
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    portfolio_series = metrics.portfolio_returns(returns, weights)
    risk = metrics.risk_metrics(portfolio_series, st.session_state.get("frequency", "Daily"))
    st.table(pd.Series(risk).to_frame("Value"))
    corr = metrics.correlation_matrix(returns)
    heatmap = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="Viridis", origin="lower")
    heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(heatmap, use_container_width=True)
    st.plotly_chart(metrics.histogram_with_var(portfolio_series, st.session_state.get("frequency", "Daily")), use_container_width=True)
    concentration = metrics.concentration(weights)
    st.table(pd.Series(concentration).to_frame("Value"))


tabs = st.tabs([
    "Summary + Chat",
    "Data Ingestion",
    "Portfolio Outlook",
    "Risk Diagnostics",
])

with tabs[0]:
    render_summary_tab()
with tabs[1]:
    render_ingestion_tab()
with tabs[2]:
    render_outlook_tab()
with tabs[3]:
    render_risk_tab()
