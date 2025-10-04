from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping

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

SESSION_DEFAULTS = {
    "prices": pd.DataFrame(),
    "meta": {},
    "universe": pd.DataFrame(),
    "portfolio_state": portfolio.PortfolioState(),
    "returns": pd.DataFrame(),
    "sector_meta": {},
    "benchmark_symbol": "None",
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def cache_key(symbols: Iterable[str], freq: str, lookback: int) -> str:
    return f"{hash(tuple(sorted(symbols)))}-{freq}-{lookback}"


@st.cache_data(show_spinner=False)
def cached_ingest(symbols: tuple[str, ...], freq: str, lookback: int, polygon_key: str | None) -> tuple[pd.DataFrame, dict]:
    frame, meta = data.fetch_prices(list(symbols), freq, lookback, polygon_key or None)
    return frame, meta


@st.cache_data(show_spinner=False)
def cached_returns(prices: pd.DataFrame, freq: str, cache_id: str) -> pd.DataFrame:
    return metrics.price_to_returns(prices, freq)


@st.cache_data(show_spinner=False)
def cached_extended_metrics(
    portfolio_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weights: tuple[tuple[str, float], ...],
    benchmark_df: pd.DataFrame | None,
    freq: str,
) -> dict[str, float]:
    benchmark_series = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_series = benchmark_df.iloc[:, 0]
    return metrics.compute_extended_metrics(
        portfolio_df.iloc[:, 0],
        asset_returns=asset_returns,
        weights=dict(weights),
        benchmark=benchmark_series,
        freq=freq,
    )


def refresh_returns(freq: str, lookback: int) -> None:
    prices = st.session_state.get("prices", pd.DataFrame())
    if prices.empty:
        st.session_state["returns"] = pd.DataFrame()
        return
    cache_id = cache_key(list(prices.columns), freq, lookback)
    st.session_state["returns"] = cached_returns(prices, freq, cache_id)


def _format_value(name: str, value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if name in {"Ann. Vol", "Max Drawdown", "Upside Capture", "Downside Capture", "Hit Rate", "Avg Up Day", "Avg Down Day", "Tail Ratio", "Pain Ratio", "ES 99%", "Cornish-Fisher VaR", "CDaR 95%", "Rolling VaR"}:
        return f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
    if name in {"CAGR", "MAR Ratio", "Calmar Ratio"}:
        return f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
    if name in {"Sharpe", "Sortino", "Treynor Ratio", "Information Ratio", "Kelly Fraction"}:
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if name.startswith("RC "):
        return f"{value:.1%}" if isinstance(value, (int, float)) else str(value)
    if name in {"Skew", "Kurtosis", "Rolling Beta"}:
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if isinstance(value, (int, float)):
        return f"{value:.2%}" if name.endswith("Rate") else f"{value:.2f}"
    return str(value)


def _log_block(
    name: str,
    start: float,
    *,
    count: int,
    portfolio_size: int,
    benchmark: bool,
    window: int | float,
) -> None:
    duration = time.perf_counter() - start
    LOGGER.info(
        "%s rendered in %.3fs | metrics=%d | portfolio_points=%d | window=%s | benchmark=%s",
        name,
        duration,
        count,
        portfolio_size,
        window,
        benchmark,
    )


def _sector_meta(universe: pd.DataFrame) -> Mapping[str, Mapping[str, str]]:
    if universe.empty:
        return {}
    return {row.symbol: {"sector": row.sector} for row in universe.itertuples(index=False)}


def _render_chart_grid(
    figures: Mapping[str, tuple],
    *,
    prefix: str,
    portfolio_size: int,
    benchmark: bool,
    window: int | float,
) -> None:
    items = list(figures.items())
    for idx in range(0, len(items), 2):
        cols = st.columns(2)
        for (name, (figure, insight, tooltip)), column in zip(items[idx : idx + 2], cols, strict=False):
            with column:
                chart_start = time.perf_counter()
                with st.expander(name, expanded=True):
                    ui.chart_block(name, figure, insight=insight, tooltip=tooltip)
                _log_block(
                    f"{prefix}_{name}",
                    chart_start,
                    count=1,
                    portfolio_size=portfolio_size,
                    benchmark=benchmark,
                    window=window,
                )


def render_summary_tab() -> None:
    st.subheader("Performance Summary & Copilot")
    returns = st.session_state.get("returns", pd.DataFrame())
    freq = st.session_state.get("frequency", "Daily")
    lookback = st.session_state.get("lookback", 3)
    universe = st.session_state.get("universe", pd.DataFrame())
    sector_meta = st.session_state.get("sector_meta", {}) or _sector_meta(universe)
    st.session_state["sector_meta"] = sector_meta

    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    asset_returns = returns[portfolio_state.symbols].dropna(how="all") if not returns.empty else pd.DataFrame()
    portfolio_series = metrics.portfolio_returns(returns, weights)

    benchmark_options = ["None"] + returns.columns.tolist()
    current_benchmark = st.session_state.get("benchmark_symbol", "None")
    if current_benchmark not in benchmark_options:
        current_benchmark = "None"
    benchmark_symbol = st.selectbox("Benchmark", options=benchmark_options, index=benchmark_options.index(current_benchmark))
    st.session_state["benchmark_symbol"] = benchmark_symbol
    benchmark_series = None
    if benchmark_symbol != "None" and not returns.empty:
        benchmark_series = returns[benchmark_symbol].reindex(portfolio_series.index).dropna()

    extended_metrics = cached_extended_metrics(
        portfolio_series.to_frame("portfolio") if not portfolio_series.empty else pd.DataFrame(columns=["portfolio"]),
        asset_returns,
        tuple(sorted(weights.items())),
        benchmark_series.to_frame("benchmark") if benchmark_series is not None else None,
        freq,
    )

    benchmark_sensitive = {
        "Treynor Ratio",
        "CAPM Alpha",
        "CAPM Beta",
        "Tracking Error",
        "Information Ratio",
        "Upside Capture",
        "Downside Capture",
        "Rolling Beta",
    }

    metric_tooltips = {
        "CAGR": "Compound annual growth rate based on observed sample.",
        "Ann. Vol": "Annualized realized volatility.",
        "Sharpe": "Sharpe ratio using sample mean/vol.",
        "Sortino": "Sortino ratio using downside deviation.",
        "Max Drawdown": "Worst peak-to-trough decline.",
        "Calmar Ratio": "CAGR divided by max drawdown magnitude.",
        "MAR Ratio": "Annual return divided by max drawdown.",
        "Omega Ratio": "Upside vs downside tail probability ratio.",
        "Ulcer Index": "Root mean square drawdown depth.",
        "Treynor Ratio": "Excess return per unit beta.",
        "CAPM Alpha": "Average excess return vs benchmark.",
        "CAPM Beta": "Systematic sensitivity to benchmark.",
        "Tracking Error": "Std dev of active returns.",
        "Information Ratio": "Active return divided by tracking error.",
        "Upside Capture": "Average relative gain in up periods.",
        "Downside Capture": "Average relative loss in down periods.",
        "Hit Rate": "Share of positive-return periods.",
        "Avg Up Day": "Mean gain on positive days.",
        "Avg Down Day": "Mean loss on negative days.",
        "Skew": "Distribution skewness.",
        "Kurtosis": "Distribution kurtosis.",
        "Tail Ratio": "95th percentile vs 5th percentile.",
        "Pain Ratio": "CAGR divided by average drawdown.",
        "Cornish-Fisher VaR": "Adjusted VaR using Cornish-Fisher expansion.",
        "ES 99%": "Expected shortfall at 99% confidence.",
        "CDaR 95%": "Average of worst 5% drawdowns.",
        "Diversification Ratio": "Weighted avg vol divided by portfolio vol.",
        "Kelly Fraction": "Fraction of capital optimal for Kelly sizing.",
        "Rolling VaR": "Most recent rolling VaR estimate.",
        "Rolling Beta": "Most recent rolling beta estimate.",
        "VaR Breaches": "Count of returns breaching 95% VaR.",
    }

    selected_metrics = {k: extended_metrics.get(k) for k in metric_tooltips if k in extended_metrics}
    rc_metrics = {k: v for k, v in extended_metrics.items() if k.startswith("RC ")}
    summary_metrics = selected_metrics | rc_metrics
    formatted = {name: (_format_value(name, val), metric_tooltips.get(name, "")) for name, val in summary_metrics.items()}

    kpi_start = time.perf_counter()
    ui.render_metric_grid(formatted, columns=4, muted_keys=benchmark_sensitive if benchmark_series is None else [])
    _log_block(
        "summary_kpis",
        kpi_start,
        count=len(formatted),
        portfolio_size=len(portfolio_series),
        benchmark=benchmark_series is not None,
        window=lookback,
    )

    if portfolio_series.empty:
        st.info("Build a portfolio in the Data Ingestion tab to see analytics.")
    else:
        figures = metrics.assemble_summary_figures(
            portfolio_series,
            asset_returns=asset_returns,
            weights=weights,
            benchmark=benchmark_series,
            freq=freq,
            meta=sector_meta,
        )
        _render_chart_grid(
            figures,
            prefix="summary_chart",
            portfolio_size=len(portfolio_series),
            benchmark=benchmark_series is not None,
            window=lookback,
        )

    summary_payload = {
        "kpis": {k: float(v) if isinstance(v, (int, float)) and not pd.isna(v) else None for k, v in summary_metrics.items()},
        "weights": weights,
        "symbols": portfolio_state.symbols,
        "benchmark": benchmark_symbol,
        "provider": st.session_state.get("meta", {}).get("provider"),
        "frequency": freq,
        "lookback_years": lookback,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state["summary_payload"] = summary_payload

    st.markdown("### Context-aware Assistant")
    provider = st.selectbox("Provider", ["Groq", "Gemini"], key="chat_provider")
    default_model = "mixtral-8x7b-32768" if provider == "Groq" else "gemini-pro"
    model = st.text_input("Model", value=st.session_state.get("chat_model", default_model), key="chat_model")
    api_key = st.text_input("API Key", type="password", value=st.session_state.get("chat_api_key", ""), help="Stored in memory only.")
    st.session_state["chat_api_key"] = api_key
    prompt = st.text_area("Question", placeholder="Ask about portfolio performance…")
    system_prompt = chat.build_system_prompt(summary_payload, {"symbols": portfolio_state.symbols})
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


def _ingestion_kpis(prices: pd.DataFrame, universe: pd.DataFrame, meta: Mapping[str, object]) -> dict[str, tuple[str, str | None]]:
    provider = meta.get("provider", "–") if isinstance(meta, Mapping) else "–"
    errors = meta.get("errors", {}) if isinstance(meta, Mapping) else {}
    missing = sum(1 for val in errors.values() if val)
    coverage = 1 - prices.isna().sum().sum() / prices.size if prices.size else 0
    start = prices.dropna(how="all").index.min() if not prices.empty else None
    end = prices.dropna(how="all").index.max() if not prices.empty else None
    kpis = {
        "Universe Symbols": (str(len(universe)), "Universe size after selection."),
        "Ingested Symbols": (str(prices.shape[1]), "Count of tickers with data."),
        "Provider": (str(provider), "Active data provider."),
        "Fallback Count": (str(len(meta.get("provider_sequence", [])) if isinstance(meta, Mapping) else 0), "Providers attempted in order."),
        "Missing Tickers": (str(missing), "Symbols lacking data."),
        "Coverage": (f"{coverage:.1%}", "Non-null observations share."),
        "Price Start": (start.strftime("%Y-%m-%d") if start else "–", "First available observation."),
        "Price End": (end.strftime("%Y-%m-%d") if end else "–", "Latest observation."),
        "Rows": (str(prices.shape[0]), "Number of time periods ingested."),
        "Universe Sectors": (str(universe["sector"].nunique() if not universe.empty else 0), "Distinct sectors represented."),
        "Lookback Years": (str(st.session_state.get("lookback", 3)), "Requested history length."),
        "Frequency": (st.session_state.get("frequency", "Daily"), "Sampling frequency."),
    }
    return kpis


def _ingestion_figures(prices: pd.DataFrame, returns: pd.DataFrame, universe: pd.DataFrame, weights: Dict[str, float]) -> dict[str, tuple]:
    figures: dict[str, tuple] = {}
    sector_counts = universe["sector"].value_counts().sort_values(ascending=False) if not universe.empty else pd.Series(dtype=int)
    figures["Sector Distribution"] = (
        px.bar(sector_counts, orientation="v", title=""),
        "Universe composition by sector.",
        None,
    )
    default_weights: Dict[str, float] = weights
    if not default_weights and not universe.empty:
        equal_weight = 1 / len(universe)
        default_weights = {row.symbol: equal_weight for row in universe.itertuples(index=False)}
    figures["Sector Donut"] = (
        metrics.sector_donut(default_weights, st.session_state.get("sector_meta", {})),
        "Weighted sector mix.",
        None,
    )
    if not prices.empty:
        sample = prices.iloc[-200:]
        figures["Price Sample"] = (
            px.line(sample, title=""),
            "Recent price trajectories for ingested symbols.",
            "Last 200 periods.",
        )
        availability = prices.notna().astype(int)
        figures["Availability Heatmap"] = (
            px.imshow(availability.T, aspect="auto", color_continuous_scale="Blues", title=""),
            "Data availability by date/ticker.",
            "1 indicates data present.",
        )
        observation_count = prices.notna().sum()
        figures["Observation Count"] = (
            px.bar(observation_count, title=""),
            "Valid observations per ticker.",
            None,
        )
        mean_price = prices.mean(axis=1)
        figures["Average Price"] = (
            px.line(mean_price, title=""),
            "Universe average price level.",
            None,
        )
    if not returns.empty:
        melted = returns.reset_index().melt(id_vars=returns.index.name or "index", var_name="symbol", value_name="return")
        figures["Return Histogram"] = (
            px.histogram(melted, x="return", color="symbol", nbins=40, opacity=0.5, title=""),
            "Return distribution per asset.",
            None,
        )
        figures["Return Box"] = (
            px.box(melted, x="symbol", y="return", title=""),
            "Cross-sectional return dispersion.",
            None,
        )
        vol = returns.std()
        figures["Volatility Bar"] = (
            px.bar(vol, title=""),
            "Per-asset realized volatility.",
            None,
        )
        figures["Rolling Coverage"] = (
            px.line(returns.notna().sum(axis=1), title=""),
            "Number of active series over time.",
            None,
        )
        figures["Mean Return"] = (
            px.bar(returns.mean(), title=""),
            "Average return by ticker.",
            None,
        )
        figures["Risk/Return"] = (
            metrics.risk_return_scatter(returns, st.session_state.get("frequency", "Daily")),
            "Mean-variance map for assets.",
            None,
        )
    return figures


def render_ingestion_tab() -> None:
    st.subheader("Data Ingestion & Portfolio Builder")
    universe_choice = st.selectbox("Universe", ["S&P 100", "S&P 500", "Random (N)"])
    random_n = st.slider("Random selection size", min_value=5, max_value=50, value=15) if universe_choice == "Random (N)" else None
    freq = st.selectbox("Frequency", ["Daily", "Monthly"], key="frequency")
    lookback = st.slider("Lookback (years)", min_value=1, max_value=10, value=3, key="lookback")

    universe_df = data.load_universe("assets/sample_universe.csv", universe_choice, random_n)
    st.session_state["universe"] = universe_df
    st.session_state["sector_meta"] = _sector_meta(universe_df)
    st.dataframe(universe_df.head(30), use_container_width=True)

    if st.button("Ingest Universe Data", type="primary"):
        symbols = universe_df["symbol"].tolist()
        if not symbols:
            st.warning("Universe has no symbols.")
        else:
            LOGGER.info("User triggered ingestion for %d symbols", len(symbols))
            with st.status("Ingesting data…", expanded=True) as status:
                progress = st.progress(0.0)
                status.write("Starting ingestion")

                def _progress(idx: int, total: int, sym: str) -> None:
                    progress.progress(idx / max(total, 1))
                    status.write(f"Loaded {sym}")

                try:
                    prices, meta = data.fetch_prices(symbols, freq, lookback, POLYGON_KEY, progress_callback=_progress)
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
    returns = st.session_state.get("returns", pd.DataFrame())

    kpi_start = time.perf_counter()
    ingestion_kpis = _ingestion_kpis(prices, universe_df, st.session_state.get("meta", {}))
    ui.render_metric_grid(ingestion_kpis, columns=4)
    _log_block(
        "ingestion_kpis",
        kpi_start,
        count=len(ingestion_kpis),
        portfolio_size=prices.shape[0],
        benchmark=False,
        window=lookback,
    )

    figures = _ingestion_figures(
        prices,
        returns,
        universe_df,
        st.session_state.get("portfolio_state", portfolio.PortfolioState()).normalized_weights(),
    )
    _render_chart_grid(
        figures,
        prefix="ingestion_chart",
        portfolio_size=prices.shape[0],
        benchmark=False,
        window=lookback,
    )

    if prices.empty:
        st.info("No prices available yet. Ingest data to activate the portfolio builder.")
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
    freq = st.session_state.get("frequency", "Daily")
    lookback = st.session_state.get("lookback", 3)
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    portfolio_series = metrics.portfolio_returns(returns, weights)
    benchmark_symbol = st.session_state.get("benchmark_symbol", "None")
    benchmark_series = None
    if benchmark_symbol != "None":
        benchmark_series = returns[benchmark_symbol].reindex(portfolio_series.index).dropna()

    extended_metrics = cached_extended_metrics(
        portfolio_series.to_frame("portfolio"),
        returns[portfolio_state.symbols].dropna(how="all") if portfolio_state.symbols else pd.DataFrame(),
        tuple(sorted(weights.items())),
        benchmark_series.to_frame("benchmark") if benchmark_series is not None else None,
        freq,
    )

    current_year = datetime.now(timezone.utc).year
    bars_per_year = metrics.ANNUALIZATION.get(freq, 252)
    periods = {
        "1Y": bars_per_year,
        "3Y": bars_per_year * 3,
        "5Y": bars_per_year * 5,
        "YTD": len(portfolio_series.loc[portfolio_series.index >= pd.Timestamp(current_year, 1, 1)]),
    }
    period_perf = metrics.period_returns(portfolio_series, periods)

    outlook_metrics = {
        "CAGR": extended_metrics.get("CAGR"),
        "Ann. Vol": extended_metrics.get("Ann. Vol"),
        "Sharpe": extended_metrics.get("Sharpe"),
        "Sortino": extended_metrics.get("Sortino"),
        "Max Drawdown": extended_metrics.get("Max Drawdown"),
        "Calmar Ratio": extended_metrics.get("Calmar Ratio"),
        "MAR Ratio": extended_metrics.get("MAR Ratio"),
        "Omega Ratio": extended_metrics.get("Omega Ratio"),
        "Treynor Ratio": extended_metrics.get("Treynor Ratio"),
        "Information Ratio": extended_metrics.get("Information Ratio"),
        "Upside Capture": extended_metrics.get("Upside Capture"),
        "Downside Capture": extended_metrics.get("Downside Capture"),
        "Tracking Error": extended_metrics.get("Tracking Error"),
    }
    for label, value in period_perf.items():
        outlook_metrics[f"Return {label}"] = value

    formatted = {name: (_format_value(name, val), "Period or risk metric.") for name, val in outlook_metrics.items()}
    benchmark_sensitive = {"Treynor Ratio", "Information Ratio", "Upside Capture", "Downside Capture", "Tracking Error"}
    kpi_start = time.perf_counter()
    ui.render_metric_grid(formatted, columns=4, muted_keys=benchmark_sensitive if benchmark_series is None else [])
    _log_block(
        "outlook_kpis",
        kpi_start,
        count=len(formatted),
        portfolio_size=len(portfolio_series),
        benchmark=benchmark_series is not None,
        window=lookback,
    )

    figures = metrics.assemble_outlook_figures(
        portfolio_series,
        asset_returns=returns[portfolio_state.symbols].dropna(how="all") if portfolio_state.symbols else pd.DataFrame(),
        weights=weights,
        benchmark=benchmark_series,
        freq=freq,
    )
    _render_chart_grid(
        figures,
        prefix="outlook_chart",
        portfolio_size=len(portfolio_series),
        benchmark=benchmark_series is not None,
        window=lookback,
    )

    with st.expander("Summary JSON", expanded=False):
        st.json(st.session_state.get("summary_payload", {}))


def render_risk_tab() -> None:
    st.subheader("Risk Metrics & Diagnostics")
    returns = st.session_state.get("returns", pd.DataFrame())
    if returns.empty:
        st.info("Ingest data first.")
        return
    freq = st.session_state.get("frequency", "Daily")
    portfolio_state: portfolio.PortfolioState = st.session_state.get("portfolio_state", portfolio.PortfolioState())
    weights = portfolio_state.normalized_weights()
    portfolio_series = metrics.portfolio_returns(returns, weights)
    benchmark_symbol = st.session_state.get("benchmark_symbol", "None")
    benchmark_series = None
    if benchmark_symbol != "None":
        benchmark_series = returns[benchmark_symbol].reindex(portfolio_series.index).dropna()

    extended_metrics = cached_extended_metrics(
        portfolio_series.to_frame("portfolio"),
        returns[portfolio_state.symbols].dropna(how="all") if portfolio_state.symbols else pd.DataFrame(),
        tuple(sorted(weights.items())),
        benchmark_series.to_frame("benchmark") if benchmark_series is not None else None,
        freq,
    )

    risk_specific = {
        "Sharpe": extended_metrics.get("Sharpe"),
        "Sortino": extended_metrics.get("Sortino"),
        "Max Drawdown": extended_metrics.get("Max Drawdown"),
        "ES 99%": extended_metrics.get("ES 99%"),
        "Cornish-Fisher VaR": extended_metrics.get("Cornish-Fisher VaR"),
        "CDaR 95%": extended_metrics.get("CDaR 95%"),
        "Tail Ratio": extended_metrics.get("Tail Ratio"),
        "Pain Ratio": extended_metrics.get("Pain Ratio"),
        "Ulcer Index": extended_metrics.get("Ulcer Index"),
        "VaR Breaches": extended_metrics.get("VaR Breaches"),
        "Diversification Ratio": extended_metrics.get("Diversification Ratio"),
        "Kelly Fraction": extended_metrics.get("Kelly Fraction"),
        "Rolling VaR": extended_metrics.get("Rolling VaR"),
        "Rolling Beta": extended_metrics.get("Rolling Beta"),
    }
    formatted = {name: (_format_value(name, val), "Risk diagnostic.") for name, val in risk_specific.items()}
    benchmark_sensitive = {"Rolling Beta"}
    kpi_start = time.perf_counter()
    ui.render_metric_grid(formatted, columns=4, muted_keys=benchmark_sensitive if benchmark_series is None else [])
    _log_block(
        "risk_kpis",
        kpi_start,
        count=len(formatted),
        portfolio_size=len(portfolio_series),
        benchmark=benchmark_series is not None,
        window=st.session_state.get("lookback", 3),
    )

    figures = metrics.assemble_risk_figures(
        portfolio_series,
        asset_returns=returns[portfolio_state.symbols].dropna(how="all") if portfolio_state.symbols else pd.DataFrame(),
        weights=weights,
        benchmark=benchmark_series,
    )
    _render_chart_grid(
        figures,
        prefix="risk_chart",
        portfolio_size=len(portfolio_series),
        benchmark=benchmark_series is not None,
        window=st.session_state.get("lookback", 3),
    )


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

