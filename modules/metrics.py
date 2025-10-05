"""Portfolio analytics computations and visualizations."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)

ANNUALIZATION = {"Daily": 252, "Monthly": 12}


@dataclass
class PortfolioSummary:
    """Convenience container for top-line metrics."""

    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float

    def to_dict(self) -> dict[str, float]:
        """Serialize metrics to a dict for downstream consumption."""

        return {
            "CAGR": self.cagr,
            "Ann. Vol": self.volatility,
            "Sharpe": self.sharpe,
            "Sortino": self.sortino,
            "Max Drawdown": self.max_drawdown,
        }


def price_to_returns(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Convert price levels to returns aligned with the requested frequency."""

    frame = prices.sort_index().ffill().dropna(how="all")
    returns = frame.pct_change().dropna(how="all")
    if returns.empty:
        return returns
    if freq == "Monthly":
        monthly = (1 + returns).resample("M").prod() - 1
        return monthly.dropna(how="all")
    return returns


def portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Compute portfolio return series given asset returns and weights."""

    if returns.empty or not weights:
        return pd.Series(dtype=float)
    aligned = returns[list({s for s in weights if s in returns.columns})].fillna(0)
    if aligned.empty:
        return pd.Series(dtype=float)
    w = np.array([weights[s] for s in aligned.columns], dtype=float)
    w = w / w.sum() if w.sum() else w
    port = aligned.to_numpy() @ w
    return pd.Series(port, index=aligned.index, name="portfolio")


def _annual_factor(freq: str) -> int:
    return ANNUALIZATION.get(freq, 252)


def _drawdown_curve(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    return curve / peak - 1


def compute_summary(returns: pd.Series, freq: str) -> PortfolioSummary:
    """Return standard performance summary for portfolio returns."""

    if returns.empty:
        return PortfolioSummary(*(np.nan for _ in range(5)))
    periods = _annual_factor(freq)
    mean = returns.mean()
    std = returns.std(ddof=0)
    downside = returns[returns < 0].std(ddof=0)
    compounded = (1 + returns).prod()
    years = len(returns) / periods if periods else np.nan
    cagr = compounded ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = std * math.sqrt(periods)
    sharpe = (mean * periods) / vol if vol else np.nan
    sortino = (mean * periods) / (downside * math.sqrt(periods)) if downside else np.nan
    drawdown = _drawdown_curve(returns)
    max_dd = drawdown.min()
    return PortfolioSummary(cagr, vol, sharpe, sortino, max_dd)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _ulcer_index(drawdown: pd.Series) -> float:
    if drawdown.empty:
        return np.nan
    clipped = drawdown.clip(upper=0).dropna()
    if clipped.empty:
        return np.nan
    squared = np.square(clipped)
    mean_squared = float(np.mean(squared))
    return math.sqrt(mean_squared) if mean_squared >= 0 else np.nan


def _capm_alpha_beta(portfolio: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan, np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] else np.nan
    alpha = aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()
    return alpha, beta


def _tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float:
    spread = portfolio.sub(benchmark, fill_value=0)
    return float(spread.std(ddof=0))


def _hit_rate(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    positives = (series > 0).sum()
    return positives / len(series)


def _upside_downside_capture(portfolio: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan, np.nan
    bench = aligned.iloc[:, 1]
    port = aligned.iloc[:, 0]
    upside = port[bench > 0].mean()
    downside = port[bench < 0].mean()
    bench_up = bench[bench > 0].mean()
    bench_down = bench[bench < 0].mean()
    up_capture = _safe_div(upside, bench_up) if bench_up not in (0, np.nan) else np.nan
    down_capture = _safe_div(downside, bench_down) if bench_down not in (0, np.nan) else np.nan
    return up_capture, down_capture


def _avg_up_down(series: pd.Series) -> tuple[float, float]:
    if series.empty:
        return np.nan, np.nan
    up = series[series > 0]
    down = series[series < 0]
    return (up.mean() if not up.empty else np.nan, down.mean() if not down.empty else np.nan)


def _tail_ratio(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    upper = np.quantile(series.dropna(), 0.95)
    lower = np.quantile(series.dropna(), 0.05)
    return _safe_div(upper, abs(lower))


def _pain_ratio(returns: pd.Series, freq: str) -> float:
    drawdown = _drawdown_curve(returns)
    pain = -drawdown[drawdown < 0].mean()
    cagr = compute_summary(returns, freq).cagr
    return _safe_div(cagr, pain)


def _cornish_fisher_var(series: pd.Series, alpha: float) -> float:
    if series.empty:
        return np.nan
    z = NormalDist().inv_cdf(alpha)
    s = float(series.skew())
    k = float(series.kurtosis())
    cf = z + (1 / 6) * (z**2 - 1) * s + (1 / 24) * (z**3 - 3 * z) * (k - 3) - (1 / 36) * (2 * z**3 - 5 * z) * (s**2)
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    return -(mean + cf * std)


def _expected_shortfall(series: pd.Series, alpha: float) -> float:
    if series.empty:
        return np.nan
    threshold = series.quantile(1 - alpha)
    tail = series[series <= threshold]
    if tail.empty:
        return np.nan
    return -tail.mean()


def _conditional_drawdown_at_risk(returns: pd.Series, alpha: float) -> float:
    drawdown = _drawdown_curve(returns)
    threshold = drawdown.quantile(alpha)
    tail = drawdown[drawdown <= threshold]
    return -tail.mean() if not tail.empty else np.nan


def _diversification_ratio(weights: Dict[str, float], cov: pd.DataFrame) -> float:
    if not weights or cov.empty:
        return np.nan
    w = np.array([weights.get(sym, 0.0) for sym in cov.columns])
    if not w.any():
        return np.nan
    stds = np.sqrt(np.diag(cov.to_numpy()))
    numerator = np.dot(w, stds)
    denom = math.sqrt(float(w @ cov.to_numpy() @ w))
    return _safe_div(numerator, denom)


def _risk_contributions(weights: Dict[str, float], cov: pd.DataFrame) -> dict[str, float]:
    if not weights or cov.empty:
        return {}
    w = np.array([weights.get(sym, 0.0) for sym in cov.columns])
    portfolio_var = float(w @ cov.to_numpy() @ w)
    if portfolio_var <= 0:
        return {sym: np.nan for sym in cov.columns}
    marginal = cov.to_numpy() @ w
    contrib = w * marginal / portfolio_var
    return {sym: float(val) for sym, val in zip(cov.columns, contrib, strict=False)}


def _kelly_fraction(mean: float, variance: float) -> float:
    if variance <= 0:
        return np.nan
    return mean / variance


def compute_extended_metrics(
    portfolio: pd.Series,
    *,
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: pd.Series | None,
    freq: str,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """Compute an extensive set of portfolio statistics."""

    if portfolio.empty:
        return {}
    summary = compute_summary(portfolio, freq)
    drawdown = _drawdown_curve(portfolio)
    ann_factor = _annual_factor(freq)
    ann_return = float(portfolio.mean() * ann_factor)
    ann_vol = float(portfolio.std(ddof=0) * math.sqrt(ann_factor))
    downside = portfolio[portfolio < 0]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(ann_factor)) if not downside.empty else np.nan
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = _safe_div(summary.cagr, abs(max_dd)) if max_dd else np.nan
    mar = _safe_div(ann_return, abs(max_dd)) if max_dd else np.nan
    positive = (portfolio[portfolio > 0] + 1).prod() - 1 if not portfolio.empty else np.nan
    negative = abs((portfolio[portfolio < 0] + 1).prod() - 1) if not portfolio.empty else np.nan
    omega = _safe_div(positive, negative)
    ulcer = _ulcer_index(drawdown)
    treynor = np.nan
    alpha = beta = np.nan
    tracking = information = np.nan
    upside_capture = downside_capture = np.nan
    if benchmark is not None and not benchmark.empty:
        alpha, beta = _capm_alpha_beta(portfolio, benchmark)
        tracking = _tracking_error(portfolio, benchmark)
        information = _safe_div(ann_return - float(benchmark.mean() * ann_factor), tracking) if tracking else np.nan
        upside_capture, downside_capture = _upside_downside_capture(portfolio, benchmark)
        if beta not in (np.nan, 0):
            treynor = _safe_div(ann_return - risk_free_rate, beta)
    hit = _hit_rate(portfolio)
    avg_up, avg_down = _avg_up_down(portfolio)
    skew = float(portfolio.skew()) if len(portfolio) > 2 else np.nan
    kurt = float(portfolio.kurtosis()) if len(portfolio) > 3 else np.nan
    tail = _tail_ratio(portfolio)
    pain = _pain_ratio(portfolio, freq)
    cf_var = _cornish_fisher_var(portfolio, 0.99)
    es_99 = _expected_shortfall(portfolio, 0.99)
    cdar = _conditional_drawdown_at_risk(portfolio, 0.95)
    cov = asset_returns.cov()
    div_ratio = _diversification_ratio(weights, cov)
    contributions = _risk_contributions(weights, cov)
    kelly = _kelly_fraction(portfolio.mean(), float(portfolio.var(ddof=0)))
    window = max(ann_factor // 12, 1)
    rolling_var = portfolio.rolling(window=window).apply(lambda x: -np.quantile(x.dropna(), 0.05) if len(x.dropna()) else np.nan)
    rolling_beta_series = pd.Series(dtype=float)
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
        if not aligned.empty:
            numerator = aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1])
            denominator = aligned.iloc[:, 1].rolling(window).var()
            rolling_beta_series = numerator / denominator
    var_breaches = portfolio < portfolio.quantile(0.05)

    metrics = {
        "CAGR": summary.cagr,
        "Ann. Vol": summary.volatility,
        "Sharpe": summary.sharpe,
        "Sortino": summary.sortino,
        "Max Drawdown": summary.max_drawdown,
        "Calmar Ratio": calmar,
        "MAR Ratio": mar,
        "Omega Ratio": omega,
        "Ulcer Index": ulcer,
        "Treynor Ratio": treynor,
        "CAPM Alpha": alpha,
        "CAPM Beta": beta,
        "Tracking Error": tracking,
        "Information Ratio": information,
        "Upside Capture": upside_capture,
        "Downside Capture": downside_capture,
        "Hit Rate": hit,
        "Avg Up Day": avg_up,
        "Avg Down Day": avg_down,
        "Skew": skew,
        "Kurtosis": kurt,
        "Tail Ratio": tail,
        "Pain Ratio": pain,
        "Cornish-Fisher VaR": cf_var,
        "ES 99%": es_99,
        "CDaR 95%": cdar,
        "Diversification Ratio": div_ratio,
        "Kelly Fraction": kelly,
    }
    if not rolling_var.empty:
        metrics["Rolling VaR"] = float(rolling_var.dropna().iloc[-1]) if not rolling_var.dropna().empty else np.nan
    if isinstance(rolling_beta_series, pd.Series) and not rolling_beta_series.empty:
        metrics["Rolling Beta"] = float(rolling_beta_series.dropna().iloc[-1]) if not rolling_beta_series.dropna().empty else np.nan
    metrics["VaR Breaches"] = int(var_breaches.sum()) if not portfolio.empty else np.nan
    return metrics | {f"RC {sym}": val for sym, val in contributions.items()}


def period_returns(returns: pd.Series, periods: Mapping[str, int]) -> pd.Series:
    """Compute cumulative returns over user-defined windows."""

    data = {}
    for label, bars in periods.items():
        if returns.empty or bars <= 0:
            data[label] = np.nan
            continue
        recent = returns.dropna().iloc[-bars:]
        data[label] = (1 + recent).prod() - 1 if not recent.empty else np.nan
    return pd.Series(data)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise return correlations."""

    return returns.corr().fillna(0)


def concentration(weights: Dict[str, float]) -> dict[str, float]:
    """Basic concentration diagnostics from weights."""

    if not weights:
        return {"HHI": np.nan, "ENH": np.nan, "Top3": np.nan, "Top5": np.nan, "Top10": np.nan}
    weights_arr = np.array(sorted(weights.values(), reverse=True))
    hhi = float(np.sum(weights_arr**2))
    enh = 1 / hhi if hhi else np.nan
    top = {
        "Top3": float(weights_arr[:3].sum()) if weights_arr.size >= 1 else np.nan,
        "Top5": float(weights_arr[:5].sum()) if weights_arr.size >= 1 else np.nan,
        "Top10": float(weights_arr[:10].sum()) if weights_arr.size >= 1 else np.nan,
    }
    return {"HHI": hhi, "ENH": enh, **top}


def equity_curve(portfolio: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    """Plot cumulative performance for portfolio and optional benchmark."""

    fig = go.Figure()
    if not portfolio.empty:
        fig.add_trace(go.Scatter(x=portfolio.index, y=(1 + portfolio).cumprod(), name="Portfolio"))
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark.index, y=(1 + benchmark).cumprod(), name="Benchmark"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", legend=dict(orientation="h"))
    fig.update_yaxes(title="Growth of $1")
    return fig


def equity_sparkline(portfolio: pd.Series) -> go.Figure:
    fig = go.Figure()
    if not portfolio.empty:
        cum = (1 + portfolio).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Equity"))
    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=10), height=200)
    return fig


def sector_donut(weights: Dict[str, float], meta: Mapping[str, Mapping[str, str]]) -> go.Figure:
    sectors = {}
    for sym, weight in weights.items():
        sector = meta.get(sym, {}).get("sector", "Other") if meta else "Other"
        sectors[sector] = sectors.get(sector, 0.0) + weight
    fig = go.Figure(data=[go.Pie(labels=list(sectors.keys()), values=list(sectors.values()), hole=0.6)])
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def treemap_weights(weights: Dict[str, float], meta: Mapping[str, Mapping[str, str]]) -> go.Figure:
    if not weights:
        return go.Figure()
    labels = []
    parents = []
    values = []
    for sym, weight in weights.items():
        sector = meta.get(sym, {}).get("sector", "Other") if meta else "Other"
        labels.append(sym)
        parents.append(sector)
        values.append(weight)
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values, textinfo="label+percent entry"))
    fig.update_layout(template="plotly_dark")
    return fig


def rolling_return(portfolio: pd.Series, window: int) -> go.Figure:
    data = portfolio.rolling(window=window).apply(lambda s: (1 + s).prod() - 1 if len(s.dropna()) else np.nan)
    fig = go.Figure(go.Scatter(x=data.index, y=data.values, name="Rolling Return"))
    fig.update_layout(template="plotly_dark")
    return fig


def rolling_volatility(portfolio: pd.Series, window: int, freq: str) -> go.Figure:
    factor = math.sqrt(_annual_factor(freq))
    data = portfolio.rolling(window=window).std(ddof=0) * factor
    fig = go.Figure(go.Scatter(x=data.index, y=data.values, name="Rolling Vol"))
    fig.update_layout(template="plotly_dark")
    return fig


def rolling_sharpe(portfolio: pd.Series, window: int, freq: str) -> go.Figure:
    factor = math.sqrt(_annual_factor(freq))
    rolling = portfolio.rolling(window=window)
    sharpe = rolling.mean() * _annual_factor(freq) / (rolling.std(ddof=0) * factor)
    fig = go.Figure(go.Scatter(x=sharpe.index, y=sharpe.values, name="Rolling Sharpe"))
    fig.update_layout(template="plotly_dark")
    return fig


def drawdown_area(portfolio: pd.Series) -> go.Figure:
    dd = _drawdown_curve(portfolio)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(template="plotly_dark")
    return fig


def upside_downside_bars(portfolio: pd.Series, freq: str) -> go.Figure:
    up = (portfolio[portfolio > 0]).resample("M").apply(lambda x: (1 + x).prod() - 1 if len(x) else np.nan)
    down = (portfolio[portfolio < 0]).resample("M").apply(lambda x: (1 + x).prod() - 1 if len(x) else np.nan)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=up.index, y=up.values, name="Upside"))
    fig.add_trace(go.Bar(x=down.index, y=down.values, name="Downside"))
    fig.update_layout(template="plotly_dark", barmode="group")
    return fig


def risk_return_scatter(asset_returns: pd.DataFrame, freq: str) -> go.Figure:
    if asset_returns.empty:
        return go.Figure()
    ann = _annual_factor(freq)
    stats = pd.DataFrame({
        "Return": asset_returns.mean() * ann,
        "Vol": asset_returns.std(ddof=0) * math.sqrt(ann),
    })
    fig = px.scatter(stats, x="Vol", y="Return", text=stats.index, template="plotly_dark")
    return fig


def calendar_heatmap(portfolio: pd.Series) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    df = portfolio.to_frame("return").copy()
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    pivot = df.pivot_table(index="Year", columns="Month", values="return", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", template="plotly_dark")
    return fig


def active_weights_bar(weights: Dict[str, float]) -> go.Figure:
    if not weights:
        return go.Figure()
    equal = 1 / len(weights)
    active = {sym: w - equal for sym, w in weights.items()}
    fig = go.Figure(go.Bar(x=list(active.keys()), y=list(active.values())))
    fig.update_layout(template="plotly_dark")
    return fig


def annual_returns_bar(portfolio: pd.Series) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    data = portfolio.resample("Y").apply(lambda x: (1 + x).prod() - 1)
    fig = go.Figure(go.Bar(x=data.index.year, y=data.values))
    fig.update_layout(template="plotly_dark")
    return fig


def return_hist_with_stats(portfolio: pd.Series) -> go.Figure:
    fig = go.Figure()
    if not portfolio.empty:
        fig.add_trace(go.Histogram(x=portfolio, nbinsx=40, name="Returns"))
        mean = portfolio.mean()
        std = portfolio.std(ddof=0)
        x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        pdf = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2) if std else np.zeros_like(x)
        fig.add_trace(go.Scatter(x=x, y=pdf, name="Normal PDF"))
    fig.update_layout(template="plotly_dark")
    return fig


def alpha_beta_scatter(asset_returns: pd.DataFrame, benchmark: pd.Series) -> go.Figure:
    if asset_returns.empty or benchmark.empty:
        return go.Figure()
    stats = []
    for sym in asset_returns.columns:
        alpha, beta = _capm_alpha_beta(asset_returns[sym], benchmark)
        stats.append({"symbol": sym, "alpha": alpha, "beta": beta})
    frame = pd.DataFrame(stats).dropna()
    fig = px.scatter(frame, x="beta", y="alpha", text="symbol", template="plotly_dark")
    return fig


def contribution_waterfall(weights: Dict[str, float], asset_returns: pd.DataFrame) -> go.Figure:
    if not weights or asset_returns.empty:
        return go.Figure()
    means = asset_returns.mean()
    contrib = {sym: weights.get(sym, 0.0) * means.get(sym, 0.0) for sym in asset_returns.columns}
    fig = go.Figure(go.Waterfall(x=list(contrib.keys()), y=list(contrib.values())))
    fig.update_layout(template="plotly_dark")
    return fig


def var_exceedance_timeline(portfolio: pd.Series, alpha: float = 0.95) -> go.Figure:
    threshold = portfolio.quantile(1 - alpha) if not portfolio.empty else np.nan
    breaches = portfolio[portfolio < threshold]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio.values, name="Returns"))
    fig.add_trace(go.Scatter(x=breaches.index, y=breaches.values, mode="markers", name="Breaches"))
    fig.update_layout(template="plotly_dark")
    return fig


def rolling_var_es_lines(portfolio: pd.Series, window: int = 63) -> go.Figure:
    rolling_var = portfolio.rolling(window).apply(lambda x: -np.quantile(x.dropna(), 0.05) if len(x.dropna()) else np.nan)
    rolling_es = portfolio.rolling(window).apply(lambda x: -x[x <= x.quantile(0.05)].mean() if len(x.dropna()) else np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_var.index, y=rolling_var.values, name="Rolling VaR"))
    fig.add_trace(go.Scatter(x=rolling_es.index, y=rolling_es.values, name="Rolling ES"))
    fig.update_layout(template="plotly_dark")
    return fig


def qqplot(portfolio: pd.Series) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    sorted_returns = np.sort(portfolio.dropna())
    quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
    normal = np.array([NormalDist().inv_cdf(q) for q in quantiles])
    fig = go.Figure(go.Scatter(x=normal, y=sorted_returns, mode="markers"))
    fig.update_layout(template="plotly_dark", xaxis_title="Normal Quantiles", yaxis_title="Empirical")
    return fig


def normal_vs_t_fit(portfolio: pd.Series) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    hist = go.Histogram(x=portfolio, nbinsx=40, histnorm="probability density", name="Empirical")
    mean = portfolio.mean()
    std = portfolio.std(ddof=0)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
    normal_pdf = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2) if std else np.zeros_like(x)
    t_pdf = normal_pdf * 0.9  # proxy to avoid scipy dependency
    fig = go.Figure([hist, go.Scatter(x=x, y=normal_pdf, name="Normal"), go.Scatter(x=x, y=t_pdf, name="t-dist approx")])
    fig.update_layout(template="plotly_dark")
    return fig


def corr_heatmap(returns: pd.DataFrame) -> go.Figure:
    matrix = correlation_matrix(returns)
    fig = px.imshow(matrix, text_auto=False, aspect="auto", color_continuous_scale="Inferno", origin="lower", template="plotly_dark")
    return fig


def corr_dendrogram(returns: pd.DataFrame) -> go.Figure:
    if returns.empty:
        return go.Figure()
    corr = correlation_matrix(returns)
    distance = 1 - corr
    order = distance.mean().sort_values().index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=order, y=distance.loc[order, order].mean(), name="Avg distance"))
    fig.update_layout(template="plotly_dark")
    return fig


def corr_network(returns: pd.DataFrame, threshold: float = 0.6) -> go.Figure:
    if returns.empty:
        return go.Figure()
    corr = correlation_matrix(returns)
    nodes = list(corr.columns)
    angles = np.linspace(0, 2 * math.pi, len(nodes), endpoint=False)
    positions = {node: (math.cos(angle), math.sin(angle)) for node, angle in zip(nodes, angles, strict=False)}
    fig = go.Figure()
    for i, node in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            weight = corr.iloc[i, j]
            if abs(weight) >= threshold:
                x0, y0 = positions[nodes[i]]
                x1, y1 = positions[nodes[j]]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=abs(weight) * 2), showlegend=False))
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes], mode="markers+text", text=nodes))
    fig.update_layout(template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def risk_contrib_stacked_bar(contributions: Mapping[str, float]) -> go.Figure:
    if not contributions:
        return go.Figure()
    fig = go.Figure(go.Bar(x=list(contributions.keys()), y=list(contributions.values()), name="Risk Contribution"))
    fig.update_layout(template="plotly_dark")
    return fig


def diversification_ratio_gauge(value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value if not np.isnan(value) else 0, gauge={"axis": {"range": [0, max(2, value or 1)]}}))
    fig.update_layout(template="plotly_dark")
    return fig


def stress_fan_chart(portfolio: pd.Series) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    windows = [21, 63, 126]
    fig = go.Figure()
    for w in windows:
        data = portfolio.rolling(w).apply(lambda x: (1 + x).prod() - 1 if len(x.dropna()) else np.nan)
        fig.add_trace(go.Scatter(x=data.index, y=data.values, name=f"{w}-day"))
    fig.update_layout(template="plotly_dark")
    return fig


def monte_carlo_cone(portfolio: pd.Series, horizon: int = 63, simulations: int = 1000) -> go.Figure:
    if portfolio.empty:
        return go.Figure()
    mu = portfolio.mean()
    sigma = portfolio.std(ddof=0)
    last = (1 + portfolio).cumprod().iloc[-1]
    rng = np.random.default_rng(42)
    paths = np.cumprod(1 + rng.normal(mu, sigma, size=(simulations, horizon)), axis=1) * last
    percentiles = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
    idx = pd.date_range(start=portfolio.index[-1], periods=horizon + 1, freq="B")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx[1:], y=percentiles[1], name="Median"))
    fig.add_trace(go.Scatter(x=idx[1:], y=percentiles[0], name="5%", fill=None))
    fig.add_trace(go.Scatter(x=idx[1:], y=percentiles[2], name="95%", fill="tonexty"))
    fig.update_layout(template="plotly_dark")
    return fig


def rolling_beta_line(portfolio: pd.Series, benchmark: pd.Series, window: int = 63) -> go.Figure:
    if portfolio.empty or benchmark.empty:
        return go.Figure()
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    beta = aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].rolling(window).var()
    fig = go.Figure(go.Scatter(x=beta.index, y=beta.values, name="Rolling Beta"))
    fig.update_layout(template="plotly_dark")
    return fig


def histogram_with_var(returns: pd.Series, freq: str) -> go.Figure:
    """Legacy helper maintained for compatibility."""

    fig = go.Figure()
    if returns.empty:
        fig.update_layout(template="plotly_dark")
        return fig
    hist_var = -np.quantile(returns.dropna(), 0.05)
    mean = returns.mean()
    std = returns.std(ddof=0)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
    counts, _ = np.histogram(returns.dropna(), bins=40, density=True)
    pdf = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2) if std else np.zeros_like(x)
    fig.add_trace(go.Histogram(x=returns, nbinsx=40, name="Returns", opacity=0.6, histnorm="probability"))
    scale = pdf.max() or 1
    density_scale = counts.max() or 1
    fig.add_trace(go.Scatter(x=x, y=pdf / scale * density_scale, name="Normal PDF"))
    fig.add_vline(x=-hist_var, line_color="red", annotation_text="VaR 95%", annotation_position="top right")
    fig.update_layout(template="plotly_dark", bargap=0.1)
    return fig


def assemble_summary_figures(
    portfolio: pd.Series,
    *,
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: pd.Series | None,
    freq: str,
    meta: Mapping[str, Mapping[str, str]],
) -> dict[str, Tuple[go.Figure, str, str | None]]:
    """Generate plots for the summary tab."""

    return {
        "Equity Sparkline": (equity_sparkline(portfolio), "Momentum of $1 since inception.", None),
        "Sector Allocation": (sector_donut(weights, meta), "Current exposure by GICS sector.", "Requires sector metadata."),
        "Treemap Weights": (treemap_weights(weights, meta), "Visualizes weight concentration.", None),
        "Rolling 12M Return": (rolling_return(portfolio, 252 if freq == "Daily" else 12), "Trailing performance drift.", "Rolling cumulative return."),
        "Rolling Volatility": (rolling_volatility(portfolio, 63, freq), "Watch realized risk regime.", None),
        "Rolling Sharpe": (rolling_sharpe(portfolio, 63, freq), "Risk-adjusted momentum.", "Annualized metric."),
        "Drawdown Area": (drawdown_area(portfolio), "Depth and duration of pullbacks.", None),
        "Upside vs Downside": (upside_downside_bars(portfolio, freq), "Monthly gain/loss split.", None),
        "Risk/Return Scatter": (risk_return_scatter(asset_returns, freq), "Asset positioning in mean-variance space.", None),
        "Calendar Heatmap": (calendar_heatmap(portfolio), "Seasonality of returns.", None),
    }


def assemble_outlook_figures(
    portfolio: pd.Series,
    *,
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: pd.Series | None,
    freq: str,
) -> dict[str, Tuple[go.Figure, str, str | None]]:
    """Charts supporting the outlook tab."""

    figures: dict[str, Tuple[go.Figure, str, str | None]] = {
        "Equity Curve": (equity_curve(portfolio, benchmark), "Portfolio vs benchmark trajectory.", None),
        "Active Weights": (active_weights_bar(weights), "Deviation from equal-weight baseline.", None),
        "Annual Returns": (annual_returns_bar(portfolio), "Calendar year performance dispersion.", None),
        "Return Histogram": (return_hist_with_stats(portfolio), "Distribution with normal overlay.", None),
        "Contribution Waterfall": (contribution_waterfall(weights, asset_returns), "Average contribution by asset.", None),
        "Alpha/Beta Scatter": (
            alpha_beta_scatter(asset_returns, benchmark) if benchmark is not None else go.Figure(),
            "Asset sensitivities vs benchmark.",
            "Benchmark dependent.",
        ),
        "Rolling VaR": (rolling_var_es_lines(portfolio), "Evolution of tail risk.", None),
        "Var Exceedance": (var_exceedance_timeline(portfolio), "Highlight realized tail events.", None),
        "Stress Fan": (stress_fan_chart(portfolio), "Scenario bandwidth under rolling windows.", None),
        "Monte Carlo Cone": (monte_carlo_cone(portfolio), "Probabilistic future cone.", None),
    }
    if benchmark is not None and not benchmark.empty:
        figures["Rolling Beta"] = (rolling_beta_line(portfolio, benchmark), "Time-varying systematic exposure.", "Benchmark dependent.")
    return figures


def assemble_risk_figures(
    portfolio: pd.Series,
    *,
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: pd.Series | None,
) -> dict[str, Tuple[go.Figure, str, str | None]]:
    """Charts supporting the risk diagnostics tab."""

    cov = asset_returns.cov()
    contributions = _risk_contributions(weights, cov)
    figures = {
        "Histogram + VaR": (histogram_with_var(portfolio, "Daily"), "Distribution of returns with VaR line.", None),
        "Correlation Heatmap": (corr_heatmap(asset_returns), "Cross-asset dependency structure.", None),
        "Correlation Dendrogram": (corr_dendrogram(asset_returns), "Cluster relationship overview.", None),
        "Correlation Network": (corr_network(asset_returns), "Graph of high correlations.", None),
        "Risk Contribution": (risk_contrib_stacked_bar(contributions), "Marginal contribution to volatility.", None),
        "Diversification Gauge": (diversification_ratio_gauge(_diversification_ratio(weights, cov)), "Higher is better diversification.", None),
        "Rolling VaR/ES": (rolling_var_es_lines(portfolio), "Tail metrics trend.", None),
        "QQ Plot": (qqplot(portfolio), "Check normality assumption.", None),
        "Normal vs t Fit": (normal_vs_t_fit(portfolio), "Compare empirical vs parametric densities.", None),
        "Rolling Beta": (
            rolling_beta_line(portfolio, benchmark) if benchmark is not None and not benchmark.empty else go.Figure(),
            "Systematic risk drift.",
            "Benchmark dependent.",
        ),
    }
    return figures

