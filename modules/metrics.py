"""Portfolio analytics computations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ANNUALIZATION = {"Daily": 252, "Monthly": 12}


@dataclass
class PortfolioSummary:
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float

    def to_dict(self) -> dict[str, float]:
        return {
            "CAGR": self.cagr,
            "Ann. Vol": self.volatility,
            "Sharpe": self.sharpe,
            "Sortino": self.sortino,
            "Max Drawdown": self.max_drawdown,
        }


def price_to_returns(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    frame = prices.sort_index().ffill().dropna(how="all")
    returns = frame.pct_change().dropna(how="all")
    if returns.empty:
        return returns
    if freq == "Monthly":
        monthly = (1 + returns).resample("M").prod() - 1
        return monthly.dropna(how="all")
    return returns


def portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    if returns.empty or not weights:
        return pd.Series(dtype=float)
    aligned = returns[list({s for s in weights if s in returns.columns})].fillna(0)
    if aligned.empty:
        return pd.Series(dtype=float)
    w = np.array([weights[s] for s in aligned.columns], dtype=float)
    w = w / w.sum() if w.sum() else w
    port = aligned.to_numpy() @ w
    return pd.Series(port, index=aligned.index, name="portfolio")


def compute_summary(returns: pd.Series, freq: str) -> PortfolioSummary:
    if returns.empty:
        return PortfolioSummary(*(np.nan for _ in range(5)))
    periods = ANNUALIZATION.get(freq, 252)
    mean = returns.mean()
    std = returns.std(ddof=0)
    downside = returns[returns < 0].std(ddof=0)
    compounded = (1 + returns).prod()
    years = len(returns) / periods if periods else np.nan
    cagr = compounded ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = std * math.sqrt(periods)
    sharpe = (mean * periods) / vol if vol else np.nan
    sortino = (mean * periods) / (downside * math.sqrt(periods)) if downside else np.nan
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak) - 1
    max_dd = drawdown.min()
    return PortfolioSummary(cagr, vol, sharpe, sortino, max_dd)


def equity_curve(portfolio: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    if not portfolio.empty:
        fig.add_trace(go.Scatter(x=portfolio.index, y=(1 + portfolio).cumprod(), name="Portfolio"))
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark.index, y=(1 + benchmark).cumprod(), name="Benchmark"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", legend=dict(orientation="h"))
    fig.update_yaxes(title="Growth of $1")
    return fig


def period_returns(returns: pd.Series, periods: Dict[str, int]) -> pd.Series:
    data = {}
    for label, bars in periods.items():
        if returns.empty:
            data[label] = np.nan
            continue
        recent = returns.dropna().iloc[-bars:]
        if recent.empty:
            data[label] = np.nan
        else:
            data[label] = (1 + recent).prod() - 1
    return pd.Series(data)


def risk_metrics(returns: pd.Series, freq: str) -> dict[str, float]:
    if returns.empty:
        return {key: np.nan for key in ["Sharpe", "Sortino", "Max Drawdown", "Hist VaR", "Param VaR", "ES"]}
    summary = compute_summary(returns, freq)
    hist_var = -np.quantile(returns.dropna(), 0.05)
    mean = returns.mean()
    std = returns.std(ddof=0)
    z = NormalDist().inv_cdf(0.05)
    parametric = -(mean + z * std)
    tail = returns[returns <= np.quantile(returns.dropna(), 0.05)]
    expected_shortfall = -tail.mean() if not tail.empty else np.nan
    return {
        "Sharpe": summary.sharpe,
        "Sortino": summary.sortino,
        "Max Drawdown": summary.max_drawdown,
        "Hist VaR": hist_var,
        "Param VaR": parametric,
        "ES": expected_shortfall,
    }


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().fillna(0)


def concentration(weights: Dict[str, float]) -> dict[str, float]:
    if not weights:
        return {"HHI": np.nan, "ENH": np.nan, "Top3": np.nan, "Top5": np.nan, "Top10": np.nan}
    weights_arr = np.array(sorted(weights.values(), reverse=True))
    hhi = float(np.sum(weights_arr ** 2))
    enh = 1 / hhi if hhi else np.nan
    top = {
        "Top3": float(weights_arr[:3].sum()) if weights_arr.size >= 1 else np.nan,
        "Top5": float(weights_arr[:5].sum()) if weights_arr.size >= 1 else np.nan,
        "Top10": float(weights_arr[:10].sum()) if weights_arr.size >= 1 else np.nan,
    }
    return {"HHI": hhi, "ENH": enh, **top}


def histogram_with_var(returns: pd.Series, freq: str) -> go.Figure:
    fig = go.Figure()
    if returns.empty:
        fig.update_layout(template="plotly_dark")
        return fig
    hist_var = -np.quantile(returns.dropna(), 0.05)
    mean = returns.mean()
    std = returns.std(ddof=0)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
    dist = NormalDist(mean, std if std else 1e-6)
    pdf = np.array([dist.pdf(val) for val in x])
    counts, _ = np.histogram(returns.dropna(), bins=40, density=True)
    fig.add_trace(go.Histogram(x=returns, nbinsx=40, name="Returns", opacity=0.6, histnorm="probability"))
    scale = pdf.max() or 1
    density_scale = counts.max() or 1
    fig.add_trace(go.Scatter(x=x, y=pdf / scale * density_scale, name="Normal PDF"))
    fig.add_vline(x=-hist_var, line_color="red", annotation_text="VaR 95%", annotation_position="top right")
    fig.update_layout(template="plotly_dark", bargap=0.1)
    return fig
