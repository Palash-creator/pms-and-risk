"""Data ingestion helpers with provider fallbacks."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Tuple

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)


def load_universe(csv_path: str, universe: str, random_n: int | None = None) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if universe == "S&P 100":
        return data.head(min(100, len(data)))
    if universe == "S&P 500":
        return data.head(min(500, len(data)))
    if universe == "Random (N)":
        if random_n is None:
            random_n = 10
        return data.sample(n=min(random_n, len(data)), random_state=42)
    return data


def _polygon_timespan(freq: str) -> str:
    return "month" if freq == "Monthly" else "day"


def _yfinance_interval(freq: str) -> str:
    return "1mo" if freq == "Monthly" else "1d"


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _polygon_request(symbol: str, start: datetime, end: datetime, freq: str, api_key: str) -> pd.Series:
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{_polygon_timespan(freq)}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    with httpx.Client(timeout=30) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    results = payload.get("results", [])
    if not results:
        return pd.Series(dtype=float)
    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df.set_index("timestamp", inplace=True)
    return df["c"].rename(symbol)


def _yfinance_request(symbol: str, start: datetime, end: datetime, freq: str) -> pd.Series:
    data = yf.download(symbol, start=start, end=end + timedelta(days=2), interval=_yfinance_interval(freq), progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    series = data["Close"].rename(symbol)
    series.index = series.index.tz_localize(None)
    return series


def _synthetic_series(symbol: str, start: datetime, end: datetime, freq: str) -> pd.Series:
    periods = 12 if freq == "Monthly" else 252
    years = max((end - start).days / 365.0, 1)
    steps = int(periods * years)
    if steps <= 0:
        steps = periods
    dt_index = (
        pd.date_range(start=start, periods=steps, freq="M" if freq == "Monthly" else "B")
        if freq == "Monthly"
        else pd.date_range(start=start, periods=steps, freq="B")
    )
    mu, sigma = 0.10, 0.20
    returns = np.random.default_rng(seed=hash(symbol) % 10000).normal(mu / periods, sigma / np.sqrt(periods), len(dt_index))
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dt_index, name=symbol)


def _parallel_fetch(
    symbols: Iterable[str],
    fetcher: Callable[[str], pd.Series],
    max_workers: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    series_map: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    sym_list = list(dict.fromkeys(symbols))
    total = len(sym_list)
    max_workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetcher, sym): sym for sym in sym_list}
        for i, future in enumerate(as_completed(future_map), start=1):
            sym = future_map[future]
            try:
                series = future.result()
                if not series.empty:
                    series_map[sym] = series
                else:
                    errors[sym] = "No data"
            except Exception as exc:  # noqa: BLE001
                errors[sym] = str(exc)
                LOGGER.exception("Failed to fetch %s", sym)
            if progress_callback:
                progress_callback(i, total, sym)
    frame = pd.DataFrame(series_map)
    frame.sort_index(inplace=True)
    return frame, errors


def fetch_prices(
    symbols: List[str],
    freq: str,
    lookback_years: int,
    polygon_key: str | None = None,
    max_workers: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    end = datetime.utcnow().date()
    start = end - relativedelta(years=lookback_years)
    LOGGER.info("Starting ingestion for %d symbols | freq=%s | lookback=%s", len(symbols), freq, lookback_years)
    provider_sequence: list[str] = []

    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.min.time())

    if polygon_key:
        provider_sequence.append("polygon")
        try:
            frame, errors = _parallel_fetch(
                symbols,
                fetcher=lambda sym: _polygon_request(sym, start_dt, end_dt, freq, polygon_key),
                max_workers=max_workers,
                progress_callback=progress_callback,
            )
            if not frame.empty:
                LOGGER.info("Polygon ingestion successful with %d columns", frame.shape[1])
                return frame, {"provider": "polygon", "errors": errors, "provider_sequence": provider_sequence}
        except Exception:  # noqa: BLE001
            LOGGER.exception("Polygon ingestion failed; falling back")

    provider_sequence.append("yfinance")
    try:
        frame, errors = _parallel_fetch(
            symbols,
            fetcher=lambda sym: _yfinance_request(sym, start_dt, end_dt, freq),
            max_workers=max_workers,
            progress_callback=progress_callback,
        )
        if not frame.empty:
            LOGGER.info("yfinance ingestion successful with %d columns", frame.shape[1])
            return frame, {"provider": "yfinance", "errors": errors, "provider_sequence": provider_sequence}
    except Exception:  # noqa: BLE001
        LOGGER.exception("yfinance ingestion failed; falling back to synthetic")

    provider_sequence.append("synthetic")
    frame, errors = _parallel_fetch(
        symbols,
        fetcher=lambda sym: _synthetic_series(sym, start_dt, end_dt, freq),
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
    LOGGER.info("Synthetic ingestion produced %d columns", frame.shape[1])
    return frame, {
        "provider": "synthetic",
        "errors": errors,
        "fallback_path": json.dumps(provider_sequence),
        "provider_sequence": provider_sequence,
    }
