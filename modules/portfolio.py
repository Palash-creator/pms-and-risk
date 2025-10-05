"""Portfolio state management utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Simple container for portfolio weights."""

    symbols: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    def normalized_weights(self) -> Dict[str, float]:
        """Return weights normalized to sum to 1."""
        if not self.weights:
            return {}
        total = sum(max(w, 0.0) for w in self.weights.values())
        if total == 0:
            n = len(self.weights)
            return {s: 1.0 / n for s in self.weights} if n else {}
        return {sym: max(w, 0.0) / total for sym, w in self.weights.items() if sym in self.symbols}

    def vector(self, ordered_symbols: Iterable[str]) -> np.ndarray:
        """Return ordered weight vector for computation."""
        weights = self.normalized_weights()
        return np.array([weights.get(sym, 0.0) for sym in ordered_symbols], dtype=float)

    def to_frame(self) -> pd.DataFrame:
        data = [{"symbol": sym, "weight": self.normalized_weights().get(sym, 0.0)} for sym in self.symbols]
        return pd.DataFrame(data)


def validate_weights(symbols: Iterable[str], weights: Dict[str, float]) -> bool:
    if not symbols:
        return False
    if not weights:
        return False
    total = sum(weights.get(sym, 0.0) for sym in symbols)
    return abs(total - 1.0) < 1e-6


def generate_dry_run(universe: pd.DataFrame, seed: int | None = None) -> PortfolioState:
    """Pick sectors and build a random diversified portfolio."""
    rng = random.Random(seed)
    if universe.empty:
        return PortfolioState()
    top_sectors = (
        universe.groupby("sector")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    selections: list[str] = []
    for sector in top_sectors:
        tickers = universe.loc[universe["sector"] == sector, "symbol"].tolist()
        rng.shuffle(tickers)
        selections.extend(tickers[:2])
    selections = selections[:10]
    if not selections:
        selections = rng.sample(universe["symbol"].tolist(), k=min(10, len(universe)))
    raw_weights = np.array([rng.random() for _ in selections])
    weights = raw_weights / raw_weights.sum()
    return PortfolioState(symbols=selections, weights=dict(zip(selections, weights)))
