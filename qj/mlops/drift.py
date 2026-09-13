"""Drift monitoring: detect when the live data moves away from training.

Markets evolve; a model trained on 2024 price dynamics may degrade in 2025.
We track two kinds of shift:

- **Distribution drift**: compare the live feature distribution to the
  training distribution (e.g. via a Kolmogorov–Smirnov test or simple
  mean/std distance). Silently catching regime changes early.
- **Prediction drift**: if you receive actual outcomes, watch the model's
  accuracy/calibration degrade over rolling windows.

This module implements a lightweight, dependency-light drift detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (0=identical, 1=very different)."""
    return _ks(a, b)


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic without scipy dependency."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return 1.0
    # Empirical CDF diff at merged points
    all_vals = np.unique(np.concatenate([a, b]))
    if len(all_vals) == 0:
        return 0.0
    ecdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    ecdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(ecdf_a - ecdf_b)))


class DriftMonitor:
    """Compare a rolling reference distribution to a streaming one."""

    def __init__(
        self,
        reference: pd.DataFrame,
        threshold: float = 0.2,
    ) -> None:
        """``reference`` is the training-time feature frame used as baseline."""
        self.reference = reference
        self.threshold = threshold
        self.reference_stats: dict[str, tuple[float, float]] = {}
        for col in reference.columns:
            vals = reference[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals):
                self.reference_stats[col] = (float(np.mean(vals)), float(np.std(vals)))

    def score_window(self, live: pd.DataFrame) -> dict[str, float]:
        """Return per-feature drift metrics and an aggregated alarm flag."""
        out: dict[str, float] = {"aggregate_drift": 0.0, "drift_alarm": False}
        if live.empty:
            return out
        n = 0
        agg = 0.0
        for col, (mu, sd) in self.reference_stats.items():
            if col not in live.columns:
                continue
            vals = live[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            # Normalized distributional distance (mean shift in std units)
            dist = abs(float(np.mean(vals)) - mu) / (sd + 1e-9)
            out[f"{col}_drift"] = dist
            agg += dist
            n += 1
        agg = agg / n if n else 0.0
        out["aggregate_drift"] = float(agg)
        out["drift_alarm"] = bool(agg > self.threshold)
        return out
