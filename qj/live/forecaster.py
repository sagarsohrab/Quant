"""Live forecasting: turn streaming candles into real-time predictions.

The live system consumes the same feature pipeline as backtests/training
but operates on a rolling window of candles it has seen in production. It
holds a trained classifier and, on each closed candle, recomputes features
and emits a probability forecast.

Key design point: features must match training exactly (same column order
and definitions). We lock the feature list at construction time.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from qj.features.engine import add_features
from qj.models.columns import default_feature_cols


class LiveForecaster:
    """Emit per-bar probability forecasts from a rolling candle window."""

    def __init__(
        self,
        model,
        feature_cols: list[str] | None = None,
        warmup: int | None = None,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols or default_feature_cols()
        # We need enough history for the longest rolling window + warmup
        self.warmup = warmup or 60
        self._rows: deque[dict] = deque(maxlen=self.warmup + 200)
        self._df: pd.DataFrame | None = None
        self._feats: pd.DataFrame | None = None

    def _rebuild(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._rows).set_index("timestamp").sort_index()
        return df

    def _features(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._rebuild()
            self._feats = add_features(self._df)
        return self._feats

    def add_candle(self, row: dict) -> None:
        """Append one closed candle and clear cached features (staleness)."""
        self._rows.append(row)
        # Invalidate cached frame/features each new candle
        self._df = None
        self._feats = None

    def predict(self) -> dict | None:
        """Return the current probability forecast, or None if warmup unmet.

        The returned dict has keys: timestamp (of last candle), prob_up,
        direction (LONG/SHORT/FLAT), and the features used. Only uses
        trailing info, so it is directly actionable at runtime.
        """
        feats = self._features()
        if len(feats) < self.warmup + 20:
            return None
        last = feats.iloc[-1]
        row = {}
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            # feature drift — inform caller rather than silently break
            row["missing_features"] = missing
        X = feats[self.feature_cols].iloc[[-1]]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob_up = float(self.model.predict_proba(X)[0, 1])
        row["timestamp"] = feats.index[-1]
        row["close"] = float(feats["close"].iloc[-1])
        row["prob_up"] = prob_up
        row["direction"] = "LONG" if prob_up > 0.5 else "FLAT"
        return row
