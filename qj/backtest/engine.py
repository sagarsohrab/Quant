"""A backtesting engine for signal-driven strategies.

Core idea: take a *forecast* (probability of up-move) and translate it
into a *position* with optional confidence-based sizing, then simulate
the equity curve from the subsequent realised returns. Handles a simple
long/flat (and optionally short) strategy with transaction costs.

The engine is deliberately signal-agnostic: pass any series of position
weights (e.g. model probabilities, rule-based signals, a baseline) and it
will tell you how you would have done.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_to_position(
    prob_up: pd.Series,
    threshold: float = 0.5,
    allow_short: bool = False,
    size: float = 1.0,
) -> pd.Series:
    """Convert a probability forecast into a position series (-1..1).

    ``threshold`` is the up-probability at which we go long (or short if
    far below and allow_short). Position is 0 (flat) when in the middle.
    """
    if allow_short:
        pos = np.where(prob_up > threshold, size, np.where(prob_up < (1 - threshold), -size, 0.0))
    else:
        pos = (prob_up > threshold).astype(float) * size
    return pd.Series(pos, index=prob_up.index, name="position")


def backtest_positions(
    close: pd.Series,
    position: pd.Series,
    commission: float = 0.001,
    slippage: float = 0.0,
    cost_model: str = "per_trade",
) -> pd.DataFrame:
    """Simulate a strategy from a position series.

    Position is applied from the *next* bar (no lookahead): you hold the
    position you decided this bar through to the next. Returns a DataFrame
    indexed by time with raw/log returns and cumulative equity ignoring
    fees (gross) and after fees (net).

    ``cost_model``: "per_trade" (fee on each entry+exit) or "per_bar"
    (fee on every bar you hold).
    """
    pos = position.reindex(close.index).fillna(0.0)
    ret = close.pct_change().fillna(0.0)

    # Trade the position decided at bar t for the return realised at t+1:
    pos_lag = pos.shift(1).fillna(0.0)

    # Transaction cost via turnover (absolute change in position)
    turnover = pos_lag.diff().abs().fillna(pos_lag.abs())
    if cost_model == "per_trade":
        cost = turnover * commission
    elif cost_model == "per_bar":
        cost = pos_lag.abs() * commission
    else:
        raise ValueError(f"Unknown cost_model {cost_model!r}")

    gross = pos_lag * ret
    net = gross - cost
    frame = pd.DataFrame(
        {
            "return": ret,
            "position": pos_lag,
            "gross": gross,
            "cost": cost,
            "net": net,
        },
        index=close.index,
    )
    frame["gross_cum"] = (1 + frame["gross"]).cumprod()
    frame["net_cum"] = (1 + frame["net"]).cumprod()
    return frame


def performance_metrics(bt: pd.DataFrame, rf: float = 0.0, periods_per_year: int = 24) -> dict:
    """Compute standard performance statistics from a backtest frame."""
    net = bt["net"]
    total_return = bt["net_cum"].iloc[-1] - 1
    n = len(net)
    ann_factor = periods_per_year

    mean_daily = net.mean()
    std_daily = net.std()
    sharpe = (mean_daily - rf / ann_factor) / std_daily * np.sqrt(ann_factor) if std_daily > 0 else np.nan

    cum = bt["net_cum"]
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    # win rate among active (non-flat) trading days
    active = net[bt["position"].abs() > 0]
    win_rate = (active > 0).mean() if len(active) else np.nan

    # Calmar = annualised return / |max drawdown|
    yrs = n / periods_per_year
    ann_return = (1 + total_return) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "total_return": float(total_return),
        "annual_return": float(ann_return) if np.isfinite(ann_return) else float("nan"),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else float("nan"),
        "win_rate": float(win_rate) if np.isfinite(win_rate) else float("nan"),
        "volatility_ann": float(std_daily * np.sqrt(ann_factor)) if np.isfinite(std_daily) else float("nan"),
    }
