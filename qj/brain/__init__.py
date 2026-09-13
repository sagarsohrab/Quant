"""WorldQuant BRAIN alpha trainer: expressions, neutralization, evaluation."""

from qj.brain.alphas import ALPHAS, get_alpha
from qj.brain.evaluate import compute_signal, forward_returns, sharpe_style, simulate_alpha, turnover
from qj.brain.neutralize import (
    cross_sectional_rank,
    demean,
    neutralize_market,
    neutralize_series,
)

__all__ = [
    "ALPHAS",
    "get_alpha",
    "compute_signal",
    "forward_returns",
    "sharpe_style",
    "simulate_alpha",
    "turnover",
    "cross_sectional_rank",
    "demean",
    "neutralize_market",
    "neutralize_series",
]
