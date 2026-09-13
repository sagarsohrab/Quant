"""Walk-forward evaluation (no-lookahead model assessment).

A single random train/test split overstates performance on time series
because overlapping trends leak across the boundary. Walk-forward splits
the data into expanding/nested training windows and rolls a validation
window forward, retraining each step — mimicking live deployment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from qj.models.forecast import make_clf, make_dataset


def walk_forward_predictions(
    frames: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 1,
    n_splits: int = 4,
    train_pct: float = 0.6,
    random_state: int = 42,
) -> pd.Series:
    """Return out-of-sample up-probabilities over a rolling window.

    Like ``walk_forward_evaluate`` but returns the model's predicted
    P(up) on every validation bar, in original time order. These are
    strictly out-of-sample predictions (each bar's pred came from a model
    trained only on data before it), so they can feed a backtest honestly.
    Rows before the first train window get NaN (no prediction yet).
    """
    X, y = make_dataset(frames, feature_cols=feature_cols, horizon=horizon)
    n = len(y)
    n_train = int(train_pct * n)
    edges = np.linspace(n_train, n - 1, n_splits + 1, dtype=int)

    prob = pd.Series(np.nan, index=y.index, name="prob_up")
    for i in range(n_splits):
        tr_end = int(edges[i])
        va_end = int(edges[i + 1])
        if va_end <= tr_end:
            continue
        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_va = X.iloc[tr_end:va_end]
        if len(y_tr) < 50 or len(X_va) == 0:
            continue
        clf = make_clf(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            prob.iloc[tr_end:va_end] = clf.predict_proba(X_va)[:, 1]
    # reindex back to the original frame's alignment
    return prob.reindex(frames.index)


def walk_forward_evaluate(
    frames: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 1,
    n_splits: int = 4,
    train_pct: float = 0.6,
    random_state: int = 42,
    return_diagnostics: bool = True,
) -> dict:
    """Roll a training/validation window forward and report honest metrics.

    Returns dict with: acc (mean), logloss, n (evaluated rows), estimator
    used, per-split accuracy list, and (if return_diagnostics) the features
    used.
    """
    X, y = make_dataset(frames, feature_cols=feature_cols, horizon=horizon)
    # Keep the temporal order for splitting.
    n = len(y)
    n_train = int(train_pct * n)
    # Build expanding training edges
    edges = np.linspace(n_train, n - 1, n_splits + 1, dtype=int)
    accs, losses = [], []
    n_eval = 0
    estimator_name = "unknown"

    for i in range(n_splits):
        tr_end = int(edges[i])
        va_end = int(edges[i + 1])
        # validation window: [tr_end, va_end)
        if va_end <= tr_end:
            continue
        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_va, y_va = X.iloc[tr_end:va_end], y.iloc[tr_end:va_end]
        if len(y_tr) < 50 or len(y_va) < 5:
            continue

        clf = make_clf(random_state=random_state)
        estimator_name = getattr(clf, "estimate_", estimator_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            probs = clf.predict_proba(X_va)[:, 1]
        accs.append(accuracy_score(y_va, preds))
        loss = log_loss(y_va, probs)
        losses.append(loss)
        n_eval += len(y_va)

    result = {
        "acc": float(np.mean(accs)) if accs else float("nan"),
        "logloss": float(np.mean(losses)) if losses else float("nan"),
        "n_splits": len(accs),
        "n_evaluated": n_eval,
        "per_split_acc": accs,
        "per_split_logloss": losses,
        "estimator": estimator_name,
    }
    if return_diagnostics:
        result["n_features"] = X.shape[1]
    return result
