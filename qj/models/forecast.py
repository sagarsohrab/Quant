"""Forecasting models: predict the *direction* of forward returns.

Why classification (up/down) instead of regression (price/return level)?
The forward return distribution is noisy and heavy-tailed; predicting its
sign is more robust and maps directly to a long/short trading decision.
We also report a probability so the backtester can size/confidence-trade.

Libraries are imported defensively: it tries LightGBM, falls back to
XGBoost, then to a plain sklearn histogram gradient boosting — so the
journey runs even on a bare box. Each estimator exposes the same API
(fit / predict / predict_proba / feature_importances_).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from qj.models.columns import default_feature_cols

_GBM = None
try:
    from lightgbm import LGBMClassifier
    _GBM = ("lightgbm", LGBMClassifier)
except Exception:  # noqa: BLE001  - OpenMP/dylib issues common on mac
    try:
        from xgboost import XGBClassifier
        _GBM = ("xgboost", XGBClassifier)
    except Exception:  # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingClassifier

        _GBM = ("sklearn-hgb", HistGradientBoostingClassifier)


def make_clf(random_state: int = 42, **kwargs):
    """Instantiate the best available gradient boosting classifier."""
    name, cls = _GBM
    if name == "lightgbm":
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=random_state,
        )
        params.update(kwargs)
        clf = cls(**params)
        clf.estimate_ = "lightgbm"
    elif name == "xgboost":
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )
        params.update(kwargs)
        clf = cls(**params)
        clf.estimate_ = "xgboost"
    else:  # sklearn HGB
        params = dict(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        )
        params.update(kwargs)
        clf = cls(**params)
        clf.estimate_ = "sklearn-hgb"
    return clf


def make_target(closing: pd.Series, horizon: int = 1) -> pd.Series:
    """True label: 1 if price rises over *horizon* bars, else 0.

    NOTE (lookahead): the label at bar t uses close[t+horizon], so it is
    only ever *fitted* in-sample and never available at prediction time.
    Callers must shift labels before training and must NOT use labels when
    producing out-of-sample / live predictions.
    """
    future = closing.shift(-horizon)
    target = future > closing
    # Treat undefined future (last `horizon` rows) as missing, not "down".
    target = target.where(future.notna())
    return target.astype(float).reindex(closing.index)


def make_dataset(
    frames: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = 1,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split an enriched frame into X (features) and y (target labels).

    The target is forward-looking; ``dropna`` removes warm-up/NaN rows and
    the final ``horizon`` rows where the target is undefined. Returns a
    feature matrix ready for train/val/test splitting.
    """
    out = frames.copy()
    out["__target__"] = make_target(out["close"], horizon)
    if feature_cols is None:
        feature_cols = [
            c for c in default_feature_cols() if c in out.columns
            and c not in {"__target__"}
        ]
    X = out[feature_cols]
    y = out["__target__"]
    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
    return X, y
