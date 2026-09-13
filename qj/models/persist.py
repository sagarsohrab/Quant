"""Persist / load trained classifiers and a one-shot train helper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from qj.features.engine import add_features
from qj.models.columns import default_feature_cols
from qj.models.forecast import make_clf, make_dataset


def train_on_frame(
    frames: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = 1,
    drop_last: int = 0,
    random_state: int = 42,
):
    """Train a classifier on the *entire* frame (for a live/production model).

    Use this after you've validated via walk-forward; it fits a final model
    on all available history so live predictions are as informed as possible.
    Returns (model, feature_cols).
    """
    feat = frames if all(c in frames.columns for c in ["log_return", "mom_5"]) else add_features(frames)
    cols = feature_cols or default_feature_cols()
    X, y = make_dataset(feat, feature_cols=cols, horizon=horizon)
    if drop_last:
        X, y = X.iloc[:-drop_last], y.iloc[:-drop_last]
    clf = make_clf(random_state=random_state)
    clf.fit(X, y)
    return clf, cols


def save_model(clf, path: str | Path, feature_cols: list[str] | None = None) -> Path:
    """Persist a trained classifier with its feature contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class": clf.__class__.__name__,
        "model": clf,
        "feature_cols": feature_cols or default_feature_cols(),
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_model(path: str | Path):
    """Load a persisted (model, feature_cols) pair."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["model"], payload["feature_cols"]
