"""Tests for mlops layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qj.mlops.drift import DriftMonitor, ks_statistic
from qj.mlops.registry import ModelRegistry


def test_ks_same_vs_different():
    a = np.random.default_rng(0).normal(0, 1, 1000)
    same = np.random.default_rng(1).normal(0, 1, 1000)
    diff = np.random.default_rng(2).normal(3, 1, 1000)
    assert ks_statistic(a, same) < ks_statistic(a, diff)


def test_registry_versions_and_latest(tmp_path):
    reg = ModelRegistry(tmp_path)
    m1 = reg.register("es", "models/es_v1.pkl", metrics={"acc": 0.52})
    m2 = reg.register("es", "models/es_v2.pkl", metrics={"acc": 0.55})
    assert m1.version == 1 and m2.version == 2
    assert reg.latest("es").version == 2
    assert len(reg.history("es")) == 2
    assert reg.list_names() == ["es"]


def test_drift_monitor_detects_shift():
    idx = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    ref = pd.DataFrame({"mom_5": np.random.default_rng(1).normal(0, 0.01, 200)}, index=idx)
    same = pd.DataFrame({"mom_5": np.random.default_rng(2).normal(0, 0.01, 200)}, index=idx)
    drifted = pd.DataFrame({"mom_5": np.random.default_rng(3).normal(2.0, 0.01, 200)}, index=idx)

    mon = DriftMonitor(ref, threshold=5.0)  # lenient so normal noise passes
    assert not mon.score_window(same)["drift_alarm"]

    strict = DriftMonitor(ref, threshold=0.5)
    assert strict.score_window(drifted)["drift_alarm"]
