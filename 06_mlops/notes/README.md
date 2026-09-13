# Layer 6: ML Ops — keep models honest in production

A model that worked offline can quietly decay as markets (or data) drift.
ML ops is the machinery to train, version, deploy, and monitor models
reliably.

## Code map
- `qj/mlops/registry.py` — `ModelRegistry` + `ModelRecord`
- `qj/mlops/drift.py` — `DriftMonitor`, Kolmogorov–Smirnov statistic

## Model registry
Every iteration should be versioned and auditable: what were its metrics,
what file is its artifact, when was it created. `ModelRegistry` is a small
JSON-backed store that auto-increments versions and tracks the `latest`
model per name.

```
registry.register("es", "models/es_v3.pkl", metrics={"acc":0.56})
registry.latest("es")   # ModelRecord(version=3, ...)
```

## Drift monitoring
Markets are non-stationary. The distribution of features you trained on
(year 1) can shift (year 2) — a "regime change". Two checks:

1. **Distribution drift**: compare live feature stats to the training-time
   reference. `DriftMonitor` computes a per-feature mean-shift (in std
   units) and raises an alarm past a threshold.
2. **Prediction drift** (if you have outcomes): watch rolling accuracy /
   calibration degrade.

The KS statistic is the classic two-sample distribution-distance test; a
dependency-free version is included.

## The big picture
This layer points at the *whole* production discipline:
- **Train** → validate → register (versioned, with metrics)
- **Serve** → monitor (features + predictions)
- **Alert / retrain** when drift exceeds tolerance

It's the same discipline that keeps LLMs and any model healthy at scale.
The from-scratch MiniGPT (`qj/llm/`) is a model like any other — it too
would sit behind a registry and a drift monitor in production.

## Try
```
uv run --extra ml python -m pytest qj/mlops/ -q
```
