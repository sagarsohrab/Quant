# Layer 3: Models — forecasting market direction

We build classifiers that predict the *direction* of forward returns
(up/down) rather than the exact price or return magnitude.

## Why classification over regression?
- Returns are noisy and fat-tailed; regressing an exact number is hard.
- Predicting the sign maps directly to a trading decision (long/flat).
- A calibrated probability lets us size positions by confidence.

## Code map
- `qj/models/columns.py` — the standard feature list
- `qj/models/forecast.py` — target + dataset construction, estimator factory
- `qj/models/walkforward.py` — honest time-series evaluation
- `qj/models/persist.py` — train/save/load helpers

## The target (label)
`make_target(close, horizon)`:
```
label_t = 1  if  close[t+horizon] > close[t]   else  0
```
We predict P(up) over the next `horizon` bars. **Lookahead warning**: the
label uses the future, so it exists only for *training* — never at
prediction time. `make_dataset` drops the final rows where the target is
undefined.

## Estimator: gradient boosting
We try LightGBM → XGBoost → sklearn HGB in order (so it runs anywhere).
Gradient boosting sequentially fits trees to the *residuals* of the
previous model, combining many weak learners into a strong one. It's the
workhorse for tabular forecasting.

## The critical part: walk-forward, not random split
Standard train/test splits leak overlapping-trend information across the
boundary and overstate performance. Walk-forward rolls an *expanding*
training window forward and validates on strictly-future data each step —
mimicking real live deployment. `walk_forward_evaluate` and
`walk_forward_predictions` implement this.

## Try
```
uv run --extra ml python -c "
from qj.data import ParquetStore
from qj.features import add_features
from qj.models import default_feature_cols, walk_forward_evaluate
mdf = ParquetStore('data').load('binance','BTCUSDT','1h')
res = walk_forward_evaluate(add_features(mdf.df), default_feature_cols(), horizon=1, n_splits=5)
print('acc', round(res['acc'],3), 'logloss', round(res['logloss'],3))
"
```
