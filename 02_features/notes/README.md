# Layer 2: Features — turning prices into signals

Models don't learn well from raw prices (non-stationary: a $1 move means
different things at $500 vs $50,000). We transform prices into *stationary,
informative* features. This is the craft of quant research.

## Code map
- `qj/features/engine.py` — `add_features()` and the primitives

## The features
| Feature | Formula (concept) | Why it matters |
|---------|-------------------|----------------|
| `log_return` | `ln(P_t / P_{t-1})` | Stationary; symmetric; variance-additive |
| `simple_return` | `P_t / P_{t-1} - 1` | Arithmetic, intuitive |
| `mom_N` | `P_t / P_{t-N} - 1` | Trend/momentum over N bars |
| `vol_N` | rolling std of log returns | Volatility regime (risk) |
| `zscore_N` | `(P - mean)/std` over N | How stretched price is from its mean |
| `min_N`,`max_N` | rolling low/high | Support/resistance levels |
| `day_of_week`,`hour_of_day` | calendar | Regular seasonality |

## The one rule: no lookahead
Every feature at time *t* must use **only** data up to *t* (values from
`rolling(window)` are computed with trailing info only). We never peek at
future bars to build a feature — that would give a model "answers" it
couldn't have in production. This is enforced by construction here.

## Read it in code
`add_features()` computes all of them in one pass. Compare a `log_return`
column to a `simple_return` on a big BTC move to see the difference.

## Try
```python
from qj.data import ParquetStore
from qj.features import add_features
mdf = ParquetStore('data').load('binance','BTCUSDT','1h')
feats = add_features(mdf.df)
print(feats.head())
```
