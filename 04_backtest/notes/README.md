# Layer 4: Backtesting — did the strategy actually work?

A model predicts; a backtest tells you whether acting on those predictions
would have made (or lost) money — honestly, with costs and no future
peeking.

## Code map
- `qj/backtest/engine.py` — positions, equity curve, metrics

## The flow
1. **Forecast → position.** `signal_to_position` turns a probability into a
   position: go long when P(up) > threshold, else flat (with optional
   short and confidence sizing).
2. **No lookahead.** The position decided at bar *t* is applied to the
   return realised at bar *t+1* (`position.shift(1)`). You never earn the
   return that *created* your signal.
3. **Costs.** A `turnover`-based fee models commissions on each trade —
   fees are often the difference between a profitable and a losing strategy.
4. **Equity curve & metrics.** `performance_metrics` computes total/ann.
   return, Sharpe, max drawdown, win rate, annualized volatility.

## Key metrics
| Metric | Meaning |
|--------|---------|
| Sharpe | return per unit of risk; >1 decent, >2 great |
| Max drawdown | worst peak-to-trough loss (sleep factor) |
| Calmar | annual return ÷ max drawdown |
| Win rate | fraction of active days profitable |

## Honest note
Our first momentum model *loses* to buy-and-hold about half the time on
choppy data — that is normal and educational. A backtest that shows
effortless profit is more often a leaky data bug than alpha. This layer
is where you *learn to distrust* a strategy before trusting it.

## Try (the full pipeline)
```
uv run --extra ml python - <<'PY'
from qj.data import ParquetStore
from qj.features import add_features
from qj.models import default_feature_cols, walk_forward_predictions
from qj.backtest import signal_to_position, backtest_positions, performance_metrics
mdf = ParquetStore('data').load('binance','BTCUSDT','1h')
feats = add_features(mdf.df)
prob = walk_forward_predictions(feats, default_feature_cols(), horizon=1, n_splits=5)
pos = signal_to_position(prob, threshold=0.5)
bt = backtest_positions(mdf.df['close'], pos, commission=0.001)
m = performance_metrics(bt, periods_per_year=24*365)
print('return %.1f%%  sharpe %.2f  maxdd %.1f%%' % (m['total_return']*100, m['sharpe'], m['max_drawdown']*100))
PY
```
