# The Quant Journey — Complete Master Guide

> *Everything that matters in this repo, explained with intuition, math, and code.
> Each section builds on the last. Read top to bottom, or jump to the layer you need.*

```
Data → Features → Models → Backtest → Live → ML Ops → LLM Architecture
```

---

## Layer 1 · Data — the foundation

### The mental model: a price is not informative

A single price tells you almost nothing — it's one point in a continuous
stream. So we **bucket time into fixed-length bars** and record the shape
of each bar. That shape is **OHLCV**:

| Field | Meaning |
|-------|---------|
| **O**pen | price at bar start |
| **H**igh | highest price reached inside the bar |
| **L**ow | lowest price reached inside the bar |
| **C**lose | price at bar end *(the all-important one)* |
| **V**olume | how much traded |

**Why five numbers and not just the last price?** Because the range
(H−L) encodes *volatility*, the direction (C vs O) encodes *movement*, and
volume encodes *conviction*. Every later layer consumes these.

In code that "canon" is just a fixed contract:

```python
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
```

### The abstraction: source-agnostic data

The pipeline should not care whether prices come from Binance, Yahoo, or
Zerodha. We enforce this with an **abstract interface** every provider
must implement:

```python
class BaseDataProvider(abc.ABC):
    @abc.abstractmethod
    def fetch_history(self, start, end):
        """download OHLCV between two timestamps"""

    @abc.abstractmethod
    def fetch_latest(self):
        """get the most recent candle"""
```

Every provider returns the same **`MarketDataFrame`** — a normalized
frame tagged with `symbol`, `source`, `freq`. So downstream code calls
`provider.fetch_history(...)` and is blind to *which* provider answered.

> **Why this matters:** swap Binance for Yahoo tomorrow and *nothing* else
> in the repo changes. This is interface design, and it's the same pattern
> serious software uses everywhere.

### Normalization: imposing order on messy providers

APIs are inconsistent (uppercase columns, different time units, volume in
different currencies). `_normalize` fixes that in five steps:

```python
df.columns = [c.lower() for c in df.columns]       # 1. lowercase
keep = [c for c in OHLCV_COLS if c in df.columns]  # 2. keep known cols only
df["timestamp"] = pd.to_datetime(df["timestamp"], unit=ts_unit, utc=True)  # 3. parse time
df[c] = pd.to_numeric(df[c], errors="coerce")       # 4. force numbers
df = df.set_index("timestamp").sort_index()          # 5. index + sort
```

**The time-unit bug (a classic):** the same integer can mean wildly
different times depending on its unit:

```python
ms = 1700000000000
pd.to_datetime(ms, unit="ms", utc=True)  # 2023-11-14  ✅ correct
pd.to_datetime(ms, unit="s",  utc=True)  # year 55840    ❌ nonsense
pd.to_datetime(ms, unit="ns", utc=True)  # 1970-01-01    ❌ nonsense
```

One wrong unit silently corrupts every timestamp. This is why `ts_unit`
is an explicit parameter.

### Why the timestamp must be the INDEX

Pandas operations (`shift`, `rolling`, `pct_change`, `reindex`) **align on
the index**. Making `timestamp` the index turns "a list of rows" into
*a time series*, which buys you:

1. **True chronological order** — `sort_index()` guarantees row 0 is the
   earliest bar. A plain column preserves row order, which APIs often
   return newest-first and with gaps.
2. **Safe joins** — merging two frames aligns by *time*, never by row.
3. **Time-aware math** — `.shift(1)`, `.rolling(N)`, `.pct_change()` are
   defined in terms of the index.

A missing bar or wrong order silently corrupts results — exactly the kind
of subtle bug that kills high-frequency strategies.

### Pagination: getting *all* the history

Binance caps a kline request at **1000 candles**. Fetching years of data
means looping:

```python
while cursor < end_ms:              # keep going until we reach the end
    batch = self._klines(cursor, end_ms=end_ms)
    if not batch:
        break
    rows.extend(batch)
    if len(batch) < KLINES_LIMIT:   # fewer than 1000 = last page
        break
    cursor = last_open + bar_ms     # resume one bar after the last
    time.sleep(0.1)                 # be polite to the free API
```

Correct resumption point + a `sleep` for rate limits = polite, complete
fetching.

---

## Layer 2 · Features — the craft of signal construction

**Why transform raw prices?** Raw prices are *non-stationary*: a $1 move
means different things at $500 vs $50,000, and the level drifts forever.
Models choke on that. We transform prices into **stationary, informative**
features.

### Returns — the stationarity fix

The most important transform. **Log returns** are preferred:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

```python
def log_returns(close):
    return np.log(close).diff()
```

**Why log and not simple?**
- **Stationary** — removes the drift in level.
- **Additive over time** — total log return = sum of daily log returns.
- **Close to symmetric** — a +10% then −10% gives ~0% net.

Simple (arithmetic) returns differ slightly but read more intuitively:

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

### Momentum — capturing trend

How much has price moved over `N` bars:

$$
\text{mom}_N = \frac{P_t}{P_{t-N}} - 1
$$

```python
def momentum(close, window):
    return close / close.shift(window) - 1
```

Positive = trending up; negative = trending down. Multiple windows
(`mom_3`, `mom_10`, `mom_20`) capture short, medium, long trend.

### Volatility — the risk regime

Realized volatility = rolling std of log returns:

$$
\text{vol}_N = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\left(r_{t-i}-\bar{r}\right)^2}
$$

```python
def rolling_vol(log_ret, window):
    return log_ret.rolling(window).std()
```

This is your **"market noise" gauge**. High vol = choppy/risk-on;
low vol = calm. It answers "is this a big or normal move?"

### Z-score — how stretched is price?

Distance of price from its rolling mean, in rolling-std units:

$$
z = \frac{P_t - \text{mean}_N(P)}{\text{std}_N(P)}
$$

```python
def rolling_zscore(close, window):
    return (close - close.rolling(window).mean()) / close.rolling(window).std()
```

| z | meaning |
|---|---------|
| +2 | price 2 std above its recent mean (stretched up) |
|  0 | right at the average |
| −2 | price 2 std below (stretched down) |

### Calendar features — seasonality

Crypto trades 24/7 but still shows regular patterns:

```python
out["day_of_week"] = out.index.dayofweek
out["hour_of_day"] = out.index.hour
```

### The ONE rule: no lookahead ⚠️

> **A feature at time `t` may only use data up to and including `t`.**
> It must *never* peek at future bars to construct itself.

Every feature here is built with `rolling()` / `shift()`, which use only
*trailing* information by construction. This rule is the difference
between an honest model and one that secretly "knows the answer."

```python
def add_features(df):
    out = df.copy()
    close = out["close"]
    out["log_return"] = log_returns(close)        # uses P_t, P_{t-1}  ✅
    out["mom_5"]      = momentum(close, 5)        # uses last 5 closes   ✅
    # ... all trailing-only ...
    return out
```

**Translation, almost literally:** "tell me what you know *now*, and I'll
decide what to do *next*."

---

## Layer 3 · Models — forecasting direction

### Why classification, not regression?

The market return is noisy and fat-tailed — predicting an *exact* number
is hopeless. So instead we predict the **direction** (up/down):

- Maps directly to a trading decision (long vs flat).
- A calibrated **probability** lets us size by confidence.
- Far more robust to noise.

### The label (target)

```python
def make_target(close, horizon=1):
    future = close.shift(-horizon)
    target = (future > close).where(future.notna())
    return target.astype(float)
```

Result: `1` if price rises over the next `horizon` bars, `0` if not.

> **Lookahead warning:** the label uses the *future*, so it exists only
> for training. `make_dataset` drops the final rows where the target is
> undefined.

### The estimator: gradient boosting

We try LightGBM → XGBoost → sklearn HGB (so it runs anywhere). Gradient
boosting trains shallow trees *sequentially*, each new tree learning the
**residuals** (mistakes) of all the previous ones:

$$
\hat{y}^{(m)} = \hat{y}^{(m-1)} + \eta \cdot h_m(x)
$$

where each $h_m$ fits the residual $y - \hat{y}^{(m-1)}$, and $\eta$ is a
small learning rate. Many weak learners combined = one strong model.

### The critical method: walk-forward (not random split)

Random train/test splits leak *overlapping trends* across the boundary and
overstate performance. **Walk-forward** rolls an expanding training window:
train on past, validate on strictly-future data, slide forward, repeat.

```
Split 1: [train ............][validate]
Split 2: [train ..................][validate]
Split 3: [train ........................][validate]
```

```python
res = walk_forward_evaluate(feats, cols, horizon=1, n_splits=5)
print(res["acc"], res["logloss"])   # honest, out-of-sample numbers
```

An accuracy of ~0.52 vs 0.50 random on BTC momentum is *typical* and
*honest* — market direction is hard. This is the layer that teaches you
to distrust a strategy before trusting it.

---

## Layer 4 · Backtesting — did it make money?

### The pipeline: forecast → position → equity

```python
prob = walk_forward_predictions(...)          # 1. out-of-sample P(up)
pos  = signal_to_position(prob, threshold=0.5)  # 2. forecast → position
bt   = backtest_positions(close, pos, commission=0.001)  # 3. simulate
m    = performance_metrics(bt)                 # 4. Sharpe, drawdown, ...
```

### Forecasting → position

Go long when P(up) > threshold, else stay flat:

```python
pos = (prob > threshold).astype(float) * size
```

### No lookahead: you act on the NEXT bar

The position decided at bar `t` is applied to the return realised at
`t+1` — you never earn the return that *created* your signal:

```python
pos_lag = pos.shift(1)   # hold what you decided LAST bar, for THIS bar
gross = pos_lag * close.pct_change()
```

### Transaction costs — the silent killer

Each trade costs money. We charge on **turnover** (how much position
changed):

```python
turnover = pos_lag.diff().abs()
cost = turnover * commission
net = gross - cost
```

Fees are frequently the difference between a profitable and a losing
strategy. Never skip them.

### The metrics

| Metric | Formula (idea) | Meaning |
|--------|----------------|---------|
| Sharpe | (return − rf) / std(return) × √periods | return per unit of risk |
| Max drawdown | min(equity / cummax − 1) | worst peak-to-trough loss |
| Win rate | P(active day > 0) | how often you're right |
| Calmar | annual return / max drawdown | risk-adjusted trend |

```python
framing = performance_metrics(bt, periods_per_year=24 * 365)
print(framing["total_return"], framing["sharpe"], framing["max_drawdown"])
```

**Honest note:** our first momentum model often *loses* to buy-and-hold on
choppy data. That's normal and educational. A backtest showing effortless
profit is usually a leaky-data bug, not alpha.

---

## Layer 5 · Live — streaming, production-style

### Two data models

| REST (`qj/data`) | WebSocket (`qj/live`) |
|------------------|------------------------|
| You **pull** history | Server **pushes** new data |
| For backfill | For real-time |
| Request/response | Persistent connection |

Binance's WebSocket (`wss://...<symbol>@kline_<interval>`) pushes each
candle the instant it closes. We consume it as an async iterator:

```python
async for frame in stream_klines("BTCUSDT", "1m"):
    print(frame["close"])    # a new candle, live
```

### The LiveForecaster loop

A live forecaster keeps a rolling window of candles and, per closed bar:

1. Rebuilds **features with the exact same `add_features`** as training
   (feature parity is non-negotiable).
2. Scores the last bar with the trained model.
3. Emits `prob_up` and a `LONG` / `FLAT` decision.

This is the backtest's loop, accelerated to real time:

```python
async for row in stream_klines(symbol, freq):
    fc.add_candle(row)
    if pred := fc.predict():
        print(pred["direction"], pred["prob_up"])
```

**Why this transfers everywhere:** stream → transform → model → act is the
same pattern behind live LLM serving, fraud detection, and every
real-time ML system.

---

## Layer 6 · ML Ops — keep models honest in production

### Model registry: version everything

Every iteration must be auditable — what metrics, what artifact, when:

```python
registry.register("es", "models/es_v3.pkl", metrics={"acc": 0.56})
registry.latest("es")   # ModelRecord(version=3, ...)
registry.history("es")  # all versions, oldest → newest
```

Auto-incrementing versions let you always know *which* model is deployed.

### Drift monitoring: markets change

A model trained on one regime can silently decay as conditions shift.
Two checks:

1. **Distribution drift** — do *live* feature stats differ from *training*
   stats? Measure per-feature mean shift (in std units):

$$
\text{drift}_f = \frac{|\bar{x}_{\text{live}} - \mu_{\text{train}}|}{\sigma_{\text{train}}}
$$

2. **KS statistic** — classic two-sample distribution-distance test,
   robust and dependency-light:

```python
from qj.mlops.drift import ks_statistic
ks_statistic(reference_values, live_values)  # 0 = identical, 1 = very different
```

```python
monitor = DriftMonitor(reference_feats, threshold=1.0)
alarm = monitor.score_window(live_feats)
if alarm["drift_alarm"]:
    retrain()   # the regime changed
```

Same discipline keeps any model — including LLMs — healthy at scale.

---

## Layer 7 · LLM Architecture — the transformer, from scratch

The MiniGPT in `qj/llm/minigpt.py` implements a real GPT-style transformer
in ~150 lines — the exact blueprint behind modern LLMs. Three conceptual
parts:

### a) Embeddings — token → vector

Characters/words become ids; embeddings map each id to a *learned* vector
so similar tokens land near each other:

```python
self.token_embed = nn.Embedding(vocab_size, d_model)
```

### b) Positional encoding — order matters

A transformer sees a *bag* of tokens with no innate sense of order, so we
inject position (GPT uses learned; the paper used sinusoids):

```python
positions = torch.arange(0, T)
x = x + self.embed(positions)   # add position info to each token row
```

### c) Self-attention — the real magic ⭐

Each token decides how much to attend to every other token using
**query / key / value** triples:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
#   Q  "how much do I care about others"
#   K  "how relevant am I to others"        → similarity = relevance
#   V  "what information do I carry"
```

**Multi-head** runs this in parallel sub-spaces, letting the model track
several relationships at once. A **causal mask** zeros out future tokens so
the model can only attend *backward* — it never sees the future:

```python
causal = torch.tril(torch.ones(T, T))        # lower triangle
scores = scores.masked_fill(causal == 0, -inf)  # blind to future
```

### d) The residual block

Each transformer block = attention + feed-forward, with **residual
connections** (add the input back so gradients flow) and **layer-norm**
for training stability:

```python
x = x + self.attn(self.ln1(x))   # residual + pre-norm
x = x + self.ff(self.ln2(x))
```

### e) Training: next-token prediction

An LLM learns by predicting the *next* token from the previous ones —
self-supervised, no labels needed:

```python
logits = model(idx)                     # (batch, seq, vocab)
loss = CrossEntropyLoss(logits.view(-1, vocab), targets.view(-1))
```

The output head projects hidden states to vocabulary-size scores; softmax
turns them into probabilities; cross-entropy measures how wrong we were.

**Try it — watch an LLM learn:**
```
uv run --extra llm python -m qj.llm.train_demo
```
Watch perplexity fall from ~25 → <2 as coherent text emerges on your MPS GPU.

### Scaling the same design to real LLMs

MiniGPT is ~160k params. Real LLMs scale the *identical* design:
- **Bigger** — more layers, wider dims, more data (GPT-1 → GPT-4).
- **Better tokenizers** — subword/BPE so a large vocab covers any text.
- **Modernization** — GELU, RMSNorm, RoPE, KV-cache for fast inference.
- **Alignment** — RLHF / preference tuning on top of the pretrained base.
- **Efficiency** — quantization (fp16 → int8), pruning, distillation.

### The bridge "from tech to LLM architecture"

| Interest | Path |
|----------|------|
| Hardware | GPUs, memory bandwidth, batch size, quantization |
| Data | Parquet, streaming (WS), storage engines, distributed compute |
| Infra | queues, containers, registries, monitoring, retries |
| Tools | PyTorch autodiff, llama.cpp (CPU), vLLM (GPU) |

---

## The whole stack in one picture

```
┌──────────────────────────────────────────────┐
│ APPLICATIONS   forecasting, backtest, live    │
├──────────────────────────────────────────────┤
│ LLM/MODEL      MiniGPT transformer (07/)      │
├──────────────────────────────────────────────┤
│ ML PIPELINE    features→train→eval→deploy     │
│                (02/,03/,06/)                  │
├──────────────────────────────────────────────┤
│ DATA/INFRA     Binance REST+WS, Parquet       │
│                (01/,05/)                      │
├──────────────────────────────────────────────┤
│ COMPUTE        CPU/GPU (MPS), memory, files   │
└──────────────────────────────────────────────┘
```

Every ML system — quant or LLM — sits on this same skeleton. **Features,
training, evaluation, and deployment are the same discipline at every
level.**

---

## Cheat-sheet of the most important ideas

| # | Idea | Where |
|---|------|-------|
| 1 | Normalize data to one canonical shape | `02` base.py |
| 2 | Timestamp must be a time index | `01` |
| 3 | **Never look ahead** when building features/labels | `02` |
| 4 | Predict *direction*, not exact value | `03` |
| 5 | Evaluate with walk-forward, not random split | `03` |
| 6 | Always charge transaction costs | `04` |
| 7 | Mirror training features exactly in live | `05` |
| 8 | Version models + monitor for drift | `06` |
| 9 | Transformer = embeddings + positions + attention | `07` |
| 10 | LLMs = next-token prediction, scaled up | `07` |
