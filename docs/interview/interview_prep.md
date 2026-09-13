# Quantitative Researcher Interview — Question Bank (WorldQuant + general QR)

Practice-ready Q/A. Each answer is concise, correct, and intuition-first. Numbers are rough-but-right where an interviewer wants a Fermi estimate. Review every section, then drill the ones you're weakest on.

---

## 1. Alpha & Signal Intuition

**Q1. What is an "alpha"?**
An alpha is a signal/expression that, when traded, is *expected* to produce return independent of common risk factors (market, industry, etc.). WorldQuant BRAIN formalizes alphas as expressions over a cross-section of assets (e.g. `rank(close/delay(close,5)-1)`) that must clear thresholds on Sharpe, turnover, and drawdown. It is the "edge" — not beta, not luck.
**Why it matters:** Almost every QR interview question orbits this: does this signal genuinely beat the market, net of costs and factors?

**Q2. What makes a signal predictive vs overfit?**
A signal is predictive if it earns out-of-sample, after costs, consistently across sub-periods and markets. It's overfit if it merely memorizes noise in the training window: too many parameters relative to data, tuned on in-sample Sharpe, non-robust to small changes. Hallmarks of overfit: great IS Sharpe, collapses OOS; family of tuned variants all "work" in-sample; rank-independent of holding period.
**Why it matters:** Distinguishing signal from noise is the single most common screening question.

**Q3. How is Sharpe ratio defined? What's a good value?**
`Sharpe = (mean(returns) − rf)/std(returns)`, annualized by `× sqrt(periods_per_year)`. It's return per unit of risk. "Good" depends on frequency: daily → ~1.0+ is respectable; hourly/high-freq → Sharpe scales with `sqrt(N)` so 5–10+ is common; monthly → 0.5–1.0 is fine. WorldQuant uses a submission-style Sharpe (often daily, thresholds in the 1.0–2.5 range depending on universe). Note it ignores skew, drawdown, and tail risk — a high-Sharpe strategy can still blow up.
**Why it matters:** You must quote the definition *with annualization* and state frequency context.

**Q4. What is turnover and capacity?**
Turnover = how much your positions change per period (mean absolute score/position change). Capacity = the dollar size at which trading your signal stops being profitable — your edge per unit trades shrinks in slippage because you move the market and others front-run you. A high-Sharpe, high-turnover alpha has low capacity. Costs (`turnover × commission`) are what turn a backtest into reality.
**Why it matters:** An alpha is worthless if costs eat the edge. Interviewers probe whether you account for this.

**Q5. Why does correlation between alphas matter?**
Combining highly-correlated alphas diversifies almost nothing — the portfolio Sharpe scales up roughly with average correlation. Factor/diversifying alphas (low pairwise correlation) combine to a much higher combined Sharpe via Sharpe improvement ≈ `sqrt(n)` upper bound when uncorrelated. Correlation also hides double-counting: two signals may be the same factor in disguise.
**Why it matters:** Adding an alpha to a book is a Sharpe-augmentation problem, not a yes/no per-alpha call.

**Q6. Explain data snooping / multiple testing. Why do 100 backtests inflate apparent risk-adjusted return?**
Testing many hypotheses and reporting the best creates selection bias: with enough trials, pure noise produces a "great" result by chance. Formally, the probability that at least one of `m` independent tests yields |t|>threshold ≈ `1−(1−p)^m`. If p=0.05 and m=100, you're virtually guaranteed ≥1 false positive. Sharpe from the *best* of 100 tried alphas is hugely biased upward. Remedies: hold out an untouched validation/test set, multiple-testing corrections (Bonferroni/Holm), or deflated Sharpe (Bailey–López de Prado).
**Why it matters:** This is *the* quant cardinal sin. Expect to be grilled.

**Q7. In-sample vs out-of-sample (IS vs OOS).**
IS = fit/measure on training data (optimistic — the model saw these labels). OOS = performance on data never used for fitting (honest estimate of live behavior). Only OOS Sharpe/predictions can validate a signal. Walk-forward is IS/OOS done correctly for time series (train on past, validate on future, roll forward).

**Q8. Why do mean reversion and momentum coexist across timescales?**
They operate on different horizons and different market microstructure:
- **Short horizon (minutes–days):** mean reversion dominates — liquidity provision, overreaction to news, dealers leaning against order flow; prices oscillate around a calm level.
- **Medium horizon (weeks–months):** momentum dominates — gradual information diffusion, herd behavior, news arriving in chunks.
Different trader populations and information-processing speeds drive each effect; there's no contradiction because each is defined over a separate return window. A short-scaled z-score alpha (revert) and a longer momentum alpha (trend) can both be profitable.

**Q9. What is "displacement" in alpha construction?**
Displacement lags the signal before taking an action — e.g. BRAIN's `delay(close, d)` or `rank(zscore(delay(close,10),20))` (see the repo's `alpha_6`). You mean-revert *where price was* `d` bars ago, not where it is now. This avoids racing the current bar (which is noisy/not yet observed at decision time) and captures the shape of the recent path rather than the instantaneous level. It's a simple way to reduce microstructure noise and turnover.
**Why it matters:** Displacement/z-scores are classic BRAIN levers; interviewers like to hear you understand *why* the construction works, not just the formula.

**Q10. What does "neutralization" mean? (market / sector / industry / sub-industry, AMM)**
Neutralization strips an alpha's exposure to common factors so it captures only idiosyncratic (asset-specific) signal. You regress the alpha's cross-sectional scores on factor exposure vectors and keep the residual: `alpha_neutral = alpha − (b0 + Σbᵢ·factorᵢ)`. WorldQuant's standard factors: **AMM** (American Market Index, broad US-equity beta), **SECTOR**, **INDUSTRY**, and **SUBIND** (Sub-Industry). Purpose: (a) make alphas "additive" in a portfolio (avoid piling onto the same factor), (b) avoid double-paying for a factor you already hold, (c) keep results attributable to your actual signal.
**Why it matters:** WorldQuant requires neutralization before submission; the repo's `neutralize_market` is a single-factor analogue (see map).

**Q11. What is volatility scaling / how do you size with it?**
Volatility targeting: scale positions by `1/vol` (inverse of a rolling realized vol) so each asset/bet contributes roughly equal risk. Without it, a few high-vol names dominate the book and Sharpe is unstable. Forms: risk parity (equal risk contribution across assets), per-alpha `1/vol` weight (see repo `alpha_5`: `rank(mom/vol20)`), or vol-targeting the book's total risk. Downside: vol itself is hard to predict and mean-reverts; scaling to estimated vol can lag regime shifts.

**Q12. How would you make a raw price/return transformation you trust?**
Use log returns (`np.log(close).diff()`) for stationarity and symmetry, drop non-stationary raw levels as features, remove NaN warm-up, and only use trailing (shifted) info so there's no lookahead. Every feature/alpha must be computable at bar `t` using only bars `≤ t`.

---

## 2. Statistics

**Q13. Standard deviation of returns — intuition?**
Std(SD) measures spread of the return distribution; daily vol = SD of daily returns, annualized by `× sqrt(252)` (daily) — assuming i.i.d., variance scales linearly with time so SD scales with `sqrt(time)`. It defines the "risk" scaling behind Sharpe and the ±1σ bands around expected return.

**Q14. Covariance vs correlation.**
`Cov(X,Y) = E[(X−μₓ)(Y−μ_y)]` — signed co-movement, scaled by units. `Corr(X,Y) = Cov/(σₓσ_y)` — normalized to [−1,1], unit-free. Correlation = covariance each variable in its own SD units. Correlation is what drives diversification; a portfolio of imperfectly-correlated assets has lower variance than the sum of parts.

**Q15. Why does overfit risk grow as you add features?**
Each feature adds degrees of freedom the model can exploit to fit noise. With `p` features and `n` samples, a linear model can interpolate when `p ≈ n`; with trees/boosting the capacity grows faster. More parameters → lower IS error, flatter/noisier OOS curve — the bias-variance curve peaks. Regularization (shrinkage, max-depth, subsampling) trades a little bias for far lower variance.

**Q16. p-hacking / multiple comparisons.**
p-hacking = choosing analyses, adding/dropping features, or running many tests until one hits p<0.05. Because a small p-value can arise by chance, inflating trials inflates false positives. Same mechanics as Q6 (data snooping). Fix: pre-register your test, correct for multiplicity, or rely on a single honest holdout.

**Q17. Return distributions: fat tails, kurtosis, skew.**
Returns are *not* normal: they have fat tails (more extreme events than normal — crashes bigger than ±5σ), positive excess **kurtosis** (both tails heavier; peaked center), and **negative skew** (left tail longer — big drops more likely than big gains). Heavy tails mean VaR/Sharpe miss tail risk; a day of −10% happens far more often than a normal model predicts. Skew matters because a high-Sharpe, negatively-skewed strategy is insurance-selling: steady small gains, rare huge losses.

**Q18. Hypothesis testing: null, p-value, and the misinterpretation.**
Null(H0) = a default (e.g., "signal has no predictive power"); p-value = P(observing a result at least this extreme **assuming H0 is true)** — *not* P(H0 is true). p<0.05 is a threshold of surprise under the null, not proof of the alternative, and p-values are vulnerable to sample size: with enough data anything is "significant". Practical use in quant: a t-stat / Sharpe-based test for whether an alpha's edge is distinguishable from noise.

**Q19. Central Limit Theorem and its limits.**
CLT: the *mean* of i.i.d. samples with finite variance converges to Normal as n→∞. Two practical limits: (1) returns have finite-ish but heavy tails, so convergence is slow — you need far more samples than you think; (2) financial data is not i.i.d. (autocorrelation, heteroskedasticity), so the classic CLT's assumptions break. Always sanity-check whether the thing you're averaging is genuinely independent and light-tailed.

**Q20. Stationarity — why returns over prices?**
A series is stationary if its distribution (mean, variance, autocovariance) is constant over time. Prices are *non-stationary* (mean drifts, volatility clusters), so models trained on price levels learn the level and it breaks as soon as price moves to a new regime. Returns are approximately stationary (mean ≈ 0, roughly constant variance), so a model trained on returns remains valid across price levels. This is why the repo's feature engine strips raw `open/high/low/close/volume` levels and models `log_return`, momentum, z-scores.

**Q21. What is a stationary process? / differencing.**
Weak stationarity: constant mean, constant variance, autocovariance depending only on lag (not time). A random walk (e.g., price) is non-stationary but its first *difference* (return) is stationary. **Differencing** (`x_t − x_{t−1}`) removes trends and unit roots; log-differencing handles multiplicative growth. Check with ADF/stationarity tests; transform first, model the transform.

**Q22. Autocorrelation / AR(1).**
Autocorrelation = correlation of a series with its own lagged values: `ρ_k = Corr(x_t, x_{t−k})`. AR(1): `x_t = φ·x_{t−1} + ε_t`. Positive φ = persistent/smooth (momentum-ish), negative φ = oscillating (mean-reverting). Return autocorrelation is usually near zero at daily frequency but can be positive sign-reversal for short windows. Autocorrelation also inflates effective sample size/reduces independence — important for correct standard errors.

**Q23. Bayesian vs frequentist intuition.**
Frequentist: probability = long-run frequency; parameters fixed/unknown; inference via p-values/CIs. Bayesian: probability = degree of belief; parameters are random; start with a **prior**, update with data via **Bayes' rule** to a **posterior**. In finance, Bayesian shrinkage is natural: prior that alphas' true Sharpe ≈ 0, and your observed estimate pulls back toward it (the prior is your prior belief that markets are mostly efficient).

**Q24. What is shrinkage?**
Shrinkage pulls an estimate toward a prior/center to reduce variance at the cost of slight bias — e.g. James–Stein or L2 regularization, or shrinking each alpha's net IC toward zero. In portfolio construction it's critical: sample covariance overfits to noise; shrinking toward a structured prior (e.g., diagonal / single-factor covariance, Ledoit–Wolf) radically improves stability and OOS Sharpe.

**Q25. Dangers of lookahead bias, leaks, survivorship bias.**
- **Lookahead/leak:** using future info at decision time (target trained without shifting, unshifted labels, features computed with future bars) → unrealistically great backtests that die live.
- **Survivorship bias:** backtesting only assets that exist today, dropping delisted/bankrupt ones → overstated universe returns (crash risk hidden).
Guard: shift labels, trailing-only features, include dead names, walk-forward retraining. The repo encodes no-lookahead everywhere (feature shifts, `pos.shift(1)`, label shift in `make_target`).

---

## 3. Linear Algebra

**Q26. Matrix–vector product — geometric meaning.**
`A·x` is a linear combination of A's columns: it maps vector `x` (weights over columns) into the column space. In regression, `X·β` = predicted values as a combination of the feature columns. It's the computational core of most quant signal math (weights × returns).

**Q27. Orthogonality & dot product as projection/correlation.**
Dot product `a·b = Σaᵢbᵢ = |a||b|cosθ`. Zero when orthogonal (θ=90°, cos=0) — no shared direction. Normalized dot product (divide by |a||b|) **is** correlation. Least-squares residualization (with centered/unit data) = removing the component along another vector = forcing orthogonality to it — exactly what `neutralize_market` does.

**Q28. Eigenvalues/eigenvectors intuition.**
For `A·v = λ·v`: `v` is a direction A doesn't rotate (only scales), `λ` is the scale factor. Eigenvectors are the "natural axes" of the transformation; `λ` measures how much A stretches along each. In finance: the covariance matrix's eigendecomposition = directions of maximal/maximal spread = the PCA components = dominant risk factors.

**Q29. PCA — what it does and why in alpha research.**
PCA finds the orthogonal directions of largest variance (the top eigenvectors of the covariance matrix) and projects data onto the top few → reduced-dimension, decorrelated features capturing the dominant factor structure. In alpha research it: (a) identifies/bots away common factor exposure, (b) compresses many correlated predictors (e.g., 20 momentums) into a few independent ones, (c) finds low-dimensional risk structure. Downside: components are statistical, not interpretable; and variance ranking ≠ return predictiveness.

**Q30. Covariance matrix & why symmetric positive semi-definite (PSD).**
`Σ_{ij} = Cov(x_i, x_j)`; symmetric because covariance is symmetric. PSD because for any weight vector `w`, `wᵀΣw = Var(wᵀx) ≥ 0` (variance is never negative). This is why correlation matrices are PSD too. Non-PSD matrices (from noisy/partial estimates) imply negative variance — impossible — hence shrinkage/cleaning is needed.

**Q31. Least squares as projection.**
OLS minimizes `||y − Xβ||²`; the solution `β = (XᵀX)⁻¹Xᵀy` is the orthogonal projection of `y` onto the column space of `X` — the "shadow" of the outcome onto features. The residuals `y − Xβ` are exactly orthogonal to every column of X. This is the direct math behind neutralization (residuals = alpha stripped of factor exposure).

**Q32. Rank deficiency / multicollinearity.**
If features are linearly dependent, `XᵀX` is singular — no unique inverse, unstable β, infinite least-squares solutions. Multicollinearity (highly correlated, e.g., correlated returns) → large variance in coefficient estimates and non-robust risk. Fix: drop/reduce features, PCA, or regularize (ridge adds λI to make `XᵀX+λI` invertible). In quant: double-counting nearly identical signals is rank-deficiency in disguise.

---

## 4. Machine Learning for Finance

**Q33. Why gradient boosting (LightGBM/XGBoost) for tabular finance?**
Gradient boosting builds an additive ensemble of shallow trees, each correcting the previous one's residual error; it handles: mixed data types, missing values, non-linear interactions, and heterogeneous feature scales with no engineering. For noisy, tabular, mostly-dense finance data it reliably beats deep nets (which need huge data) and captures interaction structure between momentum/vol/calendar features. LightGBM's histogram splitting + leaf-wise growth makes it fast on big OHLCV frames. This is exactly the repo's model choice (`qj/models/forecast.py` with LightGBM→XGBoost→sklearn HGB fallback).

**Q34. Bias–variance tradeoff.**
Bias = error from underfitting (model too simple; wrong on average). Variance = error from overfitting (too sensitive to which training sample you draw). Total error ≈ bias² + variance + irreducible noise. Boosting reduces bias (ensemble learns hard patterns) while regularization (depth, subsample, LR) controls variance — you tune to the sweet spot where OOS error is minimal.

**Q35. Why standard K-fold cross-validation is wrong for time series.**
Standard K-fold shuffles/splits the data randomly, so the model trains on **future** data and validates on **past** — lookahead and over-optimism (overlapping trends leak across folds). Correct approach: **walk-forward / expanding (origin) window** — always train on the past, validate on strictly later data, retrain and roll forward (exactly `qj/models/walkforward.py`). Purged/embargoed CV handles horizon overlap. The whole point is to mimic live deployment: a model can only ever have seen data before the moment it predicts.

**Q36. Overfitting and regularization.**
Overfitting = low train error, high test error (model fitted noise). Regularization constrains the hypothesis space: shrinkage (L2/L1), max-depth/leaf-limits, learning rate, subsampling/column-sampling, early stopping. In boosting, a low learning rate + many trees with early stopping usually wins.

**Q37. Tree feature importance — what it means / caveats.**
Gains-based importance = total reduction in loss/split impurity a feature causes across trees → a ranking of which inputs the model leaned on. Caveats: it reflects *fitting* importance, not *causal* or always OOS importance; correlated features split importance between them; it can overstate high-cardinality/smooth features. Use it for feature reduction / debugging, but validate with holdout or permutation importance.

**Q38. Classification (direction) vs regression (return level) — why classification first?**
Predicting sign (up/down) is far more robust to the heavy-tailed, noisy forward return distribution than predicting the exact return level (which has enormous, hard-to-model variance). A direction model maps directly to a long/short decision and its probability (`predict_proba`) can drive confidence-based sizing. The repo makes exactly this choice (`make_target`: 1 if `close[t+1] > close[t]`), and reports logloss + accuracy alongside the subsequent Sharpe in backtesting.

**Q39. Class imbalance.**
An imbalanced up/down target (e.g., 60/40 or worse at long horizons) biases a model toward the majority class, inflating accuracy but adding no edge. Fixes: class weights, resampling, and — most importantly — evaluate with proper metrics (precision/recall, logloss/Brier, and ultimately strategy Sharpe) rather than raw accuracy. If the market is ~balanced at short horizons, imbalance is less pathological but still worth checking.

**Q40. Evaluation metrics: accuracy, logloss/calibration, Sharpe.**
- **Accuracy:** simple but misleading when imbalanced; ignores direction magnitude and confidence.
- **Logloss/Brier:** penalize *confidence in wrong predictions* → a well-calibrated probability (a model can be accurate but badly calibrated). Lower logloss ≈ better probabilities.
- **Sharpe:** the ultimate metric — does the probability, once traded, earn risk-adjusted return net of costs? Walk-forward reports logloss and accuracy (`walk_forward_evaluate`), and `evaluate.py`/`performance_metrics` convert to Sharpe — the pipeline treats Sharpe as the bottom line.

**Q41. Why does ML often underperform simple rank signals? And the role of noise.**
Finance is a low signal-to-noise domain: tiny signal, huge noise, rapidly decaying edge. ML is trained to minimize loss (accuracy/logloss), not to maximize realized Sharpe; it can overfit microstructure, learn unstable patterns, or "spend" the signal on features that don't survive. A simple, robust rank signal (e.g., a z-scored momentum) captured in a handful of parameters often has higher OOS Sharpe per unit capacity. ML shines at *combining* many weak signals if heavily regularized, validated walk-forward, and stripped of noise — but you must resist chasing IS metrics. The honest lesson: complexity pays only when the signal genuinely exists and survives OOS.

**Q42. Design a signal research pipeline (your workflow).**
1. **Data:** clean OHLCV, no survivorship/lookahead, log returns.
2. **Features/alphas:** expressive library (momentum, z-score, displacement, volume) all trailing-only.
3. **Neutralization:** strip market/sector exposure (market/sector/sub-industry analogues).
4. **Model:** gradient boosting (LightGBM) on direction target.
5. **Validation:** walk-forward/expanding-window OOS predictions — never random K-fold.
6. **Backtest:** forecasts → positions → equity curve with turnover+costs; report Sharpe, drawdown.
7. **Drift/ops:** monitor distribution drift (KS) and registry the models.
8. **Multiple-testing discipline:** hold out a final untouched sample; correct for the number of alphas tried.
This mirrors the repo's actual layers (features → models → walkforward → backtest → mlops).

---

## 5. Probability & Estimation

**Q43. Linearity of expectation; variance of sums.**
`E(aX+bY) = aE(X)+bE(Y)` always (no independence needed). `Var(aX+bY) = a²Var(X)+b²Var(Y)+2ab·Cov(X,Y)`. Independence → covariance term vanishes. This is why diversification works: `Var` of a portfolio is less than the weighted sum of variances when assets are imperfectly correlated.

**Q44. Law of Large Numbers.**
As n→∞, the sample mean converges (in probability) to the population mean. Practical caveat: convergence for heavy-tailed / i.i.d.-violating financial data is slow and delicate — "with enough samples the estimate is reliable" only under strong assumptions that returns violate.

**Q45. When are mean/variance unreliable?**
When the distribution has heavy tails, skew, or the mean is dominated by rare events — a handful of extreme returns can move the sample mean/var hugely. Also under non-stationarity (regime changes make historical moments stale). In such cases use robust estimates: median, trimmed moments, realized vol, or Bayesian shrinkage.

**Q46. Conditional probability / Bayes' rule in modeling.**
`P(A|B) = P(B|A)P(A)/P(B)` — update belief in A given evidence B. In modeling: `P(up | features)` = posterior over direction, the prior `P(up)` (≈ 0.5 baseline), likelihood from historical feature-return links. Gaussian/GMM models, hidden Markov regimes, and Bayesian shrinkage are all applications. Your model literally computes `P(up|x)` via `predict_proba`.

**Q47. Random walk and why prices roughly follow one.**
Efficient-price random walk: next price change is unpredictable (`p_{t+1} = p_t + ε_t`, ε independent). With efficient markets revealing all info, price-level returns have ~no forecastable component at many horizons; 1-step predictability is tiny. If price were predictable at a tradable scale, it would be arbitraged away. This sets the (low) floor on how much edge even a good model should claim.

**Q48. Expected value and ergodicity in trading.**
- **Expected value:** `EV = Σ pᵢ·payoffᵢ` — the per-bet average over many independent bets.
- **Ergodicity:** a process is ergodic if time-average = ensemble-average. **Trading is often non-ergodic** because of compounding and absorbing barriers (liquidation/ruin): each bet is proportional to current wealth, so a positive-EV but high-variance strategy can have a negative *growth* rate (Kelly/log-utility framing). Key insight: a strategy can have positive expected P&L yet be ruinous for a single trader if variance is unmanaged — hence drawdown/vol controls matter as much as mean edge.

---

## 6. Numerical / Coding & Python

**Q49. Clean averaging / EWMA.**
Use the vectorized `Series.ewm` — never a Python `for` loop (slow, no NaN semantics):
```python
ewma = s.ewm(span=span, adjust=False, min_periods=span).mean()
```
`adjust=False` = recursive form `ewma_t = (1−α)ewma_{t−1} + α·x_t`; span ↔ `α = 2/(span+1)`.

**Q50. Vectorization vs loops.**
Vectorize everything with numpy/pandas operations (rolling, diff, cumprod) — they run in compiled C and handle alignment/NaN. Python loops over rows are ~100–1000× slower and invite bugs. Rule: if you're iterating over a DataFrame's rows, you're almost always doing it wrong; express it as array ops.

**Q51. Handling NaN / missing bars.**
`rolling(...).min_periods`, `.dropna()`, `.fillna()`, and `.shift()` control warm-up/alignment gaps. For missing *bars* (bad ticks), decide: forward-fill only if valid, else resample to a regular grid. Never silently mask NaN as signal — NaN in features must propagate to "no position" (the repo's `make_dataset(dropna=True)` and NaN-preserving neutralize do this).

**Q52. Efficient rolling operations.**
`pd.Series.rolling(window).mean/std/min/max/rank` are O(n) windowed ops (rolling = sliding window). Combine with `.shift()` for lookahead safety. For heavy pipelines, downsample or use `min_periods` to avoid recomputing on NaN edges.

**Q53–58. Concise reusable snippets.**
```python
import numpy as np, pandas as pd

# EWMA (span S)
def ewma(s, span=20): return s.ewm(span=span, adjust=False, min_periods=span).mean()

# Rolling z-score
def zscore(s, w=20):
    mu, sd = s.rolling(w).mean(), s.rolling(w).std()
    return (s - mu) / sd

# Sharpe from returns (annualize by periods/yr)
def sharpe(r, ppy=252):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(ppy)

# Residualize y on x (OLS) -> orthogonal residual
def residualize(y, x):
    A = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(A, y, rcond=None)[0]   # [intercept, slope]
    return y - A @ b                            # same as neutralize_market

# Cross-sectional rank (fractional 0..1, NaN-preserving)
def cs_rank(s): return s.rank(pct=True, method="average")

# Tracking error / correlation
def tracking_error(a, b): return (a - b).std()
```
These mirror real repo functions: `rolling_zscore` and `zscore`, `neutralize_market`'s `lstsq`, `cross_sectional_rank`, `performance_metrics`' Sharpe (all in `qj/features/engine.py`, `qj/brain/neutralize.py`, `qj/brain/evaluate.py`).

---

## 7. Brainteasers / Mental Math

**Q59. Estimate the total equity traded on US stock exchanges in one year.**
Rough US daily equity volume ≈ $400–500B (call it $450B). `$450B × 252 ≈ $113–114T/yr`. State assumptions and round. **Why:** tests multi-step Fermi reasoning and fluency with market scale.

**Q60. With 1,000 backtests of a random signal, what fraction will falsely look significant at α=0.05?**
Expect ~ `0.05 × 1000 = 50` false positives purely by chance; and the *best* t-stat will be far beyond the 95% threshold. This nails the multiple-testing question in numbers.

**Q61. Estimate the daily volume of BTC in USD.**
Daily BTC spot+derivatives volume ~ tens of $B; call spot ≈ $20–40B. Order-of-magnitude: "~$20–50B daily." **Why:** sanity for a crypto-alpha capacity discussion.

**Q62. What is the ±1 standard deviation and ±2σ range for a normal variable?**
±1σ ≈ 68% of mass, ±2σ ≈ 95%. Practical: if daily vol is 2%, a −4% day is about a 2σ event (~rare-ish); a −6% day is 3σ but happens more than normal predicts because of fat tails.

**Q63. A fair coin flipped 100 times: expected #heads and its 1σ range?**
Expected = 50. Var = `n·p·q = 100·0.5·0.5 = 25`, so σ = 5. Range ≈ 45–55 within 1σ. **Why:** instant binomial stats — the base for many estimation questions.

**Q64. Estimate the average height of adult humans.**
~1.7m (5'7"). Useful anchor: convert NaNs, population fractions, etc. Fermi-style reasoning start.

**Q65. If a strategy must beat a daily Sharpe of 1.0, about what t-stat/sample do you need to be confident?**
Daily Sharpe 1.0 annualized ≈ `1.0/√252 ≈ 0.063` per day. To detect at 3σ you need `n ≈ (z/Sharpe_daily)² = (3/0.063)² ≈ 2250 days ≈ 9yr`. Shows how much data honest edge detection needs — a core reason OOS rigor matters.

**Q66. Rough odds of a 1-in-100 tail event occurring in 100 independent days?**
P(at least once) = `1 − (1−0.01)^100 ≈ 1 − 0.366 ≈ 63%`. Highlights that even rare tail events are *likely* over a year — motivates the fat-tail/ruin concern.

**Q67. Variance of a portfolio of N identical, uncorrelated assets each with vol σ.**
Portfolio var = `σ²/N`, so portfolio vol = `σ/√N` — diversification halves risk per doubling of N count-ish. If correlations are ρ, vol ≈ `σ√((1/N)+(1−1/N)ρ)`, showing correlation is what kills diversification.

**Q68. Estimate Apple's daily revenue to check a number you're given.**
Apple ~ $95B/quarter = ~$380B/yr ≈ ~$1B/day. Good for cross-checking a claimed figure and demonstrating order-of-magnitude reasoning.

---

## Quick-diagnostic answers (the ones to have memorized cold)
- Sharpe annualization: `×sqrt(ppy)`.
- Cov → Corr: divide by both SDs.
- p-value definition: under-the-null probability, not P(null).
- Random-walk price: `p_{t+1}=p_t+ε`, why returns (not prices) are modeled.
- OLS projection: `β=(XᵀX)⁻¹Xᵀy`; residual ⊥ columns of X.

---

## Suggested further reading (all general, universally-accepted)
- AQR / Grinold & Kahn, *Active Portfolio Management* (alpha/factor language).
- Bailey, Borwein, López de Prado & Zhu — "The Probability of Backtest Overfitting".
- López de Prado — *Advances in Financial Machine Learning* (walk-forward, purged CV, deflated Sharpe).
- WorldQuant BRAIN docs (neutralization factors: AMM, SECTOR, INDUSTRY, SUBIND; submission bar).
- Hull / standard references for random walks, CLT, GBM.
