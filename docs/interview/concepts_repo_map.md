# Quant Concepts → Repo Map (Sagar's quant repo at `/Users/sagar.sohrab/Quant`)

Maps core QR interview concepts to concrete, real artifacts. Paths relative to `/Users/sagar.sohrab/Quant`; function names verified against the code. Where a capability exists only as a single-instrument *reduction* of a cross-sectional BRAIN concept, that limitation is stated honestly — do not overclaim.

---

## Concept → artifact table

| Interview concept | Repo artifact (real path / function) | What it does there |
|---|---|---|
| Sharpe ratio, annualization | `qj/backtest/engine.py::performance_metrics` | Computes Sharpe (`mean/std × sqrt(periods_per_year)`), vol-an, max DD, Calmar from a backtest frame. `qj/brain/evaluate.py::sharpe_style` has the BRAIN-style version (default ppy=24 for hourly bars). |
| Walk-forward / no-lookahead validation | `qj/models/walkforward.py::walk_forward_evaluate`, `walk_forward_predictions` | Rolling expanding-train window, retrain per split, report *strictly out-of-sample* acc + logloss; `predictions` variant feeds a backtest honestly. This is the anti-K-fold pattern. |
| No-lookahead mechanics | `qj/backtest/engine.py::backtest_positions` (uses `pos.shift(1)`); `qj/models/forecast.py::make_target` (label at t uses `close[t+horizon]`, shifted); `qj/features/engine.py` (all trailing shifts) | Trailing-only features, positions applied next bar, labels shifted before fit — the three places leakage is prevented. |
| Momentum / features | `qj/features/engine.py::momentum`, `add_features` (`mom_{3,5,10,20,50}`, `zscore_`, `vol_`, `min/max`) | Rolling momentum, z-scores, vol, calendar, vol_ratio — the model inputs. |
| Displacement z-score | `qj/brain/alphas.py::alpha_6` (`zscore(delay(close,10),20)`), `alpha_7` | Displaced mean-reversion — z-score of the *delayed* close; classic BRAIN expression. |
| Volatility scaling / risk parity | `qj/brain/alphas.py::alpha_5` (`rank(mom/vol20)`) | Vol-scaled momentum (calm assets weighted up). |
| Cross-sectional rank | `qj/brain/neutralize.py::cross_sectional_rank`; `qj/brain/alphas.py::rank`, `ts_rank` | Fractional rank to [0,1], NaN-preserving. **Note:** single-instrument repo → rank applied over time, not a true per-bar cross-section. |
| Residualization / orthogonalization | `qj/brain/neutralize.py::neutralize_market` (uses `np.linalg.lstsq`) | OLS residual of alpha on a market factor — `alpha − (b0 + b1·market)`; directly the least-squares-projection concept. |
| Neutralization (AMM / SECTOR / SUBIND) | `qj/brain/neutralize.py` (module docstring) + `neutralize_series`, `demean` | **Partial only:** the *concept* of AMM/Sub-Industry/Sector neutralization is documented, but only `neutralize_market` (single-factor, single-series analogue) is implemented. No real multi-factor or universe. State exactly this. |
| Turnover / transaction costs | `qj/backtest/engine.py::backtest_positions` (`turnover = pos.diff().abs()` → cost); `qj/brain/evaluate.py::turnover` | Costs = turnover × commission; per_trade vs per_bar cost models. |
| In-sample vs out-of-sample Sharpe split | `qj/brain/evaluate.py::simulate_alpha` (returns `is_sharpe` / `oos_sharpe`) | Two-half time split IS/OOS Sharpe for an alpha. **Note:** a proxy, not a submission guarantee. |
| IS/OOS via walk-forward | `qj/models/walkforward.py` | The rigorous IS/OOS method (vs the simple split above). |
| Drift monitoring / KS statistic | `qj/mlops/drift.py::ks_statistic`, `DriftMonitor` | Two-sample Kolmogorov–Smirnov + per-feature mean/std distance; fires a drift alarm when live distribution moves from training. |
| Model registry / versioning | `qj/mlops/registry.py::ModelRegistry` | JSON-backed `register/latest/history` — audit which model is deployed and its metrics. |
| Gradient boosting (LightGBM/XGBoost) | `qj/models/forecast.py::make_clf` | LightGBM → XGBoost → sklearn HGB fallback; params (depth, subsample, colsample, LR). |
| Classification vs regression (direction) | `qj/models/forecast.py:make_target`, module docstring | Predicts up/down (label `close[t+1] > close[t]`), reports `predict_proba` for sizing — the robustness argument for direction. |
| Feature contract | `qj/models/columns.py::default_feature_cols`; `qj/features/engine.py::RAW_LEVEL_COLS` | Standard feature set; raw non-stationary price levels excluded as features. |
| Model persistence & final training | `qj/models/persist.py::train_on_frame/save_model/load_model` | Train on all history after walk-forward validation; pickle model + feature contract. |
| Transformers / LLM self-attention | `qj/llm/minigpt.py::MiniGPT`, `MultiHeadAttention`, `TransformerBlock`, `PositionalEncoding` | Causal scaled dot-product attention, GPT-style block, learnable positions — from scratch, no HF. |
| EWMA / rolling z-score / residualize / cs-rank / Sharpe snippets | `qj/features/engine.py:rolling_zscore`; `qj/brain/neutralize.py:neutralize_market, cross_sectional_rank`; `qj/backtest/engine.py:performance_metrics` | The interview "snippet" answers all exist as real functions here. |

---

## Concepts NOT in the repo (say so honestly)
- No true multi-asset cross-section / universe — everything is single-symbol (BRAIN `rank`/`neutralize` are implemented as *time-series* reductions).
- No portfolio construction (covariance-based weight optimization, factor exposure matrix) beyond 1-asset sizing.
- No purged/embargoed CV; walk-forward only.
- No Kelly growth / ergodicity computation; no explicit asset-class factor models.
- No production-grade trade execution / slippage model; `slippage` param exists in `backtest_positions` but default 0.

---

## 5 suggested "how I would explain X" written pieces Sagar could produce from the repo
1. **"How I validate a signal without fooling myself"** — walk `walk_forward_predictions` ([IS/OOS, no lookahead]) with the shift-1 position rule and label-shift guard so the backtest can't cheat; contrast with naive K-fold.
2. **"What neutralization actually does, and its limits"** — explain AMM/SECTOR/SUBIND *concept* from `neutralize.py`'s docstring and `neutralize_market`'s `lstsq` residual (projection/orthogonality), and be explicit that the repo implements single-factor/single-series only.
3. **"From features to forecast to Sharpe"** — trace `add_features` → `make_target` (direction) → `make_clf` (LightGBM) → `walk_forward` → `performance_metrics`, explaining why classification+probability beats regression on noisy returns.
4. **"What a low-signal domain does to ML"** — use `forecast.py`'s classification choice, walk-forward discipline, and `evaluate.py`'s logloss-vs-Sharpe framing to argue why simple robust signals often beat complex ones.
5. **"Monitoring a live model"** — explain distribution drift with `ks_statistic`/`DriftMonitor` and version discipline with `ModelRegistry.register/latest`, i.e. the OPS story for why a model decays and how you catch it.
