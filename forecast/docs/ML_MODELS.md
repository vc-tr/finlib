# Learned Signals (ML)

Three ML strategies sit in the catalog alongside the rule-based ones and run
through the **same** backtester, walk-forward harness, execution model, and
tearsheets:

| Name | Model | Dependency |
|------|-------|------------|
| `ml_logistic` | L2 logistic regression | core (`scikit-learn`) |
| `ml_gradient_boost` | Gradient-boosted trees (depth 2) | core (`scikit-learn`) |
| `ml_lstm` | 1-layer LSTM classifier | `requirements-ml.txt` (PyTorch) |

The point of this module is **not** a high Sharpe on one ticker — a single daily
price series carries little learnable signal. The point is to show ML done the
way a quant desk requires it: **no lookahead, anywhere, ever.**

## The core guarantee: leak-free by construction

Lookahead is the single most common way ML backtests lie. We remove it
structurally rather than by convention. Every learned signal is produced by
`src/ml/walk_forward_signal`, which:

1. Builds a **causal** feature matrix — feature row `t` uses prices `≤ t` only
   (`src/ml/features.py`). No centered windows, no negative shifts.
2. Walks forward in blocks. Before predicting a block that starts at bar `p`, it
   refits the model on rows `i` with **`i + horizon ≤ p`** — i.e. only rows
   whose forward-looking label is already realized *before* the block begins.
   That inequality is the embargo that stops the `horizon`-step label from
   overlapping the prediction window.
3. Standardizes features on the **training rows only** (scaler statistics never
   see the test window).
4. Emits a position in `{-1, 0, +1}`; the backtester then applies its own
   `shift(1)`, exactly like every rule-based strategy.

```
 train (labels realized)          embargo         predict block
|-----------------------------|  (horizon bars)  |==============|
0                          p-horizon            p           p+retrain_every
```

### This is tested, not asserted

`tests/test_ml_walkforward.py::test_walkforward_is_anti_lookahead` perturbs all
prices from bar `k` onward by 40% and checks that **every signal before `k` is
bit-for-bit unchanged**. If any future information leaked into an earlier
prediction, that test fails. The same check runs for the LSTM
(`tests/test_ml_lstm.py::test_lstm_is_anti_lookahead`).

We also verify the pipeline actually *learns*: it recovers a planted edge on an
AR(1)-momentum series (directional accuracy > 0.52 out-of-sample) and the
estimators fit a linearly separable target (> 90% accuracy).

## Features (`src/ml/features.py`)

Ten causal technical features: multi-horizon returns, 12-1 momentum, rolling
volatility (10/20), 20-day z-score, RSI(14), Donchian channel position, and a
10/50 moving-average gap. All backward-looking.

## Models

- **Logistic / Gradient Boosting** (`src/ml/sklearn_model.py`) — probabilistic
  classifiers predicting `P(next-bar up)`. Logistic is the linear baseline;
  gradient boosting captures non-linear feature interactions.
- **LSTM** (`src/ml/torch_lstm.py`) — a genuinely-trained sequence model: causal
  sliding windows, Adam, **gradient clipping**, a **ReduceLROnPlateau** learning
  -rate schedule, and **early stopping** on a chronological validation tail.
  CPU-deterministic given a seed; uses Apple MPS / CUDA automatically when
  available via `device=`. torch is imported lazily, so the rest of the platform
  installs and runs without it.

A probability is converted to a position with an optional dead-band around 0.5
(`WalkForwardConfig.band`): low-confidence predictions go flat instead of
trading noise.

## Run it

```bash
# scikit-learn (core deps only)
python scripts/run_ml_demo.py --strategy ml_gradient_boost --symbol SPY --period 8y

# PyTorch LSTM
pip install -r requirements-ml.txt
python scripts/run_ml_demo.py --strategy ml_lstm --symbol SPY --period 10y

# offline, deterministic (the CI smoke test)
python scripts/run_ml_demo.py --strategy ml_logistic --synthetic
```

## Honest expectations

On a single daily series, after realistic costs, these models land somewhere
between break-even and a modest positive Sharpe, and they are regime-dependent —
exactly what you should expect, and what the `expected_result` field on each
strategy says. The value here is the **methodology**: a learned signal you can
trust because the harness makes leakage structurally impossible.
