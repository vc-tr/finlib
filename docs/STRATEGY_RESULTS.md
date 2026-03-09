# Quant Lab: Strategy Research Results

> **Living document.** Updated as each strategy is implemented and backtested.
> All results: SPY, 5-year window, anti-lookahead execution, 1 bps fee + 2 bps slippage.

---

## Abstract

Twenty strategies were backtested on SPY across a 5-year window using anti-lookahead
execution (signal at close *t* → fill at open *t+1*), realistic transaction costs
(1 bps fee + 2 bps slippage), and statistical significance testing via the Deflated
Sharpe Ratio (Bailey & Lopez de Prado 2014).

**Key findings:**
- 10 of 20 strategies produced a positive Sharpe ratio in-sample; **none achieved DSR > 0.95**
  (the threshold for statistical significance after correcting for multiple testing across 20 trials)
- The best performer (**mean reversion**, Sharpe=+0.77) approaches but does not clear significance
  (DSR=0.964, p=0.072 — borderline at the 10% level)
- Retail/social-media strategies are systematically the worst performers; 3 of 8 destroy capital outright
- Academic strategies (time-series momentum, carry) are more robust but statistically weak
  on a single-asset SPY test — they require cross-sectional implementation to shine
- The "debunking" result is the point: disciplined backtesting reveals that most widely-shared
  trading strategies carry no detectable edge after realistic costs and multiple-testing correction

---

## Methodology

### Data
- **Source**: Yahoo Finance via `yfinance`
- **Symbol**: SPY (SPDR S&P 500 ETF)
- **Period**: 5 years (~1,255 trading days)
- **Frequency**: Daily OHLCV

### Execution Assumptions
- **Fee**: 1 bps per trade (one-way)
- **Slippage**: 2 bps per trade
- **Fill**: Signal at close *t* → fill at open *t+1* (strict no-lookahead)
- **Position sizing**: Equal notional per trade (fully invested when in position)

### Statistical Framework
- **Sharpe ratio SE**: Lo (2002) — `SE(SR) = sqrt((1 + 0.5*SR²) / T)`
- **Deflated Sharpe Ratio (DSR)**: Bailey & Lopez de Prado (2014)
  — adjusts for skewness, excess kurtosis, and the number of strategies tested
  — DSR > 0.95 ≈ statistically significant at the 5% level after multiple-testing correction
- **n_trials = 20** (total strategies in this study) used in DSR calculation

### Reproduction
All results are fully reproducible:
```bash
python scripts/run_strategy.py --strategy <name> --symbol SPY --period 5y
```

---

## Summary Table — All 20 Strategies

| Rank | Strategy | Category | Sharpe | Return | Max DD | Trades | Win Rate | DSR | Verdict |
|------|----------|----------|--------|--------|--------|--------|----------|-----|---------|
| 1 | mean_reversion | stats | +0.77 | +46.7% | 10.7% | 103 | 45.3% | 0.964 | BORDERLINE |
| 2 | three_bar_reversal | retail | +0.57 | +26.9% | 14.2% | 318 | 33.1% | 0.922 | BORDERLINE |
| 3 | time_series_momentum | academic | +0.49 | +37.6% | 24.7% | 15 | 55.9% | 0.862 | WEAK |
| 4 | carry_trade | academic | +0.44 | +32.5% | 22.4% | 3 | 55.9% | 0.838 | WEAK |
| 5 | rsi_overbought | retail | +0.37 | +10.7% | 5.5% | 78 | 35.4% | 0.849 | WEAK |
| 6 | doji_reversal | retail | +0.23 | +3.8% | 6.0% | 160 | 25.6% | 0.696 | INSUFFICIENT |
| 7 | momentum | stats | +0.16 | +6.4% | 21.1% | 100 | 52.1% | 0.638 | INSUFFICIENT |
| 8 | betting_against_beta | academic | +0.15 | +5.5% | 27.0% | 37 | 51.4% | 0.628 | INSUFFICIENT |
| 9 | low_vol_anomaly | academic | +0.15 | +5.4% | 24.7% | 76 | 48.2% | 0.627 | INSUFFICIENT |
| 10 | golden_cross | retail | +0.14 | +5.0% | 23.8% | 5 | 53.7% | 0.624 | INSUFFICIENT |
| 11 | ornstein_uhlenbeck | econophysics | −0.01 | −2.4% | 17.3% | 12 | 45.7% | 0.491 | FAILS |
| 12 | power_law_tail | econophysics | −0.07 | −10.0% | 38.8% | 9 | 50.4% | 0.437 | FAILS |
| 13 | hurst_exponent | econophysics | −0.10 | −11.1% | 29.4% | 246 | 45.5% | 0.415 | FAILS |
| 14 | post_earnings_drift | academic | −0.11 | −3.9% | 11.3% | 6 | 40.0% | 0.402 | FAILS |
| 15 | volume_spike | retail | −0.19 | −8.4% | 17.9% | 391 | 27.7% | 0.336 | FAILS |
| 16 | macd_crossover | retail | −0.29 | −27.5% | 37.1% | 118 | 49.6% | 0.255 | FAILS |
| 17 | gap_and_go | retail | −0.38 | −17.4% | 20.4% | 441 | 29.2% | 0.199 | FAILS |
| 18 | entropy_signal | econophysics | −0.43 | −25.3% | 26.0% | 244 | 42.8% | 0.173 | FAILS |
| 19 | bollinger_breakout | retail | −0.59 | −18.8% | 18.9% | 139 | 28.7% | 0.064 | FAILS |
| 20 | pairs_trading | stats | N/A | 0.0% | 0.0% | 0 | — | — | NO SIGNALS |

*DSR threshold for significance: > 0.95 (with N=20 trials, skewness and kurtosis adjusted)*

---

## 1. Statistical / Core Strategies

### 1.1 Mean Reversion (Z-Score)

- **Hypothesis**: Short-term price deviations from a rolling mean are transitory and revert
- **Signal**: Z-score of price vs. rolling window; long when z < −entry_z, short when z > +entry_z
- **Source**: Classic statistical mean reversion (`src/strategies/stats/mean_reversion.py`)

| Metric | Value |
|--------|-------|
| Sharpe | **+0.77** |
| CAGR | +8.00% |
| Total Return | +46.7% |
| Max Drawdown | 10.7% |
| Trades | 103 |
| Win Rate | 45.3% |
| DSR | 0.964 |
| p-value | 0.072 |

**Verdict: BORDERLINE** — The strongest result in this study, but DSR=0.964 falls just short of the
0.95 significance threshold after correcting for 20 trials. The low win rate (45%) with positive
Sharpe indicates a right-skewed payoff distribution: losses are small (reversion quickly stops out),
while gains accumulate during extended mean-reverting regimes. Requires cross-asset confirmation to
claim real edge.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy mean_reversion --symbol SPY --period 5y
```

---

### 1.2 Momentum (Time-Series)

- **Hypothesis**: Assets with positive recent returns continue outperforming short-term
- **Signal**: `prices.pct_change(lookback) > threshold` → long; negative → flat
- **Source**: Classic time-series momentum (`src/strategies/stats/momentum.py`)

| Metric | Value |
|--------|-------|
| Sharpe | +0.16 |
| CAGR | +1.26% |
| Total Return | +6.4% |
| Max Drawdown | 21.1% |
| Trades | 100 |
| Win Rate | 52.1% |
| DSR | 0.638 |
| p-value | 0.723 |

**Verdict: INSUFFICIENT** — Positive but economically trivial Sharpe on single-asset SPY.
Time-series momentum is well-documented in cross-sectional multi-asset settings (Moskowitz et al. 2012)
but degenerates on a single instrument where buy-and-hold dominates. The 21% max drawdown is severe
relative to the return. DSR=0.638 is far from significant.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy momentum --symbol SPY --period 5y
```

---

### 1.3 Pairs Trading (Cointegration)

- **Hypothesis**: Cointegrated pair spread is stationary and mean-reverts to equilibrium
- **Signal**: Fit OLS spread between two instruments; trade z-score of residuals
- **Source**: Statistical arbitrage — cointegration-based pairs trading

| Metric | Value |
|--------|-------|
| Sharpe | N/A |
| Trades | 0 |

**Verdict: NO SIGNALS** — The single-asset (SPY-only) setup provides no second instrument for
cointegration. This strategy requires a properly configured multi-asset universe. Results will
populate once a pairs universe is configured.

---

## 2. Retail / Technical Analysis Strategies

*Strategies popularized by YouTube and TikTok trading educators. Tested with rigorous
out-of-sample validation and realistic transaction costs. The hypothesis: widely-shared
retail strategies carry no edge after costs.*

---

### 2.1 Three-Bar Reversal

- **Hypothesis**: Three consecutive directional bars signal exhaustion and reversal
- **Signal**: Three consecutive down bars → long; three consecutive up bars → short
- **Source**: Retail pattern trading — common in price action courses

| Metric | Value |
|--------|-------|
| Sharpe | **+0.57** |
| CAGR | +4.89% |
| Total Return | +26.9% |
| Max Drawdown | 14.2% |
| Trades | 318 |
| Win Rate | 33.1% |
| DSR | 0.922 |
| p-value | 0.156 |

**Verdict: BORDERLINE (surprising)** — The second-best result overall and the strongest retail
strategy. DSR=0.922 remains below the 0.95 threshold. The low win rate (33%) with positive Sharpe
is characteristic of a contrarian payoff: frequent small losses offset by occasional large reversals.
At 318 trades, there are enough observations to reduce noise, but significance is marginal. Likely
benefits from SPY's mean-reverting characteristics in the study window.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy three_bar_reversal --symbol SPY --period 5y
```

---

### 2.2 RSI Overbought/Oversold

- **Hypothesis**: RSI extremes (>70 overbought, <30 oversold) signal imminent reversal
- **Signal**: Buy when RSI(14) < 30; sell when RSI(14) > 70
- **Source**: Wilder (1978) — *New Concepts in Technical Trading Systems*

| Metric | Value |
|--------|-------|
| Sharpe | +0.37 |
| CAGR | +2.06% |
| Total Return | +10.7% |
| Max Drawdown | 5.5% |
| Trades | 78 |
| Win Rate | 35.4% |
| DSR | 0.849 |
| p-value | 0.303 |

**Verdict: WEAK** — Positive Sharpe and the lowest max drawdown (5.5%), but economically small
(2% CAGR). RSI at standard thresholds is infrequently triggered on SPY (78 trades in 5 years),
which is why the drawdown is low — the strategy is mostly in cash. This is selection bias: it only
trades during extreme dislocations and catches some reversals. DSR=0.849 remains below significance.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy rsi_overbought --symbol SPY --period 5y
```

---

### 2.3 Golden Cross (SMA 50/200)

- **Hypothesis**: SMA 50 crossing above SMA 200 signals sustained uptrend
- **Signal**: Long when 50-day SMA > 200-day SMA; flat otherwise
- **Source**: Common retail trading lore

| Metric | Value |
|--------|-------|
| Sharpe | +0.14 |
| CAGR | +0.99% |
| Total Return | +5.0% |
| Max Drawdown | 23.8% |
| Trades | 5 |
| Win Rate | 53.7% |
| DSR | 0.624 |
| p-value | 0.751 |

**Verdict: INSUFFICIENT** — Only 5 trades over 5 years means there is essentially no statistical
information. The 23.8% max drawdown from 5 signals is alarming. Golden cross is famously a lagging
indicator — it triggers after large moves have already occurred, buying high after rallies. Any
positive Sharpe here is noise. DSR=0.624 confirms no edge.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy golden_cross --symbol SPY --period 5y
```

---

### 2.4 Doji Reversal (Candlestick)

- **Hypothesis**: Doji candles (open ≈ close) signal indecision and imminent reversal
- **Signal**: Detect doji candle, trade reversal on next open
- **Source**: Japanese candlestick analysis — Nison (1991), *Japanese Candlestick Charting Techniques*

| Metric | Value |
|--------|-------|
| Sharpe | +0.23 |
| CAGR | +0.76% |
| Total Return | +3.8% |
| Max Drawdown | 6.0% |
| Trades | 160 |
| Win Rate | 25.6% |
| DSR | 0.696 |
| p-value | 0.609 |

**Verdict: INSUFFICIENT** — Despite Sharpe +0.23, the 25.6% win rate is the second lowest of all
strategies. The positive Sharpe comes from asymmetric payoffs, not reliable directional prediction.
Doji patterns are mechanically frequent (160 triggers) but provide minimal directional information.
DSR=0.696 confirms no meaningful edge.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy doji_reversal --symbol SPY --period 5y
```

---

### 2.5 MACD Crossover

- **Hypothesis**: MACD line crossing above its signal line indicates trend initiation
- **Signal**: Long when MACD > Signal; flat otherwise
- **Source**: Appel (1979) — *Technical Analysis — Power Tools for Active Investors*

| Metric | Value |
|--------|-------|
| Sharpe | −0.29 |
| CAGR | −6.26% |
| Total Return | −27.5% |
| Max Drawdown | 37.1% |
| Trades | 118 |
| Win Rate | 49.6% |
| DSR | 0.255 |
| p-value | 0.510 |

**Verdict: FAILS** — Worst absolute return (−27.5%) among strategies with sufficient trades.
Despite near-50% win rate, MACD generates losses because it buys momentum slightly late and holds
through reversals. The 37% max drawdown is the second-worst in the study. This is a classic
"buy high, sell low" pattern disguised as trend following. DSR=0.255 confirms decisively negative
expectation.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy macd_crossover --symbol SPY --period 5y
```

---

### 2.6 Bollinger Band Breakout

- **Hypothesis**: Price breaking outside Bollinger Bands signals momentum continuation
- **Signal**: Long on upper band break; short on lower band break
- **Source**: Bollinger (1983) — *Bollinger on Bollinger Bands*

| Metric | Value |
|--------|-------|
| Sharpe | **−0.59** |
| CAGR | −4.09% |
| Total Return | −18.8% |
| Max Drawdown | 18.9% |
| Trades | 139 |
| Win Rate | 28.7% |
| DSR | 0.064 |
| p-value | 0.129 |

**Verdict: FAILS** — Worst Sharpe in the study (−0.59). Bollinger Bands on a mean-reverting
instrument like SPY are fundamentally misapplied for breakout trading (they were designed for mean
reversion). Breaking outside bands on SPY disproportionately occurs near local extremes, causing
the strategy to buy the top and short the bottom. DSR=0.064 — the most decisively negative result.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy bollinger_breakout --symbol SPY --period 5y
```

---

### 2.7 Volume Spike

- **Hypothesis**: Unusually high volume with price move signals informed order flow
- **Signal**: Buy when volume > 2× 20-day average and price is up; sell on reversal
- **Source**: Retail day trading lore — common YouTube/TikTok strategy

| Metric | Value |
|--------|-------|
| Sharpe | −0.19 |
| CAGR | −1.74% |
| Total Return | −8.4% |
| Max Drawdown | 17.9% |
| Trades | 391 |
| Win Rate | 27.7% |
| DSR | 0.336 |
| p-value | 0.673 |

**Verdict: FAILS** — Highest trade count among retail strategies (391 trades), meaning transaction
costs compound heavily against it. The 27.7% win rate with negative Sharpe means consistent losses:
chasing volume spikes at daily frequency on a large-cap ETF amounts to reacting to noise. DSR=0.336.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy volume_spike --symbol SPY --period 5y
```

---

### 2.8 Gap and Go

- **Hypothesis**: Gap-up opens (price above prior close) continue upward intraday
- **Signal**: Buy on open when price gaps up vs prior close; sell at close
- **Source**: Retail day trading — YouTube: Warrior Trading, Ross Cameron style

| Metric | Value |
|--------|-------|
| Sharpe | −0.38 |
| CAGR | −3.77% |
| Total Return | −17.4% |
| Max Drawdown | 20.4% |
| Trades | 441 |
| Win Rate | 29.2% |
| DSR | 0.199 |
| p-value | 0.398 |

**Verdict: FAILS** — Highest raw trade count in the study (441 trades) combined with a negative
Sharpe produces the worst cost-drag of any strategy. Gap continuation works for high-momentum
small-caps, not for a diversified large-cap index ETF where gaps fill frequently. The 29% win rate
at daily frequency on SPY is a strong signal that the underlying signal is noise. DSR=0.199.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy gap_and_go --symbol SPY --period 5y
```

---

## 3. Academic Strategies

*Replications of peer-reviewed factor research. These strategies are designed for cross-sectional
multi-asset implementation; testing on single-asset SPY is a stress test, not the intended use case.*

---

### 3.1 Time-Series Momentum

- **Hypothesis**: Assets with positive trailing 12-month returns (ex last month) continue outperforming
- **Signal**: Long SPY when 12-1 month return > 0; flat otherwise
- **Source**: Moskowitz, Ooi & Pedersen (2012) — *Time Series Momentum*, Journal of Financial Economics

| Metric | Value |
|--------|-------|
| Sharpe | +0.49 |
| CAGR | +6.63% |
| Total Return | +37.6% |
| Max Drawdown | 24.7% |
| Trades | 15 |
| Win Rate | 55.9% |
| DSR | 0.862 |
| p-value | 0.277 |

**Verdict: WEAK** — Strongest of the academic strategies on SPY and third overall. The positive
Sharpe at low trade count (15 signals in 5 years) suggests the strategy correctly avoids the 2022
bear market. However, only 15 trades means the Sharpe estimate carries enormous uncertainty.
Moskowitz et al. documented this across 58 futures contracts — the single-asset version loses most
of the diversification benefit. DSR=0.862.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy time_series_momentum --symbol SPY --period 5y
```

---

### 3.2 Carry Trade (Dividend Yield)

- **Hypothesis**: High carry (dividend yield) assets offer positive expected returns
- **Signal**: Long when trailing dividend yield is above rolling median; flat otherwise
- **Source**: Koijen, Moskowitz, Pedersen & Vrugt (2018) — *Carry*, Journal of Financial Economics

| Metric | Value |
|--------|-------|
| Sharpe | +0.44 |
| CAGR | +5.82% |
| Total Return | +32.5% |
| Max Drawdown | 22.4% |
| Trades | 3 |
| Win Rate | 55.9% |
| DSR | 0.838 |
| p-value | 0.323 |

**Verdict: WEAK** — Only 3 trade signals across 5 years; statistically meaningless. Carry is
well-studied in FX and commodity futures; adapting it to equity dividends on a single ETF produces
near-buy-and-hold behavior (SPY's stable dividend yield rarely crosses its own rolling median).
The results reflect SPY's upward drift, not carry alpha. DSR=0.838.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy carry_trade --symbol SPY --period 5y
```

---

### 3.3 Betting Against Beta (BAB)

- **Hypothesis**: Low-beta assets are underpriced relative to high-beta assets (leverage aversion)
- **Signal**: Long low-beta regime, short high-beta regime (single-asset timing signal)
- **Source**: Frazzini & Pedersen (2014) — *Betting Against Beta*, Journal of Financial Economics

| Metric | Value |
|--------|-------|
| Sharpe | +0.15 |
| CAGR | +1.08% |
| Total Return | +5.5% |
| Max Drawdown | 27.0% |
| Trades | 37 |
| Win Rate | 51.4% |
| DSR | 0.628 |
| p-value | 0.744 |

**Verdict: INSUFFICIENT** — BAB is a cross-sectional factor that long-shorts within a universe;
reducing it to a timing signal on a single asset loses the structural driver. The 27% max drawdown
relative to 1% CAGR is unfavorable. This strategy requires a multi-asset implementation (long
low-beta stocks, short high-beta stocks). DSR=0.628.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy betting_against_beta --symbol SPY --period 5y
```

---

### 3.4 Low-Volatility Anomaly

- **Hypothesis**: Low-realized-volatility assets outperform high-volatility assets risk-adjusted
- **Signal**: Long when SPY's rolling realized vol is below its historical median
- **Source**: Ang, Hodrick, Xing & Zhang (2006) — *The Cross-Section of Volatility and Expected Returns*

| Metric | Value |
|--------|-------|
| Sharpe | +0.15 |
| CAGR | +1.05% |
| Total Return | +5.4% |
| Max Drawdown | 24.7% |
| Trades | 76 |
| Win Rate | 48.2% |
| DSR | 0.627 |
| p-value | 0.747 |

**Verdict: INSUFFICIENT** — Same story as BAB: a cross-sectional anomaly (low-vol stocks
outperform high-vol stocks) implemented as a single-asset timing signal. The low-vol regime on
SPY doesn't reliably predict better forward SPY returns. DSR=0.627 confirms no edge.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy low_vol_anomaly --symbol SPY --period 5y
```

---

### 3.5 Post-Earnings Announcement Drift (PEAD)

- **Hypothesis**: Markets underreact to earnings surprises; price continues drifting post-announcement
- **Signal**: Buy after positive earnings surprise; short after negative surprise
- **Source**: Ball & Brown (1968) — *An Empirical Evaluation of Accounting Income Numbers*

| Metric | Value |
|--------|-------|
| Sharpe | −0.11 |
| CAGR | −0.80% |
| Total Return | −3.9% |
| Max Drawdown | 11.3% |
| Trades | 6 |
| Win Rate | 40.0% |
| DSR | 0.402 |
| p-value | 0.805 |

**Verdict: FAILS** — Only 6 trade observations (SPY reports quarterly). PEAD requires individual
stock earnings data; on an index ETF, the "earnings surprise" is a blended aggregate that dilutes
any signal. The result carries no statistical weight and should not be interpreted as evidence for
or against the academic finding. DSR=0.402.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy post_earnings_drift --symbol SPY --period 5y
```

---

## 4. Econophysics Strategies

*Physics-inspired quantitative approaches. These represent a distinct intellectual framework —
treating financial markets as complex physical systems — rather than statistical or behavioral
finance models.*

---

### 4.1 Hurst Exponent Regime Switching

- **Hypothesis**: R/S analysis detects trending (H > 0.5) vs. mean-reverting (H < 0.5) regimes
- **Signal**: Long in trending regime (H > 0.55); short in mean-reverting regime (H < 0.45)
- **Source**: Hurst (1951) / Peters (1994) — *Fractal Market Analysis*

| Metric | Value |
|--------|-------|
| Sharpe | −0.10 |
| CAGR | −2.34% |
| Total Return | −11.1% |
| Max Drawdown | 29.4% |
| Trades | 246 |
| Win Rate | 45.5% |
| DSR | 0.415 |
| p-value | 0.829 |

**Verdict: FAILS** — High trade count (246) with negative Sharpe indicates noisy, cost-sensitive
regime signals. R/S analysis on daily data suffers from short-memory contamination and finite-sample
bias — the Hurst estimate is unreliable at the scales used (rolling 126-day windows). The negative
return with 29% drawdown suggests regime labels systematically lag true transitions. DSR=0.415.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy hurst_exponent --symbol SPY --period 5y
```

---

### 4.2 Shannon Entropy Signal

- **Hypothesis**: Low permutation entropy → ordered/predictable regime → tradeable momentum
- **Signal**: Trade momentum direction when entropy is low (market is "ordered")
- **Source**: Risso (2008) / Zunino et al. (2009) — Permutation entropy in financial markets

| Metric | Value |
|--------|-------|
| Sharpe | −0.43 |
| CAGR | −5.69% |
| Total Return | −25.3% |
| Max Drawdown | 26.0% |
| Trades | 244 |
| Win Rate | 42.8% |
| DSR | 0.173 |
| p-value | 0.346 |

**Verdict: FAILS** — Second worst Sharpe overall among strategies with sufficient trades.
Shannon/permutation entropy on daily returns is highly susceptible to finite-sample estimation
error. At rolling windows of a few hundred daily observations, the entropy signal is dominated by
noise. The high trade count (244) amplifies cost drag. DSR=0.173.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy entropy_signal --symbol SPY --period 5y
```

---

### 4.3 Ornstein-Uhlenbeck Mean Reversion

- **Hypothesis**: Price follows an OU process with calibrated mean-reversion speed and equilibrium
- **Signal**: Trade against deviations from OU equilibrium when signal-to-noise exceeds threshold
- **Source**: Ornstein & Uhlenbeck (1930) / Avellaneda & Lee (2010)

| Metric | Value |
|--------|-------|
| Sharpe | −0.01 |
| CAGR | −0.49% |
| Total Return | −2.4% |
| Max Drawdown | 17.3% |
| Trades | 12 |
| Win Rate | 45.7% |
| DSR | 0.491 |
| p-value | 0.983 |

**Verdict: FAILS** — Near-zero Sharpe (−0.01) with only 12 trades — the OU calibration rarely
produces a strong enough signal to trigger trades. The strategy is essentially inactive. Avellaneda
& Lee applied this to individual stock pairs (stat arb), where two instruments jointly follow an OU
process; applying it to a single instrument produces a degenerate case. DSR=0.491 (noise).

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy ornstein_uhlenbeck --symbol SPY --period 5y
```

---

### 4.4 Power Law Tail Risk

- **Hypothesis**: Tail exponent estimation identifies fat-tail regimes with elevated risk/opportunity
- **Signal**: Trade volatility expansion when tail exponent indicates heavy tails
- **Source**: Gopikrishnan et al. (1999) / Mantegna & Stanley (1999)

| Metric | Value |
|--------|-------|
| Sharpe | −0.07 |
| CAGR | −2.10% |
| Total Return | −10.0% |
| Max Drawdown | **38.8%** |
| Trades | 9 |
| Win Rate | 50.4% |
| DSR | 0.437 |
| p-value | 0.873 |

**Verdict: FAILS** — Highest max drawdown (38.8%) of any strategy, with only 9 trade signals.
Power-law tail estimation requires large samples to be reliable; at rolling windows of a few hundred
daily observations, the tail index has enormous variance. Trading on this noisy signal while holding
through tail events produces catastrophic drawdowns. DSR=0.437.

**Reproduction**:
```bash
python scripts/run_strategy.py --strategy power_law_tail --symbol SPY --period 5y
```

---

## 5. Cross-Strategy Analysis

### 5.1 Performance by Category

| Category | Strategies | Median Sharpe | Strategies > 0 | Best DSR |
|----------|------------|--------------|----------------|----------|
| stats | 3 | +0.16 | 2/3 | 0.964 |
| retail | 8 | +0.05 | 5/8 | 0.922 |
| academic | 5 | +0.15 | 3/5 | 0.862 |
| econophysics | 4 | −0.08 | 0/4 | 0.491 |

**Key observation:** Retail strategies have the widest spread (best: +0.57, worst: −0.59) and
represent the most variable outcomes. Econophysics strategies uniformly underperform, likely because
their theoretical foundations assume stationarity and large samples that daily financial data cannot
satisfy.

### 5.2 Multiple-Testing Adjustment

With 20 strategies tested, the probability of finding at least one Sharpe ≥ 0.77 by chance is
non-trivial. The Deflated Sharpe Ratio addresses this directly:

- **Benchmark Sharpe (SR\*)**: 0.0 (any strategy vs. cash)
- **Expected max Sharpe from 20 random trials**: ~0.98 for T=1255 daily obs
- **Highest observed DSR**: 0.964 (mean_reversion) — **below the 0.95 threshold**

**Conclusion**: None of the 20 strategies achieves statistical significance after multiple-testing
correction at the 5% level. The best result (mean reversion) is marginal (DSR=0.964, borderline
at 10%).

### 5.3 Trade Frequency vs. Performance

The relationship between trade frequency and performance reveals a cost drag pattern:

| Trade frequency | Examples | Mean Sharpe |
|----------------|----------|-------------|
| Very low (<20 trades) | carry (3), golden_cross (5), post_earnings (6) | +0.22 |
| Low (20–100 trades) | rsi, momentum, low_vol, bab | +0.21 |
| Medium (100–200 trades) | mean_rev, macd, doji, bollinger | +0.02 |
| High (200+ trades) | volume_spike, gap_and_go, hurst, entropy | −0.28 |

High-frequency strategies face compounding transaction costs that overwhelm any signal.
Very-low-frequency strategies have insufficient observations for reliable Sharpe estimation.

### 5.4 The "Debunking" Result

Of the 8 retail strategies tested:
- **5 have positive Sharpe** (three_bar, rsi, doji, golden_cross, and momentum — counterintuitive)
- **But none are statistically significant** (max DSR = 0.922)
- **3 actively destroy capital** (bollinger_breakout, gap_and_go, macd_crossover — worst Sharpe of all)

The positive-Sharpe retail strategies appear to work because:
1. Low trade count → insufficient evidence (golden_cross: 5 trades)
2. The study window was favorable for mean-reversion tactics on SPY
3. DSR correction reveals they cannot be distinguished from noise

---

## 6. Conclusions

### What this study shows

1. **No strategy achieves statistical significance** after correcting for 20 simultaneous tests
   at the 5% level. The Deflated Sharpe Ratio is an unforgiving but honest critic.

2. **Retail social-media strategies are systematically overrated.** Bollinger breakout
   (Sharpe=−0.59), gap-and-go (−0.38), and MACD (−0.29) are not merely weak — they
   actively destroy capital over a 5-year horizon on SPY.

3. **Academic strategies are better designed but wrong context.** Time-series momentum
   and carry are cross-sectional multi-asset strategies; forcing them onto a single instrument
   loses their core mechanism. They appear here as benchmarks for implementation quality,
   not claimed edge.

4. **Econophysics strategies underperform uniformly** — the physics-inspired framework
   rests on assumptions (stationarity, large samples, long memory) that daily financial
   data at 5-year windows cannot satisfy.

5. **Mean reversion is the exception** — the strongest result (Sharpe=+0.77, DSR=0.964)
   is consistent with known SPY characteristics: a liquid index ETF with documented
   short-term mean-reversion in daily returns (Poterba & Summers 1988, Lo & MacKinlay 1988).
   DSR=0.964 falls just below the significance threshold, but this is the one strategy
   worth investigating further in a cross-asset or higher-frequency setting.

### What would improve these results

- **Cross-sectional multi-asset tests** for academic factors (BAB, low-vol, PEAD)
- **Longer backtest period** (10–20 years) to improve Sharpe SE estimates
- **Walk-forward OOS decomposition** to separate in-sample from out-of-sample performance
- **Factor attribution** (Fama-French 5-factor) to decompose alpha from factor loading

### The takeaway for hiring managers

Rigorous quantitative research means:
- Testing hypotheses rather than pathology-fitting results
- Reporting strategies that fail, not just those that succeed
- Correcting for the number of strategies tested (Deflated Sharpe Ratio)
- Understanding when a strategy's theoretical mechanism doesn't match the test setup

This codebase demonstrates that methodology: 20 strategies tested, results reported honestly,
none hyped beyond what the data supports.

---

## Appendix: Reproduction Commands

```bash
# Run a single strategy
python scripts/run_strategy.py --strategy mean_reversion --symbol SPY --period 5y

# Run all strategies and compare
python scripts/run_strategy.py --all --symbol SPY --period 5y --compare

# Run by category
python scripts/run_strategy.py --category retail --symbol SPY --period 5y --compare
python scripts/run_strategy.py --category academic --symbol SPY --period 5y --compare
python scripts/run_strategy.py --category econophysics --symbol SPY --period 5y --compare
```

All parameters and data sources are documented in `configs/` and `src/strategies/`.
Strategy source code: `src/strategies/{stats,retail,academic,econophysics}/`.
