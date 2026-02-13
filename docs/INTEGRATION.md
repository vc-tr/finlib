# Integration with Trading Platforms

This guide explains how to connect quant-forecast strategies and models to external trading platforms for paper trading and live execution. All recommended options below offer **free** paper trading.

---

## Recommended Free Platforms

| Platform | Paper Trading | Live Trading | Python SDK | Best For |
|----------|---------------|--------------|------------|----------|
| **Alpaca** | ✅ Free | Commission-free | `alpaca-py` | Stocks, ETFs, crypto; API-first |
| **Tradier** | ✅ Sandbox | Paid brokerage | `requests` / `tradier-python` | Stocks, options; flexible API |
| **Interactive Brokers** | ✅ Paper | Low commissions | `ib_insync` | Global markets; professional |

---

## 1. Alpaca (Recommended)

**Why Alpaca:** Free paper + live, commission-free, no local server, full Python SDK.

### Setup

```bash
pip install alpaca-py
```

1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Create a **Paper Trading** account (no funding required)
3. Get API keys: Dashboard → API Keys → Generate

### Connect and Trade

```python
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from datetime import datetime, timedelta

# Paper trading (use paper=True)
trading_client = TradingClient(
    "YOUR_API_KEY_ID",
    "YOUR_API_SECRET_KEY",
    paper=True
)

# Verify connection
account = trading_client.get_account()
print(f"Account status: {account.status}")

# Place a market order (example)
order = trading_client.submit_order(
    symbol="SPY",
    qty=1,
    side="buy",
    type="market",
    time_in_force="day"
)
```

### Integrate with quant-forecast

```python
from src.strategies import MeanReversionStrategy
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from alpaca.trading.client import TradingClient

# 1. Get latest prices
fetcher = YahooDataFetcher()
df = fetcher.fetch_ohlcv("SPY", "1d", period="30d")
prices = df["close"]

# 2. Generate signals
strategy = MeanReversionStrategy(lookback=20, entry_z=2.0)
signals, _ = strategy.backtest_returns(prices)
current_signal = signals.iloc[-1]  # -1=short, 0=flat, 1=long

# 3. Execute via Alpaca
client = TradingClient("KEY", "SECRET", paper=True)
if current_signal == 1:
    client.submit_order(symbol="SPY", qty=1, side="buy", type="market", time_in_force="day")
elif current_signal == -1:
    client.submit_order(symbol="SPY", qty=1, side="sell", type="market", time_in_force="day")
```

### Environment Variables

```bash
export ALPACA_API_KEY_ID="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

Use in code: `os.environ.get("ALPACA_API_KEY_ID")` — never hardcode keys.

---

## 2. Tradier

**Why Tradier:** Free sandbox, good for options, REST API.

### Setup

```bash
pip install requests  # or: pip install tradier-python
```

1. Sign up at [tradier.com](https://tradier.com)
2. Get Sandbox token: [sandbox.tradier.com](https://sandbox.tradier.com) → API Access

### Basic Usage

```python
import requests

BASE = "https://sandbox.tradier.com/v1"
headers = {
    "Authorization": f"Bearer YOUR_SANDBOX_TOKEN",
    "Accept": "application/json"
}

# Get quotes
r = requests.get(f"{BASE}/markets/quotes?symbols=SPY", headers=headers)
print(r.json())

# Place order (sandbox)
r = requests.post(f"{BASE}/accounts/ACCOUNT_ID/orders", headers=headers, data={
    "class": "equity",
    "symbol": "SPY",
    "side": "buy",
    "quantity": "1",
    "type": "market"
})
```

### Integrate with quant-forecast

Same pattern: run your strategy → get signal → call Tradier REST API to place order.

---

## 3. Interactive Brokers (IBKR)

**Why IBKR:** Global markets, paper account, professional tools. Requires TWS or IB Gateway running.

### Setup

```bash
pip install ib_insync
```

1. Open IBKR Paper Trading account
2. Install [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
3. Enable API: TWS → File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients

### Basic Usage

```python
from ib_insync import IB, Stock, MarketOrder

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 paper, 7496 live

contract = Stock('SPY', 'SMART', 'USD')
order = MarketOrder('BUY', 1)
trade = ib.placeOrder(contract, order)

ib.disconnect()
```

---

## General Integration Pattern

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  quant-forecast │────▶│  Signal / Order   │────▶│ Trading Platform│
│  (strategy or   │     │  (your bridge)    │     │  (Alpaca, etc.) │
│   model)        │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Steps

1. **Data**: Use `YahooDataFetcher` or platform’s data API
2. **Signal**: Run strategy (`MeanReversionStrategy`, `MomentumStrategy`, etc.) or model prediction
3. **Bridge**: Map signal (-1, 0, 1) or prediction to order (buy/sell/size)
4. **Execute**: Call platform API to place order
5. **Risk**: Add position sizing, max exposure, stop-loss in your bridge

### Example Bridge (Alpaca)

```python
def signal_to_order(signal: float, symbol: str, size: int = 1):
    """Convert quant-forecast signal to Alpaca order."""
    if signal > 0:
        return {"side": "buy", "symbol": symbol, "qty": size}
    elif signal < 0:
        return {"side": "sell", "symbol": symbol, "qty": size}
    return None  # flat, no order
```

---

## Recommendations

| Use Case | Recommendation |
|----------|----------------|
| **Quick start, US stocks** | Alpaca paper trading |
| **Options, advanced orders** | Tradier sandbox |
| **Global markets, futures** | Interactive Brokers |
| **Deep learning predictions** | Train with `train.py`, load model, predict → bridge → Alpaca |

### Before Live Trading

1. Paper trade for at least 1–2 months
2. Add transaction costs and slippage to backtests
3. Use small position sizes initially
4. Implement circuit breakers (max daily loss, max drawdown)
