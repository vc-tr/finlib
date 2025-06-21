# src/pipeline/data_fetcher_yahoo.py
import time
import pandas as pd
import yfinance as yf
from .data_fetcher import DataFetcher

class YahooDataFetcher(DataFetcher):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_ohlcv(self, symbol: str, interval: str, period: str = "5d") -> pd.DataFrame:
        """
        Fetch via yfinance.download with basic retry on exception.
        """
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    symbol,
                    interval=interval,
                    period=period,
                    progress=False,
                )
                # Rename to lowercase and drop Adj Close
                df = df.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close",
                    "Volume": "volume"
                })
                return df[["open", "high", "low", "close", "volume"]]
            except Exception as e:
                last_exc = e
                time.sleep(self.retry_delay)
        # If we get here, all retries failed
        raise last_exc