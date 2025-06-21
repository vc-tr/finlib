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
        Fetch via yfinance.download with basic retry on exception,
        flatten MultiIndex columns, rename to lowercase.
        """
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    symbol,
                    interval=interval,
                    period=period,
                    progress=False,
                    auto_adjust=False,        # explicit to avoid FutureWarning
                    threads=False             # safer for small requests
                )

                # If MultiIndex (field, ticker), drop the ticker level:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Rename to lowercase and drop any unwanted columns
                df = df.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close",
                    "Volume": "volume"
                })

                # Return only the 5 fields we want
                return df[["open", "high", "low", "close", "volume"]]

            except Exception as e:
                last_exc = e
                time.sleep(self.retry_delay)

        # If we get here, all retries failed
        raise last_exc