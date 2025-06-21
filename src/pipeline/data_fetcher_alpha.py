# src/pipeline/data_fetcher_alpha.py
import os
import pandas as pd
import requests
from .data_fetcher import DataFetcher
from dotenv import load_dotenv

load_dotenv()

class AlphaVantageDataFetcher(DataFetcher):
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_KEY")
        if not self.api_key:
            raise ValueError("AlphaVantage API key not set")

    def fetch_ohlcv(self, symbol: str, interval: str, outputsize: str = "compact") -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        r = requests.get(self.BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get(f"Time Series ({interval})", {})
        df = pd.DataFrame.from_dict(data, orient="index", dtype=float)
        df = df.rename(columns={
            "1. open": "open", "2. high": "high",
            "3. low": "low", "4. close": "close",
            "5. volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        return df.sort_index()[["open", "high", "low", "close", "volume"]]