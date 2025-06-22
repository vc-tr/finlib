# scripts/benchmark_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill

def main():
    fetcher = YahooDataFetcher(max_retries=1, retry_delay=0)
    print("Fetching 5 days of 1m SPY data…")
    df_raw = fetcher.fetch_ohlcv("SPY", "1m", period="5d")

    # Benchmark reindex/backfill
    print(f"Raw bars: {len(df_raw)}")
    start = time.time()
    df_clean = reindex_and_backfill(df_raw)
    elapsed = time.time() - start

    print(f"Cleaned bars: {len(df_clean)}")
    print(f"reindex_and_backfill took {elapsed:.3f} seconds")

if __name__ == "__main__":
    main()