import torch
from train import run_training

def test_run_training_smoke(monkeypatch):
    # monkeypatch fetcher to return tiny synthetic data
    from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
    class DummyFetcher(YahooDataFetcher):
        def fetch_ohlcv(self, *args, **kwargs):
            import pandas as pd
            # 100 minutes of linear growth
            idx = pd.date_range("2025-01-01", periods=100, freq="1T")
            df = pd.DataFrame({
                "open": list(range(100)),
                "high": list(range(100)),
                "low":  list(range(100)),
                "close":list(range(100)),
                "volume":[1]*100,
            }, index=idx)
            return df
    monkeypatch.setattr("train.YahooDataFetcher", DummyFetcher)
    
    result = run_training(epochs=3, patience=1)
    assert isinstance(result, dict)
    assert result["epochs_ran"] <= 3
    assert result["best_val_loss"] >= 0.0