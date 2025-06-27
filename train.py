import os
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.models.lstm import PriceLSTM

# train.py
import mlflow, torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# …
# old HYPERPARAMS   # max grad norm
# (keep SEQ_LEN, BATCH_SIZE, HIDDEN_DIM, LR as before)
# hyperparams
SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS     = 20
PATIENCE   = 5       # early-stop patience
CLIP_VALUE = 1.0  

def prepare_data(symbol="SPY"):
    fetcher = YahooDataFetcher(max_retries=1, retry_delay=0)
    df = fetcher.fetch_ohlcv(symbol, "1m", period="2d")
    df = reindex_and_backfill(df)
    # use close price and pct_change as feature+label
    series = df["close"].pct_change().dropna()
    # Pre-allocate numpy arrays to avoid slow tensor creation
    n_samples = len(series) - SEQ_LEN
    X = np.zeros((n_samples, SEQ_LEN), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = series.iloc[i : i + SEQ_LEN].values
        y[i] = series.iloc[i + SEQ_LEN]
    
    # Convert to tensors efficiently 
    X = torch.from_numpy(X).unsqueeze(-1)  # (N,SEQ_LEN,1)
    y = torch.from_numpy(y).unsqueeze(-1)  # (N,1)
    ds = TensorDataset(X, y)
    n_train = int(len(ds) * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    return train_ds, val_ds

def train():
    # MLflow setup
    mlflow.set_experiment("quant-lstm-baseline")
    with mlflow.start_run():
        # log hyperparams
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
        })

        # data
        train_ds, val_ds = prepare_data()
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # model
        model = PriceLSTM(input_dim=1, hidden_dim=HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        # one epoch
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_loss)

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        mlflow.log_metric("val_loss", val_loss)

        # Save model artifact with proper signature to avoid warnings
        sample_input = torch.randn(1, SEQ_LEN, 1)
        mlflow.pytorch.log_model(
            model, 
            "price_lstm_model",
            input_example=sample_input.numpy()  # Convert to numpy for MLflow
        )

        print(f"Train Loss: {avg_loss:.6f}  Val Loss: {val_loss:.6f}")

if __name__ == "__main__":
    train()