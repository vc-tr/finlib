import os
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import argparse
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.models.lstm import PriceLSTM
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 20
PATIENCE = 5       # early-stop patience
CLIP_VALUE = 1.0   # max grad norm


def get_loss_function(loss_name: str) -> nn.Module:
    """
    Get loss function based on name.
    
    Args:
        loss_name: Name of loss function ('mse', 'mae', 'huber')
        
    Returns:
        PyTorch loss function
        
    Raises:
        ValueError: If loss_name is not supported
    """
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'huber': nn.SmoothL1Loss(beta=1.0)
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                        f"Supported options: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]

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

def run_training(epochs=EPOCHS, patience=PATIENCE, loss_name='mse'):
    """
    Extract training logic into a reusable function.
    Returns dict of final metrics for easier testing and automation.
    
    Args:
        epochs: Number of training epochs
        patience: Early stopping patience
        loss_name: Loss function name ('mse', 'mae', 'huber')
    """
    # MLflow setup
    mlflow.set_experiment("quant-lstm-baseline")
    with mlflow.start_run():
        # TensorBoard setup
        from datetime import datetime
        tb = SummaryWriter(f"runs/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create hyperparams dict for TensorBoard
        hparams = {
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
            "epochs": epochs,
            "patience": patience,
            "clip_value": CLIP_VALUE,
            "loss_function": loss_name
        }
        tb.add_hparams(hparams, {})  # log hyperparams
        
        # log hyperparams
        mlflow.log_params(hparams)

        # data
        train_ds, val_ds = prepare_data()
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # model, optimizer, scheduler, criterion
        model = PriceLSTM(input_dim=1, hidden_dim=HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = get_loss_function(loss_name)
        
        # Early stopping variables
        best_val = float("inf")
        wait = 0
        epochs_ran = 0

        # Training loop with multiple epochs
        for epoch in range(1, epochs + 1):
            epochs_ran = epoch
            
            # — Train —
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            
            # — Validate —
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # — Scheduler & early-stop —
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", current_lr, step=epoch)
            
            # TensorBoard logging
            tb.add_scalar("Loss/train", train_loss, epoch)
            tb.add_scalar("Loss/val", val_loss, epoch)
            tb.add_scalar("LR", current_lr, epoch)
            
            # Every few epochs, log weight histograms
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    tb.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        tb.add_histogram(f"{name}.grad", param.grad, epoch)
            
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                # Save the best checkpoint
                sample_input = torch.randn(1, SEQ_LEN, 1)
                mlflow.pytorch.log_model(
                    model, 
                    "best_model",
                    input_example=sample_input.numpy()
                )
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

        print(f"Training completed. Best validation loss: {best_val:.6f}")
        
        # Close TensorBoard writer
        tb.close()
        
        # Return dict of final metrics
        return {
            "best_val_loss": best_val,
            "epochs_ran": epochs_ran
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for time series forecasting')
    parser.add_argument('--loss', type=str, default='mse', 
                       choices=['mse', 'mae', 'huber'],
                       help='Loss function to use (default: mse)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                       help=f'Early stopping patience (default: {PATIENCE})')
    parser.add_argument("--model_type", type=str, default="lstm",
                       choices=["lstm","bilstm","gru"], help="Which RNN variant")
    args = parser.parse_args()
    
    print(f"Starting training with {args.loss.upper()} loss function...")
    print(f"Configuration: {args.epochs} epochs, patience={args.patience}")
    
    result = run_training(epochs=args.epochs, patience=args.patience, loss_name=args.loss)
    print("Training completed:", result)