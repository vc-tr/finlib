"""
Tests for training functionality.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm import PriceLSTM


def test_lstm_model():
    """Test that LSTM model can be created and run forward pass."""
    model = PriceLSTM(input_dim=1, hidden_dim=32)
    x = torch.randn(4, 10, 1)  # batch_size=4, seq_len=10, input_dim=1
    output = model(x)
    assert output.shape == (4, 1)  # batch_size=4, output_dim=1


def test_loss_functions():
    """Test that all loss functions work correctly."""
    # Create dummy data
    y_pred = torch.randn(10, 1)
    y_true = torch.randn(10, 1)
    
    # Test MSE loss
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(y_pred, y_true)
    assert isinstance(mse_value, torch.Tensor)
    assert mse_value.item() >= 0
    
    # Test MAE loss
    mae_loss = nn.L1Loss()
    mae_value = mae_loss(y_pred, y_true)
    assert isinstance(mae_value, torch.Tensor)
    assert mae_value.item() >= 0
    
    # Test Huber loss
    huber_loss = nn.SmoothL1Loss(beta=1.0)
    huber_value = huber_loss(y_pred, y_true)
    assert isinstance(huber_value, torch.Tensor)
    assert huber_value.item() >= 0


def test_loss_function_selection():
    """Test loss function selection logic."""
    import sys
    sys.path.append('..')  # Add parent directory
    from train import get_loss_function
    
    # Test all supported loss functions
    for loss_name in ['mse', 'mae', 'huber']:
        loss_fn = get_loss_function(loss_name)
        assert isinstance(loss_fn, nn.Module)
        
        # Test that it can compute loss
        y_pred = torch.randn(5, 1)
        y_true = torch.randn(5, 1)
        loss_value = loss_fn(y_pred, y_true)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.item() >= 0
    
    # Test invalid loss function
    with pytest.raises(ValueError, match="Unsupported loss function"):
        get_loss_function('invalid_loss')


def test_training_with_all_loss_functions():
    """
    Smoke test: Run training with all three loss functions to ensure they work.
    This is a minimal training run to catch any integration issues.
    """
    import sys
    sys.path.append('..')  # Add parent directory
    from train import run_training
    
    # Test each loss function with minimal training
    loss_functions = ['mse', 'mae', 'huber']
    
    for loss_name in loss_functions:
        print(f"\n🧪 Testing training with {loss_name.upper()} loss...")
        
        try:
            # Run minimal training (1 epoch, low patience)
            result = run_training(epochs=1, patience=1, loss_name=loss_name)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'best_val_loss' in result
            assert 'epochs_ran' in result
            assert result['epochs_ran'] >= 1
            
            print(f"✅ {loss_name.upper()} loss training completed successfully")
            print(f"   Best validation loss: {result['best_val_loss']:.6f}")
            print(f"   Epochs run: {result['epochs_ran']}")
            
        except Exception as e:
            pytest.fail(f"Training with {loss_name} loss failed: {str(e)}")


def test_loss_function_gradients():
    """Test that all loss functions can compute gradients properly."""
    # Create model and data
    model = PriceLSTM(input_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create dummy dataset
    X = torch.randn(20, 10, 1)  # 20 samples, seq_len=10, input_dim=1
    y = torch.randn(20, 1)      # 20 targets
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)
    
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'huber': nn.SmoothL1Loss(beta=1.0)
    }
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"🧪 Testing gradient computation with {loss_name.upper()} loss...")
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        batch_x, batch_y = next(iter(dataloader))
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in model.parameters())
        
        assert has_gradients, f"No gradients computed for {loss_name} loss"
        print(f"✅ {loss_name.upper()} loss gradients computed successfully")


def test_loss_function_properties():
    """Test mathematical properties of loss functions."""
    # Test with known values
    y_pred = torch.tensor([[1.0], [2.0], [3.0]])
    y_true = torch.tensor([[1.0], [2.0], [3.0]])  # Perfect predictions
    
    # MSE should be 0 for perfect predictions
    mse_loss = nn.MSELoss()
    mse_perfect = mse_loss(y_pred, y_true)
    assert mse_perfect.item() == 0.0, f"MSE should be 0 for perfect predictions, got {mse_perfect.item()}"
    
    # MAE should be 0 for perfect predictions
    mae_loss = nn.L1Loss()
    mae_perfect = mae_loss(y_pred, y_true)
    assert mae_perfect.item() == 0.0, f"MAE should be 0 for perfect predictions, got {mae_perfect.item()}"
    
    # Huber should be 0 for perfect predictions
    huber_loss = nn.SmoothL1Loss(beta=1.0)
    huber_perfect = huber_loss(y_pred, y_true)
    assert huber_perfect.item() == 0.0, f"Huber should be 0 for perfect predictions, got {huber_perfect.item()}"
    
    # Test with non-zero error
    y_pred_error = torch.tensor([[1.0], [2.0], [3.0]])
    y_true_error = torch.tensor([[0.0], [3.0], [2.0]])  # Some errors
    
    mse_error = mse_loss(y_pred_error, y_true_error)
    mae_error = mae_loss(y_pred_error, y_true_error)
    huber_error = huber_loss(y_pred_error, y_true_error)
    
    # All should be positive for non-zero errors
    assert mse_error.item() > 0, "MSE should be positive for non-zero errors"
    assert mae_error.item() > 0, "MAE should be positive for non-zero errors"
    assert huber_error.item() > 0, "Huber should be positive for non-zero errors"
    
    print("✅ All loss functions have correct mathematical properties")


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running loss function tests...")
    test_loss_functions()
    test_loss_function_selection()
    test_loss_function_gradients()
    test_loss_function_properties()
    print("✅ All basic tests passed!")
    
    # Run training smoke tests (these take longer)
    print("\n🚀 Running training smoke tests...")
    test_training_with_all_loss_functions()
    print("✅ All training smoke tests passed!")
    
    print("\n🎉 All tests completed successfully!")