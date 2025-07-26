#!/usr/bin/env python3
"""
Loss Function Analysis for Time Series Forecasting

This script compares MSE, MAE, and Huber (Smooth L1) loss functions:
1. Derives gradient sensitivity to outliers for each loss
2. Runs 1-epoch training on synthetic time series data
3. Logs train/val loss for each loss function
4. Provides recommendations for financial time series forecasting
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleLSTM(nn.Module):
    """Simple LSTM model for loss function comparison."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])


def generate_synthetic_timeseries(n_points: int = 100, noise_level: float = 0.1, 
                                outlier_prob: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series with outliers for loss function testing.
    
    Args:
        n_points: Number of data points
        noise_level: Standard deviation of Gaussian noise
        outlier_prob: Probability of outlier occurrence
        
    Returns:
        Tuple of (features, targets) where features are sequences and targets are next values
    """
    # Generate base trend with some seasonality
    t = np.linspace(0, 4*np.pi, n_points)
    trend = 0.1 * t  # Linear trend
    seasonality = 0.5 * np.sin(t) + 0.3 * np.sin(2*t)  # Multiple frequencies
    base_signal = trend + seasonality
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, n_points)
    signal = base_signal + noise
    
    # Add outliers
    outlier_mask = np.random.random(n_points) < outlier_prob
    outlier_values = np.random.normal(0, 3*noise_level, n_points)  # 3x larger noise
    signal[outlier_mask] += outlier_values[outlier_mask]
    
    # Create sequences for LSTM (sequence_length = 10)
    seq_len = 10
    X, y = [], []
    
    for i in range(seq_len, n_points):
        X.append(signal[i-seq_len:i])
        y.append(signal[i])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def analyze_gradient_sensitivity() -> Dict[str, Dict]:
    """
    Analyze gradient sensitivity to outliers for each loss function.
    
    Returns:
        Dictionary with gradient analysis for each loss function
    """
    print("🔍 Analyzing Gradient Sensitivity to Outliers...")
    
    # Create synthetic data with known outliers
    y_true = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Introduce outliers
    y_true_with_outliers = y_true.clone()
    y_true_with_outliers[3] = 5.0  # Large outlier
    y_true_with_outliers[7] = -3.0  # Negative outlier
    
    # Define loss functions
    losses = {
        'mse': nn.MSELoss(reduction='none'),
        'mae': nn.L1Loss(reduction='none'),
        'huber': nn.SmoothL1Loss(reduction='none', beta=1.0)
    }
    
    analysis = {}
    
    for loss_name, loss_fn in losses.items():
        print(f"\n📊 {loss_name.upper()} Loss Analysis:")
        
        # Calculate losses
        loss_normal = loss_fn(y_pred, y_true)
        loss_with_outliers = loss_fn(y_pred, y_true_with_outliers)
        
        # Calculate gradients (simplified - in practice gradients depend on model parameters)
        # For demonstration, we'll show how the loss changes with prediction changes
        gradients = []
        for i in range(len(y_pred)):
            # Calculate how loss changes with small changes in prediction
            y_pred_plus = y_pred.clone()
            y_pred_plus[i] += 0.01
            y_pred_minus = y_pred.clone()
            y_pred_minus[i] -= 0.01
            
            loss_plus = loss_fn(y_pred_plus, y_true_with_outliers).mean()
            loss_minus = loss_fn(y_pred_minus, y_true_with_outliers).mean()
            
            # Approximate gradient magnitude
            grad_magnitude = abs(loss_plus - loss_minus) / 0.02
            gradients.append(grad_magnitude.item())
        
        # Calculate sensitivity metrics
        normal_loss_mean = loss_normal.mean().item()
        outlier_loss_mean = loss_with_outliers.mean().item()
        outlier_impact = outlier_loss_mean / normal_loss_mean if normal_loss_mean > 0 else float('inf')
        
        analysis[loss_name] = {
            'normal_loss_mean': normal_loss_mean,
            'outlier_loss_mean': outlier_loss_mean,
            'outlier_impact_ratio': outlier_impact,
            'gradient_magnitudes': gradients,
            'max_gradient': max(gradients),
            'min_gradient': min(gradients),
            'gradient_std': np.std(gradients)
        }
        
        print(f"  Normal loss mean: {normal_loss_mean:.6f}")
        print(f"  Outlier loss mean: {outlier_loss_mean:.6f}")
        print(f"  Outlier impact ratio: {outlier_impact:.2f}x")
        print(f"  Max gradient magnitude: {max(gradients):.6f}")
        print(f"  Gradient std: {np.std(gradients):.6f}")
    
    return analysis


def run_loss_comparison_experiment(n_points: int = 100, batch_size: int = 16, 
                                 learning_rate: float = 0.01) -> Dict[str, List[float]]:
    """
    Run 1-epoch training experiment comparing all three loss functions.
    
    Args:
        n_points: Number of data points in synthetic dataset
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary with training and validation losses for each loss function
    """
    print(f"\n🚀 Running Loss Function Comparison Experiment...")
    print(f"Dataset size: {n_points} points, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Generate synthetic data
    X, y = generate_synthetic_timeseries(n_points=n_points)
    
    # Split data (80% train, 20% val)
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).unsqueeze(-1)  # (N, seq_len, 1)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)  # (N, 1)
    X_val = torch.from_numpy(X_val).unsqueeze(-1)
    y_val = torch.from_numpy(y_val).unsqueeze(-1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss functions
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'huber': nn.SmoothL1Loss(beta=1.0)
    }
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n📈 Training with {loss_name.upper()} loss...")
        
        # Initialize model, optimizer
        model = SimpleLSTM(input_dim=1, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop (1 epoch)
        model.train()
        train_losses = []
        val_losses = []
        
        # Training
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Train Loss = {loss.item():.6f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                val_loss = loss_fn(predictions, y_batch)
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        results[loss_name] = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"  Final Train Loss: {avg_train_loss:.6f}")
        print(f"  Final Val Loss: {avg_val_loss:.6f}")
    
    return results


def create_visualizations(gradient_analysis: Dict, training_results: Dict, 
                         save_path: str = "figures/loss_analysis.png"):
    """
    Create visualizations for loss function comparison.
    
    Args:
        gradient_analysis: Results from gradient sensitivity analysis
        training_results: Results from training experiment
        save_path: Path to save the visualization
    """
    print(f"\n📊 Creating visualizations...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Gradient sensitivity comparison
    loss_names = list(gradient_analysis.keys())
    outlier_impacts = [gradient_analysis[name]['outlier_impact_ratio'] for name in loss_names]
    
    ax1.bar(loss_names, outlier_impacts, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('Outlier Impact Ratio (Higher = More Sensitive)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Impact Ratio')
    ax1.grid(True, alpha=0.3)
    
    # 2. Training loss comparison
    train_losses = [training_results[name]['train_loss'] for name in loss_names]
    val_losses = [training_results[name]['val_loss'] for name in loss_names]
    
    x = np.arange(len(loss_names))
    width = 0.35
    
    ax2.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
    ax2.bar(x + width/2, val_losses, width, label='Val Loss', alpha=0.8)
    ax2.set_title('Training vs Validation Loss (1 Epoch)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss function shapes
    x_range = np.linspace(-3, 3, 1000)
    y_true = torch.zeros_like(torch.tensor(x_range))
    
    for i, loss_name in enumerate(loss_names):
        if loss_name == 'mse':
            loss_values = (x_range - y_true.numpy()) ** 2
        elif loss_name == 'mae':
            loss_values = np.abs(x_range - y_true.numpy())
        elif loss_name == 'huber':
            huber_loss = nn.SmoothL1Loss(beta=1.0, reduction='none')
            loss_values = huber_loss(torch.tensor(x_range), y_true).numpy()
        
        ax3.plot(x_range, loss_values, label=loss_name.upper(), linewidth=2)
    
    ax3.set_title('Loss Function Shapes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Prediction Error (y_pred - y_true)')
    ax3.set_ylabel('Loss Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)
    
    # 4. Gradient magnitude comparison
    gradient_stds = [gradient_analysis[name]['gradient_std'] for name in loss_names]
    ax4.bar(loss_names, gradient_stds, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_title('Gradient Standard Deviation (Stability Measure)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Gradient Std')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📁 Visualization saved to: {save_path}")
    
    plt.show()


def generate_recommendations(gradient_analysis: Dict, training_results: Dict) -> str:
    """
    Generate recommendations based on analysis results.
    
    Args:
        gradient_analysis: Results from gradient sensitivity analysis
        training_results: Results from training experiment
        
    Returns:
        String with recommendations
    """
    print("\n💡 Generating recommendations...")
    
    # Analyze results
    mse_outlier_impact = gradient_analysis['mse']['outlier_impact_ratio']
    mae_outlier_impact = gradient_analysis['mae']['outlier_impact_ratio']
    huber_outlier_impact = gradient_analysis['huber']['outlier_impact_ratio']
    
    mse_val_loss = training_results['mse']['val_loss']
    mae_val_loss = training_results['mae']['val_loss']
    huber_val_loss = training_results['huber']['val_loss']
    
    # Determine best performer
    best_loss = min([mse_val_loss, mae_val_loss, huber_val_loss])
    best_loss_name = 'mse' if mse_val_loss == best_loss else 'mae' if mae_val_loss == best_loss else 'huber'
    
    # Determine most robust to outliers
    most_robust = min([mse_outlier_impact, mae_outlier_impact, huber_outlier_impact])
    most_robust_name = 'mse' if mse_outlier_impact == most_robust else 'mae' if mae_outlier_impact == most_robust else 'huber'
    
    recommendations = f"""
## Loss Function Analysis Results

### Performance Summary
- **MSE Validation Loss**: {mse_val_loss:.6f}
- **MAE Validation Loss**: {mae_val_loss:.6f}  
- **Huber Validation Loss**: {huber_val_loss:.6f}

### Outlier Sensitivity
- **MSE Outlier Impact**: {mse_outlier_impact:.2f}x
- **MAE Outlier Impact**: {mae_outlier_impact:.2f}x
- **Huber Outlier Impact**: {huber_outlier_impact:.2f}x

### Key Findings
1. **Best Validation Performance**: {best_loss_name.upper()} ({best_loss:.6f})
2. **Most Robust to Outliers**: {most_robust_name.upper()} ({most_robust:.2f}x impact)
3. **Gradient Stability**: {gradient_analysis['huber']['gradient_std']:.6f} (Huber) vs {gradient_analysis['mse']['gradient_std']:.6f} (MSE)

### Recommendations for Financial Time Series

**Primary Recommendation: HUBER LOSS**
- Combines benefits of MSE (smooth gradients) and MAE (robustness to outliers)
- Particularly suitable for financial data with frequent outliers
- Provides stable training dynamics

**Secondary Options:**
- **MSE**: Use when data is clean and outliers are rare
- **MAE**: Use when outliers are frequent but training stability is less critical

**Implementation Notes:**
- Huber loss with β=1.0 provides good balance
- Consider adaptive β values based on data characteristics
- Monitor gradient norms during training for stability
"""
    
    return recommendations


def main():
    """Main function to run the complete loss analysis."""
    parser = argparse.ArgumentParser(description='Loss Function Analysis for Time Series')
    parser.add_argument('--n_points', type=int, default=100, help='Number of data points')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations')
    parser.add_argument('--output', type=str, default='docs/loss_choice.md', help='Output markdown file')
    
    args = parser.parse_args()
    
    print("🔬 Loss Function Analysis for Time Series Forecasting")
    print("=" * 60)
    
    # Run gradient sensitivity analysis
    gradient_analysis = analyze_gradient_sensitivity()
    
    # Run training comparison experiment
    training_results = run_loss_comparison_experiment(
        n_points=args.n_points,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Create visualizations
    if args.save_viz:
        create_visualizations(gradient_analysis, training_results)
    
    # Generate recommendations
    recommendations = generate_recommendations(gradient_analysis, training_results)
    
    # Save to markdown file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(recommendations)
    
    print(f"\n📄 Analysis saved to: {args.output}")
    print("\n🎉 Loss function analysis completed!")


if __name__ == "__main__":
    main() 