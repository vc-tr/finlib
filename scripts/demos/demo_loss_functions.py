#!/usr/bin/env python3
"""
Demonstration script for loss function integration.

This script shows how to use the updated train.py with different loss functions.
"""

import subprocess
import sys
import os

def run_training_with_loss(loss_name: str, epochs: int = 1, patience: int = 1):
    """Run training with a specific loss function."""
    print(f"\n🚀 Running training with {loss_name.upper()} loss...")
    print("-" * 50)
    
    cmd = [
        sys.executable, "train.py",
        "--loss", loss_name,
        "--epochs", str(epochs),
        "--patience", str(patience)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running training: {e}")


def main():
    """Main demonstration function."""
    print("🔬 Loss Function Integration Demo")
    print("=" * 50)
    print("This demo shows how to use different loss functions with the training pipeline.")
    
    # Test all three loss functions
    loss_functions = ['mse', 'mae', 'huber']
    
    for loss_name in loss_functions:
        run_training_with_loss(loss_name)
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n💡 Usage examples:")
    print("  python train.py --loss mse --epochs 20")
    print("  python train.py --loss mae --epochs 20")
    print("  python train.py --loss huber --epochs 20")
    print("\n📖 For more options:")
    print("  python train.py --help")


if __name__ == "__main__":
    main() 