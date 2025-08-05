#!/usr/bin/env python3
"""
Simple script to launch TensorBoard for viewing training logs.
"""
import subprocess
import sys
import os

def main():
    """Launch TensorBoard pointing to the runs directory."""
    log_dir = "runs"
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory '{log_dir}' does not exist.")
        print("Make sure you've run training first to generate logs.")
        sys.exit(1)
    
    print(f"Launching TensorBoard with log directory: {log_dir}")
    print("TensorBoard will be available at: http://localhost:6006")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        subprocess.run(["tensorboard", "--logdir", log_dir], check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("Error: TensorBoard not found. Please install it with:")
        print("pip install tensorboard")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running TensorBoard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 