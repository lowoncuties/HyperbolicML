#!/usr/bin/env python
"""
Test script to demonstrate how to use the optimized loss functions with the existing codebase.
"""

import os
import sys
import time
import yaml
import numpy as np

# Add parent directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Import functions from the hyperXGB codebase
from xgb.utils import set_seeds
from xgb.hyper_trainer import run_train_UCI


def main():
    """
    This function demonstrates how to use the existing training scripts with the
    optimized loss functions. The optimization is already integrated, so you can
    just run your existing code as normal.
    """
    print("Testing optimized loss functions with existing training scripts...")

    # Example UCI dataset test
    # This is similar to what's in main_run_UCIE.py

    # Generate a simple configuration
    params = {
        "seed": 42,
        "class_label": 0,
        "method": "Pxgboost0",  # Use Poincare XGBoost
        "space": "hyperbolic",
        "max_depth": 6,
        "n_estimators": 20,
        "subsample": 0.8,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.5,
        "colsample_bytree": 0.8,
        "eta": 0.1,
        "gamma": 0.3,
        "folds": 1,  # Just do one fold for testing
        "dataname": "iris",  # Use a simple dataset for testing
        "data_num": 1,  # Only one dataset
    }

    # Save the configuration to a temporary file
    config_path = "temp_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(params, f)

    # Set the seeds
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])

    # Time the execution
    start_time = time.time()

    # Run the training
    print("\nRunning training with optimized loss functions...")
    run_train_UCI(params, config_path)

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)

    print(
        "\nTo run a full benchmark comparing original vs optimized implementations:"
    )
    print("1. First run the benchmark_simple.py script to measure speedup")
    print("2. Use the optimized implementation for all your training needs")
    print(
        "3. If you need to revert, just comment out batch_* functions in hyperutils.py"
    )


if __name__ == "__main__":
    main()
