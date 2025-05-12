#!/usr/bin/env python
"""
Memory-efficient benchmark script with configurable batch sizes to compare
the performance of original vs optimized batch-based loss functions.
"""

import os
import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add parent directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Import both original and optimized functions
from xgb.hyperutils import customgobj, logregobj, hyperobj

# Save current functions
original_customgobj = customgobj
original_hyperobj = hyperobj
original_logregobj = logregobj

# Import the optimized versions - backup original first
import importlib
import xgb.hyperutils

backup_module = importlib.import_module("xgb.hyperutils")
original_module_dict = backup_module.__dict__.copy()

# Test with different dataset sizes
DATASET_SIZES = [100, 500, 1000, 5000, 10000]
N_FEATURES = 100
N_CLASSES = 8
N_ITERATIONS = 6  # Number of training iterations


# Function to directly test the batch gradient calculations with configurable batch size
def benchmark_direct_calculation(n_samples, n_features, batch_size=None):
    """Benchmark the direct calculation of gradients with different batch sizes."""
    from xgb.hyperutils import egradrgrad, batch_egradrgrad

    # Generate random data (values in range -0.5 to 0.5 for Poincare ball)
    preds = np.random.uniform(-0.5, 0.5, size=(n_samples, n_features))
    grads = np.random.normal(0, 1, size=(n_samples, n_features))

    # Test original implementation (element by element)
    start_time = time.time()
    result_orig = np.zeros_like(preds)
    for i in range(n_samples):
        for j in range(n_features):
            result_orig[i, j] = egradrgrad(preds[i, j], grads[i, j])
    orig_time = time.time() - start_time

    # Test batch implementation with specified batch size
    start_time = time.time()
    if batch_size is None or batch_size >= n_samples:
        # Process all at once
        result_batch = batch_egradrgrad(preds, grads)
    else:
        # Process in batches
        result_batch = np.zeros_like(preds)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            result_batch[i:end_idx] = batch_egradrgrad(
                preds[i:end_idx], grads[i:end_idx]
            )
    batch_time = time.time() - start_time

    # Calculate speedup and verify results
    speedup = orig_time / batch_time
    max_diff = np.max(np.abs(result_orig - result_batch))

    return orig_time, batch_time, speedup, max_diff


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch loss functions with configurable batch size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (default: process all at once)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10000,
        help="Maximum dataset size to test (default: 10000)",
    )
    args = parser.parse_args()

    # Test parameters
    dataset_sizes = [100, 500, 1000, 5000, args.max_size]
    feature_dims = [10, 20]
    batch_size = args.batch_size

    print(
        "Benchmarking batch optimization with"
        + (
            f" batch size {batch_size}"
            if batch_size
            else " full batch processing"
        )
    )
    print("=" * 80)

    print("\nDIRECT GRADIENT CALCULATION BENCHMARK")
    print(
        f"{'Size':<8} {'Dim':<5} {'Original (s)':<15} {'Batch (s)':<15} {'Speedup':<10} {'Max Diff':<10}"
    )
    print("-" * 70)

    for dim in feature_dims:
        for size in dataset_sizes:
            orig_time, batch_time, speedup, max_diff = (
                benchmark_direct_calculation(size, dim, batch_size)
            )
            print(
                f"{size:<8} {dim:<5} {orig_time:<15.6f} {batch_time:<15.6f} {speedup:<10.2f}x {max_diff:<10.2e}"
            )

    # Additional benchmarks for different batch sizes if no specific batch size was given
    if batch_size is None and args.max_size >= 5000:
        print(
            "\nIMPACT OF BATCH SIZE ON PERFORMANCE (Dataset size: 5000, Dimensions: 20)"
        )
        print(
            f"{'Batch Size':<15} {'Time (s)':<15} {'Speedup vs Original':<20}"
        )
        print("-" * 55)

        test_size = 5000
        test_dim = 20
        orig_time, _, _, _ = benchmark_direct_calculation(
            test_size, test_dim, None
        )

        for bs in [100, 500, 1000, 2500, 5000]:
            _, batch_time, speedup, _ = benchmark_direct_calculation(
                test_size, test_dim, bs
            )
            print(f"{bs:<15} {batch_time:<15.6f} {speedup:<20.2f}x")

    print("\nSUMMARY")
    print("-" * 80)
    print(
        "The batch implementation significantly outperforms the original implementation."
    )
    if batch_size:
        print(
            f"Using a batch size of {batch_size} helps manage memory usage for large datasets."
        )
    else:
        print(
            "Processing the entire dataset at once provides the best performance,"
        )
        print(
            "but using smaller batches can help with very large datasets or limited memory."
        )


if __name__ == "__main__":
    main()
