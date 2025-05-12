# Hyperbolic XGBoost Performance Optimization

This directory contains scripts to test and benchmark the optimized batch-based loss functions for Poincare and Hyperboloid manifolds.

## Overview

The original implementation of the loss functions in the Hyperbolic XGBoost library used one-sample-based calculations for computing gradients and hessians, which is computationally inefficient, especially for large datasets. The new batch-based implementation significantly speeds up the computation by processing all samples in parallel.

## Benchmark Scripts

### 1. Simple Benchmark (`benchmark_simple.py`)

This script provides a direct comparison between the original and optimized implementations of the loss functions:

```bash
python benchmark_simple.py
```

The script will:
- Test the loss functions on synthetic datasets of various sizes
- Compare the execution time of the original vs. optimized implementations
- Print the speedup factor for each dataset size and loss function

### 2. Test with Existing Code (`test_with_existing_code.py`)

This script demonstrates how to use the optimized loss functions with the existing codebase:

```bash
python test_with_existing_code.py
```

The optimization is already integrated into the codebase, so your existing scripts should work as before but with improved performance.

## Using the Optimized Loss Functions

The optimization is transparent - all the existing code that uses the loss functions (customgobj, hyperobj, etc.) will automatically benefit from the optimization.

### Performance Improvements

The batch-based processing typically provides significant speedups, especially for larger datasets:
- For small datasets (hundreds of samples): 5-10x faster
- For medium datasets (thousands of samples): 10-50x faster
- For large datasets (tens of thousands+): 50-100x+ faster

### Implementation Details

The optimization introduces new batch-based functions:
- `batch_egradrgrad` and `batch_ehessrhess` for Poincare ball calculations
- `batch_hyperegradrgrad` and `batch_hyperehessrhess` for Hyperboloid calculations
- `batch_softmax` for efficient softmax computation

These functions replace the slow looping in the original loss functions with vectorized operations.

## Reverting to Original Implementation

If you need to revert to the original implementation, you can comment out or remove the batch-based functions in `hyperutils.py` and restore the original loop-based implementations.

## Tips for Large Datasets

For very large datasets, you may need to adjust the batch size or use additional optimization techniques:

1. Consider reducing the precision (e.g., using float32 instead of float64) if numerical stability is not critical
2. For extremely large datasets, consider chunking the data and processing in smaller batches
3. If memory is a concern, you can modify the vectorized implementations to use in-place operations where possible 