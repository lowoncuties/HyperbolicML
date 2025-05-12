#!/usr/bin/env python
"""
Benchmark old vs batched hyperbolic XGBoost objectives.
"""

import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

# ————————————————————————————————————————————
# Ensure xgb package is on the path
# ————————————————————————————————————————————
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

# ————————————————————————————————————————————
# Import old (non‑batched) implementations
# ————————————————————————————————————————————
from xgb.hyperutils import (
    customgobj as orig_customgobj,
    hyperobj as orig_hyperobj,
)

# ————————————————————————————————————————————
# Import new batched implementations
# ————————————————————————————————————————————
from xgb.hyperutils_batch import (
    customgobj as batched_customgobj,
    hyperobj as batched_hyperobj,
)

# ————————————————————————————————————————————
# Benchmark settings
# ————————————————————————————————————————————
DATASET_SIZES = [100, 500, 1000, 5000, 10000]
N_FEATURES = 20
N_CLASSES = 4
N_ITERATIONS = 3


def make_dmatrix(n_samples: int) -> xgb.DMatrix:
    """Generate synthetic classification data in an XGBoost DMatrix."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=N_FEATURES // 2,
        n_classes=N_CLASSES,
        random_state=42,
    )
    return xgb.DMatrix(X, y)


def make_preds(n_samples: int) -> np.ndarray:
    """Generate random logits in [-5,5] for multi‑class."""
    return np.random.uniform(-5.0, 5.0, size=(n_samples, N_CLASSES))


def time_obj(obj_fn, preds: np.ndarray, dtrain: xgb.DMatrix) -> float:
    """Time a single call to an XGBoost custom objective."""
    t0 = time.time()
    grad, hess = obj_fn(preds, dtrain)
    return time.time() - t0


def main():
    header = (
        f"{'N':<6}  "
        f"{'Orig Poinc':<12}  {'Batch Poinc':<12}  {'Speedup':<8}  "
        f"{'Orig Hyp':<12}  {'Batch Hyp':<12}  {'Speedup':<8}"
    )
    print(header)
    print("-" * len(header))

    for n in DATASET_SIZES:
        dtrain = make_dmatrix(n)
        preds = make_preds(n)

        # Poincaré comparison
        t_orig_p = np.mean(
            [
                time_obj(orig_customgobj, preds, dtrain)
                for _ in range(N_ITERATIONS)
            ]
        )
        t_bat_p = np.mean(
            [
                time_obj(batched_customgobj, preds, dtrain)
                for _ in range(N_ITERATIONS)
            ]
        )

        # Hyperboloid comparison
        t_orig_h = np.mean(
            [
                time_obj(orig_hyperobj, preds, dtrain)
                for _ in range(N_ITERATIONS)
            ]
        )
        t_bat_h = np.mean(
            [
                time_obj(batched_hyperobj, preds, dtrain)
                for _ in range(N_ITERATIONS)
            ]
        )

        print(
            f"{n:<6}  "
            f"{t_orig_p:<12.6f}  {t_bat_p:<12.6f}  {(t_orig_p/t_bat_p):<8.2f}  "
            f"{t_orig_h:<12.6f}  {t_bat_h:<12.6f}  {(t_orig_h/t_bat_h):<8.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
