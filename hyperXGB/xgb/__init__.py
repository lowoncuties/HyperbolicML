"""
XGB Core: Hyperbolic XGBoost Implementation

Core hyperbolic XGBoost implementation with vectorized manifold operations
for the HyperbolicML library. Provides optimized gradient and Hessian
computations on hyperbolic manifolds.
"""

import sys

# Add the current package to sys.modules as 'xgb' so imports work
if "xgb" not in sys.modules:
    sys.modules["xgb"] = sys.modules[__name__]

# Import all public functions and classes
from .hyperutils_batch import (
    batch_softmax,
    batch_poincare_rgrad_helper,
    batch_poincare_rhess,
    batch_hyperboloid_rgrad_vectorized,
    batch_hyperboloid_rhess_vectorized,
    custom_multiclass_obj,
    customgobj,
    hyperobj,
    logregobj,
    multiclass_eval,
    _one_hot,
    _prepare_weights,
)

from .hyperboloid_batch import HyperboloidBatch
from .poincare import PoincareBall

__all__ = [
    # Batch processing functions
    "batch_softmax",
    "batch_poincare_rgrad_helper",
    "batch_poincare_rhess",
    "batch_hyperboloid_rgrad_vectorized",
    "batch_hyperboloid_rhess_vectorized",
    # Objective functions
    "custom_multiclass_obj",
    "customgobj",
    "hyperobj",
    "logregobj",
    "multiclass_eval",
    # Utility functions
    "_one_hot",
    "_prepare_weights",
    # Classes
    "HyperboloidBatch",
    "PoincareBall",
]
