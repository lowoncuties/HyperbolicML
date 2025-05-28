"""
HyperXGB: XGBoost Integration for HyperbolicML

XGBoost integration module providing hyperbolic machine learning capabilities
with gradient boosting on Poincaré ball and hyperboloid manifolds.
"""

# Import xgb submodule to make it available
from . import xgb

# Import main functions for convenience
from .xgb.hyperutils_batch import (
    egradrgrad,
    batch_egradrgrad,
    batch_ehessrhess,
    predict,
    batch_softmax,
    batch_poincare_rgrad_helper,
    batch_poincare_rhess,
    batch_hyperboloid_rgrad,
    batch_hyperboloid_rhess,
    custom_multiclass_obj,
    customgobj,
    hyperobj,
    logregobj,
    multiclass_eval,
    _one_hot,
    _prepare_weights,
)
from .xgb.hyperboloid_batch import HyperboloidBatch

__all__ = [
    "xgb",
    "egradrgrad",
    "batch_egradrgrad",
    "batch_ehessrhess",
    "predict",
    "batch_softmax",
    "batch_poincare_rgrad_helper",
    "batch_poincare_rhess",
    "batch_hyperboloid_rgrad",
    "batch_hyperboloid_rhess",
    "custom_multiclass_obj",
    "customgobj",
    "hyperobj",
    "logregobj",
    "multiclass_eval",
    "_one_hot",
    "_prepare_weights",
    "HyperboloidBatch",
]
