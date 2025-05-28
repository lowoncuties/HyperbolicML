"""
HyperbolicML: High-Performance Hyperbolic Machine Learning Library

A comprehensive library for machine learning on hyperbolic manifolds,
providing efficient implementations of hyperbolic embeddings, transformations, and algorithms.

Features:
- Hyperbolic machine learning algorithms (including XGBoost integration)
- Poincaré ball and hyperboloid manifold operations
- Fast vectorized batch processing utilities
- Seamless integration with popular ML frameworks
- Support for hyperbolic embeddings and neural networks
"""

__version__ = "1.0.0"


# Initialize error variables
_XGB_ERROR = None
_API_ERROR = None
_MANIFOLDS_ERROR = None

# Import the xgb module to make it available
try:
    from .hyperXGB import xgb

    _XGB_AVAILABLE = True
except ImportError as e:
    _XGB_ERROR = str(e)
    _XGB_AVAILABLE = False
    xgb = None

# Import main API classes
try:
    from .hyperXGB.xgb.hyperutils_batch import (
        HyperbolicXGBoost,
        HyperbolicClassifier,
        HyperbolicRegressor,
    )

    _API_AVAILABLE = True
except ImportError as e:
    _API_ERROR = str(e)
    _API_AVAILABLE = False

# Import manifold utilities
try:
    from .hyperXGB.xgb.poincare import PoincareBall as PoincareManifold
    from .hyperXGB.xgb.hyperboloid1 import Hyperbolic as HyperboloidManifold

    _MANIFOLDS_AVAILABLE = True
except ImportError as e:
    _MANIFOLDS_ERROR = str(e)
    _MANIFOLDS_AVAILABLE = False

# Define what's available for import
__all__ = ["__version__"]

if _XGB_AVAILABLE:
    __all__.append("xgb")

if _API_AVAILABLE:
    __all__.extend(
        ["HyperbolicXGBoost", "HyperbolicClassifier", "HyperbolicRegressor"]
    )

if _MANIFOLDS_AVAILABLE:
    __all__.extend(["PoincareManifold", "HyperboloidManifold"])


def __getattr__(name):
    if name == "xgb" and not _XGB_AVAILABLE:
        raise ImportError(f"Could not import xgb module: {_XGB_ERROR}")
    elif (
        name
        in ["HyperbolicXGBoost", "HyperbolicClassifier", "HyperbolicRegressor"]
        and not _API_AVAILABLE
    ):
        raise ImportError(f"Could not import API classes: {_API_ERROR}")
    elif (
        name in ["PoincareManifold", "HyperboloidManifold"]
        and not _MANIFOLDS_AVAILABLE
    ):
        raise ImportError(
            f"Could not import manifold classes: {_MANIFOLDS_ERROR}"
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
