"""
Hyperbolic Manifolds

Provides access to the underlying manifold implementations for advanced users
who need direct access to geometric operations.
"""

from .xgb.poincare import PoincareBall
from .xgb.hyperboloid_batch import HyperboloidBatch


class PoincareManifold:
    """
    Poincaré Ball Manifold Wrapper

    Provides a clean interface to Poincaré ball operations for advanced users.
    For most use cases, the HyperbolicXGBoost API is recommended instead.
    """

    def __init__(self):
        self._manifold = PoincareBall(n=1)

    def distance(self, x, y):
        """Compute hyperbolic distance between points."""
        return self._manifold.dist(x, y)

    def exp_map(self, x, v):
        """Exponential map from point x in direction v."""
        return self._manifold.expmap(x, v)

    def log_map(self, x, y):
        """Logarithmic map from x to y."""
        return self._manifold.logmap(x, y)

    def parallel_transport(self, x, y, v):
        """Parallel transport vector v from x to y."""
        return self._manifold.ptransp(x, y, v)


class HyperboloidManifold:
    """
    Hyperboloid Manifold Wrapper

    Provides a clean interface to hyperboloid operations for advanced users.
    Uses the optimized batch implementation for performance.
    """

    def __init__(self, n=1, k=1):
        self._manifold = HyperboloidBatch(n=n, k=k)

    def filldata_batch(self, X):
        """Convert spatial coordinates to hyperboloid points."""
        return self._manifold.filldata_batch(X)

    def egrad2rgrad_batch(self, X, G):
        """Convert Euclidean to Riemannian gradients."""
        return self._manifold.egrad2rgrad_batch(X, G)

    def ehess2rhess_batch(self, X, grad, hess, U):
        """Convert Euclidean to Riemannian Hessians."""
        return self._manifold.ehess2rhess_batch(X, grad, hess, U)


__all__ = ["PoincareManifold", "HyperboloidManifold"]
