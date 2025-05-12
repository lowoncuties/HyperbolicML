import numpy as np
import xgboost as xgb
from xgb.poincare import PoincareBall
from xgb.hyperboloid1 import Hyperbolic

# Legacy gradient converters for backward compatibility


def egradrgrad(pred: float, grad: float) -> float:
    """
    Convert a scalar Euclidean gradient to the Riemannian gradient on a 1D Poincaré ball.

    Parameters
    ----------
    pred : float
        A point on the Poincaré ball (|pred| < 1).
    grad : float
        The Euclidean gradient at `pred`.

    Returns
    -------
    rgrad : float
        The Riemannian gradient: grad / lambda(pred)^2, where lambda = 2/(1 - pred^2).
    """
    lam = 2.0 / (1.0 - pred**2)
    return grad / (lam**2)


def batch_egradrgrad(preds: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of Euclidean gradients to Riemannian gradients on a 1D Poincaré ball.

    Parameters
    ----------
    preds : np.ndarray, shape (N,)
        Points on the ball (|preds[i]| < 1).
    grads : np.ndarray, shape (N,)
        Corresponding Euclidean gradients.

    Returns
    -------
    rgrads : np.ndarray, shape (N,)
        Riemannian gradients: grads / lambda(preds)^2.
    """
    lam = 2.0 / (1.0 - preds**2)
    return grads / lam**2


def batch_ehessrhess(
    preds: np.ndarray, grads: np.ndarray, hesses: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Vectorized conversion of Euclidean Hessians to Riemannian Hessians on a 1D Poincaré ball.

    Parameters
    ----------
    preds : np.ndarray, shape (N,)
        Points on the ball.
    grads : np.ndarray, shape (N,)
        Euclidean gradients at `preds`.
    hesses : np.ndarray, shape (N,)
        Euclidean Hessians at `preds`.
    u : np.ndarray, shape (N,)
        Tangent vector (ignored in this 1D simplification).

    Returns
    -------
    rhesses : np.ndarray, shape (N,)
        Riemannian Hessians: (hesses - 2*preds*grads^2) / lambda(preds)^2.
    """
    lam = 2.0 / (1.0 - preds**2)
    correction = 2.0 * preds * grads**2
    return (hesses - correction) / lam**2


# ————————————————————————————————————————————
# Global manifold instances
# ————————————————————————————————————————————
POINCARE = PoincareBall(n=1, k=1)
HYPERBOLOID = Hyperbolic(n=2, k=1)

# ————————————————————————————————————————————
# Prediction helper
# ————————————————————————————————————————————


def predict(booster: xgb.Booster, X: xgb.DMatrix) -> np.ndarray:
    """
    Predict class labels from raw booster outputs.

    Parameters
    ----------
    booster : xgb.Booster
        Trained XGBoost model.
    X : xgb.DMatrix
        Feature matrix in XGBoost DMatrix format.

    Returns
    -------
    classes : np.ndarray, shape (n_samples,)
        Predicted class indices (argmax of raw scores).
    """
    logits = booster.predict(X, output_margin=True)
    return np.argmax(logits, axis=1)


# ————————————————————————————————————————————
# Core batch operations
# ————————————————————————————————————————————


def batch_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities in a numerically stable way over the last axis.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
        Raw scores for N samples and C classes.

    Returns
    -------
    probs : np.ndarray, shape (N, C)
        Softmax probabilities.
    """
    clipped = np.clip(logits, None, 15.0)
    exps = np.exp(clipped - clipped.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def batch_poincare_rgrad(x: np.ndarray, eucl_grad: np.ndarray) -> np.ndarray:
    """
    Vectorized Riemannian gradient on a 1D Poincaré ball.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Points on the ball.
    eucl_grad : np.ndarray, shape (N,)
        Euclidean gradients at each point.

    Returns
    -------
    rgrad : np.ndarray, shape (N,)
        Riemannian gradients: eucl_grad / lambda(x)^2.
    """
    lam = 2.0 / (1.0 - x**2)
    return eucl_grad / lam**2


def batch_poincare_rhess(
    x: np.ndarray, eucl_grad: np.ndarray, eucl_hess: np.ndarray
) -> np.ndarray:
    """
    Vectorized Riemannian Hessian on a 1D Poincaré ball.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Points on the ball.
    eucl_grad : np.ndarray, shape (N,)
        Euclidean gradients.
    eucl_hess : np.ndarray, shape (N,)
        Euclidean Hessians.

    Returns
    -------
    rhess : np.ndarray, shape (N,)
        Riemannian Hessians: (eucl_hess - 2 x eucl_grad^2) / lambda(x)^2.
    """
    lam = 2.0 / (1.0 - x**2)
    correction = 2.0 * x * eucl_grad**2
    return (eucl_hess - correction) / lam**2


def batch_hyperboloid_rgrad(x: np.ndarray, eucl_grad: np.ndarray) -> np.ndarray:
    """
    Vectorized Riemannian gradient on a 1D hyperboloid model (spatial component).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Spatial coordinates of hyperboloid points.
    eucl_grad : np.ndarray, shape (N,)
        Euclidean gradients (spatial part).

    Returns
    -------
    rgrad : np.ndarray, shape (N,)
        Spatial components of Riemannian gradients.
    """
    t = np.sqrt(1.0 + x**2)
    pts = np.stack([t, x], axis=1)
    grads = np.stack([np.zeros_like(x), eucl_grad], axis=1)
    rgrads = HYPERBOLOID.egrad2rgrad(pts, grads)
    return rgrads[:, 1]


def batch_hyperboloid_rhess(
    x: np.ndarray, eucl_grad: np.ndarray, eucl_hess: np.ndarray
) -> np.ndarray:
    """
    Vectorized Riemannian Hessian on a 1D hyperboloid model (spatial component).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Spatial coordinates of hyperboloid points.
    eucl_grad : np.ndarray, shape (N,)
        Euclidean gradients (spatial part).
    eucl_hess : np.ndarray, shape (N,)
        Euclidean Hessians (spatial part).

    Returns
    -------
    rhess : np.ndarray, shape (N,)
        Spatial components of Riemannian Hessians.
    """
    t = np.sqrt(1.0 + x**2)
    pts = np.stack([t, x], axis=1)
    grads = np.stack([np.zeros_like(x), eucl_grad], axis=1)
    hesss = np.stack([np.zeros_like(x), eucl_hess], axis=1)
    rhesss = HYPERBOLOID.ehess2rhess(pts, grads, hesss, hesss)
    return rhesss[:, 1]


# ————————————————————————————————————————————
# Utility functions
# ————————————————————————————————————————————


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode integer class labels.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Integer labels in [0, num_classes).
    num_classes : int
        Number of classes.

    Returns
    -------
    one_hot : np.ndarray, shape (N, num_classes)
        One-hot encoded label matrix.
    """
    oh = np.zeros((labels.size, num_classes), dtype=float)
    oh[np.arange(labels.size), labels.astype(int)] = 1.0
    return oh


def _prepare_weights(dtrain: xgb.DMatrix) -> np.ndarray:
    """
    Extract sample weights from DMatrix or default to ones.

    Parameters
    ----------
    dtrain : xgb.DMatrix
        XGBoost DMatrix with optional weights.

    Returns
    -------
    weights : np.ndarray, shape (n_samples,)
        Sample weights.
    """
    w = dtrain.get_weight()
    return w if w.size else np.ones(dtrain.num_row(), dtype=float)


# ————————————————————————————————————————————
# Custom objectives and metrics
# ————————————————————————————————————————————


def custom_multiclass_obj(
    predt: np.ndarray, dtrain: xgb.DMatrix, manifold: str = "poincare"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-class softprob objective with optional Riemannian corrections.

    Parameters
    ----------
    predt : np.ndarray, shape (n_samples, n_classes)
        Raw leaf-weight scores (logits) from XGBoost.
    dtrain : xgb.DMatrix
        DMatrix containing true labels and optional weights.
    manifold : str, optional
        Which geometry to use: 'poincare', 'hyperboloid', or 'euclid'.

    Returns
    -------
    grad : np.ndarray, shape (n_samples*n_classes, 1)
        Flattened gradient for XGBoost.
    hess : np.ndarray, shape (n_samples*n_classes, 1)
        Flattened Hessian for XGBoost.
    """
    n, k = predt.shape
    labels = dtrain.get_label().astype(int)
    weights = _prepare_weights(dtrain)

    probs = batch_softmax(predt)
    one_hot = _one_hot(labels, k)

    eu_grad = (probs - one_hot) * weights[:, None]
    eu_hess = np.maximum(2.0 * probs * (1 - probs) * weights[:, None], 1e-6)

    flat_x = predt.ravel()
    flat_grad = eu_grad.ravel()
    flat_hess = eu_hess.ravel()

    if manifold == "poincare":
        rgrad = batch_poincare_rgrad(flat_x, flat_grad)
        rhess = batch_poincare_rhess(flat_x, flat_grad, flat_hess)
    elif manifold == "hyperboloid":
        rgrad = batch_hyperboloid_rgrad(flat_x, flat_grad)
        rhess = batch_hyperboloid_rhess(flat_x, flat_grad, flat_hess)
    else:
        rgrad, rhess = flat_grad, flat_hess

    return rgrad.reshape(-1, 1), rhess.reshape(-1, 1)


def customgobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Poincaré-ball multi-class XGBoost objective.
    """
    return custom_multiclass_obj(predt, dtrain, manifold="poincare")


def hyperobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hyperboloid multi-class XGBoost objective.
    """
    return custom_multiclass_obj(predt, dtrain, manifold="hyperboloid")


def logregobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean (standard softmax) multi-class XGBoost objective.
    """
    return custom_multiclass_obj(predt, dtrain, manifold="euclid")


def multiclass_eval(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[str, float]:
    """
    Accuracy metric (error rate) for multi-class classification.

    Returns
    -------
    name : str
        Metric name, 'PyMError'.
    result : float
        Error rate = mean(pred != true).
    """
    labels = dtrain.get_label().astype(int)
    preds = predt.argmax(axis=1)
    return "PyMError", float((preds != labels).mean())
