import numpy as np
import xgboost as xgb
from xgb.poincare import (
    PoincareBall,
)  # Assuming this path is correct for your project
from xgb.hyperboloid1 import Hyperbolic  # Assuming this path is correct

# ————————————————————————————————————————————
# Legacy gradient converters for backward compatibility
# (These seem to be for a different context or older version,
# kept as per your original structure, but not directly used by the new custom_multiclass_obj)
# ————————————————————————————————————————————


def egradrgrad(pred: float, grad: float) -> float:
    """
    Convert a scalar Euclidean gradient to the Riemannian gradient on a 1D Poincaré ball.
    Assumes 'pred' is already on the ball.
    """
    # Add epsilon for stability if pred can be exactly +/-1
    epsilon = 1e-15
    denominator = 1.0 - pred**2
    if abs(denominator) < epsilon:  # Handles pred being very close to +/-1
        # Gradient becomes very small or scaling factor blows up.
        # Depending on convention, could return 0 or a very large/small number.
        # For grad / (lambda^2) = grad * ((1-pred^2)/2)^2, this approaches 0.
        return 0.0
    lam = 2.0 / denominator
    return grad / (lam**2)


def batch_egradrgrad(preds: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of Euclidean gradients to Riemannian gradients on a 1D Poincaré ball.
    Assumes 'preds' are already on the ball.
    """
    epsilon = 1e-15
    one_minus_preds_sq = 1.0 - preds**2
    # Ensure stability: ((1-pred^2)/2)^2
    scaling_factor = (np.maximum(one_minus_preds_sq, epsilon) / 2.0) ** 2
    return grads * scaling_factor


def batch_ehessrhess(
    preds: np.ndarray, grads: np.ndarray, hesses: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Vectorized conversion of Euclidean Hessians to Riemannian Hessians on a 1D Poincaré ball.
    Assumes 'preds' are already on the ball. 'u' is ignored in 1D.
    Formula: (hesses - 2*preds*grads^2) / lambda(preds)^2
    """
    epsilon = 1e-15
    one_minus_preds_sq = 1.0 - preds**2
    scaling_factor = (
        np.maximum(one_minus_preds_sq, epsilon) / 2.0
    ) ** 2  # This is 1/lambda^2
    correction = 2.0 * preds * grads**2
    return (hesses - correction) * scaling_factor


# ————————————————————————————————————————————
# Global manifold instances
# Ensure these are correctly initialized for your project's dimensions/curvature
# ————————————————————————————————————————————
POINCARE = PoincareBall(n=1, k=1)  # For 1D Poincare embeddings per class
HYPERBOLOID = Hyperbolic(
    n=2, k=1
)  # For 1D spatial component on 2D ambient Hyperboloid


# ————————————————————————————————————————————
# Prediction helper
# ————————————————————————————————————————————
def predict(booster: xgb.Booster, X: xgb.DMatrix) -> np.ndarray:
    """Predict class labels from raw booster outputs (logits)."""
    logits = booster.predict(X, output_margin=True)
    if logits.ndim > 1 and logits.shape[1] > 1:  # Multi-class
        return np.argmax(logits, axis=1)
    else:  # Binary or single output regression-like
        # For binary classification, XGBoost with custom obj might still output single logit
        # This part depends on how you interpret single output for binary.
        # Assuming standard binary: > 0 for class 1.
        # If your labels are 0/1, (logits > 0).astype(int) is common.
        # The testing script handles this based on objective.
        # Here, returning argmax is safe if only one column (effectively identity).
        return (
            np.argmax(logits, axis=1)
            if logits.ndim > 1
            else (logits > 0).astype(int)
        )


# ————————————————————————————————————————————
# Core batch operations
# ————————————————————————————————————————————
def batch_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities in a numerically stable way over the last axis."""
    # Clip logits to prevent overflow in exp, especially for large positive values
    clipped_logits = np.clip(
        logits, None, 15.0
    )  # Max value for exp to avoid large numbers
    exps = np.exp(
        clipped_logits - np.max(clipped_logits, axis=1, keepdims=True)
    )
    return exps / np.sum(exps, axis=1, keepdims=True)


# ————————————————————————————————————————————
# Riemannian operations Helpers (Potentially for standalone use or other manifolds)
# The custom_multiclass_obj for Poincare below now calculates directly.
# ————————————————————————————————————————————
def batch_poincare_rgrad_helper(
    x_on_ball: np.ndarray, eucl_grad_wrt_x: np.ndarray
) -> np.ndarray:
    """
    Vectorized Riemannian gradient on a 1D Poincaré ball.
    x_on_ball: points already on the ball (e.g., m = tanh(z)).
    eucl_grad_wrt_x: Euclidean gradient with respect to x_on_ball (e.g., dL/dm).
    Returns: Riemannian gradient w.r.t. x_on_ball: dL/dm_R = (dL/dm_E) / lambda(m)^2
    """
    epsilon = 1e-8  # For stability
    one_minus_x_squared = 1.0 - x_on_ball**2
    safe_one_minus_x_squared = np.maximum(one_minus_x_squared, epsilon)
    # scaling_factor is ((1-m^2)/2)^2
    scaling_factor = (safe_one_minus_x_squared / 2.0) ** 2
    return eucl_grad_wrt_x * scaling_factor


def batch_poincare_rhess_helper(
    x_on_ball: np.ndarray,
    eucl_grad_wrt_x: np.ndarray,
    eucl_hess_wrt_x: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Riemannian Hessian on a 1D Poincaré ball.
    x_on_ball: points already on the ball (e.g., m = tanh(z)).
    eucl_grad_wrt_x: Euclidean gradient with respect to x_on_ball (dL/dm_E).
    eucl_hess_wrt_x: Euclidean Hessian with respect to x_on_ball (d^2L/dm^2_E).
    Returns: Riemannian Hessian w.r.t. x_on_ball.
    The formula for pullback Hessian of L(z) is different from R_Hess(m) of L(m).
    This helper should implement R_Hess(m) of L(m).
    A common formula for R_Hess(m) of L(m) in 1D:
    (d^2L/dm^2_E + (2m / (1-m^2)) * dL/dm_E ) * ((1-m^2)/2)^2
    The term `2*preds*grads^2` in the original `batch_ehessrhess` seems to be from the z-space pullback.
    For clarity, if this is a general m-space to m-space Riemannian Hessian:
    """
    epsilon = 1e-8  # For stability
    one_minus_x_squared = 1.0 - x_on_ball**2
    safe_one_minus_x_squared = np.maximum(one_minus_x_squared, epsilon)

    scaling_factor = (safe_one_minus_x_squared / 2.0) ** 2  # ((1-m^2)/2)^2

    # Christoffel symbol related term for d^2L/dm^2_R from dL/dm_E and d^2L/dm^2_E
    # d^2L/dm^2_R = (d^2L/dm^2_E - Gamma * dL/dm_E) / lambda_conformal^2
    # Gamma_m^m_m = -2m / (1-m^2) for Poincare disk metric g = (4/(1-m^2)^2) dm^2
    # So, term is + (2*m / (1-m^2)) * eucl_grad_wrt_x
    # This uses the Levi-Civita connection.
    christoffel_term_factor = (
        2.0 * x_on_ball
    ) / safe_one_minus_x_squared  # (2m / (1-m^2))

    # Applying scaling factor:
    # (eucl_hess_wrt_x + christoffel_term_factor * eucl_grad_wrt_x) * scaling_factor
    # This is a common form for the Riemannian Hessian components on the manifold.
    # The original `(hesses - 2*preds*grads^2) / lam**2` was for the z-space pullback.
    # If these helpers are truly general m-to-m, this would be more standard:
    riemannian_hess = (
        eucl_hess_wrt_x + christoffel_term_factor * eucl_grad_wrt_x
    ) * scaling_factor

    # However, to keep it closer to your original `batch_ehessrhess` if that formula was intended for m-space:
    # correction_original = 2.0 * x_on_ball * eucl_grad_wrt_x**2
    # riemannian_hess = (eucl_hess_wrt_x - correction_original) * scaling_factor

    # Given the main Poincare logic is now direct, these helpers are less critical.
    # Reverting to a structure similar to your `batch_ehessrhess` for consistency if used:
    correction_term_from_z_pullback_style = 2.0 * x_on_ball * eucl_grad_wrt_x**2
    riemannian_hess = (
        eucl_hess_wrt_x - correction_term_from_z_pullback_style
    ) * scaling_factor

    return np.maximum(riemannian_hess, epsilon)


def batch_hyperboloid_rgrad(
    x_spatial: np.ndarray, eucl_grad_spatial: np.ndarray
) -> np.ndarray:
    """Riemannian gradient on a 1D hyperboloid model (spatial component)."""
    t_coord = np.sqrt(1.0 + x_spatial**2)
    # Ambient points: (t, x_spatial)
    ambient_pts = np.stack([t_coord, x_spatial], axis=1)
    # Euclidean gradients in ambient space (grad_t = 0 for spatial perturbations)
    ambient_eucl_grads = np.stack(
        [np.zeros_like(x_spatial), eucl_grad_spatial], axis=1
    )

    riemannian_grads_ambient = HYPERBOLOID.egrad2rgrad(
        ambient_pts, ambient_eucl_grads
    )
    return riemannian_grads_ambient[:, 1]  # Return only the spatial component


def batch_hyperboloid_rhess(
    x_spatial: np.ndarray,
    eucl_grad_spatial: np.ndarray,
    eucl_hess_spatial: np.ndarray,
) -> np.ndarray:
    """Riemannian Hessian on a 1D hyperboloid model (spatial component)."""
    t_coord = np.sqrt(1.0 + x_spatial**2)
    ambient_pts = np.stack([t_coord, x_spatial], axis=1)
    ambient_eucl_grads = np.stack(
        [np.zeros_like(x_spatial), eucl_grad_spatial], axis=1
    )
    # Euclidean Hessian in ambient space (assuming Hessian w.r.t. spatial coord, others zero)
    # The 'u' vector for ehess2rhess is often the direction of the 2nd derivative.
    # In 1D, this is often taken along the gradient or a basis vector.
    # Your original code used 'hesss' for both 'eucl_hess' and 'u'.
    ambient_eucl_hesses = np.stack(
        [np.zeros_like(x_spatial), eucl_hess_spatial], axis=1
    )

    riemannian_hesses_ambient = HYPERBOLOID.ehess2rhess(
        ambient_pts,
        ambient_eucl_grads,
        ambient_eucl_hesses,
        ambient_eucl_hesses,
    )
    return riemannian_hesses_ambient[:, 1]  # Return only the spatial component


# ————————————————————————————————————————————
# Utility functions
# ————————————————————————————————————————————
def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer class labels."""
    oh = np.zeros((labels.size, num_classes), dtype=float)
    oh[np.arange(labels.size), labels.astype(int)] = 1.0
    return oh


def _prepare_weights(dtrain: xgb.DMatrix) -> np.ndarray:
    """Extract sample weights from DMatrix or default to ones."""
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
    predt are the raw logits (z).
    """
    n_samples, n_classes = predt.shape
    labels = dtrain.get_label().astype(int)
    weights = _prepare_weights(dtrain)

    probs = batch_softmax(predt)
    one_hot_labels = _one_hot(labels, n_classes)

    # Euclidean gradients (dL/dz_E) and Hessians (d^2L/dz^2_E) w.r.t. logits z
    eu_grad = (probs - one_hot_labels) * weights[:, None]
    min_hess_val = 1e-7  # Consistent minimum Hessian value
    eu_hess = np.maximum(
        2.0 * probs * (1.0 - probs) * weights[:, None], min_hess_val
    )

    flat_x_logits = predt.ravel()  # These are the logits z
    flat_eu_grad = eu_grad.ravel()  # This is dL/dz_E
    flat_eu_hess = eu_hess.ravel()  # This is d^2L/dz^2_E

    rgrad = flat_eu_grad  # Default to Euclidean
    rhess = flat_eu_hess  # Default to Euclidean

    if manifold == "poincare":
        epsilon_stability = 1e-8  # Small constant for numerical stability

        # Map logits z to Poincaré disk points m = tanh(z)
        m_on_disk = np.tanh(flat_x_logits)

        # --- Standard Poincaré pullback metric formulas ---
        # The scaling factor is ( (1-m^2)/2 )^2 which is 1/lambda_conformal(m)^2
        # where lambda_conformal(m) = 2 / (1-m^2) is the conformal factor for the metric.

        one_minus_m_squared = 1.0 - m_on_disk**2
        # Ensure the base (1-m^2) is not excessively small before squaring or division.
        safe_one_minus_m_squared = np.maximum(
            one_minus_m_squared, epsilon_stability
        )

        # This is ((1-m^2)/2)^2, which is 1 / (lambda_conformal(m)^2)
        current_scaling_factor = (safe_one_minus_m_squared / 2.0) ** 2

        rgrad = flat_eu_grad * current_scaling_factor

        hess_curvature_correction = 2.0 * m_on_disk * (flat_eu_grad**2)

        rhess = (
            flat_eu_hess - hess_curvature_correction
        ) * current_scaling_factor

        # Ensure Riemannian Hessian is positive for XGBoost stability
        rhess = np.maximum(
            rhess, epsilon_stability
        )  # Using a small positive floor

    elif manifold == "hyperboloid":
        rgrad = batch_hyperboloid_rgrad(flat_x_logits, flat_eu_grad)
        rhess = batch_hyperboloid_rhess(
            flat_x_logits, flat_eu_grad, flat_eu_hess
        )
        rhess = np.maximum(
            rhess, min_hess_val
        )  # Ensure hyperboloid hessian is positive

    # For 'euclid' (or any other unhandled manifold string), it uses the defaults:
    # rgrad = flat_eu_grad, rhess = flat_eu_hess

    return rgrad.reshape(-1, 1), rhess.reshape(-1, 1)


# --- Wrapper functions for specific manifolds ---
def customgobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Poincaré-ball multi-class XGBoost objective."""
    return custom_multiclass_obj(predt, dtrain, manifold="poincare")


def hyperobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Hyperboloid multi-class XGBoost objective."""
    return custom_multiclass_obj(predt, dtrain, manifold="hyperboloid")


def logregobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean (standard softmax) multi-class XGBoost objective."""
    return custom_multiclass_obj(predt, dtrain, manifold="euclid")


# --- Evaluation Metric (already provided) ---
def multiclass_eval(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[str, float]:
    """Accuracy metric (error rate) for multi-class classification."""
    labels = dtrain.get_label().astype(int)
    # predt are raw scores if custom obj is used
    preds_idx = predt.argmax(axis=1)
    return "PyMError", float((preds_idx != labels).mean())


# Overall Performance Summary (Averages per Model Type):

#   Hyperboloid:
#     Avg. Obj Speedup:    780.74x
#     Avg. Train Speedup:  34.47x
#     Avg. Reg. Rounds:    68.6 (20/50 stopped early)
#     Avg. Accuracy (Reg): 0.8034 / (Batch): 0.7914
#     Avg. F1-Macro (Reg): 0.7372 / (Batch): 0.7265

#   Logistic Regression:
#     Avg. Obj Speedup:    87.29x
#     Avg. Train Speedup:  13.03x
#     Avg. Reg. Rounds:    88.2 (10/50 stopped early)
#     Avg. Accuracy (Reg): 0.8036 / (Batch): 0.8036
#     Avg. F1-Macro (Reg): 0.7379 / (Batch): 0.7379

#   Poincare:
#     Avg. Obj Speedup:    245.52x
#     Avg. Train Speedup:  21.05x
#     Avg. Reg. Rounds:    82.9 (10/50 stopped early)
#     Avg. Accuracy (Reg): 0.7698 / (Batch): 0.6723
#     Avg. F1-Macro (Reg): 0.7050 / (Batch): 0.5757
