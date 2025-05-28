import numpy as np
import xgboost as xgb
from .poincare import (
    PoincareBall,
)  # Assuming this path is correct for your project
from .hyperboloid1 import Hyperbolic  # Assuming this path is correct
from .hyperboloid_batch import (
    batch_hyperboloid_rgrad_vectorized,
    batch_hyperboloid_rhess_vectorized,
    batch_poincare_rgrad_vectorized,
    batch_poincare_rhess_vectorized,
)

# ————————————————————————————————————————————
# Legacy gradient converters for backward compatibility
# (These seem to be for a different context or older version,
# kept as per your original structure, but not directly used by the new custom_multiclass_obj)
# ————————————————————————————————————————————


def egradrgrad(pred: float, grad: float) -> float:
    """
    Convert Euclidean to Riemannian gradient on 1D Poincaré ball.

    Applies conformal rescaling: ∇ᴿf = ∇ᴱf / λ(x)²
    where λ(x) = 2 / (1 - x²) is the conformal factor.

    Args:
        pred: Point on Poincaré ball [-1, 1]
        grad: Euclidean gradient at the point

    Returns:
        Riemannian gradient on the manifold
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
    Vectorized conversion of Euclidean to Riemannian gradients on 1D Poincaré ball.

    Computes: ∇ᴿf = ∇ᴱf × ((1 - x²)/2)² for all points simultaneously.
    This exactly matches PoincareBall.euclidean_to_riemannian_gradient.

    Args:
        preds: Points on Poincaré ball, shape (n,)
        grads: Euclidean gradients, shape (n,)

    Returns:
        Riemannian gradients, shape (n,)
    """
    return batch_poincare_rgrad_vectorized(preds, grads)


def batch_ehessrhess(
    preds: np.ndarray, grads: np.ndarray, hesses: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Vectorized conversion of Euclidean to Riemannian Hessians on 1D Poincaré ball.

    Uses Koszul formula: ∇²ᴿf = (∇²ᴱf - Γˣₓₓ(∇ᴱf)²) / λ(x)²
    where Γˣₓₓ = 2x/(1-x²) is the Christoffel symbol and λ(x) = 2/(1-x²).
    Exactly matches PoincareBall.euclidean_to_riemannian_hessian.

    Args:
        preds: Points on the Poincaré ball, shape (n,)
        grads: Euclidean gradients, shape (n,)
        hesses: Euclidean Hessians, shape (n,)
        u: Tangent vectors (used in Koszul formula), shape (n,)

    Returns:
        Riemannian Hessians computed using exact Koszul formula, shape (n,)
    """
    return batch_poincare_rhess_vectorized(preds, grads, hesses)


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
    """
    Predict class labels from raw booster outputs.

    For multi-class: ŷ = argmax(softmax(z_i)) where z are logits.
    For binary: ŷ = I[z > 0] where I is the indicator function.

    Args:
        booster: Trained XGBoost model
        X: Input features as DMatrix

    Returns:
        Predicted class labels as integer array
    """
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
    """
    Compute numerically stable softmax probabilities.

    Formula: p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
    where z are the input logits.

    Args:
        logits: Raw logits, shape (n_samples, n_classes)

    Returns:
        Softmax probabilities, shape (n_samples, n_classes)
    """
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
    Vectorized Riemannian gradient computation on 1D Poincaré ball.

    Computes: ∇ᴿf = ∇ᴱf × ((1 - x²)/2)² where the scaling factor
    is 1/λ(x)² with λ(x) = 2/(1-x²) being the conformal factor.

    Args:
        x_on_ball: Points already on the ball (e.g., m = tanh(z)), shape (n,)
        eucl_grad_wrt_x: Euclidean gradient w.r.t. x_on_ball (e.g., dL/dm), shape (n,)

    Returns:
        Riemannian gradient w.r.t. x_on_ball: dL/dm_R = (dL/dm_E) / λ(m)², shape (n,)
    """
    epsilon = 1e-8  # For stability
    one_minus_x_squared = 1.0 - x_on_ball**2
    safe_one_minus_x_squared = np.maximum(one_minus_x_squared, epsilon)
    # scaling_factor is ((1-m^2)/2)^2
    scaling_factor = (safe_one_minus_x_squared / 2.0) ** 2
    return eucl_grad_wrt_x * scaling_factor


def batch_poincare_rhess(
    x_on_ball: np.ndarray,
    eucl_grad_wrt_x: np.ndarray,
    eucl_hess_wrt_x: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Riemannian Hessian computation on 1D Poincaré ball.

    Uses exact Koszul formula: ∇²ᴿf = (∇²ᴱf - Γˣₓₓ(∇ᴱf)²) / λ(x)²
    where Γˣₓₓ = 2x/(1-x²) is the Christoffel symbol.
    Exactly matches PoincareBall.euclidean_to_riemannian_hessian.

    Args:
        x_on_ball: Points already on the ball (e.g., m = tanh(z)), shape (n,)
        eucl_grad_wrt_x: Euclidean gradient w.r.t. x_on_ball, shape (n,)
        eucl_hess_wrt_x: Euclidean Hessian w.r.t. x_on_ball, shape (n,)

    Returns:
        Riemannian Hessian w.r.t. x_on_ball computed using Koszul formula, shape (n,)
    """
    # For the Poincaré objective, the tangent vector u is typically the same as the Hessian
    # This matches how the original ehessrhess is called: ehessrhess(pred, grad, hess, hess)
    tangent_vector = eucl_hess_wrt_x

    return batch_ehessrhess(
        x_on_ball, eucl_grad_wrt_x, eucl_hess_wrt_x, tangent_vector
    )


def batch_hyperboloid_rgrad(preds: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """
    Vectorized Riemannian gradient conversion for hyperboloid manifold.

    Converts Euclidean gradients to Riemannian gradients using the
    hyperboloid manifold structure with Minkowski inner product.

    Args:
        preds: Spatial coordinates on hyperboloid, shape (n,)
        grads: Euclidean gradients, shape (n,)

    Returns:
        Riemannian gradients on hyperboloid, shape (n,)
    """
    return batch_hyperboloid_rgrad_vectorized(preds, grads)


def batch_hyperboloid_rhess(
    preds: np.ndarray, grads: np.ndarray, hess: np.ndarray
) -> np.ndarray:
    """
    Vectorized Riemannian Hessian conversion for hyperboloid manifold.

    Converts Euclidean Hessians to Riemannian Hessians using the
    hyperboloid manifold connection and Minkowski metric structure.

    Args:
        preds: Spatial coordinates on hyperboloid, shape (n,)
        grads: Euclidean gradients, shape (n,)
        hess: Euclidean Hessians, shape (n,)

    Returns:
        Riemannian Hessians on hyperboloid, shape (n,)
    """
    return batch_hyperboloid_rhess_vectorized(preds, grads, hess)


# ————————————————————————————————————————————
# Utility functions
# ————————————————————————————————————————————
def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Computation: e_i[j] = 1 if j == label_i else 0

    Args:
        labels: Integer class labels, shape (n_samples,)
        num_classes: Total number of classes

    Returns:
        One-hot encoded labels, shape (n_samples, num_classes)
    """
    oh = np.zeros((labels.size, num_classes), dtype=float)
    oh[np.arange(labels.size), labels.astype(int)] = 1.0
    return oh


def _prepare_weights(dtrain: xgb.DMatrix) -> np.ndarray:
    """
    Extract sample weights from XGBoost DMatrix or return uniform weights.

    Args:
        dtrain: XGBoost DMatrix containing training data and labels

    Returns:
        Sample weights array, shape (n_samples,)
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
    Unified multi-class objective with manifold-specific Riemannian corrections.

    Computes gradients and Hessians for softmax cross-entropy loss with
    optional Riemannian metric corrections for hyperbolic manifolds.
    Base loss: L = -Σ_i y_i log(p_i) where p_i = softmax(z_i)

    Args:
        predt: Raw logits, shape (n_samples, n_classes)
        dtrain: XGBoost DMatrix with labels and weights
        manifold: Manifold type ("poincare", "hyperboloid", or "euclid")

    Returns:
        Tuple of (gradients, hessians) with Riemannian corrections applied
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
    """
    Poincaré-ball multi-class XGBoost objective with vectorized conversions.

    Computes softmax cross-entropy loss with Poincaré ball Riemannian corrections:
    - Maps logits z to Poincaré disk: m = tanh(z)
    - Applies conformal factor: λ(m) = 2/(1-m²)
    - Converts gradients: ∇ᴿf = ∇ᴱf / λ(m)²

    Args:
        predt: Raw logits, shape (n_samples, n_classes)
        dtrain: XGBoost DMatrix with labels and weights

    Returns:
        Tuple of (Riemannian gradients, Riemannian Hessians)
    """
    n_samples, n_classes = predt.shape
    labels = dtrain.get_label().astype(int)
    weights = _prepare_weights(dtrain)

    probs = batch_softmax(predt)
    one_hot_labels = _one_hot(labels, n_classes)

    # Euclidean gradients and Hessians w.r.t. logits
    eu_grad = (probs - one_hot_labels) * weights[:, None]
    min_hess_val = 1e-7  # Consistent minimum Hessian value
    eu_hess = np.maximum(
        2.0 * probs * (1.0 - probs) * weights[:, None], min_hess_val
    )

    # Use vectorized Poincaré conversions for speed
    flat_preds = predt.ravel()
    flat_eu_grad = eu_grad.ravel()
    flat_eu_hess = eu_hess.ravel()

    # Convert using vectorized functions
    rgrad = batch_egradrgrad(flat_preds, flat_eu_grad)

    # For hess, we need the u parameter
    u = np.tanh(flat_preds)
    rhess = batch_ehessrhess(flat_preds, flat_eu_grad, flat_eu_hess, u)

    return rgrad.reshape(-1, 1), rhess.reshape(-1, 1)


def hyperobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hyperboloid multi-class XGBoost objective.

    Computes softmax cross-entropy loss with hyperboloid Riemannian corrections
    using the Minkowski inner product and hyperboloid embedding.

    Args:
        predt: Raw logits, shape (n_samples, n_classes)
        dtrain: XGBoost DMatrix with labels and weights

    Returns:
        Tuple of (Riemannian gradients, Riemannian Hessians)
    """
    return custom_multiclass_obj(predt, dtrain, manifold="hyperboloid")


def logregobj(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean (standard softmax) multi-class XGBoost objective.

    Computes standard softmax cross-entropy loss without manifold corrections:
    L = -Σ_i y_i log(softmax(z_i))

    Args:
        predt: Raw logits, shape (n_samples, n_classes)
        dtrain: XGBoost DMatrix with labels and weights

    Returns:
        Tuple of (Euclidean gradients, Euclidean Hessians)
    """
    return custom_multiclass_obj(predt, dtrain, manifold="euclid")


# --- Evaluation Metric (already provided) ---
def multiclass_eval(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[str, float]:
    """
    Accuracy metric (error rate) for multi-class classification.

    Computes: error = (1/n) Σ_i I[argmax(p_i) ≠ y_i]
    where I[·] is the indicator function

    Args:
        predt: Raw logits or probabilities, shape (n_samples, n_classes)
        dtrain: XGBoost DMatrix with true labels

    Returns:
        Tuple of (metric_name, error_rate)
    """
    labels = dtrain.get_label().astype(int)
    # predt are raw scores if custom obj is used
    preds_idx = predt.argmax(axis=1)
    return "PyMError", float((preds_idx != labels).mean())


def _handle_binary_classification(predt: np.ndarray) -> np.ndarray:
    """
    Handle binary classification by ensuring predictions are 2D.

    XGBoost sends 1D arrays for binary classification but our batch functions
    expect 2D arrays for multiclass. This function converts binary to multiclass format.
    """
    if predt.ndim == 1:
        # Binary classification: convert to 2-class format
        # XGBoost binary predictions are logits for P(class=1)
        # Convert to [logit_class_0, logit_class_1] format
        n_samples = len(predt)
        multiclass_predt = np.zeros((n_samples, 2))
        multiclass_predt[:, 0] = -predt  # logit for class 0
        multiclass_predt[:, 1] = predt  # logit for class 1
        return multiclass_predt
    return predt


def custom_multiclass_obj_with_binary_support(
    predt: np.ndarray, dtrain, manifold: str = "poincare"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified objective with binary classification support.
    Handles both DMatrix and numpy array inputs for compatibility with sklearn API.
    """
    # Handle binary classification
    predt_2d = _handle_binary_classification(predt)

    # Extract labels and weights depending on input type
    if hasattr(dtrain, "get_label"):
        # DMatrix object (low-level XGBoost API)
        labels = dtrain.get_label().astype(int)
        weights = _prepare_weights(dtrain)
    else:
        # Numpy array (sklearn API)
        labels = dtrain.astype(int)
        weights = np.ones(len(labels))  # Default weights

    n_samples, n_classes = predt_2d.shape

    probs = batch_softmax(predt_2d)
    one_hot_labels = _one_hot(labels, n_classes)

    # Euclidean gradients (dL/dz_E) and Hessians (d^2L/dz^2_E) w.r.t. logits z
    eu_grad = (probs - one_hot_labels) * weights[:, None]
    min_hess_val = 1e-7  # Consistent minimum Hessian value
    eu_hess = np.maximum(
        2.0 * probs * (1.0 - probs) * weights[:, None], min_hess_val
    )

    flat_x_logits = predt_2d.ravel()  # These are the logits z
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

    grad_reshaped = rgrad.reshape(n_samples, n_classes)
    hess_reshaped = rhess.reshape(n_samples, n_classes)

    # If original was 1D (binary), return 1D
    if predt.ndim == 1:
        # For binary, XGBoost expects 1D gradients/hessians
        # Take the difference: grad_class1 - grad_class0
        grad_binary = grad_reshaped[:, 1] - grad_reshaped[:, 0]
        hess_binary = (
            hess_reshaped[:, 1] + hess_reshaped[:, 0]
        )  # Sum for binary
        return grad_binary, hess_binary

    return grad_reshaped.ravel(), hess_reshaped.ravel()


# Update wrapper functions to use binary support
def customgobj_with_binary_support(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Poincaré objective with binary classification support."""
    return custom_multiclass_obj_with_binary_support(predt, dtrain, "poincare")


def hyperobj_with_binary_support(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Hyperboloid objective with binary classification support."""
    return custom_multiclass_obj_with_binary_support(
        predt, dtrain, "hyperboloid"
    )


def logregobj_with_binary_support(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean objective with binary classification support."""
    return custom_multiclass_obj_with_binary_support(predt, dtrain, "euclid")
