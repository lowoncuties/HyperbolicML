import numpy as np
import xgboost as xgb
from .poincare import PoincareBall
from .hyperboloid1 import Hyperbolic

# kRows, kClasses = 280, 4  # for gaussion data

# # https://www.kaggle.com/code/kevalm/xgboost-implementation-on-iris-dataset-python
# # https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
# # https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html#sphx-glr-python-examples-custom-softmax-py


# predict the results
def predict(booster: xgb.Booster, X):
    """A customized prediction function that converts raw prediction to
    target class.

    """
    # Output margin means we want to obtain the raw prediction obtained from
    # tree leaf weight.
    predt = booster.predict(X, output_margin=True)
    out = np.zeros(predt.shape[0])
    for r in range(predt.shape[0]):
        # the class with maximum prob (not strictly prob as it haven't gone
        # through softmax yet so it doesn't sum to 1, but result is the same
        # for argmax).
        i = np.argmax(predt[r])
        out[r] = i
    return out


# for poincare ball
def egradrgrad(preds, grad):
    caredisk = PoincareBall(1)
    return caredisk.euclidean_to_riemannian_gradient(preds, grad)


# for poincare ball
def ehessrhess(preds, grad, hess, u):
    caredisk = PoincareBall(1)
    return caredisk.euclidean_to_riemannian_hessian(preds, grad, hess, u)


# for hyperbolid
def hyperegradrgrad(preds, grad):
    n = grad.shape[0]
    hyperbo = Hyperbolic(n, 1)
    pred = hyperbo.filldata(preds)
    grads = hyperbo.filldata(grad)
    fgrad = hyperbo.egrad2rgrad(pred, grads)
    return fgrad[:, 1]


# for hyperbolid
def hyperehessrhess(preds, grad, hess, u):
    n = grad.shape[0]
    hyperbo = Hyperbolic(n, 1)
    pred = hyperbo.filldata(preds)
    grads = hyperbo.filldata(grad)
    ehess = np.array(hess)
    ehess = hyperbo.filldata(ehess)
    fhess = hyperbo.ehess2rhess(pred, grads, ehess, ehess)
    return fhess[:, 1]


def softmax(x):
    """Softmax function with x as input vector."""
    x = np.where(x > 15, 15, x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-6)


def customgobj(predt: np.ndarray, data: xgb.DMatrix):
    """Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    """
    kRows, kClasses = predt.shape
    labels = data.get_label()
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            rg = egradrgrad(predt[r, c], g)
            rh = ehessrhess(predt[r, c], g, h, h)
            grad[r, c] = rg
            hess[r, c] = rh

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess


def logregobj(predt: np.ndarray, data: xgb.DMatrix):
    """Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    """
    kRows, kClasses = predt.shape
    labels = data.get_label()
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess


def accMetric(predt: np.ndarray, dtrain: xgb.DMatrix):
    ytrue = dtrain.get_label()
    # Like custom objective, the predt is untransformed leaf weight when custom objective
    # is provided.

    # With the use of `custom_metric` parameter in train function, custom metric receives
    # raw input only when custom objective is also being used.  Otherwise custom metric
    # will receive transformed prediction.
    # ---added by KONG, AS eval data is not the same as train, we can not assert as this result
    # assert predt.shape == (kRows, kClasses)
    row, column = predt.shape
    out = np.zeros(row)
    for r in range(predt.shape[0]):
        i = np.argmax(predt[r])
        out[r] = i

    assert ytrue.shape == out.shape

    errors = np.zeros(row)
    errors[ytrue != out] = 1.0
    return "PyMError", np.sum(errors) / row


def hyperobj(predt: np.ndarray, data: xgb.DMatrix):
    """Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    """
    kRows, kClasses = predt.shape
    labels = data.get_label()
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            rg = hyperegradrgrad(predt[r, c], g)
            rh = hyperehessrhess(predt[r, c], g, h, h)
            grad[r, c] = rg
            hess[r, c] = rh

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess


# this is for multi-label
def logregobj_binary(preds, dtrain):
    scale_pos_weight = 3.0
    # labels = dtrain.get_label()
    labels = dtrain
    weights = np.where(labels == 1.0, scale_pos_weight, 1.0)
    # preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad * weights, hess * weights


# this is for multi-label--poincare
def customobj_binary(preds, dtrain):
    scale_pos_weight = 3.0
    # labels = dtrain.get_label()
    labels = dtrain
    weights = np.where(labels == 1.0, scale_pos_weight, 1.0)
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    rg = egradrgrad(preds, grad)
    rh = ehessrhess(preds, grad, hess, hess)

    return rg * weights, rh * weights


# this is for multi-label--poincare
def testregobj(predt: np.ndarray, data: xgb.DMatrix):
    """Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    """
    kRows, kClasses = data.shape
    labels = data
    # if data.get_weight().size == 0:
    #     # Use 1 as weight if we don't have custom weight.
    #     weights = np.ones((kRows, 1), dtype=float)
    # else:
    #     weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    # assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g
            h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess


# this is for multi-label--poincare
def hyperobj_binary(preds, dtrain):
    scale_pos_weight = 3.0
    # labels = dtrain.get_label()
    labels = dtrain
    weights = np.where(labels == 1.0, scale_pos_weight, 1.0)
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    rg = hyperegradrgrad(preds, grad)
    rh = hyperehessrhess(preds, grad, hess, hess)

    return rg * weights, rh * weights
