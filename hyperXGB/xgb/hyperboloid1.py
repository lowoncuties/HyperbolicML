import numpy as np
# https://github.com/mctorch/mctorch

def multiprod(A, B):
    # Added just to be parallel to manopt/pymanopt implemenetation
    return np.matmul(A, B)


def multitransp(A):
    # First check if we have been given just one matrix
    return np.transpose(A)

class Manifold(object):
    """
    Base class for manifold constraints

    All manifolds should subclass this class:

        import torch.nn as nn
        class XManifold(nn.Manifold):
            def __init__(self, args):
                super(Manifold, self).__init__()

            def dim(self):
                ...

    All functions map to corresponding functions in
    Manopt `<http://www.manopt.org>` and its python dervivation
    pymanopt `<https://github.com/pymanopt/pymanopt>`

    All functions should be converted to torch counterparts.

    """

    def __init__(self):
        self._dim = None
        self._size = None

    def __str__(self):
        """
        Name of the manifold
        """

    def dim(self):
        """
        Dimension of the manifold
        """
        return self._dim

    def size(self):
        """
        Returns tuple denoting size of a point on manifold
        """
        return self._size

    def proj(self, X, G):
        """
        Project into the tangent space. Usually the same as egrad2rgrad
        """
        raise NotImplementedError

    def egrad2rgrad(self, X, G):
        """
        A mapping from the Euclidean gradient G into the tangent space
        to the manifold at X. For embedded manifolds, this is simply the
        projection of G on the tangent space at X.
        """
        return self.proj(X, G)

    def ehess2rhess(self, X, egrad, Hess, U):
        """
        Convert Euclidean into Riemannian Hessian.
        """
        raise NotImplementedError

    def randvec(self, X):
        """
        Returns a random, unit norm vector in the tangent space at X.
        """
        raise NotImplementedError


class Hyperbolic(Manifold):
    """
    Class for Hyperbolic manifold with shape (k x N) or N

    With k > 1 it applies product of k Hyperbolas
    """

    def __init__(self, n, k=1):
        if n < 1:
            raise ValueError("Need n >= 2 Value supplied was n = {}".format(n))
        if k < 1:
            raise ValueError("Need k >= 1 Value supplied was k = {}".format(k))

        super(Manifold, self).__init__()
        # Set the dimensions of the Hyperbolic manifold
        self._n = n
        self._k = k
        self.eps = 1e-7
        self.name = 'Hyperboloid'
        self.min_norm = 1e-15
        self.max_norm = 1e6
        # Set dimension #TODO: confirm
        self._dim = self._k * (self._n)
        if k == 1:
            self._size = n
        else:
            self._size = k*n

    def __str__(self):
        if self._k == 1:
            return "Hyperbolic manifold ({})".format(self._n)
        elif self._k >= 2:
            return "Product Hyperbolic manifold ({})^{}".format(
                self._n, self._k)

    def rand(self):
        """
        Generate random Hyperbolic point in range (-0.001, 0.001)
        """
        u_range = (-0.001, 0.001)
        if self._k == 1:
            X = np.random.randn(self._n)# * (u_range[1] - u_range[0]) + u_range[0]
            X[0] = np.sqrt(1 + np.sum(X[1:]**2))
            return X

        X = np.random.randn(self._k, self._n)# * (u_range[1] - u_range[0]) + u_range[0]
        X[:, 0] = np.sqrt(1 + np.sum(X[:, 1:]**2, axis=1))
        return X

    def _lorentz_scalar_product(self, u, v):
        if u.shape == v.shape:
            if len(v.shape) == 1:
                val = np.sum(u*v) - 2*u[0]*v[0]
                return val
            elif len(v.shape) == 2:
                val = np.sum(u*v, axis=1) - 2*u[:, 0]*v[:, 0]
                return val
            raise ValueError("u, v can not be {}-dimensional".format(len(v.shape)))
        raise ValueError("u,v shape should be same")

    def proj(self, X, U):
        result = self._lorentz_scalar_product(X, U)
        return U + np.stack((result,) *X.shape[1], axis=1) * X

    def egrad2rgrad(self, X, U):
        temp = U
        temp[:,0] = -U[:, 0]
        return self.proj(X, temp)

    def inner_minkowski_columns(self, U, V):
        return -U[:, 0]*V[:,0] + np.sum(U[:, 1:]*V[:,1:], axis=1)

    def ehess2rhess(self, X, grad, Hess, U):
        """
        Convert Euclidean into Riemannian Hessian.
        """
        egrad = grad.copy()
        eHess = Hess.copy()
        egrad[:, 0] = -egrad[:, 0]
        eHess[:, 0] = -eHess[:, 0]
        if self._k == 1:
            inners = self.inner_minkowski_columns(X, egrad)
            timesres = U * np.stack((inners,) *U.shape[1], axis=1)
            return  self.proj(X, timesres + eHess)
        else:
            inners = self.inner_minkowski_columns(X, egrad)
            timesres = multitransp(U) * inners
            return self.proj(X, multitransp(timesres + eHess ))


    def expmap0(self, u, c=1):
        K = 1. / c
        sqrtK = K ** 0.5
        # d = u.shape[1] - 1
        # x = u.narrow(-1, 1, d).view(-1, d)
        index = np.arange(1, u.shape[1])
        x = u[:, index]

        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        # x_norm = torch.clamp(x_norm, min=self.min_norm)
        x_norm = np.clip(x_norm, self.min_norm, max(x_norm))
        theta = x_norm / sqrtK
        res = np.ones_like(u)
        res[:, 0:1] = sqrtK * np.cosh(np.clip(theta, -15, 15)) # it does not clamp just use cosh--KONG, edit by Kong
        res[:, 1:] = sqrtK * np.sinh(np.clip(theta, -15, 15)) * x / x_norm
        return self.proj_hyper(res, c)

    def proj_tan0(self, u, c=1):
        narrowed = u[:, [0]]
        vals = np.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def proj_hyper(self, x, c=1):
        K = 1. / c
        # d = x.size(-1) - 1
        # y = x.narrow(-1, 1, d)
        index = np.arange(1, x.shape[1])
        y = x[:, index]

        # y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        y_sqnorm = np.linalg.norm(y, axis=1, keepdims=True) ** 2
        mask = np.ones_like(x)
        mask[:, 0] = 0
        vals = np.zeros_like(x)
        # vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        vals[:, 0:1] = np.sqrt(np.clip(K + y_sqnorm, self.eps, max(K + y_sqnorm)))
        return vals + mask * x

    def from_poincare(self, x, c=1):
        """Convert from Poincare ball model to hyperboloid model.

        Note: converting a point from poincare ball to hyperbolic is
            reversible, i.e. p == to_poincare(from_poincare(p)).

        Args:
            x: torch.tensor of shape (..., dim)
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise
        Returns:
            torch.tensor of shape (..., dim+1)
        To do:
            Add some capping to make things numerically stable. This is only needed in the case ideal == False
        """
        K = 1. / c
        sqrtK = K ** 0.5
        # eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
        eucl_squared_norm = np.sum(x * x, axis=1, keepdims=True)

        # sqrtK * np.concatenate((1 + eucl_squared_norm, 2 * 1 * x), axis=-1)/np.clip(K - eucl_squared_norm, self.min_norm, max(K - eucl_squared_norm))
        # return sqrtK * torch.cat((K + eucl_squared_norm, 2 * sqrtK * x), dim=-1) / (
        #         K - eucl_squared_norm).clamp_min(self.min_norm)
        return sqrtK * np.concatenate((1 + eucl_squared_norm, 2 * 1 * x), axis=-1) / np.clip(K - eucl_squared_norm,
                                                                                             self.min_norm,
                                                                                             max(K - eucl_squared_norm + 1e-7))


    def filldata(self, X):
        if X.shape == () or X.shape[0] == 1:
            Y = np.zeros([1, 2])
            Y[:, 1] = X
            Y[:, 0] = np.sqrt(1 + X ** 2)
        else:
            Y = np.zeros([X.shape[0], 2])
            Y[:, 1] = X
            Y[:, 0] = np.sqrt(1 + np.sum(X[:]**2, axis=1))
        Y = self.proj_tan0(Y)
        Y = self.expmap0(Y)
        return Y

    def tangentToHyperdata(self, X):
        Y = np.zeros([X.shape[0], X.shape[1]+1])
        index = np.arange(1, X.shape[1]+1)
        Y[:, index] = X
        Y = self.proj_tan0(Y)
        Y = self.expmap0(Y)
        return Y

if __name__ == "__main__":
    # n- number of point, k- dimension ->k=2
    hyper =Hyperbolic(3, 1)
    np.random.seed(42)
    data = hyper.rand()
    # test gradient
    data = [[1.3096,   -0.8456], [1.1524 ,  -0.5727]]
    Y = [[-0.6490, -0.7585], [1.1812,  -1.1096]]
    # test hession
    data = [[1.1455,   -0.5587], [1.0158 ,   0.1784], [1.0192,  -0.1969]]
    Y = [ [  -0.6490,   -0.7585,   -0.8456], [1.1812 ,  -1.1096,   -0.5727]]

    data = np.array(data)

    cost = 0.5*(np.linalg.norm(np.transpose(data) - Y,'fro')) **2
    grade = np.zeros(2)
    grade = np.transpose(data) - Y
    grade = np.transpose(grade)
    z= hyper.egrad2rgrad(data, grade)

    u = [[0.4478,   -0.9181], [0.1063,    0.6052], [0.0101,   -0.0524]]
    u = np.array(u)
    ehess = u
    uu = hyper.ehess2rhess(data, grade, ehess, u)
    print(z)
