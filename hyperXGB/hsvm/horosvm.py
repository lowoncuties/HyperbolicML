import numpy as np
import torch

import pymanopt
from pymanopt.manifolds import  Product, Sphere, SymmetricPositiveDefinite
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient

from sklearn.preprocessing import label_binarize

from hsvm.platt import *



def busemann(x, p, keepdims=True):
    """
    x: [..., d]
    p: [..., d]
    Returns: [..., 1] if keepdims==True else [...]
    <w,x>_B = 0.5 * ln (1- |x|^2 / |w-x|^2)
    """

    xnorm = torch.norm(x, dim=-1, keepdim=True)
    pnorm = torch.norm(p, dim=-1)
    p = p / (pnorm)

    deno = torch.norm(p - x, dim=-1, keepdim=True) ** 2
    num = (1 - xnorm ** 2)
    ans = 0.5 * torch.log(num / deno.clamp(1e-8))

    if not keepdims:
        ans = ans.squeeze(-1)
    return ans

def hinge_loss(x, y, margin=1000):

    """
    x: [..., d]
    y: [..., d]
    Returns: [..., 1]
    """
    return torch.max(torch.zeros_like(x), margin - torch.sum(x * y, dim=-1, keepdim=True))


def train_horo_svm(X, y, C, batch_size, max_iter = 1000, verbosity = 0):
    """
    X: [n, d]
    y: [n, 1]
    """
    n, d = X.shape
    X_ = torch.from_numpy(X).float()
    y_ = torch.from_numpy(y).int()

    # man = Product([Euclidean(1), Sphere(d), Euclidean(1)])
    man = Product([SymmetricPositiveDefinite(1), Sphere(d), SymmetricPositiveDefinite(1)])
    # man = Product([SymmetricPositiveDefinite(1), Sphere(d), Euclidean(1)])

    shuffle_idx = np.random.permutation(n)

    @pymanopt.function.pytorch(man)
    def cost(mu, omega, b):
        loss = 0
        for i in range(0, n, batch_size):
            idx = shuffle_idx[i: i+batch_size]
            X_batch = X_[idx, :]
            y_batch = y_[idx]
            decision_values = mu.squeeze()* busemann(X_batch, omega)- b.squeeze()
            margin_loss =  0.5 * mu.squeeze()**2
            misclass_loss = hinge_loss(decision_values, y_batch)
            loss += margin_loss + C * misclass_loss.sum()
        return loss
    
    init_mu = np.ones(1)
    init_mu = np.expand_dims(init_mu, axis=1)
    mean = np.mean(X[np.where(y == 1)], axis=0)
    init_omega = mean / np.linalg.norm(mean)
    init_b = np.ones(1) 
    init_b = np.expand_dims(init_b, axis=1)

    # solver = SteepestDescent(verbosity=verbosity)
    solver = ConjugateGradient(verbosity=verbosity)
    problem = Problem(manifold=man,cost=cost) 
    theta = solver.run(problem,initial_point= [init_mu,init_omega,init_b]).point

    mu = theta[0]
    omega = theta[1]
    b = theta[2]
    
    return mu, omega, b

def poincare_dot(X, mu, omega, b):
    """
    poincare inner product
    """
    ans = mu * (busemann_np(X, omega) - b)
    return ans.squeeze()

def busemann_np(x, p, keepdims=True):
    """
    x: [..., d]
    p: [..., d]
    Returns: [..., 1] if keepdims==True else [...]
    <w,x>_B = 0.5 * ln (1- |x|^2 / |w-x|^2)
    """

    xnorm = np.linalg.norm(x, axis=-1, keepdims=True)
    pnorm = np.linalg.norm(p, axis=-1)
    p = p / (pnorm)
    deno = np.linalg.norm(p - x, axis= -1, keepdims = True) ** 2
    num = (1 - xnorm ** 2)
    ans = 0.5 * np.log(num / deno.clip(1e-8))
    if not keepdims:
        ans = ans.squeeze()
    return ans

class horo_svm(object):
    """SVM classifier.
    Parameters
    ------------
    C: hyperparameter
    
    Attributes
    -------------
    
    """
    def __init__(self,  C = 1.0, batch_size = 20, max_iter = 1000000000,  verbose=0, multiclass = False):
        self.C = C
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.multiclass = multiclass

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ------------
        X: array, shape=[n_samples,n_features]
        y: array, shape=n_samples,)
        Returns
        ----------
        self: object
        """

        self.y_train = y


        if self.multiclass:
            y_binary = label_binarize(y, classes=np.unique(y))
            y_binary = 2 * y_binary - 1 #convert to -1 and 1
            self.class_labels = np.unique(y)
            self.mu_ = []
            self.omega_ = []
            self.b_ = []
            self.platt_coefs = []
            for i in range(y_binary.shape[-1]):
                self.mu_i, self.omega_i, self.bi = train_horo_svm(X, y_binary[:,i], 
                                                self.C, self.batch_size, self.verbose)
                self.mu_.append(self.mu_i)
                self.omega_.append(self.omega_i)
                self.b_.append(self.bi)

            
            for i in range(y_binary.shape[-1]):
                decision_values = poincare_dot(X, self.mu_[i], self.omega_[i], self.b_[i])
                yi_train = (self.y_train == self.class_labels[i]).astype('int')
                # convert labels = {0, 1} to {-1, 1}
                yi_train = 2 * yi_train - 1
                # get platt coefs A, B
                ab = SigmoidTrain(deci=decision_values, label=yi_train, prior1 = None, prior0 = None)
                self.platt_coefs.append(ab)
        else:
            classes = np.unique(y)
            self.class_labels_ = {'neg_class': np.min(classes), 'pos_class': np.max(classes)}            
            y_binary = 2 * (self.y_train == self.class_labels_['pos_class']) - 1 #convert to -1 and 1
            self.mu, self.omega, self.b = train_horo_svm(X, y_binary[:,0],
                                self.C, self.batch_size, self.verbose)

            decision_value = poincare_dot(X, self.mu, self.omega, self.b)
            # yi_train = 2 * self.y_train - 1
            yi_train = 2*(self.y_train == self.class_labels_['pos_class']).astype('int')-1
            ab = SigmoidTrain(decision_value, yi_train) 
            self.platt_coefs = ab

        return self
    
    
    def predict(self, X):
        """Return the predicted class label.
        Parameters
        ------------
        X: array, shape=[n_samples,n_features]
        Returns
        ----------
        y_pred: array, shape=[n_samples,]
        """
        if self.multiclass:
            n_classes = len(self.class_labels)
            y_probs = np.zeros((X.shape[0], n_classes))
            for i in range(n_classes):
                decision_values = poincare_dot(X, self.mu_[i], self.omega_[i], self.b_[i])
                for j in range(X.shape[0]):
                    y_probs[j, i] = SigmoidPredict(decision_values[j], self.platt_coefs[i])
            y_pred = self.class_labels[np.argmax(y_probs, axis=1)]

        else:
            y_pred = np.zeros((X.shape[0],))
            decision_value = poincare_dot(X, self.mu, self.omega, self.b)
            y_pred = np.where(decision_value < 0, self.class_labels_['neg_class'], self.class_labels_['pos_class'])

        return y_pred

    def predict_proba(self, X):
        """
        Predict probability from Platt method and hyperbolic decision function vals
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        """
       
        if self.multiclass:
            n_classes = len(self.class_labels)
            
            # we find probabilities of belonging to each class 
            y_probs = np.zeros((X.shape[0], n_classes))
            
            # find each class prediction score and apply Platt probability scaling
            for i in range(n_classes):
               for i in range(n_classes):
                decision_values = poincare_dot(X, self.mu_[i], self.omega_[i], self.b_[i])
                for j in range(X.shape[0]):
                    y_probs[j, i] = SigmoidPredict(decision_values[j], self.platt_coefs[i])
            return y_probs
        
        else:
            y_probs = np.zeros((X.shape[0], ))
            decision_value = poincare_dot(X, self.mu, self.omega, self.b)
            # for i in range(X.shape[0]):
            #     y_probs[i] = SigmoidPredict(deci=decision_value[i], AB=self.platt_coefs)
            # return y_probs 
            return decision_value

    
    def accuracy(self, X, y):
        """Return the accuracy of the model.
        Parameters
        ------------
        X: array, shape=[n_samples,n_features]
        y: array, shape=(n_samples,)
        Returns
        ----------
        accuracy: float
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy