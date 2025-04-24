import torch
import scipy.io
import numpy as np
from hsvm.utils import norm
from hsvm.utils import poincare_pts_to_hyperboloid
from sklearn.preprocessing import label_binarize

def get_training_data(source, class_label, seed):
    mat = scipy.io.loadmat(source+'data_' + str('{:0>3}'.format(class_label)) + '.mat')

    X, y = mat['B'], mat['label'].ravel().astype(np.int8)
    # if not in poincare disk, project points within the disk
    if (norm(X, axis=1) > 1).any():
        out_pts = norm(X, axis=1) > 1
        num_pts = np.sum(out_pts)
        X[out_pts] = X[out_pts] / norm(X[out_pts], axis=0) - np.repeat(1e-3, num_pts).reshape(-1,1)

    shuffle_index = np.random.permutation(len(y))
    X = X[shuffle_index]
    X = poincare_pts_to_hyperboloid(X, metric='minkowski')
    labels_binary = label_binarize(y, classes=[1, 2, 3, 4])
    y = labels_binary[:, class_label-1]
    return torch.as_tensor(X), y


def get_testing_data(source, class_label, seed):
    mat = scipy.io.loadmat(source+'data_' + str('{:0>3}'.format(class_label)) + '.mat')

    X, y = mat['B'], mat['label'].ravel().astype(np.int8)
    # if not in poincare disk, project points within the disk
    if (norm(X, axis=1) > 1).any():
        out_pts = norm(X, axis=1) > 1
        num_pts = np.sum(out_pts)
        X[out_pts] = X[out_pts] / norm(X[out_pts], axis=0) - np.repeat(1e-3, num_pts).reshape(-1,1)

    shuffle_index = np.random.permutation(len(y))
    X = X[shuffle_index]
    X = poincare_pts_to_hyperboloid(X, metric='minkowski')
    labels_binary = label_binarize(y, classes=[1, 2, 3, 4])
    y = labels_binary[:, class_label-1]
    return torch.as_tensor(X), y


def get_space():
    return 'hyperbolic'

def get_multi_class():
    return True