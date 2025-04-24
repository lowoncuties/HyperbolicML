
import torch
import scipy.io


def get_training_data(source, class_label, seed):
    mat = scipy.io.loadmat(source+'data_' + str('{:0>3}'.format(class_label)) + '.mat')
    data = mat["B"]
    labels = mat["label"]
    labels = labels - 1
    return torch.as_tensor(data), labels


def get_testing_data(source, class_label, seed):
    mat = scipy.io.loadmat(source + 'data_' + str('{:0>3}'.format(class_label)) + '.mat')
    data = mat["B"]
    labels = mat["label"]
    labels = labels - 1
    return torch.as_tensor(data), labels


def get_space():
    return 'hyperbolic'