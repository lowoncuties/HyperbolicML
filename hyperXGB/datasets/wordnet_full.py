
import torch
import numpy as np
from sklearn.model_selection import train_test_split

label_to_name = {
    1: 'animal',
    2: 'group',
    3: 'worker',
    4: 'mammal',
    5: 'tree',
    6: 'solid',
    7: 'occupation',
    8: 'rodent'    ,                                                                                                                                                                                                                                                                          
}


def get_training_data(source, class_label, seed):
    print('Running', label_to_name[class_label], 'experiment')
    print('Running', source)
    data = np.load((source + '//embeddings.npy'))
    # print(data)
    labels = np.load(source + '//' + label_to_name[class_label] + '_labels.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed,
                                                              stratify=labels)
    train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.2, random_state=3,
                                                              stratify=train_labels)

    # return torch.as_tensor(train), train_labels[:, None]
    return torch.as_tensor(train), train_labels


def get_testing_data(source, class_label, seed):
    data = np.load(source + '//embeddings.npy')
    labels = np.load(source + '//' + label_to_name[class_label] + '_labels.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed,
                                                              stratify=labels)

    # return torch.as_tensor(test), test_labels[:, None]
    return torch.as_tensor(test), test_labels


def get_validation_data(source, class_label, seed):
    data = np.load(source + '//embeddings.npy')
    labels = np.load(source + '//' + label_to_name[class_label] + '_labels.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed,
                                                              stratify=labels)
    train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.2, random_state=3,
                                                              stratify=train_labels)

    # return torch.as_tensor(test), test_labels[:, None]
    return torch.as_tensor(test), test_labels


def get_space():
    return 'hyperbolic'

def get_multi_class():
    return False