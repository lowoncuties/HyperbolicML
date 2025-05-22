import torch
import numpy as np
from pathlib import Path

# we add a empty '', to match the index of [1, len+1]
label_to_name = [
    "",
    "abalone",
    "acute-inflammation",
    "acute-nephritis",
    "annealing",
    "audiology-std",
    "balance-scale",
    "balloons",
    "blood",
    "breast-cancer",
    "breast-cancer-wisc",
    "breast-cancer-wisc-diag",
    "breast-cancer-wisc-prog",
    "breast-tissue",
    "car",
    "congressional-voting",
    "conn-bench-sonar-mines-rocks",
    "conn-bench-vowel-deterding",
    "contrac",
    "credit-approval",
    "cylinder-bands",
    "dermatology",
    "echocardiogram",
    "ecoli",
    "energy-y1",
    "energy-y2",
    "fertility",
    "flags",
    "glass",
    "haberman-survival",
    "hayes-roth",
    "heart-cleveland",
    "heart-hungarian",
    "heart-switzerland",
    "heart-va",
    "hepatitis",
    "horse-colic",
    "ilpd-indian-liver",
    "ionosphere",
    "iris",
    "led-display",
    "lenses",
    "libras",
    "lung-cancer",
    "lymphography",
    "mammographic",
    "molec-biol-promoter",
    "monks-1",
    "monks-2",
    "monks-3",
    "oocytes_merluccius_nucleus_4d",
    "oocytes_merluccius_states_2f",
    "oocytes_trisopterus_nucleus_2f",
    "oocytes_trisopterus_states_5b",
    "parkinsons",
    "pima",
    "pittsburg-bridges-MATERIAL",
    "pittsburg-bridges-REL-L",
    "pittsburg-bridges-SPAN",
    "pittsburg-bridges-T-OR-D",
    "pittsburg-bridges-TYPE",
    "planning",
    "post-operative",
    "primary-tumor",
    "seeds",
    "soybean",
    "spect",
    "spectf",
    "statlog-australian-credit",
    "statlog-german-credit",
    "statlog-heart",
    "statlog-image",
    "statlog-vehicle",
    "synthetic-control",
    "teaching",
    "tic-tac-toe",
    "titanic",
    "trains",
    "vertebral-column-2clases",
    "vertebral-column-3clases",
    "wine",
    "wine-quality-red",
    "yeast",
    "zoo",
]


def get_training_data(source, class_label, seed):
    print("Running", label_to_name[class_label], "experiment")

    dataset = UCIDataset(label_to_name[class_label], source)
    cv = seed % 4
    X_test, y_test, data, labels, valx, valy = dataset.getitem(cv)
    labels = np.expand_dims(labels, axis=1)
    labels = labels.astype(np.uint8)
    return torch.as_tensor(data), labels


def get_testing_data(source, class_label, seed):
    dataset = UCIDataset(label_to_name[class_label], source)
    cv = seed % 4
    X_test, y_test, data, labels, valx, valy = dataset.getitem(cv)
    labels = np.expand_dims(y_test, axis=1)  # change test data
    labels = labels.astype(np.uint8)

    return torch.as_tensor(X_test), labels


def get_validation_data(source, class_label, seed):
    dataset = UCIDataset(label_to_name[class_label], source)
    cv = seed % 4
    X_test, y_test, data, labels, valx, valy = dataset.getitem(cv)
    labels = np.expand_dims(valy, axis=1)  # change test data
    labels = labels.astype(np.uint8)

    return torch.as_tensor(valx), labels


class UCIDataset:
    def __init__(self, dataset, parent=None):

        self.root = Path(parent) / dataset
        data_file = sorted(self.root.glob(f"{dataset}*.dat"))[0]
        label_file = sorted(self.root.glob("label*.dat"))[0]
        val_file = sorted(self.root.glob("validation*.dat"))[0]
        fold_index = sorted(self.root.glob("folds*.dat"))[0]
        self.dataX = np.loadtxt(data_file, delimiter=",")
        self.dataY = np.loadtxt(label_file, delimiter=",")
        self.validation = np.loadtxt(val_file, delimiter=",")
        self.folds_index = np.loadtxt(fold_index, delimiter=",")
        self.n_CV = self.folds_index.shape[1]
        # types = np.unique(self.dataY)
        # self.n_types = types.size
        # # One hot coding for the target
        # self.dataY_tmp = np.zeros((self.dataY.size, self.n_types))
        # for i in range(self.n_types):
        #     for j in range(self.dataY.size):  # remove this loop
        #         if self.dataY[j] == types[i]:
        #             self.dataY_tmp[j, i] = 1

    def getitem(self, CV):
        full_train_idx = np.where(self.folds_index[:, CV] == 0)[0]
        train_idx = np.where(
            (self.folds_index[:, CV] == 0) & (self.validation[:, CV] == 0)
        )[0]
        test_idx = np.where(self.folds_index[:, CV] == 1)[0]
        val_idx = np.where(self.validation[:, CV] == 1)[0]

        full_train_x = self.dataX[full_train_idx, :]
        testX = self.dataX[test_idx, :]
        testY = self.dataY[test_idx]
        full_train_y = self.dataY[full_train_idx]

        trainX = self.dataX[train_idx, :]
        trainY = self.dataY[train_idx]
        evalX = self.dataX[val_idx, :]
        evalY = self.dataY[val_idx]
        # return testX, testY, full_train_x, full_train_y
        return testX, testY, trainX, trainY, evalX, evalY


def get_space():
    return "euclidean"


def get_multi_class():
    # this need to be check again during process
    return False
