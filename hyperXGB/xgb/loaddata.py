dfolder =  'C:\\Users\\16499\\Downloads\\data_py\\data\\'

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/06/28 17:51
@Author: Merc2
'''
import numpy as np
from pathlib import Path

class UCIDataset:
    def __init__(self, dataset, parent=dfolder):

        self.root = Path(parent) / dataset
        data_file = sorted(self.root.glob(f'{dataset}*.dat'))[0]
        label_file = sorted(self.root.glob('label*.dat'))[0]
        val_file = sorted(self.root.glob('validation*.dat'))[0]
        fold_index = sorted(self.root.glob('folds*.dat'))[0]
        self.dataX = np.loadtxt(data_file, delimiter=',')
        self.dataY = np.loadtxt(label_file, delimiter=',')
        self.validation = np.loadtxt(val_file, delimiter=',')
        self.folds_index = np.loadtxt(fold_index, delimiter=',')
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
        train_idx = np.where((self.folds_index[:, CV] == 0) & (self.validation[:, CV] == 0))[0]
        test_idx = np.where(self.folds_index[:, CV] == 1)[0]
        val_idx = np.where(self.validation[:, CV] == 1)[0]
        trainX = self.dataX[train_idx, :]
        # trainY = self.dataY_tmp[train_idx, :]
        trainY = self.dataY[train_idx]
        testX = self.dataX[test_idx, :]
        # testY = self.dataY_tmp[test_idx, :]
        testY = self.dataY[test_idx]
        evalX = self.dataX[val_idx, :]
        # evalY = self.dataY_tmp[val_idx, :]
        evalY = self.dataY[val_idx]
        full_train_x = self.dataX[full_train_idx, :]
        # full_train_y = self.dataY_tmp[full_train_idx, :]
        full_train_y = self.dataY[full_train_idx]
        return trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y


# usage

# dataset = UCIDataset('loc/to/dataset')
# trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = dataset.getitem(0) # 0 is the CV index
# trainX is the training data
# trainY is the training label
# evalX is the validation data
# evalY is the validation label
# testX is the test data
# testY is the test label
# full_train_x is the full training data
# full_train_y is the full training label
# if __name__ == "__main__":
#     dataset = UCIDataset('iris')
#     trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = dataset.getitem(0) # 0 is the CV index
#     trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = dataset.getitem(1)