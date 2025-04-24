
import logging
import os
from joblib import Parallel, delayed
from functools import partial

import numpy as np

from sklearn.utils import resample
from sklearn.metrics import f1_score, average_precision_score,accuracy_score

import torch
from ignite.utils import setup_logger

from .loaddata import UCIDataset
from .utils import archive_code, expanduservars, _build_datasets, kfold_split, save_results, save_results_acc
from .node import Node
from .geom.poincare import expmap0
from .visualization import visualize_decision_boundary, plot_tree, visualize_splits

LOGGER = logging.getLogger(__name__)

dataname =['abalone', 'acute-inflammation', 'acute-nephritis', 'annealing', 'audiology-std', 'balance-scale', 'balloons', 'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 'breast-cancer-wisc-prog', 'breast-tissue', 'car', 'congressional-voting', 'conn-bench-sonar-mines-rocks', 'conn-bench-vowel-deterding', 'contrac', 'credit-approval', 'cylinder-bands', 'dermatology', 'echocardiogram', 'ecoli', 'energy-y1', 'energy-y2', 'fertility', 'flags', 'glass', 'haberman-survival', 'hayes-roth', 'heart-cleveland', 'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic', 'ilpd-indian-liver', 'ionosphere', 'iris', 'led-display', 'lenses', 'libras', 'lung-cancer', 'lymphography', 'mammographic', 'molec-biol-promoter', 'monks-1', 'monks-2', 'monks-3', 'oocytes_merluccius_nucleus_4d', 'oocytes_merluccius_states_2f', 'oocytes_trisopterus_nucleus_2f', 'oocytes_trisopterus_states_5b', 'parkinsons', 'pima', 'pittsburg-bridges-MATERIAL', 'pittsburg-bridges-REL-L', 'pittsburg-bridges-SPAN', 'pittsburg-bridges-T-OR-D', 'pittsburg-bridges-TYPE', 'planning', 'post-operative', 'primary-tumor', 'seeds', 'soybean', 'spect', 'spectf', 'statlog-australian-credit', 'statlog-german-credit', 'statlog-heart', 'statlog-image', 'statlog-vehicle', 'synthetic-control', 'teaching', 'tic-tac-toe', 'titanic', 'trains', 'vertebral-column-2clases', 'vertebral-column-3clases', 'wine', 'wine-quality-red', 'yeast', 'zoo']


def tree_predict(hyp_test_x, tree):
    predictions = []
    probabilities = []
    for x in hyp_test_x:
        predicted_class, probability = tree.predict(x)

        predictions.append(predicted_class)
        probabilities.append(probability)

    return predictions, probabilities


def make_tree(hyp_train_x, train_y, params, num_tree):
    hyp_train_x, train_y = resample(hyp_train_x, train_y, random_state=params["seed"] + num_tree)

    tree = Node(hyp_train_x, train_y, 0, params)
    tree.grow()

    # if params["visualize"]:
    #     plot_tree(tree, hyp_train_x, train_y, "tree_train" + str(num_tree) + ".png")
    
    return tree


def evaluate_forest(trees, hyp_test_x, labels, params):
    result = Parallel(n_jobs=params["num_jobs"])(delayed(partial(tree_predict, hyp_test_x))(trees[i]) for i in range(params["num_trees"]))
    result = np.array(result)
    
    test_y_preds = result[:, 0]
    test_y_probs = result[:, 1]

    max_probs = np.zeros(len(hyp_test_x))
    test_y_pred = np.zeros(len(hyp_test_x))
    for label in labels:
        mask = (test_y_preds == label).astype(int)
        label_probs = np.sum(test_y_probs * mask, axis=0)
        
        test_y_pred[label_probs > max_probs] = label
        max_probs[label_probs > max_probs] = label_probs[label_probs > max_probs]

    return test_y_pred, label_probs


def run_train(params: dict, params_file):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    archive_code(output_path, params_file)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    LOGGER.info("Using seed %d on class %d", params["seed"], params["class_label"])

    full_train_x, full_train_y, full_test_x, full_test_y, space = _build_datasets(params)

    hyp_f1_micro = 0
    hyp_f1_macro = 0
    hyp_acc = 0
    hyp_aupr = 0

    labels = np.unique(np.concatenate((full_train_y, full_test_y)))
    
    #  Cross validation
    for i in range(params["folds"]):
        if params["folds"] > 1:
            train_x, train_y, test_x, test_y = kfold_split(full_train_x, full_train_y, i, params)
            LOGGER.info("%d train and %d test samples for fold %d", len(train_x), len(test_x), i)
        else:
            train_x = full_train_x
            train_y = full_train_y
            test_x = full_test_x
            test_y = full_test_y
        
        if space == 'euclidean':
            LOGGER.info("Mapping to hyperbolic space")
            hyp_train_x = expmap0(train_x, 1.0)
            hyp_test_x = expmap0(test_x, 1.0)
        else:
            hyp_train_x = train_x
            hyp_test_x = test_x

        trees = Parallel(n_jobs=params["num_jobs"])(delayed(partial(make_tree, hyp_train_x, train_y, params))(i) for i in range(params["num_trees"]))

        average_depth = np.mean([tree.get_depth() for tree in trees])

        if params["visualize"]:
            plot_tree(trees[0], hyp_test_x, test_y, "tree_test" + str(0) + ".png")

        test_y_pred, test_y_probs = evaluate_forest(trees, hyp_test_x, labels, params)
        
        aupr = average_precision_score(test_y[:, 0], 1 - test_y_probs, pos_label=0) if params["dataset_file"] in ['datasets.wordnet_full', 'datasets.wn_full_5d', 'datasets.wn_full_10d'] else 0
        
        f1_micro = f1_score(test_y, test_y_pred, average="micro")
        f1_macro = f1_score(test_y, test_y_pred, average="macro")
        acc =       accuracy_score(test_y, test_y_pred)
        LOGGER.info("Hyperbolic tree f1 micro: %.4f, f1 macro: %.4f, AUPR: %.4f. Mean depth of %.2f", f1_micro, f1_macro, aupr, average_depth)

        if params["visualize"] and hyp_train_x.shape[-1] == 2:
            feature_1, feature_2 = np.meshgrid(
                np.linspace(-1, 1, 100),
                np.linspace(-1, 1, 100)
            )

            grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
            grid = torch.as_tensor(grid)

            grid_preds, _ = evaluate_forest(trees, grid, labels, params)

            visualize_decision_boundary(hyp_train_x, train_y, feature_1, feature_2, grid_preds, 'train' + str(i) + '.png')
            visualize_decision_boundary(hyp_test_x, test_y, feature_1, feature_2, grid_preds, 'test' + str(i) + '.png')
            visualize_splits(trees[0], hyp_train_x, train_y, 'splits.png')

        hyp_f1_micro += f1_micro
        hyp_f1_macro += f1_macro
        hyp_aupr += aupr
        hyp_acc  += acc
    save_results(output_path, params, hyp_f1_micro, hyp_f1_macro, hyp_aupr, hyp_acc)

def setup_train(params: dict, cv = 0):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    # archive_code(output_path, params_file)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    # LOGGER.info("Using seed %d on class %d", params["seed"], params["class_label"])
    nid = [20, 30, 32, 35, 36, 8, 9, 12, 45, 72, 76, 28, 13, 40, 44, 50, 55, 69]
    for id in range(18):
        # for each data parameter tuning.
        dname = dataname[nid[id]]  # matlab index is from 1
        params["folds"] = 5
        classifier = my_classifier()
        classifier.set_data(dname, cv=cv)
        for run in range(5):
            params["seed"] = params["seed"] + 1
            classifier.train(params)

        save_results_acc(output_path, params, classifier.acc, cv)

import pandas as pd

class my_classifier:

    def init_(self, name=None, cv = 0):
        # df = pd.read_csv(('newdata//' + name + '//' + 'labels_py.dat'), delimiter=',', header=None)
        dfolder = 'C:\\Users\\16499\\Downloads\\data_py\\data\\'
        df = pd.read_csv((dfolder + '//' + name + '//' + 'labels_py.dat'), delimiter=',', header=None)
        y = df.to_numpy()
        y = y.squeeze()
        dataset = UCIDataset(name)
        trainX, trainY, evalX, evalY, X_test, y_test, X_train, y_train = dataset.getitem(cv)

        kClasses = np.max(y) + 1

        kRows, kCols = X_train.shape
        if kRows < 100:
            self.kRounds = 5
        elif kRows < 250 and kRows > 99:
            self.kRounds = int(kRows / 20)  # 20  (middle choice is invalid
        else:
            self.kRounds = int(kRows / 50)  # 40
        X_train = torch.as_tensor(X_train)
        X_test = torch.as_tensor(X_test)
        return  X_test, y_test, X_train, y_train, kClasses

    def set_data(self, name=None, cv = 0):
      self.name = name
      self.cv = cv
      self.acc = 0

    def train(self, params: dict, space: str = 'euclidean'):
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        X_test, y_test, X_train, y_train, kClasses = self.init_(self.name, self.cv)
        params['class_label'] = kClasses
        params['num_trees'] = self.kRounds
        labels = np.unique(np.concatenate((y_train, y_test)))
        if space == 'euclidean':
            LOGGER.info("Mapping to hyperbolic space")
            hyp_train_x = expmap0(X_train, 1.0)
            hyp_test_x = expmap0(X_test, 1.0)
        else:
            hyp_train_x = X_train
            hyp_test_x = X_test

        trees = Parallel(n_jobs=params["num_jobs"])(
            delayed(partial(make_tree, hyp_train_x, y_train, params))(i) for i in range(params["num_trees"]))

        average_depth = np.mean([tree.get_depth() for tree in trees])
        test_y_pred, test_y_probs = evaluate_forest(trees, hyp_test_x, labels, params)
        acc = accuracy_score(y_test, test_y_pred)
        self.acc = self.acc + acc

def run_train_(params: dict, params_file):
    output_path = expanduservars(params['output_path'])
    full_train_x, full_train_y, full_test_x, full_test_y, space = _build_datasets(params)

    hyp_f1_micro = 0
    hyp_f1_macro = 0
    hyp_acc = 0
    hyp_aupr = 0

    labels = np.unique(np.concatenate((full_train_y, full_test_y)))

    #  Cross validation
    for i in range(params["folds"]):
        if params["folds"] > 1:
            train_x, train_y, test_x, test_y = kfold_split(full_train_x, full_train_y, i, params)
            LOGGER.info("%d train and %d test samples for fold %d", len(train_x), len(test_x), i)
        else:
            train_x = full_train_x
            train_y = full_train_y
            test_x = full_test_x
            test_y = full_test_y

        if space == 'euclidean':
            LOGGER.info("Mapping to hyperbolic space")
            hyp_train_x = expmap0(train_x, 1.0)
            hyp_test_x = expmap0(test_x, 1.0)
        else:
            hyp_train_x = train_x
            hyp_test_x = test_x

        trees = Parallel(n_jobs=params["num_jobs"])(
            delayed(partial(make_tree, hyp_train_x, train_y, params))(i) for i in range(params["num_trees"]))

        average_depth = np.mean([tree.get_depth() for tree in trees])

        if params["visualize"]:
            plot_tree(trees[0], hyp_test_x, test_y, "tree_test" + str(0) + ".png")

        test_y_pred, test_y_probs = evaluate_forest(trees, hyp_test_x, labels, params)

        aupr = average_precision_score(test_y[:, 0], 1 - test_y_probs, pos_label=0) if params["dataset_file"] in [
            'datasets.wordnet_full', 'datasets.wn_full_5d', 'datasets.wn_full_10d'] else 0

        f1_micro = f1_score(test_y, test_y_pred, average="micro")
        f1_macro = f1_score(test_y, test_y_pred, average="macro")
        acc = accuracy_score(test_y, test_y_pred)
        LOGGER.info("Hyperbolic tree f1 micro: %.4f, f1 macro: %.4f, AUPR: %.4f. Mean depth of %.2f", f1_micro,
                    f1_macro, aupr, average_depth)

        if params["visualize"] and hyp_train_x.shape[-1] == 2:
            feature_1, feature_2 = np.meshgrid(
                np.linspace(-1, 1, 100),
                np.linspace(-1, 1, 100)
            )

            grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
            grid = torch.as_tensor(grid)

            grid_preds, _ = evaluate_forest(trees, grid, labels, params)

            visualize_decision_boundary(hyp_train_x, train_y, feature_1, feature_2, grid_preds,
                                        'train' + str(i) + '.png')
            visualize_decision_boundary(hyp_test_x, test_y, feature_1, feature_2, grid_preds, 'test' + str(i) + '.png')
            visualize_splits(trees[0], hyp_train_x, train_y, 'splits.png')

        hyp_f1_micro += f1_micro
        hyp_f1_macro += f1_macro
        hyp_aupr += aupr
        hyp_acc += acc
    save_results(output_path, params, hyp_f1_micro, hyp_f1_macro, hyp_aupr, hyp_acc)