
import yaml
from xgb.utils import set_seeds
import pandas as pd

from xgb.hyper_trainer_svm import run_train_UCI
import numpy as np

## for svm parameter
_kernel = ["linear", "poly", "rbf", "sigmoid"],
_C = np.array([0.01, 0.1, 1, 10]),
_degree = np.array([1, 2, 3, 4]),
_coef = np.array([0.1, 1, 5, 10]),
_gamma = ["auto", "scale"],
_shrinking = [True, False],

def config_array_svm(confini, array):
    confini['kernel'] = _kernel[0][array.D - 1]
    confini['C'] = _C[0][array.E - 1]
    confini['degree'] = _degree[0][array.F - 1]
    confini['coef0'] = _coef[0][array.A - 1]
    id = (array.B - 1)%2
    confini['shrinking'] = _shrinking[0][id]
    id = (array.C - 1) % 2
    confini['gamma'] = _gamma[0][id]
    return confini

_bootstrap = [True, False],
_min_samples_leaf = np.array([1, 3, 5, 6]),
_min_samples_split = np.array([2, 8, 10, 20, 40]),
_max_depth = np.array([10, 20, 30, 40]),
_min_impurity_decrease = np.array([0.0, 0.01, 0.05, 0.1]),
_criterion = ['gini', 'entropy', 'log_loss', 'gini'],


def config_array_rf(confini, array):
    confini['min_samples_leaf'] = _min_samples_leaf[0][array.D - 1]
    confini['min_samples_split'] = _min_samples_split[0][array.E - 1]
    confini['max_depth'] = _max_depth[0][array.F - 1]
    confini['min_impurity_decrease'] = _min_impurity_decrease[0][array.A - 1]
    confini['criterion'] = _criterion[0][array.B - 1]
    id = (array.C - 1)%2
    confini['bootstrap'] = _bootstrap[0][id]
    confini['n_estimators'] = 100
    return confini


def tune_main_():
    method = ['randomforest']
    file = 'UCIdata'
    df = pd.read_csv(('param//orthogonalarray' + str(256) + '.csv'), header=0)
    del df[df.columns[0]]

    params_file = "configs/params." + file + ".yml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    for idarray in range(2):
        set_seeds(params["seed"])
        params['method'] = method[0] + str(idarray)
        confini =dict()
        if params['method'][:3]=='SVM':
            confini = config_array_svm(confini, df.iloc[idarray])
        else:
            confini = config_array_rf(confini, df.iloc[idarray])

        run_train_UCI(params, params_file, confini)
tune_main_()

