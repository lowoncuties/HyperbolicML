import yaml
from xgb.utils import set_seeds

from xgb.hyper_trainer_feb2 import run_train, run_train_UCI
import numpy as np


# _c = np.array([0.1, 1, 5, 10]),  # range: (0,1] (def 1)
# _lr = np.array([0.001, 0.01, 0.1, 0.0001]),  # 0-1(def 0.3)
# # _gamma = np.concatenate((np.arange(0.0, 1.1, 0.1), np.arange(10, 100, 5)), axis=0),  # 0- inf
# _batch_size = np.array([10, 20, 16, 32]),
#
#
# _beta = np.array([0, 0.1, 0.5, 10]),  # range: (0,1] (def 1)
# _subsample_size = np.array([0.1, 0.4, 0.8, 1]),  # 0-1(def 0.3)
# _min_impurity_decrease = np.array([0.0, 0.01, 0.05, 0.1]),

_c = np.array([0.05, 0.01, 3, 0.1, 1, 5, 10]),  # range: (0,1] (def 1)
_lr = np.array([0.005, 0.05, 0.0005, 0.001, 0.01, 0.1, 0.0001]),  # 0-1(def 0.3)
_batch_size = np.array([4, 8, 24, 10, 20, 16, 32]),


# _beta = np.array([0.05, 0.08, 0.8, 0.01, 0.1, 0.5, 1]),  # range: (0,1] (def 1)
_max_depth = np.array([3, 6, 8, 12, 16, 10, 14]),  # range: (0,1] (def 1)
_subsample_size = np.array([0.05, 0.7, 0.5, 0.1, 0.4, 0.8, 1]),  # 0-1(def 0.3)
_min_impurity_decrease = np.array([0.03, 0.07, 0.09, 0.0, 0.01, 0.05, 0.1]),

def config_array_rf(confini, array):
    confini['max_depth'] = _max_depth[0][array.A - 1]
    confini['subsample_size'] = _subsample_size[0][array.B - 1]
    confini['min_impurity_decrease'] = _min_impurity_decrease[0][array.C - 1]

    return confini  # , _num_boost_round[0][array.H - 1]



def config_array_svm(confini, array):
    confini['c'] = _c[0][array.A - 1]
    confini['lr'] = _lr[0][array.B - 1]
    confini['batch_size'] = _batch_size[0][array.C - 1]

    return confini  # , _num_boost_round[0][array.H - 1]


def tune_main():
    import pandas
    method = ['LinearHSVM' ] #'LinearHSVM', 'horoRF'
    file = 'wordnet'
    arr = 256
    for med in range(1):
        df = pandas.read_csv(('param//rforthogonalarray' + str(arr) + '.csv'), header=0)
        del df[df.columns[0]]

        params_file = "configs/params." + file + ".yml"
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        for idarray in range(49, 20, -1):
            params['seed'] = 42
            # params["seed"] = params["seed"] + params["class_label"]
            set_seeds(params["seed"])
            params['method'] = method[med] + str(idarray)
            if med == 0:
                params = config_array_svm(params, df.iloc[idarray])
            else:
                params = config_array_rf(params, df.iloc[idarray])

            if file == 'UCIdata' or file == 'UCIdataE' or file=='wordnet':
                run_train_UCI(params, params_file)
            else:
                run_train(params, params_file)



tune_main()

