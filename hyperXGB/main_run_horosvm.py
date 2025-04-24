
import yaml
from xgb.utils import set_seeds
import pandas as pd

from xgb.hyper_trainer_feb2 import run_train_UCI
import numpy as np

_c = np.array([0.05, 0.01, 3, 0.1, 1.0, 5.0, 10.0]),  # range: (0,1] (def 1)
_trail = np.array([1e4, 1e5, 1e6, 5e4, 5e5, 1e3, 5e3]).astype(int),  # 0-1(def 0.3)
_batch_size = np.array([4, 8, 24, 10, 20, 16, 32]),


def config_array_svm(confini, array):
    confini['C'] = _c[0][array.A - 1]
    confini['ntrail'] = _trail[0][array.B - 1]
    confini['batch_size'] = _batch_size[0][array.C - 1]

    return confini  # , _num_boost_round[0][array.H - 1]


def tune_main_():
    method = ['horoSVM']
    file = 'UCIdatanew'
    df = pd.read_csv(('param//orthogonalarray' + str(256) + '.csv'), header=0)
    del df[df.columns[0]]

    params_file = "configs/params." + file + ".yml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    for idarray in range(1):
        set_seeds(params["seed"])
        params['method'] = method[0] + str(idarray)
        confini =dict()
        confini = config_array_svm(confini, df.iloc[idarray])
        params['C'] = confini['C']
        params['ntrail'] = confini['ntrail']
        params['batch_size'] = confini['batch_size']

        run_train_UCI(params, params_file)

import time
t = time.time()
tune_main_()
print(time.time()-t)
