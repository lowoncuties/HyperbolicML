import yaml
from xgb.utils import set_seeds
import pandas
import numpy as np
from xgb.hyper_trainer import run_train, run_train_UCI

_subsample = np.array([0.5, 0.7, 0.8, 0.9]),  # range: (0,1] (def 1)
_eta = np.array([0.01, 0.05, 0.2, 0.1]),  # 0-1(def 0.3)
# _gamma = np.concatenate((np.arange(0.0, 1.1, 0.1), np.arange(10, 100, 5)), axis=0),  # 0- inf
_gamma = np.array([0.9, 0.6, 0.3, 0.0]),
_colsample_bytree = np.array([1.0, 0.7, 0.5, 0.9]),  # of (0, 1], the default value of 1,
_colsample_bylevel = np.array([0.9, 0.7, 0.5, 1.0]),  # of (0, 1], the default value of 1,
_colsample_bynode = np.array([0.5, 0.9, 1.0, 0.7]),  # of (0, 1], the default value of 1,

def config_array(confini, array):
    confini['subsample'] = _subsample[0][array.D - 1]
    confini['colsample_bylevel'] = _colsample_bylevel[0][array.E - 1]
    confini['colsample_bynode'] = _colsample_bynode[0][array.F - 1]
    confini['colsample_bytree'] = _colsample_bytree[0][array.A - 1]
    confini['eta'] = _eta[0][array.B - 1]
    confini['gamma'] = _gamma[0][array.C - 1]
    return confini  # , _num_boost_round[0][array.H - 1]


def main():
    import sys
    params_file = sys.argv[1]
    print(params_file)
    # params_file = "configs/params.karate.yml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Different seed for each of the 5 network embeddings
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])

    run_train(params, params_file)

#main()
def tune_main():

    method = ['Pxgboost']
    file = 'UCIdataE'
    for arr in [256]:
        df = pandas.read_csv(('orthogonalarray' + str(arr) + '.csv'), header=0)
        del df[df.columns[0]]

        params_file = "configs/params." + file + ".yml"
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
            
        if params['method'] == 'Exgboost':
            params['space'] = 'original'
        
        for idarray in range(arr):
            params['seed'] = 42
            params['max_depth'] = 10
            params['n_estimators'] = 50

            params["seed"] = params["seed"] + params["class_label"]
            set_seeds(params["seed"])
            
            params['method'] = method[0] + str(idarray)
            params = config_array(params, df.iloc[idarray])

            if file == 'UCIdata' or file == 'UCIdataE':
                run_train_UCI(params, params_file)
            else:
                run_train(params, params_file)
tune_main()