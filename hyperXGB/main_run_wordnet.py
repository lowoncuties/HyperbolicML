import yaml
from xgb.utils import set_seeds


from xgb.hyper_trainer import  run_train_UCI
from xgb.hyper_trainer import run_train

def main():
    import sys
    # params_file = sys.argv[1]
    # print(params_file)

    #method = ['randomforest', 'hsvm', 'Exgboost', 'Pxgboost', 'SVM', 'LinearHSVM', 'randomforest', 'horoRF']
    method = [ 'tuneEPxgboost','tunesingXGB', 'tuneSVM', 'tuneRF'] # 'Exgboost', 'Pxgboost', 'EPxgboost_m',
#'UCIdata', 'UCIdataE', 'gauss', 'karate', 'polblogs', 'polbooks',
    paramfile = [ 'gausshyper',
                 'football']  # , 'wordnet'

    for med in method:
        for file in ['wordnet']:
            # Different seed for each of the 5 network embeddings
            params_file = "configs/params." + file + ".yml"
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)

            params['seed'] = 42
            params["seed"] = params["seed"] + params["class_label"]
            set_seeds(params["seed"])
            params['method'] = med

            if params['method'] == 'Exgboost':
                params['space'] = 'original'

            if file == 'UCIdata' or file == 'UCIdataE' or file == 'wordnet':
                run_train_UCI(params, params_file)
            else:
                run_train(params, params_file)


main()
