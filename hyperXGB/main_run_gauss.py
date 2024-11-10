import yaml
from xgb.utils import set_seeds


from xgb.hyper_trainer import run_train_gaussian

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

    run_train_gaussian(params, params_file)

main()
