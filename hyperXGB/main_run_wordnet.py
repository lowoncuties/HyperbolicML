#!/usr/bin/env python3
"""
Run the WordNet dataset experiments with correct Python path setup.
This script ensures the packages can be properly imported.
"""

import os
import sys
import yaml

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import required modules
from xgb.utils import set_seeds
from xgb.hyper_trainer import run_train_UCI


def main():
    # Use a fixed path for the parameter file
    params_file = os.path.join(current_dir, "configs/params.wordnet.yml")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    # Correct the path to the data directory
    # Change 'data//wordnet//' to the correct path
    params["source"] = os.path.join(current_dir, "data")

    # Configure the run
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])

    # Run the training
    run_train_UCI(params, params_file)


if __name__ == "__main__":
    main()
