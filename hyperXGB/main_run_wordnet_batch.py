#!/usr/bin/env python3
"""
Run the WordNet dataset experiments with the batch version of the hyperutils implementation.
This script demonstrates using hyperutils_batch.py instead of the regular hyperutils.py.
"""

import os
import sys
import yaml
import numpy as np
import xgboost as xgb
from sklearn import metrics

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import required modules
from xgb.utils import set_seeds
from xgb.hyperutils_batch import (
    custom_multiclass_obj,
    logregobj,
    multiclass_eval,
    customgobj,
    hyperobj,
    predict,
)


def main():
    # Use a fixed path for the parameter file
    params_file = os.path.join(current_dir, "configs/params.wordnet.yml")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    # Correct the path to the data directory
    params["source"] = os.path.join(current_dir, "data")

    # Configure the run
    method = params.get("method", "Pxgboost")
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])

    # Load data
    dataset_file = params["dataset_file"]
    dataset_module = __import__(dataset_file, fromlist=[""])

    # Get data
    print(f"Loading data from {params['source']}")
    train_x, train_y = dataset_module.get_training_data(
        params["source"], params["class_label"], params["seed"]
    )
    test_x, test_y = dataset_module.get_testing_data(
        params["source"], params["class_label"], params["seed"]
    )
    val_x, val_y = dataset_module.get_validation_data(
        params["source"], params["class_label"], params["seed"]
    )

    # Convert to numpy arrays
    train_x = train_x.numpy()
    test_x = test_x.numpy()
    val_x = val_x.numpy()

    # Get unique classes
    num_classes = np.unique(train_y).shape[0]
    print(f"Number of classes: {num_classes}")

    # Convert to DMatrix format
    dtrain = xgb.DMatrix(train_x, train_y)
    dval = xgb.DMatrix(val_x, val_y)
    dtest = xgb.DMatrix(test_x, test_y)

    # Set up XGBoost parameters
    xgb_params = {
        "max_depth": params.get("max_depth", 6),
        "eta": params.get("eta", 0.1),
        "gamma": params.get("gamma", 0.0),
        "subsample": params.get("subsample", 0.7),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "colsample_bylevel": params.get("colsample_bylevel", 1.0),
        "num_class": num_classes,
        "disable_default_eval_metric": True,
    }

    print(f"Training with method: {method}")
    print(f"XGBoost parameters: {xgb_params}")

    # Select the objective function based on the method
    if method == "Exgboost":
        obj_func = logregobj
    elif method == "Pxgboost":
        obj_func = customgobj
    else:
        # Default to the hyperbolic version
        obj_func = hyperobj

    # Train the model
    num_boost_round = params.get("round", 100)
    evals_result = {}

    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=obj_func,
        custom_metric=multiclass_eval,
        evals_result=evals_result,
        evals=[(dval, "validation")],
        verbose_eval=10,
    )

    # Make predictions
    preds = predict(booster, dtest)

    # Evaluate
    accuracy = metrics.accuracy_score(test_y, preds)
    f1_macro = metrics.f1_score(test_y, preds, average="macro")
    f1_micro = metrics.f1_score(test_y, preds, average="micro")

    print("\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")

    # Save the model if needed
    output_path = params.get("output_path", "./logs/output_wordnet_batch")
    os.makedirs(output_path, exist_ok=True)
    booster.save_model(os.path.join(output_path, f"model_{method}.json"))

    # Log results to file
    with open(os.path.join(output_path, "results.txt"), "a") as f:
        f.write(f"Method: {method}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Micro: {f1_micro:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
