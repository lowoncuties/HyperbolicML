#!/usr/bin/env python3
"""
Compare the performance of the regular and batch implementations of hyperutils.
This script runs both implementations on the WordNet dataset and evaluates their
performance using various metrics.
"""

import os
import sys
import time
import yaml
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import both implementations for comparison
from xgb.hyperutils import logregobj as regular_logregobj
from xgb.hyperutils import customgobj as regular_customgobj
from xgb.hyperutils import hyperobj as regular_hyperobj
from xgb.hyperutils import predict as regular_predict

from xgb.hyperutils_batch import logregobj as batch_logregobj
from xgb.hyperutils_batch import customgobj as batch_customgobj
from xgb.hyperutils_batch import hyperobj as batch_hyperobj
from xgb.hyperutils_batch import predict as batch_predict

from xgb.utils import set_seeds


def load_wordnet_data(params, label_idx):
    """Load data from the WordNet dataset with the specified label."""
    # Configure path
    params["source"] = os.path.join(current_dir, "data")
    params["class_label"] = label_idx

    # Set seed for reproducibility
    params["seed"] = 42 + label_idx
    set_seeds(params["seed"])

    # Import dataset module
    dataset_file = params["dataset_file"]
    dataset_module = __import__(dataset_file, fromlist=[""])

    # Get the data
    print(f"Loading WordNet data (label {label_idx})...")
    train_x, train_y = dataset_module.get_training_data(
        params["source"], params["class_label"], params["seed"]
    )

    test_x, test_y = dataset_module.get_testing_data(
        params["source"], params["class_label"], params["seed"]
    )

    # Convert to numpy arrays
    train_x = train_x.numpy()
    test_x = test_x.numpy()

    # Create DMatrix
    dtrain = xgb.DMatrix(train_x, train_y)
    dtest = xgb.DMatrix(test_x, test_y)

    # Generate predictions matching the size of the train_x
    n_samples = train_x.shape[0]
    n_classes = np.unique(train_y).shape[0]
    predt = np.random.rand(n_samples, n_classes) * 0.5  # Keep values small

    print(f"Dataset loaded: {n_samples} samples, {n_classes} classes")

    return dtrain, dtest, predt, test_y, n_classes


def compare_speed_and_metrics(
    dtrain,
    dtest,
    test_y,
    n_classes,
    name,
    regular_fn,
    batch_fn,
    params,
    n_runs=5,
):
    """Compare both implementation speed and model performance metrics."""
    # --- PART 1: Compare implementation speed with gradient/hessian metrics ---
    regular_times = []
    batch_times = []
    predt = np.random.rand(dtrain.num_row(), n_classes) * 0.5

    # Warm-up run
    _ = regular_fn(predt, dtrain)
    _ = batch_fn(predt, dtrain)

    # Timing runs for the objective functions
    for _ in range(n_runs):
        start = time.time()
        regular_result = regular_fn(predt, dtrain)
        regular_times.append(time.time() - start)

        start = time.time()
        batch_result = batch_fn(predt, dtrain)
        batch_times.append(time.time() - start)

    reg_mean = np.mean(regular_times)
    batch_mean = np.mean(batch_times)
    speedup = reg_mean / batch_mean

    # Verify that both implementations produce similar results
    reg_grad, reg_hess = regular_result
    batch_grad, batch_hess = batch_result

    grad_diff = np.abs(reg_grad - batch_grad).mean()
    hess_diff = np.abs(reg_hess - batch_hess).mean()

    # --- PART 2: Compare model training and performance metrics ---
    xgb_params = {
        "max_depth": params.get("max_depth", 6),
        "eta": params.get("eta", 0.1),
        "gamma": params.get("gamma", 0.0),
        "subsample": params.get("subsample", 0.7),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "colsample_bylevel": params.get("colsample_bylevel", 1.0),
        "num_class": n_classes,
        "disable_default_eval_metric": True,
    }

    # Time training with regular implementation
    start_time = time.time()
    regular_booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params.get("round", 30),  # Reduced for speed
        obj=regular_fn,
        verbose_eval=False,
    )
    regular_train_time = time.time() - start_time

    # Time training with batch implementation
    start_time = time.time()
    batch_booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params.get("round", 30),  # Reduced for speed
        obj=batch_fn,
        verbose_eval=False,
    )
    batch_train_time = time.time() - start_time

    # Make predictions with both models
    regular_preds = regular_predict(regular_booster, dtest)
    batch_preds = batch_predict(batch_booster, dtest)

    # Calculate metrics for regular implementation
    regular_accuracy = metrics.accuracy_score(test_y, regular_preds)
    regular_f1_macro = metrics.f1_score(test_y, regular_preds, average="macro")
    regular_f1_micro = metrics.f1_score(test_y, regular_preds, average="micro")
    regular_precision = metrics.precision_score(
        test_y, regular_preds, average="macro"
    )
    regular_recall = metrics.recall_score(
        test_y, regular_preds, average="macro"
    )

    # Calculate metrics for batch implementation
    batch_accuracy = metrics.accuracy_score(test_y, batch_preds)
    batch_f1_macro = metrics.f1_score(test_y, batch_preds, average="macro")
    batch_f1_micro = metrics.f1_score(test_y, batch_preds, average="micro")
    batch_precision = metrics.precision_score(
        test_y, batch_preds, average="macro"
    )
    batch_recall = metrics.recall_score(test_y, batch_preds, average="macro")

    # Print comprehensive comparison
    print(f"\n{name} Implementation Comparison:")
    print("\n--- Speed Comparison ---")
    print(f"  Regular: {reg_mean:.5f} seconds (avg over {n_runs} runs)")
    print(f"  Batch:   {batch_mean:.5f} seconds (avg over {n_runs} runs)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Gradient mean absolute difference: {grad_diff:.8f}")
    print(f"  Hessian mean absolute difference: {hess_diff:.8f}")

    print("\n--- Training Time ---")
    print(f"  Regular: {regular_train_time:.5f} seconds")
    print(f"  Batch:   {batch_train_time:.5f} seconds")
    print(f"  Speedup: {regular_train_time/batch_train_time:.2f}x")

    print("\n--- Model Performance Metrics ---")
    print("  Regular implementation:")
    print(f"    Accuracy:  {regular_accuracy:.4f}")
    print(f"    F1-Macro:  {regular_f1_macro:.4f}")
    print(f"    F1-Micro:  {regular_f1_micro:.4f}")
    print(f"    Precision: {regular_precision:.4f}")
    print(f"    Recall:    {regular_recall:.4f}")

    print("  Batch implementation:")
    print(f"    Accuracy:  {batch_accuracy:.4f}")
    print(f"    F1-Macro:  {batch_f1_macro:.4f}")
    print(f"    F1-Micro:  {batch_f1_micro:.4f}")
    print(f"    Precision: {batch_precision:.4f}")
    print(f"    Recall:    {batch_recall:.4f}")

    return {
        "name": name,
        # Speed metrics
        "regular_mean": reg_mean,
        "batch_mean": batch_mean,
        "speedup": speedup,
        "grad_diff": grad_diff,
        "hess_diff": hess_diff,
        "sample_size": predt.shape[0],
        # Training time
        "regular_train_time": regular_train_time,
        "batch_train_time": batch_train_time,
        "train_speedup": regular_train_time / batch_train_time,
        # Regular implementation performance metrics
        "regular_accuracy": regular_accuracy,
        "regular_f1_macro": regular_f1_macro,
        "regular_f1_micro": regular_f1_micro,
        "regular_precision": regular_precision,
        "regular_recall": regular_recall,
        # Batch implementation performance metrics
        "batch_accuracy": batch_accuracy,
        "batch_f1_macro": batch_f1_macro,
        "batch_f1_micro": batch_f1_micro,
        "batch_precision": batch_precision,
        "batch_recall": batch_recall,
    }


def plot_results(results, output_path):
    """Create plots to visualize the performance comparison."""
    plt.figure(figsize=(12, 8))

    # Create a grid with 2 rows and 2 columns
    plt.subplot(2, 2, 1)

    # Group by implementation type
    types = ["Logistic Regression", "Poincare", "Hyperboloid"]
    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            sample_sizes = [r["sample_size"] for r in type_results]
            speedups = [r["speedup"] for r in type_results]
            plt.plot(sample_sizes, speedups, marker="o", label=impl_type)

    plt.xlabel("Number of Samples")
    plt.ylabel("Speedup Factor (×)")
    plt.title("Batch Implementation Speedup")
    plt.grid(True)
    plt.legend()

    # Performance comparison (regular vs batch) - subplot 2
    plt.subplot(2, 2, 2)

    # Prepare data for grouped bar chart
    labels = []
    regular_times = []
    batch_times = []

    # Pick one example for each type for the bar chart
    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_times.append(largest["regular_mean"])
            batch_times.append(largest["batch_mean"])

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, regular_times, width, label="Regular")
    plt.bar(x + width / 2, batch_times, width, label="Batch")
    plt.yscale("log")  # Log scale for better visibility
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison (Log Scale)")
    plt.xticks(x, labels)
    plt.legend()

    # Model accuracy comparison - subplot 3
    plt.subplot(2, 2, 3)

    # Prepare data for grouped bar chart
    labels = []
    regular_accs = []
    batch_accs = []

    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_accs.append(largest["regular_accuracy"])
            batch_accs.append(largest["batch_accuracy"])

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, regular_accs, width, label="Regular")
    plt.bar(x + width / 2, batch_accs, width, label="Batch")
    plt.ylim(0, 1.0)  # Accuracy scale
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(x, labels)
    plt.legend()

    # F1 Score comparison - subplot 4
    plt.subplot(2, 2, 4)

    # Prepare data for grouped bar chart
    labels = []
    regular_f1s = []
    batch_f1s = []

    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_f1s.append(largest["regular_f1_macro"])
            batch_f1s.append(largest["batch_f1_macro"])

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, regular_f1s, width, label="Regular")
    plt.bar(x + width / 2, batch_f1s, width, label="Batch")
    plt.ylim(0, 1.0)  # F1 scale
    plt.ylabel("F1-Macro Score")
    plt.title("Model F1-Macro Score Comparison")
    plt.xticks(x, labels)
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "implementation_comparison.png"))
    print(
        f"Results plot saved to {os.path.join(output_path, 'implementation_comparison.png')}"
    )

    # Create a second figure for additional metrics
    plt.figure(figsize=(14, 10))

    # Precision and Recall subplot
    plt.subplot(2, 2, 1)

    # Prepare data for grouped bar chart
    labels = []
    regular_precision = []
    batch_precision = []
    regular_recall = []
    batch_recall = []

    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_precision.append(largest["regular_precision"])
            batch_precision.append(largest["batch_precision"])
            regular_recall.append(largest["regular_recall"])
            batch_recall.append(largest["batch_recall"])

    x = np.arange(len(labels))
    width = 0.175  # Narrower bars for 4 groups

    plt.bar(
        x - 1.5 * width, regular_precision, width, label="Regular Precision"
    )
    plt.bar(x - 0.5 * width, batch_precision, width, label="Batch Precision")
    plt.bar(x + 0.5 * width, regular_recall, width, label="Regular Recall")
    plt.bar(x + 1.5 * width, batch_recall, width, label="Batch Recall")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Precision and Recall Comparison")
    plt.xticks(x, labels)
    plt.legend()

    # Training time subplot
    plt.subplot(2, 2, 2)

    # Prepare data
    labels = []
    regular_train_times = []
    batch_train_times = []
    train_speedups = []

    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_train_times.append(largest["regular_train_time"])
            batch_train_times.append(largest["batch_train_time"])
            train_speedups.append(largest["train_speedup"])

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, regular_train_times, width, label="Regular")
    plt.bar(x + width / 2, batch_train_times, width, label="Batch")
    plt.ylabel("Training Time (s)")
    plt.title("Model Training Time Comparison")
    plt.xticks(x, labels)
    plt.legend()

    # Training speedup subplot
    plt.subplot(2, 2, 3)
    plt.bar(labels, train_speedups)
    plt.ylabel("Speedup Factor")
    plt.title("Training Speedup (Regular vs Batch)")
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)

    # F1-Micro score subplot
    plt.subplot(2, 2, 4)

    # Prepare data
    labels = []
    regular_f1_micro = []
    batch_f1_micro = []

    for impl_type in types:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            # Choose the largest sample size
            largest = max(type_results, key=lambda x: x["sample_size"])
            labels.append(impl_type)
            regular_f1_micro.append(largest["regular_f1_micro"])
            batch_f1_micro.append(largest["batch_f1_micro"])

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, regular_f1_micro, width, label="Regular")
    plt.bar(x + width / 2, batch_f1_micro, width, label="Batch")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-Micro Score")
    plt.title("F1-Micro Score Comparison")
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "additional_metrics.png"))
    print(
        f"Additional metrics saved to {os.path.join(output_path, 'additional_metrics.png')}"
    )


def main():
    print(
        "Comparing regular and batch implementations using WordNet dataset..."
    )

    # Load the WordNet parameters
    params_file = os.path.join(current_dir, "configs/params.wordnet.yml")
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    # Different label indices for WordNet (using several to get different dataset sizes)
    # 1=animal, 2=group, 3=worker, 4=mammal, etc.
    label_indices = [1, 3, 4, 5]  # Reduced to save time

    # Store results for all comparisons
    results = []

    for label_idx in label_indices:
        dtrain, dtest, predt, test_y, n_classes = load_wordnet_data(
            params, label_idx
        )

        label_name = params["dataset_file"].split(".")[-1]
        dataset_name = f"WordNet-{label_name}-label{label_idx}"

        # Test with logregobj (standard Euclidean)
        print(f"\n===== Testing Logistic Regression ({dataset_name}) =====")
        results.append(
            compare_speed_and_metrics(
                dtrain,
                dtest,
                test_y,
                n_classes,
                f"Logistic Regression ({dataset_name})",
                regular_logregobj,
                batch_logregobj,
                params,
            )
        )

        # Test with customgobj (Poincaré)
        print(f"\n===== Testing Poincaré ({dataset_name}) =====")
        results.append(
            compare_speed_and_metrics(
                dtrain,
                dtest,
                test_y,
                n_classes,
                f"Poincare ({dataset_name})",
                regular_customgobj,
                batch_customgobj,
                params,
            )
        )

        # Test with hyperobj (Hyperboloid)
        print(f"\n===== Testing Hyperboloid ({dataset_name}) =====")
        results.append(
            compare_speed_and_metrics(
                dtrain,
                dtest,
                test_y,
                n_classes,
                f"Hyperboloid ({dataset_name})",
                regular_hyperobj,
                batch_hyperobj,
                params,
            )
        )

    # Create output directory if needed
    output_path = os.path.join(current_dir, "logs")
    os.makedirs(output_path, exist_ok=True)

    # Plot and save the results
    plot_results(results, output_path)

    # Save detailed results as text file
    with open(
        os.path.join(output_path, "comprehensive_comparison.txt"), "w"
    ) as f:
        f.write("WordNet Comprehensive Comparison:\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"Model: {r['name']}\n")
            f.write("=" * 40 + "\n")

            f.write("Speed Comparison:\n")
            f.write(
                f"  Regular implementation: {r['regular_mean']:.5f} seconds\n"
            )
            f.write(f"  Batch implementation: {r['batch_mean']:.5f} seconds\n")
            f.write(f"  Speedup: {r['speedup']:.2f}x\n")
            f.write(f"  Gradient difference: {r['grad_diff']:.8f}\n")
            f.write(f"  Hessian difference: {r['hess_diff']:.8f}\n\n")

            f.write("Training Time:\n")
            f.write(
                f"  Regular implementation: {r['regular_train_time']:.5f} seconds\n"
            )
            f.write(
                f"  Batch implementation: {r['batch_train_time']:.5f} seconds\n"
            )
            f.write(f"  Speedup: {r['train_speedup']:.2f}x\n\n")

            f.write("Regular Implementation Metrics:\n")
            f.write(f"  Accuracy: {r['regular_accuracy']:.4f}\n")
            f.write(f"  F1-Macro: {r['regular_f1_macro']:.4f}\n")
            f.write(f"  F1-Micro: {r['regular_f1_micro']:.4f}\n")
            f.write(f"  Precision: {r['regular_precision']:.4f}\n")
            f.write(f"  Recall: {r['regular_recall']:.4f}\n\n")

            f.write("Batch Implementation Metrics:\n")
            f.write(f"  Accuracy: {r['batch_accuracy']:.4f}\n")
            f.write(f"  F1-Macro: {r['batch_f1_macro']:.4f}\n")
            f.write(f"  F1-Micro: {r['batch_f1_micro']:.4f}\n")
            f.write(f"  Precision: {r['batch_precision']:.4f}\n")
            f.write(f"  Recall: {r['batch_recall']:.4f}\n\n")

            f.write("\n" + "=" * 60 + "\n\n")

    # Print overall summary
    print("\nOverall Performance Summary:")
    for impl_type in ["Logistic Regression", "Poincare", "Hyperboloid"]:
        type_results = [r for r in results if impl_type in r["name"]]
        if type_results:
            avg_speedup = np.mean([r["speedup"] for r in type_results])
            avg_train_speedup = np.mean(
                [r["train_speedup"] for r in type_results]
            )

            regular_acc = np.mean([r["regular_accuracy"] for r in type_results])
            batch_acc = np.mean([r["batch_accuracy"] for r in type_results])

            regular_f1 = np.mean([r["regular_f1_macro"] for r in type_results])
            batch_f1 = np.mean([r["batch_f1_macro"] for r in type_results])

            print(f"\n  {impl_type}:")
            print(f"    Operation speedup: {avg_speedup:.2f}x")
            print(f"    Training speedup: {avg_train_speedup:.2f}x")
            print(
                f"    Accuracy: {regular_acc:.4f} (regular) vs {batch_acc:.4f} (batch)"
            )
            print(
                f"    F1-Macro: {regular_f1:.4f} (regular) vs {batch_f1:.4f} (batch)"
            )


if __name__ == "__main__":
    main()
