#!/usr/bin/env python3
"""
Compare the performance of the regular and batch implementations of hyperutils
on various datasets. This script runs both implementations and evaluates their
performance using various metrics, saving the results to an Excel file.
"""

import os
import sys
import time
import glob
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics

# Add the current directory to Python path for local module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import local modules after sys.path modification
from xgb.hyperutils import logregobj as regular_logregobj
from xgb.hyperutils import customgobj as regular_customgobj
from xgb.hyperutils import hyperobj as regular_hyperobj
from xgb.hyperutils import predict as regular_predict
from xgb.hyperutils_batch import logregobj as batch_logregobj
from xgb.hyperutils_batch import customgobj as batch_customgobj
from xgb.hyperutils_batch import hyperobj as batch_hyperobj
from xgb.hyperutils_batch import predict as batch_predict
from xgb.utils import set_seeds


# Custom Callback for Time-Based Early Stopping
class TrainingTimeStopper(xgb.callback.TrainingCallback):
    def __init__(self, max_allowed_training_time, min_time_for_stop=0.5):
        super().__init__()
        # Ensure max_allowed_training_time is not excessively small
        self.max_allowed_training_time = max(
            max_allowed_training_time, min_time_for_stop
        )
        self.start_time_actual_training = 0
        self.stopped_early = False
        self.rounds_completed = 0

    def before_training(self, model):
        self.start_time_actual_training = time.time()
        self.stopped_early = False  # Reset for safety if callback is reused
        self.rounds_completed = 0
        return model

    def after_iteration(self, model, epoch, evals_log):
        current_duration = time.time() - self.start_time_actual_training
        self.rounds_completed = epoch + 1  # epoch is 0-indexed
        if current_duration > self.max_allowed_training_time:
            print(
                f"    Regular model training stopped early at iter {epoch}: "
                f"exceeded {self.max_allowed_training_time:.2f}s "
                f"(curr: {current_duration:.2f}s)"
            )
            self.stopped_early = True
            return True  # Signal XGBoost to stop
        return False

    def after_training(self, model):
        # If not stopped early, rounds_completed might be based on
        # num_boost_round but self.rounds_completed from after_iteration
        # should be correct for completed rounds.
        # model.num_boosted_rounds() is also an option after training.
        return model


def load_dataset_data(params, dataset_config_name):
    """Load data from the specified dataset configuration."""
    params["source"] = os.path.join(current_dir, "data")
    params["seed"] = 42 + sum(ord(c) for c in dataset_config_name) % 100
    set_seeds(params["seed"])

    dataset_file_path = params["dataset_file"]
    dataset_module_name = dataset_file_path.replace("/", ".")
    dataset_module = __import__(dataset_module_name, fromlist=[""])

    print(
        f"Loading data for {dataset_config_name} "
        f"(using module {dataset_module_name})..."
    )

    if "class_label" in params:
        train_x, train_y = dataset_module.get_training_data(
            params["source"], params.get("class_label", None), params["seed"]
        )
        test_x, test_y = dataset_module.get_testing_data(
            params["source"], params.get("class_label", None), params["seed"]
        )
    else:
        train_x, train_y = dataset_module.get_training_data(
            params["source"], params["seed"]
        )
    test_x, test_y = dataset_module.get_testing_data(
        params["source"], params["seed"]
    )

    train_x = train_x.numpy()
    test_x = test_x.numpy()

    dtrain = xgb.DMatrix(train_x, train_y)
    dtest = xgb.DMatrix(test_x, test_y)

    n_samples = train_x.shape[0]
    unique_labels = np.unique(train_y)
    n_classes = len(unique_labels) if len(unique_labels) > 1 else 2

    if n_classes == 2 and len(unique_labels) == 1:
        print(
            f"Warning: Only one class ({unique_labels[0]}) present in train_y "
            f"for {dataset_config_name}. XGBoost setup might need adjustment."
        )

    predt_cols = n_classes if n_classes > 1 else 1
    predt = np.random.rand(n_samples, predt_cols) * 0.5

    print(
        f"Dataset {dataset_config_name} loaded: "
        f"{n_samples} samples, {n_classes} classes"
    )
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
    """Compare implementation speed and model performance metrics."""
    regular_times, batch_times = [], []
    predt_shape_cols = n_classes if n_classes > 1 else 1
    predt = np.random.rand(dtrain.num_row(), predt_shape_cols) * 0.5

    _ = regular_fn(predt, dtrain)
    _ = batch_fn(predt, dtrain)  # Warm-up obj calls

    for _ in range(n_runs):
        start = time.time()
        reg_res = regular_fn(predt, dtrain)
        regular_times.append(time.time() - start)
        start = time.time()
        batch_res = batch_fn(predt, dtrain)
        batch_times.append(time.time() - start)

    reg_mean, batch_mean = np.mean(regular_times), np.mean(batch_times)
    obj_speedup = reg_mean / batch_mean if batch_mean > 0 else float("inf")
    grad_diff = np.abs(reg_res[0] - batch_res[0]).mean()
    hess_diff = np.abs(reg_res[1] - batch_res[1]).mean()

    xgb_p = {
        "max_depth": params.get("max_depth", 6),
        "eta": params.get("eta", 0.1),
        "gamma": params.get("gamma", 0.0),
        "subsample": params.get("subsample", 0.7),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "colsample_bylevel": params.get("colsample_bylevel", 1.0),
        "disable_default_eval_metric": True,
    }
    if n_classes > 1:
        xgb_p["num_class"] = n_classes
    objective = params.get(
        "objective", "multi:softprob" if n_classes > 1 else "binary:logistic"
    )
    xgb_p["objective"] = objective

    num_boost_round = params.get("round", 30)
    common_train_args = {
        "params": xgb_p,
        "dtrain": dtrain,
        "num_boost_round": num_boost_round,
        "verbose_eval": False,
    }

    # Train batch model first to get its training time
    start_batch_train = time.time()
    batch_booster = xgb.train(obj=batch_fn, **common_train_args)
    batch_train_time = time.time() - start_batch_train

    # Prepare for regular model training with time-based stopping
    max_regular_allowed_time = batch_train_time * 50
    # Add a minimum sensible time, e.g. 1 sec, if batch time is tiny.
    # Or more rounds-based, e.g., ensure at least N rounds can run
    # if batch is super fast.
    # For now, direct 50x as requested, with callback's internal
    # min_time_for_stop.
    time_stopper_cb = TrainingTimeStopper(
        max_allowed_training_time=max_regular_allowed_time
    )

    start_reg_train = time.time()
    # xgb.train will handle early stopping if the callback returns True
    reg_booster = xgb.train(
        obj=regular_fn, callbacks=[time_stopper_cb], **common_train_args
    )
    # No try-except block needed here for EarlyStopException

    regular_train_time = time.time() - start_reg_train
    regular_model_stopped_early = time_stopper_cb.stopped_early
    # Get actual rounds from booster, as callback might not be perfect
    # if num_boost_round is 0
    regular_model_rounds_completed = (
        reg_booster.num_boosted_rounds()
        if hasattr(reg_booster, "num_boosted_rounds")
        else time_stopper_cb.rounds_completed
    )

    reg_preds_raw = regular_predict(reg_booster, dtest)
    batch_preds_raw = batch_predict(batch_booster, dtest)

    if objective == "multi:softprob":
        if (
            reg_preds_raw.ndim > 1 and reg_preds_raw.shape[1] > 1
        ):  # Check for actual multi-class output
            reg_preds = np.argmax(reg_preds_raw, axis=1)
        else:  # Binary case or 1D probability output
            reg_preds = (reg_preds_raw > 0.5).astype(int)

        if batch_preds_raw.ndim > 1 and batch_preds_raw.shape[1] > 1:
            batch_preds = np.argmax(batch_preds_raw, axis=1)
        else:
            batch_preds = (batch_preds_raw > 0.5).astype(int)

    elif objective == "binary:logistic":
        reg_preds = (reg_preds_raw > 0.5).astype(int)
        batch_preds = (batch_preds_raw > 0.5).astype(int)
    else:
        reg_preds, batch_preds = reg_preds_raw, batch_preds_raw

    m_args = {"average": "macro", "zero_division": 0}
    reg_acc = metrics.accuracy_score(test_y, reg_preds)
    reg_f1M = metrics.f1_score(test_y, reg_preds, **m_args)
    reg_f1m = metrics.f1_score(
        test_y, reg_preds, average="micro", zero_division=0
    )
    reg_prec = metrics.precision_score(test_y, reg_preds, **m_args)
    reg_recall = metrics.recall_score(test_y, reg_preds, **m_args)
    batch_acc = metrics.accuracy_score(test_y, batch_preds)
    batch_f1M = metrics.f1_score(test_y, batch_preds, **m_args)
    batch_f1m = metrics.f1_score(
        test_y, batch_preds, average="micro", zero_division=0
    )
    batch_prec = metrics.precision_score(test_y, batch_preds, **m_args)
    batch_recall = metrics.recall_score(test_y, batch_preds, **m_args)

    train_time_speedup = (
        regular_train_time / batch_train_time
        if batch_train_time > 0
        else float("inf")
    )
    print(f"\n{name} Comparison:")
    print(
        f"  Speed: Regular {reg_mean:.5f}s, Batch {batch_mean:.5f}s, "
        f"Speedup {obj_speedup:.2f}x"
    )
    print(f"  Diffs: Grad {grad_diff:.8f}, Hess {hess_diff:.8f}")
    print(
        f"  Train Time: Regular {regular_train_time:.5f}s "
        f"(Rounds: {regular_model_rounds_completed}), "
        f"Batch {batch_train_time:.5f}s, Speedup {train_time_speedup:.2f}x"
    )
    if regular_model_stopped_early:
        print(
            "    NOTE: Regular model training was stopped early due to "
            "time limit."
        )
    print(
        f"  Regular Metrics: Acc {reg_acc:.4f}, F1-Macro {reg_f1M:.4f}, "
        f"F1-Micro {reg_f1m:.4f}, Prec {reg_prec:.4f}, Recall {reg_recall:.4f}"
    )
    print(
        f"  Batch Metrics:   Acc {batch_acc:.4f}, F1-Macro {batch_f1M:.4f}, "
        f"F1-Micro {batch_f1m:.4f}, Prec {batch_prec:.4f}, "
        f"Recall {batch_recall:.4f}"
    )

    return {
        "name": name,
        "regular_obj_time_mean": reg_mean,
        "batch_obj_time_mean": batch_mean,
        "obj_speedup": obj_speedup,
        "grad_diff": grad_diff,
        "hess_diff": hess_diff,
        "sample_size": dtrain.num_row(),
        "regular_train_time": regular_train_time,
        "regular_model_stopped_early": regular_model_stopped_early,
        "regular_model_rounds_completed": regular_model_rounds_completed,
        "batch_train_time": batch_train_time,
        "train_speedup": train_time_speedup,
        "regular_accuracy": reg_acc,
        "regular_f1_macro": reg_f1M,
        "regular_f1_micro": reg_f1m,
        "regular_precision": reg_prec,
        "regular_recall": reg_recall,
        "batch_accuracy": batch_acc,
        "batch_f1_macro": batch_f1M,
        "batch_f1_micro": batch_f1m,
        "batch_precision": batch_prec,
        "batch_recall": batch_recall,
        "n_classes": n_classes,
    }


def plot_results(results, output_path):
    """Create plots to visualize the performance comparison."""
    if not results:
        print("No results to plot.")
        return
    plt.figure(figsize=(16, 14))
    model_types = sorted(list(set(r["name"].split(" (")[0] for r in results)))

    plot_configs = [
        (
            "Obj. Speedup vs Sample Size",
            "obj_speedup",
            "Sample Size",
            "Speedup (x)",
            False,
            True,
            [],
        ),
        (
            "Obj. Execution Time (Log)",
            ["regular_obj_time_mean", "batch_obj_time_mean"],
            "Time (s, log)",
            "Model Type",
            True,
            False,
            ["Regular", "Batch"],
        ),
        (
            "Avg Model Accuracy",
            ["regular_accuracy", "batch_accuracy"],
            "Avg Accuracy",
            "Model Type",
            True,
            False,
            ["Regular", "Batch"],
        ),
        (
            "Avg F1-Macro",
            ["regular_f1_macro", "batch_f1_macro"],
            "Avg F1-Macro",
            "Model Type",
            True,
            False,
            ["Regular", "Batch"],
        ),
    ]

    for i, (
        title,
        y_keys,
        xlabel,
        ylabel,
        is_bar,
        is_line,
        bar_legends,
    ) in enumerate(plot_configs):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        if is_line:
            for mt in model_types:
                mt_res = sorted(
                    [
                        r
                        for r in results
                        if r["name"].startswith(mt)
                        and y_keys in r
                        and isinstance(r.get(y_keys), (int, float))
                        and not np.isnan(r.get(y_keys))
                    ],
                    key=lambda x: x["sample_size"],
                )
                if mt_res:
                    plt.plot(
                        [r["sample_size"] for r in mt_res],
                        [r[y_keys] for r in mt_res],
                        marker="o",
                        label=mt,
                    )
            if any(model_types):
                plt.legend()
        elif is_bar:
            bar_lbls, d_reg, d_batch = [], [], []
            for mt in model_types:
                mt_r = [
                    r
                    for r in results
                    if r["name"].startswith(mt)
                    and all(
                        isinstance(r.get(k), (int, float))
                        and not np.isnan(r.get(k))
                        for k in y_keys
                    )
                ]
                if mt_r:
                    bar_lbls.append(mt)
                    d_reg.append(np.mean([r[y_keys[0]] for r in mt_r]))
                    d_batch.append(np.mean([r[y_keys[1]] for r in mt_r]))
            if bar_lbls:
                x = np.arange(len(bar_lbls))
                width = 0.35
                plt.bar(x - width / 2, d_reg, width, label=bar_legends[0])
                plt.bar(x + width / 2, d_batch, width, label=bar_legends[1])
                plt.xticks(x, bar_lbls, rotation=45, ha="right")
                plt.legend()
                if "log" in title.lower():
                    plt.yscale("log")
                if any(s in title for s in ["Accuracy", "F1"]):
                    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "summary_plot1.png"))
    plt.close()
    print(f"Plot1: {os.path.join(output_path, 'summary_plot1.png')}")

    plt.figure(figsize=(16, 10))
    plot_configs_2 = [
        (
            "Avg Precision & Recall",
            (
                [
                    ("regular_precision", "batch_precision"),
                    ("regular_recall", "batch_recall"),
                ]
            ),
            "Avg Score",
            ["Reg Prec", "Batch Prec", "Reg Recall", "Batch Recall"],
        ),
        (
            "Avg Training Time (Log)",
            ([("regular_train_time", "batch_train_time")]),
            "Avg Train Time (s, log)",
            ["Regular", "Batch"],
        ),
        (
            "Avg Training Speedup",
            "train_speedup",
            "Avg Speedup (x)",
            ["Speedup"],
        ),
        (
            "Avg F1-Micro",
            ([("regular_f1_micro", "batch_f1_micro")]),
            "Avg F1-Micro",
            ["Regular", "Batch"],
        ),
    ]

    for i, (title, y_key_config, ylabel, bar_legends) in enumerate(
        plot_configs_2
    ):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Model Type")
        plt.grid(True, alpha=0.3)
        bar_labels_mt, data_for_bars = [], []
        is_single_metric_bar = isinstance(y_key_config, str)

        for mt in model_types:
            mt_results = [r for r in results if r["name"].startswith(mt)]
            if not mt_results:
                continue
            current_mt_bar_values, valid_mt_data = [], True
            if is_single_metric_bar:
                vals = [
                    r.get(y_key_config)
                    for r in mt_results
                    if isinstance(r.get(y_key_config), (int, float))
                    and not np.isnan(r.get(y_key_config))
                    and r.get(y_key_config) != float("inf")
                ]
                if not vals:
                    valid_mt_data = False
                else:
                    current_mt_bar_values.append(np.mean(vals))
            else:
                for y_keys_pair in y_key_config:
                    for y_key in y_keys_pair:
                        vals = [
                            r.get(y_key)
                            for r in mt_results
                            if isinstance(r.get(y_key), (int, float))
                            and not np.isnan(r.get(y_key))
                        ]
                        if not vals:
                            valid_mt_data = False
                            break
                        current_mt_bar_values.append(np.mean(vals))
                    if not valid_mt_data:
                        break
            if valid_mt_data and current_mt_bar_values:
                bar_labels_mt.append(mt)
                data_for_bars.append(current_mt_bar_values)

        if not bar_labels_mt:
            plt.text(0.5, 0.5, "No data", ha="center")
            continue
        x = np.arange(len(bar_labels_mt))
        num_metric_sets = len(data_for_bars[0]) if data_for_bars else 0
        if num_metric_sets == 0:
            plt.text(0.5, 0.5, "No data for bars", ha="center")
            continue  # Should be caught by bar_labels_mt check

        bar_width = (0.8 / num_metric_sets) if num_metric_sets > 0 else 0.8
        for bar_idx in range(num_metric_sets):
            current_set_values = [
                model_data[bar_idx] for model_data in data_for_bars
            ]
            bar_offset = (bar_idx - (num_metric_sets - 1) / 2) * bar_width
            plt.bar(
                x + bar_offset,
                current_set_values,
                bar_width,
                label=bar_legends[bar_idx],
            )

        plt.xticks(x, bar_labels_mt, rotation=45, ha="right")
        if any(bar_legends):
            plt.legend()
        if "log" in title.lower():
            plt.yscale("log")
        if any(s in title for s in ["Accuracy", "F1", "Precision", "Recall"]):
            plt.ylim(0, 1.05)
        if "Speedup" in title and is_single_metric_bar:
            plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "summary_plot2.png"))
    plt.close()
    print(f"Plot2: {os.path.join(output_path, 'summary_plot2.png')}")


def save_results_to_excel(
    results, output_path, filename="comparison_results.xlsx"
):
    """Saves the collected results to an Excel file."""
    if not results:
        print("No results to save.")
        return
    df = pd.DataFrame(results)
    if "name" not in df.columns:
        print("Skipping Excel: 'name' column missing.")
        return

    df[["Model_Type", "Dataset_Full"]] = df["name"].str.split(
        " \(", expand=True
    )
    df["Dataset_Full"] = df["Dataset_Full"].str.replace("\)$", "", regex=True)
    try:
        split_ds = df["Dataset_Full"].str.rsplit("-", n=2, expand=True)
        df["Dataset_Base"] = split_ds[0]
        df["Dataset_Variant"] = split_ds[1] if len(split_ds.columns) > 1 else ""
        df["Dataset_Label_Info"] = (
            split_ds[2] if len(split_ds.columns) > 2 else ""
        )
    except Exception:
        df["Dataset_Base"], df["Dataset_Variant"], df["Dataset_Label_Info"] = (
            df["Dataset_Full"],
            "",
            "",
        )

    cols = [
        "Model_Type",
        "Dataset_Full",
        "Dataset_Base",
        "Dataset_Variant",
        "Dataset_Label_Info",
        "sample_size",
        "n_classes",
        "obj_speedup",
        "regular_train_time",
        "regular_model_stopped_early",
        "regular_model_rounds_completed",
        "batch_train_time",
        "train_speedup",
        "grad_diff",
        "hess_diff",
        "regular_accuracy",
        "batch_accuracy",
        "regular_f1_macro",
        "batch_f1_macro",
        "regular_f1_micro",
        "batch_f1_micro",
        "regular_precision",
        "batch_precision",
        "regular_recall",
        "batch_recall",
    ]
    # Ensure all desired columns exist, adding them with NaN if not present,
    # and order them
    current_cols = df.columns.tolist()
    # final_cols was unused, so removed its calculation.
    # df.reindex will add any missing columns from 'cols' with NaN.
    df = df.reindex(columns=cols)

    excel_path = os.path.join(output_path, filename)
    try:
        df.to_excel(excel_path, index=False, sheet_name="ComparisonResults")
        print(f"Results saved to {excel_path}")
    except Exception as e:
        print(
            f"Error saving to Excel ({excel_path}): {e}. "
            "Is openpyxl installed?"
        )


def main():
    print("Comparing implementations across datasets...")
    config_path = os.path.join(current_dir, "configs")
    config_files = glob.glob(os.path.join(config_path, "params.*.yml"))
    if not config_files:
        print(f"No config files in {config_path}")
        return

    all_results = []
    for cfg_file in config_files:
        cfg_name = (
            os.path.basename(cfg_file)
            .replace("params.", "")
            .replace(".yml", "")
        )
        print(f"\n\n===== PROCESSING DATASET: {cfg_name} =====")
        with open(cfg_file, "r") as f:
            params = yaml.safe_load(f)
        try:
            dtrain, dtest, _, test_y, n_cl = load_dataset_data(params, cfg_name)
        except Exception as e:
            print(f"Error loading {cfg_name}: {e}. Skipping.")
            err_res = {
                "name": f"ERROR ({cfg_name})",
                "error_message": str(e),
                "sample_size": 0,
            }
            num_fields = [
                "obj_speedup",
                "regular_train_time",
                "regular_model_stopped_early",
                "regular_model_rounds_completed",
                "batch_train_time",
                "train_speedup",
                "grad_diff",
                "hess_diff",
                "regular_accuracy",
                "batch_accuracy",
                "regular_f1_macro",
                "batch_f1_macro",
                "regular_f1_micro",
                "batch_f1_micro",
                "regular_precision",
                "batch_precision",
                "regular_recall",
                "batch_recall",
                "n_classes",
            ]
            for k in num_fields:
                err_res[k] = float("nan")
            all_results.append(err_res)
            continue

        model_tests = [
            ("Logistic Regression", regular_logregobj, batch_logregobj),
            ("Poincare", regular_customgobj, batch_customgobj),
            ("Hyperboloid", regular_hyperobj, batch_hyperobj),
        ]
        for model_desc, reg_fn, batch_fn in model_tests:
            model_name = f"{model_desc} ({cfg_name})"
            print(f"\n--- Testing {model_name} ---")
            all_results.append(
                compare_speed_and_metrics(
                    dtrain,
                    dtest,
                    test_y,
                    n_cl,
                    model_name,
                    reg_fn,
                    batch_fn,
                    params,
                )
            )

    output_dir = os.path.join(current_dir, "logs", "all_datasets_comparison")
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_excel(all_results, output_dir)

    valid_results = [
        r
        for r in all_results
        if "error_message" not in r and r.get("sample_size", 0) > 0
    ]
    if valid_results:
        plot_results(valid_results, output_dir)
    else:
        print("No valid results with sample_size > 0 to plot.")

    txt_path = os.path.join(output_dir, "comprehensive_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Comprehensive Comparison:\n" + "=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"Entry: {r.get('name', 'N/A')}\n")
            if "error_message" in r:
                f.write(f"  ERROR: {r['error_message']}\n\n")
                continue
            f.write("-" * 40 + "\n")

            def fmt_m(
                lbl,
                val,
                u="s",
                fmt=".5f",
                is_speedup=False,
                is_bool=False,
                is_int=False,
            ):
                if is_bool:
                    v_str = str(val)
                elif is_int:
                    v_str = (
                        str(int(val))
                        if isinstance(val, (int, float)) and not np.isnan(val)
                        else str(val)
                    )
                elif isinstance(val, (int, float)) and not np.isnan(val):
                    v_str = f"{val:{fmt}}"
                else:
                    v_str = str(val)
                u_str = (
                    "x"
                    if is_speedup
                    else (
                        u
                        if val is not None
                        and not (isinstance(val, float) and np.isnan(val))
                        and not is_bool
                        and not is_int
                        else ""
                    )
                )
                return f"  {lbl:<28}: {v_str} {u_str}\n"

            f.write(fmt_m("Reg. Obj Time", r.get("regular_obj_time_mean")))
            f.write(fmt_m("Batch Obj Time", r.get("batch_obj_time_mean")))
            f.write(
                fmt_m(
                    "Obj Speedup",
                    r.get("obj_speedup"),
                    fmt=".2f",
                    is_speedup=True,
                )
            )
            f.write(fmt_m("Grad Diff", r.get("grad_diff"), u="", fmt=".8f"))
            f.write(
                fmt_m("Hess Diff", r.get("hess_diff"), u="", fmt=".8f") + "\n"
            )

            f.write(fmt_m("Reg. Train Time", r.get("regular_train_time")))
            f.write(
                fmt_m(
                    "Reg. Stopped Early",
                    r.get("regular_model_stopped_early"),
                    u="",
                    is_bool=True,
                )
            )
            f.write(
                fmt_m(
                    "Reg. Rounds Completed",
                    r.get("regular_model_rounds_completed"),
                    u="",
                    is_int=True,
                )
            )
            f.write(fmt_m("Batch Train Time", r.get("batch_train_time")))
            f.write(
                fmt_m(
                    "Train Speedup",
                    r.get("train_speedup"),
                    fmt=".2f",
                    is_speedup=True,
                )
                + "\n"
            )

            for rt_prefix in ["regular", "batch"]:
                f.write(f"{rt_prefix.capitalize()} Metrics:\n")
                f.write(
                    fmt_m(
                        "  Accuracy",
                        r.get(f"{rt_prefix}_accuracy"),
                        u="",
                        fmt=".4f",
                    )
                )
                f.write(
                    fmt_m(
                        "  F1-Macro",
                        r.get(f"{rt_prefix}_f1_macro"),
                        u="",
                        fmt=".4f",
                    )
                )
                f.write(
                    fmt_m(
                        "  F1-Micro",
                        r.get(f"{rt_prefix}_f1_micro"),
                        u="",
                        fmt=".4f",
                    )
                )
                f.write(
                    fmt_m(
                        "  Precision",
                        r.get(f"{rt_prefix}_precision"),
                        u="",
                        fmt=".4f",
                    )
                )
                f.write(
                    fmt_m(
                        "  Recall",
                        r.get(f"{rt_prefix}_recall"),
                        u="",
                        fmt=".4f",
                    )
                    + ("\n" if rt_prefix == "batch" else "")
                )
            f.write(
                fmt_m(
                    "Sample Size",
                    r.get("sample_size"),
                    u="",
                    fmt="",
                    is_int=True,
                )
            )
            f.write(
                fmt_m(
                    "Num Classes", r.get("n_classes"), u="", fmt="", is_int=True
                )
                + "\n\n"
            )
    print(f"Text summary: {txt_path}")

    print("\nOverall Performance Summary (Averages per Model Type):")
    if not valid_results:
        print("No valid results to summarize.")
    else:
        model_types_summary = sorted(
            list(set(r["name"].split(" (")[0] for r in valid_results))
        )
        for mt in model_types_summary:
            mt_r = [r for r in valid_results if r["name"].startswith(mt)]
            if not mt_r:
                continue

            def mean_if_num(vals):
                V = [
                    v
                    for v in vals
                    if isinstance(v, (int, float))
                    and not np.isnan(v)
                    and v != float("inf")
                ]
                return np.mean(V) if V else float("nan")

            print(f"\n  {mt}:")
            obj_speedup_avg = mean_if_num([r.get("obj_speedup") for r in mt_r])
            train_speedup_avg = mean_if_num(
                [r.get("train_speedup") for r in mt_r]
            )
            reg_acc_avg = mean_if_num([r.get("regular_accuracy") for r in mt_r])
            batch_acc_avg = mean_if_num([r.get("batch_accuracy") for r in mt_r])
            reg_f1_avg = mean_if_num([r.get("regular_f1_macro") for r in mt_r])
            batch_f1_avg = mean_if_num([r.get("batch_f1_macro") for r in mt_r])

            # Avg rounds for regular models, and how many were stopped early
            avg_reg_rounds = mean_if_num(
                [r.get("regular_model_rounds_completed") for r in mt_r]
            )
            num_stopped_early = sum(
                1 for r in mt_r if r.get("regular_model_stopped_early") is True
            )
            total_reg_models = len(
                [
                    r
                    for r in mt_r
                    if r.get("regular_model_rounds_completed") is not None
                ]
            )

            print(f"    Avg. Obj Speedup:    {obj_speedup_avg:.2f}x")
            print(f"    Avg. Train Speedup:  {train_speedup_avg:.2f}x")
            if total_reg_models > 0:
                print(
                    f"    Avg. Reg. Rounds:    {avg_reg_rounds:.1f} "
                    f"({num_stopped_early}/{total_reg_models} stopped early)"
                )
            print(
                f"    Avg. Accuracy (Reg): {reg_acc_avg:.4f} / "
                f"(Batch): {batch_acc_avg:.4f}"
            )
            print(
                f"    Avg. F1-Macro (Reg): {reg_f1_avg:.4f} / "
                f"(Batch): {batch_f1_avg:.4f}"
            )
    print("\nScript finished.")


if __name__ == "__main__":
    main()
