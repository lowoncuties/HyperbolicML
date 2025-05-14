# HyperXGB Experiment Logs & Analysis

This directory contains the output logs, results, and plots from running the `test_all_datasets.py` script located in the parent `hyperXGB` directory.

## 1. Testing Methodology

The primary goal of these experiments is to compare the performance and speed of different custom objective function implementations for XGBoost, specifically focusing on:

*   **Regular (Iterative) Implementations**: Custom objectives where calculations (e.g., for gradients and Hessians) are often performed in Python loops, typically found in `xgb/hyperutils.py`.
*   **Batch (Vectorized) Implementations**: Custom objectives optimized for speed using NumPy vectorization for calculations, found in `xgb/hyperutils_batch.py`.

Comparisons are made across various datasets and geometric manifolds:

*   **Datasets**: The script iterates through all `params.*.yml` configuration files found in the `hyperXGB/configs/` directory. Each configuration file defines parameters for a specific dataset and its loading mechanism.
*   **Geometric Manifolds/Objectives**:
    *   **Logistic Regression (Euclidean)**: Standard softmax objective for multi-class classification (or logistic for binary).
    *   **Poincaré Ball**: Custom objective designed for hyperbolic geometry using the Poincaré ball model.
    *   **Hyperboloid Model**: Custom objective designed for hyperbolic geometry using the Hyperboloid (Lorentz) model.

For each dataset and objective combination, both the "regular" and "batch" versions are trained and evaluated.

### Metrics Collected:
The script collects a comprehensive set of metrics, including:
*   **Speed Metrics**:
    *   Mean time to compute the objective function (gradient and Hessian) for regular and batch versions.
    *   Speedup of batch objective computation over regular.
    *   Total training time for regular and batch models.
    *   Speedup of batch training time over regular.
*   **Gradient & Hessian Differences**:
    *   Mean absolute difference between gradients computed by regular and batch methods.
    *   Mean absolute difference between Hessians computed by regular and batch methods.
*   **Model Performance Metrics** (for both regular and batch trained models):
    *   Accuracy
    *   F1-Score (Macro and Micro averages)
    *   Precision (Macro average)
    *   Recall (Macro average)
*   **Dataset Information**:
    *   Sample size
    *   Number of classes

## 2. Early Stopping for Regular Models

A critical aspect of the testing methodology is the implementation of an early stopping mechanism for the "regular" model training.

*   **Criterion**: The training of a regular model is stopped if its cumulative training time exceeds **50 times** the total training time of the corresponding batch model (for the same dataset and objective function).
*   **Justification**: On some datasets, the regular (iterative) implementations can be exceptionally slow, potentially taking hundreds of minutes to train on machines like a MacBook Pro M3. The 50x threshold was chosen as a practical limit to prevent excessively long experiment runtimes while still allowing the regular model to train for a reasonable duration if it's not drastically slower.
*   **Outcome**: If a regular model's training is stopped early:
    *   The model trained up to that point (i.e., a partially trained model) is used for inference and metric calculation.
    *   The results will indicate that the model was stopped early and the number of training rounds completed.

## 3. Output Files

All outputs from the `test_all_datasets.py` script are saved in a subdirectory within this `logs` folder, typically named `all_datasets_comparison/`. This subdirectory contains:

*   **`comparison_results.xlsx`**: An Excel spreadsheet containing all the collected metrics in a structured format. Each row represents a specific model (e.g., "Poincare"), dataset configuration, and implementation type (regular/batch info combined).
*   **`summary_plot1.png` & `summary_plot2.png`**: PNG image files containing various plots that visualize the comparison results, such as speedups, execution times, and performance metrics across different model types and sample sizes.
*   **`comprehensive_summary.txt`**: A text file providing a detailed summary of all results for each test run, including all metrics and whether regular models were stopped early.

## 4. Results (Example of one run)

```text
Overall Performance Summary (Averages per Model Type):

  Hyperboloid:
    Avg. Obj Speedup:    917.95x
    Avg. Train Speedup:  43.38x
    Avg. Reg. Rounds:    46.0 (2/3 stopped early)
    Avg. Accuracy (Reg): 0.9562 / (Batch): 0.9567
    Avg. F1-Macro (Reg): 0.8634 / (Batch): 0.8647

  Logistic Regression:
    Avg. Obj Speedup:    130.48x
    Avg. Train Speedup:  20.51x
    Avg. Reg. Rounds:    79.0 (1/3 stopped early)
    Avg. Accuracy (Reg): 0.9567 / (Batch): 0.9567
    Avg. F1-Macro (Reg): 0.8649 / (Batch): 0.8648

  Poincare:
    Avg. Obj Speedup:    383.62x
    Avg. Train Speedup:  31.73x
    Avg. Reg. Rounds:    71.0 (1/3 stopped early)
    Avg. Accuracy (Reg): 0.9333 / (Batch): 0.9214
    Avg. F1-Macro (Reg): 0.8424 / (Batch): 0.8342
```

## 5. How to Run the Experiments

To regenerate the results found in this logs directory:

1.  Ensure you have all necessary Python dependencies installed, including:
    `xgboost`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml`, `openpyxl`.
2.  Navigate to the `HyperbolicML/hyperXGB/` directory in your terminal.
3.  Execute the main script:
    ```bash
    python test_all_datasets.py
    ```
4.  The script will process all dataset configurations and save the outputs in the `HyperbolicML/hyperXGB/logs/all_datasets_comparison/` directory.


## 6. Visualizing Results: Understanding the Plots

The script generates two main summary plots, `summary_plot1.png` and `summary_plot2.png`, located in the `all_datasets_comparison/` subdirectory (relative to this `logs` directory). These plots provide a visual overview of the performance and speed comparisons. 

### Example Plots

**`Summary Plot 1: Objective Speedups, Execution Times, Accuracy, F1-Macro`**

![Summary Plot 1: Objective Speedups, Execution Times, Accuracy, F1-Macro](all_datasets_comparison/summary_plot1.png)

**`Summary Plot 2: Precision/Recall, Training Times, Training Speedups, F1-Micro`**

![Summary Plot 2: Precision/Recall, Training Times, Training Speedups, F1-Micro](all_datasets_comparison/summary_plot2.png)

---
This README provides a snapshot of the testing setup and observations. For the most detailed and up-to-date information, always refer to the source code of `test_all_datasets.py` and the helper utility scripts.