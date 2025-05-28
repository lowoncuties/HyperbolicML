"""
Basic Usage Example for HyperbolicML

This example demonstrates how to use the HyperbolicXGBoost API for
classification tasks with different manifold types.
"""

import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the HyperbolicML API
from HyperbolicML import HyperbolicXGBoost, HyperbolicClassifier


def example_automatic_manifold_selection():
    """Example showing automatic manifold selection."""
    print("=== Automatic Manifold Selection ===")

    # Generate a dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model with automatic manifold selection
    model = HyperbolicXGBoost(
        manifold="auto", n_estimators=100, random_state=42, learning_rate=0.1
    )

    # Fit the model
    print("Training model...")
    model.fit(X_train, y_train, verbose=True)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Selected manifold: {model.manifold_}")
    print(f"Number of classes: {model.n_classes_}")
    print()


def example_manifold_comparison():
    """Example comparing different manifolds on the same dataset."""
    print("=== Manifold Comparison ===")

    # Generate a dataset
    X, y = make_blobs(n_samples=800, centers=4, n_features=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    manifolds = ["euclidean", "poincare", "hyperboloid"]
    results = {}

    for manifold in manifolds:
        print(f"Training with {manifold} manifold...")

        model = HyperbolicClassifier(
            manifold=manifold, n_estimators=50, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        results[manifold] = accuracy
        print(f"{manifold.capitalize()} accuracy: {accuracy:.4f}")

    # Show results
    print("\n=== Results Summary ===")
    best_manifold = max(results, key=results.get)
    for manifold, accuracy in results.items():
        marker = " ⭐" if manifold == best_manifold else ""
        print(f"{manifold.capitalize()}: {accuracy:.4f}{marker}")
    print()


def example_with_validation():
    """Example showing early stopping with validation set."""
    print("=== Early Stopping Example ===")

    # Generate a larger dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_classes=5,
        n_informative=20,
        random_state=42,
    )

    # Split into train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # Create model with early stopping
    model = HyperbolicXGBoost(
        manifold="poincare",
        n_estimators=200,
        early_stopping_rounds=10,
        learning_rate=0.1,
        random_state=42,
    )

    # Fit with validation set
    print("Training with early stopping...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Final test accuracy: {accuracy:.4f}")

    # Show feature importance
    importances = model.feature_importances_
    top_features = np.argsort(importances)[-5:][::-1]
    print("Top 5 most important features:")
    for i, idx in enumerate(top_features):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    print()


def example_custom_parameters():
    """Example showing custom XGBoost parameters."""
    print("=== Custom Parameters Example ===")

    X, y = make_classification(
        n_samples=500, n_features=15, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create model with custom parameters
    model = HyperbolicXGBoost(
        manifold="hyperboloid",
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
    )

    print("Training with custom parameters...")
    model.fit(X_train, y_train, verbose=True)

    # Detailed evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Prediction probabilities shape: {y_proba.shape}")
    print(f"Average confidence: {np.mean(np.max(y_proba, axis=1)):.4f}")


if __name__ == "__main__":
    print("HyperbolicML Basic Usage Examples")
    print("=" * 50)
    print()

    # Run all examples
    example_automatic_manifold_selection()
    example_manifold_comparison()
    example_with_validation()
    example_custom_parameters()

    print("All examples completed successfully! 🎉")
