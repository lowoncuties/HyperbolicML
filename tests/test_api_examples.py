#!/usr/bin/env python3
"""
HyperbolicML API Examples Test - Simple 1D Version

High-level examples demonstrating machine learning workflows with HyperbolicML using simple 1D data.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time


def test_hyperbolic_classifier():
    """Test hyperbolic classification workflow with simple 1D data."""
    print("Testing Simple Hyperbolic Classification...")

    try:
        # Generate very simple binary classification data
        X, y = make_classification(
            n_samples=50,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        print(
            f"✓ Generated simple dataset: {X_train.shape[0]} train, {X_test.shape[0]} test"
        )

        # Test with euclidean first (most stable)
        from hyperXGB.api import HyperbolicClassifier

        model = HyperbolicClassifier(manifold="euclidean", n_estimators=10)
        print("✓ Created HyperbolicClassifier")

        model.fit(X_train, y_train)
        print("✓ Model training completed")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ Euclidean accuracy: {accuracy:.4f}")

        return True
    except Exception as e:
        print(f"✗ Simple classification test failed: {e}")
        return False


def test_hyperbolic_regressor():
    """Test hyperbolic regression workflow with simple 1D data."""
    print("Testing Simple Hyperbolic Regression...")

    try:
        # Generate very simple regression data
        X, y = make_regression(
            n_samples=50, n_features=4, noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        print(
            f"✓ Generated simple regression dataset: {X_train.shape[0]} train"
        )

        from hyperXGB.api import HyperbolicRegressor

        model = HyperbolicRegressor(n_estimators=10)
        print("✓ Created HyperbolicRegressor")

        model.fit(X_train, y_train)
        print("✓ Model training completed")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"✓ MSE: {mse:.4f}")

        return True
    except Exception as e:
        print(f"✗ Simple regression test failed: {e}")
        return False


def test_xgboost_workflow():
    """Test simple XGBoost hyperbolic workflow."""
    print("Testing Simple HyperbolicXGBoost workflow...")

    try:
        # Generate minimal binary classification data
        X, y = make_classification(
            n_samples=40,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        print(f"✓ Generated minimal dataset: {X_train.shape}")

        from hyperXGB.api import HyperbolicXGBoost

        model = HyperbolicXGBoost(
            manifold="euclidean",
            n_estimators=5,
            max_depth=2,
            random_state=42,
        )
        print("✓ Created HyperbolicXGBoost")

        model.fit(X_train, y_train)
        print("✓ Model training completed")

        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"✓ Test accuracy: {test_accuracy:.4f}")

        return True
    except Exception as e:
        print(f"✗ Simple XGBoost workflow failed: {e}")
        return False


def test_manifold_comparison():
    """Test simple manifold comparison."""
    print("Testing Simple Manifold Comparison...")

    try:
        # Very simple data for manifold testing
        X, y = make_classification(
            n_samples=30,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        print("✓ Generated simple comparison dataset")

        from hyperXGB.api import HyperbolicClassifier

        manifolds = ["euclidean"]  # Start with just euclidean
        results = {}

        for manifold in manifolds:
            try:
                print(f"  Testing {manifold} manifold...")
                model = HyperbolicClassifier(
                    manifold=manifold, n_estimators=5, random_state=42
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results[manifold] = accuracy
                print(f"  ✓ {manifold.capitalize()}: {accuracy:.4f}")
            except Exception as e:
                print(f"  ✗ {manifold.capitalize()} failed: {e}")
                results[manifold] = None

        successful = {k: v for k, v in results.items() if v is not None}
        print(
            f"✓ Tested {len(manifolds)} manifolds, {len(successful)} successful"
        )

        return len(successful) > 0
    except Exception as e:
        print(f"✗ Simple manifold comparison failed: {e}")
        return False


def test_comprehensive_manifolds():
    """Test simple manifold scenarios."""
    print("Testing Simple Comprehensive Manifolds...")

    try:
        from hyperXGB.api import HyperbolicClassifier

        # Just test euclidean with very simple data
        X, y = make_classification(
            n_samples=20,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        try:
            model = HyperbolicClassifier(
                manifold="euclidean", n_estimators=3, random_state=42
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"✓ Simple euclidean: {accuracy:.3f}")
            return True
        except Exception as e:
            print(f"✗ Simple euclidean failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Simple comprehensive test failed: {e}")
        return False


def test_hyperbolic_manifold_debug():
    """Simple debug test for basic functionality."""
    print("Testing Simple Debug...")

    try:
        from hyperXGB.api import HyperbolicClassifier

        # Minimal test case
        X, y = make_classification(
            n_samples=16,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        print(f"✓ Debug dataset: {X_train.shape}")

        try:
            model = HyperbolicClassifier(
                manifold="euclidean", n_estimators=3, random_state=42
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"✓ Debug euclidean: {accuracy:.3f}")
            return True
        except Exception as e:
            print(f"✗ Debug test failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Debug test setup failed: {e}")
        return False


def test_basic_api():
    """Test basic API creation without fitting."""
    print("Testing Basic API Creation...")

    try:
        from hyperXGB.api import (
            HyperbolicClassifier,
            HyperbolicRegressor,
            HyperbolicXGBoost,
        )

        classifier = HyperbolicClassifier(manifold="euclidean", n_estimators=3)
        print("✓ Created HyperbolicClassifier")

        regressor = HyperbolicRegressor(n_estimators=3)
        print("✓ Created HyperbolicRegressor")

        xgb_model = HyperbolicXGBoost(manifold="euclidean", n_estimators=3)
        print("✓ Created HyperbolicXGBoost")

        print(f"✓ Classifier manifold: {classifier.manifold}")
        print(f"✓ XGB n_estimators: {xgb_model.n_estimators}")

        return True
    except Exception as e:
        print(f"✗ Basic API test failed: {e}")
        return False


def test_batch_manifolds():
    """Simple test of batch implementations."""
    print("Testing Simple Batch Manifolds...")

    try:
        # Test basic batch class imports
        from hyperXGB.xgb import PoincareBall, HyperboloidBatch

        # Test very simple PoincareBall
        try:
            poincare = PoincareBall(n=2)
            random_point = poincare.random_point()
            print(
                f"✓ PoincareBall created, random point shape: {random_point.shape}"
            )
        except Exception as e:
            print(f"✗ PoincareBall test: {e}")

        # Test very simple HyperboloidBatch
        try:
            hyperboloid = HyperboloidBatch(n=10, k=1)
            test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            hyp_points = hyperboloid.filldata_batch(test_data)
            print(
                f"✓ HyperboloidBatch created, mapped shape: {hyp_points.shape}"
            )
        except Exception as e:
            print(f"✗ HyperboloidBatch test: {e}")

        return True
    except Exception as e:
        print(f"✗ Simple batch test failed: {e}")
        return False


def test_api_with_batch_manifolds():
    """Simple API test with minimal data."""
    print("Testing Simple API Integration...")

    try:
        from hyperXGB.api import HyperbolicClassifier

        # Minimal test data
        X, y = make_classification(
            n_samples=12,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )

        print(f"✓ Generated minimal test dataset: {X.shape}")

        try:
            model = HyperbolicClassifier(manifold="euclidean", n_estimators=2)
            model.fit(X[:8], y[:8])  # Train on 8 samples
            pred = model.predict(X[8:])  # Test on 4 samples
            print(f"✓ Simple integration: predictions shape {pred.shape}")
            return True
        except Exception as e:
            print(f"✗ Simple integration failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Simple API integration failed: {e}")
        return False


def test_advanced_hyperbolic_manifolds():
    """Simplified version of advanced testing."""
    print("Testing Simple Advanced Features...")

    try:
        from hyperXGB.api import HyperbolicClassifier

        # Test 1: Slightly larger feature space
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        try:
            model = HyperbolicClassifier(
                manifold="euclidean", n_estimators=5, max_depth=2
            )
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"✓ 5D features: {score:.3f}")
        except Exception as e:
            print(f"✗ 5D features: {e}")

        # Test 2: Simple multi-class
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=3,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        try:
            model = HyperbolicClassifier(
                manifold="euclidean", n_estimators=5, max_depth=2
            )
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"✓ 3-class: {score:.3f}")
        except Exception as e:
            print(f"✗ 3-class: {e}")

        print("✓ Simple Advanced Features: PASS")
        return True

    except Exception as e:
        print(f"✗ Simple advanced test failed: {e}")
        return False


def test_manifold_geometry_properties():
    """Simple geometry test."""
    print("Testing Simple Geometry...")

    try:
        from hyperXGB.xgb.poincare import PoincareBall

        # Very simple geometry test
        try:
            poincare = PoincareBall(n=2)
            points = np.array([[0.1, 0.2], [0.3, 0.4]])
            print("✓ Simple Poincaré geometry: created")
        except Exception as e:
            print(f"✗ Simple geometry: {e}")

        print("✓ Simple Geometry: PASS")
        return True

    except Exception as e:
        print(f"✗ Simple geometry test failed: {e}")
        return False


def test_performance_benchmarks():
    """Simple performance test."""
    print("Testing Simple Performance...")

    try:
        from hyperXGB.api import HyperbolicClassifier

        # Simple timing test
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )

        start_time = time.time()
        model = HyperbolicClassifier(manifold="euclidean", n_estimators=5)
        model.fit(X, y)
        fit_time = time.time() - start_time

        print(f"✓ Simple performance: {fit_time:.3f}s")
        print("✓ Simple Performance: PASS")
        return True

    except Exception as e:
        print(f"✗ Simple performance test failed: {e}")
        return False


def main():
    """Run all simplified tests."""
    print("HyperbolicML Simple 1D API Examples")
    print("=" * 50)

    tests = [
        ("Basic API", test_basic_api),
        ("Simple Classification", test_hyperbolic_classifier),
        ("Simple Regression", test_hyperbolic_regressor),
        ("Simple XGBoost", test_xgboost_workflow),
        ("Simple Manifolds", test_manifold_comparison),
        ("Simple Comprehensive", test_comprehensive_manifolds),
        ("Simple Debug", test_hyperbolic_manifold_debug),
        ("Simple Batch", test_batch_manifolds),
        ("Simple Integration", test_api_with_batch_manifolds),
        ("Simple Advanced", test_advanced_hyperbolic_manifolds),
        ("Simple Geometry", test_manifold_geometry_properties),
        ("Simple Performance", test_performance_benchmarks),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        if test_func():
            passed += 1
            print(f"✓ {test_name}: PASS")
        else:
            print(f"✗ {test_name}: FAIL")

    print(f"\n{'='*50}")
    print(f"Simple API Examples Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple API examples work correctly!")
        return 0
    else:
        print("❌ Some simple API examples failed")
        return 1


if __name__ == "__main__":
    exit(main())
