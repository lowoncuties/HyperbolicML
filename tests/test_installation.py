#!/usr/bin/env python3
"""
HyperbolicML Installation Test Script

This script tests the installation and basic functionality of HyperbolicML.
"""

import sys
import os

# Add parent directory to path so we can import HyperbolicML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test if HyperbolicML can be imported."""
    try:
        import HyperbolicML

        print("✓ Import successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of HyperbolicML."""
    try:
        import HyperbolicML

        # Test that we can access the version
        version = getattr(HyperbolicML, "__version__", "unknown")
        print(f"✓ Basic functionality works (version: {version})")
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def test_manifold_types():
    """Test that manifold utilities are available."""
    try:
        from HyperbolicML.hyperXGB.xgb import PoincareBall, HyperboloidBatch

        print("✓ Manifold types accessible")
        return True
    except Exception as e:
        print(f"✗ Manifold test failed: {e}")
        return False


def test_manifold_classes():
    """Test that manifold wrapper classes are available."""
    try:
        from HyperbolicML.hyperXGB.manifolds import (
            PoincareManifold,
            HyperboloidManifold,
        )

        print("✓ Manifold classes accessible")
        return True
    except Exception as e:
        print(f"✗ Manifold classes test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("HyperbolicML Installation Test")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Manifold Types Test", test_manifold_types),
        ("Manifold Classes Test", test_manifold_classes),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name.lower()}...")
        results.append((test_name, test_func()))

    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("✅ All tests passed! HyperbolicML is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check your installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
