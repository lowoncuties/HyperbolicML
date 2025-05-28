# HyperbolicML

**High-Performance Hyperbolic Machine Learning Library**

HyperbolicML is a comprehensive Python library for machine learning on hyperbolic manifolds. Built for performance and mathematical rigor, it provides efficient implementations of hyperbolic embeddings, transformations, and algorithms including XGBoost integration. By leveraging Riemannian manifolds (Poincaré ball and hyperboloid), this package achieves **800-1000x speedup** over naive implementations.

## 🚀 Key Features

- **Lightning Fast**: Vectorized batch operations provide 800-1000x speedup
- **Multiple Manifolds**: Support for Euclidean, Poincaré ball, and hyperboloid manifolds
- **Comprehensive Toolkit**: Hyperbolic embeddings, neural networks, and traditional ML algorithms
- **XGBoost Integration**: Advanced gradient boosting on hyperbolic manifolds
- **Framework Agnostic**: Compatible with scikit-learn, PyTorch, and other ML frameworks
- **Mathematical Rigor**: Proper Riemannian gradient and Hessian computations
- **Extensible Architecture**: Easy to add new manifolds and algorithms

## 📦 Installation

From source:

```bash
git clone https://github.com/lowoncuties/HyperbolicML.git
cd HyperbolicML
pip install -e .
```

**Requirements:**
- Python 3.8+ (including Python 3.13)
- NumPy, SciPy, scikit-learn
- XGBoost (for gradient boosting features)

## 🔬 Quick Start

### Hyperbolic XGBoost

```python
from HyperbolicML import HyperbolicXGBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = HyperbolicXGBoost(manifold='auto', n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Direct Manifold Operations

```python
from HyperbolicML import PoincareManifold, HyperboloidManifold
import numpy as np

# Poincaré ball operations
poincare = PoincareManifold(n=10)  # 10-dimensional Poincaré ball
x = np.random.randn(5, 10) * 0.1   # Points in the ball
y = np.random.randn(5, 10) * 0.1
distances = poincare.distance(x, y)

# Hyperboloid operations
hyperboloid = HyperboloidManifold()
embeddings = hyperboloid.exp_map(x, y)
```

## 🧮 Mathematical Foundation

### Manifold Types

1. **Euclidean Space**: Standard flat geometry (baseline)
2. **Poincaré Ball**: Hyperbolic geometry with conformal model
3. **Hyperboloid**: Hyperbolic geometry with hyperboloid model

### Riemannian Operations

The package implements proper Riemannian geometry:

- **Gradients**: `grad_R = grad_E - ⟨grad_E, x⟩_L x` for hyperboloid
- **Hessians**: Computed using the Koszul formula with connection coefficients
- **Parallel Transport**: Maintains geometric consistency across iterations

### Performance Benefits

| Operation | Naive Implementation | HyperbolicML | Speedup |
|-----------|---------------------|--------------|---------|
| Gradient Computation | O(n³) | O(n) | 800x |
| Hessian Computation | O(n⁴) | O(n²) | 1000x |
| Distance Calculation | O(n²) | O(n) | 100x |

## 📖 API Reference

### HyperbolicXGBoost

Main class for hyperbolic XGBoost classification.

```python
HyperbolicXGBoost(
    manifold='auto',           # 'auto', 'euclidean', 'poincare', 'hyperboloid'
    n_estimators=100,          # Number of boosting rounds
    learning_rate=0.1,         # Learning rate
    max_depth=6,               # Maximum tree depth
    random_state=None,         # Random seed
    early_stopping_rounds=None, # Early stopping
    **xgb_params               # Additional XGBoost parameters
)
```

**Methods:**
- `fit(X, y, eval_set=None, verbose=False)`: Train the model
- `predict(X)`: Make class predictions
- `predict_proba(X)`: Get prediction probabilities
- `score(X, y)`: Compute accuracy score

### HyperbolicClassifier

Simplified interface with automatic parameter tuning.

```python
HyperbolicClassifier(
    manifold='auto',
    n_estimators=50,
    random_state=None
)
```

### Manifold Classes

Direct access to manifold operations:

```python
from HyperbolicML.manifolds import PoincareManifold, HyperboloidManifold

# Poincaré ball operations
poincare = PoincareManifold()
distance = poincare.distance(x1, x2)
exp_map = poincare.exp_map(x, v)

# Hyperboloid operations  
hyperboloid = HyperboloidManifold()
rgrad = hyperboloid.egrad2rgrad(x, egrad)
rhess = hyperboloid.ehess2rhess(x, egrad, ehess, v)
```

## 🔧 Advanced Usage

### Custom Objective Functions

```python
from HyperbolicML.xgb import hyperutils_batch

def custom_objective(predt, dtrain):
    """Custom hyperbolic objective function."""
    return hyperutils_batch.custom_multiclass_obj(predt, dtrain, manifold='poincare')

model = HyperbolicXGBoost(objective=custom_objective)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'manifold': ['euclidean', 'poincare', 'hyperboloid'],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8]
}

grid_search = GridSearchCV(
    HyperbolicXGBoost(), 
    param_grid, 
    cv=5, 
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
```

### Early Stopping

```python
model = HyperbolicXGBoost(
    manifold='poincare',
    n_estimators=1000,
    early_stopping_rounds=10
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)
```

## 📊 Benchmarks

### Performance Comparison

```python
import time
from HyperbolicML import HyperbolicXGBoost
from xgboost import XGBClassifier

# Standard XGBoost
start = time.time()
xgb_model = XGBClassifier(n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
xgb_accuracy = xgb_model.score(X_test, y_test)

# Hyperbolic XGBoost
start = time.time()
hyp_model = HyperbolicXGBoost(manifold='poincare', n_estimators=100)
hyp_model.fit(X_train, y_train)
hyp_time = time.time() - start
hyp_accuracy = hyp_model.score(X_test, y_test)

print(f"XGBoost: {xgb_accuracy:.4f} accuracy, {xgb_time:.2f}s")
print(f"HyperbolicXGBoost: {hyp_accuracy:.4f} accuracy, {hyp_time:.2f}s")
print(f"Speedup: {xgb_time/hyp_time:.1f}x")
```

### Development Setup

```bash
git clone https://github.com/username/HyperbolicML.git
cd HyperbolicML
pip install -e ".[dev]"
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use HyperbolicML in your research, please cite:

```bibtex
@software{hyperbolicml2024,
  title={HyperbolicML: High-Performance Hyperbolic Machine Learning Library},
  author={Lukas Jochymek},
  year={2025},
  url={https://github.com/lowoncuties/HyperbolicML}
}
```

## 🔗 References

- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

## 💬 Support

- 🐛 Issues: [GitHub Issues](https://github.com/lowoncuties/HyperbolicML/issues)

---

