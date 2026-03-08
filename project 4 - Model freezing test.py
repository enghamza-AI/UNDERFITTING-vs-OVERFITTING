#Project 4: Model Freezing Test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Settings
true_fn = lambda x: 1.5 * x**2 - 2 * x + 0.8
noise_std = 0.8
degree = 1                  # FIXED low complexity (linear model) → high bias

# Training set sizes to test
train_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]

val_mses = []

# Large fixed validation set (same for all — fresh but reproducible)
n_val = 500
X_val = np.random.uniform(-3, 3, n_val).reshape(-1, 1)
y_val = true_fn(X_val) + np.random.normal(0, noise_std, n_val).reshape(-1, 1)

for n_train in train_sizes:
    # Generate training data
    X_train = np.random.uniform(-3, 3, n_train).reshape(-1, 1)
    y_train = true_fn(X_train) + np.random.normal(0, noise_std, n_train).reshape(-1, 1)
    
    # Model — fixed low complexity
    poly = PolynomialFeatures(degree)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X_train, y_train.ravel())
    
    # Validate on large hold-out
    y_pred_val = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_mses.append(val_mse)
    
    print(f"Train size: {n_train:4d} → Val MSE: {val_mse:6.3f}")

# Plot the scaling curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, val_mses, 'o-', color='teal', linewidth=2.5, markersize=8)
plt.xscale('log')
plt.xlabel('Training set size (log scale)')
plt.ylabel('Validation MSE')
plt.title('Model Freezing Test\nFixed low-capacity (linear) model → more data reduces variance but bias floor remains')
plt.grid(True, alpha=0.3)
plt.ylim(0, max(val_mses) * 1.1)
plt.show()