#Project 2: Memorization Detector

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Tiny dataset — only 8 points (very small → easy to memorize)
n_tiny = 8
X_tiny = np.linspace(-2.5, 2.5, n_tiny).reshape(-1, 1)
true_fn = lambda x: 1.5 * x**2 - 2 * x + 0.8
y_tiny = true_fn(X_tiny) + np.random.normal(0, 0.4, n_tiny).reshape(-1, 1)

# High-capacity model — high polynomial degree to force memorization
degree = 12
poly = PolynomialFeatures(degree)
model = make_pipeline(poly, LinearRegression())

# Train on tiny data
model.fit(X_tiny, y_tiny.ravel())

# Final train MSE (should be extremely low)
y_pred_train = model.predict(X_tiny)
train_mse = mean_squared_error(y_tiny, y_pred_train)
print(f"Final train MSE on {n_tiny} points (degree {degree}): {train_mse:.6f}")
print("→ Near-zero train error = model memorized the points perfectly")

# Smooth curve for visualization
X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
y_plot_true = true_fn(X_plot)
y_plot_pred = model.predict(X_plot)

# Plot
plt.figure(figsize=(12, 7))
plt.scatter(X_tiny, y_tiny, color='red', s=100, label=f'Tiny training data ({n_tiny} points)', zorder=5)
plt.plot(X_plot, y_plot_true, color='green', linestyle='--', linewidth=2.5, label='True underlying function')
plt.plot(X_plot, y_plot_pred, color='blue', linewidth=2.5, label=f'High-degree poly fit (deg {degree})')

plt.title('Memorization Detector\nNear-perfect train fit on tiny data → severe overfitting')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3.2, 3.2)
plt.ylim(-6, 10)
plt.show()