#project 1 - train vs validation tracker
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Data: quadratic + moderate noise
n_samples = 200
X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
true_fn = lambda x: 1.5 * x**2 - 2 * x + 0.8
y = true_fn(X) + np.random.normal(0, 0.8, n_samples).reshape(-1, 1)

#Fixed train / validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

degrees = list(range(1, 16))
train_mses = []
val_mses = []

for deg in degrees:
    #creating model for this complexity
    poly = PolynomialFeatures(deg)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X_train, y_train.ravel())

    #Error
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)


    train_mses.append(train_mse)
    val_mses.append(val_mse)

    print(f"Degree {deg:2d} | Train MSE: {train_mse:6.3f} | Val MSE: {val_mse:6.3f}")


plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mses, 'o-', color='blue', label='Train MSE')
plt.plot(degrees, val_mses,   'o-', color='orange', label='Validation MSE')
plt.xlabel('Model complexity (polynomial degree)')
plt.ylabel('Mean Squared Error')
plt.title('Train vs Validation Tracker\nDivergence shows overfitting onset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  
plt.show()
