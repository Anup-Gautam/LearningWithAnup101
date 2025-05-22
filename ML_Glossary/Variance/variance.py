import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def population_variance(data):
    mean = np.mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def sample_variance(data):
    mean = np.mean(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)

# Example 1: Intuition Behind Variance
data1 = [2, 4, 6, 8, 10]
data2 = [5, 5, 6, 6, 6]

print("Population Variance of data1:", population_variance(data1))
print("Population Variance of data2:", population_variance(data2))

# Example: Sensitivity to Outliers
data_with_outlier = [5, 6, 100]
print("Variance with outlier:", population_variance(data_with_outlier))

# Example: High Variance (Overfitting) vs Low Variance (Underfitting)
# Generate synthetic nonlinear data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Underfitting Model: Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

# Overfitting Model: Deep Decision Tree
tree = DecisionTreeRegressor(max_depth=None)
tree.fit(X, y)
y_pred_tree = tree.predict(X)

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, s=20, edgecolor="black", label="Data")
plt.plot(X, y_pred_linear, color="blue", label="Linear Regression")
plt.title("Low Variance (Underfitting)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, s=20, edgecolor="black", label="Data")
plt.plot(X, y_pred_tree, color="red", label="Deep Decision Tree")
plt.title("High Variance (Overfitting)")
plt.legend()

plt.tight_layout()
plt.savefig("/mnt/data/variance_examples_plot.png")
plt.show()
