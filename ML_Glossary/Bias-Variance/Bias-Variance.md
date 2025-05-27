# Bias-Variance Tradeoff: A Comprehensive Guide

## Introduction

The **Bias-Variance Tradeoff** provides a theoretical foundation to understand why models fail and how they can be improved. This framework is essential in machine learning, as it helps diagnose model behavior and guides the choice of appropriate solutions.

### Total Prediction Error

The total expected error for a model \( f'(x) \) trying to predict the true function \( f(x) \) at a point \( x \) can be broken down as:

\[ \mathbb{E}[(f'(x) - f(x))^2] = \text{Bias}^2[f'(x)] + \text{Variance}[f'(x)] + \text{Irreducible Error} \]

#### Legend:

- \( f(x) \): True function
- \( f'(x) \): Predicted function by the model
- \( \mathbb{E} \): Expectation (average over multiple datasets or models)
- **Bias**: Systematic deviation from the true function
- **Variance**: Sensitivity to training data variations
- **Irreducible Error**: Noise in the data that cannot be eliminated

...

```python
# Example for High Bias (Underfitting)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate quadratic data
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X**2 + np.random.normal(0, 1, size=X.shape)

# Linear model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Linear Model')
plt.legend()
plt.title('High Bias Example')
plt.show()
```

```python
# Example for High Variance (Overfitting)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Overfitting model (high variance)
high_variance_model = DecisionTreeRegressor(max_depth=None)
high_variance_model.fit(X_train, y_train)

train_mse = mean_squared_error(y_train, high_variance_model.predict(X_train))
test_mse = mean_squared_error(y_test, high_variance_model.predict(X_test))

print(f"Training MSE: {train_mse:.2f}")
print(f"Validation MSE: {test_mse:.2f}")
```

...
