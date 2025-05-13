# 📊 Normal Distribution (Gaussian Distribution)

The **Normal Distribution**, also called the **Gaussian Distribution**, is the most widely used probability distribution for real-valued random variables. It forms the **famous bell-shaped curve** and is foundational in statistics and machine learning.

---

## 🔍 Key Characteristics

- **Bell-shaped Curve**: The distribution is symmetric around its center.
- **Symmetry**: The left and right sides of the curve are mirror images.
- **Central Tendency**: In a normal distribution,
  ```
  Mean = Median = Mode
  ```
- **Empirical Rule (68-95-99.7)**:
  - 68% of data lies within **±1σ** (1 standard deviation)
  - 95% of data lies within **±2σ**
  - 99.7% of data lies within **±3σ**

---

## 📐 Probability Density Function

The probability of a value `x` in a normal distribution is given by:

```math
f(x) = (1 / √(2πσ²)) * e^(-(x - μ)² / (2σ²))
```

Where:

- `μ` = Mean (center of the distribution)
- `σ` = Standard Deviation
- `x` = Data point
- `e` = Euler’s number (~2.718)

---

## ⚙️ Applications of Normal Distribution

### 1. 🧮 Feature Scaling & Standardization

In machine learning, it's common to normalize data to a standard scale:

```bash
# Standardize a dataset to mean=0 and std=1 using Python (sklearn)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

> Many models like **Logistic Regression**, **SVM**, **Neural Networks**, and **PCA** assume or perform better with standardized data.

---

### 2. 🚨 Outlier Detection

Because 99.7% of data in a normal distribution falls within ±3σ, values outside this range are considered **outliers**.

```bash
# Detect outliers in a list using Z-score
import numpy as np
from scipy.stats import zscore

data = np.array([10, 12, 11, 14, 13, 100])  # 100 is an outlier
z_scores = zscore(data)
outliers = data[np.abs(z_scores) > 3]
print(outliers)  # Output: [100]
```

> Useful in **fraud detection**, **quality control**, and **data cleaning**.

---

### 3. 📏 Estimating Errors in Linear Regression

Residuals (differences between actual and predicted values) are assumed to be normally distributed in **Linear Regression**.

```bash
# Visualizing residuals using matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
residuals = actual - predicted
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
```

---

### 4. 🧠 Weight Initialization in Neural Networks

Weights in neural networks are often initialized using a **normal distribution** to help the model converge efficiently.

```bash
# PyTorch: Initialize weights using normal distribution
import torch.nn as nn
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
```

> Proper weight initialization prevents vanishing/exploding gradients and speeds up training.

---
