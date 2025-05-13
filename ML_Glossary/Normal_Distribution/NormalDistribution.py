import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, zscore

# Set style
sns.set(style="whitegrid")

# 1. Normal Distribution Curve with 68-95-99.7 Rule
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Normal Distribution', color='blue')
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.3, label='68%')
plt.fill_between(x, y, where=(x >= -2) & (x <= 2), alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= -3) & (x <= 3), alpha=0.1, label='99.7%')
plt.title('Normal Distribution Curve with Standard Deviation Ranges')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# 2. Standardization Example
data = np.random.normal(loc=50, scale=10, size=1000)
standardized = (data - np.mean(data)) / np.std(data)

plt.figure(figsize=(10, 6))
sns.histplot(standardized, kde=True, bins=30)
plt.title("Standardized Data (Mean = 0, Std = 1)")
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# 3. Outlier Detection using Z-score
outlier_data = np.append(np.random.normal(50, 5, 1000), [100, 105])
z_scores = zscore(outlier_data)
outliers = outlier_data[np.abs(z_scores) > 3]

plt.figure(figsize=(10, 6))
sns.histplot(outlier_data, bins=30, kde=True)
for o in outliers:
    plt.axvline(o, color='red', linestyle='--', label='Outlier' if 'Outlier' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title("Outlier Detection using Z-score")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 4. Residuals in Linear Regression
actual = np.random.normal(50, 5, 100)
predicted = actual + np.random.normal(0, 2, 100)
residuals = actual - predicted

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution (Assumed Normal in Linear Regression)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.axvline(np.mean(residuals), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# 5. Neural Network Weight Initialization
weights = np.random.normal(loc=0, scale=0.01, size=1000)

plt.figure(figsize=(10, 6))
sns.histplot(weights, bins=30, kde=True)
plt.title("Weight Initialization using Normal Distribution (mean=0, std=0.01)")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.show()
