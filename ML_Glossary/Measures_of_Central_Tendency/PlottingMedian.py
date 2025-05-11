import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

def plot_median_vs_mean(data, title):
    mean_val = np.mean(data)
    median_val = np.median(data)

    plt.figure(figsize=(8, 4))
    sns.histplot(data, bins=10, kde=True, color="skyblue", edgecolor="black")
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', label=f'Median = {median_val:.2f}')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 1. Outlier Resistance Example
data_outlier = [10, 15, 25, 30, 40, 1000]
plot_median_vs_mean(data_outlier, "Outlier Resistance: Median vs Mean")

# 2. Symmetric Distribution Example
data_symmetric = [70, 75, 80, 85, 90]
plot_median_vs_mean(data_symmetric, "Symmetric Data: Mean = Median")

# 3. Unequally Important Values (Sales)
data_sales = [1000, 1200, 1100, 5000, 1150]
plot_median_vs_mean(data_sales, "All Values Matter: Use Mean")

# 4. Median Stability (Small Changes)
data_before = [100, 101, 102, 103, 104]
data_after = [100, 101, 102, 103, 110]

plot_median_vs_mean(data_before, "Before Change: Stable Median")
plot_median_vs_mean(data_after, "After Change: Median Unchanged, Mean Increased")
