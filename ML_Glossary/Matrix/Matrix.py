import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define input features (Age, Income) for 3 samples
X = np.array([
    [25, 50000],
    [40, 80000],
    [22, 30000]
])

# Step 2: Normalize features for better scaling
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Step 3: Define weights (e.g., learned in a linear regression model)
W = np.array([
    [0.3],
    [0.00001]
])

# Step 4: Perform matrix multiplication to get predictions
Y = X_norm @ W  # Equivalent to np.dot(X_norm, W)

# Step 5: Plot the input features and predictions
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Input features (scatter)
axs[0].scatter(X_norm[:, 0], X_norm[:, 1], color='blue')
axs[0].set_title("Normalized Input Features")
axs[0].set_xlabel("Age (normalized)")
axs[0].set_ylabel("Income (normalized)")
axs[0].grid(True)

# Plot 2: Predictions (bar plot)
axs[1].bar(range(1, 4), Y.flatten(), color='green')
axs[1].set_title("Predicted Output (Y)")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Prediction")
axs[1].set_xticks([1, 2, 3])
axs[1].grid(True)

# Show the plots
plt.tight_layout()
plt.show()
