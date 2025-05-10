import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

# Set a consistent style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Create a figure with subplots
fig = plt.figure(figsize=(20, 16))

# ========== 1. Why Mean is important: Shows best value to minimize error ==========
ax1 = fig.add_subplot(3, 2, 1)

# Example data
data = [2, 4, 6, 8]
mean_value = np.mean(data)

# Calculate errors for different guesses
guesses = np.arange(1, 10)
squared_errors = []

for guess in guesses:
    error = sum((np.array(data) - guess)**2)
    squared_errors.append(error)

# Plot the squared error for each guess
ax1.plot(guesses, squared_errors, 'o-', linewidth=2, markersize=10)
ax1.axvline(x=mean_value, color='red', linestyle='--', label=f'Mean = {mean_value}')
ax1.axvline(x=guesses[np.argmin(squared_errors)], color='green', linestyle='--', 
           label=f'Min Error at x = {guesses[np.argmin(squared_errors)]}')

# Highlight the minimum error point
min_error_idx = np.argmin(squared_errors)
ax1.scatter([guesses[min_error_idx]], [squared_errors[min_error_idx]], 
           color='red', s=200, zorder=5, label=f'Minimum Error: {squared_errors[min_error_idx]}')

ax1.set_title('Why Mean is Important: Minimizes Squared Error', fontsize=14)
ax1.set_xlabel('Guess Value', fontsize=12)
ax1.set_ylabel('Sum of Squared Errors', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotations
for i, txt in enumerate(squared_errors):
    ax1.annotate(f"{txt}", (guesses[i], squared_errors[i]), 
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9)

# ========== 2. Problem with Outliers ==========
ax2 = fig.add_subplot(3, 2, 2)

# Example data with outlier
data_with_outlier = [10, 12, 11, 1000]
mean_with_outlier = np.mean(data_with_outlier)
median_with_outlier = np.median(data_with_outlier)

# Create DataFrame for seaborn
df_outlier = pd.DataFrame({
    'Value': data_with_outlier,
    'Type': ['Normal', 'Normal', 'Normal', 'Outlier']
})

# Plot the data
colors = ['#3498db', '#3498db', '#3498db', '#e74c3c']
ax2.bar(range(len(data_with_outlier)), data_with_outlier, color=colors)
ax2.axhline(y=mean_with_outlier, color='red', linestyle='-', 
           linewidth=2, label=f'Mean = {mean_with_outlier:.2f}')
ax2.axhline(y=median_with_outlier, color='green', linestyle='--', 
           linewidth=2, label=f'Median = {median_with_outlier:.2f}')

ax2.set_title('Problem with Outliers', fontsize=14)
ax2.set_xlabel('Data Points', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_xticks(range(len(data_with_outlier)))
ax2.set_xticklabels(['Point 1', 'Point 2', 'Point 3', 'Outlier'])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add annotations for the first 3 points
for i in range(3):
    ax2.annotate(f"{data_with_outlier[i]}", 
                (i, data_with_outlier[i]), 
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9)

# Add annotation for the outlier
ax2.annotate(f"{data_with_outlier[3]}", 
            (3, data_with_outlier[3]), 
            xytext=(0, -20), textcoords='offset points',
            ha='center', fontsize=10, fontweight='bold')

# Add explanatory text
ax2.text(0.5, 0.5, "Mean (258.25) is pulled far from\nthe majority of data points\ndue to the outlier.",
        transform=ax2.transAxes, ha='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# ========== 3. Non-Symmetrical Data ==========
ax3 = fig.add_subplot(3, 2, 3)

# Simulate income data with right skew (lognormal distribution)
np.random.seed(42)
income_data = np.random.lognormal(mean=10, sigma=1, size=1000)

# Calculate statistics
mean_income = np.mean(income_data)
median_income = np.median(income_data)
mode_income = stats.mode(income_data.round(-3)).mode  # Rounded to nearest thousand

# Plot the distribution
sns.histplot(income_data, bins=50, kde=True, ax=ax3)
ax3.axvline(x=mean_income, color='red', linestyle='-', 
           linewidth=2, label=f'Mean = {mean_income:.2f}')
ax3.axvline(x=median_income, color='green', linestyle='--', 
           linewidth=2, label=f'Median = {median_income:.2f}')
ax3.axvline(x=mode_income, color='blue', linestyle=':', 
           linewidth=2, label=f'Approx. Mode = {mode_income:.2f}')

ax3.set_title('Non-Symmetrical Data: Income Distribution', fontsize=14)
ax3.set_xlabel('Income', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Add explanatory text
ax3.text(0.7, 0.8, "Mean is pulled to the right\nby high-income individuals.\nMedian better represents\nthe 'typical' income.",
        transform=ax3.transAxes, ha='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# ========== 4. Categorical Data ==========
ax4 = fig.add_subplot(3, 2, 4)

# Example categorical data
categories = ['Red', 'Blue', 'Green', 'Yellow', 'Purple']
frequencies = [42, 35, 27, 20, 15]

# Create a bar plot for categories
bars = ax4.bar(categories, frequencies, color=['red', 'blue', 'green', 'yellow', 'purple'])

# Calculate the mode (most frequent category)
mode_idx = np.argmax(frequencies)
mode_cat = categories[mode_idx]

# Highlight the mode
bars[mode_idx].set_edgecolor('black')
bars[mode_idx].set_linewidth(3)

ax4.set_title('Categorical Data: Color Preferences', fontsize=14)
ax4.set_xlabel('Color', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Add text for most common category
ax4.text(mode_idx, frequencies[mode_idx] + 5, 
        f"Mode: '{mode_cat}'\n(Most frequent)",
        ha='center', fontsize=12, fontweight='bold')

# Add explanatory text
ax4.text(0.5, 0.8, "With categorical data, we can't calculate a mean.\nWe use mode (most frequent) instead.",
        transform=ax4.transAxes, ha='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# ========== 5. Imbalanced Dataset ==========
ax5 = fig.add_subplot(3, 2, 5)

# Create imbalanced fraud detection data
normal = np.ones(99)
fraud = np.zeros(1)
data_imbalanced = np.concatenate([normal, fraud])
labels = ['Normal' if x == 1 else 'Fraud' for x in data_imbalanced]

# Create a DataFrame
df_imbalanced = pd.DataFrame({
    'Transaction': range(1, len(data_imbalanced) + 1),
    'Class': labels
})

# Plot
colors = ['#3498db' if label == 'Normal' else '#e74c3c' for label in labels]
ax5.bar(df_imbalanced['Transaction'], np.ones(len(df_imbalanced)), color=colors)

# Calculate mean and add a line for it
mean_imbalanced = np.mean(data_imbalanced)
ax5.axhline(y=0.5, color='red', linestyle='-', 
           linewidth=2, label=f'Decision Boundary = 0.5')

ax5.set_title('Imbalanced Dataset: Fraud Detection (99% Normal, 1% Fraud)', fontsize=14)
ax5.set_xlabel('Transaction ID', fontsize=12)
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['Fraud', 'Normal'])
ax5.legend(fontsize=10)

# Create custom legend for classes
legend_elements = [
    Patch(facecolor='#3498db', label='Normal (99%)'),
    Patch(facecolor='#e74c3c', label='Fraud (1%)')
]
ax5.legend(handles=legend_elements, fontsize=10, loc='upper right')

# Add fraud annotation
fraud_idx = np.where(data_imbalanced == 0)[0][0]
ax5.annotate('Fraud transaction', 
            (fraud_idx + 1, 0), 
            xytext=(0, -30), textcoords='offset points',
            ha='center', va='top', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Add explanatory text
ax5.text(0.5, 0.3, "Mean-based prediction would always predict 'Normal'\ndue to class imbalance (mean=0.99).\nNeed other metrics like precision/recall.",
        transform=ax5.transAxes, ha='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# ========== 6. Summary of when to use/not use Mean ==========
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')


# Adjust layout and show
plt.tight_layout()
plt.savefig('mean_advantages_disadvantages.png', dpi=300, bbox_inches='tight')
plt.show()