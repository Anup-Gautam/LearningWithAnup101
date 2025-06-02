# Complete Guide to Outlier Detection Methods

## Table of Contents

1. [Introduction to Outliers](#introduction-to-outliers)
2. [Types of Outliers](#types-of-outliers)
3. [Statistical Methods](#statistical-methods)
4. [Distance-Based Methods](#distance-based-methods)
5. [Model-Based Approaches](#model-based-approaches)
6. [Robust Statistical Methods](#robust-statistical-methods)
7. [Multivariate Outlier Detection](#multivariate-outlier-detection)
8. [Time Series Outlier Detection](#time-series-outlier-detection)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Practical Considerations](#practical-considerations)

---

## Introduction to Outliers

Outliers are data points that significantly deviate from the expected pattern in a dataset. They can indicate errors, fraud, or genuinely interesting phenomena that warrant investigation.

**Real-world Example**: In a dataset of employee salaries at a company, most employees earn between $40,000-$120,000 annually, but the CEO earns $2,000,000. The CEO's salary is an outlier that reflects a genuine business reality rather than an error.

---

## Types of Outliers

### Point Outliers

Individual data points that are anomalous with respect to the entire dataset.

**Example**: In a dataset of human heights, finding a person who is 3 feet tall (dwarfism) or 8 feet tall (gigantism).

**Code Example**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate normal height data (in inches)
np.random.seed(42)
heights = np.random.normal(68, 4, 1000)  # Mean: 68 inches, Std: 4 inches

# Add point outliers
outliers = [36, 96]  # 3 feet and 8 feet
all_heights = np.concatenate([heights, outliers])

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=30, alpha=0.7, label='Normal Heights')
plt.scatter(outliers, [5, 5], color='red', s=100, label='Point Outliers')
plt.xlabel('Height (inches)')
plt.ylabel('Frequency')
plt.title('Point Outliers in Height Data')
plt.legend()
plt.show()
```

### Contextual Outliers

Data points that are anomalous in a specific context but normal otherwise.

**Example**: A temperature of 80°F (27°C) is normal in July in New York but would be highly unusual in January.

**Code Example**:

```python
import pandas as pd
import numpy as np

# Create temperature data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
np.random.seed(42)

# Seasonal temperature pattern
day_of_year = dates.dayofyear
seasonal_temp = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
noise = np.random.normal(0, 5, len(dates))
temperatures = seasonal_temp + noise

# Add contextual outlier: 80°F in January
temperatures[10] = 80  # January 11th

# Create DataFrame
weather_data = pd.DataFrame({
    'date': dates,
    'temperature': temperatures,
    'month': dates.month
})

# Identify contextual outlier
january_temps = weather_data[weather_data['month'] == 1]['temperature']
january_mean = january_temps.mean()
january_std = january_temps.std()

print(f"January average: {january_mean:.1f}°F")
print(f"January std: {january_std:.1f}°F")
print(f"Outlier temperature: {temperatures[10]:.1f}°F")
print(f"Z-score for outlier: {(temperatures[10] - january_mean) / january_std:.2f}")
```

### Collective Outliers

A collection of data points that together form an anomalous pattern.

**Example**: Network traffic that suddenly spikes for several hours, indicating a potential DDoS attack.

**Code Example**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate normal network traffic (requests per minute)
np.random.seed(42)
time_points = np.arange(0, 1440)  # 24 hours in minutes
normal_traffic = np.random.poisson(50, 1440)  # Average 50 requests/minute

# Add collective outlier: DDoS attack from 2 PM to 4 PM (840-960 minutes)
attack_period = slice(840, 960)
normal_traffic[attack_period] += np.random.poisson(500, 120)  # +500 requests/minute

plt.figure(figsize=(12, 6))
plt.plot(time_points, normal_traffic)
plt.axvspan(840, 960, alpha=0.3, color='red', label='Collective Outlier Period')
plt.xlabel('Time (minutes from midnight)')
plt.ylabel('Requests per minute')
plt.title('Network Traffic with Collective Outlier (DDoS Attack)')
plt.legend()
plt.show()
```

---

## Statistical Methods

### Z-Score Method

The Z-score measures how many standard deviations a data point is from the mean.

**Formula**:

```
z = (x - μ) / σ
```

Where:

- x = data point
- μ = mean of the dataset
- σ = standard deviation of the dataset

**Interpretation**: Points with |z| > 2 or 3 are typically considered outliers.

**Real-world Example**: Detecting unusually high blood pressure readings in medical data.

**Code Implementation**:

```python
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using Z-score method

    Parameters:
    data: array-like, input data
    threshold: float, Z-score threshold (default: 3)

    Returns:
    outliers: boolean array indicating outlier positions
    z_scores: array of Z-scores
    """
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    return outliers, z_scores

# Example: Blood pressure data
np.random.seed(42)
systolic_bp = np.random.normal(120, 15, 1000)  # Normal: 120 ± 15 mmHg

# Add some outliers
systolic_bp = np.concatenate([systolic_bp, [200, 210, 80, 70]])

# Detect outliers
outliers, z_scores = detect_outliers_zscore(systolic_bp, threshold=2.5)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Outlier values: {systolic_bp[outliers]}")
print(f"Outlier Z-scores: {z_scores[outliers]}")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(systolic_bp, bins=30, alpha=0.7, color='skyblue')
plt.scatter(systolic_bp[outliers], np.zeros_like(systolic_bp[outliers]),
           color='red', s=50, label='Outliers')
plt.xlabel('Systolic Blood Pressure (mmHg)')
plt.ylabel('Frequency')
plt.title('Blood Pressure Distribution with Outliers')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(systolic_bp)), z_scores, alpha=0.6)
plt.axhline(y=2.5, color='red', linestyle='--', label='Threshold')
plt.scatter(np.where(outliers)[0], z_scores[outliers], color='red', s=50)
plt.xlabel('Data Point Index')
plt.ylabel('|Z-Score|')
plt.title('Z-Scores with Outlier Threshold')
plt.legend()

plt.tight_layout()
plt.show()
```

**Mathematical Example**:
Given blood pressure readings: [115, 118, 122, 125, 130, 200]

- Mean (μ) = 135
- Standard deviation (σ) = 32.4
- For the value 200: z = (200 - 135) / 32.4 = 2.01

Since |z| > 2, this could be considered an outlier.

### Interquartile Range (IQR) Method

IQR method is robust to non-normal distributions and doesn't assume normality.

**Formula**:

- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

**Real-world Example**: Detecting fraudulent transaction amounts in credit card data.

**Code Implementation**:

```python
def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using IQR method

    Parameters:
    data: array-like, input data
    multiplier: float, IQR multiplier (default: 1.5)

    Returns:
    outliers: boolean array indicating outlier positions
    bounds: tuple of (lower_bound, upper_bound)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers, (lower_bound, upper_bound)

# Example: Credit card transaction amounts
np.random.seed(42)
# Most transactions are small amounts
normal_transactions = np.random.exponential(50, 10000)  # Mean: $50

# Add some fraudulent high-value transactions
fraud_transactions = [5000, 7500, 10000, 12000]
all_transactions = np.concatenate([normal_transactions, fraud_transactions])

# Detect outliers
outliers, bounds = detect_outliers_iqr(all_transactions)

print(f"Q1: ${np.percentile(all_transactions, 25):.2f}")
print(f"Q3: ${np.percentile(all_transactions, 75):.2f}")
print(f"IQR: ${np.percentile(all_transactions, 75) - np.percentile(all_transactions, 25):.2f}")
print(f"Lower bound: ${bounds[0]:.2f}")
print(f"Upper bound: ${bounds[1]:.2f}")
print(f"Number of outliers: {np.sum(outliers)}")
print(f"Outlier amounts: ${all_transactions[outliers]}")

# Box plot visualization
plt.figure(figsize=(10, 6))
plt.boxplot(all_transactions, vert=False)
plt.scatter(all_transactions[outliers], np.ones_like(all_transactions[outliers]),
           color='red', s=50, alpha=0.7, label='Outliers')
plt.xlabel('Transaction Amount ($)')
plt.title('Credit Card Transactions - IQR Outlier Detection')
plt.legend()
plt.show()
```

### Modified Z-Score

Uses median absolute deviation (MAD) instead of standard deviation, making it more robust.

**Formula**:

```
Modified Z-score = 0.6745 × (x - median) / MAD
```

Where MAD = median(|x - median(x)|)

**Code Implementation**:

```python
def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using Modified Z-score method

    Parameters:
    data: array-like, input data
    threshold: float, Modified Z-score threshold (default: 3.5)

    Returns:
    outliers: boolean array indicating outlier positions
    modified_z_scores: array of Modified Z-scores
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    outliers = np.abs(modified_z_scores) > threshold

    return outliers, modified_z_scores

# Example with same blood pressure data
outliers_mod, mod_z_scores = detect_outliers_modified_zscore(systolic_bp)

print(f"Modified Z-Score Method:")
print(f"Number of outliers: {np.sum(outliers_mod)}")
print(f"Outlier values: {systolic_bp[outliers_mod]}")

# Compare with regular Z-score
print(f"\nComparison:")
print(f"Regular Z-score outliers: {np.sum(outliers)}")
print(f"Modified Z-score outliers: {np.sum(outliers_mod)}")
```

### Grubbs' Test

Statistical test for detecting outliers in univariate data assuming normal distribution.

**Formula**:

```
G = max|x_i - x̄| / s
```

Where s is the sample standard deviation.

**Code Implementation**:

```python
from scipy.stats import t

def grubbs_test(data, alpha=0.05):
    """
    Perform Grubbs' test for outliers

    Parameters:
    data: array-like, input data
    alpha: float, significance level

    Returns:
    is_outlier: bool, whether the most extreme point is an outlier
    test_statistic: float, Grubbs test statistic
    critical_value: float, critical value for the test
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Find the most extreme point
    deviations = np.abs(data - mean)
    max_deviation_idx = np.argmax(deviations)

    # Calculate test statistic
    G = deviations[max_deviation_idx] / std

    # Calculate critical value
    t_critical = t.ppf(1 - alpha/(2*n), n-2)
    critical_value = ((n-1) * np.sqrt(t_critical**2)) / (np.sqrt(n) * np.sqrt(n-2 + t_critical**2))

    is_outlier = G > critical_value

    return is_outlier, G, critical_value, max_deviation_idx

# Example
test_data = np.array([2.1, 2.3, 2.4, 2.2, 2.5, 2.4, 2.3, 2.6, 5.1])  # 5.1 is potential outlier

is_outlier, G, critical_value, outlier_idx = grubbs_test(test_data)

print(f"Grubbs' Test Results:")
print(f"Test statistic G: {G:.3f}")
print(f"Critical value: {critical_value:.3f}")
print(f"Is outlier: {is_outlier}")
print(f"Suspected outlier value: {test_data[outlier_idx]}")
```

---

## Distance-Based Methods

### k-Nearest Neighbors (k-NN) Distance

Identifies outliers based on their distance to k nearest neighbors.

**Real-world Example**: Detecting unusual customer purchasing patterns in e-commerce.

**Code Implementation**:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def detect_outliers_knn(data, k=5, threshold_percentile=95):
    """
    Detect outliers using k-NN distance method

    Parameters:
    data: array-like, shape (n_samples, n_features)
    k: int, number of nearest neighbors
    threshold_percentile: float, percentile threshold for outlier detection

    Returns:
    outliers: boolean array indicating outlier positions
    distances: array of distances to k-th nearest neighbor
    """
    # Fit k-NN model
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because first neighbor is the point itself
    nbrs.fit(data)

    # Find distances to k-th nearest neighbor
    distances, indices = nbrs.kneighbors(data)
    knn_distances = distances[:, k]  # Distance to k-th neighbor (excluding self)

    # Define threshold based on percentile
    threshold = np.percentile(knn_distances, threshold_percentile)
    outliers = knn_distances > threshold

    return outliers, knn_distances

# Example: Customer purchase behavior
# Features: [purchase_frequency, average_amount, days_since_last_purchase]
np.random.seed(42)

# Normal customers
normal_customers = np.random.multivariate_normal(
    mean=[10, 100, 15],  # 10 purchases/month, $100 avg, 15 days since last
    cov=[[4, 10, 2], [10, 400, 5], [2, 5, 25]],
    size=1000
)

# Outlier customers (unusual behavior)
outlier_customers = np.array([
    [50, 500, 1],   # Very frequent, high-value purchases
    [1, 50, 90],    # Very infrequent purchases
    [30, 1000, 2],  # Extremely high spending
])

all_customers = np.vstack([normal_customers, outlier_customers])

# Detect outliers
outliers, distances = detect_outliers_knn(all_customers, k=5, threshold_percentile=97)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Outlier customer indices: {np.where(outliers)[0]}")

# Visualize (2D projection)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(all_customers[~outliers, 0], all_customers[~outliers, 1],
           alpha=0.6, label='Normal Customers')
plt.scatter(all_customers[outliers, 0], all_customers[outliers, 1],
           color='red', s=100, label='Outlier Customers')
plt.xlabel('Purchase Frequency (per month)')
plt.ylabel('Average Purchase Amount ($)')
plt.title('Customer Behavior - k-NN Outlier Detection')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(distances, bins=30, alpha=0.7, color='skyblue')
plt.axvline(x=np.percentile(distances, 97), color='red', linestyle='--',
           label='97th Percentile Threshold')
plt.xlabel('Distance to 5th Nearest Neighbor')
plt.ylabel('Frequency')
plt.title('Distribution of k-NN Distances')
plt.legend()

plt.tight_layout()
plt.show()
```

### Local Outlier Factor (LOF)

Measures local density deviation compared to neighbors. Effective for datasets with varying densities.

**Mathematical Concept**:
LOF compares the local density of a point with the local densities of its neighbors. A point with a substantially lower density than its neighbors is considered an outlier.

**Code Implementation**:

```python
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(data, n_neighbors=20, contamination=0.1):
    """
    Detect outliers using Local Outlier Factor

    Parameters:
    data: array-like, shape (n_samples, n_features)
    n_neighbors: int, number of neighbors to consider
    contamination: float, proportion of outliers in the dataset

    Returns:
    outliers: boolean array indicating outlier positions
    lof_scores: array of LOF scores
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(data)
    lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores

    outliers = outlier_labels == -1

    return outliers, lof_scores

# Example: Sensor readings with varying density regions
np.random.seed(42)

# Create data with different density regions
cluster1 = np.random.normal([2, 2], 0.5, (100, 2))  # Dense cluster
cluster2 = np.random.normal([8, 8], 1.0, (50, 2))   # Less dense cluster
scattered = np.random.uniform([0, 0], [10, 10], (20, 2))  # Scattered points

# Add clear outliers
clear_outliers = np.array([[0, 10], [10, 0], [-1, -1]])

sensor_data = np.vstack([cluster1, cluster2, scattered, clear_outliers])

# Detect outliers using LOF
outliers, lof_scores = detect_outliers_lof(sensor_data, n_neighbors=15, contamination=0.1)

print(f"Number of outliers detected: {np.sum(outliers)}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(sensor_data[~outliers, 0], sensor_data[~outliers, 1],
           alpha=0.6, label='Normal Points')
plt.scatter(sensor_data[outliers, 0], sensor_data[outliers, 1],
           color='red', s=100, label='Outliers (LOF)')
plt.xlabel('Sensor Reading 1')
plt.ylabel('Sensor Reading 2')
plt.title('Sensor Data - LOF Outlier Detection')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(lof_scores, bins=30, alpha=0.7, color='skyblue')
plt.xlabel('LOF Score')
plt.ylabel('Frequency')
plt.title('Distribution of LOF Scores')
plt.axvline(x=1, color='red', linestyle='--', label='LOF = 1 (threshold)')
plt.legend()

plt.tight_layout()
plt.show()
```

### DBSCAN Clustering

Identifies outliers as noise points that don't belong to any cluster.

**Parameters**:

- **eps**: Maximum distance between two points to be neighbors
- **min_samples**: Minimum number of points required to form a cluster

**Code Implementation**:

```python
from sklearn.cluster import DBSCAN

def detect_outliers_dbscan(data, eps=0.5, min_samples=5):
    """
    Detect outliers using DBSCAN clustering

    Parameters:
    data: array-like, shape (n_samples, n_features)
    eps: float, maximum distance between two samples
    min_samples: int, minimum samples in a neighborhood

    Returns:
    outliers: boolean array indicating outlier positions
    cluster_labels: array of cluster labels (-1 for outliers)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)

    outliers = cluster_labels == -1  # -1 indicates noise/outliers

    return outliers, cluster_labels

# Example: Network intrusion detection
# Features: [packet_size, connection_duration, data_transfer_rate]
np.random.seed(42)

# Normal network traffic (3 different types)
web_traffic = np.random.multivariate_normal([1000, 30, 50], [[100, 0, 0], [0, 25, 0], [0, 0, 100]], 300)
email_traffic = np.random.multivariate_normal([500, 10, 20], [[50, 0, 0], [0, 4, 0], [0, 0, 25]], 200)
file_transfer = np.random.multivariate_normal([5000, 120, 200], [[1000, 0, 0], [0, 400, 0], [0, 0, 1000]], 100)

# Suspicious traffic (potential intrusions)
suspicious_traffic = np.array([
    [50000, 1, 10000],  # Large packet, short duration, high rate - DDoS
    [100, 300, 1],      # Small packet, long duration, low rate - slow scan
    [10000, 5, 5000],   # Large packet, short duration, very high rate
])

network_data = np.vstack([web_traffic, email_traffic, file_transfer, suspicious_traffic])

# Detect outliers using DBSCAN
outliers, labels = detect_outliers_dbscan(network_data, eps=2000, min_samples=10)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

# Visualize (2D projection using first two features)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Plot clusters
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Outliers in red
        class_member_mask = (labels == k)
        xy = network_data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=100, c='red', marker='x', label='Outliers')
    else:
        # Clusters
        class_member_mask = (labels == k)
        xy = network_data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], alpha=0.7, label=f'Cluster {k}')

plt.xlabel('Packet Size (bytes)')
plt.ylabel('Connection Duration (seconds)')
plt.title('Network Traffic - DBSCAN Outlier Detection')
plt.legend()

plt.subplot(1, 2, 2)
# Distribution of cluster assignments
unique, counts = np.unique(labels, return_counts=True)
plt.bar([str(l) if l != -1 else 'Outliers' for l in unique], counts,
        color=['red' if l == -1 else 'skyblue' for l in unique])
plt.xlabel('Cluster Label')
plt.ylabel('Number of Points')
plt.title('Cluster Size Distribution')

plt.tight_layout()
plt.show()
```

---

## Model-Based Approaches

### Isolation Forest

Isolates outliers by randomly selecting features and split values. Outliers require fewer splits to isolate.

**Principle**: Anomalies are few and different, so they can be isolated with fewer random partitions.

**Real-world Example**: Credit card fraud detection based on transaction patterns.

**Code Implementation**:

```python
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(data, contamination=0.1, random_state=42):
    """
    Detect outliers using Isolation Forest

    Parameters:
    data: array-like, shape (n_samples, n_features)
    contamination: float, proportion of outliers in the dataset
    random_state: int, random state for reproducibility

    Returns:
    outliers: boolean array indicating outlier positions
    anomaly_scores: array of anomaly scores
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(data)
    anomaly_scores = iso_forest.decision_function(data)

    outliers = outlier_labels == -1

    return outliers, anomaly_scores

# Example: Credit card fraud detection
np.random.seed(42)

# Normal transactions: [amount, time_of_day, merchant_category, location_risk]
normal_transactions = np.column_stack([
    np.random.exponential(50, 5000),  # Amount: mostly small
    np.random.uniform(0, 24, 5000),   # Time: throughout day
    np.random.randint(1, 10, 5000),   # Merchant category
    np.random.beta(2, 5, 5000)        # Location risk: mostly low
])

# Fraudulent transactions (different patterns)
fraud_transactions = np.array([
    [2000, 3, 8, 0.9],    # High amount, odd hour, high-risk location
    [5000, 2, 5, 0.8],    # Very high amount, very late
    [1500, 4, 9, 0.95],   # High amount, unusual merchant, high risk
    [3000, 1, 7, 0.85],   # High amount, very early
    [500, 23, 3, 0.9],    # Normal amount but high-risk location and late hour
])

all_transactions = np.vstack([normal_transactions, fraud_transactions])

# Create labels for evaluation (last 5 are fraud)
true_labels = np.concatenate([np.zeros(5000), np.ones(5)])

# Detect outliers using Isolation Forest
outliers, anomaly_scores = detect_outliers_isolation_forest(
    all_transactions, contamination=0.001  # Expect 0.1% fraud
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Actual number of fraud cases: {np.sum(true_labels)}")

# Calculate detection accuracy
detected_fraud = outliers[-5:]  # Check last 5 transactions
print(f"Fraud cases detected: {np.sum(detected_fraud)} out of 5")

# Visualize anomaly scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(all_transactions[~outliers, 0], all_transactions[~outliers, 1],
           alpha=0.6, label='Normal Transactions', s=20)
plt.scatter(all_transactions[outliers, 0], all_transactions[outliers, 1],
           color='red', s=100, label='Detected Outliers')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Time of Day (hour)')
plt.title('Credit Card Transactions - Isolation Forest')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(anomaly_scores[:-5], bins=50, alpha=0.7, label='Normal Transactions', density=True)
plt.hist(anomaly_scores[-5:], bins=10, alpha=0.7, color='red', label='Actual Fraud', density=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Distribution of Anomaly Scores')
plt.legend()

plt.tight_layout()
plt.show()
```

### One-Class SVM

Learns a boundary around normal data points and identifies outliers as points outside this boundary.

**Mathematical Concept**: Finds a hyperplane that separates the normal data from the origin with maximum margin.

**Code Implementation**:

````python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def detect_outliers_one_class_svm(data, nu=0.05, kernel='rbf', gamma='scale'):
    """
    Detect outliers using One-Class SVM

    Parameters:
    data: array-like, shape (n_samples, n_features)
    nu: float, upper bound on fraction of training errors and lower bound on support vectors
    kernel: str, kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma: str or float, kernel coefficient

    Returns:
    outliers: boolean array indicating outlier positions
    decision_scores: array of decision function values
    """
    # Standardize the data for better SVM performance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit One-Class SVM
    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    outlier_labels = oc_svm.fit_predict(data_scaled)
    decision_scores = oc_svm.decision_function(data_scaled)

    outliers = outlier_labels == -1

    return outliers, decision_scores

# Example: Manufacturing quality control
# Features: [temperature, pressure, vibration, power_consumption]
np.random.seed(42)

# Normal operating conditions
normal_operations = np.random.multivariate_normal(
    mean=[150, 2.5, 0.1, 100],  # Temp: 150°C, Pressure: 2.5 bar, etc.
    cov=[[25, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.001, 0], [0, 0, 0, 100]],
    size=1000
)

# Abnormal conditions (equipment malfunction)
abnormal_operations = np.array([
    [200, 3.5, 0.5, 150],  # Overheating, high pressure, high vibration
    [100, 1.5, 0.05, 50],  # Under-temperature, low pressure
    [160, 2.4, 0.8, 120],  # Normal temp/pressure but excessive vibration
    [180, 4.0, 0.2, 200],  # High temp, excessive pressure
])

manufacturing_data = np.vstack([normal_operations, abnormal_operations])

# Detect outliers using One-Class SVM
outliers, decision_scores = detect_outliers_one_class_svm(
    manufacturing_data, nu=0.01, kernel='rbf'
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Outlier detection on last 4 samples (known abnormal): {outliers[-4:]}")

# Visualize (2D projection)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(manufacturing_data[~outliers, 0], manufacturing_data[~outliers, 1],
           alpha=0.6, label='Normal Operations', s=20)
plt.scatter(manufacturing_data[outliers, 0], manufacturing_data[outliers, 1],
           color='red', s=100, label='Abnormal Operations')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (bar)')
plt.title('Manufacturing Process - One-Class SVM')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(decision_scores[:-4], bins=30, alpha=0.7, label='Normal Operations', density=True)
plt.hist(decision_scores[-4:], bins=5, alpha=0.7, color='red', label='Known Abnormal', density=True)
plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Decision Score')
plt.ylabel('Density')
plt.title('Distribution of Decision Scores')
plt.legend()

plt.tight_layout()
plt.show()

### Autoencoders

Neural networks trained to reconstruct input data. Points with high reconstruction error are considered outliers.

**Architecture**: Input → Encoder → Latent Space → Decoder → Output

**Code Implementation**:
```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler

def create_autoencoder(input_dim, encoding_dim=32):
    """
    Create an autoencoder model

    Parameters:
    input_dim: int, dimension of input data
    encoding_dim: int, dimension of encoding layer

    Returns:
    autoencoder: keras model
    encoder: keras model for encoding
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

def detect_outliers_autoencoder(data, contamination=0.05, epochs=100, validation_split=0.2):
    """
    Detect outliers using Autoencoder reconstruction error

    Parameters:
    data: array-like, shape (n_samples, n_features)
    contamination: float, expected proportion of outliers
    epochs: int, number of training epochs
    validation_split: float, fraction of data for validation

    Returns:
    outliers: boolean array indicating outlier positions
    reconstruction_errors: array of reconstruction errors
    """
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create and train autoencoder
    autoencoder, encoder = create_autoencoder(data.shape[1])

    # Train on normal data only (assuming most data is normal)
    history = autoencoder.fit(
        data_scaled, data_scaled,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        verbose=0
    )

    # Calculate reconstruction errors
    reconstructed = autoencoder.predict(data_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(data_scaled - reconstructed), axis=1)

    # Define threshold based on contamination rate
    threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
    outliers = reconstruction_errors > threshold

    return outliers, reconstruction_errors, history

# Example: Network intrusion detection
np.random.seed(42)
tf.random.set_seed(42)

# Normal network behavior: [bytes_sent, bytes_received, duration, num_connections]
normal_network = np.random.multivariate_normal(
    mean=[1000, 1500, 60, 5],
    cov=[[10000, 5000, 100, 10], [5000, 15000, 150, 15],
         [100, 150, 400, 5], [10, 15, 5, 4]],
    size=2000
)

# Anomalous network behavior (intrusions)
intrusion_patterns = np.array([
    [50000, 100, 1, 100],    # Data exfiltration: high outbound, low inbound
    [100, 50000, 1, 1],      # Data infiltration: low outbound, high inbound
    [5000, 5000, 300, 50],   # Prolonged suspicious activity
    [100, 100, 1, 200],      # Port scanning: many connections, little data
])

network_data = np.vstack([normal_network, intrusion_patterns])

# Detect outliers using Autoencoder
outliers, errors, history = detect_outliers_autoencoder(
    network_data, contamination=0.002, epochs=50
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Intrusions detected: {outliers[-4:]}")

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training History')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(network_data[~outliers, 0], network_data[~outliers, 1],
           alpha=0.6, label='Normal Traffic', s=20)
plt.scatter(network_data[outliers, 0], network_data[outliers, 1],
           color='red', s=100, label='Detected Intrusions')
plt.xlabel('Bytes Sent')
plt.ylabel('Bytes Received')
plt.title('Network Traffic Analysis')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(errors[:-4], bins=50, alpha=0.7, label='Normal Traffic', density=True)
plt.hist(errors[-4:], bins=5, alpha=0.7, color='red', label='Known Intrusions', density=True)
plt.axvline(x=np.percentile(errors, 99.8), color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.title('Distribution of Reconstruction Errors')
plt.legend()

plt.tight_layout()
plt.show()

### Gaussian Mixture Models (GMM)

Identifies outliers as points with low probability under the learned distribution.

**Mathematical Concept**: Models data as a mixture of Gaussian distributions and computes likelihood of each point.

**Code Implementation**:
```python
from sklearn.mixture import GaussianMixture

def detect_outliers_gmm(data, n_components=3, contamination=0.05):
    """
    Detect outliers using Gaussian Mixture Model

    Parameters:
    data: array-like, shape (n_samples, n_features)
    n_components: int, number of Gaussian components
    contamination: float, expected proportion of outliers

    Returns:
    outliers: boolean array indicating outlier positions
    log_likelihood: array of log-likelihood scores
    """
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    # Calculate log-likelihood scores
    log_likelihood = gmm.score_samples(data)

    # Define threshold based on contamination rate
    threshold = np.percentile(log_likelihood, contamination * 100)
    outliers = log_likelihood < threshold

    return outliers, log_likelihood

# Example: Customer segmentation and anomaly detection
np.random.seed(42)

# Customer segments: [age, annual_income, spending_score]
# Segment 1: Young, low income, moderate spending
segment1 = np.random.multivariate_normal([25, 30000, 40],
                                       [[16, 0, 0], [0, 25000000, 0], [0, 0, 100]], 300)

# Segment 2: Middle-aged, high income, high spending
segment2 = np.random.multivariate_normal([45, 80000, 80],
                                       [[36, 0, 0], [0, 100000000, 0], [0, 0, 100]], 200)

# Segment 3: Elderly, moderate income, low spending
segment3 = np.random.multivariate_normal([65, 50000, 20],
                                       [[64, 0, 0], [0, 64000000, 0], [0, 0, 64]], 150)

# Anomalous customers
anomalous_customers = np.array([
    [20, 150000, 95],  # Very young with very high income and spending
    [70, 200000, 5],   # Elderly with very high income but very low spending
    [35, 25000, 90],   # Middle-aged with low income but very high spending
])

customer_data = np.vstack([segment1, segment2, segment3, anomalous_customers])

# Detect outliers using GMM
outliers, log_likelihood = detect_outliers_gmm(customer_data, n_components=3, contamination=0.01)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Anomalous customers detected: {outliers[-3:]}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(customer_data[~outliers, 1], customer_data[~outliers, 2],
           alpha=0.6, label='Normal Customers', s=20)
plt.scatter(customer_data[outliers, 1], customer_data[outliers, 2],
           color='red', s=100, label='Anomalous Customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Customer Analysis - GMM Outlier Detection')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(log_likelihood[:-3], bins=30, alpha=0.7, label='Normal Customers', density=True)
plt.hist(log_likelihood[-3:], bins=5, alpha=0.7, color='red', label='Known Anomalies', density=True)
plt.axvline(x=np.percentile(log_likelihood, 1), color='red', linestyle='--', label='Threshold')
plt.xlabel('Log-Likelihood')
plt.ylabel('Density')
plt.title('Distribution of Log-Likelihood Scores')
plt.legend()

plt.tight_layout()
plt.show()

---

## Robust Statistical Methods

### RANSAC (Random Sample Consensus)

Fits models while being robust to outliers by iteratively selecting random subsets.

**Algorithm**:
1. Randomly select minimal subset of data
2. Fit model to subset
3. Count inliers (points close to model)
4. Repeat and keep best model

**Real-world Example**: Fitting a line to data with measurement errors.

**Code Implementation**:
```python
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np

def detect_outliers_ransac(X, y, residual_threshold=None, max_trials=100):
    """
    Detect outliers using RANSAC regression

    Parameters:
    X: array-like, shape (n_samples, n_features), input features
    y: array-like, shape (n_samples,), target values
    residual_threshold: float, maximum residual for inlier
    max_trials: int, maximum number of iterations

    Returns:
    outliers: boolean array indicating outlier positions
    inlier_mask: boolean array indicating inlier positions
    ransac_model: fitted RANSAC model
    """
    ransac = RANSACRegressor(
        base_estimator=LinearRegression(),
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42
    )

    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outliers = ~inlier_mask

    return outliers, inlier_mask, ransac

# Example: Sensor calibration with measurement errors
np.random.seed(42)

# True linear relationship: y = 2x + 1 + noise
X_true = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * X_true.ravel() + 1 + np.random.normal(0, 0.5, 100)

# Add outliers (measurement errors)
outlier_indices = [20, 35, 50, 65, 80]
y_with_outliers = y_true.copy()
y_with_outliers[outlier_indices] += np.random.uniform(-10, 10, len(outlier_indices))

# Detect outliers using RANSAC
outliers, inliers, ransac_model = detect_outliers_ransac(
    X_true, y_with_outliers, residual_threshold=2.0
)

# Fit regular linear regression for comparison
lr = LinearRegression()
lr.fit(X_true, y_with_outliers)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Actual outlier indices: {outlier_indices}")
print(f"Detected outlier indices: {np.where(outliers)[0]}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_true[inliers], y_with_outliers[inliers],
           alpha=0.7, label='Inliers', s=30)
plt.scatter(X_true[outliers], y_with_outliers[outliers],
           color='red', s=100, label='Detected Outliers')

# Plot regression lines
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
plt.plot(X_plot, ransac_model.predict(X_plot),
         'g-', label='RANSAC Regression', linewidth=2)
plt.plot(X_plot, lr.predict(X_plot),
         'r--', label='Regular Regression', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sensor Calibration - RANSAC vs Regular Regression')
plt.legend()

plt.subplot(1, 2, 2)
# Residuals analysis
residuals_ransac = np.abs(y_with_outliers - ransac_model.predict(X_true))
residuals_lr = np.abs(y_with_outliers - lr.predict(X_true))

plt.scatter(X_true[inliers], residuals_ransac[inliers],
           alpha=0.7, label='RANSAC Inliers', s=30)
plt.scatter(X_true[outliers], residuals_ransac[outliers],
           color='red', s=100, label='RANSAC Outliers')
plt.axhline(y=2.0, color='red', linestyle='--', label='RANSAC Threshold')

plt.xlabel('X')
plt.ylabel('Absolute Residual')
plt.title('Residuals Analysis')
plt.legend()

plt.tight_layout()
plt.show()
````

### Robust Regression (Huber)

Uses robust loss functions that are less sensitive to outliers than ordinary least squares.

**Mathematical Formula**:
Huber loss function:

L(r) = {
0.5 _ r² if |r| ≤ δ
δ _ (|r| - 0.5 \* δ) if |r| > δ
}

Where r is the residual and δ is the threshold parameter.

**Code Implementation**:

```python
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error

def robust_regression_analysis(X, y, epsilon=1.35):
    """
    Perform robust regression using Huber loss

    Parameters:
    X: array-like, input features
    y: array-like, target values
    epsilon: float, Huber loss threshold parameter

    Returns:
    huber_model: fitted Huber regression model
    outlier_weights: array of sample weights (low weights indicate outliers)
    """
    huber = HuberRegressor(epsilon=epsilon, max_iter=100)
    huber.fit(X, y)

    # Get sample weights (outliers have lower weights)
    outlier_weights = huber.fit(X, y).sample_weight_

    return huber, outlier_weights

# Example: Economic data analysis with outliers
np.random.seed(42)

# GDP vs Education spending relationship
education_spending = np.linspace(2, 15, 50).reshape(-1, 1)  # % of GDP
gdp_growth = 0.3 * education_spending.ravel() + 2 + np.random.normal(0, 0.3, 50)

# Add outliers (countries with unusual economic conditions)
outlier_countries = [10, 25, 40]  # Indices of outlier countries
gdp_growth[outlier_countries] += [-2, 3, -1.5]  # Economic shocks/booms

# Compare different regression methods
lr = LinearRegression()
huber, weights = robust_regression_analysis(education_spending, gdp_growth)

lr.fit(education_spending, gdp_growth)

# Identify outliers based on low weights
outlier_threshold = np.percentile(weights, 20)  # Bottom 20% weights
detected_outliers = weights < outlier_threshold

print(f"Countries with unusual economic patterns:")
print(f"Actual outlier indices: {outlier_countries}")
print(f"Detected outlier indices: {np.where(detected_outliers)[0]}")

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(education_spending, gdp_growth, alpha=0.7, s=50)
plt.scatter(education_spending[outlier_countries], gdp_growth[outlier_countries],
           color='red', s=100, label='Known Outliers')

X_plot = np.linspace(2, 15, 100).reshape(-1, 1)
plt.plot(X_plot, lr.predict(X_plot), 'r--', label='OLS Regression', linewidth=2)
plt.plot(X_plot, huber.predict(X_plot), 'g-', label='Huber Regression', linewidth=2)

plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('GDP Growth Rate (%)')
plt.title('Economic Analysis - Robust vs OLS Regression')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(education_spending, weights, alpha=0.7)
plt.scatter(education_spending[detected_outliers], weights[detected_outliers],
           color='red', s=100, label='Low Weight (Outliers)')
plt.axhline(y=outlier_threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('Sample Weight')
plt.title('Huber Regression - Sample Weights')
plt.legend()

plt.subplot(1, 3, 3)
# Residuals comparison
residuals_lr = np.abs(gdp_growth - lr.predict(education_spending))
residuals_huber = np.abs(gdp_growth - huber.predict(education_spending))

plt.scatter(education_spending, residuals_lr, alpha=0.7, label='OLS Residuals')
plt.scatter(education_spending, residuals_huber, alpha=0.7, label='Huber Residuals')
plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('Absolute Residual')
plt.title('Residuals Comparison')
plt.legend()

plt.tight_layout()
plt.show()

---

## Multivariate Outlier Detection

### Mahalanobis Distance

Measures distance accounting for covariance structure of the data.

**Mathematical Formula**:
```

D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

````
Where:
- x = data point
- μ = mean vector
- Σ = covariance matrix

**Real-world Example**: Medical diagnosis based on multiple biomarkers.

**Code Implementation**:
```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def detect_outliers_mahalanobis(data, contamination=0.05):
    """
    Detect outliers using Mahalanobis distance

    Parameters:
    data: array-like, shape (n_samples, n_features)
    contamination: float, expected proportion of outliers

    Returns:
    outliers: boolean array indicating outlier positions
    mahal_distances: array of Mahalanobis distances
    threshold: threshold value used
    """
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # Calculate Mahalanobis distances
    try:
        inv_cov = np.linalg.inv(cov)
        mahal_distances = np.array([
            mahalanobis(x, mean, inv_cov) for x in data
        ])
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if covariance matrix is singular
        inv_cov = np.linalg.pinv(cov)
        mahal_distances = np.array([
            np.sqrt((x - mean).T @ inv_cov @ (x - mean)) for x in data
        ])

    # Mahalanobis distance follows chi-square distribution
    # Use chi-square critical value for threshold
    threshold = np.sqrt(chi2.ppf(1 - contamination, df=data.shape[1]))
    outliers = mahal_distances > threshold

    return outliers, mahal_distances, threshold

# Example: Medical diagnosis - multiple biomarkers
np.random.seed(42)

# Normal patients: [glucose, cholesterol, blood_pressure, bmi]
normal_patients = np.random.multivariate_normal(
    mean=[100, 200, 120, 25],
    cov=[[100, 20, 10, 5], [20, 400, 30, 15],
         [10, 30, 225, 8], [5, 15, 8, 16]],
    size=500
)

# Patients with health issues (outliers)
abnormal_patients = np.array([
    [180, 300, 160, 35],  # Diabetic, high cholesterol, hypertension, obese
    [60, 150, 90, 18],    # Hypoglycemic, underweight
    [120, 400, 140, 30],  # Very high cholesterol
    [200, 250, 180, 40],  # Severe diabetes and hypertension
])

medical_data = np.vstack([normal_patients, abnormal_patients])
feature_names = ['Glucose', 'Cholesterol', 'Blood Pressure', 'BMI']

# Detect outliers using Mahalanobis distance
outliers, distances, threshold = detect_outliers_mahalanobis(
    medical_data, contamination=0.01
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Mahalanobis threshold: {threshold:.2f}")
print(f"Abnormal patients detected: {outliers[-4:]}")

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(medical_data[~outliers, 0], medical_data[~outliers, 1],
           alpha=0.6, label='Normal Patients', s=30)
plt.scatter(medical_data[outliers, 0], medical_data[outliers, 1],
           color='red', s=100, label='Abnormal Patients')
plt.xlabel('Glucose Level')
plt.ylabel('Cholesterol Level')
plt.title('Medical Data - Mahalanobis Distance')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(distances[:-4], bins=30, alpha=0.7, label='Normal Patients', density=True)
plt.hist(distances[-4:], bins=5, alpha=0.7, color='red', label='Known Abnormal', density=True)
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Density')
plt.title('Distribution of Mahalanobis Distances')
plt.legend()

plt.subplot(1, 3, 3)
# Show outlier scores for each patient
outlier_indices = np.where(outliers)[0]
plt.scatter(range(len(distances)), distances, alpha=0.6, label='All Patients')
plt.scatter(outlier_indices, distances[outlier_indices],
           color='red', s=100, label='Detected Outliers')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Patient Index')
plt.ylabel('Mahalanobis Distance')
plt.title('Patient Outlier Scores')
plt.legend()

plt.tight_layout()
plt.show()

### Principal Component Analysis (PCA) for Outlier Detection

Projects data to principal component space where outliers often become more apparent.

**Principle**: Outliers often have unusual patterns in the principal component space, especially in later components.

**Code Implementation**:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def detect_outliers_pca(data, n_components=None, contamination=0.05):
    """
    Detect outliers using PCA reconstruction error

    Parameters:
    data: array-like, shape (n_samples, n_features)
    n_components: int, number of principal components (None for auto)
    contamination: float, expected proportion of outliers

    Returns:
    outliers: boolean array indicating outlier positions
    reconstruction_errors: array of reconstruction errors
    pca_model: fitted PCA model
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Determine number of components if not specified
    if n_components is None:
        # Use components that explain 95% of variance
        pca_temp = PCA()
        pca_temp.fit(data_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.95) + 1

    # Fit PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # Reconstruct data
    data_reconstructed = pca.inverse_transform(data_pca)

    # Calculate reconstruction errors
    reconstruction_errors = np.sum((data_scaled - data_reconstructed) ** 2, axis=1)

    # Define threshold
    threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
    outliers = reconstruction_errors > threshold

    return outliers, reconstruction_errors, pca

# Example: Image quality control in manufacturing
np.random.seed(42)

# Simulate image features: [brightness, contrast, sharpness, color_balance, noise_level]
# Normal images
normal_images = np.random.multivariate_normal(
    mean=[128, 50, 80, 1.0, 10],
    cov=[[100, 10, 5, 0, 2], [10, 25, 8, 0, 1], [5, 8, 16, 0, 1],
         [0, 0, 0, 0.01, 0], [2, 1, 1, 0, 4]],
    size=800
)

# Defective images (outliers)
defective_images = np.array([
    [50, 20, 30, 0.5, 25],    # Very dark, low contrast, blurry
    [200, 80, 90, 1.5, 5],    # Over-exposed, high contrast
    [128, 45, 20, 1.0, 30],   # Normal brightness but very blurry and noisy
    [100, 10, 85, 0.2, 8],    # Low contrast, poor color balance
])

image_data = np.vstack([normal_images, defective_images])
feature_names = ['Brightness', 'Contrast', 'Sharpness', 'Color Balance', 'Noise Level']

# Detect outliers using PCA
outliers, reconstruction_errors, pca_model = detect_outliers_pca(
    image_data, n_components=3, contamination=0.01
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"PCA components used: {pca_model.n_components_}")
print(f"Variance explained: {np.sum(pca_model.explained_variance_ratio_):.3f}")
print(f"Defective images detected: {outliers[-4:]}")

# Visualize
plt.figure(figsize=(15, 10))

# Original data (2D projection)
plt.subplot(2, 3, 1)
plt.scatter(image_data[~outliers, 0], image_data[~outliers, 1],
           alpha=0.6, label='Normal Images', s=30)
plt.scatter(image_data[outliers, 0], image_data[outliers, 1],
           color='red', s=100, label='Defective Images')
plt.xlabel('Brightness')
plt.ylabel('Contrast')
plt.title('Original Feature Space')
plt.legend()

# PCA space
data_scaled = StandardScaler().fit_transform(image_data)
data_pca = pca_model.transform(data_scaled)

plt.subplot(2, 3, 2)
plt.scatter(data_pca[~outliers, 0], data_pca[~outliers, 1],
           alpha=0.6, label='Normal Images', s=30)
plt.scatter(data_pca[outliers, 0], data_pca[outliers, 1],
           color='red', s=100, label='Defective Images')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Feature Space')
plt.legend()

# Reconstruction errors
plt.subplot(2, 3, 3)
plt.hist(reconstruction_errors[:-4], bins=30, alpha=0.7, label='Normal Images', density=True)
plt.hist(reconstruction_errors[-4:], bins=5, alpha=0.7, color='red', label='Known Defective', density=True)
plt.axvline(x=np.percentile(reconstruction_errors, 99), color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.title('PCA Reconstruction Errors')
plt.legend()

# Explained variance
plt.subplot(2, 3, 4)
plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1),
        pca_model.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')

# Feature loadings
plt.subplot(2, 3, 5)
loadings = pca_model.components_.T
for i, feature in enumerate(feature_names):
    plt.scatter(loadings[i, 0], loadings[i, 1], s=100)
    plt.annotate(feature, (loadings[i, 0], loadings[i, 1]),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('PC1 Loading')
plt.ylabel('PC2 Loading')
plt.title('Feature Loadings on PC1 vs PC2')
plt.grid(True, alpha=0.3)

# Outlier scores per sample
plt.subplot(2, 3, 6)
plt.scatter(range(len(reconstruction_errors)), reconstruction_errors, alpha=0.6)
plt.scatter(np.where(outliers)[0], reconstruction_errors[outliers],
           color='red', s=100, label='Detected Outliers')
plt.axhline(y=np.percentile(reconstruction_errors, 99), color='red', linestyle='--')
plt.xlabel('Image Index')
plt.ylabel('Reconstruction Error')
plt.title('Individual Image Outlier Scores')
plt.legend()

plt.tight_layout()
plt.show()

### Minimum Covariance Determinant (MCD)

Robust estimator of covariance matrix that is less affected by outliers.

**Principle**: Finds the subset of observations whose covariance matrix has the smallest determinant.

**Code Implementation**:
```python
from sklearn.covariance import EllipticEnvelope

def detect_outliers_mcd(data, contamination=0.1, support_fraction=None):
    """
    Detect outliers using Minimum Covariance Determinant

    Parameters:
    data: array-like, shape (n_samples, n_features)
    contamination: float, proportion of outliers in the dataset
    support_fraction: float, proportion of points to include in support

    Returns:
    outliers: boolean array indicating outlier positions
    mahalanobis_distances: array of robust Mahalanobis distances
    """
    # Fit MCD estimator
    mcd = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=42
    )

    outlier_labels = mcd.fit_predict(data)
    mahalanobis_distances = mcd.mahalanobis(data)

    outliers = outlier_labels == -1

    return outliers, mahalanobis_distances

# Example: Financial portfolio risk analysis
np.random.seed(42)

# Stock returns: [tech_stocks, financial_stocks, energy_stocks, healthcare_stocks]
# Normal market conditions
normal_returns = np.random.multivariate_normal(
    mean=[0.08, 0.06, 0.05, 0.07],  # Expected annual returns
    cov=[[0.04, 0.01, 0.005, 0.015], [0.01, 0.025, 0.008, 0.01],
         [0.005, 0.008, 0.06, 0.005], [0.015, 0.01, 0.005, 0.03]],
    size=1000
)

# Market crisis periods (outliers)
crisis_returns = np.array([
    [-0.4, -0.3, -0.5, -0.2],  # Financial crisis
    [-0.3, -0.4, -0.6, -0.1],  # Tech bubble burst
    [0.5, 0.2, 0.8, 0.3],      # Speculative bubble
    [-0.2, -0.5, -0.3, -0.4],  # General market crash
])

portfolio_returns = np.vstack([normal_returns, crisis_returns])
sector_names = ['Technology', 'Financial', 'Energy', 'Healthcare']

# Detect outliers using MCD
outliers, robust_distances = detect_outliers_mcd(
    portfolio_returns, contamination=0.005
)

print(f"Number of outlier periods detected: {np.sum(outliers)}")
print(f"Crisis periods detected: {outliers[-4:]}")

# Compare with standard Mahalanobis distance
standard_outliers, standard_distances, _ = detect_outliers_mahalanobis(
    portfolio_returns, contamination=0.005
)

# Visualize
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(portfolio_returns[~outliers, 0], portfolio_returns[~outliers, 1],
           alpha=0.6, label='Normal Periods', s=30)
plt.scatter(portfolio_returns[outliers, 0], portfolio_returns[outliers, 1],
           color='red', s=100, label='Crisis Periods (MCD)')
plt.xlabel('Technology Returns')
plt.ylabel('Financial Returns')
plt.title('Portfolio Returns - MCD Outlier Detection')
plt.legend()

plt.subplot(2, 3, 2)
plt.scatter(portfolio_returns[~standard_outliers, 0], portfolio_returns[~standard_outliers, 1],
           alpha=0.6, label='Normal Periods', s=30)
plt.scatter(portfolio_returns[standard_outliers, 0], portfolio_returns[standard_outliers, 1],
           color='orange', s=100, label='Crisis Periods (Standard)')
plt.xlabel('Technology Returns')
plt.ylabel('Financial Returns')
plt.title('Portfolio Returns - Standard Mahalanobis')
plt.legend()

plt.subplot(2, 3, 3)
plt.hist(robust_distances[:-4], bins=30, alpha=0.7, label='Normal Periods', density=True)
plt.hist(robust_distances[-4:], bins=5, alpha=0.7, color='red', label='Known Crises', density=True)
plt.xlabel('Robust Mahalanobis Distance')
plt.ylabel('Density')
plt.title('Distribution of Robust Distances')
plt.legend()

plt.subplot(2, 3, 4)
# Time series view
time_periods = range(len(portfolio_returns))
plt.plot(time_periods, robust_distances, alpha=0.7, label='Robust Distance')
plt.scatter(np.where(outliers)[0], robust_distances[outliers],
           color='red', s=50, label='Detected Outliers', zorder=5)
plt.xlabel('Time Period')
plt.ylabel('Robust Mahalanobis Distance')
plt.title('Outlier Detection Over Time')
plt.legend()

plt.subplot(2, 3, 5)
# Comparison of methods
plt.scatter(standard_distances, robust_distances, alpha=0.6)
plt.scatter(standard_distances[outliers], robust_distances[outliers],
           color='red', s=100, label='MCD Outliers')
plt.scatter(standard_distances[standard_outliers], robust_distances[standard_outliers],
           color='orange', s=100, marker='^', label='Standard Outliers')
plt.xlabel('Standard Mahalanobis Distance')
plt.ylabel('Robust Mahalanobis Distance')
plt.title('Comparison of Distance Methods')
plt.legend()

# Correlation matrix heatmap
plt.subplot(2, 3, 6)
correlation_matrix = np.corrcoef(portfolio_returns.T)
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(sector_names)), sector_names, rotation=45)
plt.yticks(range(len(sector_names)), sector_names)
plt.title('Portfolio Correlation Matrix')

plt.tight_layout()
plt.show()

---

## Time Series Outlier Detection

### Seasonal Decomposition

Separates time series into trend, seasonal, and residual components to identify anomalies.

**Components**:
- **Trend**: Long-term movement
- **Seasonal**: Regular periodic patterns
- **Residual**: Random fluctuations (where outliers are detected)

**Code Implementation**:
```python
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

def detect_outliers_seasonal_decomposition(data, model='additive', period=None, contamination=0.05):
    """
    Detect outliers using seasonal decomposition

    Parameters:
    data: array-like, time series data
    model: str, 'additive' or 'multiplicative'
    period: int, period of seasonality (None for auto-detection)
    contamination: float, expected proportion of outliers

    Returns:
    outliers: boolean array indicating outlier positions
    residuals: array of residual values
    decomposition: statsmodels decomposition result
    """
    # Create time series
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data, model=model, period=period)
    residuals = decomposition.resid.dropna()

    # Detect outliers in residuals using IQR method
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 2.5  # More sensitive threshold for time series

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)

    # Align with original data indices
    outliers = np.zeros(len(data), dtype=bool)
    outliers[residuals.index] = outlier_mask

    return outliers, residuals.values, decomposition

# Example: E-commerce website traffic analysis
np.random.seed(42)

# Generate synthetic daily website traffic data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
n_days = len(dates)

# Base traffic with trend
trend = np.linspace(1000, 1500, n_days)

# Weekly seasonality (higher traffic on weekends)
weekly_pattern = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)

# Annual seasonality (higher traffic during holidays)
annual_pattern = 150 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)

# Random noise
noise = np.random.normal(0, 50, n_days)

# Combine components
traffic = trend + weekly_pattern + annual_pattern + noise

# Add outliers (special events, technical issues)
outlier_days = [50, 120, 200, 300, 350]  # Specific days with unusual traffic
traffic[outlier_days] += [800, -500, 1200, -300, 600]  # Traffic spikes and drops

# Create DataFrame
traffic_data = pd.DataFrame({
    'date': dates,
    'traffic': traffic
})

# Detect outliers using seasonal decomposition
outliers, residuals, decomp = detect_outliers_seasonal_decomposition(
    traffic_data['traffic'], model='additive', period=7
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Outlier dates: {dates[outliers]}")
print(f"Actual outlier days: {[dates[i] for i in outlier_days]}")

# Visualize
plt.figure(figsize=(15, 12))

# Original time series with outliers
plt.subplot(4, 1, 1)
plt.plot(dates, traffic, alpha=0.7, label='Website Traffic')
plt.scatter(dates[outliers], traffic[outliers], color='red', s=50, label='Detected Outliers', zorder=5)
plt.ylabel('Daily Visitors')
plt.title('Website Traffic Analysis - Seasonal Decomposition')
plt.legend()

# Trend component
plt.subplot(4, 1, 2)
plt.plot(dates, decomp.trend, color='blue', label='Trend')
plt.ylabel('Trend')
plt.legend()

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(dates, decomp.seasonal, color='green', label='Seasonal')
plt.ylabel('Seasonal')
plt.legend()

# Residuals with outliers
plt.subplot(4, 1, 4)
plt.plot(dates[1:-1], residuals, alpha=0.7, label='Residuals')  # Skip first and last (NaN in decomposition)
residual_outliers = outliers[1:-1]  # Adjust for NaN values
plt.scatter(dates[1:-1][residual_outliers], residuals[residual_outliers],
           color='red', s=50, label='Detected Outliers', zorder=5)
plt.ylabel('Residuals')
plt.xlabel('Date')
plt.legend()

plt.tight_layout()
plt.show()

### ARIMA-based Outlier Detection

Fits autoregressive models and identifies outliers as points with large residuals.

**Mathematical Concept**: ARIMA(p,d,q) models use past values and errors to predict current values. Large prediction errors indicate outliers.

**Code Implementation**:
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

def detect_outliers_arima(data, order=(1,1,1), contamination=0.05):
    """
    Detect outliers using ARIMA model residuals

    Parameters:
    data: array-like, time series data
    order: tuple, ARIMA order (p,d,q)
    contamination: float, expected proportion of outliers

    Returns:
    outliers: boolean array indicating outlier positions
    residuals: array of model residuals
    model: fitted ARIMA model
    """
    # Fit ARIMA model
    model = ARIMA(data, order=order)
    fitted_model = model.fit()

    # Get residuals
    residuals = fitted_model.resid

    # Detect outliers using standardized residuals
    std_residuals = residuals / np.std(residuals)
    threshold = np.percentile(np.abs(std_residuals), (1 - contamination) * 100)

    outliers = np.abs(std_residuals) > threshold

    return outliers, residuals, fitted_model

# Example: Stock price anomaly detection
np.random.seed(42)

# Generate synthetic stock price data with random walk + trend
n_days = 365
dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

# Base stock price evolution (random walk with drift)
returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with slight upward trend
prices = [100]  # Starting price

for i in range(1, n_days):
    prices.append(prices[-1] * (1 + returns[i]))

prices = np.array(prices)

# Add outliers (market events)
event_days = [50, 150, 250, 300]
price_shocks = [0.15, -0.20, 0.12, -0.08]  # 15% jump, 20% drop, etc.

for day, shock in zip(event_days, price_shocks):
    prices[day:] *= (1 + shock)  # Persistent effect

# Create DataFrame
stock_data = pd.DataFrame({
    'date': dates,
    'price': prices
})

# Detect outliers using ARIMA
outliers, residuals, arima_model = detect_outliers_arima(
    stock_data['price'], order=(2,1,1), contamination=0.02
)

print(f"ARIMA Model Summary:")
print(f"AIC: {arima_model.aic:.2f}")
print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Outlier dates: {dates[outliers]}")

# Calculate daily returns for analysis
daily_returns = np.diff(prices) / prices[:-1]

# Visualize
plt.figure(figsize=(15, 10))

# Stock price with outliers
plt.subplot(3, 1, 1)
plt.plot(dates, prices, alpha=0.8, label='Stock Price')
plt.scatter(dates[outliers], prices[outliers], color='red', s=50, label='Detected Outliers', zorder=5)
plt.ylabel('Stock Price ($)')
plt.title('Stock Price Analysis - ARIMA Outlier Detection')
plt.legend()

# Model residuals
plt.subplot(3, 1, 2)
plt.plot(dates, residuals, alpha=0.7, label='ARIMA Residuals')
plt.scatter(dates[outliers], residuals[outliers], color='red', s=50, label='Outliers', zorder=5)
plt.ylabel('Residuals')
plt.legend()

# Daily returns distribution
plt.subplot(3, 1, 3)
plt.hist(daily_returns, bins=50, alpha=0.7, density=True, label='All Returns')
outlier_returns = daily_returns[outliers[1:]]  # Adjust for diff operation
plt.hist(outlier_returns, bins=10, alpha=0.8, color='red', density=True, label='Outlier Returns')
plt.xlabel('Daily Returns')
plt.ylabel('Density')
plt.title('Distribution of Daily Returns')
plt.legend()

plt.tight_layout()
plt.show()

# Model diagnostics
print(f"\nModel Diagnostics:")
print(f"Ljung-Box test p-value: {acorr_ljungbox(residuals, lags=10, return_df=True)['lb_pvalue'].iloc[-1]:.4f}")

### Change Point Detection

Identifies sudden changes in time series behavior.

**Methods**:
- **CUSUM**: Cumulative sum control chart
- **Bayesian Change Point**: Probabilistic approach
- **Binary Segmentation**: Recursive partitioning

**Code Implementation**:
```python
import ruptures as rpt  # You may need: pip install ruptures

def detect_change_points(data, model='rbf', min_size=10, jump=5):
    """
    Detect change points in time series

    Parameters:
    data: array-like, time series data
    model: str, change point detection model ('l1', 'l2', 'rbf', 'normal')
    min_size: int, minimum segment size
    jump: int, jump size for computational efficiency

    Returns:
    change_points: list of change point indices
    segments: list of segment boundaries
    """
    # Use Binary Segmentation algorithm
    algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
    algo.fit(data)

    # Detect change points (automatic number selection)
    change_points = algo.predict(pen=np.log(len(data)))

    # Create segments
    segments = [0] + change_points

    return change_points[:-1], segments  # Remove last point (end of series)

def detect_outliers_change_points(data, window_size=50, threshold=2.0):
    """
    Detect outliers around change points

    Parameters:
    data: array-like, time series data
    window_size: int, size of window around change points
    threshold: float, standard deviation threshold

    Returns:
    outliers: boolean array indicating outlier positions
    change_points: list of detected change points
    """
    change_points, _ = detect_change_points(data)
    outliers = np.zeros(len(data), dtype=bool)

    # Check for outliers around each change point
    for cp in change_points:
        start = max(0, cp - window_size // 2)
        end = min(len(data), cp + window_size // 2)

        segment = data[start:end]
        mean_segment = np.mean(segment)
        std_segment = np.std(segment)

        # Mark points that deviate significantly from segment statistics
        for i in range(start, end):
            if abs(data[i] - mean_segment) > threshold * std_segment:
                outliers[i] = True

    return outliers, change_points

# Example: Network performance monitoring
np.random.seed(42)

# Generate network latency data with regime changes
n_points = 1000
time_points = np.arange(n_points)

# Different network states
latency = np.zeros(n_points)

# Normal operation (0-300)
latency[0:300] = np.random.normal(50, 5, 300)

# Network congestion (300-600)
latency[300:600] = np.random.normal(120, 15, 300)

# After optimization (600-800)
latency[600:800] = np.random.normal(30, 3, 200)

# Server issues (800-1000)
latency[800:1000] = np.random.normal(200, 25, 200)

# Add some outliers (network spikes)
spike_indices = [100, 250, 450, 750, 900]
latency[spike_indices] += np.random.uniform(100, 300, len(spike_indices))

# Detect change points and outliers
try:
    outliers, change_points = detect_outliers_change_points(latency, window_size=100, threshold=2.5)
    change_points_detected, segments = detect_change_points(latency)

    print(f"Number of change points detected: {len(change_points_detected)}")
    print(f"Change points at indices: {change_points_detected}")
    print(f"Number of outliers detected: {np.sum(outliers)}")

    # Visualize
    plt.figure(figsize=(15, 8))

    # Main time series plot
    plt.subplot(2, 1, 1)
    plt.plot(time_points, latency, alpha=0.7, label='Network Latency')

    # Mark change points
    for cp in change_points_detected:
        plt.axvline(x=cp, color='blue', linestyle='--', alpha=0.7, label='Change Point' if cp == change_points_detected[0] else "")

    # Mark outliers
    plt.scatter(time_points[outliers], latency[outliers], color='red', s=50, label='Detected Outliers', zorder=5)

    plt.ylabel('Latency (ms)')
    plt.title('Network Performance Monitoring - Change Point Detection')
    plt.legend()

    # Segment analysis
    plt.subplot(2, 1, 2)
    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    for i, (start, end) in enumerate(zip([0] + change_points_detected, change_points_detected + [len(latency)])):
        segment_data = latency[start:end]
        plt.plot(range(start, end), segment_data, color=colors[i % len(colors)],
                label=f'Segment {i+1} (μ={np.mean(segment_data):.1f})', linewidth=2)

    plt.ylabel('Latency (ms)')
    plt.xlabel('Time Point')
    plt.title('Identified Network Performance Segments')
    plt.legend()

    plt.tight_layout()
    plt.show()

except ImportError:
    print("Ruptures package not available. Using alternative change point detection:")

    # Simple CUSUM-based change point detection
    def simple_cusum_change_points(data, threshold=5):
        """Simple CUSUM change point detection"""
        mean_data = np.mean(data)
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))

        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean_data - 1)
            cusum_neg[i] = max(0, cusum_neg[i-1] - data[i] + mean_data - 1)

        change_points = []
        for i in range(len(data)):
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)

        return change_points

    change_points_simple = simple_cusum_change_points(latency, threshold=100)
    print(f"Simple CUSUM change points: {change_points_simple[:10]}")  # Show first 10

### Sliding Window Approach

Detects outliers by comparing current observations with recent historical data.

**Code Implementation**:
```python
def detect_outliers_sliding_window(data, window_size=50, threshold=2.5, step_size=1):
    """
    Detect outliers using sliding window approach

    Parameters:
    data: array-like, time series data
    window_size: int, size of sliding window
    threshold: float, standard deviation threshold
    step_size: int, step size for sliding window

    Returns:
    outliers: boolean array indicating outlier positions
    anomaly_scores: array of anomaly scores
    """
    outliers = np.zeros(len(data), dtype=bool)
    anomaly_scores = np.zeros(len(data))

    for i in range(window_size, len(data), step_size):
        # Define window
        window_start = max(0, i - window_size)
        window_data = data[window_start:i]

        # Calculate statistics for window
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)

        # Check current point
        if window_std > 0:  # Avoid division by zero
            z_score = abs(data[i] - window_mean) / window_std
            anomaly_scores[i] = z_score

            if z_score > threshold:
                outliers[i] = True

    return outliers, anomaly_scores

# Example: Real-time sensor monitoring
np.random.seed(42)

# Generate sensor temperature data
n_hours = 720  # 30 days of hourly data
time_hours = np.arange(n_hours)

# Base temperature with daily cycle
daily_cycle = 10 * np.sin(2 * np.pi * time_hours / 24)
base_temp = 20 + daily_cycle + np.random.normal(0, 1, n_hours)

# Add gradual sensor drift
drift = 0.01 * time_hours
sensor_temp = base_temp + drift

# Add sensor malfunctions (outliers)
malfunction_hours = [100, 250, 400, 500, 650]
sensor_temp[malfunction_hours] += np.random.uniform(-15, 15, len(malfunction_hours))

# Detect outliers using sliding window
outliers, anomaly_scores = detect_outliers_sliding_window(
    sensor_temp, window_size=48, threshold=3.0  # 48-hour window
)

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Malfunction hours detected: {np.array(malfunction_hours)[np.isin(malfunction_hours, np.where(outliers)[0])]}")

# Visualize
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(time_hours, sensor_temp, alpha=0.7, label='Sensor Temperature')
plt.scatter(time_hours[outliers], sensor_temp[outliers], color='red', s=50, label='Detected Outliers', zorder=5)
plt.scatter(malfunction_hours, sensor_temp[malfunction_hours], color='orange', s=100, marker='^',
           label='Known Malfunctions', zorder=5)
plt.ylabel('Temperature (°C)')
plt.title('Real-time Sensor Monitoring - Sliding Window Detection')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_hours, anomaly_scores, alpha=0.7, label='Anomaly Scores')
plt.axhline(y=3.0, color='red', linestyle='--', label='Threshold')
plt.scatter(time_hours[outliers], anomaly_scores[outliers], color='red', s=50, zorder=5)
plt.ylabel('Anomaly Score (Z-score)')
plt.xlabel('Time (hours)')
plt.title('Anomaly Scores Over Time')
plt.legend()

plt.tight_layout()
plt.show()

---

## Evaluation Metrics

Understanding how to evaluate outlier detection performance is crucial for selecting the right method and tuning parameters.

### Classification Metrics

When ground truth labels are available, we can treat outlier detection as a binary classification problem.

**Code Implementation**:
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

def evaluate_outlier_detection(y_true, y_pred, y_scores=None):
    """
    Comprehensive evaluation of outlier detection performance

    Parameters:
    y_true: array-like, true binary labels (1 for outlier, 0 for normal)
    y_pred: array-like, predicted binary labels
    y_scores: array-like, anomaly scores (optional, for AUC calculation)

    Returns:
    metrics: dict, containing all evaluation metrics
    """
    # Basic classification metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }

    # AUC-ROC if scores provided
    if y_scores is not None:
        auc_roc = roc_auc_score(y_true, y_scores)
        metrics['auc_roc'] = auc_roc

    return metrics

def plot_evaluation_results(y_true, y_pred, y_scores=None, method_name="Outlier Detection"):
    """Plot comprehensive evaluation results"""
    metrics = evaluate_outlier_detection(y_true, y_pred, y_scores)

    plt.figure(figsize=(15, 5))

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Outlier'],
                yticklabels=['Normal', 'Outlier'])
    plt.title(f'{method_name}\nConfusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Metrics bar plot
    plt.subplot(1, 3, 2)
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics['precision'], metrics['recall'],
                    metrics['f1_score'], metrics['specificity']]

    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'coral', 'gold'])
    plt.ylim(0, 1)
    plt.title(f'{method_name}\nPerformance Metrics')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    # ROC Curve (if scores available)
    if y_scores is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.subplot(1, 3, 3)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{method_name}\nROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Score distribution if no scores
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, 'No anomaly scores\nprovided for ROC curve',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('ROC Curve Not Available')

    plt.tight_layout()
    plt.show()

    return metrics

# Example: Comprehensive evaluation of multiple methods
np.random.seed(42)

# Generate synthetic dataset
n_normal = 1000
n_outliers = 50

# Normal data: 2D Gaussian
normal_data = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_normal)

# Outlier data: points far from normal distribution
outlier_data = np.random.uniform(-4, 4, (n_outliers, 2))
outlier_data = outlier_data[np.linalg.norm(outlier_data, axis=1) > 2.5]  # Keep only distant points

# Combine data
all_data = np.vstack([normal_data, outlier_data[:n_outliers]])
true_labels = np.concatenate([np.zeros(n_normal), np.ones(len(outlier_data[:n_outliers]))])

# Apply multiple detection methods
methods = {}

# 1. Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_predictions = iso_forest.fit_predict(all_data)
iso_scores = iso_forest.decision_function(all_data)
methods['Isolation Forest'] = {
    'predictions': (iso_predictions == -1).astype(int),
    'scores': -iso_scores  # Convert to positive scores
}

# 2. Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_predictions = lof.fit_predict(all_data)
lof_scores = -lof.negative_outlier_factor_
methods['LOF'] = {
    'predictions': (lof_predictions == -1).astype(int),
    'scores': lof_scores
}

# 3. One-Class SVM
from sklearn.svm import OneClassSVM
oc_svm = OneClassSVM(nu=0.05, kernel='rbf')
svm_predictions = oc_svm.fit_predict(all_data)
svm_scores = -oc_svm.decision_function(all_data)
methods['One-Class SVM'] = {
    'predictions': (svm_predictions == -1).astype(int),
    'scores': svm_scores
}

# 4. Mahalanobis Distance
def mahalanobis_method(data, contamination=0.05):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)

    distances = np.array([
        mahalanobis(x, mean, inv_cov) for x in data
    ])

    threshold = np.percentile(distances, (1 - contamination) * 100)
    predictions = (distances > threshold).astype(int)

    return predictions, distances

mahal_predictions, mahal_scores = mahalanobis_method(all_data)
methods['Mahalanobis'] = {
    'predictions': mahal_predictions,
    'scores': mahal_scores
}

# Evaluate all methods
print("Outlier Detection Method Comparison")
print("=" * 50)

results_summary = []

for method_name, method_data in methods.items():
    print(f"\n{method_name}:")
    metrics = plot_evaluation_results(
        true_labels,
        method_data['predictions'],
        method_data['scores'],
        method_name
    )

    # Store results for comparison
    results_summary.append({
        'Method': method_name,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'AUC-ROC': metrics.get('auc_roc', 'N/A')
    })

    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")

# Summary comparison
results_df = pd.DataFrame(results_summary)
print(f"\nSummary Comparison:")
print(results_df.to_string(index=False))

# Best method selection
best_f1_method = results_df.loc[results_df['F1-Score'].idxmax(), 'Method']
print(f"\nBest method by F1-Score: {best_f1_method}")

---

## Practical Considerations

### Dealing with High-Dimensional Data

High-dimensional data poses unique challenges for outlier detection due to the "curse of dimensionality."

**Problems**:
1. **Distance concentration**: All points appear equally distant
2. **Sparse data**: Data becomes increasingly sparse
3. **Noise accumulation**: Irrelevant features add noise

**Solutions**:

**Code Implementation**:
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE

def handle_high_dimensional_outliers(data, method='pca', n_components=10):
    """
    Handle high-dimensional data for outlier detection

    Parameters:
    data: array-like, high-dimensional data
    method: str, dimensionality reduction method ('pca', 'feature_selection', 'tsne')
    n_components: int, number of components/features to keep

    Returns:
    reduced_data: array-like, reduced dimensional data
    transformer: fitted transformer object
    """
    if method == 'pca':
        # PCA for linear dimensionality reduction
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        transformer = pca

    elif method == 'feature_selection':
        # Feature selection (requires labels - using variance for unsupervised)
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.1)  # Remove low-variance features
        reduced_data = selector.fit_transform(data)
        transformer = selector

    elif method == 'tsne':
        # t-SNE for non-linear dimensionality reduction
        tsne = TSNE(n_components=min(n_components, 3), random_state=42)
        reduced_data = tsne.fit_transform(data)
        transformer = tsne

    return reduced_data, transformer

# Example: High-dimensional gene expression data
np.random.seed(42)

# Simulate gene expression data (1000 genes, 200 samples)
n_genes = 1000
n_samples = 200
n_outliers = 10

# Normal samples: correlated gene expression
normal_samples = np.random.multivariate_normal(
    mean=np.zeros(n_genes),
    cov=np.eye(n_genes) + 0.1 * np.random.random((n_genes, n_genes)),
    size=n_samples - n_outliers
)

# Outlier samples: different expression pattern
outlier_samples = np.random.normal(2, 1, (n_outliers, n_genes))

# Combine data
gene_expression = np.vstack([normal_samples, outlier_samples])
true_labels = np.concatenate([np.zeros(n_samples - n_outliers), np.ones(n_outliers)])

print(f"Original data shape: {gene_expression.shape}")

# Apply different dimensionality reduction methods
reduction_methods = ['pca', 'tsne']
detection_results = {}

for reduction_method in reduction_methods:
    # Reduce dimensionality
    reduced_data, transformer = handle_high_dimensional_outliers(
        gene_expression, method=reduction_method, n_components=10
    )

    print(f"Reduced data shape ({reduction_method}): {reduced_data.shape}")

    # Apply outlier detection on reduced data
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(reduced_data)
    scores = -iso_forest.decision_function(reduced_data)

    # Evaluate
    metrics = evaluate_outlier_detection(
        true_labels,
        (predictions == -1).astype(int),
        scores
    )

    detection_results[reduction_method] = metrics

    print(f"{reduction_method.upper()} + Isolation Forest:")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")

# Compare with direct high-dimensional detection
iso_forest_direct = IsolationForest(contamination=0.05, random_state=42)
predictions_direct = iso_forest_direct.fit_predict(gene_expression)
scores_direct = -iso_forest_direct.decision_function(gene_expression)

metrics_direct = evaluate_outlier_detection(
    true_labels,
    (predictions_direct == -1).astype(int),
    scores_direct
)

print(f"\nDirect High-Dimensional Detection:")
print(f"  F1-Score: {metrics_direct['f1_score']:.3f}")
print(f"  AUC-ROC: {metrics_direct['auc_roc']:.3f}")

### Handling Imbalanced Data

Outliers are typically rare, making the dataset highly imbalanced.

**Strategies**:
1. **Adjust contamination parameter**: Set based on domain knowledge
2. **Cost-sensitive learning**: Penalize false negatives more
3. **Ensemble methods**: Combine multiple detectors
4. **Threshold tuning**: Optimize based on business cost

**Code Implementation**:
```python
def optimize_threshold_for_imbalanced_data(y_true, y_scores, cost_fp=1, cost_fn=10):
    """
    Optimize threshold for imbalanced outlier detection considering costs

    Parameters:
    y_true: array-like, true binary labels
    y_scores: array-like, anomaly scores
    cost_fp: float, cost of false positive
    cost_fn: float, cost of false negative

    Returns:
    optimal_threshold: float, threshold that minimizes total cost
    thresholds: array, all tested thresholds
    costs: array, corresponding total costs
    """
    # Test range of thresholds
    thresholds = np.percentile(y_scores, np.linspace(1, 99, 99))
    costs = []

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate total cost
        total_cost = cost_fp * fp + cost_fn * fn
        costs.append(total_cost)

    # Find optimal threshold
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, thresholds, np.array(costs)

# Example: Credit card fraud detection with cost considerations
np.random.seed(42)

# Simulate credit card transaction data
n_normal = 10000
n_fraud = 100  # Only 1% fraud (highly imbalanced)

# Normal transactions: small amounts, regular patterns
normal_transactions = np.column_stack([
    np.random.exponential(50, n_normal),      # Amount
    np.random.uniform(6, 22, n_normal),       # Hour of day
    np.random.randint(1, 8, n_normal),        # Merchant category
    np.random.beta(2, 5, n_normal)            # Risk score
])

# Fraudulent transactions: different patterns
fraud_transactions = np.column_stack([
    np.random.exponential(200, n_fraud),      # Higher amounts
    np.random.uniform(0, 6, n_fraud),         # Unusual hours
    np.random.randint(6, 10, n_fraud),        # High-risk categories
    np.random.beta(5, 2, n_fraud)             # Higher risk scores
])

# Combine data
transaction_data = np.vstack([normal_transactions, fraud_transactions])
fraud_labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

# Train outlier detector
iso_forest = IsolationForest(contamination=0.01, random_state=42)
fraud_scores = -iso_forest.fit_predict(transaction_data)
fraud_scores = -iso_forest.decision_function(transaction_data)

# Cost considerations:
# False Positive cost: $5 (investigation cost)
# False Negative cost: $500 (average fraud loss)
optimal_threshold, thresholds, costs = optimize_threshold_for_imbalanced_data(
    fraud_labels, fraud_scores, cost_fp=5, cost_fn=500
)

# Compare with default threshold (contamination-based)
default_threshold = np.percentile(fraud_scores, 99)  # 1% contamination

print(f"Credit Card Fraud Detection - Cost Optimization")
print(f"Dataset: {n_normal} normal, {n_fraud} fraud transactions")
print(f"Default threshold (99th percentile): {default_threshold:.3f}")
print(f"Optimal threshold (cost-minimizing): {optimal_threshold:.3f}")

# Evaluate both approaches
for threshold_name, threshold in [("Default", default_threshold), ("Optimal", optimal_threshold)]:
    predictions = (fraud_scores > threshold).astype(int)
    metrics = evaluate_outlier_detection(fraud_labels, predictions, fraud_scores)

    # Calculate costs
    total_cost = 5 * metrics['false_positives'] + 500 * metrics['false_negatives']

    print(f"\n{threshold_name} Threshold Results:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Total Cost: ${total_cost:,}")

# Visualize cost optimization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, costs)
plt.axvline(x=optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
plt.axvline(x=default_threshold, color='blue', linestyle='--', label='Default Threshold')
plt.xlabel('Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Cost vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Show precision-recall trade-off
precisions, recalls = [], []
for threshold in thresholds:
    predictions = (fraud_scores > threshold).astype(int)
    if np.sum(predictions) > 0:  # Avoid division by zero
        precision = precision_score(fraud_labels, predictions)
        recall = recall_score(fraud_labels, predictions)
        precisions.append(precision)
        recalls.append(recall)

plt.plot(recalls, precisions, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

### Domain Knowledge Integration

Incorporating domain expertise is crucial for effective outlier detection.

**Strategies**:
1. **Feature engineering**: Create domain-specific features
2. **Rule-based filtering**: Apply business rules before detection
3. **Interpretable models**: Use models that provide explanations
4. **Human-in-the-loop**: Combine automated detection with expert review

**Code Implementation**:
```python
def create_domain_features(data, domain='financial'):
    """
    Create domain-specific features for better outlier detection

    Parameters:
    data: array-like, raw feature data
    domain: str, domain type ('financial', 'medical', 'network')

    Returns:
    enhanced_data: array-like, data with additional domain features
    feature_names: list, names of all features
    """
    if domain == 'financial':
        # Financial transaction features
        # Assuming data columns: [amount, time_hour, merchant_cat, user_age]
        amounts = data[:, 0]
        time_hours = data[:, 1]
        merchant_cats = data[:, 2]
        user_ages = data[:, 3]

        # Create domain features
        amount_log = np.log1p(amounts)  # Log-transformed amount
        is_weekend = ((time_hours // 24) % 7 >= 5).astype(int)  # Weekend indicator
        is_night = ((time_hours % 24 < 6) | (time_hours % 24 > 22)).astype(int)  # Night hours
        high_risk_merchant = (merchant_cats >= 7).astype(int)  # High-risk merchant
        amount_user_age_ratio = amounts / (user_ages + 1)  # Amount relative to age

        enhanced_data = np.column_stack([
            data,  # Original features
            amount_log,
            is_weekend,
            is_night,
            high_risk_merchant,
            amount_user_age_ratio
        ])

        feature_names = [
            'amount', 'time_hour', 'merchant_cat', 'user_age',
            'amount_log', 'is_weekend', 'is_night', 'high_risk_merchant', 'amount_age_ratio'
        ]

    return enhanced_data, feature_names

def apply_business_rules(data, feature_names, domain='financial'):
    """
    Apply domain-specific business rules for outlier detection

    Parameters:
    data: array-like, feature data
    feature_names: list, feature names
    domain: str, domain type

    Returns:
    rule_outliers: boolean array, points flagged by business rules
    rule_explanations: list, explanations for each flagged point
    """
    rule_outliers = np.zeros(len(data), dtype=bool)
    rule_explanations = [''] * len(data)

    if domain == 'financial':
        amount_idx = feature_names.index('amount')
        time_idx = feature_names.index('time_hour')
        night_idx = feature_names.index('is_night')

        for i in range(len(data)):
            reasons = []

            # Rule 1: Very high amount transactions
            if data[i, amount_idx] > 5000:
                reasons.append(f"High amount: ${data[i, amount_idx]:.2f}")

            # Rule 2: Night transactions over $1000
            if data[i, night_idx] == 1 and data[i, amount_idx] > 1000:
                reasons.append(f"Night transaction: ${data[i, amount_idx]:.2f} at {data[i, time_idx]:.1f}h")

            # Rule 3: Multiple transactions in short time (simplified)
            # This would require timestamp data in practice

            if reasons:
                rule_outliers[i] = True
                rule_explanations[i] = "; ".join(reasons)

    return rule_outliers, rule_explanations

# Example: Comprehensive fraud detection system
np.random.seed(42)

# Generate enhanced transaction data
n_transactions = 5000
base_data = np.column_stack([
    np.random.exponential(100, n_transactions),    # Amount
    np.random.uniform(0, 24*7, n_transactions),    # Time (hours in week)
    np.random.randint(1, 10, n_transactions),      # Merchant category
    np.random.uniform(18, 80, n_transactions)      # User age
])

# Add some clear fraud cases
fraud_indices = [100, 500, 1000, 2000, 3000]
base_data[fraud_indices, 0] = np.random.uniform(8000, 15000, len(fraud_indices))  # Very high amounts
base_data[fraud_indices, 1] = np.random.uniform(24, 30, len(fraud_indices))       # Night hours

# Create domain features
enhanced_data, feature_names = create_domain_features(base_data, domain='financial')

print(f"Enhanced feature set: {feature_names}")
print(f"Data shape: {enhanced_data.shape}")

# Apply business rules
rule_outliers, rule_explanations = apply_business_rules(enhanced_data, feature_names)

print(f"Business rules flagged: {np.sum(rule_outliers)} transactions")

# Apply ML-based detection
iso_forest = IsolationForest(contamination=0.02, random_state=42)
ml_predictions = iso_forest.fit_predict(enhanced_data)
ml_outliers = (ml_predictions == -1)

# Combine rule-based and ML-based detection
combined_outliers = rule_outliers | ml_outliers

print(f"Rule-based outliers: {np.sum(rule_outliers)}")
print(f"ML-based outliers: {np.sum(ml_outliers)}")
print(f"Combined outliers: {np.sum(combined_outliers)}")
print(f"Overlap: {np.sum(rule_outliers & ml_outliers)}")

# Analyze flagged transactions
flagged_indices = np.where(combined_outliers)[0]
print(f"\nSample flagged transactions:")
for i in flagged_indices[:5]:  # Show first 5
    print(f"Transaction {i}:")
    print(f"  Amount: ${enhanced_data[i, 0]:.2f}")
    print(f"  Time: {enhanced_data[i, 1]:.1f}h")
    print(f"  Rule explanation: {rule_explanations[i] if rule_explanations[i] else 'ML-only detection'}")

# Visualize detection results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(enhanced_data[~rule_outliers, 0], enhanced_data[~rule_outliers, 1],
           alpha=0.6, label='Normal (Rules)', s=20)
plt.scatter(enhanced_data[rule_outliers, 0], enhanced_data[rule_outliers, 1],
           color='red', s=100, label='Rule-based Outliers')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Time (hours)')
plt.title('Rule-based Detection')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(enhanced_data[~ml_outliers, 0], enhanced_data[~ml_outliers, 1],
           alpha=0.6, label='Normal (ML)', s=20)
plt.scatter(enhanced_data[ml_outliers, 0], enhanced_data[ml_outliers, 1],
           color='orange', s=100, label='ML-based Outliers')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Time (hours)')
plt.title('ML-based Detection')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(enhanced_data[~combined_outliers, 0], enhanced_data[~combined_outliers, 1],
           alpha=0.6, label='Normal', s=20)
plt.scatter(enhanced_data[rule_outliers & ~ml_outliers, 0], enhanced_data[rule_outliers & ~ml_outliers, 1],
           color='red', s=100, label='Rule-only', marker='^')
plt.scatter(enhanced_data[ml_outliers & ~rule_outliers, 0], enhanced_data[ml_outliers & ~rule_outliers, 1],
           color='orange', s=100, label='ML-only', marker='s')
plt.scatter(enhanced_data[rule_outliers & ml_outliers, 0], enhanced_data[rule_outliers & ml_outliers, 1],
           color='purple', s=100, label='Both Methods', marker='*')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Time (hours)')
plt.title('Combined Detection')
plt.legend()

plt.tight_layout()
plt.show()

### Scalability Considerations

For large datasets, computational efficiency becomes critical.

**Strategies**:
1. **Sampling**: Use representative subsets for training
2. **Online learning**: Process data streams incrementally
3. **Distributed computing**: Parallelize across multiple machines
4. **Approximate methods**: Trade accuracy for speed

**Code Implementation**:
```python
import time
from sklearn.utils import shuffle

def scalable_outlier_detection(data, method='sampling', sample_ratio=0.1, chunk_size=1000):
    """
    Scalable outlier detection for large datasets

    Parameters:
    data: array-like, large dataset
    method: str, scalability approach ('sampling', 'chunking', 'online')
    sample_ratio: float, ratio of data to sample
    chunk_size: int, size of chunks for processing

    Returns:
    outliers: boolean array, outlier predictions
    processing_time: float, time taken for detection
    """
    start_time = time.time()

    if method == 'sampling':
        # Sample subset for training, apply to full dataset
        n_samples = int(len(data) * sample_ratio)
        sample_indices = np.random.choice(len(data), n_samples, replace=False)
        sample_data = data[sample_indices]

        # Train on sample
        detector = IsolationForest(contamination=0.05, random_state=42)
        detector.fit(sample_data)

        # Apply to full dataset
        outlier_labels = detector.predict(data)
        outliers = (outlier_labels == -1)

    elif method == 'chunking':
        # Process data in chunks
        outliers = np.zeros(len(data), dtype=bool)

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]

            # Train detector on chunk
            detector = IsolationForest(contamination=0.05, random_state=42)
            chunk_labels = detector.fit_predict(chunk)

            # Store results
            outliers[i:i+chunk_size] = (chunk_labels == -1)

    elif method == 'online':
        # Simplified online detection using sliding window
        outliers = np.zeros(len(data), dtype=bool)
        window_size = min(chunk_size, 500)

        for i in range(window_size, len(data)):
            # Use sliding window for detection
            window_data = data[i-window_size:i]

            # Simple threshold-based detection on recent data
            current_point = data[i]
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)

            # Z-score based detection
            z_scores = np.abs((current_point - window_mean) / (window_std + 1e-8))
            outliers[i] = np.any(z_scores > 3)

    processing_time = time.time() - start_time
    return outliers, processing_time

# Example: Scalability comparison on large dataset
np.random.seed(42)

# Generate large synthetic dataset
n_large = 50000
large_normal_data = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_large)
large_outlier_data = np.random.uniform(-4, 4, (int(n_large * 0.02), 2))
large_outlier_data = large_outlier_data[np.linalg.norm(large_outlier_data, axis=1) > 2.5]

large_dataset = np.vstack([large_normal_data, large_outlier_data])
large_dataset = shuffle(large_dataset, random_state=42)

print(f"Large dataset size: {len(large_dataset):,} samples")

# Compare different scalability approaches
scalability_methods = ['sampling', 'chunking', 'online']
scalability_results = {}

for method in scalability_methods:
    print(f"\nTesting {method} approach...")

    outliers, proc_time = scalable_outlier_detection(
        large_dataset,
        method=method,
        sample_ratio=0.1,
        chunk_size=1000
    )

    scalability_results[method] = {
        'outliers_detected': np.sum(outliers),
        'processing_time': proc_time,
        'outlier_rate': np.sum(outliers) / len(large_dataset)
    }

    print(f"  Outliers detected: {np.sum(outliers):,}")
    print(f"  Processing time: {proc_time:.2f} seconds")
    print(f"  Outlier rate: {np.sum(outliers) / len(large_dataset):.3%}")

# Baseline: Standard detection on full dataset
print(f"\nBaseline: Standard Isolation Forest on full dataset...")
start_time = time.time()
baseline_detector = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
baseline_labels = baseline_detector.fit_predict(large_dataset)
baseline_time = time.time() - start_time
baseline_outliers = np.sum(baseline_labels == -1)

print(f"  Outliers detected: {baseline_outliers:,}")
print(f"  Processing time: {baseline_time:.2f} seconds")

# Summary comparison
print(f"\nScalability Comparison Summary:")
print(f"{'Method':<15} {'Outliers':<10} {'Time (s)':<10} {'Speedup':<10}")
print("-" * 50)

baseline_speedup = 1.0
print(f"{'Baseline':<15} {baseline_outliers:<10} {baseline_time:<10.2f} {baseline_speedup:<10.1f}x")

for method, results in scalability_results.items():
    speedup = baseline_time / results['processing_time']
    print(f"{method.capitalize():<15} {results['outliers_detected']:<10} {results['processing_time']:<10.2f} {speedup:<10.1f}x")

---

## Real-World Applications

### Fraud Detection in Financial Services

Complete implementation of a fraud detection system combining multiple techniques.

**Code Implementation**:
```python
class FraudDetectionSystem:
    """
    Comprehensive fraud detection system combining multiple methods
    """

    def __init__(self, contamination=0.001):
        self.contamination = contamination
        self.models = {}
        self.feature_names = None
        self.scalers = {}
        self.is_fitted = False

    def create_features(self, transactions_df):
        """Create features for fraud detection"""
        features = transactions_df.copy()

        # Time-based features
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)

        # Amount-based features
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_zscore'] = (features['amount'] - features['amount'].mean()) / features['amount'].std()

        # User-based features (requires historical data in practice)
        user_stats = features.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_transaction_count']
        features = features.merge(user_stats, on='user_id')

        # Merchant-based features
        merchant_stats = features.groupby('merchant_id')['amount'].agg(['mean', 'count']).reset_index()
        merchant_stats.columns = ['merchant_id', 'merchant_avg_amount', 'merchant_transaction_count']
        features = features.merge(merchant_stats, on='merchant_id')

        # Risk indicators
        features['amount_user_ratio'] = features['amount'] / (features['user_avg_amount'] + 1)
        features['unusual_time'] = ((features['is_night'] == 1) & (features['amount'] > features['user_avg_amount'])).astype(int)

        return features

    def apply_business_rules(self, features):
        """Apply business rules for fraud detection"""
        rule_flags = pd.DataFrame(index=features.index)

        # Rule 1: High amount transactions
        rule_flags['high_amount'] = (features['amount'] > 5000).astype(int)

        # Rule 2: Night transactions with high amounts
        rule_flags['night_high_amount'] = (
            (features['is_night'] == 1) &
            (features['amount'] > 2000)
        ).astype(int)

        # Rule 3: Amount significantly higher than user's average
        rule_flags['unusual_amount_for_user'] = (
            features['amount_user_ratio'] > 5
        ).astype(int)

        # Rule 4: Multiple high-risk indicators
        risk_score = (
            rule_flags['high_amount'] +
            rule_flags['night_high_amount'] +
            rule_flags['unusual_amount_for_user']
        )
        rule_flags['high_risk'] = (risk_score >= 2).astype(int)

        return rule_flags

    def fit(self, transactions_df):
        """Fit the fraud detection models"""
        # Create features
        features = self.create_features(transactions_df)

        # Select numerical features for ML models
        ml_features = [
            'amount', 'amount_log', 'amount_zscore', 'hour', 'day_of_week',
            'user_avg_amount', 'user_transaction_count', 'merchant_avg_amount',
            'amount_user_ratio', 'is_weekend', 'is_night'
        ]

        X = features[ml_features].fillna(0)
        self.feature_names = ml_features

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)

        # Fit multiple models
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.models['isolation_forest'].fit(X_scaled)

        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )
        self.models['lof'].fit(X_scaled)

        self.models['one_class_svm'] = OneClassSVM(
            nu=self.contamination,
            kernel='rbf'
        )
        self.models['one_class_svm'].fit(X_scaled)

        self.is_fitted = True

    def predict(self, transactions_df):
        """Predict fraud for new transactions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create features
        features = self.create_features(transactions_df)

        # Apply business rules
        rule_flags = self.apply_business_rules(features)

        # ML-based predictions
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scalers['standard'].transform(X)

        ml_predictions = {}
        for model_name, model in self.models.items():
            predictions = model.predict(X_scaled)
            ml_predictions[model_name] = (predictions == -1).astype(int)

        # Ensemble prediction (majority vote)
        ensemble_pred = (
            ml_predictions['isolation_forest'] +
            ml_predictions['lof'] +
            ml_predictions['one_class_svm']
        )
        ml_fraud = (ensemble_pred >= 2).astype(int)  # At least 2 models agree

        # Combine rule-based and ML-based predictions
        rule_fraud = (rule_flags['high_risk'] == 1).astype(int)
        combined_fraud = ((rule_fraud == 1) | (ml_fraud == 1)).astype(int)

        # Create results dataframe
        results = pd.DataFrame({
            'transaction_id': transactions_df.index,
            'rule_based_fraud': rule_fraud,
            'ml_based_fraud': ml_fraud,
            'combined_fraud': combined_fraud,
            'fraud_probability': ensemble_pred / 3,  # Normalized score
            'amount': transactions_df['amount'],
            'user_id': transactions_df['user_id']
        })

        return results

# Example usage of the fraud detection system
np.random.seed(42)

# Generate synthetic transaction data
n_transactions = 10000
transaction_data = {
    'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='15min'),
    'amount': np.random.exponential(100, n_transactions),
    'user_id': np.random.randint(1, 1000, n_transactions),
    'merchant_id': np.random.randint(1, 100, n_transactions)
}

# Add some fraudulent transactions
fraud_indices = np.random.choice(n_transactions, 50, replace=False)
for idx in fraud_indices:
    transaction_data['amount'][idx] = np.random.uniform(3000, 10000)  # High amounts
    # Some fraudulent transactions at night
    if np.random.random() > 0.5:
        hour = np.random.choice([1, 2, 3, 23])
        transaction_data['timestamp'][idx] = transaction_data['timestamp'][idx].replace(hour=hour)

transactions_df = pd.DataFrame(transaction_data)

print(f"Generated {len(transactions_df):,} transactions")
print(f"Date range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}")

# Initialize and train fraud detection system
fraud_detector = FraudDetectionSystem(contamination=0.005)
fraud_detector.fit(transactions_df)

# Predict fraud
fraud_results = fraud_detector.predict(transactions_df)

# Analyze results
print(f"\nFraud Detection Results:")
print(f"Rule-based detections: {fraud_results['rule_based_fraud'].sum():,}")
print(f"ML-based detections: {fraud_results['ml_based_fraud'].sum():,}")
print(f"Combined detections: {fraud_results['combined_fraud'].sum():,}")

# Show flagged transactions
flagged_transactions = fraud_results[fraud_results['combined_fraud'] == 1].head(10)
print(f"\nSample flagged transactions:")
print(flagged_transactions[['transaction_id', 'amount', 'fraud_probability', 'rule_based_fraud', 'ml_based_fraud']])

### Network Security Intrusion Detection

Implementation for detecting network intrusions and cyber attacks.

**Code Implementation**:
```python
class NetworkIntrusionDetector:
    """
    Network intrusion detection system using multiple outlier detection methods
    """

    def __init__(self, contamination=0.01):
        self.contamination = contamination
        self.models = {}
        self.feature_extractors = {}
        self.is_fitted = False

    def extract_network_features(self, network_logs):
        """Extract features from network traffic logs"""
        features = network_logs.copy()

        # Basic traffic features
        features['bytes_per_packet'] = features['total_bytes'] / (features['packet_count'] + 1)
        features['duration_log'] = np.log1p(features['duration'])
        features['bytes_log'] = np.log1p(features['total_bytes'])

        # Rate-based features
        features['bytes_per_second'] = features['total_bytes'] / (features['duration'] + 1)
        features['packets_per_second'] = features['packet_count'] / (features['duration'] + 1)

        # Port-based features
        features['is_common_port'] = features['dest_port'].isin([21, 22, 23, 25, 53, 80, 110, 443, 993, 995]).astype(int)
        features['is_high_port'] = (features['dest_port'] > 1024).astype(int)

        # Protocol features
        protocol_encoding = {'TCP': 1, 'UDP': 2, 'ICMP': 3}
        features['protocol_encoded'] = features['protocol'].map(protocol_encoding).fillna(0)

        # Time-based features
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['is_business_hours'] = ((features['hour'] >= 9) & (features['hour'] <= 17)).astype(int)

        # Connection patterns
        source_counts = features.groupby('source_ip').size().reset_index(name='source_connection_count')
        dest_counts = features.groupby('dest_ip').size().reset_index(name='dest_connection_count')

        features = features.merge(source_counts, on='source_ip', how='left')
        features = features.merge(dest_counts, on='dest_ip', how='left')

        return features

    def detect_anomalous_patterns(self, features):
        """Detect specific attack patterns"""
        patterns = pd.DataFrame(index=features.index)

        # DDoS pattern: High packet rate, short duration
        patterns['ddos_pattern'] = (
            (features['packets_per_second'] > features['packets_per_second'].quantile(0.95)) &
            (features['duration'] < features['duration'].quantile(0.1))
        ).astype(int)

        # Port scanning: Multiple connections to different ports from same source
        patterns['port_scan_pattern'] = (
            features['source_connection_count'] > features['source_connection_count'].quantile(0.99)
        ).astype(int)

        # Data exfiltration: Large data transfer outside business hours
        patterns['data_exfil_pattern'] = (
            (features['total_bytes'] > features['total_bytes'].quantile(0.95)) &
            (features['is_business_hours'] == 0)
        ).astype(int)

        # Unusual protocol usage
        patterns['unusual_protocol'] = (
            features['protocol_encoded'] == 3  # ICMP is less common
        ).astype(int)

        return patterns

    def fit(self, network_logs):
        """Train the intrusion detection models"""
        # Extract features
        features = self.extract_network_features(network_logs)

        # Select features for ML models
        ml_features = [
            'total_bytes', 'packet_count', 'duration', 'dest_port',
            'bytes_per_packet', 'bytes_per_second', 'packets_per_second',
            'duration_log', 'bytes_log', 'protocol_encoded', 'hour',
            'is_common_port', 'is_high_port', 'is_business_hours',
            'source_connection_count', 'dest_connection_count'
        ]

        X = features[ml_features].fillna(0)

        # Scale features
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train models
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.models['isolation_forest'].fit(X_scaled)

        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )
        self.models['lof'].fit(X_scaled)

        # DBSCAN for cluster-based detection
        from sklearn.cluster import DBSCAN
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        self.models['dbscan'].fit(X_scaled)

        self.feature_names = ml_features
        self.is_fitted = True

    def predict(self, network_logs):
        """Predict network intrusions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract features
        features = self.extract_network_features(network_logs)

        # Detect specific patterns
        patterns = self.detect_anomalous_patterns(features)

        # ML-based detection
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        iso_pred = self.models['isolation_forest'].predict(X_scaled)
        lof_pred = self.models['lof'].predict(X_scaled)

        # DBSCAN prediction (outliers are labeled as -1)
        dbscan_labels = self.models['dbscan'].fit_predict(X_scaled)
        dbscan_pred = (dbscan_labels == -1).astype(int) * 2 - 1  # Convert to -1/1 format

        # Ensemble prediction
        ml_ensemble = (
            (iso_pred == -1).astype(int) +
            (lof_pred == -1).astype(int) +
            (dbscan_pred == -1).astype(int)
        )

        # Pattern-based detection
        pattern_score = (
            patterns['ddos_pattern'] +
            patterns['port_scan_pattern'] +
            patterns['data_exfil_pattern'] +
            patterns['unusual_protocol']
        )

        # Combine detections
        intrusion_probability = (ml_ensemble / 3 + pattern_score / 4) / 2

        # Final prediction
        intrusion_detected = (
            (ml_ensemble >= 2) |  # At least 2 ML models agree
            (pattern_score >= 2)  # At least 2 pattern matches
        ).astype(int)

        # Create results
        results = pd.DataFrame({
            'connection_id': network_logs.index,
            'source_ip': network_logs['source_ip'],
            'dest_ip': network_logs['dest_ip'],
            'dest_port': network_logs['dest_port'],
            'intrusion_detected': intrusion_detected,
            'intrusion_probability': intrusion_probability,
            'ml_score': ml_ensemble,
            'pattern_score': pattern_score,
            'ddos_pattern': patterns['ddos_pattern'],
            'port_scan_pattern': patterns['port_scan_pattern'],
            'data_exfil_pattern': patterns['data_exfil_pattern']
        })

        return results

# Generate synthetic network traffic data
np.random.seed(42)

n_connections = 5000
network_data = {
    'timestamp': pd.date_range('2023-01-01', periods=n_connections, freq='1s'),
    'source_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_connections)],
    'dest_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_connections)],
    'dest_port': np.random.choice([21, 22, 23, 25, 53, 80, 443] + list(range(1024, 65536)),
                                 n_connections, p=[0.05]*7 + [0.65]),
    'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_connections, p=[0.7, 0.25, 0.05]),
    'total_bytes': np.random.exponential(1000, n_connections),
    'packet_count': np.random.poisson(10, n_connections),
    'duration': np.random.exponential(5, n_connections)
}

# Add attack traffic
attack_indices = np.random.choice(n_connections, 100, replace=False)

for idx in attack_indices:
    attack_type = np.random.choice(['ddos', 'port_scan', 'data_exfil'])

    if attack_type == 'ddos':
        # DDoS: High packet rate, short duration
        network_data['packet_count'][idx] = np.random.randint(100, 1000)
        network_data['duration'][idx] = np.random.uniform(0.1, 1.0)
        network_data['total_bytes'][idx] = network_data['packet_count'][idx] * np.random.randint(50, 200)

    elif attack_type == 'port_scan':
        # Port scanning: Multiple connections
        source_ip = f"192.168.1.{np.random.randint(200, 255)}"  # Suspicious source
        network_data['source_ip'][idx] = source_ip
        network_data['dest_port'][idx] = np.random.randint(1, 1024)  # Scanning low ports

    elif attack_type == 'data_exfil':
        # Data exfiltration: Large transfer outside business hours
        network_data['total_bytes'][idx] = np.random.uniform(50000, 500000)
        network_data['timestamp'][idx] = network_data['timestamp'][idx].replace(hour=np.random.choice([2, 3, 4]))

network_logs = pd.DataFrame(network_data)

print(f"Generated {len(network_logs):,} network connections")

# Train intrusion detection system
intrusion_detector = NetworkIntrusionDetector(contamination=0.02)
intrusion_detector.fit(network_logs)

# Detect intrusions
intrusion_results = intrusion_detector.predict(network_logs)

# Analyze results
print(f"\nIntrusion Detection Results:")
print(f"Total intrusions detected: {intrusion_results['intrusion_detected'].sum():,}")
print(f"DDoS patterns detected: {intrusion_results['ddos_pattern'].sum():,}")
print(f"Port scan patterns detected: {intrusion_results['port_scan_pattern'].sum():,}")
print(f"Data exfiltration patterns detected: {intrusion_results['data_exfil_pattern'].sum():,}")

# Show detected intrusions
detected_intrusions = intrusion_results[intrusion_results['intrusion_detected'] == 1].head(10)
print(f"\nSample detected intrusions:")
print(detected_intrusions[['source_ip', 'dest_ip', 'dest_port', 'intrusion_probability', 'ml_score', 'pattern_score']])

---

## Conclusion

This comprehensive guide has covered the major approaches to outlier detection, from basic statistical methods to advanced machine learning techniques. Each method has its strengths and is suited to different types of data and applications.

### Key Takeaways

**Method Selection Guidelines**:
- **Statistical methods** (Z-score, IQR): Best for univariate, normally distributed data
- **Distance-based methods** (LOF, k-NN): Effective for datasets with varying densities
- **Model-based approaches** (Isolation Forest, Autoencoders): Good for high-dimensional, complex patterns
- **Time series methods**: Essential for temporal data with trends and seasonality

**Implementation Best Practices**:
1. **Understand your data**: Distribution, dimensionality, and domain characteristics
2. **Combine multiple methods**: Ensemble approaches often perform better
3. **Incorporate domain knowledge**: Business rules and expert insight are invaluable
4. **Consider computational constraints**: Balance accuracy with scalability
5. **Evaluate thoroughly**: Use appropriate metrics and validation strategies

**Real-World Considerations**:
- **Imbalanced data**: Adjust thresholds and consider cost-sensitive approaches
- **High dimensionality**: Use dimensionality reduction or feature selection
- **Streaming data**: Implement online or sliding window approaches
- **Interpretability**: Ensure results can be explained to stakeholders

The choice of outlier detection method should always be driven by your specific use case, data characteristics, and business requirements. Start with simpler methods to establish baselines, then progress to more sophisticated approaches as needed.

Remember that outlier detection is often just the first step in a larger analytical workflow. The real value comes from understanding why outliers occur and taking appropriate action based on the insights gained.
````
