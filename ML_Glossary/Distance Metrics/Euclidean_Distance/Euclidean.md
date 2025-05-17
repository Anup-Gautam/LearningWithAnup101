# Euclidean Distance

Euclidean distance is the straight-line distance between two points in Euclidean space. It is the most intuitive metric for measuring how far apart two points are. Below, we explain the concept using examples and Python code.

---

## Formula

The Euclidean distance between two points **p** and **q** in _n_-dimensional space is calculated as:

$$
d(p,q) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2 + \dots + (q_n - p_n)^2}
$$

Or, more compactly:

\[
d(p,q) = \sqrt{\sum\_{i=1}^{n} (q_i - p_i)^2}
\]

### Python Example

```python
import math

def euclidean_distance(p, q):
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

# 2D Example
p1 = (1, 2)
q1 = (4, 6)
print("2D Distance:", euclidean_distance(p1, q1))

# 3D Example
p2 = (1, 2, 3)
q2 = (4, 6, 9)
print("3D Distance:", euclidean_distance(p2, q2))
```

**Output:**

```
2D Distance: 5.0
3D Distance: 8.366600265340756
```

---

## Key Properties

### 1. Non-negativity

```python
print(euclidean_distance((2, 3), (2, 3)))
```

**Output:**

```
0.0
```

---

### 2. Symmetry

```python
print(euclidean_distance((1, 2), (3, 4)))
print(euclidean_distance((3, 4), (1, 2)))
```

**Output:**

```
2.8284271247461903
2.8284271247461903
```

---

### 3. Triangle Inequality

```python
p = (1, 1)
q = (4, 5)
r = (2, 2)

d_pq = euclidean_distance(p, q)
d_pr = euclidean_distance(p, r)
d_rq = euclidean_distance(r, q)

print(d_pq <= d_pr + d_rq)
```

**Output:**

```
True
```

---

### 4. Rotation and Translation Invariance

```python
p = (1, 2)
q = (4, 6)
p_translated = (11, 12)
q_translated = (14, 16)

print(euclidean_distance(p, q))
print(euclidean_distance(p_translated, q_translated))
```

**Output:**

```
5.0
5.0
```

---

### 5. Scale Sensitivity

```python
p_scaled = (1, 1000)
q_scaled = (4, 1000)

print("Without scaling:", euclidean_distance(p_scaled, q_scaled))
```

**Output:**

```
Without scaling: 3.0
```

---

## Use Cases

### 1. K-Nearest Neighbors (KNN)

```python
data = [(1, 2), (4, 5), (7, 8)]
query = (2, 3)

neighbors = sorted(data, key=lambda point: euclidean_distance(point, query))
print("Nearest Neighbor:", neighbors[0])
```

**Output:**

```
Nearest Neighbor: (1, 2)
```

---

### 2. K-Means Clustering

```python
centroids = [(2, 2), (8, 8)]
point = (3, 4)

closest = min(centroids, key=lambda c: euclidean_distance(c, point))
print("Assigned to centroid:", closest)
```

**Output:**

```
Assigned to centroid: (2, 2)
```

---

### 3. Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print("PCA Distance:", euclidean_distance(X_transformed[0], X_transformed[1]))
```

**Output:**

```
PCA Distance: 0.29669331963716057
```

---

### 4. Anomaly Detection

```python
dataset = [(1, 1), (1.1, 1.2), (100, 100)]

reference = (1, 1)
distances = [euclidean_distance(reference, point) for point in dataset]
print("Distances:", distances)
```

**Output:**

```
Distances: [0.0, 0.22360679774997916, 140.0071426749364]
```

---

### 5. Recommendation Systems

```python
user1 = [5, 3, 0, 1]
user2 = [4, 0, 0, 1]

print("User similarity:", euclidean_distance(user1, user2))
```

**Output:**

```
User similarity: 3.1622776601683795
```

---

## When to Use Euclidean Distance

- When features are normalized or scaled similarly
- When clusters are compact and spherical
- When dimensions are relatively low (avoiding the curse of dimensionality)

Avoid using Euclidean distance when:

- Features vary in scale
- There are too many dimensions (e.g., >50) without dimensionality reduction
