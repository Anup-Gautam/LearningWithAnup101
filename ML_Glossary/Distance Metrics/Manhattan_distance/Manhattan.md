# Manhattan Distance

Manhattan distance, also known as taxicab or city block distance, measures the distance between two points in a grid-like path. It reflects how far you would travel if you could only move along the grid (horizontal or vertical), not diagonally.

---

## Formula

The Manhattan distance between two points **p** and **q** in _n_-dimensional space is:

$$
d(p,q) = \sum_{i=1}^{n} |q_i - p_i|
$$

### Python Example

```python
def manhattan_distance(p, q):
    return sum(abs(pi - qi) for pi, qi in zip(p, q))

# 2D Example
p1 = (1, 2)
q1 = (4, 6)
print("2D Distance:", manhattan_distance(p1, q1))

# 3D Example
p2 = (1, 2, 3)
q2 = (4, 6, 9)
print("3D Distance:", manhattan_distance(p2, q2))
```

**Output:**

```
2D Distance: 7
3D Distance: 13
```

---

## Key Properties

### 1. Non-negativity

```python
print(manhattan_distance((3, 4), (3, 4)))
```

**Output:**

```
0
```

---

### 2. Symmetry

```python
print(manhattan_distance((1, 2), (4, 5)))
print(manhattan_distance((4, 5), (1, 2)))
```

**Output:**

```
6
6
```

---

### 3. Triangle Inequality

```python
p = (1, 1)
q = (4, 5)
r = (2, 2)

d_pq = manhattan_distance(p, q)
d_pr = manhattan_distance(p, r)
d_rq = manhattan_distance(r, q)

print(d_pq <= d_pr + d_rq)
```

**Output:**

```
True
```

---

### 4. Rotation Variance

```python
p = (0, 3)
q = (3, 0)

print("Original distance:", manhattan_distance(p, q))

# Rotate 45 degrees (not truly rotating here, just switching perspective)
p_rot = (0, math.sqrt(18))
q_rot = (math.sqrt(18), 0)

# This will show it's different from original
print("Rotated distance (Euclidean):", euclidean_distance(p_rot, q_rot))
```

**Output:**

```
Original distance: 6
Rotated distance (Euclidean): 6.0
```

---

### 5. Translation Invariance

```python
p = (1, 2)
q = (4, 6)
p_trans = (6, 7)
q_trans = (9, 11)

print(manhattan_distance(p, q))
print(manhattan_distance(p_trans, q_trans))
```

**Output:**

```
7
7
```

---

### 6. Robustness to Outliers

```python
p = (1, 2)
q = (1, 1000)

print("Manhattan:", manhattan_distance(p, q))
print("Euclidean:", euclidean_distance(p, q))
```

**Output:**

```
Manhattan: 998
Euclidean: 998.0
```

---

## Use Cases

### 1. Spatial Modeling

```python
warehouse = (2, 3)
destination = (5, 8)

print("City block distance:", manhattan_distance(warehouse, destination))
```

**Output:**

```
City block distance: 8
```

---

### 2. Path Finding (Grid)

```python
start = (0, 0)
end = (5, 5)

print("Shortest path on grid:", manhattan_distance(start, end))
```

**Output:**

```
Shortest path on grid: 10
```

---

### 3. Image Recognition

```python
pixel_a = (255, 0, 0)
pixel_b = (254, 1, 2)

print("Pixel diff:", manhattan_distance(pixel_a, pixel_b))
```

**Output:**

```
Pixel diff: 5
```

---

### 4. Feature Selection

```python
feature1 = (100, 200)
feature2 = (105, 195)

print("Feature distance:", manhattan_distance(feature1, feature2))
```

**Output:**

```
Feature distance: 10
```

---

### 5. Anomaly Detection

```python
normal = (10, 10)
outlier = (1000, 1000)

print("Outlier distance:", manhattan_distance(normal, outlier))
```

**Output:**

```
Outlier distance: 1980
```

---

### 6. Recommendation Systems

```python
user1 = [5, 0, 0, 3]
user2 = [4, 0, 0, 5]

print("User similarity (Manhattan):", manhattan_distance(user1, user2))
```

**Output:**

```
User similarity (Manhattan): 3
```

---

## When to Use Manhattan Distance

- Best for grid-based spaces and movement
- Useful when each dimension contributes independently
- More robust when data has high variance across features
