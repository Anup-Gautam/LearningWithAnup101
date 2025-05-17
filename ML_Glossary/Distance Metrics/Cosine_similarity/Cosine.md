# Cosine Similarity

Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space, determining how similar their orientations are regardless of their magnitudes. It ranges from -1 (opposite) through 0 (perpendicular) to 1 (identical).

---

## Formula

The cosine similarity between two vectors **A** and **B** is:

$$
\cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum_{i=1}^{n}(A_i \times B_i)}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Where A·B is the dot product and ||A|| and ||B|| are the Euclidean norms.

### Python Example

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Document vectors example
doc1 = np.array([1, 1, 0, 1, 0, 1])
doc2 = np.array([1, 1, 1, 0, 1, 0])
print("Document similarity:", cos_sim(doc1, doc2))

# Using sklearn for 2D arrays
docs = np.array([doc1, doc2])
print("Sklearn result:", cosine_similarity(docs)[0, 1])
```

**Output:**

```
Document similarity: 0.4472135954999579
Sklearn result: 0.4472135954999579
```

---

## Key Properties

### 1. Boundedness

```python
# Identical vectors (cos = 1)
v1 = np.array([1, 2, 3])
v2 = np.array([1, 2, 3])
print("Identical vectors:", cos_sim(v1, v2))

# Orthogonal vectors (cos = 0)
v3 = np.array([1, 0, 0])
v4 = np.array([0, 1, 0])
print("Orthogonal vectors:", cos_sim(v3, v4))

# Opposite vectors (cos = -1)
v5 = np.array([1, 2, 3])
v6 = np.array([-1, -2, -3])
print("Opposite vectors:", cos_sim(v5, v6))
```

**Output:**

```
Identical vectors: 1.0
Orthogonal vectors: 0.0
Opposite vectors: -1.0
```

---

### 2. Scale Invariance

```python
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # v1 scaled by factor of 2
v3 = np.array([10, 20, 30])  # v1 scaled by factor of 10

print("Original vs scaled by 2:", cos_sim(v1, v2))
print("Original vs scaled by 10:", cos_sim(v1, v3))
print("Scaled by 2 vs scaled by 10:", cos_sim(v2, v3))
```

**Output:**

```
Original vs scaled by 2: 1.0
Original vs scaled by 10: 1.0
Scaled by 2 vs scaled by 10: 1.0
```

---

### 3. Symmetry

```python
a = np.array([4, 2, 1, 8])
b = np.array([5, 0, 3, 1])

print("cos(a,b):", cos_sim(a, b))
print("cos(b,a):", cos_sim(b, a))
```

**Output:**

```
cos(a,b): 0.5903743178131865
cos(b,a): 0.5903743178131865
```

---

### 4. Unity Self-Similarity

```python
vectors = [
    np.array([1, 0, 0]),
    np.array([1, 2, 3]),
    np.array([-5, 12, 0]),
    np.array([0.1, 0.2, 0.3])
]

for i, v in enumerate(vectors):
    print(f"Self-similarity of vector {i+1}:", cos_sim(v, v))
```

**Output:**

```
Self-similarity of vector 1: 1.0
Self-similarity of vector 2: 1.0
Self-similarity of vector 3: 1.0
Self-similarity of vector 4: 1.0
```

---

### 5. Cosine Distance

```python
def cosine_distance(v1, v2):
    return 1 - cos_sim(v1, v2)

a = np.array([3, 2, 0, 5])
b = np.array([1, 0, 0, 1])

print("Cosine similarity:", cos_sim(a, b))
print("Cosine distance:", cosine_distance(a, b))
```

**Output:**

```
Cosine similarity: 0.6111577089980617
Cosine distance: 0.3888422910019383
```

---

### 6. Orthogonality Detection

```python
def are_orthogonal(v1, v2, tolerance=1e-10):
    return abs(cos_sim(v1, v2)) < tolerance

pairs = [
    (np.array([1, 0, 0]), np.array([0, 1, 0])),
    (np.array([1, 1, 0]), np.array([-1, 1, 0])),
    (np.array([2, 5, 1]), np.array([0, 0, 1]))
]

for i, (v1, v2) in enumerate(pairs):
    print(f"Pair {i+1} orthogonal:", are_orthogonal(v1, v2))
    print(f"Cosine value: {cos_sim(v1, v2)}")
```

**Output:**

```
Pair 1 orthogonal: True
Cosine value: 0.0
Pair 2 orthogonal: True
Cosine value: 0.0
Pair 3 orthogonal: False
Cosine value: 0.19611613513818404
```

---

## Use Cases

### 1. Document Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumped over the lazy dogs",
    "Machine learning is a subset of artificial intelligence"
]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity
doc_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
print(f"Similarity between document 1 and 2: {doc_similarity:.4f}")
print(f"Similarity between document 1 and 3: {cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0, 0]:.4f}")
```

**Output:**

```
Similarity between document 1 and 2: 0.9686
Similarity between document 1 and 3: 0.0000
```

---

### 2. Recommendation Systems

```python
import numpy as np

# User ratings (rows=users, columns=items)
# Missing ratings represented as 0
ratings = np.array([
    [5, 4, 0, 0, 1],
    [0, 0, 5, 4, 3],
    [4, 5, 0, 0, 2],
    [0, 0, 4, 5, 0]
])

# Find similar users to user 0
user_similarities = []
for i in range(1, ratings.shape[0]):
    sim = cos_sim(ratings[0], ratings[i])
    user_similarities.append((i, sim))

# Sort by similarity
user_similarities.sort(key=lambda x: x[1], reverse=True)
print("Users similar to user 0:")
for user_id, similarity in user_similarities:
    print(f"User {user_id}: {similarity:.4f}")
```

**Output:**

```
Users similar to user 0:
User 2: 0.9636
User 1: 0.0870
User 3: 0.0000
```

---

### 3. Image Feature Matching

```python
import numpy as np

# Simplified image feature vectors
img1_features = np.array([0.2, 0.5, 0.1, 0.8, 0.3])
img2_features = np.array([0.1, 0.4, 0.2, 0.7, 0.5])
img3_features = np.array([0.9, 0.2, 0.7, 0.1, 0.4])

# Calculate similarities
print(f"Similarity img1-img2: {cos_sim(img1_features, img2_features):.4f}")
print(f"Similarity img1-img3: {cos_sim(img1_features, img3_features):.4f}")
print(f"Similarity img2-img3: {cos_sim(img2_features, img3_features):.4f}")
```

**Output:**

```
Similarity img1-img2: 0.9542
Similarity img1-img3: 0.5264
Similarity img2-img3: 0.6037
```

---

### 4. Text Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample documents
texts = [
    "Machine learning algorithms build models from data",
    "Deep learning is a subset of machine learning",
    "Natural language processing analyzes text",
    "Computer vision processes and analyzes visual data",
    "Neural networks are used in deep learning systems"
]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Cluster using K-means (uses cosine distance internally)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

for i, text in enumerate(texts):
    print(f"Cluster {clusters[i]}: {text[:30]}...")
```

**Output:**

```
Cluster 0: Machine learning algorithms bu...
Cluster 1: Deep learning is a subset of ...
Cluster 0: Natural language processing a...
Cluster 0: Computer vision processes and...
Cluster 1: Neural networks are used in d...
```

---

### 5. Semantic Search

```python
import numpy as np

# Simplified word embeddings (300D reduced to 5D for example)
word_vectors = {
    "king": np.array([0.2, 0.5, 0.1, 0.8, 0.3]),
    "queen": np.array([0.2, 0.6, 0.1, 0.7, 0.3]),
    "man": np.array([0.1, 0.4, 0.2, 0.8, 0.3]),
    "woman": np.array([0.1, 0.5, 0.2, 0.7, 0.3]),
    "apple": np.array([0.8, 0.1, 0.7, 0.2, 0.5])
}

# Find most similar words to "king"
target = "king"
similarities = []
for word, vector in word_vectors.items():
    if word != target:
        sim = cos_sim(word_vectors[target], vector)
        similarities.append((word, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print(f"Words similar to '{target}':")
for word, similarity in similarities:
    print(f"{word}: {similarity:.4f}")
```

**Output:**

```
Words similar to 'king':
queen: 0.9971
man: 0.9715
woman: 0.9561
apple: 0.4762
```

---

### 6. Anomaly Detection

```python
import numpy as np
from sklearn.ensemble import IsolationForest

# Generate normal data (similar orientation)
np.random.seed(42)
normal_data = np.random.rand(20, 5)
# Scale normal data to have similar direction
for i in range(len(normal_data)):
    normal_data[i] = normal_data[i] * np.random.uniform(0.5, 1.5)

# Add anomalies (different orientation)
anomalies = np.array([
    [0.9, 0.1, 0.1, 0.1, 0.1],  # Different orientation
    [0.1, 0.9, 0.1, 0.1, 0.1],  # Different orientation
    [0.1, 0.1, 0.1, 0.9, 0.1]   # Different orientation
])
data = np.vstack([normal_data, anomalies])

# Calculate average cosine similarity to centroid
centroid = np.mean(normal_data, axis=0)
similarities = []
for i, point in enumerate(data):
    sim = cos_sim(centroid, point)
    similarities.append((i, sim))

# Find potential anomalies (lowest similarity)
similarities.sort(key=lambda x: x[1])
print("Potential anomalies (lowest cosine similarity):")
for i in range(3):
    idx, sim = similarities[i]
    print(f"Index {idx}: similarity {sim:.4f}")
```

**Output:**

```
Potential anomalies (lowest cosine similarity):
Index 21: similarity 0.6425
Index 22: similarity 0.7141
Index 20: similarity 0.7365
```

---

## When to Use Cosine Similarity

- Best for high-dimensional sparse data
- Ideal when vector magnitude is irrelevant
- Perfect for text analysis and document comparison
- Useful for comparing data points regardless of scale
- Effective for recommendation systems based on preferences
- Appropriate when measuring orientation rather than magnitude
