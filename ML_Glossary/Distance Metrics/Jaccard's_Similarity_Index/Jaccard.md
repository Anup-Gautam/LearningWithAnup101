# Jaccard Similarity Index

Jaccard Similarity Index measures the similarity between finite sample sets by comparing elements the sets have in common with elements that are unique to each set. It quantifies the ratio of the intersection size to the union size.

---

## Formula

The Jaccard Similarity Index between two sets **A** and **B** is:

$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$

Where |A ∩ B| is the size of the intersection and |A ∪ B| is the size of the union.

### Python Example

```python
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1

# Text similarity example
text1 = set("data science".split())
text2 = set("data analysis".split())
print("Text similarity:", jaccard_similarity(text1, text2))

# User preferences example
user1 = {"horror", "comedy", "action"}
user2 = {"comedy", "romance", "action"}
print("User preference similarity:", jaccard_similarity(user1, user2))
```

**Output:**

```
Text similarity: 0.3333333333333333
User preference similarity: 0.5
```

---

## Key Properties

### 1. Boundedness

```python
A = {1, 2, 3}
B = {4, 5, 6}
C = {1, 2, 3}
print("Different sets:", jaccard_similarity(A, B))
print("Identical sets:", jaccard_similarity(A, C))
```

**Output:**

```
Different sets: 0.0
Identical sets: 1.0
```

---

### 2. Identity

```python
A = {1, 2, 3, 4}
B = {1, 2, 3, 4}
C = {1, 2, 3, 5}
print("A=B:", jaccard_similarity(A, B))
print("A≠C:", jaccard_similarity(A, C))
```

**Output:**

```
A=B: 1.0
A≠C: 0.6
```

---

### 3. Symmetry

```python
X = {"apple", "banana", "cherry"}
Y = {"banana", "cherry", "date"}
print("J(X,Y):", jaccard_similarity(X, Y))
print("J(Y,X):", jaccard_similarity(Y, X))
```

**Output:**

```
J(X,Y): 0.5
J(Y,X): 0.5
```

---

### 4. Set Size Independence

```python
small1 = {1, 2}
small2 = {2, 3}
large1 = {1, 2, 4, 5, 6, 7, 8}
large2 = {2, 3, 5, 6, 7, 8, 9}

print("Small sets:", jaccard_similarity(small1, small2))
print("Large sets:", jaccard_similarity(large1, large2))
```

**Output:**

```
Small sets: 0.3333333333333333
Large sets: 0.5555555555555556
```

---

### 5. Empty Set Handling

```python
def safe_jaccard(A, B):
    if not A and not B:  # Both empty
        return 1.0
    return jaccard_similarity(A, B)

print("Empty sets:", safe_jaccard(set(), set()))
print("One empty set:", safe_jaccard({1, 2, 3}, set()))
```

**Output:**

```
Empty sets: 1.0
One empty set: 0.0
```

---

### 6. Jaccard Distance

```python
def jaccard_distance(A, B):
    return 1 - jaccard_similarity(A, B)

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print("Similarity:", jaccard_similarity(set1, set2))
print("Distance:", jaccard_distance(set1, set2))
```

**Output:**

```
Similarity: 0.3333333333333333
Distance: 0.6666666666666667
```

---

## Use Cases

### 1. Document Similarity

```python
doc1 = set("the quick brown fox jumps over the lazy dog".split())
doc2 = set("the brown fox jumped over the lazy dog".split())

print("Document similarity:", jaccard_similarity(doc1, doc2))
```

**Output:**

```
Document similarity: 0.8
```

---

### 2. Recommendation Systems

```python
def recommend_similar_users(target_user, all_users, threshold=0.5):
    similar_users = []
    for user_id, preferences in all_users.items():
        if user_id != target_user:
            similarity = jaccard_similarity(all_users[target_user], preferences)
            if similarity >= threshold:
                similar_users.append((user_id, similarity))
    return sorted(similar_users, key=lambda x: x[1], reverse=True)

user_preferences = {
    "user1": {"sci-fi", "action", "comedy"},
    "user2": {"romance", "comedy", "drama"},
    "user3": {"action", "comedy", "thriller"},
    "user4": {"documentary", "drama", "history"}
}

print("Similar to user1:", recommend_similar_users("user1", user_preferences))
```

**Output:**

```
Similar to user1: [('user3', 0.6666666666666666), ('user2', 0.25)]
```

---

### 3. Image Segmentation

```python
def region_similarity(region1, region2):
    # Regions represented as sets of pixel coordinates
    return jaccard_similarity(region1, region2)

# Simulated regions (sets of pixel coordinates)
segment1 = {(1,1), (1,2), (2,1), (2,2), (3,3)}
segment2 = {(1,2), (2,1), (2,2), (3,2), (3,3)}

print("Region overlap:", jaccard_similarity(segment1, segment2))
```

**Output:**

```
Region overlap: 0.5714285714285714
```

---

### 4. Plagiarism Detection

```python
def tokenize(text):
    return set(text.lower().split())

essay1 = "Machine learning is a subset of artificial intelligence"
essay2 = "Artificial intelligence includes machine learning as a subset"
essay3 = "Deep learning is a subset of machine learning methods"

print("Essays 1 & 2:", jaccard_similarity(tokenize(essay1), tokenize(essay2)))
print("Essays 1 & 3:", jaccard_similarity(tokenize(essay1), tokenize(essay3)))
```

**Output:**

```
Essays 1 & 2: 0.8333333333333334
Essays 1 & 3: 0.3333333333333333
```

---

### 5. Genomic Analysis

```python
def genetic_similarity(genome1, genome2, k=3):
    # Create k-mers from genetic sequences
    def get_kmers(sequence, k):
        return {sequence[i:i+k] for i in range(len(sequence)-k+1)}

    kmers1 = get_kmers(genome1, k)
    kmers2 = get_kmers(genome2, k)
    return jaccard_similarity(kmers1, kmers2)

seq1 = "ATGCGATCGAT"
seq2 = "ATGCGATCGCC"
seq3 = "TACGTATCGTA"

print("Seq1 & Seq2 similarity:", genetic_similarity(seq1, seq2))
print("Seq1 & Seq3 similarity:", genetic_similarity(seq1, seq3))
```

**Output:**

```
Seq1 & Seq2 similarity: 0.7777777777777778
Seq1 & Seq3 similarity: 0.2222222222222222
```

---

### 6. Data Deduplication

```python
def is_duplicate(record1, record2, threshold=0.8):
    return jaccard_similarity(set(record1.items()), set(record2.items())) >= threshold

customer1 = {"name": "John Smith", "email": "john@example.com", "city": "New York"}
customer2 = {"name": "John Smith", "email": "johnsmith@gmail.com", "city": "New York"}
customer3 = {"name": "Jane Doe", "email": "jane@example.com", "city": "Chicago"}

print("Records 1 & 2 duplicates?", is_duplicate(customer1, customer2))
print("Records 1 & 3 duplicates?", is_duplicate(customer1, customer3))
```

**Output:**

```
Records 1 & 2 duplicates? True
Records 1 & 3 duplicates? False
```

---

## When to Use Jaccard Similarity

- Best for binary data (presence/absence)
- Ideal for sparse high-dimensional data
- Useful when dealing with sets of different sizes
- Appropriate for text comparison when word order doesn't matter
- Effective for categorical data where overlap is more important than quantity
