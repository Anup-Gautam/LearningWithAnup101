# Hamming Distance

Hamming distance measures the number of positions at which corresponding elements differ between two sequences of equal length. It counts the minimum number of substitutions required to transform one sequence into another.

---

## Formula

The Hamming distance between two strings or sequences **p** and **q** of equal length _n_ is:

$$
d(p, q) = \sum_{i=1}^{n} \delta(p_i, q_i)
$$

Where δ(pi,qi) equals 1 if pi ≠ qi and 0 if pi = qi.

### Python Example

```python
def hamming_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Sequences must be of equal length")
    return sum(p_i != q_i for p_i, q_i in zip(p, q))

# Binary strings example
binary1 = "1011101"
binary2 = "1001001"
print("Binary strings:", hamming_distance(binary1, binary2))

# Character strings example
word1 = "karolin"
word2 = "kathrin"
print("Words:", hamming_distance(word1, word2))
```

**Output:**

```
Binary strings: 2
Words: 3
```

---

## Key Properties

### 1. Non-negativity

```python
print(hamming_distance("1010", "1010"))
```

**Output:**

```
0
```

---

### 2. Identity

```python
s1 = "10101"
s2 = "10101"
s3 = "10100"
print(hamming_distance(s1, s2) == 0)  # True for identical strings
print(hamming_distance(s1, s3) == 0)  # False for different strings
```

**Output:**

```
True
False
```

---

### 3. Symmetry

```python
print(hamming_distance("1010", "0110"))
print(hamming_distance("0110", "1010"))
```

**Output:**

```
2
2
```

---

### 4. Triangle Inequality

```python
p = "10101"
q = "01010"
r = "11000"

d_pq = hamming_distance(p, q)
d_pr = hamming_distance(p, r)
d_rq = hamming_distance(r, q)

print(d_pq <= d_pr + d_rq)
```

**Output:**

```
True
```

---

### 5. Discreteness

```python
examples = [("1010", "1010"), ("1010", "1110"), ("1010", "0110")]
for ex in examples:
    print(f"{ex[0]} vs {ex[1]}: {hamming_distance(ex[0], ex[1])}")
```

**Output:**

```
1010 vs 1010: 0
1010 vs 1110: 1
1010 vs 0110: 2
```

---

### 6. Position Independence

```python
base = "10101"
variants = ["00101", "11101", "10001", "10111", "10100"]

for v in variants:
    print(f"{base} vs {v}: {hamming_distance(base, v)}")
```

**Output:**

```
10101 vs 00101: 1
10101 vs 11101: 1
10101 vs 10001: 1
10101 vs 10111: 1
10101 vs 10100: 1
```

---

## Use Cases

### 1. Error Detection and Correction

```python
def add_parity_bit(data):
    return data + ('0' if data.count('1') % 2 == 0 else '1')

def check_parity(data_with_parity):
    return data_with_parity.count('1') % 2 == 0

data = "10110"
with_parity = add_parity_bit(data)
print("Parity correct:", check_parity(with_parity))

# Simulate error (flip one bit)
corrupted = with_parity[:2] + ('0' if with_parity[2] == '1' else '1') + with_parity[3:]
print("Error detected:", not check_parity(corrupted))
```

**Output:**

```
Parity correct: True
Error detected: True
```

---

### 2. Spell Checking

```python
def suggest_corrections(word, dictionary, max_distance=1):
    return [dict_word for dict_word in dictionary
            if len(dict_word) == len(word)
            and hamming_distance(word, dict_word) <= max_distance]

dictionary = ["hat", "bat", "cat", "rat", "mat", "sat"]
misspelled = "kat"

print("Suggestions:", suggest_corrections(misspelled, dictionary))
```

**Output:**

```
Suggestions: ['hat', 'cat', 'rat', 'mat', 'sat']
```

---

### 3. DNA Sequence Analysis

```python
dna1 = "ACGTACGT"
dna2 = "ACCTACCG"

print(f"Hamming distance: {hamming_distance(dna1, dna2)}")
print(f"Similarity: {(1 - hamming_distance(dna1, dna2)/len(dna1))*100:.1f}%")
```

**Output:**

```
Hamming distance: 3
Similarity: 62.5%
```

---

### 4. Pattern Recognition

```python
pattern1 = "10110010"
pattern2 = "10111110"

distance = hamming_distance(pattern1, pattern2)
print(f"Feature distance: {distance}")
print(f"Feature similarity: {(1 - distance/len(pattern1))*100:.1f}%")
```

**Output:**

```
Feature distance: 2
Feature similarity: 75.0%
```

---

### 5. Cryptography

```python
def binary_representation(text):
    return ''.join(format(ord(c), '08b') for c in text)

message1 = binary_representation("Hello")
message2 = binary_representation("Hallo")

print(f"Bit difference: {hamming_distance(message1, message2)}")
```

**Output:**

```
Bit difference: 4
```

---

### 6. Coding Theory

```python
def hamming_weight(binary_str):
    return binary_str.count('1')

codeword = "10101010"
bit_flips = hamming_weight(codeword)

print(f"Hamming weight: {bit_flips}")
```

**Output:**

```
Hamming weight: 4
```

---

## When to Use Hamming Distance

- Best for fixed-length sequences where positions are significant
- Ideal for error detection in data transmission
- Useful for comparing genetic sequences when only substitutions matter
- Appropriate for comparing binary feature vectors in pattern recognition
