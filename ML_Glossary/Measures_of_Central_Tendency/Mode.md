# Mode in Machine Learning

## What is Mode?

- **Mode** is the value that occurs most frequently in a dataset.
- A dataset can have:
  - One mode (**unimodal**)
  - Multiple modes (**multimodal**)

---

## Why is Mode Important?

### 1. Handling Categorical Data

```bash
# Example: Filling missing values in a categorical column
colors = ["blue", "red", "blue", "green", None]
# Mode is "blue" -> Fill None with "blue"
filled_colors = ["blue", "red", "blue", "green", "blue"]
```

### 2. Classification Problems

```bash
# Example: Predicting the majority class
labels = ["spam", "not spam", "spam", "spam", "not spam"]
# Mode is "spam" -> Baseline classifier predicts "spam"
```

### 3. Detecting Repeated Behaviors

```bash
# Example: Most frequent product bought
purchases = ["apple", "banana", "apple", "apple", "orange"]
# "apple" is the most common item
```

---

## When Not to Use Mode

### 1. Data with High Cardinality

```bash
# Example: Temperatures where each value is unique
temps = [72.3, 72.4, 72.5, 72.6]
# Mode doesn’t exist or is meaningless
```

### 2. Multimodal Distribution

```bash
# Example: Scores with multiple peaks
scores = [70, 80, 80, 90, 90]
# Both 80 and 90 are modes -> Ambiguous result
```

### 3. Summarizing Numerical Columns

```bash
# Example: Summarizing ages
ages = [22, 25, 25, 30, 30, 30, 45]
# Mean or median is more representative than mode (30)
```

### 4. Where Minority Class is Also Important

```bash
# Example: Fraud detection
transactions = ["legit"] * 95 + ["fraud"] * 5
# Mode is "legit" -> Ignores rare but critical "fraud" class
```

---

## Final Note

It might seem like **mode** is pretty useless, but just understand that it is **highly situational, not useless**.  
You will see frequency-based patterns many times during your journey with ML.
