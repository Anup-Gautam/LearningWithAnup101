# Understanding Median

The **median** is the middle value in a dataset that has been arranged in ascending or descending order. It is a key measure of central tendency that splits the dataset into two equal halves.

---

## How to Calculate the Median

### 1. Odd Number of Values

**Formula:**

```bash
Median = ((n + 1) / 2)th value
```

> Note: This gives the **position** of the median in the sorted dataset, not the value itself.

**Example:**

```bash
Dataset: 3, 7, 9
Sorted: 3, 7, 9
n = 3
Position = (3 + 1) / 2 = 2
Median = 7
```

---

### 2. Even Number of Values

**Formula:**

```bash
Median = ( (n/2)th value + (n/2 + 1)th value ) / 2
```

**Example:**

```bash
Dataset: 5, 10, 15, 20
Sorted: 5, 10, 15, 20
n = 4
Median = (10 + 15) / 2 = 12.5
```

---

## Why Median Is Important

### 1. Outlier Resistance

The median is not affected by extremely large or small values.

**Example:**

```bash
Customer Purchase Amounts: 10, 15, 25, 30, 40, 1000
Mean = (10 + 15 + 25 + 30 + 40 + 1000) / 6 = 186.67
Median = (25 + 30) / 2 = 27.5
```

→ Median better represents a typical purchase compared to the inflated mean.

---

### 2. Outlier Detection

Values far from the median can be considered outliers.

**Example:**

```bash
Purchase Data: 10, 15, 25, 30, 40, 1000
Median = 27.5
1000 is far from 27.5 → Likely an outlier
```

---

### 3. Estimating Missing Values

Median can be used to impute missing values.

**Example:**

```bash
Age Data: 25, 30, 35, ?, 40, 45, 50
Known Values: 25, 30, 35, 40, 45, 50
Median = (35 + 40) / 2 = 37.5
```

→ Missing value can be estimated as 37.5

---

## When Not to Use Median

### 1. Symmetric / Normally Distributed Data

In perfectly symmetric distributions, mean and median are similar. Mean can offer more informative insights.

**Example:**

```bash
Test Scores: 70, 75, 80, 85, 90
Mean = 80
Median = 80
```

→ Both are fine, but mean may be preferred when calculating averages.

---

### 2. When Every Value Matters Equally

Use the mean if all data points are equally important, like in financial summaries.

**Example:**

```bash
Daily Sales: 1000, 1200, 1100, 5000, 1150
Mean = 1890
Median = 1150
```

→ Mean is better for revenue projections.

---

### 3. Weak to Small Changes

If only a few values change, the median may remain the same while the mean changes.

**Example:**

```bash
Original: 100, 101, 102, 103, 104
Mean = 102
Median = 102

Modified: 100, 101, 102, 103, 110
Mean = 103.2
Median = 102
```

→ Median stays the same even with a significant change in the last value.

---

## Summary

| Feature              | Median | Mean |
| -------------------- | ------ | ---- |
| Outlier Resistance   | Yes    | No   |
| Good for Skewed Data | Yes    | No   |
| Equal Weight to All  | No     | Yes  |
| Sensitive to Change  | Less   | More |
