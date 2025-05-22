# Variance

## What is Variance?

**Definition**: Variance measures how far each number in a dataset is from the mean and from every other number in the set.

### Population Variance Formula

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

where,

- $x_i$: Each individual value
- $\mu$: Mean of the population
- $N$: Number of data points

### Sample Variance Formula

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Where:

- $x_i$: Each individual value
- $\bar{x}$: Sample mean
- $n$: Sample size

The sample formula uses **Bessel’s correction** to give an unbiased estimate of the population variance.

## Intuition Behind Variance

Example 1: [2, 4, 6, 8, 10]

- Mean = 6
- Squared differences = [16, 4, 0, 4, 16]
- Variance = 8

Example 2: [5, 5, 6, 6, 6]

- Mean = 5.6
- Squared differences = [0.36, 0.36, 0.16, 0.16, 0.16]
- Variance ≈ 0.24

## Properties of Variance

- **Always Non-Negative**: Square terms are never negative
- **Sensitive to Outliers**: One extreme value can skew the variance
- **Unit Issues**: Units are squared (e.g., kg²)
- **Assumes Symmetry**: Assumes symmetric data in some models

## Applications of Variance

- **Hypothesis Testing**: Used in t-tests, ANOVA
- **Bias-Variance Tradeoff**: Core concept in machine learning
- **Feature Selection**: Features with near-zero variance can be removed
- **Principal Component Analysis (PCA)**: Uses variance to find important features

## Variance in Machine Learning

### High Variance (Overfitting)

- Model memorizes training data
- Poor generalization to test data

Example: Deep decision tree fits all noise in the data

### Low Variance (Underfitting)

- Model too simple to capture patterns
- High bias, poor training performance

Example: Linear regression on a nonlinear dataset

### Optimal Error Balance

```
Total Error = Bias^2 + Variance + Irreducible Error
```

Goal is to minimize total error by balancing bias and variance.

## Real-World Analogy

**Dartboard Analogy**:

- **Low variance, high bias**: Darts clustered far from bullseye
- **High variance, low bias**: Darts spread around bullseye
- **Low variance, low bias**: Darts clustered at bullseye (ideal)

## Summary Table

| Aspect         | Low Variance   | High Variance        |
| -------------- | -------------- | -------------------- |
| Data Spread    | Tight          | Widely spread        |
| Model Behavior | Underfitting   | Overfitting          |
| Generalization | May be poor    | Often poor           |
| Fix            | Add complexity | Simplify, regularize |

---
