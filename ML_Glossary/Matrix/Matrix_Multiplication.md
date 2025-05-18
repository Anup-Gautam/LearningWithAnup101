# Matrix Multiplication:

Matrix multiplication is the process of combining two matrices where the number of columns in the first matrix must equal the number of rows in the second. The result is a new matrix that represents a composition of transformations or operations.

### The Rule

If

- Matrix **A** is of size `m × n`
- Matrix **B** is of size `n × p`  
  Then,
- The result **C = A × B** is of size `m × p`

Each element `C[i][j]` is the dot product of the `i-th row of A` and `j-th column of B`.

## Why It Matters in Machine Learning

Matrix multiplication is fundamental in how data flows through a machine learning model. It’s used for predictions, transformations, and learning.

### Scenario 1: Linear Regression

Given:

- Input features: `X` = `[age, income]`, shape: `n × 2`
- Weights: `W` = `[w1, w2]`, shape: `2 × 1`
- Prediction: `Y = X × W`, shape: `n × 1`

Each multiplication combines features with weights to produce a predicted output.

### Scenario 2: Neural Networks

In a hidden layer:

- Inputs: `X`, shape `[n × d_in]`
- Weights: `W`, shape `[d_in × d_out]`
- Output: `X × W`, shape `[n × d_out]`

Matrix multiplication projects data into new spaces, helping the model learn complex representations.

### Scenario 3: Batch Training

Matrix multiplication supports batch processing for efficiency:

- Batch of inputs: `X`, shape `[batch_size × input_dim]`
- Weights: `W`, shape `[input_dim × output_dim]`
- Result: `X × W`, shape `[batch_size × output_dim]`

This enables parallel computation for training.

### Scenario 4: Gradient Backpropagation

During training, matrix multiplication is used to compute gradients for updating weights. Backpropagation chains together matrix multiplications to determine how changes in weights affect the final output.

## Summary in a Single Flow

```
Input Data (Matrix) → × Weight Matrix → + Bias → Activation → Next Layer
```

Each layer's forward pass is a matrix multiplication. The backward pass also relies on matrix multiplication.
