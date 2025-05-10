# Understanding Measures of Central Tendency

- It's a single value that attempts to describe a set of data by identifying the central position of the data
- In this course we will be talking specifically about 3 measures:
  - Mean
  - Median
  - Mode

## Mean

- Sum of all values divided by the number of values in a data set
- Mean(x̄) = (x₁ + x₂ + x₃ + ... + xₙ)/n
- where x₁, x₂, x₃ … xₙ are the values and n is the total number of values

### What does Mean denote in a dataset?

- It is considered to be the value that is most common
- It is essentially a model of your dataset. Think of it as a summary of your data
- This doesn't mean that the mean value exists in your dataset
- **Important**: It gives you the lowest prediction error in the dataset.

### Example:

Dataset = [2, 4, 6, 8]
Mean = (2 + 4 + 6 + 8)/4 = 20/4 = 5

- Let's say you want a number to represent the whole set. Let's say we have 3 guesses.
- We measure error by (actual - guess)² { We will learn more about errors in later parts}

For 4:

- (2 - 4)² = 4
- (4 - 4)² = 0
- (6 - 4)² = 4
- (8 - 4)² = 16
- Squared Error = 4 + 0 + 4 + 16 = 24

For 5:

- (2 - 5)² = 9
- (4 - 5)² = 1
- (6 - 5)² = 1
- (8 - 5)² = 9
- Squared Error = 9 + 1 + 1 + 9 = 20

For 6:

- (2 - 6)² = 16
- (4 - 6)² = 4
- (6 - 6)² = 0
- (8 - 6)² = 4
- Squared Error = 16 + 4 + 0 + 4 = 24

As you can see, 5 gives the lowest total squared error.

### Why is Mean important?

- It shows the best single value to use when you want to minimize how wrong the guess is
- It is most commonly used for regression models and predictive analysis

### When should we not use Mean?

- When the data set has outliers.
  - Example: Data = 10, 12, 11, 1000
  - Mean = 258.25 - this is not representative at all
- When the data is not symmetrical
  - Example: Income, Housing Prices
- When working with categorical data
  - Example: Can't calculate mean for data like "red", "blue" or "cat", "dog"
- When working with a highly imbalanced dataset
  - Example: In fraud detection, 99% are normal and 1% fraud - mean prediction might always say normal
