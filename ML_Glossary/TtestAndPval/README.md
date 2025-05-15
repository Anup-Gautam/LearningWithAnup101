# ML Glossary - T-test and P-value

This repository contains implementations and explanations of T-tests and P-values, fundamental statistical concepts used in hypothesis testing and data analysis.

## Ttest.py

This script demonstrates various types of T-tests and their applications, including:

- One-sample T-test
- Two-sample T-test (Independent)
- Paired T-test
- P-value calculations
- Effect size measurements
- Confidence intervals

## TtestAndPval.md

This markdown file contains detailed explanations of:

- T-test theory and assumptions
- Types of T-tests and when to use them
- P-value interpretation
- Null and alternative hypotheses
- Statistical significance
- Common pitfalls and best practices

## Setup Instructions

### 1. Create a Virtual Environment

```bash
# Navigate to the project directory
cd LearningWithAnup101/ML_Glossary

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Required Packages

```bash
# Install the required packages
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas
pip install scipy
pip install statsmodels
```

### 3. Run the Script

```bash
# Make sure you're in the TtestAndPval directory
cd TtestAndPval

# Run the T-test script
python Ttest.py
```

The script will generate visualizations and print statistical results to the console.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- statsmodels

## Output

### Ttest.py demonstrates:

1. One-sample T-test

   - Comparing sample mean to population mean
   - P-value calculation
   - Confidence intervals
   - Visualization of results

2. Two-sample T-test

   - Independent samples comparison
   - Equal/unequal variance assumptions
   - Effect size calculation
   - Results visualization

3. Paired T-test

   - Before/after comparisons
   - Dependent samples analysis
   - Difference distribution
   - Statistical significance

4. P-value Analysis
   - Interpretation guidelines
   - Common significance levels
   - Multiple testing considerations
   - Visualization of p-value distributions

## Example Usage

```python
# One-sample T-test
from scipy import stats
import numpy as np

# Sample data
sample = np.array([1, 2, 3, 4, 5])
population_mean = 3

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample, population_mean)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

## Additional Resources

- TtestAndPval.md contains detailed theoretical explanations
- Code comments provide implementation details
- References to statistical literature
- Best practices for hypothesis testing

## Common Statistical Terms

- **T-statistic**: Measures the difference between groups relative to the variation within groups
- **P-value**: Probability of obtaining results as extreme as the observed results
- **Significance Level (α)**: Threshold for rejecting the null hypothesis (commonly 0.05)
- **Effect Size**: Measure of the magnitude of the difference between groups
- **Confidence Interval**: Range of values likely to contain the true population parameter
