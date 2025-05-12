# ML Glossary - Measures of Central Tendency

This repository contains visualizations and explanations of various machine learning concepts, starting with Measures of Central Tendency.

## Mean Visualizations

### PlottingMean.py

This script creates a comprehensive visualization showing different aspects of the mean, including:

- Why mean is important (minimizing squared error)
- Problems with outliers
- Non-symmetrical data analysis
- Categorical data handling
- Imbalanced dataset considerations

## Median Visualizations

### PlottingMedian.py

This script creates a comprehensive visualization showing different aspects of the median, including:

- Outlier resistance
- Symmetric distribution analysis
- Unequally important values
- Median stability with small changes

### Median.py

This script demonstrates practical examples of using the median in data analysis, including:

- Basic median calculations
- Handling outliers
- Comparing mean vs median
- Real-world applications

## Mode Visualizations

### PlottingMode.py

This script creates a comprehensive visualization showing different aspects of the mode, including:

- Categorical data analysis
- Discrete data handling
- Multiple modes (bimodal/multimodal distributions)
- Mode vs Mean vs Median comparison

### Mode.py

This script demonstrates practical examples of using the mode in data analysis, including:

- Basic mode calculations
- Handling categorical data
- Finding most frequent values
- Real-world applications

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
```

### 3. Run the Scripts

```bash
# Make sure you're in the Measures_of_Central_Tendency directory
cd Measures_of_Central_Tendency

# Run the Mean visualization
python PlottingMean.py

# Run the Median visualization
python PlottingMedian.py

# Run the Median examples
python Median.py

# Run the Mode visualization
python PlottingMode.py

# Run the Mode examples
python Mode.py
```

The scripts will generate visualizations saved as:

- 'mean_advantages_disadvantages.png'
- 'median_advantages_disadvantages.png'
- 'mode_advantages_disadvantages.png'

in the current directory and display the plots.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib
- seaborn
- pandas
- scipy

## Output

### Mean Visualization (PlottingMean.py)

The script generates a figure with 6 subplots:

1. Why Mean is important: Shows best value to minimize error
2. Problem with Outliers
3. Non-Symmetrical Data
4. Categorical Data
5. Imbalanced Dataset
6. Summary of when to use/not use Mean

The visualization will be saved as 'mean_advantages_disadvantages.png' in the current directory.

### Median Visualization (PlottingMedian.py)

The script generates a figure with 4 subplots:

1. Outlier Resistance: Shows how median is less affected by outliers
2. Symmetric Data: Demonstrates when mean equals median
3. All Values Matter: Shows when to use mean instead of median
4. Median Stability: Illustrates how median remains stable with small changes

The visualization will be saved as 'median_advantages_disadvantages.png' in the current directory.

### Median Examples (Median.py)

This script provides:

- Interactive examples of median calculations
- Comparison with mean in different scenarios
- Practical applications and use cases

The visualizations will be saved as PNG files in the current directory.

### Mode Visualization (PlottingMode.py)

The script generates a figure with 4 subplots:

1. Categorical Data: Shows how mode is the best measure for categorical data
2. Discrete Data: Demonstrates mode's effectiveness with discrete values
3. Multiple Modes: Illustrates bimodal and multimodal distributions
4. Comparison: Shows when to use mode vs mean vs median

The visualization will be saved as 'mode_advantages_disadvantages.png' in the current directory.

### Mode Examples (Mode.py)

This script provides:

- Interactive examples of mode calculations
- Handling categorical and discrete data
- Finding most frequent values in datasets
- Practical applications and use cases

The visualizations will be saved as PNG files in the current directory.
