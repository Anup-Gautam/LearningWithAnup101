# ML Glossary - Measures of Central Tendency

This repository contains visualizations and explanations of various machine learning concepts, starting with Measures of Central Tendency.

## PlottingMean.py

This script creates a comprehensive visualization showing different aspects of the mean, including:

- Why mean is important (minimizing squared error)
- Problems with outliers
- Non-symmetrical data analysis
- Categorical data handling
- Imbalanced dataset considerations

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

### 3. Run the Script

```bash
# Make sure you're in the Measures_of_Central_Tendency directory
cd Measures_of_Central_Tendency

# Run the script
python PlottingMean.py
```

The script will generate a visualization saved as 'mean_advantages_disadvantages.png' in the current directory and display the plot.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib
- seaborn
- pandas
- scipy

## Output

The script generates a figure with 6 subplots:

1. Why Mean is important: Shows best value to minimize error
2. Problem with Outliers
3. Non-Symmetrical Data
4. Categorical Data
5. Imbalanced Dataset
6. Summary of when to use/not use Mean

The visualization will be saved as 'mean_advantages_disadvantages.png' in the current directory.
