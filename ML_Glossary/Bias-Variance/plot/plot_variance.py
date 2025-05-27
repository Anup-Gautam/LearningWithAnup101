import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("../models/data.csv")
X_simple = df[["SquareFootage"]]
X_full = df[["SquareFootage", "Bedrooms", "Age"]]
y = df["Price"]

model_names = [
    "simple_model.json",
    "complex_model.json",
    "simple_model_fixed.json",
    "complex_model_fixed.json"
]

models = {}
for file in model_names:
    with open(os.path.join("../results", file), "r") as f:
        data = json.load(f)
        models[data["model"]] = data["mse"]

# --- Plots ---
plt.figure(figsize=(14, 6))

# Scatter plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['SquareFootage'], y=df['Price'])
plt.title("Scatterplot: SquareFootage vs Price")

# Bar chart of model MSEs
plt.subplot(1, 2, 2)
sns.barplot(x=list(models.keys()), y=list(models.values()))
plt.xticks(rotation=15)
plt.ylabel("Mean Squared Error")
plt.title("Bias and Variance Across Models")

plt.tight_layout()
plt.show()
