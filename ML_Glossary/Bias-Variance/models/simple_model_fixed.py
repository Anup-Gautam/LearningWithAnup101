import pandas as pd
import json
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('data.csv')

# Features and target (use more features)
X = df[["SquareFootage", "Bedrooms", "Age"]]
y = df["Price"]

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train the model
model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

results = {"model": "Simple Linear Regression with Ridge Regularization", "mse": mse}

with open("../results/simple_model_fixed.json", "w") as f:
    json.dump(results, f)