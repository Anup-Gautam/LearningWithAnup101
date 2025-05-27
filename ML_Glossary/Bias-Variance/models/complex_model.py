import pandas as pd
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('data.csv')

# Features and target
X = df[["SquareFootage", "Bedrooms", "Age"]]
y = df["Price"]

poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

results = {"model": "Complex Polynomial Regression", "mse": mse}

with open("../results/complex_model.json", "w") as f:
    json.dump(results, f)