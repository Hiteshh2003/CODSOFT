# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = 'D:/Downloads/advertising.csv'
advertising_data = pd.read_csv('D:/Downloads/advertising.csv')

# Display the first few rows of the dataset to understand its structure
print("Sample of the Advertising dataset:")
print(advertising_data.head())

# Basic statistics of the dataset
print("\nSummary statistics of the dataset:")
print(advertising_data.describe())

# Check for any missing values
print("\nMissing values in the dataset:")
print(advertising_data.isnull().sum())

# Visualize the relationships between variables
sns.pairplot(advertising_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.7)
plt.suptitle('Pairplot of Sales vs Advertising Expenditure')
plt.show()

# Step 2: Preprocess the data
# Separate features (X) and target variable (y)
X = advertising_data[['TV', 'Radio', 'Newspaper']]
y = advertising_data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a machine learning model (Linear Regression)
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Coefficients and intercept
print("\nCoefficients:")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
print("\nIntercept:", model.intercept_)
