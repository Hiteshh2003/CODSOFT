# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the dataset
file_path = 'D:/Downloads/IRIS.csv'
iris_data = pd.read_csv('D:/Downloads/IRIS.csv')

# Display the first few rows of the dataset to understand its structure
print("Sample of the Iris dataset:")
print(iris_data.head())

# Step 2: Exploratory Data Analysis (EDA)
# Basic statistics of the dataset
print("\nSummary statistics of the dataset:")
print(iris_data.describe())

# Check for any missing values
print("\nMissing values in the dataset:")
print(iris_data.isnull().sum())

# Visualize the distribution of each feature by species
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(data=iris_data, x='sepal_length', hue='species', kde=True)
plt.subplot(2, 2, 2)
sns.histplot(data=iris_data, x='sepal_width', hue='species', kde=True)
plt.subplot(2, 2, 3)
sns.histplot(data=iris_data, x='petal_length', hue='species', kde=True)
plt.subplot(2, 2, 4)
sns.histplot(data=iris_data, x='petal_width', hue='species', kde=True)
plt.tight_layout()
plt.show()

# Step 3: Preprocessing the data
# Separate features (X) and target variable (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train a machine learning model (K-Nearest Neighbors Classifier)
# Initialize the classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
# Predictions on the test set
y_pred = knn.predict(X_test)

# Model evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
