import pandas as pd

# Use raw strings or forward slashes to avoid Unicode errors
train_data = pd.read_csv(r'C:\Users\HITESH\OneDrive\Desktop\Titanic-Dataset.csv')

# Display the first few rows of the training data
print(train_data.head())




# Inspect the data
print(train_data.info())
print(train_data.describe())

# Visualize the data
import seaborn as sns
import matplotlib.pyplot as plt

# Countplot for survival
sns.countplot(x='Survived', data=train_data)
plt.show()

# Countplot for Pclass vs Survived
sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.show()

# Countplot for Sex vs Survived
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.show()

# Histogram for Age
sns.histplot(train_data['Age'], bins=30, kde=True)
plt.show()



'D:/Downloads/IRIS.csv'