import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_excel('patent.xlsx')

# Basic Overview
print("Dataset Shape:", data.shape)
print("\nColumn Information:")
print(data.info())
print("\nFirst Few Rows:")
print(data.head())

# Summary Statistics
print("\nSummary Statistics (Numerical Data):")
print(data.describe())

# Checking for Missing Values
missing_values = data.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values)

# Visualizations

# 1. Distribution of the Number of Claims
plt.figure(figsize=(10, 5))
sns.histplot(data['Number of Claims'], bins=10, kde=True)
plt.title("Distribution of the Number of Claims")
plt.xlabel("Number of Claims")
plt.ylabel("Frequency")
plt.show()

# 2. Patents by Legal Status
plt.figure(figsize=(8, 5))
sns.countplot(y='Legal Status', data=data)
plt.title("Count of Patents by Legal Status")
plt.xlabel("Count")
plt.ylabel("Legal Status")
plt.show()

# 3. Technology Domain Analysis
plt.figure(figsize=(10, 5))
sns.countplot(y='Technology Domain', data=data, order=data['Technology Domain'].value_counts().index)
plt.title("Distribution of Technology Domains")
plt.xlabel("Count")
plt.ylabel("Technology Domain")
plt.show()

# 4. Filing vs Publication Date
data['Filing Date'] = pd.to_datetime(data['Filing Date'], errors='coerce')
data['Publication Date'] = pd.to_datetime(data['Publication Date'], errors='coerce')

# Time taken from Filing to Publication
data['Time to Publish (days)'] = (data['Publication Date'] - data['Filing Date']).dt.days
plt.figure(figsize=(10, 5))
sns.histplot(data['Time to Publish (days)'].dropna(), bins=15, kde=True)
plt.title("Time from Filing to Publication (days)")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.show()

# 5. Correlation Analysis (where applicable)
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
