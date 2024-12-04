import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from fancyimpute import IterativeSVD

# Load data
print("Semestrálna práca z USU")
v7_missing = pd.read_csv('data/v7_missing.csv')
v7_complete = pd.read_csv('data/v7_complete.csv')

# Dataset overview
print("Missing Data Overview:")
print(v7_missing.describe())
print("\nComplete Data Overview:")
print(v7_complete.describe())

# Check for missing values
print("\nCount of missing values in each column:")
print(v7_missing.isnull().sum())

# Replace missing values (or zeros, if they represent missing) with NaN
v7_missing.replace(0, np.nan, inplace=True)

# Impute missing values using PCA
imputer = IterativeSVD()
v7_imputed = imputer.fit_transform(v7_missing)

# Convert imputed data back to DataFrame
v7_imputed_df = pd.DataFrame(v7_imputed, columns=v7_missing.columns)

# Standardize both datasets (complete and imputed) for comparison
scaler = StandardScaler()
v7_complete_scaled = scaler.fit_transform(v7_complete)
v7_imputed_scaled = scaler.fit_transform(v7_imputed_df)
v7_missing_scaled = scaler.fit_transform(v7_missing)

# Compute the error (Mean Squared Error)
mse = mean_squared_error(v7_complete_scaled, v7_imputed_scaled)
print(f"\nMean Squared Error between complete and imputed data: {mse:.4f}")

# Visualize the comparison of value distributions
plt.figure(figsize=(10, 6))
plt.hist(v7_complete_scaled.flatten(), bins=30, alpha=0.5, label='Complete Data')
plt.hist(v7_missing_scaled.flatten(), bins=30, alpha=0.5, label='Original Data')
plt.hist(v7_imputed_scaled.flatten(), bins=30, alpha=0.5, label='Imputed Data')
plt.legend()
plt.title('Comparison of Value Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
