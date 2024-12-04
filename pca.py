import numpy as np
import pandas as pd

# Load the dataset
v7_missing = pd.read_csv('data/v7_missing.csv')
v7_complete = pd.read_csv('data/v7_complete.csv')

# Replace missing values (or zeros) with NaN
v7_missing.replace(0, np.nan, inplace=True)

# 1. Replace missing values with the column mean as an initial guess
initial_guess = v7_missing.fillna(v7_missing.mean())

# Convert to NumPy array for matrix operations
data = initial_guess.values

# Standardize the data (subtract mean and divide by std)
mean = np.nanmean(data, axis=0)
std = np.nanstd(data, axis=0)
standardized_data = (data - mean) / std

# 2. Iteratively apply PCA-based reconstruction
tolerance = 1e-6  # Convergence threshold
max_iterations = 100
for iteration in range(max_iterations):
    # Fill missing values with the last reconstructed data
    missing_mask = np.isnan(data)
    standardized_data[missing_mask] = 0  # Set missing values to 0 for covariance calculation

    # Compute covariance matrix
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Project data onto the principal components
    scores = standardized_data @ eigenvectors

    # Reconstruct the data
    reconstructed = scores @ eigenvectors.T

    # Rescale back to the original scale
    reconstructed = reconstructed * std + mean

    # Replace only the missing values in the original data
    prev_data = data.copy()
    data[missing_mask] = reconstructed[missing_mask]

    # Check for convergence
    if np.linalg.norm(data - prev_data) < tolerance:
        print(f"Converged after {iteration + 1} iterations")
        break

# Final imputed dataset
imputed_data = pd.DataFrame(data, columns=v7_missing.columns)

# Compare original and imputed datasets
print("Imputed Data:")
print(imputed_data.describe())

print(v7_complete.describe())
