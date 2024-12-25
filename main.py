import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.decomposition import PCA


def load_data():
    """Load datasets."""
    v7_missing = pd.read_csv('data/v7_missing.csv')
    v7_complete = pd.read_csv('data/v7_complete.csv')
    return v7_missing, v7_complete


def describe_data(v7_missing, v7_complete):
    """Print descriptive statistics for both datasets."""
    print("Missing dataset:")
    print(v7_missing.describe())
    print("Complete dataset:")
    print(v7_complete.describe())


def print_missing_stats(v7_missing, v7_complete):
    """Print missing value statistics."""
    print("Missing values (missing dataset):")
    print(v7_missing.isnull().sum())
    print("Missing values (complete dataset):")
    print(v7_complete.isnull().sum())


def extract_original_and_imputed_values(v7_complete, v7_missing, missing_mask):
    """Extract original and imputed values for the missing positions."""
    original_vals = v7_complete[missing_mask].values.flatten()
    imputed_vals = v7_missing[missing_mask].values.flatten()
    valid_idx = ~np.isnan(original_vals)  # Remove NaN values
    return original_vals[valid_idx], imputed_vals[valid_idx]


def impute_missing_values(v7_missing):
    """Impute missing values using column means."""
    return v7_missing.fillna(v7_missing.mean())


def calculate_metrics(original_vals, imputed_vals):
    """Calculate and return MAE and RMSE metrics."""
    mae = mean_absolute_error(original_vals, imputed_vals)
    rmse = root_mean_squared_error(original_vals, imputed_vals)
    return mae, rmse


def pca_imputation_sklearn(data, missing_mask, n_components=None, max_iter=10, tol=1e-4):
    """Impute missing values using PCA."""
    data = data.copy()
    last_error = None

    for iteration in range(max_iter):
        # Perform PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(transformed)

        # Convert reconstructed matrix back to DataFrame
        reconstructed_df = pd.DataFrame(reconstructed, index=data.index, columns=data.columns)

        # Update missing values using the reconstructed matrix
        data[missing_mask] = reconstructed_df[missing_mask]

        # Calculate reconstruction error
        error = np.linalg.norm(data.values - reconstructed, ord='fro')
        if last_error is not None and abs(last_error - error) < tol:
            print(f"PCA Imputation converged after {iteration + 1} iterations.")
            break
        last_error = error

    return data


def pca_imputation_moja(data, missing_mask, n_components=None, max_iter=10, tol=1e-4):
    """Impute missing values using PCA."""
    data = data.copy()
    last_error = None

    for iteration in range(max_iter):
        # Perform custom PCA
        _, reconstructed = pca_moja(data.values, n_components=n_components)

        # Update missing values using the reconstructed matrix
        data.values[missing_mask] = reconstructed[missing_mask]

        # Calculate reconstruction error
        error = np.linalg.norm(data.values - reconstructed, ord='fro')
        if last_error is not None and abs(last_error - error) < tol:
            print(f"PCA Imputation converged after {iteration + 1} iterations.")
            break
        last_error = error

    return data


def pca_moja(data, n_components=None):
    """Implementation of PCA using NumPy."""
    # Center the data (subtract mean)
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Retain only the top n_components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    # Project the data onto the principal components
    transformed = np.dot(centered_data, eigenvectors)

    # Reconstruct the data
    reconstructed = np.dot(transformed, eigenvectors.T) + mean

    return transformed, reconstructed



def main():
    print("Semestrálna práca z USU")
    print("Data imputation using PCA")

    # Load data
    v7_missing, v7_complete = load_data()

    # # Describe data
    # describe_data(v7_missing, v7_complete)

    # Compute missing value statistics
    # print_missing_stats(v7_missing, v7_complete)
    missing_mask = v7_missing.isnull()

    # Impute missing values with column means
    v7_imputed_mean = impute_missing_values(v7_missing)
    # print("After imputing missing values with column means:")
    # print(v7_missing_imputed_mean.describe())

    # Standardize data
    v7_standardized = (v7_imputed_mean - v7_imputed_mean.mean()) / v7_imputed_mean.std()

    # Impute missing values with PCA
    #v7_imputed_pca = pca_imputation_sklearn(v7_standardized, missing_mask, n_components=6)
    v7_imputed_pca = pca_imputation_moja(v7_standardized, missing_mask, n_components=6)

    # De-standardize back to original scale
    v7_imputed_pca = v7_imputed_pca * v7_imputed_mean.std() + v7_imputed_mean.mean()

    # print("After imputing missing values with PCA:")
    # print(v7_imputed_pca.describe())

    # Extract original and imputed values
    original_vals_mean, imputed_vals_mean = extract_original_and_imputed_values(v7_complete, v7_imputed_mean, missing_mask)
    original_vals_pca, imputed_vals_pca = extract_original_and_imputed_values(v7_complete, v7_imputed_pca, missing_mask)

    # Calculate metrics
    mae_mean, rmse_mean = calculate_metrics(original_vals_mean, imputed_vals_mean)
    mae_pca, rmse_pca = calculate_metrics(original_vals_pca, imputed_vals_pca)
    print("Imputing missing values with MEAN:")
    print("Mean Absolute Error:", mae_mean)
    print("Root Mean Squared Error:", rmse_mean)

    print("Imputing missing values with PCA:")
    print("Mean Absolute Error:", mae_pca)
    print("Root Mean Squared Error:", rmse_pca)

    print("Difference MAE ... MEAN - PCA:", mae_mean - mae_pca)
    print("Difference RMSE ... MEAN - PCA:", rmse_mean - rmse_pca)


if __name__ == "__main__":
    main()
