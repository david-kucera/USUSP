import numpy as np


def PCA(data, n_components=None):
    """Implementation of PCA using NumPy."""
    # Center the data (subtract mean)
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("PCA can only be applied to numeric data.")

    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort in reverse order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only top n_components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    transformed = np.dot(centered_data, eigenvectors)
    reconstructed = np.dot(transformed, eigenvectors.T) + mean

    return reconstructed


def PCA_Impute(data, missing_mask, n_components=None, max_iter=10, tol=1e-4):
    """Impute missing values using PCA."""
    data = data.copy()
    last_error = None

    for iteration in range(max_iter):
        reconstructed = PCA(data.values, n_components=n_components)
        data.values[missing_mask] = reconstructed[missing_mask]
        error = np.linalg.norm(data.values - reconstructed, ord='fro')
        if last_error is not None and abs(last_error - error) < tol:
            break
        last_error = error

    return data
