import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def PCA_Impute(data, missing_mask, n_components=None, max_iter=10, tol=1e-4):
    data = data.copy()
    last_error = None

    for iteration in range(max_iter):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(transformed)

        reconstructed_df = pd.DataFrame(reconstructed, index=data.index, columns=data.columns)
        data[missing_mask] = reconstructed_df[missing_mask]

        error = np.linalg.norm(data.values - reconstructed, ord='fro')
        if last_error is not None and abs(last_error - error) < tol:
            print(f"PCA Imputation converged after {iteration + 1} iterations.")
            break
        last_error = error

    return data
