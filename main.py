import pandas as pd
import numpy as np

from pca import PCA_Impute
# from pca_sklearn import PCA_Impute

from metrics import MAE, RMSE
from plot import plot_mae, plot_rmse


def load_data():
    """Load datasets and drop the first column."""
    v7_missing = pd.read_csv('data/v7_missing.csv').iloc[:, 1:]
    v7_complete = pd.read_csv('data/v7_complete.csv').iloc[:, 1:]
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
    valid_idx = ~np.isnan(original_vals)
    return original_vals[valid_idx], imputed_vals[valid_idx]


def impute_missing_values(v7_missing):
    """Impute missing values using column means."""
    return v7_missing.fillna(v7_missing.mean())


def calculate_metrics(original_vals, imputed_vals):
    """Calculate and return MAE and RMSE metrics."""
    mae = MAE(original_vals, imputed_vals)
    rmse = RMSE(original_vals, imputed_vals)
    return mae, rmse


def main(component_count=None):
    print("Semestrálna práca z USU")
    print("Data imputation using PCA")
    print("Component count:", component_count)

    # Load data
    v7_missing, v7_complete = load_data()
    missing_mask = v7_missing.isnull()

    # Impute missing values with column means
    v7_imputed_mean = impute_missing_values(v7_missing)

    # Standardize data
    v7_standardized = (v7_imputed_mean - v7_imputed_mean.mean()) / v7_imputed_mean.std()

    # Impute missing values with PCA
    v7_imputed_pca = PCA_Impute(v7_standardized, missing_mask, n_components=component_count)
    # De-standardize back to original scale
    v7_imputed_pca = v7_imputed_pca * v7_imputed_mean.std() + v7_imputed_mean.mean()

    # Save to file
    v7_imputed_pca.to_csv('data/v7_imputed_pca.csv')
    v7_imputed_mean.to_csv('data/v7_imputed_mean.csv')

    # Extract original and imputed values
    original_vals, imputed_vals_mean = extract_original_and_imputed_values(v7_complete, v7_imputed_mean, missing_mask)
    _, imputed_vals_pca = extract_original_and_imputed_values(v7_complete, v7_imputed_pca, missing_mask)

    # Calculate metrics
    mae_mean, rmse_mean = calculate_metrics(original_vals, imputed_vals_mean)
    mae_pca, rmse_pca = calculate_metrics(original_vals, imputed_vals_pca)
    print("Imputing missing values with MEAN:")
    print("Mean Absolute Error:", mae_mean)
    print("Root Mean Squared Error:", rmse_mean)

    print("Imputing missing values with PCA:")
    print("Mean Absolute Error:", mae_pca)
    print("Root Mean Squared Error:", rmse_pca)

    diff_mae = mae_mean - mae_pca
    diff_rmse = rmse_mean - rmse_pca
    print("Difference MAE ... MEAN - PCA:", diff_mae)
    print("Difference RMSE ... MEAN - PCA:", diff_rmse)
    # return mae_pca, rmse_pca
    return diff_mae, diff_rmse


def experiment_components():
    maes = []
    rmses = []
    component_counts = list(range(1, 11))
    for i in component_counts:
        mae, rmse = main(i)
        maes.append(mae)
        rmses.append(rmse)

    plot_mae(component_counts, maes)
    plot_rmse(component_counts, rmses)


if __name__ == "__main__":
    try :
        np.random.seed(42)
        main(5)
        # experiment_components()
    except Exception as e :
        print(e)