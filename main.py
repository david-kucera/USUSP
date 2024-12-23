import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


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
    v7_missing.fillna(v7_missing.mean(), inplace=True)
    return v7_missing


def calculate_metrics(original_vals, imputed_vals):
    """Calculate and return MAE and RMSE metrics."""
    mae = mean_absolute_error(original_vals, imputed_vals)
    rmse = root_mean_squared_error(original_vals, imputed_vals)
    return mae, rmse


def main():
    print("Semestrálna práca z USU")
    print("Data imputation using PCA")

    # Load data
    v7_missing, v7_complete = load_data()

    # Describe data
    describe_data(v7_missing, v7_complete)

    # Compute missing value statistics
    print_missing_stats(v7_missing, v7_complete)
    missing_mask = v7_missing.isnull()

    # Impute missing values with column means
    v7_missing = impute_missing_values(v7_missing)
    print("After imputing missing values with column means:")
    print(v7_missing.describe())

    # Extract original and imputed values
    original_vals, imputed_vals = extract_original_and_imputed_values(v7_complete, v7_missing, missing_mask)

    # Calculate metrics
    mae, rmse = calculate_metrics(original_vals, imputed_vals)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)


if __name__ == "__main__":
    main()
