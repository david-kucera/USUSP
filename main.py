import pandas as pd

print("Semestrálna práca z USU")
print("Data imputation using PCA")
v7_missing = pd.read_csv('data/v7_missing.csv')
v7_complete = pd.read_csv('data/v7_complete.csv')

print("missing dataset:")
print(v7_missing.describe())
print("complete dataset:")
print(v7_complete.describe())

print("missing values")
print(v7_missing.isnull().sum())
print(v7_complete.isnull().sum())

# NOTES FROM ISL book
# we could replace missing values by column means...
# we need to do it using PCA -- MATRIX COMPLETION
# -- appropriate if the missingness is random (--- true in our case)
# -- could be used in recommender systems

# matrix completion algorithm
# 1 - create a complete data matrix
#       - missing values = mean of column

v7_missing.fillna(v7_missing.mean(), inplace=True)
print(v7_missing.describe())
print(v7_complete.isnull().sum())

# 2 - optimize iteratively
#       - solve PCA (12.13)
#       - update missing values
#       - check the objective (12.14) - stop when error stops decreasing significantly
# 3 - return the completed matrix with updated missing values

