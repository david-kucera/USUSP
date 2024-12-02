import pandas as pd

print("Semestrálna práca z USU")
v7_missing = pd.read_csv('data/v7_missing.csv')
#print(v7_missing.head)
print(v7_missing.describe())
v7_complete = pd.read_csv('data/v7_complete.csv')
#print(v7_complete.head)
print(v7_complete.describe())