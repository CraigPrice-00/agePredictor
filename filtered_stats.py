import pandas as pd

df = pd.read_csv('filtered_dataset.csv')

print(df.head())
print(len(df))
print(min(df['age']))
print(max(df['age']))