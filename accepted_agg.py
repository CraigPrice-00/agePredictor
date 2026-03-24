import os
import pandas as pd

aggregation_directory = "dataset2"

minAge = 5
maxAge = 70
countPer = int(20000 / (maxAge - minAge + 1))

listRows = []

for folder in os.listdir(aggregation_directory):

    path = os.path.join(aggregation_directory, folder)

    age = int(folder)

    files = []

    for file in os.listdir(path):
        files.append(os.path.join(path, file))


    for file_path in files:
        listRows.append({
            "file_path": file_path,
            "age": age
        })

df = pd.DataFrame(listRows)

df.to_csv("dataset_2_unbalanced.csv", index=False)

print(df.head())
print(len(df))
print(min(df["age"]))
print(max(df["age"]))