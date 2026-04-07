import os
import pandas as pd

aggregation_directories = ["./balancer/uniform/uniform70real", "./balancer/uniform/uniform100real",
                           "./balancer/population/population70real", "./balancer/population/population100real"]
save_names = ["uniform70.csv", "uniform100.csv", "population70.csv", "population100.csv"]
minAge = 5
maxAge = 70
#countPer = int(20000 / (maxAge - minAge + 1))

for index, aggregation_directory in enumerate(aggregation_directories):
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

    df.to_csv(save_names[index], index=False)

print(df.head())
print(len(df))
print(min(df["age"]))
print(max(df["age"]))