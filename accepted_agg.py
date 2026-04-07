import os
import pandas as pd

balancer = "./balancer"
population = "/population/"
uniform = "/uniform/"

aggregation_directories = [balancer + uniform + "uniform70real",
                           balancer + uniform + "uniform100real",
                           balancer + population + "population70real",
                           balancer + population + "population100real",
                           balancer + "/final/real/uniform_split/test",
                           balancer + "/final/real/population_split/test"]
save_names = ["uniform70.csv", "uniform100.csv", "population70.csv", "population100.csv", "uniformTest.csv", "population_test.csv"]

minAge = 5
maxAge = 70

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

    df.to_csv("./synthetic_csv/" + save_names[index], index=False)

    print(save_names[index])
    print(df.head())
    print(len(df))
    print(min(df["age"]))
    print(max(df["age"]))