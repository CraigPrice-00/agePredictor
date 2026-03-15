import pandas as pd
import os

#Helper program to combine datasets and ensure correct path

wiki = pd.read_csv("./wiki_filtered.csv")
imdb = pd.read_csv("./imdb_filtered.csv")

wiki = wiki[['age', 'file_path']]
imdb = imdb[['age', 'file_path']]

BASE_DIR_WIKI = r"D:/agePredictor/dataset/wiki_crop"
BASE_DIR_IMDB = r"D:/agePredictor/dataset/imdb_crop"

wiki['file_path'] = wiki['file_path'].apply(
    lambda x: os.path.join(BASE_DIR_WIKI, x)
)
imdb['file_path'] = imdb['file_path'].apply(
    lambda x: os.path.join(BASE_DIR_IMDB, x)
)

dataset = pd.concat([wiki, imdb], ignore_index=True)
print(f"Saving data of length {len(dataset)} to .csv")
dataset.to_csv("./filtered_dataset.csv", index=False)


