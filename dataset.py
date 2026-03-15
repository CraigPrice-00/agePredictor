import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class AgeDataset(Dataset):
    def __init__(self, csv, transform=None, max_age=100):
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.max_age = max_age

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['file_path']
        age = float(self.df.iloc[idx]['age'])

        age = age / self.max_age

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)
