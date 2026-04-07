import sys

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchvision import transforms
from dataset import AgeDataset
from model import build_model
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE


IMAGE_SIZE = 128
BATCH_SIZE = 32

trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
def load_model(model_p):
    save_model = torch.load(model_p, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(save_model["model_state_dict"])
    model.eval()
    return model

def eval(csv, modelp):
    evaluation_dataset = AgeDataset(csv, transform=trans)

    loader = DataLoader(
        evaluation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = load_model(modelp)

    errors = []

    with torch.no_grad():
        for images, ages in tqdm(loader, desc="evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            preds = model(images)

            pred_years = preds * AGE_RANGE + MIN_AGE
            true_years = ages * AGE_RANGE + MIN_AGE

            batch_errors = (pred_years - true_years).cpu().numpy().flatten()
            errors.extend(batch_errors)

    errors = np.array(errors)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    medae = np.median(np.abs(errors))
    bias = np.mean(errors)              # positive = overpredicting
    std_err = np.std(errors)
    max_err = np.max(np.abs(errors))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Median AE: {medae:.2f}")
    print(f"Bias: {bias:.2f}")
    print(f"Std Error: {std_err:.2f}")
    print(f"Max Error: {max_err:.2f}")
    print(f"Samples: {len(errors)}")

def singlepred(filepath, modelpath):

    model = load_model(modelpath)
    pred_img = Image.open(filepath).convert('RGB')
    pred_img = trans(pred_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(pred_img)

    predicted_age = pred.item() * AGE_RANGE + MIN_AGE
    print(f"Predicted Age: {round(predicted_age)}")

    return None

if __name__ == '__main__':
    if sys.argv[1] == "eval":
        if len(sys.argv) == 2:
            csv_path = "./synthetic_csv/uniformTest.csv"
            model_path = "./best_models_synth/uniform70.pth"
            eval(csv_path, model_path)
        elif len(sys.argv) == 4:
            eval(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "pred":
        singlepred(sys.argv[2], sys.argv[3])
    else:
        print("Usage: python test.py [ eval <csv_path> <model_path> | pred <filepath> <model_path> ]")