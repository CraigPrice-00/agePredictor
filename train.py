import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import AgeDataset
from model import build_model

CSV_PATH = "filtered_dataset.csv"
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
MIN_AGE = 1
MAX_AGE = 119
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

SAVE_PATH = "best_age_model.pth"

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def denormalize_age(age_norm):
    return age_norm * AGE_RANGE + MIN_AGE

def evaluate_metrics(model, loader):
    model.eval()

    errors = []

    with torch.no_grad():
        for images, ages in tqdm(loader, desc="eval"):
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

    return {
        "mae": mae,
        "rmse": rmse,
        "median_ae": medae,
        "bias": bias,
        "std_error": std_err,
        "max_error": max_err,
        "n": len(errors),
    }

def run():
    print("Device:", DEVICE)

    # Read and split CSV
    df = pd.read_csv(CSV_PATH)
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Write temp split files (simple + reliable)
    train_csv = "_train_split.csv"
    val_csv = "_val_split.csv"
    test_csv = "_test_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_tf, val_tf = get_transforms()

    train_ds = AgeDataset(train_csv, transform=train_tf)
    val_ds = AgeDataset(val_csv, transform=val_tf)
    test_ds = AgeDataset(test_csv, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model
    model = build_model().to(DEVICE)

    # Loss/optim
    criterion = nn.L1Loss()  # MAE in normalized space
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # AMP (mixed precision) if CUDA
    use_amp = (DEVICE == "cuda")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train
        model.train()
        running_loss = 0.0

        for images, ages in tqdm(train_loader, desc="train"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, ages)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss (normalized MAE): {train_loss:.4f}")

        model.eval()
        abs_err_sum_years = 0.0
        n = 0

        with torch.no_grad():
            for images, ages in tqdm(val_loader, desc="val"):
                images = images.to(DEVICE, non_blocking=True)
                ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)  # (B,1)

                preds = model(images)

                pred_years = preds * AGE_RANGE + MIN_AGE
                true_years = ages * AGE_RANGE + MIN_AGE
                abs_err_sum_years += torch.sum(torch.abs(pred_years - true_years)).item()

                n += ages.size(0)

        val_mae = abs_err_sum_years / max(1, n)
        print(f"Val MAE (years): {val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mae": val_mae,
                    "image_size": IMAGE_SIZE,
                    "max_age": MAX_AGE,
                    "min_age": MIN_AGE,
                },
                SAVE_PATH
            )
            print(f"Saved best checkpoint -> {SAVE_PATH}")

        scheduler.step()
        print("LR:", scheduler.get_last_lr()[0])

    print("Finished Training, running final test evaluation")

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_metrics(model, test_loader)

    print(f"Test MAE (years): {metrics['mae']:.2f}")
    print(f"Test RMSE (years): {metrics['rmse']:.2f}")
    print(f"Test Median AE (years): {metrics['median_ae']:.2f}")
    print(f"Test Bias (years): {metrics['bias']:.2f}")
    print(f"Test Std Error (years): {metrics['std_error']:.2f}")
    print(f"Test Max Error (years): {metrics['max_error']:.2f}")
    print(f"Test Samples: {metrics['n']}")

    os.remove(train_csv)
    os.remove(val_csv)
    os.remove(test_csv)


if __name__ == "__main__":
    run()