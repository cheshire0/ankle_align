#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_evaluation.py

Final evaluation on TEST set only.

Inputs:
- /data/processed/metadata.csv
- /app/output/model_best_v2.pt

Outputs (/app/output):
- test_metrics.csv
- confusion_matrix.csv
- confusion_matrix.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path("/data/processed")
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path("/app/output")
MODEL_PATH = OUTPUT_DIR / "model_best_v2.pt"

ID_TO_LABEL = {
    0: "1_Pronacio",
    1: "2_Neutralis",
    2: "3_Szupinacio",
}


# -------------------------
# Dataset (NO augmentation)
# -------------------------
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["out_path"]
        y = int(row["label_id"])

        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr)

        return x, y


# -------------------------
# Model definition (must match v2)
# -------------------------
class GAPCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
        return x


# -------------------------
# Evaluation
# -------------------------
def main():
    if not META_PATH.exists():
        raise FileNotFoundError("metadata.csv not found")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("model_best_v2.pt not found")

    df = pd.read_csv(META_PATH)
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[test_df["out_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())]

    if len(test_df) == 0:
        raise RuntimeError("Test set is empty")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    img_size = ckpt["img_size"]

    model = GAPCNN(num_classes=3)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = TestDataset(test_df, img_size=img_size)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating"):
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

    # Metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID_TO_LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[ID_TO_LABEL[i] for i in range(3)],
        columns=[ID_TO_LABEL[i] for i in range(3)],
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(3)
    plt.xticks(ticks, [ID_TO_LABEL[i] for i in range(3)], rotation=45)
    plt.yticks(ticks, [ID_TO_LABEL[i] for i in range(3)])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    print("Evaluation complete.")
    print("Saved:")
    print("- test_metrics.csv")
    print("- confusion_matrix.csv")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()
