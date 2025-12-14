#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_evaluation.py

Final evaluation on TEST set.

Inputs:
- /data/processed/metadata.csv
- /app/output/model_final.pt              (required)
- /app/output/model_baseline.pt           (optional but recommended)

Outputs (/app/output):
Per model (baseline / final):
- test_metrics_<tag>.csv
- confusion_matrix_<tag>.csv
- confusion_matrix_<tag>.png

Summary:
- test_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import setup_logger

logger = setup_logger()

DATA_DIR = Path("/data/processed")
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path("/app/output")

MODEL_FINAL_PATH = OUTPUT_DIR / "model_final.pt"
MODEL_BASELINE_PATH = OUTPUT_DIR / "model_baseline.pt"

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
# Models (must match training)
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


class TinyCNNBaseline(nn.Module):
    """
    Final model: 2 conv blocks + FC (must match 02_train_final.py)
    """
    def __init__(self, num_classes: int = 3, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        feat_size = img_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * feat_size * feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _infer_model_type(ckpt: dict) -> str:
    """
    Decide which model class to instantiate.
    Prefer explicit ckpt["model_name"], else infer from state_dict keys.
    """
    name = ckpt.get("model_name")
    if isinstance(name, str):
        # normalize common names
        if name in ("GAPCNN", "GAPCNNBaseline"):
            return "GAPCNN"
        if name in ("TinyCNNBaseline", "TinyCNNFinal", "TinyCNN"):
            return "TinyCNNBaseline"
        return name

    state = ckpt.get("model_state", {})
    keys = list(state.keys()) if isinstance(state, dict) else []
    if any(k.startswith("backbone.") for k in keys) or any(k.startswith("head.") for k in keys):
        return "GAPCNN"
    if any(k.startswith("features.") for k in keys) or any(k.startswith("classifier.") for k in keys):
        return "TinyCNNBaseline"
    return "UNKNOWN"


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for xb, yb in tqdm(loader, desc="Evaluating"):
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(preds.numpy().tolist())

    return y_true, y_pred


def save_confusion_matrix(cm: np.ndarray, tag: str) -> Path:
    png_path = OUTPUT_DIR / f"confusion_matrix_{tag}.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (Test) – {tag}")
    plt.colorbar()
    ticks = np.arange(3)
    plt.xticks(ticks, [ID_TO_LABEL[i] for i in range(3)], rotation=45)
    plt.yticks(ticks, [ID_TO_LABEL[i] for i in range(3)])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return png_path


def evaluate_checkpoint(model_path: Path, test_df: pd.DataFrame, tag: str) -> Dict[str, object]:
    logger.info(f"--- Evaluating {tag} ---")
    logger.info(f"Checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    img_size = int(ckpt.get("img_size", 224))

    model_type = _infer_model_type(ckpt)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Image size: {img_size}")

    if model_type == "TinyCNNBaseline":
        model = TinyCNNBaseline(num_classes=3, img_size=img_size)
    elif model_type == "GAPCNN":
        model = GAPCNN(num_classes=3)
    else:
        raise RuntimeError(f"Cannot infer model type for checkpoint: {model_path} (got '{model_type}')")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    ds = TestDataset(test_df, img_size=img_size)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    y_true, y_pred = run_eval(model, loader)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID_TO_LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(report).transpose()
    metrics_path = OUTPUT_DIR / f"test_metrics_{tag}.csv"
    metrics_df.to_csv(metrics_path, index=True)
    logger.info(f"Saved metrics CSV: {metrics_path}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[ID_TO_LABEL[i] for i in range(3)],
        columns=[ID_TO_LABEL[i] for i in range(3)],
    )
    cm_path = OUTPUT_DIR / f"confusion_matrix_{tag}.csv"
    cm_df.to_csv(cm_path, index=True)
    logger.info(f"Saved confusion matrix CSV: {cm_path}")

    logger.info("Confusion matrix (rows=true, cols=pred):")
    for row in cm.tolist():
        logger.info(str(row))

    png_path = save_confusion_matrix(cm, tag=tag)
    logger.info(f"Saved confusion matrix PNG: {png_path}")

    acc = float(report.get("accuracy", 0.0))
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))

    logger.info(f"{tag} | test_accuracy={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")

    return {
        "model": tag,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "metrics_csv": str(metrics_path),
        "cm_csv": str(cm_path),
        "cm_png": str(png_path),
    }


def main():
    logger.info("Starting FINAL evaluation on TEST set (baseline + final if available)")
    logger.info(f"Metadata path: {META_PATH}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    if not META_PATH.exists():
        raise FileNotFoundError("metadata.csv not found")

    if not MODEL_FINAL_PATH.exists():
        raise FileNotFoundError("model_final.pt not found (required)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META_PATH)
    logger.info(f"Loaded metadata rows: {len(df)}")

    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[test_df["out_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())].copy()

    if len(test_df) == 0:
        raise RuntimeError("Test set is empty")

    logger.info(f"Test samples: {len(test_df)}")
    logger.info("Test label distribution:")
    for label, cnt in test_df["label_id"].value_counts().sort_index().items():
        logger.info(f"  {ID_TO_LABEL.get(int(label), str(label))}: {cnt}")

    summary_rows: List[Dict[str, object]] = []

    # Baseline (optional)
    if MODEL_BASELINE_PATH.exists():
        summary_rows.append(evaluate_checkpoint(MODEL_BASELINE_PATH, test_df, tag="baseline"))
    else:
        logger.warning("Baseline checkpoint not found: /app/output/model_baseline.pt (skipping baseline evaluation)")

    # Final (required)
    summary_rows.append(evaluate_checkpoint(MODEL_FINAL_PATH, test_df, tag="final"))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary comparison: {summary_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
