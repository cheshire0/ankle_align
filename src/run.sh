#!/bin/bash
set -e  # stop on first error

echo "=== AnkleAlign pipeline start ==="

echo "[1/3] Data preprocessing..."
python -u /app/src/01_data_processing.py --clean

echo "[2/3] Model training (v2)..."
python -u /app/src/02_train_v2.py \
  --epochs 80 \
  --patience 10 \
  --batch-size 16

echo "[3/3] Final evaluation on test set..."
python -u /app/src/03_evaluation.py

echo "=== Pipeline finished successfully ==="