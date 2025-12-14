#!/bin/bash
set -e
set -o pipefail

echo "============================================================"
echo " AnkleAlign – Deep Learning Pipeline"
echo " Start time: $(date)"
echo "============================================================"

echo
echo "[1/4] DATA PREPROCESSING"
echo "------------------------------------------------------------"
python -u /app/src/01_data_processing.py --clean

echo
echo "[2/4] BASELINE MODEL TRAINING (GAP CNN)"
echo "------------------------------------------------------------"
# GAP CNN baseline -> writes /app/output/model_baseline.pt
python -u /app/src/02_train_baseline.py \
  --epochs 80 \
  --patience 10 \
  --batch-size 16

echo
echo "[3/4] FINAL MODEL TRAINING (TinyCNN)"
echo "------------------------------------------------------------"
# TinyCNN final -> writes /app/output/model_final.pt
python -u /app/src/02_train_final.py

echo
echo "[4/4] FINAL EVALUATION (TEST SET)"
echo "------------------------------------------------------------"
python -u /app/src/03_evaluation.py

echo
echo "============================================================"
echo " Pipeline finished successfully"
echo " End time: $(date)"
echo "============================================================"
