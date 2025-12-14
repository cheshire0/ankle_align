# Deep Learning Class (VITMMA19) Project Work 

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Sebők Lili (XV0M8Z)
- **Aiming for +1 Mark**: Yes

### Solution Description

The objective of this project is the automatic classification of ankle alignment from foot images into three clinically relevant categories: pronation, neutral alignment, and supination. The dataset consists of images annotated using Label Studio and organized into per-Neptun folders, where filename collisions across folders are possible.

A complete deep learning pipeline was implemented, covering data preparation, baseline modeling, incremental model improvement, evaluation, and deployment. All steps are fully containerized using Docker and can be reproduced on a CPU-only system.

Two models are trained and evaluated:

1. A **baseline convolutional neural network**, serving as a reference model.
2. An **improved GAP-based convolutional neural network**, designed for better generalization and parameter efficiency.

The improved model incorporates early stopping and class weighting and explicitly avoids rotation-based augmentation due to medical interpretability concerns.

## Data Preparation

Raw data is expected in the following format:

```
/data/raw/
  <NEPTUN_CODE>/
    images...
    label_studio_export.json
```

The data preparation process is fully automated by `01_data_processing.py` and includes:

* Parsing Label Studio JSON exports
* Matching annotations to images
* Skipping missing, unreadable, or invalid samples
* Prefixing filenames with Neptun codes to avoid collisions
* Resizing images to a fixed resolution
* Stratified splitting into train, validation, and test sets
* Writing metadata and a detailed processing report

The processed dataset is stored under `/data/processed/` and includes a `metadata.csv` file used by all subsequent scripts.


## Model Development

### Baseline Model

* Architecture: Small CNN with two convolutional blocks and a fully connected classifier
* Purpose: Reference model for comparison
* Characteristics:

  * No class weighting
  * No early stopping
  * Standard cross-entropy loss

### Improved Model (v2)

* Architecture: CNN with Global Average Pooling (GAP)
* Improvements over baseline:

  * Significantly fewer parameters
  * Better regularization through GAP
  * Class-weighted loss to handle imbalance
  * Early stopping based on validation accuracy
  * Data augmentation limited to cropping, translation, and color jitter (no rotation)


## Final Evaluation

Final evaluation is performed exclusively on the held-out test set using `03_evaluation.py`. Both the baseline and improved models are evaluated using identical test data.

### Evaluation Metrics

* Overall accuracy
* Precision, recall, and F1-score per class
* Macro-averaged F1-score
* Weighted F1-score
* Confusion matrix

All metrics are saved as CSV files and logged to `log/run.log`.

### Quantitative Results (Test Set)


## Final Evaluation Results (Test Set)

### Model Comparison

| Model                     | Parameters | Test Accuracy |   Macro F1 | Weighted F1 |
| ------------------------- | ---------: | ------------: | ---------: | ----------: |
| **Baseline (GAPCNN)**     |    980,003 |        0.4355 |     0.3356 |      0.3882 |
| **Final Model (TinyCNN)** | 12,850,659 |    **0.6129** | **0.5721** |  **0.6086** |


### Confusion Matrix – Baseline (GAPCNN)

Rows = true label, columns = predicted label

| True \ Pred    | Pronáció | Neutrális | Szupináció |
| -------------- | -------: | --------: | ---------: |
| **Pronáció**   |        3 |        16 |          6 |
| **Neutrális**  |        4 |        22 |          3 |
| **Szupináció** |        2 |         4 |          2 |


### Confusion Matrix – Final Model (TinyCNN)

| True \ Pred    | Pronáció | Neutrális | Szupináció |
| -------------- | -------: | --------: | ---------: |
| **Pronáció**   |       15 |         9 |          1 |
| **Neutrális**  |        8 |        20 |          1 |
| **Szupináció** |        3 |         2 |          3 |


### Interpretation

* The **baseline GAPCNN** provides a lightweight reference model with limited capacity.
* The **final TinyCNN model** significantly improves accuracy and F1-scores at the cost of higher parameter count.
* This demonstrates **incremental model development**, satisfying the *Outstanding level* requirement.
* Both models show strongest performance on the **Neutral** class; **Supination** remains the hardest class due to class imbalance.


## Logging

All scripts use a unified logging configuration implemented in `src/utils.py`. The logs include:

* Hyperparameter configuration
* Data loading and preprocessing confirmation
* Model architecture and parameter counts
* Epoch-level training and validation metrics
* Final test evaluation results

All logs are written to standard output and captured by Docker into `log/run.log`.


## ML as a Service (Backend + GUI)

The trained improved model is deployed as an ML service:

* **Backend:** FastAPI
* **Frontend:** Gradio-based web GUI

The service allows users to upload an image and receive a predicted ankle alignment class with confidence scores.


### Extra Credit Justification

The project fulfills the outstanding-level requirements by:

* Providing a clear baseline and an incrementally improved model
* Applying medically justified augmentation constraints
* Performing advanced evaluation with multiple metrics
* Deploying the trained model as an interactive ML service
* Ensuring full reproducibility through Docker and detailed logging


### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

### Build

```bash
docker build -t anklealign .
```

### Run Full Pipeline (with logging)

```bash
docker run --rm \
  -v /absolute/path/to/data:/data \
  -v /absolute/path/to/output:/app/output \
  anklealign:latest \
  bash /app/src/run.sh > log/run.log 2>&1
```

### Run API Service

```bash
docker run --rm -p 8000:8000 \
  -v /absolute/path/to/output:/app/output \
  anklealign:latest \
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions


The repository is structured as follows:

* **`src/`**: Contains the source code for the machine learning pipeline.

  * `01_data_processing.py`: Loads Label Studio exports, matches images, cleans the dataset, performs resizing and stratified train/val/test split, and generates metadata and reports.
  * `02_train_baseline.py`: GAP-based CNN training script with class weighting, early stopping, and no-rotation augmentation.
  * `03_evaluation.py`: Final evaluation on the test set, producing accuracy, precision/recall/F1 metrics, and confusion matrix (CSV + PNG).
  * `api.py`: FastAPI-based ML-as-a-service backend with an optional GUI frontend for inference.
  * `run.sh`: End-to-end pipeline runner executing preprocessing, training, and evaluation inside the container.
  * `utils.py`: Shared utilities, including centralized logging configuration.
  * `config.py`: Configuration file defining hyperparameters and paths used across scripts.

* **`log/`**: Contains log files.

  * `run.log`: Full log of a complete pipeline run, including configuration, data preparation, training progress, and final evaluation metrics.


* **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.