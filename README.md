# Multi-Class Respiratory Diagnostic Model

A deep learning model for classifying respiratory diseases from lung sound recordings using transfer learning on ResNet-18. Trained on the ICBHI 2017 Respiratory Sound Database and externally validated on the Kaggle Asthma Detection Dataset v2.

---

## Classes

| Label | Description |
|---|---|
| COPD | Chronic Obstructive Pulmonary Disease |
| Pneumonia | Bacterial/viral lung infection |
| Bronchiectasis | Permanent airway widening |
| Healthy | No respiratory condition |
| URTI | Upper Respiratory Tract Infection |

---

## Model Architecture

- Base: ResNet-18 (ImageNet pretrained)
- Input: MFCC features (40 coefficients × 680 time frames) replicated to 3 channels
- Head: Dropout(0.7) → Linear(512) → ReLU → Dropout(0.7) → Linear(num_classes)
- Loss: Weighted CrossEntropyLoss (handles class imbalance)
- Optimizer: Adam (lr=0.0001, weight_decay=5e-4)
- Scheduler: ReduceLROnPlateau
- Early Stopping: patience=8

---

## Results

### Internal Validation (ICBHI 2017 — 80/20 Stratified Split)

| Metric | Score |
|---|---|
| Accuracy | 94.5% |
| Macro F1 | 0.945 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| COPD | 1.00 | 0.97 | 0.99 |
| Pneumonia | 0.97 | 0.97 | 0.97 |
| Bronchiectasis | 0.97 | 0.95 | 0.96 |
| Healthy | 0.88 | 0.97 | 0.91 |
| URTI | 0.97 | 0.85 | 0.91 |

### External Validation (Kaggle Asthma Detection Dataset v2 — 1,211 samples)

| Metric | Score |
|---|---|
| Accuracy | 47.9% |
| Macro F1 | 0.355 |
| Weighted F1 | 0.410 |

> Domain shift is the primary failure mode — training data (ICBHI) used clinical-grade stethoscopes; external data used different recording devices and environments. Bronchiectasis recall dropped from 95% → 8% across datasets.

---

## Project Structure

```
├── preprocess.py                    # Audio loading, augmentation, MFCC extraction, train/val split
├── train.py                         # Model training with early stopping and TensorBoard logging
├── test-script.py                   # Single-file inference script
├── cross_validate_by_kaggle.py      # Batch external validation (v1)
├── external_validate_by_kaggle.py   # Batch external validation (v2, handles missing classes)
├── main-external-validation.py      # External validation entry point
├── model-check-kaggle-data.py       # Dataset inspection utility
├── model_metrics_full.txt           # Full internal + external metrics report
├── external_validation_metrics.txt  # External validation metrics summary
└── external_validation_result.csv   # Per-sample prediction results
```

---

## Setup

```bash
pip install torch torchvision torchaudio librosa scikit-learn numpy pandas tqdm tensorboard
```

---

## Usage

### 1. Preprocess Data
```bash
python preprocess.py
```
Loads `.wav` files, applies augmentation to balance classes (~200 samples each), extracts MFCC features, and saves `.npy` arrays.

### 2. Train
```bash
python train.py
```
Trains ResNet-18 for up to 40 epochs with early stopping. Saves best model weights and TensorBoard logs.

### 3. External Validation
```bash
python external_validate_by_kaggle.py
```
Runs batch inference on an external dataset using a metadata CSV and reports classification metrics.

---

## Audio Preprocessing Pipeline

| Step | Detail |
|---|---|
| Sample Rate | 44,100 Hz |
| Target Duration | 5 seconds (padded/trimmed) |
| MFCC Coefficients | 40 |
| Time Frames | 680 |
| FFT Size | 2048 |
| Hop Length | 512 |
| Augmentation | Time-stretch (±5%), pitch-shift (±1 step), Gaussian noise |

---

## Data

- Training: [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/)
- External Validation: [Kaggle Asthma Detection Dataset v2](https://www.kaggle.com/)

Original class imbalance: 792 COPD vs 16 Bronchiectasis (49.5x ratio). Resolved via cap-count augmentation to ~200 samples per class.

---

## Key Design Decisions

- MFCC treated as a 2D image → enables transfer learning from ImageNet weights
- Aggressive dropout (0.7) to reduce overfitting on a small dataset (~1,000 samples)
- Macro F1 used as primary metric (not accuracy) due to class imbalance
- External validation on a completely different dataset to measure real-world generalization
