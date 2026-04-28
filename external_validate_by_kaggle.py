import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import librosa
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import softmax
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ======================================================================
# EXTERNAL VALIDATION SCRIPT (BATCH PREDICTION)
# ======================================================================

# --- 1. CONFIGURATION (CRITICAL: MATCHES SINGLE-FILE TEST) ---
# ----------------------------------------------------------------------
METADATA_INPUT_FILE = '/home/ubuntu/lung_project/KAGGLE/external_validation_metadata.csv'
MODEL_WEIGHTS_PATH = '/home/ubuntu/lung_project/models/resnet18_robust_final_best_model.pth'

# Feature Extraction Parameters (MUST match single-file test script)
SAMPLE_RATE = 44100
TARGET_DURATION = 7.86
N_MFCC = 40
MAX_MFCC_LENGTH = 680

# IMPORTANT: CLASS_LABELS must be in the exact order the model expects (numerical index 0, 1, 2, ...)
# This order is taken from your model's Class Mapping: {0: 'Bronchiectasis', 1: 'COPD', 2: 'Healthy', 3: 'Pneumonia', 4: 'URTI'}
CLASS_LABELS = np.array(['Bronchiectasis', 'COPD', 'Healthy', 'Pneumonia', 'URTI'])
NUM_CLASSES = len(CLASS_LABELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION (IDENTICAL TO train.py) ---
# ----------------------------------------------------------------------
class ResNet18(nn.Module):
    """ResNet18 model adapted for 1-channel spectrogram input via 3-channel repetition."""
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- 3. PREPROCESSING FUNCTION (HARMONIZED WITH SINGLE-FILE TEST) ---
# ----------------------------------------------------------------------
def extract_features(audio_data, sr, n_mfcc, max_len):
    """Extracts and standardizes MFCC features using the exact same parameters."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

    # Padding/Truncation for fixed length
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    elif mfccs.shape[1] < max_len:
        padding_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, padding_width)), mode='constant')

    return mfccs

def preprocess_audio_to_tensor(audio_path):
    """Loads, processes, and prepares the audio tensor for the ResNet model."""
    target_samples = int(TARGET_DURATION * SAMPLE_RATE)

    # 1. Load, Trim/Pad Audio
    data, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    trimmed_data = data[:target_samples]
    if len(trimmed_data) < target_samples:
        padding_needed = target_samples - len(trimmed_data)
        trimmed_data = np.pad(trimmed_data, (0, padding_needed), mode='constant')

    # 2. Extract Features
    features = extract_features(trimmed_data, SAMPLE_RATE, N_MFCC, MAX_MFCC_LENGTH)

    # 3. Convert to PyTorch Tensor, add batch/channel dims (1, 1, H, W)
    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 4. Replicate 3 times for ResNet input (1, 3, H, W)
    X_final = X_tensor.repeat(1, 3, 1, 1).to(DEVICE)

    return X_final

# --- 4. MAIN EXECUTION AND PREDICTION LOGIC ---
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("="*80)
    print("| Starting External Validation Script (Batch Processing) |")
    print("="*80)

    # Initialize Model
    try:
        model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        model.eval()
        print(f"[LOG] Model weights loaded successfully from {MODEL_WEIGHTS_PATH}.")
    except Exception as e:
        print(f"[ERROR] Failed to load trained model: {e}")
        exit()

    # Load the external validation metadata
    try:
        df_validation = pd.read_csv(METADATA_INPUT_FILE)
        print(f"[LOG] Loaded {len(df_validation)} samples for external validation from {METADATA_INPUT_FILE}.")
    except Exception as e:
        print(f"[ERROR] Failed to load metadata CSV: {METADATA_INPUT_FILE}. Error: {e}")
        exit()

    # --- Prediction Loop ---
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for index, row in tqdm(df_validation.iterrows(), total=len(df_validation), desc="[PREDICT] Processing Samples"):
            file_path = row['Local_Path']
            true_label_text = row['True_Label']

            try:
                # A. Preprocess and create tensor
                X_tensor = preprocess_audio_to_tensor(file_path)

                # B. Predict
                outputs = model(X_tensor)
                _, predicted_class_index = torch.max(outputs.data, 1)

                # C. Convert prediction index back to text label
                predicted_label_text = CLASS_LABELS[predicted_class_index.item()]

                true_labels.append(true_label_text)
                predicted_labels.append(predicted_label_text)

            except Exception as e:
                # Append the true label so we know which sample failed
                true_labels.append(true_label_text)
                predicted_labels.append("PROCESSING_ERROR")

    # Cleanup results and prepare for metrics
    df_results = pd.DataFrame({'True_Label': true_labels, 'Predicted_Label': predicted_labels})

    # Remove rows that failed processing (if any)
    df_results = df_results[df_results['Predicted_Label'] != 'PROCESSING_ERROR']

    # --- FINAL VALIDATION CHECK (Critical for fixing ValueError) ---
    # Filter the list of labels to only those present in the actual results (y_true)
    y_true = df_results['True_Label'].values
    y_pred = df_results['Predicted_Label'].values

    # Calculate effective labels present in y_true, sorted by CLASS_LABELS order
    effective_labels = np.intersect1d(CLASS_LABELS, np.unique(y_true))

    if len(y_true) == 0:
        print("[ERROR] No valid samples were processed. External validation failed.")
        exit()

    # ----------------------------------------------------------------------
    # STEP 5: CALCULATE AND REPORT METRICS
    # ----------------------------------------------------------------------
    print("\n--- STEP 5: Calculating External Validation Metrics ---")

    # 1. Classification Report - Use 'effective_labels' to avoid ValueError from missing classes (like URTI)
    print("\n[REPORT] Detailed Classification Metrics (Generalization Score):")
    report = classification_report(y_true, y_pred, labels=effective_labels, zero_division=0)
    print(report)

    # 2. Confusion Matrix - Use 'effective_labels' for the same reason
    conf_matrix = confusion_matrix(y_true, y_pred, labels=effective_labels)
    print("\n[REPORT] Confusion Matrix (Rows=True Label, Columns=Predicted Label):")
    print("Class Order (for Matrix):", effective_labels)
    print(conf_matrix)

    # 3. Save Results
    results_path = '/home/ubuntu/lung_project/external_validation_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n[OUTPUT] Full results saved to: {results_path}")

    print("="*80)
    print("| External Validation Complete. Analyze the Confusion Matrix! |")
    print("="*80)