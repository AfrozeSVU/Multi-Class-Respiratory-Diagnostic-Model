import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --- CONFIGURATION ---
# IMPORTANT: --- CHANGE THIS PATH TO YOUR NEW AUDIO FILE ---
 
NEW_AUDIO_PATH='/home/ubuntu/lung_project/test_folder/P10Pneumonia52A.wav'
#NEW_AUDIO_PATH='/home/ubuntu/lung_project/test_folder/193_1b2_LL_mc_AKGC417L.WAV'


MODELS_PATH = '/home/ubuntu/lung_project/models/'
MODEL_FILENAME = 'resnet18_robust_final_best_model.pth'

# These parameters MUST match preprocess.py
SAMPLE_RATE = 44100
TARGET_DURATION = 7.86
N_MFCC = 40
MAX_MFCC_LENGTH = 680
CLASS_LABELS = np.array(['Bronchiectasis', 'COPD', 'Healthy', 'Pneumonia', 'URTI'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. MODEL DEFINITION (MUST MATCH train.py) ---
class ResNet18(nn.Module):
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


# --- 2. PREPROCESSING FUNCTION (MUST MATCH preprocess.py) ---
def extract_features(audio_data, sr, n_mfcc, max_len):
    """Extracts MFCC features using the exact same parameters as training."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

    # Padding/Truncation
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    elif mfccs.shape[1] < max_len:
        padding_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, padding_width)), mode='constant')

    return mfccs


def preprocess_new_audio(audio_path):
    """Loads, trims, and extracts features from the new audio file."""
    target_samples = int(TARGET_DURATION * SAMPLE_RATE)

    try:
        # Load audio
        data, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Trim/Pad to target duration
        trimmed_data = data[:target_samples]
        if len(trimmed_data) < target_samples:
            padding_needed = target_samples - len(trimmed_data)
            trimmed_data = np.pad(trimmed_data, (0, padding_needed), mode='constant')

        # Extract features
        features = extract_features(trimmed_data, SAMPLE_RATE, N_MFCC, MAX_MFCC_LENGTH)

        # Convert to PyTorch Tensor and 3-channel format (C, H, W)
        X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        X_final = X_tensor.repeat(1, 3, 1, 1).to(DEVICE)

        return X_final

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None


# --- 3. MAIN PREDICTION LOGIC ---
if __name__ == '__main__':
    num_classes = len(CLASS_LABELS)
    model = ResNet18(num_classes).to(DEVICE)
    model_path = os.path.join(MODELS_PATH, MODEL_FILENAME)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Model weights loaded successfully from {model_path}.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Did you run train.py?")
        exit()

    # Preprocess new audio
    input_tensor = preprocess_new_audio(NEW_AUDIO_PATH)

    if input_tensor is not None:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = torch.argmax(outputs, dim=1).item()

        predicted_class = CLASS_LABELS[predicted_index]
        confidence = probabilities[0, predicted_index].item() * 100

        # --- Display Results ---
        print("\n" + "=" * 40)
        print("         LUNG SOUND DIAGNOSIS")
        print("=" * 40)
        print(f"Input File: {os.path.basename(NEW_AUDIO_PATH)}")
        print(f"Predicted Diagnosis: \033[92m{predicted_class}\033[0m")
        print(f"Confidence: {confidence:.2f}%")
        print("=" * 40 + "\n")

        # Optional: Top 3 predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, k=3)
        print("Top 3 Predictions:")
        for i in range(3):
            class_name = CLASS_LABELS[top_k_indices[0, i].item()]
            prob = top_k_probs[0, i].item() * 100
            print(f"- {class_name}: {prob:.2f}%")
