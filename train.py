# train_model_updated.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = '/home/ubuntu/lung_project/KAGGLE/set2/processed_data/'
MODELS_PATH = '/home/ubuntu/lung_project/KAGGLE/set2/models/'
METRICS_PATH = '/home/ubuntu/lung_project/KAGGLE/set2/metrics/'
TENSORBOARD_PATH = '/home/ubuntu/lung_project/KAGGLE/set2/tensorboard_logs/'
RUN_VERSION = 'resnet18_robust_final'

BATCH_SIZE = 64
N_EPOCHS = 40
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
EARLY_STOPPING_PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- DATASET ---
class MFCCDataset(Dataset):
    def __init__(self, X, Y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.X = X_tensor.repeat(1, 3, 1, 1)  # 3-channel for ResNet
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_data(processed_path):
    """Load preprocessed train/val datasets."""
    X_train = np.load(os.path.join(processed_path, 'X_train.npy'))
    Y_train = np.load(os.path.join(processed_path, 'Y_train.npy'))
    X_val = np.load(os.path.join(processed_path, 'X_val.npy'))
    Y_val = np.load(os.path.join(processed_path, 'Y_val.npy'))
    classes = np.load(os.path.join(processed_path, 'Y_label_classes.npy'))
    print(f"Train: {X_train.shape}, {Y_train.shape}; Val: {X_val.shape}, {Y_val.shape}")
    return X_train, Y_train, X_val, Y_val, classes


# --- MODEL ---
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


# --- EVALUATION ---
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    total_loss, all_labels, all_predictions = 0.0, [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset) if criterion else 0
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    report = classification_report(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    return avg_loss, f1, acc, report, cm


# --- TRAINING ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path, writer):
    best_val_f1 = 0.0
    epochs_no_improve = 0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_f1, val_acc, _, _ = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_f1)

        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Validation", val_loss, epoch+1)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch+1)
        writer.add_scalar("F1/Validation", val_f1, epoch+1)

        print(f"\n--- Epoch {epoch+1} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model (F1: {best_val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    writer.close()
    print("Training complete.")
    return best_val_f1


# --- MAIN ---
if __name__ == '__main__':
    start_time = time.time()

    # Load preprocessed data
    X_train, Y_train, X_val, Y_val, classes = load_data(PROCESSED_DATA_PATH)
    NUM_CLASSES = len(classes)

    # Class weights for imbalance
    class_counts = np.bincount(Y_train)
    weights = len(Y_train) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class Weights: {weights}")

    # DataLoaders
    train_loader = DataLoader(MFCCDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MFCCDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = ResNet18(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)

    best_model_path = os.path.join(MODELS_PATH, f'{RUN_VERSION}_best_model.pth')
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_PATH, RUN_VERSION))

    print("\n--- Starting Training ---")
    best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, best_model_path, writer)

    # Evaluate final model
    best_model = ResNet18(NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path))
    _, final_f1, final_acc, report, cm = evaluate_model(best_model, val_loader)

    # Save metrics
    metrics_path = os.path.join(METRICS_PATH, f'metrics_{RUN_VERSION}.txt')
    runtime = time.time() - start_time
    with open(metrics_path, 'w') as f:
        f.write(f"--- Run: {RUN_VERSION} ---\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
        f.write(f"Best F1: {best_f1:.4f}\nFinal Acc: {final_acc:.4f}\n\n{report}")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
        f.write(f"\nClass Mapping: {dict(zip(range(NUM_CLASSES), classes))}")

    print(f"\n--- Final Results ({RUN_VERSION}) ---")
    print(f"Final F1: {final_f1:.4f}, Accuracy: {final_acc:.4f}")
    print(f"Metrics saved to {metrics_path}")
    print(f"TensorBoard logs saved to {TENSORBOARD_PATH}")
    print("Training and Evaluation complete.")
