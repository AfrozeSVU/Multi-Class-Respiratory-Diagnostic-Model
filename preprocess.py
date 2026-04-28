 # preprocess_updated.py
import os
import copy
import librosa
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/lung_project/KAGGLE/dataset/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2'
PROCESSED_DATA_PATH = '/home/ubuntu/lung_project/KAGGLE/set2/processed_data/'

TARGET_DURATION = 5            # seconds
SAMPLE_RATE = 44100
N_MFCC = 40
MAX_MFCC_LENGTH = int(TARGET_DURATION * SAMPLE_RATE / 512)
MIN_SAMPLES_FOR_CLASS = 2
MAX_AUGMENTATION_COUNT = 400   # Target per class

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- UTILITY FUNCTIONS ---
def load_all_audio_files(data_path):
    audio_paths, labels = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                label = os.path.basename(root).lower()
                file_path = os.path.join(root, file)
                audio_paths.append(file_path)
                labels.append(label)
    unique_classes = sorted(list(set(labels)))
    print(f"\nFound {len(audio_paths)} audio files across {len(unique_classes)} classes: {unique_classes}")
    return list(zip(audio_paths, labels))


def load_and_trim_audio(file_label_pairs, sr, target_duration):
    all_audio_data = {}
    target_samples = int(target_duration * sr)
    for filepath, label in tqdm(file_label_pairs, desc="Loading and Trimming Audio"):
        try:
            data, sr_load = librosa.load(filepath, sr=sr)
            if len(data) < target_samples:
                data = np.pad(data, (0, target_samples - len(data)), mode='constant')
            else:
                data = data[:target_samples]
            filename = os.path.basename(filepath)
            all_audio_data[filename] = {'data': data, 'sample_rate': sr_load, 'diagnosis': label}
        except Exception as e:
            print(f"Could not process {filepath}: {e}")
            continue
    return all_audio_data


def augment_data(data, sr, rate, pitch, noise_factor, target_duration):
    target_samples = int(target_duration * sr)
    data_padded = np.pad(data, (0, target_samples), mode='constant')
    stretched_data = librosa.effects.time_stretch(data_padded, rate=rate)
    shifted_data = librosa.effects.pitch_shift(stretched_data, sr=sr, n_steps=pitch)
    shifted_data = shifted_data[:target_samples]
    noise = np.random.uniform(low=-1, high=1, size=shifted_data.shape) * noise_factor
    final_data = shifted_data + noise
    return final_data[:target_samples]


def extract_features(audio_data, sr, n_mfcc, max_len):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    elif mfccs.shape[1] < max_len:
        padding_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, padding_width)), mode='constant')
    return mfccs


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    file_label_pairs = load_all_audio_files(DATA_PATH)
    trimmed_audio_raw = load_and_trim_audio(file_label_pairs, SAMPLE_RATE, TARGET_DURATION)

    if not trimmed_audio_raw:
        print(" No audio data found. Check DATA_PATH or subfolders.")
        exit()

    # Keep only classes with enough samples
    raw_counts = Counter(info['diagnosis'] for info in trimmed_audio_raw.values())
    classes_to_keep = [cls for cls, count in raw_counts.items() if count >= MIN_SAMPLES_FOR_CLASS]
    trimmed_audio = {fn: info for fn, info in trimmed_audio_raw.items() if info['diagnosis'] in classes_to_keep}

    print("\nInitial Trimmed Dataset Counts:", Counter(info['diagnosis'] for info in trimmed_audio.values()))

    # --- 1. Split original samples into train & validation ---
    filenames = list(trimmed_audio.keys())
    labels = [trimmed_audio[fn]['diagnosis'] for fn in filenames]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(filenames, labels))

    original_train = {filenames[i]: trimmed_audio[filenames[i]] for i in train_idx}
    val_data = {filenames[i]: trimmed_audio[filenames[i]] for i in val_idx}  # validation only real

    print("\nOriginal training counts:", Counter(info['diagnosis'] for info in original_train.values()))
    print("Validation counts:", Counter(info['diagnosis'] for info in val_data.values()))

    # --- 2. Augment training data ---
    augmented_train = copy.deepcopy(original_train)
    for target_class in classes_to_keep:
        class_samples = [info for info in original_train.values() if info['diagnosis'] == target_class]
        current_count = len(class_samples)
        if current_count < MAX_AUGMENTATION_COUNT:
            augment_needed = MAX_AUGMENTATION_COUNT - current_count
            aug_count = 0
            num_per_original = int(np.ceil(augment_needed / current_count))
            for i in range(num_per_original):
                for j, sample_info in enumerate(class_samples):
                    if aug_count >= augment_needed:
                        break
                    rate = np.random.choice([0.95, 1.05])
                    pitch = np.random.choice([-1, 1])
                    noise_factor = np.random.uniform(0.0005, 0.002)
                    new_data = augment_data(sample_info['data'], SAMPLE_RATE, rate, pitch, noise_factor, TARGET_DURATION)
                    new_filename = f"AUG_{target_class}_{i}_{j}.wav"
                    augmented_train[new_filename] = {'data': new_data, 'sample_rate': SAMPLE_RATE, 'diagnosis': target_class}
                    aug_count += 1
                if aug_count >= augment_needed:
                    break

    print("\nTotal training samples after augmentation:", len(augmented_train))

    # --- 3. Extract MFCC features ---
    def features_labels_dict(data_dict):
        X, Y = [], []
        for info in data_dict.values():
            feat = extract_features(info['data'], info['sample_rate'], N_MFCC, MAX_MFCC_LENGTH)
            X.append(feat)
            Y.append(info['diagnosis'])
        return np.array(X), np.array(Y)

    X_train, Y_train_text = features_labels_dict(augmented_train)
    X_val, Y_val_text = features_labels_dict(val_data)

    # --- 4. Encode labels ---
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train_text)
    Y_val = le.transform(Y_val_text)
    print(f"\nLabel encoding map: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Training feature matrix: {X_train.shape}, Validation feature matrix: {X_val.shape}")

    # --- 5. Save processed datasets ---
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_train.npy'), Y_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_val.npy'), Y_val)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_label_classes.npy'), le.classes_)

    print(f"\n Saved processed train/validation splits to: {PROCESSED_DATA_PATH}")
