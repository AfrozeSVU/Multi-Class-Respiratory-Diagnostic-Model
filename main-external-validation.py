 
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# The local directory where you downloaded the external raw audio files
# from S3 in Step 1.1.
EXTERNAL_RAW_DATA_ROOT = '/home/ubuntu/lung_project/KAGGLE/dataset/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2/'
# The file name for your new metadata CSV
METADATA_OUTPUT_FILE = '/home/ubuntu/lung_project/KAGGLE/external_validation_metadata.csv'
# The 5 classes you expect to find
CLASSES = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']


# ======================================================================
# METADATA GENERATION SCRIPT
# ======================================================================

print("="*80)
print("| Starting External Validation Metadata Generation |")
print("="*80)

metadata_list = []
total_files = 0

for class_name in CLASSES:
    class_path = os.path.join(EXTERNAL_RAW_DATA_ROOT, class_name)

    if not os.path.exists(class_path):
        print(f"[WARNING] Class directory not found: {class_path}. Skipping.")
        continue

    # List all audio files in the directory
    files = [f for f in os.listdir(class_path) if f.endswith(('.wav', '.mp3', '.ogg'))]

    print(f"[LOG] Found {len(files)} files for class: {class_name}")

    for file_name in files:
        full_file_path = os.path.join(class_path, file_name)

        # Verify the file is actually an audio file and not a zero-byte file
        try:
            # Quick check to ensure file is readable/valid before adding to list
            librosa.get_duration(path=full_file_path)

            metadata_list.append({
                'Local_Path': full_file_path,
                'True_Label': class_name
            })
            total_files += 1
        except Exception as e:
            print(f"[ERROR] Skipping file {file_name}. Failed to read audio: {e}")

# Create and save the final DataFrame
df_metadata = pd.DataFrame(metadata_list)

# Final check of the generated data distribution
print("\n[SUMMARY] External Validation Set Distribution:")
print(df_metadata['True_Label'].value_counts().to_string())

# Save to CSV
df_metadata.to_csv(METADATA_OUTPUT_FILE, index=False)

print(f"\n[OUTPUT] Metadata saved to: {METADATA_OUTPUT_FILE}")
print(f"[OUTPUT] Total files added to validation set: {total_files}")
print("="*80)