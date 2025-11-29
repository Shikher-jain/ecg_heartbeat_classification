# CLI script to preprocess raw ECG data

# scripts/preprocess_data.py

import numpy as np
from src.preprocess import preprocess_dataset
import os

RAW_X_PATH = "../data/raw/X_raw.npy"
RAW_Y_PATH = "../data/raw/y_raw.npy"
PROC_X_PATH = "../data/processed/X.npy"
PROC_Y_PATH = "../data/processed/y.npy"

def main():
    # Load raw data
    X = np.load(RAW_X_PATH, allow_pickle=True)
    y = np.load(RAW_Y_PATH, allow_pickle=True)

    # Preprocess
    X_proc, y_proc = preprocess_dataset(X, y, target_length=187)

    # Save processed data
    os.makedirs(os.path.dirname(PROC_X_PATH), exist_ok=True)
    np.save(PROC_X_PATH, X_proc)
    np.save(PROC_Y_PATH, y_proc)

    print(f"Processed data saved: {PROC_X_PATH}, {PROC_Y_PATH}")

if __name__ == "__main__":
    main()
