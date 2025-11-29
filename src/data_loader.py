# Load raw or processed ECG data

# src/data_loader.py

import numpy as np
import os

def load_raw_data(x_path: str, y_path: str):
    """
    Load raw ECG data from .npy files.
    Returns X, y as numpy arrays.
    """
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Data files not found.")
    
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def load_processed_data(x_path: str, y_path: str):
    """
    Load preprocessed ECG data from .npy files.
    """
    return load_raw_data(x_path, y_path)
