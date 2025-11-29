# Filters, R-peak detection, segmentation, normalization

# src/preprocess.py

import numpy as np
from scipy.signal import resample

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize ECG signal to range [-1, 1]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return 2 * (signal - min_val) / (max_val - min_val) - 1

def segment_beats(signals: np.ndarray, target_length: int = 187) -> np.ndarray:
    """
    Resample all beats to a fixed length.
    - signals: 2D array (n_beats, variable_length)
    - target_length: desired length of each beat
    """
    n_beats = signals.shape[0]
    segmented = np.zeros((n_beats, target_length), dtype=np.float32)
    for i in range(n_beats):
        segmented[i] = resample(signals[i], target_length)
    return segmented

def preprocess_dataset(X: np.ndarray, y: np.ndarray, target_length: int = 187):
    """
    Normalize and segment all beats.
    Returns preprocessed X and y.
    """
    X_proc = np.array([normalize_signal(x) for x in X])
    X_proc = segment_beats(X_proc, target_length)
    y_proc = np.array(y, dtype=np.int32)
    return X_proc, y_proc
