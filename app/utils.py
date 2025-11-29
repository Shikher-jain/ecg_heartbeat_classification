# Shared utilities (preprocessing helpers, plotting, etc.)

# app/utils.py

import numpy as np
import logging
from config import LOG_FILE, LOG_LEVEL

# Setup logger
logger = logging.getLogger("ecg-api")
logger.setLevel(LOG_LEVEL)
fh = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

def preprocess_beat(arr: np.ndarray, input_length: int = None) -> np.ndarray:
    """
    Prepare a single ECG beat for model input.
    - arr: 1D numpy array
    - input_length: expected model input length
    Returns: reshaped array (1, input_length, 1)
    """
    if arr.ndim != 1:
        raise ValueError("Expected 1D array for a single beat.")
    if input_length is not None and arr.shape[0] != input_length:
        raise ValueError(f"Beat length {arr.shape[0]} does not match model input length {input_length}.")
    return arr.reshape(1, -1, 1).astype("float32")

def batch_preprocess(arr: np.ndarray, input_length: int = None) -> np.ndarray:
    """
    Prepare multiple beats for batch prediction.
    - arr: 2D numpy array (n_beats × beat_length)
    Returns: reshaped array (n_beats, beat_length, 1)
    """
    if arr.ndim != 2:
        raise ValueError("Expected 2D array for batch of beats.")
    if input_length is not None and arr.shape[1] != input_length:
        raise ValueError(f"Each beat must have length {input_length}.")
    return arr.reshape(arr.shape[0], arr.shape[1], 1).astype("float32")
