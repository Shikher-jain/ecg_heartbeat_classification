# Training pipeline (train/val split, model saving)

# src/train.py

import numpy as np
from sklearn.model_selection import train_test_split
from model import build_cnn_model
from preprocess import preprocess_dataset
import os
import tensorflow as tf

# Paths
RAW_DATA_PATH = "../data/processed/X.npy"
RAW_LABEL_PATH = "../data/processed/y.npy"
MODEL_SAVE_PATH = "../saved_models/ecg_cnn.h5"

def main():
    # Load processed dataset
    X = np.load(RAW_DATA_PATH)
    y = np.load(RAW_LABEL_PATH)

    # Preprocess
    X, y = preprocess_dataset(X, y, target_length=187)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    model = build_cnn_model(input_length=187)

    # Train
    history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                        validation_data=(X_val, y_val))

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
