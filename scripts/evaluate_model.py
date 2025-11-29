# Script to evaluate trained model

# scripts/evaluate_model.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_PATH = "../saved_models/ecg_cnn.h5"
X_PATH = "../data/processed/X.npy"
Y_PATH = "../data/processed/y.npy"

def main():
    # Load model and data
    model = tf.keras.models.load_model(MODEL_PATH)
    X = np.load(X_PATH).reshape(-1, 187, 1)
    y = np.load(Y_PATH)

    # Predict
    preds = model.predict(X).reshape(-1)
    preds_label = (preds > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y, preds_label)
    prec = precision_score(y, preds_label)
    rec = recall_score(y, preds_label)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

if __name__ == "__main__":
    main()
