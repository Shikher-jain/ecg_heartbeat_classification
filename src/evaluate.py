# Evaluation pipeline + metrics

# src/evaluate.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model_path: str, X_path: str, y_path: str):
    """
    Load trained model and dataset, compute metrics.
    """
    model = tf.keras.models.load_model(model_path)
    X = np.load(X_path).reshape(-1, 187, 1)
    y = np.load(y_path)

    preds = model.predict(X).reshape(-1)
    preds_label = (preds > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds_label),
        "precision": precision_score(y, preds_label),
        "recall": recall_score(y, preds_label)
    }
    return metrics
