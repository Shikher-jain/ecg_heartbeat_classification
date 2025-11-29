# Configuration variables (paths, model, params)

# app/config.py

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/ecg_cnn.h5")

# Model input length (optional; inferred from model)
INPUT_LENGTH = None

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "../logs/ecg_api.log")

# CORS
ALLOWED_ORIGINS = ["*"]  # Change in production to specific domains

# FastAPI settings
API_TITLE = "ECG Heartbeat Classification API"
API_VERSION = "1.0"
API_HOST = "0.0.0.0"
API_PORT = 8000
