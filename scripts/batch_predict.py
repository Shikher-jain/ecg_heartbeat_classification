# Script for batch predictions

# scripts/batch_predict.py

import numpy as np
import requests

API_URL = "http://localhost:8000/upload-file"
BATCH_FILE = "../data/processed/X.npy"

def main():
    with open(BATCH_FILE, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            print("Batch Predictions:")
            print(response.json())
        else:
            print("Error:", response.json())

if __name__ == "__main__":
    main()
