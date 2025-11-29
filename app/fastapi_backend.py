# Main FastAPI server


# app/fastapi_backend.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import io
from typing import Optional
import logging

from config import *
from utils import preprocess_beat, batch_preprocess, logger
from request_schema import SingleBeatRequest, BatchBeatRequest

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model: Optional[tf.keras.Model] = None

@app.on_event("startup")
def load_model():
    global model, INPUT_LENGTH
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        try:
            INPUT_LENGTH = model.input_shape[1]
        except Exception:
            INPUT_LENGTH = None
        logger.info(f"Model loaded successfully from {MODEL_PATH}, input_length={INPUT_LENGTH}")
    except Exception as e:
        model = None
        logger.error(f"Failed to load model at startup: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "input_length": INPUT_LENGTH}

@app.get("/version")
def version():
    return {"api_version": API_VERSION, "model_loaded": model is not None}

@app.post("/predict/beat")
async def predict_beat(request: SingleBeatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        arr = np.array(request.ecg_signal, dtype="float32")
        x = preprocess_beat(arr, INPUT_LENGTH)
        pred = float(model.predict(x)[0][0])
        label = "Arrhythmia" if pred > 0.5 else "Normal"
        return {"prediction_score": pred, "label": label}
    except Exception as e:
        logger.error(f"Error predicting beat: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchBeatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        arr = np.array(request.beats, dtype="float32")
        x = batch_preprocess(arr, INPUT_LENGTH)
        preds = model.predict(x).reshape(-1).tolist()
        labels = ["Arrhythmia" if p > 0.5 else "Normal" for p in preds]
        return {"predictions": [{"score": float(s), "label": l} for s, l in zip(preds, labels)]}
    except Exception as e:
        logger.error(f"Error predicting batch: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a .npy file (1D or 2D). Returns prediction(s).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        content = await file.read()
        arr = np.load(io.BytesIO(content), allow_pickle=False)
        if arr.ndim == 1:
            x = preprocess_beat(arr, INPUT_LENGTH)
            pred = float(model.predict(x)[0][0])
            label = "Arrhythmia" if pred > 0.5 else "Normal"
            return {"prediction_score": pred, "label": label}
        elif arr.ndim == 2:
            x = batch_preprocess(arr, INPUT_LENGTH)
            preds = model.predict(x).reshape(-1).tolist()
            labels = ["Arrhythmia" if p > 0.5 else "Normal" for p in preds]
            return {"predictions": [{"score": float(s), "label": l} for s, l in zip(preds, labels)]}
        else:
            raise HTTPException(status_code=400, detail="Unsupported array dimensions.")
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=400, detail=str(e))

