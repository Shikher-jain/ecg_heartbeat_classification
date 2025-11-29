# Streamlit frontend

# app/streamlit_app.py

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import io

API_URL = "http://localhost:8000"  # FastAPI backend URL

st.set_page_config(page_title="ECG Heartbeat Classification", layout="wide")

st.title("ECG Heartbeat Classification Dashboard")
st.write("Upload your ECG beats and get predictions (Normal / Arrhythmia)")

# ---- Single Beat Upload ----
st.subheader("Single Beat Prediction")
single_file = st.file_uploader("Upload a single ECG beat (.npy)", type=["npy"], key="single")

if single_file:
    try:
        arr = np.load(single_file, allow_pickle=False)
        st.write("Shape:", arr.shape)

        # Plot the beat
        st.line_chart(arr)

        # Send to FastAPI
        response = requests.post(f"{API_URL}/upload-file", files={"file": single_file})
        if response.status_code == 200:
            st.success("Prediction:")
            st.json(response.json())
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to process file: {e}")

# ---- Batch Beat Upload ----
st.subheader("Batch Prediction")
batch_file = st.file_uploader("Upload multiple ECG beats (.npy)", type=["npy"], key="batch")

if batch_file:
    try:
        arr = np.load(batch_file, allow_pickle=False)
        st.write("Batch shape:", arr.shape)
        for i, beat in enumerate(arr):
            st.write(f"Beat {i+1}")
            st.line_chart(beat)

        # Send to FastAPI
        response = requests.post(f"{API_URL}/upload-file", files={"file": batch_file})
        if response.status_code == 200:
            st.success("Batch Predictions:")
            st.json(response.json())
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to process batch file: {e}")

# ---- About ----
st.sidebar.title("About")
st.sidebar.info(
    """
This dashboard allows you to upload **ECG beats** and get predictions for Normal vs Arrhythmia.
Backend powered by **FastAPI** and **1D CNN** model.
"""
)
