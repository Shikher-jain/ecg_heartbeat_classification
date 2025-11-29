# Pydantic models for API

# app/request_schema.py

from pydantic import BaseModel
from typing import List

class SingleBeatRequest(BaseModel):
    ecg_signal: List[float]  # 1D array of floats representing a single beat

class BatchBeatRequest(BaseModel):
    beats: List[List[float]]  # 2D array of floats representing multiple beats
