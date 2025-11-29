# tests/test_preprocess.py

import numpy as np
from src.preprocess import normalize_signal, segment_beats

def test_normalize_signal():
    arr = np.array([0, 2, 4, 6])
    norm = normalize_signal(arr)
    assert np.all(norm >= -1) and np.all(norm <= 1)

def test_segment_beats():
    arr = np.array([[1,2,3,4],[5,6,7,8]])
    seg = segment_beats(arr, target_length=4)
    assert seg.shape == (2,4)
