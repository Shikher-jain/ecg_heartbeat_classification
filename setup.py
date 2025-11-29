# setup.py

from setuptools import setup, find_packages
import os

# Read dependencies from requirements.txt
def parse_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    # ignore comments and empty lines
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name="ecg_heartbeat_classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=parse_requirements(),
    entry_points={
        "console_scripts": [
            "train-ecg=src.train:main",
            "preprocess-ecg=scripts.preprocess_data:main",
            "evaluate-ecg=scripts.evaluate_model:main",
            "batch-predict-ecg=scripts.batch_predict:main"
        ]
    },
    python_requires=">=3.10",
    author="Your Name",
    description="ECG Heartbeat Classification using 1D CNN",
)
