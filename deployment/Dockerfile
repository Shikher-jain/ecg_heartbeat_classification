# deployment/Dockerfile

# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
