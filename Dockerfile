# Base image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS / numpy / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        sentence-transformers==2.2.2 \
        huggingface_hub==0.11.1 \
        -r requirements.txt

# Pre-download the model so it's cached inside the container
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application
COPY . .

# Expose the Render port (Render will inject $PORT)
EXPOSE 8080

# Run the app with dynamic port from $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
