FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# Use a lightweight Python base
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system deps (optional, if you need them for pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Fix huggingface_hub / sentence-transformers import issue
# Pin to versions that are compatible
RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    huggingface_hub==0.11.1 \
    -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Render will use (not strictly required but good practice)
EXPOSE 10000

# Start Uvicorn with dynamic port from $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
