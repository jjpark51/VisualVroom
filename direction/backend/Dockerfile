FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \  
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY checkpoints/ ./checkpoints/

# Create directory for debug images
RUN mkdir -p debug_images && chmod 777 debug_images

# Expose the port
EXPOSE 8888

# Run the application
CMD ["python", "main.py"]