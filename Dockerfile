# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement list and install dependencies first (layer caching)
COPY requirements.txt .

# Install system dependencies needed for torchtext (libtorch deps, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && pip install --no-cache-dir -r requirements.txt \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchtext==0.18.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html


# Copy the entire project
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Environment variables for Flask and Gunicorn
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start Gunicorn with 4 worker threads (tune as needed)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
