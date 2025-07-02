FROM python:3.9-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all application files
COPY . .

# Install uv for Python package management
RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml using uv with --system flag
RUN uv pip install --system --no-cache -e .

# Make sure the /data directories exist
RUN mkdir -p /app/data/embeddings /app/data/visualizations /app/data/audio_chunks

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD streamlit run /app/streamlit/streamlit_app.py --server.port=8080 --server.address=0.0.0.0
