FROM python:3.9-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml for dependency installation
COPY pyproject.toml .

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Make sure the /data directories exist
RUN mkdir -p /app/data/embeddings /app/data/visualizations /app/data/audio_chunks

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD streamlit run streamlit/embedding_visualizer_streamlit.py --server.port=8080 --server.address=0.0.0.0
