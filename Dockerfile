FROM python:3.10-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the taqsim-ai subdirectory
COPY taqsim-ai/ /app/taqsim-ai/

# Copy pyproject.toml for installation
COPY pyproject.toml /app/

# Install uv for Python package management
RUN pip install --no-cache-dir uv

# Install only the necessary dependencies for the Streamlit visualization
RUN uv pip install --system --no-cache -e .

# Make sure the /data directories exist
RUN mkdir -p /app/data/embeddings /app/data/visualizations /app/data/audio_chunks

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD streamlit run /app/taqsim-ai/streamlit/streamlit_app.py --server.port=8080 --server.address=0.0.0.0

# docker run -p 8080:8080 -v $(pwd)/data:/app/data taqsim-ai
