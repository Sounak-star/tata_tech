FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2-mesa \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install CPU-only PyTorch first to save massive amounts of space and build time
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Hugging face runs on port 7860
ENV PORT=7860
EXPOSE 7860

# Run the server
CMD ["python", "-m", "brain.server"]
