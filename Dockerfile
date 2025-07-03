FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for aeneas
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak \
    libespeak-dev \
    libsndfile1 \
    libmagic1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix for aeneas: avoid numpy >= 1.23 and setuptools >= 60
RUN pip install --no-cache-dir "numpy<1.23" "setuptools<60" && \
    echo "✅ Installed versions:" && \
    python -m pip show numpy setuptools

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose the default Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]