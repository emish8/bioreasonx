# Use a lightweight python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio utilities and compiler tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create audio temp folder
RUN mkdir -p data/audio_temp data/chroma_db

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Copy and set execution permissions on startup script
RUN chmod +x start.sh

# Run startup script
CMD ["./start.sh"]
