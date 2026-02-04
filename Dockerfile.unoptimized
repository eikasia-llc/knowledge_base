# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Define mount point for ephemeral repository storage
ENV REPO_MOUNT_POINT=/tmp/knowledge_base_repo

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "src/app.py"]
