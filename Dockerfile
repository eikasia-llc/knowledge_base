# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Enable bytecode compilation and unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies (only what's needed for wheels)
# Note: Streamlit and tiktoken have wheels, but this is good practice for future-proofing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install git - REQUIRED for GitManager in app.py
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy source code
COPY src/ ./src/

# Environment Variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV REPO_MOUNT_POINT=/tmp/knowledge_base_repo

EXPOSE 8080

CMD ["streamlit", "run", "src/app.py"]
