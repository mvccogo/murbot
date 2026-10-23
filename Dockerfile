FROM python:3.11-slim

# Install system audio and opus libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    libopus-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies before the project source is copied.
RUN uv sync --frozen --no-cache --no-install-project

# Copy project source and data directories
COPY src/ ./src/
COPY data/ ./data/
RUN uv sync --frozen --no-cache

# Run the voice bot
CMD ["uv", "run", "discord-voice-bot"]
