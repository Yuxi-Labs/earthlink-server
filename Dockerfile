# Base stage - common dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libexpat1 \
    osm2pgsql \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project metadata
COPY pyproject.toml README.md ./

# Install dependencies first (cached layer)
RUN uv pip install --system .[dev]

# Development stage - includes tests and dev dependencies
FROM base AS dev

# Copy all source code and tests
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage - minimal, no tests
FROM base AS prod

# Copy only source code (no tests)
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
