# Base stage - common dependencies
FROM python:3.13-slim AS base

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
RUN pip install --no-cache-dir uv

# Copy project metadata
COPY pyproject.toml README.md ./

# Install dependencies first (cached layer)
RUN uv pip install --system .[dev] && \
    rm -rf /root/.cache/pip /root/.cache/uv

# Development stage - includes tests and dev dependencies
FROM base AS dev

# Copy all source code and tests
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage - minimal, no tests
FROM python:3.13-slim AS prod

WORKDIR /app

# Install build dependencies + runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project metadata
COPY pyproject.toml README.md ./

# Install ONLY production dependencies (no dev/test packages)
# PyTorch is ~8GB but required for agent neural networks
RUN uv pip install --system --no-cache . && \
    pip uninstall -y uv && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /root/.cache/pip /root/.cache/uv /tmp/*

# Copy only source code (no tests)
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
