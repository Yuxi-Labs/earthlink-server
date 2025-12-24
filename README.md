# Earthlink Server

Python backend for Earthlink - autonomous agent training platform.

## Requirements

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL with PostGIS extension
- Redis

## Installation

Install dependencies:

```bash
pip install -e .
```

## Running

Start all services:

```bash
docker compose up -d
```

Run database migrations:

```bash
docker compose exec api alembic upgrade head
```

## Development

Start development server with hot reload:

```bash
docker compose up
```

Run tests:

```bash
pytest
```

Check code quality:

```bash
ruff check .
mypy src/
```

## API Documentation

API docs available at http://localhost:8000/docs when server is running.

## Database

Connect to PostgreSQL:

```bash
psql -h localhost -U earthlink -d earthlink
```

Run migrations:

```bash
alembic upgrade head
```

Create new migration:

```bash
alembic revision --autogenerate -m "description"
```

## Scripts

Utility scripts are in `scripts/` directory.

## Configuration

Configuration via environment variables. See `docker-compose.yml` for defaults.

## License

MIT License - Copyright (c) 2025 William Sawyerr

See LICENSE file for details.