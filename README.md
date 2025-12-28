# Earthlink Server

Python backend for Earthlink - autonomous agent training platform.

## Requirements

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL with PostGIS extension
- Redis, ChromaDB, Ollama

## Quick Start

See [SETUP.md](/_docs/SETUP.md) for comprehensive setup guide.

Initialize database:

```bash
python scripts/init_database.py
```

Start all services:

```bash
docker compose up -d
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

- OpenAPI docs: http://localhost:8000/docs
- Health checks: http://localhost:8000/health/status
- WebSocket commands: ws://localhost:8000/ws/commands/stream

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

- `scripts/init_database.py` - Initialize database (extensions, tables, seed data)
- `scripts/earthlink_cli.py` - CLI tool for agent/simulation control

## CLI Usage

```bash
python scripts/earthlink_cli.py agent list
python scripts/earthlink_cli.py sim status
python scripts/earthlink_cli.py health
```

## Configuration

Configuration via environment variables. See `docker-compose.yml` for defaults.

## License

MIT License - Copyright (c) 2025 William Sawyerr

See LICENSE file for details.