"""Ingestion module - import world data into the database.

Structure mirrors data/worlds/ and simulation/worlds/:

ingestion/worlds/        →  data/worlds/             →  simulation/worlds/
├── base/                   ├── base/                   ├── base/
│   └── earth/              │   └── earth/              │   └── earth/
└── exo/                    └── exo/                    └── exo/
    ├── real/                   ├── real/                   ├── real/
    ├── virtual/                ├── virtual/                ├── virtual/
    └── mixed/                  └── mixed/                  └── mixed/

Ingest world data → Store world data → Simulate world

Usage:
    from src.ingestion import IngestionRunner
    from src.ingestion.worlds.base.earth import NaturalEarthIngester
    
    # Ingest Earth
    runner = IngestRunner()
    await runner.ingest_earth()
    
    # Future: other worlds
    await runner.ingest_mars()      # NASA data
    await runner.ingest_roblox()    # Roblox API
"""

from .worlds.base.earth import NaturalEarthIngester, OSMIngester, GeofabrikIngester
from .runner import IngestionRunner

__all__ = [
    # Earth ingesters
    "NaturalEarthIngester",
    "OSMIngester",
    "GeofabrikIngester",
    # Runner
    "IngestionRunner",
]
