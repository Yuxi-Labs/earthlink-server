"""Natural Earth ingestion - country boundaries, places, physical features.

Data source: https://www.naturalearthdata.com/

Datasets:
- Countries: 10m admin boundaries
- Populated places: Cities, towns
- Lakes, rivers, coastlines
"""

import asyncio
import json
import zipfile
from pathlib import Path
from typing import Any

import aiohttp
import fiona
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Natural Earth datasets
DATASETS = {
    "countries": {
        "url": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        "type": "regions",
        "layer": "ne_10m_admin_0_countries",
    },
    "populated_places": {
        "url": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip",
        "type": "features",
        "layer": "ne_10m_populated_places",
    },
    "lakes": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip",
        "type": "features",
        "layer": "ne_10m_lakes",
    },
    "rivers": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "type": "features",
        "layer": "ne_10m_rivers_lake_centerlines",
    },
    "coastline": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
        "type": "features",
        "layer": "ne_10m_coastline",
    },
}


class NaturalEarthIngester:
    """Ingest Natural Earth data into PostGIS."""

    def __init__(self, db_url: str, data_dir: Path | None = None):
        self.db_url = db_url
        self.data_dir = data_dir or Path("/app/data/natural_earth")
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def download_dataset(self, name: str, config: dict[str, Any]) -> Path:
        """Download and extract a Natural Earth dataset."""
        print(f"[NaturalEarth] Downloading {name}...")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / f"{name}.zip"
        extract_dir = self.data_dir / name

        if not zip_path.exists():
            async with aiohttp.ClientSession() as session:
                async with session.get(config["url"]) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Failed to download {name}: {resp.status}")
                    content = await resp.read()
                    zip_path.write_bytes(content)

        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise RuntimeError(f"No shapefile found in {extract_dir}")

        return shp_files[0]

    async def ingest_features(self, shapefile: Path, dataset_name: str) -> int:
        """Ingest point/line features from shapefile."""
        print(f"[NaturalEarth] Ingesting features from {shapefile.name}...")

        count = 0
        batch: list[dict[str, Any]] = []
        batch_size = 100

        with fiona.open(shapefile) as src:
            for feature in src:
                geom = shape(feature["geometry"])
                props = feature["properties"]

                if isinstance(geom, Point):
                    geom_wkt = geom.wkt
                elif isinstance(geom, LineString):
                    geom_wkt = geom.wkt
                else:
                    continue

                name = props.get("NAME") or props.get("name") or props.get("NAME_EN", "Unknown")
                feature_type = self._get_feature_type(dataset_name)
                metadata = self._extract_metadata(props)

                batch.append({
                    "name": name,
                    "feature_type": feature_type,
                    "geometry": geom_wkt,
                    "metadata": json.dumps(metadata),
                })

                if len(batch) >= batch_size:
                    await self._insert_features(batch)
                    count += len(batch)
                    batch = []

            if batch:
                await self._insert_features(batch)
                count += len(batch)

        print(f"[NaturalEarth] Inserted {count} features from {dataset_name}")
        return count

    async def ingest_regions(self, shapefile: Path, dataset_name: str) -> int:
        """Ingest polygon regions from shapefile."""
        print(f"[NaturalEarth] Ingesting regions from {shapefile.name}...")

        count = 0
        batch: list[dict[str, Any]] = []
        batch_size = 50

        with fiona.open(shapefile) as src:
            for feature in src:
                geom = shape(feature["geometry"])
                props = feature["properties"]

                if isinstance(geom, (Polygon, MultiPolygon)):
                    geom_wkt = geom.wkt
                else:
                    continue

                name = props.get("NAME") or props.get("ADMIN") or props.get("NAME_EN", "Unknown")
                region_type = self._get_region_type(dataset_name)
                metadata = self._extract_metadata(props)

                batch.append({
                    "name": name,
                    "region_type": region_type,
                    "geometry": geom_wkt,
                    "metadata": json.dumps(metadata),
                })

                if len(batch) >= batch_size:
                    await self._insert_regions(batch)
                    count += len(batch)
                    batch = []

            if batch:
                await self._insert_regions(batch)
                count += len(batch)

        print(f"[NaturalEarth] Inserted {count} regions from {dataset_name}")
        return count

    async def _insert_features(self, batch: list[dict[str, Any]]) -> None:
        """Batch insert features."""
        async with self.async_session() as session:
            for item in batch:
                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, feature_type, geom, properties)
                        VALUES (:name, :feature_type, ST_GeomFromText(:geometry, 4326), :metadata::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            await session.commit()

    async def _insert_regions(self, batch: list[dict[str, Any]]) -> None:
        """Batch insert regions."""
        async with self.async_session() as session:
            for item in batch:
                await session.execute(
                    text("""
                        INSERT INTO geo_regions (name, region_type, geom, properties)
                        VALUES (:name, :region_type, ST_GeomFromText(:geometry, 4326), :metadata::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            await session.commit()

    def _get_feature_type(self, dataset: str) -> str:
        """Determine feature type from dataset name."""
        if "populated_places" in dataset:
            return "city"
        elif "lake" in dataset:
            return "lake"
        elif "river" in dataset:
            return "river"
        elif "coast" in dataset:
            return "coastline"
        return "landmark"

    def _get_region_type(self, dataset: str) -> str:
        """Determine region type from dataset name."""
        if "countries" in dataset or "admin_0" in dataset:
            return "country"
        elif "states" in dataset or "admin_1" in dataset:
            return "state"
        return "administrative"

    def _extract_metadata(self, props: dict) -> dict[str, Any]:
        """Extract relevant metadata from properties."""
        metadata: dict[str, Any] = {}
        preserve = [
            "ISO_A2", "ISO_A3", "POP_EST", "GDP_MD_EST", "CONTINENT",
            "REGION_UN", "SUBREGION", "LATITUDE", "LONGITUDE", "TIMEZONE",
        ]
        for field in preserve:
            if field in props and props[field] is not None:
                metadata[field.lower()] = props[field]
        return metadata

    async def ingest_all(self, datasets: list[str] | None = None) -> dict[str, int]:
        """Ingest all (or specified) Natural Earth datasets."""
        target = datasets or list(DATASETS.keys())
        results: dict[str, int] = {}

        print(f"[NaturalEarth] Starting ingestion: {', '.join(target)}")

        for name in target:
            if name not in DATASETS:
                print(f"[NaturalEarth] Unknown dataset: {name}")
                continue

            config = DATASETS[name]
            try:
                shapefile = await self.download_dataset(name, config)
                if config["type"] == "features":
                    results[name] = await self.ingest_features(shapefile, name)
                elif config["type"] == "regions":
                    results[name] = await self.ingest_regions(shapefile, name)
            except Exception as e:
                print(f"[NaturalEarth] Error processing {name}: {e}")
                results[name] = 0

        await self.engine.dispose()
        return results

