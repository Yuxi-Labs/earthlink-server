"""Geofabrik ingestion - pre-processed OSM regional extracts.

Data source: https://download.geofabrik.de/

Geofabrik provides OSM data as shapefiles for entire regions.
Good for bulk imports of a specific area (country, continent).
"""

import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1-nogeom.json"


class GeofabrikIngester:
    """Ingest OSM shapefiles from Geofabrik."""

    def __init__(self, db_url: str, data_dir: Path | None = None):
        self.db_url = db_url
        self.data_dir = data_dir or Path("/app/data/geofabrik")
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self._index_cache: dict | None = None

    def get_index(self) -> dict:
        """Fetch Geofabrik index of available regions."""
        if self._index_cache is None:
            print("[Geofabrik] Fetching region index...")
            resp = requests.get(GEOFABRIK_INDEX, timeout=30)
            resp.raise_for_status()
            self._index_cache = resp.json()
            print(f"[Geofabrik] {len(self._index_cache['features'])} regions available")
        return self._index_cache

    def find_region(self, region_id: str) -> dict[str, Any]:
        """Find region metadata from index."""
        index = self.get_index()
        for feature in index["features"]:
            if feature["properties"]["id"] == region_id:
                return feature["properties"]

        # List some available regions
        print(f"[Geofabrik] Region '{region_id}' not found. Available:")
        for feature in index["features"][:20]:
            props = feature["properties"]
            if "shp" in props.get("urls", {}):
                print(f"  - {props['id']}: {props['name']}")

        raise ValueError(f"Region '{region_id}' not found")

    def list_regions(self) -> list[str]:
        """List available regions with shapefiles."""
        index = self.get_index()
        regions = []
        for feature in index["features"]:
            props = feature["properties"]
            if "shp" in props.get("urls", {}):
                regions.append(props["id"])
        return regions

    async def download_region(self, region_id: str) -> Path:
        """Download Geofabrik shapefile for region."""
        region = self.find_region(region_id)
        urls = region.get("urls", {})
        shp_url = urls.get("shp")

        if not shp_url:
            raise ValueError(f"No shapefile for region '{region_id}'")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / f"{region_id.replace('/', '_')}.zip"
        extract_dir = self.data_dir / region_id.replace("/", "_")

        if not zip_path.exists():
            print(f"[Geofabrik] Downloading {region['name']}...")
            resp = requests.get(shp_url, stream=True, timeout=600)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not extract_dir.exists():
            print(f"[Geofabrik] Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return extract_dir

    async def ingest_region(
        self,
        region_id: str,
        layers: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Ingest a Geofabrik region into the database.
        
        Args:
            region_id: Geofabrik region ID (e.g., "australia-oceania/australia")
            layers: Specific layers to import (default: all standard layers)
        """
        extract_dir = await self.download_region(region_id)
        
        default_layers = [
            "gis_osm_places_free_1",      # Cities, towns
            "gis_osm_pois_free_1",        # POIs
            "gis_osm_buildings_a_free_1", # Buildings
            "gis_osm_roads_free_1",       # Roads
            "gis_osm_waterways_free_1",   # Waterways
        ]
        target_layers = layers or default_layers
        results: dict[str, int] = {}

        for layer in target_layers:
            shp_path = extract_dir / f"{layer}.shp"
            if not shp_path.exists():
                print(f"[Geofabrik] Layer {layer} not found, skipping")
                continue

            print(f"[Geofabrik] Ingesting {layer}...")
            try:
                count = await self._ingest_shapefile(shp_path, layer)
                results[layer] = count
            except Exception as e:
                print(f"[Geofabrik] Error ingesting {layer}: {e}")
                results[layer] = 0

        await self.engine.dispose()
        return results

    async def _ingest_shapefile(self, shp_path: Path, layer: str) -> int:
        """Ingest a single shapefile layer."""
        gdf = gpd.read_file(shp_path)
        count = 0

        # Determine target table based on layer name
        if "places" in layer:
            table = "places"
            name_col = "name"
            type_col = "fclass"
        elif "pois" in layer:
            table = "pois"
            name_col = "name"
            type_col = "fclass"
        elif "buildings" in layer:
            table = "buildings"
            name_col = "name"
            type_col = "type"
        elif "roads" in layer:
            table = "roads"
            name_col = "name"
            type_col = "fclass"
        elif "waterways" in layer:
            table = "water_features"
            name_col = "name"
            type_col = "fclass"
        else:
            print(f"[Geofabrik] Unknown layer type: {layer}")
            return 0

        async with self.async_session() as session:
            batch_size = 1000
            batch = []

            for _, row in gdf.iterrows():
                name = row.get(name_col)
                if not name or str(name) == "nan":
                    name = "Unknown"

                feature_type = row.get(type_col, "unknown")
                geom = row.geometry

                if geom is None:
                    continue

                batch.append({
                    "name": str(name),
                    "type": str(feature_type),
                    "geom": geom.wkt,
                })

                if len(batch) >= batch_size:
                    await self._insert_batch(session, table, batch)
                    count += len(batch)
                    batch = []

            if batch:
                await self._insert_batch(session, table, batch)
                count += len(batch)

            await session.commit()

        print(f"[Geofabrik] Inserted {count} records into {table}")
        return count

    async def _insert_batch(
        self,
        session: AsyncSession,
        table: str,
        batch: list[dict[str, Any]],
    ) -> None:
        """Insert batch of records into appropriate table."""
        for item in batch:
            if table == "places":
                await session.execute(
                    text("""
                        INSERT INTO places (name, place_type, geom)
                        VALUES (:name, :type, ST_GeomFromText(:geom, 4326))
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            elif table == "pois":
                await session.execute(
                    text("""
                        INSERT INTO pois (name, poi_type, geom)
                        VALUES (:name, :type, ST_GeomFromText(:geom, 4326))
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            elif table == "buildings":
                await session.execute(
                    text("""
                        INSERT INTO buildings (name, building_type, footprint)
                        VALUES (:name, :type, ST_GeomFromText(:geom, 4326))
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            elif table == "roads":
                await session.execute(
                    text("""
                        INSERT INTO roads (name, road_type, geom)
                        VALUES (:name, :type, ST_GeomFromText(:geom, 4326))
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )
            elif table == "water_features":
                await session.execute(
                    text("""
                        INSERT INTO water_features (name, water_type, geom)
                        VALUES (:name, :type, ST_GeomFromText(:geom, 4326))
                        ON CONFLICT DO NOTHING
                    """),
                    item,
                )

