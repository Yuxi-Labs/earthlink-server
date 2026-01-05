"""OpenStreetMap ingestion via Overpass API.

Queries OSM data directly from Overpass API - no file downloads needed.
Good for targeted queries (cities, POIs in specific areas).
"""

import json
from typing import Any

import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

OVERPASS_API = "https://overpass-api.de/api/interpreter"

# Pre-defined Overpass QL queries
QUERIES = {
    "major_cities": """
        [out:json][timeout:120];
        (
          node["place"="city"]["population">"100000"];
        );
        out body;
    """,
    "capital_cities": """
        [out:json][timeout:60];
        (
          node["place"="city"]["capital"];
        );
        out body;
    """,
    "airports": """
        [out:json][timeout:120];
        (
          node["aeroway"="aerodrome"]["iata"];
        );
        out body;
    """,
}


class OSMIngester:
    """Ingest OpenStreetMap data via Overpass API."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def query_overpass(self, query: str) -> dict[str, Any]:
        """Execute Overpass API query."""
        async with aiohttp.ClientSession() as session:
            async with session.post(OVERPASS_API, data={"data": query}) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Overpass API error: {resp.status}")
                return await resp.json()

    async def ingest_cities(self) -> int:
        """Ingest major cities and capitals from OSM."""
        print("[OSM] Fetching major cities (pop > 100k)...")
        data = await self.query_overpass(QUERIES["major_cities"])
        count = 0

        async with self.async_session() as session:
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue

                tags = element.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue

                lat = element.get("lat")
                lon = element.get("lon")
                properties = json.dumps({
                    "population": tags.get("population"),
                    "place": tags.get("place"),
                    "osm_id": element.get("id"),
                    "country": tags.get("is_in:country"),
                })

                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'city', :properties::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": name, "lon": lon, "lat": lat, "properties": properties},
                )
                count += 1

            await session.commit()

        print(f"[OSM] Ingested {count} major cities")

        # Also fetch capitals
        print("[OSM] Fetching capital cities...")
        data = await self.query_overpass(QUERIES["capital_cities"])
        capital_count = 0

        async with self.async_session() as session:
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue

                tags = element.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue

                lat = element.get("lat")
                lon = element.get("lon")
                properties = json.dumps({
                    "capital": tags.get("capital"),
                    "population": tags.get("population"),
                    "place": tags.get("place"),
                    "osm_id": element.get("id"),
                })

                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'city', :properties::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": name, "lon": lon, "lat": lat, "properties": properties},
                )
                capital_count += 1

            await session.commit()

        print(f"[OSM] Ingested {capital_count} capital cities")
        return count + capital_count

    async def ingest_airports(self) -> int:
        """Ingest major airports with IATA codes."""
        print("[OSM] Fetching airports...")
        data = await self.query_overpass(QUERIES["airports"])
        count = 0

        async with self.async_session() as session:
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue

                tags = element.get("tags", {})
                name = tags.get("name", tags.get("iata", "Unknown"))
                lat = element.get("lat")
                lon = element.get("lon")

                properties = json.dumps({
                    "iata": tags.get("iata"),
                    "icao": tags.get("icao"),
                    "osm_id": element.get("id"),
                })

                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'airport', :properties::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": name, "lon": lon, "lat": lat, "properties": properties},
                )
                count += 1

            await session.commit()

        print(f"[OSM] Ingested {count} airports")
        return count

    async def ingest_region(
        self,
        bbox: tuple[float, float, float, float],
        feature_types: list[str] | None = None,
    ) -> int:
        """
        Ingest OSM features within a bounding box.
        
        Args:
            bbox: (south, west, north, east) coordinates
            feature_types: List of OSM keys to fetch (e.g., ["amenity", "shop"])
        """
        south, west, north, east = bbox
        types = feature_types or ["amenity", "tourism", "historic"]

        query_parts = []
        for t in types:
            query_parts.append(f'node["{t}"]({south},{west},{north},{east});')

        query = f"""
            [out:json][timeout:180];
            (
                {chr(10).join(query_parts)}
            );
            out body;
        """

        print(f"[OSM] Fetching features in bbox {bbox}...")
        data = await self.query_overpass(query)
        count = 0

        async with self.async_session() as session:
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue

                tags = element.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue

                lat = element.get("lat")
                lon = element.get("lon")

                # Determine feature type from tags
                feature_type = "poi"
                for t in types:
                    if t in tags:
                        feature_type = tags[t]
                        break

                properties = json.dumps({
                    "osm_id": element.get("id"),
                    "tags": tags,
                })

                await session.execute(
                    text("""
                        INSERT INTO pois (name, geom, poi_type, category, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :feature_type, :category, :properties::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "name": name,
                        "lon": lon,
                        "lat": lat,
                        "feature_type": feature_type,
                        "category": types[0] if types else "poi",
                        "properties": properties,
                    },
                )
                count += 1

            await session.commit()

        print(f"[OSM] Ingested {count} POIs from region")
        return count

    async def ingest_all(self) -> dict[str, int]:
        """Ingest all standard OSM datasets."""
        results = {
            "cities": await self.ingest_cities(),
            "airports": await self.ingest_airports(),
        }
        await self.engine.dispose()
        return results

