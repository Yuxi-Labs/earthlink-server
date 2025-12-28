"""
Ingest OpenStreetMap data via Overpass API into PostGIS.
Modern API-based approach - no downloading zip files from flaky servers.
"""

import asyncio
import os
from typing import Dict, Any
import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


OVERPASS_API = "https://overpass-api.de/api/interpreter"

# Overpass QL queries - focused on populated areas to avoid timeouts
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
    """
}


class OSMIngester:
    """Ingest OpenStreetMap data via Overpass API."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def query_overpass(self, query: str) -> Dict[str, Any]:
        """Execute Overpass API query."""
        async with aiohttp.ClientSession() as session:
            async with session.post(OVERPASS_API, data={"data": query}) as resp:
                if resp.status != 200:
                    raise Exception(f"Overpass API error: {resp.status}")
                return await resp.json()
    
    async def ingest_cities(self):
        """Ingest major cities and capitals from OSM."""
        print("Fetching major cities (pop > 100k) from OpenStreetMap...")
        data = await self.query_overpass(QUERIES["major_cities"])
        
        async with self.async_session() as session:
            count = 0
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue
                
                tags = element.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                
                lat = element.get("lat")
                lon = element.get("lon")
                population = tags.get("population")
                
                properties = {
                    "population": population,
                    "place": tags.get("place"),
                    "osm_id": element.get("id"),
                    "country": tags.get("is_in:country")
                }
                
                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'city', :properties::jsonb)
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": name, "lon": lon, "lat": lat, "properties": str(properties).replace("'", '"')}
                )
                count += 1
            
            await session.commit()
            print(f"Ingested {count} major cities")
        
        # Also fetch capitals
        print("Fetching capital cities from OpenStreetMap...")
        data = await self.query_overpass(QUERIES["capital_cities"])
        
        async with self.async_session() as session:
            count = 0
            for element in data.get("elements", []):
                if element.get("type") != "node":
                    continue
                
                tags = element.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                
                lat = element.get("lat")
                lon = element.get("lon")
                
                properties = {
                    "capital": tags.get("capital"),
                    "population": tags.get("population"),
                    "place": tags.get("place"),
                    "osm_id": element.get("id"),
                    "country": tags.get("is_in:country")
                }
                
                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'city', :properties::jsonb)
                        ON CONFLICT (name, feature_type) DO UPDATE SET properties = EXCLUDED.properties
                    """),
                    {"name": name, "lon": lon, "lat": lat, "properties": str(properties).replace("'", '"')}
                )
                count += 1
            
            await session.commit()
            print(f"Ingested {count} capital cities")
    
    async def ingest_all(self):
        """Ingest all datasets."""
        await self.ingest_cities()


async def main():
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink")
    ingester = OSMIngester(db_url)
    
    print("Starting OpenStreetMap ingestion...")
    await ingester.ingest_all()
    print("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
