"""
Ingest geographic data using OSM Nominatim API.
Simple REST API for geocoding and place search.
"""

import asyncio
import os
from typing import List, Dict, Any
import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


NOMINATIM_API = "https://nominatim.openstreetmap.org"

# Major cities to query
MAJOR_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "London", "Paris", "Berlin", "Madrid", "Rome",
    "Tokyo", "Beijing", "Shanghai", "Seoul", "Mumbai",
    "São Paulo", "Mexico City", "Cairo", "Lagos", "Johannesburg",
    "Sydney", "Melbourne", "Toronto", "Vancouver", "Moscow"
]


class NominatimIngester:
    """Ingest geographic data from OSM Nominatim."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def search_place(self, place_name: str) -> List[Dict[str, Any]]:
        """Search for a place using Nominatim."""
        async with aiohttp.ClientSession() as session:
            params = {
                "q": place_name,
                "format": "json",
                "addressdetails": 1,
                "limit": 1
            }
            headers = {
                "User-Agent": "Earthlink/0.0.1"
            }
            
            async with session.get(f"{NOMINATIM_API}/search", params=params, headers=headers) as resp:
                if resp.status != 200:
                    print(f"Error searching for {place_name}: {resp.status}")
                    return []
                return await resp.json()
    
    async def ingest_cities(self):
        """Ingest major cities."""
        print(f"Ingesting {len(MAJOR_CITIES)} major cities...")
        
        async with self.async_session() as session:
            count = 0
            for city_name in MAJOR_CITIES:
                results = await self.search_place(city_name)
                
                if not results:
                    print(f"  No results for {city_name}")
                    continue
                
                result = results[0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                
                address = result.get("address", {})
                properties = {
                    "osm_id": result.get("osm_id"),
                    "osm_type": result.get("osm_type"),
                    "country": address.get("country"),
                    "country_code": address.get("country_code"),
                    "display_name": result.get("display_name")
                }
                
                await session.execute(
                    text("""
                        INSERT INTO geo_features (name, geom, feature_type, properties)
                        VALUES (:name, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'city', cast(:properties as jsonb))
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": city_name, "lon": lon, "lat": lat, "properties": str(properties).replace("'", '"')}
                )
                count += 1
                print(f"  ✓ {city_name}")
                
                # Rate limiting - be nice to Nominatim
                await asyncio.sleep(1)
            
            await session.commit()
            print(f"\nIngested {count} cities")
    
    async def ingest_all(self):
        """Ingest all datasets."""
        await self.ingest_cities()


async def main():
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink")
    ingester = NominatimIngester(db_url)
    
    print("Starting OSM Nominatim ingestion...")
    await ingester.ingest_all()
    print("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
