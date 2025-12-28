"""
Ingest OpenStreetMap data from Geofabrik extracts.
Downloads regional shapefiles and imports into PostGIS.

Geofabrik provides pre-processed OSM data as shapefiles.
Technical docs: https://download.geofabrik.de/technical.html
Index API: https://download.geofabrik.de/index-v1-nogeom.json
"""

import asyncio
import os
import zipfile
from pathlib import Path
import json
import requests
import geopandas as gpd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


# Geofabrik configuration
GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1-nogeom.json"
DATA_DIR = Path("/app/data/geofabrik")


class GeofabrikIngester:
    """Download and ingest OSM shapefiles from Geofabrik."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.index_cache = None
    
    def get_index(self):
        """Fetch Geofabrik index of available regions."""
        if self.index_cache is None:
            print("Fetching Geofabrik index...")
            resp = requests.get(GEOFABRIK_INDEX, timeout=30)
            resp.raise_for_status()
            self.index_cache = resp.json()
            print(f"Index loaded: {len(self.index_cache['features'])} regions available")
        return self.index_cache
    
    def find_region(self, region_id: str) -> dict:
        """Find region metadata from index by ID."""
        index = self.get_index()
        for feature in index['features']:
            if feature['properties']['id'] == region_id:
                return feature['properties']
        
        # If not found, list some available regions
        print(f"\nRegion '{region_id}' not found.")
        print("\nSome available regions with shapefiles:")
        count = 0
        for feature in index['features']:
            props = feature['properties']
            if 'shp' in props.get('urls', {}):
                print(f"  - {props['id']}: {props['name']}")
                count += 1
                if count >= 20:
                    print("  ... (and more)")
                    break
        
        raise ValueError(f"Region '{region_id}' not found")
    
    async def download_region(self, region_id: str) -> Path:
        """Download Geofabrik shapefile for region using JSON index."""
        region = self.find_region(region_id)
        
        # Get shapefile URL from index
        urls = region.get('urls', {})
        shp_url = urls.get('shp')
        if not shp_url:
            raise ValueError(f"No shapefile available for region '{region_id}'")
        
        region_name = region['name']
        print(f"\nDownloading: {region_name}")
        print(f"URL: {shp_url}")
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / f"{region_id.replace('/', '_')}.zip"
        extract_dir = DATA_DIR / region_id.replace('/', '_')
        
        # Download if not cached
        if not zip_path.exists():
            response = requests.get(shp_url, stream=True, timeout=600)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                raise Exception(f"Got HTML instead of ZIP. URL may be incorrect: {shp_url}")
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size / 1024 / 1024:.1f} MB")
            
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"Progress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.0f}%)", end='\r')
            
            print(f"\nDownloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"Using cached file: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Validate ZIP file
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_test:
                if zip_test.testzip() is not None:
                    raise zipfile.BadZipFile("ZIP file is corrupted")
        except zipfile.BadZipFile as e:
            print(f"Cached ZIP is corrupted: {e}")
            print("Deleting and re-downloading...")
            zip_path.unlink()
            extract_dir.rmdir() if extract_dir.exists() else None
            # Re-download
            response = requests.get(shp_url, stream=True, timeout=600)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            print(f"Re-downloading {total_size / 1024 / 1024:.1f} MB...")
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"Progress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.0f}%)", end='\r')
            print(f"\nDownloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Extract
        if not extract_dir.exists():
            print(f"Extracting to {extract_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            files = list(extract_dir.glob('*'))
            print(f"Extracted {len(files)} files")
            # List .shp files
            shp_files = list(extract_dir.glob('*.shp'))
            if shp_files:
                print(f"Found {len(shp_files)} shapefiles:")
                for shp in shp_files[:10]:  # Show first 10
                    print(f"  - {shp.name}")
        else:
            print(f"Using cached extraction: {extract_dir}")
        
        return extract_dir
    
    async def ingest_places(self, extract_dir: Path):
        """Ingest populated places (cities, towns) from shapefile."""
        places_shp = extract_dir / "gis_osm_places_free_1.shp"
        
        if not places_shp.exists():
            print(f"Skipping places - {places_shp.name} not found")
            return
        
        print(f"\nIngesting places from {places_shp.name}...")
        gdf = gpd.read_file(places_shp)
        
        # Filter to cities and towns
        if 'fclass' in gdf.columns:
            gdf = gdf[gdf['fclass'].isin(['city', 'town'])]
        
        print(f"Found {len(gdf)} cities/towns")
        
        async with self.async_session() as session:
            inserted = 0
            for idx, row in gdf.iterrows():
                try:
                    geom_wkt = row.geometry.wkt if row.geometry else None
                    name = row.get('name', f'Place_{idx}')
                    fclass = row.get('fclass', 'place')
                    population = row.get('population')
                    
                    props = {"population": population} if population else {}
                    props_json = json.dumps(props) if props else None
                    
                    await session.execute(
                        text("""
                            INSERT INTO geo_features (name, feature_type, geom)
                            VALUES (:name, :ftype, ST_GeomFromText(:geom, 4326))
                        """),
                        {
                            "name": name,
                            "ftype": fclass,
                            "geom": geom_wkt
                        }
                    )
                    inserted += 1
                    
                    if inserted % 100 == 0:
                        await session.commit()
                        print(f"Inserted {inserted} places...", end='\r')
                        
                except Exception as e:
                    print(f"\nError inserting place {name}: {e}")
                    continue
            
            await session.commit()
            print(f"\nInserted {inserted} places")
    
    async def ingest_water(self, extract_dir: Path):
        """Ingest water features (lakes, rivers) from shapefile."""
        water_shp = extract_dir / "gis_osm_water_a_free_1.shp"
        
        if not water_shp.exists():
            print(f"Skipping water - {water_shp.name} not found")
            return
        
        print(f"\nIngesting water features from {water_shp.name}...")
        gdf = gpd.read_file(water_shp)
        
        # Filter to major water bodies
        if 'fclass' in gdf.columns:
            gdf = gdf[gdf['fclass'].isin(['water', 'lake', 'reservoir'])]
        
        print(f"Found {len(gdf)} water features")
        
        inserted = 0
        for idx, row in gdf.iterrows():
            async with self.async_session() as session:
                try:
                    geom_wkt = row.geometry.wkt if row.geometry else None
                    name = row.get('name', f'Water_{idx}')
                    fclass = row.get('fclass', 'water')
                    
                    props_json = json.dumps({})
                    
                    await session.execute(
                        text("""
                            INSERT INTO geo_features (name, feature_type, geom)
                            VALUES (:name, :ftype, ST_GeomFromText(:geom, 4326))
                        """),
                        {
                            "name": name,
                            "ftype": fclass,
                            "geom": geom_wkt
                        }
                    )
                    await session.commit()
                    inserted += 1
                    
                    if inserted % 50 == 0:
                        print(f"Inserted {inserted} water features...", end='\r')
                        
                except Exception as e:
                    await session.rollback()
                    print(f"\nError inserting water feature {name}: {e}")
                    continue
        
        print(f"\nInserted {inserted} water features")
    
    async def run(self, region_id: str):
        """Download and ingest OSM data for a region."""
        print(f"=== Geofabrik OSM Ingestion ===")
        print(f"Region: {region_id}\n")
        
        extract_dir = await self.download_region(region_id)
        
        await self.ingest_places(extract_dir)
        await self.ingest_water(extract_dir)
        
        # Verify counts
        async with self.async_session() as session:
            result = await session.execute(text("SELECT COUNT(*), feature_type FROM geo_features GROUP BY feature_type"))
            rows = result.fetchall()
            
            print(f"\n=== Ingestion Complete ===")
            print("Features by type:")
            total = 0
            for count, ftype in rows:
                print(f"  {ftype}: {count}")
                total += count
            print(f"Total features: {total}")


async def main():
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink")
    ingester = GeofabrikIngester(db_url)
    
    # Start with US Massachusetts (small region for testing)
    # Other options: "us/new-york", "europe/germany", "asia/japan", etc.
    await ingester.run("us/massachusetts")


if __name__ == "__main__":
    asyncio.run(main())
