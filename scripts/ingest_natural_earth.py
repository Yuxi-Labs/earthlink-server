"""
Ingest Natural Earth data into PostGIS for Earthlink digital twin.

Downloads and imports:
- Cultural vectors: countries, populated places, roads
- Physical vectors: coastlines, lakes, rivers

Data source: https://www.naturalearthdata.com/
"""

import asyncio
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiohttp
import fiona
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, shape
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Natural Earth download URLs
DATASETS = {
    "countries": {
        "url": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        "type": "regions",
        "layer": "ne_10m_admin_0_countries"
    },
    "populated_places": {
        "url": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip",
        "type": "features",
        "layer": "ne_10m_populated_places"
    },
    "lakes": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip",
        "type": "features",
        "layer": "ne_10m_lakes"
    },
    "rivers": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "type": "features",
        "layer": "ne_10m_rivers_lake_centerlines"
    },
    "coastline": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
        "type": "features",
        "layer": "ne_10m_coastline"
    }
}

DATA_DIR = Path(__file__).parent.parent / "data" / "natural_earth"


class NaturalEarthIngester:
    """Manages downloading and ingesting Natural Earth data."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
    async def download_dataset(self, name: str, config: Dict[str, Any]) -> Path:
        """Download and extract a Natural Earth dataset."""
        print(f"Downloading {name}...")
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / f"{name}.zip"
        extract_dir = DATA_DIR / name
        
        # Download if not already present
        if not zip_path.exists():
            async with aiohttp.ClientSession() as session:
                async with session.get(config["url"]) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download {name}: {resp.status}")
                    
                    content = await resp.read()
                    with open(zip_path, "wb") as f:
                        f.write(content)
        
        # Extract
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        
        # Find shapefile
        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise Exception(f"No shapefile found in {extract_dir}")
        
        return shp_files[0]
    
    async def ingest_features(self, shapefile: Path, dataset_name: str):
        """Ingest point/line features from shapefile."""
        print(f"Ingesting features from {shapefile.name}...")
        
        with fiona.open(shapefile) as src:
            feature_count = 0
            batch_size = 100
            batch = []
            
            for feature in src:
                geom = shape(feature["geometry"])
                props = feature["properties"]
                
                # Convert geometry to WKT
                if isinstance(geom, (Point, LineString)):
                    geom_wkt = geom.wkt
                    geom_type = "point" if isinstance(geom, Point) else "line"
                else:
                    # Skip multipolygons/complex geometries for features table
                    continue
                
                # Extract relevant properties
                name = props.get("NAME") or props.get("name") or props.get("NAME_EN", "Unknown")
                feature_type = self._determine_feature_type(dataset_name, props)
                metadata = self._extract_metadata(props)
                
                batch.append({
                    "name": name,
                    "feature_type": feature_type,
                    "geometry": geom_wkt,
                    "metadata": metadata
                })
                
                if len(batch) >= batch_size:
                    await self._insert_features(batch)
                    feature_count += len(batch)
                    batch = []
                    print(f"  Inserted {feature_count} features...")
            
            # Insert remaining
            if batch:
                await self._insert_features(batch)
                feature_count += len(batch)
        
        print(f"  Total features inserted: {feature_count}")
    
    async def ingest_regions(self, shapefile: Path, dataset_name: str):
        """Ingest polygon regions from shapefile."""
        print(f"Ingesting regions from {shapefile.name}...")
        
        with fiona.open(shapefile) as src:
            region_count = 0
            batch_size = 50
            batch = []
            
            for feature in src:
                geom = shape(feature["geometry"])
                props = feature["properties"]
                
                # Convert to simple polygon or multipolygon
                if isinstance(geom, Polygon):
                    geom_wkt = geom.wkt
                elif isinstance(geom, MultiPolygon):
                    geom_wkt = geom.wkt
                else:
                    continue
                
                name = props.get("NAME") or props.get("ADMIN") or props.get("NAME_EN", "Unknown")
                region_type = self._determine_region_type(dataset_name, props)
                metadata = self._extract_metadata(props)
                
                batch.append({
                    "name": name,
                    "region_type": region_type,
                    "geometry": geom_wkt,
                    "metadata": metadata
                })
                
                if len(batch) >= batch_size:
                    await self._insert_regions(batch)
                    region_count += len(batch)
                    batch = []
                    print(f"  Inserted {region_count} regions...")
            
            if batch:
                await self._insert_regions(batch)
                region_count += len(batch)
        
        print(f"  Total regions inserted: {region_count}")
    
    async def _insert_features(self, batch: List[Dict[str, Any]]):
        """Batch insert features into database."""
        async with self.async_session() as session:
            for item in batch:
                query = text("""
                    INSERT INTO geo_features (name, feature_type, geometry, metadata)
                    VALUES (:name, :feature_type, ST_GeomFromText(:geometry, 4326), :metadata::jsonb)
                    ON CONFLICT (name, feature_type) DO NOTHING
                """)
                await session.execute(query, item)
            await session.commit()
    
    async def _insert_regions(self, batch: List[Dict[str, Any]]):
        """Batch insert regions into database."""
        async with self.async_session() as session:
            for item in batch:
                query = text("""
                    INSERT INTO geo_regions (name, region_type, geometry, metadata)
                    VALUES (:name, :region_type, ST_GeomFromText(:geometry, 4326), :metadata::jsonb)
                    ON CONFLICT (name, region_type) DO NOTHING
                """)
                await session.execute(query, item)
            await session.commit()
    
    def _determine_feature_type(self, dataset: str, props: Dict) -> str:
        """Determine feature type from dataset and properties."""
        if "populated_places" in dataset:
            return "city"
        elif "lake" in dataset:
            return "lake"
        elif "river" in dataset:
            return "river"
        elif "coast" in dataset:
            return "coastline"
        return "landmark"
    
    def _determine_region_type(self, dataset: str, props: Dict) -> str:
        """Determine region type from dataset and properties."""
        if "countries" in dataset or "admin_0" in dataset:
            return "country"
        elif "states" in dataset or "admin_1" in dataset:
            return "state"
        return "administrative"
    
    def _extract_metadata(self, props: Dict) -> Dict[str, Any]:
        """Extract relevant metadata from shapefile properties."""
        # Common fields to preserve
        metadata = {}
        
        preserve_fields = [
            "ISO_A2", "ISO_A3", "POP_EST", "GDP_MD_EST", "CONTINENT",
            "REGION_UN", "SUBREGION", "ADM0_A3", "LATITUDE", "LONGITUDE",
            "TIMEZONE", "ELEVATION", "FEATURECLA"
        ]
        
        for field in preserve_fields:
            if field in props and props[field] is not None:
                metadata[field.lower()] = props[field]
        
        return metadata
    
    async def run(self, datasets: Optional[List[str]] = None):
        """Download and ingest Natural Earth datasets."""
        target_datasets = datasets or list(DATASETS.keys())
        
        print(f"Starting Natural Earth ingestion for: {', '.join(target_datasets)}")
        
        for name in target_datasets:
            if name not in DATASETS:
                print(f"Warning: Unknown dataset '{name}', skipping")
                continue
            
            config = DATASETS[name]
            
            try:
                # Download
                shapefile = await self.download_dataset(name, config)
                
                # Ingest based on type
                if config["type"] == "features":
                    await self.ingest_features(shapefile, name)
                elif config["type"] == "regions":
                    await self.ingest_regions(shapefile, name)
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
        
        print("Ingestion complete!")
        await self.engine.dispose()


async def main():
    """Main ingestion entry point."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://earthlink:earthlink@localhost/earthlink"
    )
    
    ingester = NaturalEarthIngester(db_url)
    
    # Start with essential datasets for MVP
    await ingester.run(datasets=[
        "countries",
        "populated_places",
        "lakes",
        "coastline"
    ])


if __name__ == "__main__":
    asyncio.run(main())
