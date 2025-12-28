#!/usr/bin/env python3
"""
Download Europe OSM data from Geofabrik
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, destination: Path):
    """Download file with progress bar"""
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if destination.exists():
        print(f"✓ File already exists: {destination}")
        print(f"  Size: {destination.stat().st_size / (1024**3):.2f} GB")
        response = input("  Re-download? (y/N): ")
        if response.lower() != 'y':
            return
    
    print(f"Downloading: {url}")
    print(f"Destination: {destination}")
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Download complete: {destination.stat().st_size / (1024**3):.2f} GB")

def main():
    # Download directory - temp location inside container
    data_dir = Path("/tmp/geofabrik")
    
    # South America extract
    south_america_url = "https://download.geofabrik.de/south-america-latest.osm.pbf"
    south_america_file = data_dir / "south-america-latest.osm.pbf"
    
    print("=" * 80)
    print("DOWNLOADING SOUTH AMERICA OSM DATA")
    print("=" * 80)
    print(f"Source: Geofabrik")
    print(f"URL: {south_america_url}")
    print(f"Expected size: ~3GB")
    print(f"Coverage: All of South America")
    print("=" * 80)
    print()
    
    try:
        download_file(south_america_url, south_america_file)
        
        print()
        print("=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"File: {south_america_file}")
        print(f"Size: {south_america_file.stat().st_size / (1024**3):.2f} GB")
        print()
        print("Next steps:")
        print("1. Convert PBF to shapefiles (or use osm2pgsql)")
        print("2. Run ingestion script")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

if __name__ == "__main__":
    main()
