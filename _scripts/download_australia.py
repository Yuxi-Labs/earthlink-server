#!/usr/bin/env python3
"""
Download Australia & Oceania OSM data from Geofabrik
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
    
    # Australia & Oceania extract (~1.4 GB)
    oceania_url = "https://download.geofabrik.de/australia-oceania-latest.osm.pbf"
    oceania_file = data_dir / "australia-oceania-latest.osm.pbf"
    
    print("=" * 60)
    print("Geofabrik OSM Download - Australia & Oceania")
    print("=" * 60)
    print(f"\nRegion: Australia & Oceania")
    print(f"Size: ~1.4 GB")
    print(f"Source: Geofabrik")
    print()
    
    download_file(oceania_url, oceania_file)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nFile location: {oceania_file}")
    print("\nNext steps:")
    print("  1. Run ingest script to load into database")
    print("  2. Run transform script to create Earthlink tables")
    print()

if __name__ == "__main__":
    main()
