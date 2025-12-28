#!/usr/bin/env python3
"""
Check available data sources for Earthlink digital twin
"""

import requests
import json

def check_data_sources():
    print("=" * 80)
    print("EARTHLINK DATA SOURCE OPTIONS")
    print("=" * 80)
    
    # Planet OSM
    print("\n1. FULL PLANET (from planet.openstreetmap.org)")
    print("   URL: https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf")
    print("   Size: ~70GB compressed (PBF)")
    print("   Uncompressed: ~1.5TB")
    print("   Coverage: Entire Earth")
    print("   Format: PBF (Protocolbuffer Binary Format)")
    print("   Update: Weekly")
    print("   Pros: Complete world, single download")
    print("   Cons: Massive size, slow to process")
    
    # Geofabrik continents
    print("\n2. GEOFABRIK CONTINENTS")
    r = requests.get('https://download.geofabrik.de/index-v1-nogeom.json')
    data = r.json()
    
    continents = [f for f in data['features'] if f['properties'].get('parent') == 'planet']
    
    print(f"   Found {len(continents)} continents:\n")
    
    total_size_mb = 0
    for continent in sorted(continents, key=lambda x: x['properties']['name']):
        name = continent['properties']['name']
        urls = continent['properties']['urls']
        pbf_url = urls.get('pbf', 'N/A')
        
        # Try to get file size
        if pbf_url != 'N/A':
            try:
                head = requests.head(pbf_url, allow_redirects=True, timeout=5)
                size_bytes = int(head.headers.get('content-length', 0))
                size_mb = size_bytes / (1024 * 1024)
                total_size_mb += size_mb
                print(f"   - {name:20s} {size_mb:8.1f} MB   {pbf_url}")
            except:
                print(f"   - {name:20s} {'?':>8s} MB   {pbf_url}")
    
    print(f"\n   Total continents size: ~{total_size_mb/1024:.1f} GB")
    print("   Pros: Smaller downloads, can get specific regions")
    print("   Cons: Multiple files to manage")
    
    # Recommendation
    print("\n3. RECOMMENDATION")
    print("   For Earthlink digital twin:")
    print("   - START: North America (~14GB) - good variety, well-mapped")
    print("   - THEN ADD: Europe (~28GB) - dense urban areas")
    print("   - THEN ADD: Other continents as needed")
    print("   - OR: Download full planet if storage/bandwidth allows")
    
    print("\n4. ALTERNATIVE: Regional extracts")
    print("   - Single country/state for testing: Massachusetts (42MB)")
    print("   - Multiple countries: USA states individually")
    print("   - Specific regions of interest")
    
    print("\n" + "=" * 80)
    print("CURRENT STATUS")
    print("=" * 80)
    print("Downloaded: Massachusetts (42MB)")
    print("Ingested: 20,292 features (places + water)")
    print("Schema: Ready for full Earth data (3D, raster, topology)")
    print("\nNext step: Choose planet, continent(s), or specific regions")
    print("=" * 80)

if __name__ == "__main__":
    check_data_sources()
