#!/usr/bin/env python3
"""
Seed geo data into PostgreSQL/PostGIS tables.

Populates geo_features (point locations) and geo_regions (polygon areas)
with sample Earth locations for agent exploration.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.database import async_session_maker


FEATURES = [
    # Major cities (Points)
    {
        "name": "New York City",
        "feature_type": "city",
        "lon": -74.0060,
        "lat": 40.7128,
        "properties": {
            "country": "USA",
            "population": 8336817,
            "timezone": "America/New_York",
        },
    },
    {
        "name": "London",
        "feature_type": "city",
        "lon": -0.1276,
        "lat": 51.5074,
        "properties": {
            "country": "UK",
            "population": 8982000,
            "timezone": "Europe/London",
        },
    },
    {
        "name": "Tokyo",
        "feature_type": "city",
        "lon": 139.6917,
        "lat": 35.6895,
        "properties": {
            "country": "Japan",
            "population": 13960000,
            "timezone": "Asia/Tokyo",
        },
    },
    {
        "name": "Paris",
        "feature_type": "city",
        "lon": 2.3522,
        "lat": 48.8566,
        "properties": {
            "country": "France",
            "population": 2161000,
            "timezone": "Europe/Paris",
        },
    },
    {
        "name": "Sydney",
        "feature_type": "city",
        "lon": 151.2093,
        "lat": -33.8688,
        "properties": {
            "country": "Australia",
            "population": 5312000,
            "timezone": "Australia/Sydney",
        },
    },
    # Natural landmarks
    {
        "name": "Mount Everest",
        "feature_type": "mountain",
        "lon": 86.9250,
        "lat": 27.9881,
        "properties": {
            "elevation_m": 8849,
            "range": "Himalayas",
            "first_ascent": 1953,
        },
    },
    {
        "name": "Grand Canyon",
        "feature_type": "canyon",
        "lon": -112.1129,
        "lat": 36.0544,
        "properties": {
            "depth_m": 1857,
            "length_km": 446,
            "country": "USA",
        },
    },
    {
        "name": "Amazon Rainforest Center",
        "feature_type": "rainforest",
        "lon": -62.2159,
        "lat": -3.4653,
        "properties": {
            "area_km2": 5500000,
            "biodiversity": "extreme",
        },
    },
    {
        "name": "Great Barrier Reef",
        "feature_type": "reef",
        "lon": 147.7000,
        "lat": -18.2871,
        "properties": {
            "length_km": 2300,
            "country": "Australia",
        },
    },
    {
        "name": "Sahara Desert Center",
        "feature_type": "desert",
        "lon": 9.0000,
        "lat": 23.4162,
        "properties": {
            "area_km2": 9200000,
            "climate": "hot desert",
        },
    },
    # Oceans (representative points)
    {
        "name": "Pacific Ocean (Mariana Trench)",
        "feature_type": "ocean",
        "lon": 142.1996,
        "lat": 11.3493,
        "properties": {
            "depth_m": -10994,
            "deepest_point": "Challenger Deep",
        },
    },
    {
        "name": "Atlantic Ocean Center",
        "feature_type": "ocean",
        "lon": -30.0000,
        "lat": 0.0000,
        "properties": {"area_km2": 106460000},
    },
]


REGIONS = [
    # Continents (simplified polygons)
    {
        "name": "North America Sample",
        "region_type": "continent",
        "coords": [
            [
                [-170, 15],  # West coast Mexico
                [-170, 70],  # Alaska
                [-50, 70],  # Greenland
                [-50, 25],  # East coast Florida
                [-100, 15],  # Mexico
                [-170, 15],  # Close polygon
            ]
        ],
        "properties": {"area_km2": 24709000},
    },
    {
        "name": "Europe Sample",
        "region_type": "continent",
        "coords": [
            [
                [-10, 36],  # Portugal
                [40, 36],  # Turkey
                [40, 71],  # Norway
                [-10, 71],  # Iceland
                [-10, 36],  # Close
            ]
        ],
        "properties": {"area_km2": 10180000},
    },
    # National parks
    {
        "name": "Yellowstone National Park",
        "region_type": "park",
        "coords": [
            [
                [-111.15, 44.13],
                [-109.83, 44.13],
                [-109.83, 45.11],
                [-111.15, 45.11],
                [-111.15, 44.13],
            ]
        ],
        "properties": {
            "country": "USA",
            "established": 1872,
            "area_km2": 8991,
        },
    },
    {
        "name": "Serengeti National Park",
        "region_type": "park",
        "coords": [
            [
                [34.5, -3.3],
                [35.3, -3.3],
                [35.3, -1.5],
                [34.5, -1.5],
                [34.5, -3.3],
            ]
        ],
        "properties": {
            "country": "Tanzania",
            "area_km2": 14763,
            "wildlife": "Big Five",
        },
    },
]


async def seed_geo_data():
    """Seed geo features and regions into the database."""
    async with async_session_maker() as session:
        # Get raw connection for direct SQL execution
        conn = await session.connection()
        raw_conn = await conn.get_raw_connection()

        # Clear existing data
        print("Clearing existing geo data...")
        await raw_conn.execute("DELETE FROM geo_features")
        await raw_conn.execute("DELETE FROM geo_regions")

        # Insert features (points)
        print(f"\nInserting {len(FEATURES)} geo features...")
        for feature in FEATURES:
            await raw_conn.execute(
                """
                INSERT INTO geo_features (name, feature_type, geom, properties)
                VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4), 4326), $5::jsonb)
                """,
                feature["name"],
                feature["feature_type"],
                feature["lon"],
                feature["lat"],
                json.dumps(feature["properties"]),
            )
            print(f"  ✓ {feature['name']} ({feature['feature_type']})")

        # Insert regions (polygons)
        print(f"\nInserting {len(REGIONS)} geo regions...")
        for region in REGIONS:
            # Convert coords to WKT polygon format
            coords_str = ",".join([f"{lon} {lat}" for lon, lat in region["coords"][0]])
            wkt = f"POLYGON(({coords_str}))"

            await raw_conn.execute(
                """
                INSERT INTO geo_regions (name, region_type, geom, properties)
                VALUES ($1, $2, ST_SetSRID(ST_GeomFromText($3), 4326), $4::jsonb)
                """,
                region["name"],
                region["region_type"],
                wkt,
                json.dumps(region["properties"]),
            )
            print(f"  ✓ {region['name']} ({region['region_type']})")

        await session.commit()

        # Verify counts
        result = await raw_conn.fetchval("SELECT COUNT(*) FROM geo_features")
        feature_count = result

        result = await raw_conn.fetchval("SELECT COUNT(*) FROM geo_regions")
        region_count = result

        print(f"\n✓ Seeded {feature_count} features and {region_count} regions successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("Earthlink Geo Data Seeder")
    print("=" * 60)
    asyncio.run(seed_geo_data())
