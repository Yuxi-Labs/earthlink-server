"""Ingestion runner - automated data ingestion for Earthlink.

Orchestrates ingestion from multiple sources to build world digital twins.
Structure mirrors simulation/worlds/ - base vs exo, real vs virtual.
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .worlds.base.earth import NaturalEarthIngester, OSMIngester, GeofabrikIngester


@dataclass
class IngestionResult:
    """Result of an ingestion run."""

    source: str
    world: str
    world_type: str  # "base", "exo/real", "exo/virtual", "exo/mixed"
    started_at: datetime
    completed_at: datetime | None = None
    records_ingested: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class IngestionRunner:
    """
    Orchestrates data ingestion for all worlds.
    
    World types:
    - base/earth: Training ground (Natural Earth, OSM, Geofabrik)
    - exo/real: Real celestial bodies (NASA data)
    - exo/virtual: Virtual worlds (game APIs)
    - exo/mixed: Mixed reality
    
    Usage:
        runner = IngestionRunner()
        
        # Base world (Earth)
        await runner.ingest_earth()
        
        # Exo worlds - real (future)
        await runner.ingest_mars()
        await runner.ingest_moon()
        
        # Exo worlds - virtual (future)
        await runner.ingest_roblox()
    """

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://earthlink:earthlink@localhost/earthlink"
        )
        self.results: list[IngestionResult] = []

    # =========================================================================
    # BASE WORLD: EARTH
    # =========================================================================

    async def ingest_earth_natural(
        self,
        datasets: list[str] | None = None,
    ) -> IngestionResult:
        """Ingest Natural Earth data (global basemap)."""
        result = IngestionResult(
            source="natural_earth",
            world="earth",
            world_type="base",
            started_at=datetime.now(UTC),
        )

        try:
            ingester = NaturalEarthIngester(self.db_url)
            counts = await ingester.ingest_all(datasets)
            result.records_ingested = sum(counts.values())
            result.details = counts
        except Exception as e:
            result.errors.append(str(e))

        result.completed_at = datetime.now(UTC)
        self.results.append(result)
        return result

    async def ingest_earth_osm(self) -> IngestionResult:
        """Ingest OpenStreetMap data via Overpass API."""
        result = IngestionResult(
            source="osm",
            world="earth",
            world_type="base",
            started_at=datetime.now(UTC),
        )

        try:
            ingester = OSMIngester(self.db_url)
            counts = await ingester.ingest_all()
            result.records_ingested = sum(counts.values())
            result.details = counts
        except Exception as e:
            result.errors.append(str(e))

        result.completed_at = datetime.now(UTC)
        self.results.append(result)
        return result

    async def ingest_earth_geofabrik(
        self,
        region_id: str,
        layers: list[str] | None = None,
    ) -> IngestionResult:
        """Ingest Geofabrik regional data (OSM shapefiles)."""
        result = IngestionResult(
            source=f"geofabrik:{region_id}",
            world="earth",
            world_type="base",
            started_at=datetime.now(UTC),
        )

        try:
            ingester = GeofabrikIngester(self.db_url)
            counts = await ingester.ingest_region(region_id, layers)
            result.records_ingested = sum(counts.values())
            result.details = counts
        except Exception as e:
            result.errors.append(str(e))

        result.completed_at = datetime.now(UTC)
        self.results.append(result)
        return result

    async def ingest_earth(
        self,
        geofabrik_regions: list[str] | None = None,
    ) -> list[IngestionResult]:
        """Ingest all Earth data sources."""
        results = []

        print("\n" + "=" * 60)
        print("EARTH: Natural Earth (global basemap)")
        print("=" * 60)
        results.append(await self.ingest_earth_natural())

        print("\n" + "=" * 60)
        print("EARTH: OpenStreetMap (cities, airports)")
        print("=" * 60)
        results.append(await self.ingest_earth_osm())

        if geofabrik_regions:
            for region in geofabrik_regions:
                print("\n" + "=" * 60)
                print(f"EARTH: Geofabrik ({region})")
                print("=" * 60)
                results.append(await self.ingest_earth_geofabrik(region))

        return results

    # =========================================================================
    # EXO WORLDS - REAL (NASA data, satellite imagery)
    # =========================================================================

    async def ingest_mars(self) -> list[IngestionResult]:
        """Ingest Mars data from NASA sources. (Future)"""
        print("[Mars] No ingesters implemented yet")
        print("  Future: MRO HiRISE, MOLA topography, rover data")
        return []

    async def ingest_moon(self) -> list[IngestionResult]:
        """Ingest Moon data from NASA sources. (Future)"""
        print("[Moon] No ingesters implemented yet")
        print("  Future: LRO imagery, LOLA topography, Apollo data")
        return []

    async def ingest_venus(self) -> list[IngestionResult]:
        """Ingest Venus data. (Future)"""
        print("[Venus] No ingesters implemented yet")
        print("  Future: Magellan radar mapping")
        return []

    async def ingest_mercury(self) -> list[IngestionResult]:
        """Ingest Mercury data. (Future)"""
        print("[Mercury] No ingesters implemented yet")
        print("  Future: MESSENGER mission data")
        return []

    # =========================================================================
    # EXO WORLDS - VIRTUAL (Game APIs, world exports)
    # =========================================================================

    async def ingest_roblox(self, place_id: str | None = None) -> list[IngestionResult]:
        """Ingest Roblox world data. (Future)"""
        print("[Roblox] No ingesters implemented yet")
        print("  Future: Place data, terrain, assets via Roblox API")
        return []

    async def ingest_minecraft(self, world_path: str | None = None) -> list[IngestionResult]:
        """Ingest Minecraft world data. (Future)"""
        print("[Minecraft] No ingesters implemented yet")
        print("  Future: World saves, block data, structures")
        return []

    # =========================================================================
    # ALL WORLDS
    # =========================================================================

    async def ingest_all(
        self,
        worlds: list[str] | None = None,
        earth_geofabrik_regions: list[str] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest data for specified worlds.
        
        Args:
            worlds: List of worlds (default: ["earth"])
            earth_geofabrik_regions: Geofabrik regions for Earth
        """
        target_worlds = worlds or ["earth"]
        all_results: list[IngestionResult] = []

        for world in target_worlds:
            print("\n" + "#" * 60)
            print(f"# INGESTING: {world.upper()}")
            print("#" * 60)

            if world == "earth":
                results = await self.ingest_earth(earth_geofabrik_regions)
            elif world == "mars":
                results = await self.ingest_mars()
            elif world == "moon":
                results = await self.ingest_moon()
            elif world == "venus":
                results = await self.ingest_venus()
            elif world == "mercury":
                results = await self.ingest_mercury()
            elif world == "roblox":
                results = await self.ingest_roblox()
            elif world == "minecraft":
                results = await self.ingest_minecraft()
            else:
                print(f"Unknown world: {world}")
                results = []

            all_results.extend(results)

        self._print_summary(all_results)
        return all_results

    def _print_summary(self, results: list[IngestionResult]) -> None:
        """Print ingestion summary."""
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)

        # Group by world type
        by_type: dict[str, list[IngestionResult]] = {}
        for r in results:
            by_type.setdefault(r.world_type, []).append(r)

        total_records = 0
        total_time = 0.0

        for world_type in ["base", "exo/real", "exo/virtual", "exo/mixed"]:
            if world_type not in by_type:
                continue
            
            print(f"\n{world_type.upper()}:")
            for result in by_type[world_type]:
                status = "✓" if result.success else "✗"
                print(f"  {status} {result.world}/{result.source}: {result.records_ingested:,} records ({result.duration_seconds:.1f}s)")
                if result.errors:
                    for error in result.errors:
                        print(f"      ERROR: {error}")
                total_records += result.records_ingested
                total_time += result.duration_seconds

        print("\n" + "-" * 60)
        print(f"TOTAL: {total_records:,} records in {total_time:.1f}s")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Earthlink data ingestion")
    parser.add_argument(
        "--world",
        choices=["earth", "mars", "moon", "venus", "mercury", "roblox", "minecraft", "all"],
        default="earth",
        help="World to ingest",
    )
    parser.add_argument(
        "--source",
        choices=["all", "natural-earth", "osm", "geofabrik"],
        default="all",
        help="Data source (Earth only)",
    )
    parser.add_argument(
        "--region",
        help="Geofabrik region ID",
    )
    parser.add_argument(
        "--db-url",
        help="Database URL",
    )

    args = parser.parse_args()
    runner = IngestionRunner(args.db_url)

    if args.world == "all":
        regions = [args.region] if args.region else None
        await runner.ingest_all(earth_geofabrik_regions=regions)
    elif args.world == "earth":
        if args.source == "all":
            regions = [args.region] if args.region else None
            await runner.ingest_earth(geofabrik_regions=regions)
        elif args.source == "natural-earth":
            await runner.ingest_earth_natural()
        elif args.source == "osm":
            await runner.ingest_earth_osm()
        elif args.source == "geofabrik":
            if not args.region:
                print("ERROR: --region required for geofabrik")
                return
            await runner.ingest_earth_geofabrik(args.region)
    elif args.world == "mars":
        await runner.ingest_mars()
    elif args.world == "moon":
        await runner.ingest_moon()
    elif args.world == "roblox":
        await runner.ingest_roblox()
    elif args.world == "minecraft":
        await runner.ingest_minecraft()


if __name__ == "__main__":
    asyncio.run(main())
