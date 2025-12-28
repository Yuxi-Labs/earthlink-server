"""
Create minimal sample geographic data for initial testing.
This populates geo tables with a few sample locations to unblock agent development.
Full Natural Earth ingestion can be run later when network issues resolve.
"""

import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


async def create_sample_data():
    """Create sample geographic features and regions."""
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink")
    engine = create_async_engine(db_url, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Sample cities (features)
        cities = [
            ("New York", "POINT(-74.006 40.7128)", "Major city, United States"),
            ("London", "POINT(-0.1276 51.5074)", "Capital of United Kingdom"),
            ("Tokyo", "POINT(139.6917 35.6895)", "Capital of Japan"),
            ("Paris", "POINT(2.3522 48.8566)", "Capital of France"),
            ("Sydney", "POINT(151.2093 -33.8688)", "Major city, Australia"),
        ]
        
        print("Inserting sample cities...")
        for name, geom_wkt, description in cities:
            await session.execute(
                text("""
                    INSERT INTO geo_features (name, geom, feature_type, properties)
                    VALUES (:name, ST_GeomFromText(:geom, 4326), 'city', :properties)
                    ON CONFLICT DO NOTHING
                """),
                {"name": name, "geom": geom_wkt, "properties": f'{{"description": "{description}"}}'}
            )
        
        # Sample countries (regions)
        countries = [
            ("United States", "POLYGON((-125 49, -66 49, -66 24, -125 24, -125 49))", "North American country"),
            ("United Kingdom", "POLYGON((-6 59, 2 59, 2 50, -6 50, -6 59))", "European island nation"),
            ("Japan", "POLYGON((129 45, 145 45, 145 30, 129 30, 129 45))", "East Asian island nation"),
        ]
        
        print("Inserting sample countries...")
        for name, geom_wkt, description in countries:
            await session.execute(
                text("""
                    INSERT INTO geo_regions (name, geom, region_type, properties)
                    VALUES (:name, ST_GeomFromText(:geom, 4326), 'country', :properties)
                    ON CONFLICT DO NOTHING
                """),
                {"name": name, "geom": geom_wkt, "properties": f'{{"description": "{description}"}}'}
            )
        
        await session.commit()
        print("Sample geographic data created successfully!")
        
        # Verify counts
        result = await session.execute(text("SELECT COUNT(*) FROM geo_features"))
        feature_count = result.scalar()
        result = await session.execute(text("SELECT COUNT(*) FROM geo_regions"))
        region_count = result.scalar()
        
        print(f"Total features: {feature_count}")
        print(f"Total regions: {region_count}")


if __name__ == "__main__":
    asyncio.run(create_sample_data())
