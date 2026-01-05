"""Add unique indexes on osm_id for OSM-derived tables

Revision ID: add_unique_osm_id_indexes
Revises: 003_osm_tables
Create Date: 2026-01-05

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "add_unique_osm_id_indexes"
down_revision = "003_osm_tables"
branch_labels = None
depends_on = None


TABLES = [
    "buildings",
    "roads",
    "places",
    "water_features",
    "pois",
    "land_cover",
    "boundaries",
]


def _dedupe_and_index(table: str):
    """Remove duplicate osm_id rows and add a partial unique index."""
    op.execute(
        f"""
        DELETE FROM {table} t
        USING (
            SELECT id, ROW_NUMBER() OVER (PARTITION BY osm_id ORDER BY id) AS rn
            FROM {table}
            WHERE osm_id IS NOT NULL
        ) d
        WHERE t.id = d.id AND d.rn > 1;
        """
    )

    op.execute(
        f"""CREATE UNIQUE INDEX IF NOT EXISTS ux_{table}_osm_id
        ON {table} (osm_id)
        WHERE osm_id IS NOT NULL;"""
    )


def upgrade() -> None:
    for table in TABLES:
        _dedupe_and_index(table)


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"DROP INDEX IF EXISTS ux_{table}_osm_id;")
