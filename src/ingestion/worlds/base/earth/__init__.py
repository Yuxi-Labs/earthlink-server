"""Earth data ingestion from Natural Earth, OSM, and Geofabrik."""

from .natural_earth import NaturalEarthIngester
from .osm import OSMIngester
from .geofabrik import GeofabrikIngester

__all__ = ["NaturalEarthIngester", "OSMIngester", "GeofabrikIngester"]

