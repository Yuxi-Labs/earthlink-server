"""Worlds module.

Structure:
- base/     - Primary simulation world (Earth)
- exo/      - Other worlds
  - real/   - Real celestial bodies (Mars, Moon, etc.)
  - virtual/- Fully virtual worlds
  - mixed/  - Mixed reality worlds
"""

from .config import WorldConfig, SpatialConfig
from .base.earth import Earth, EarthConfig

__all__ = [
    "WorldConfig",
    "SpatialConfig",
    "Earth",
    "EarthConfig",
]
