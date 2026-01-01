"""
Generation capability: create maps, summaries, and theories from agent data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
from uuid import UUID, uuid4

import numpy as np

from .reasoning import Hypothesis


@dataclass
class GeneratedMap:
    """Lightweight map representation produced from exploration data."""

    id: UUID = field(default_factory=uuid4)
    map_type: str = "route"
    center: tuple[float, float] | None = None
    bounds: dict[str, float] = field(default_factory=dict)
    coverage_km2: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "map_type": self.map_type,
            "center": self.center,
            "bounds": self.bounds,
            "coverage_km2": self.coverage_km2,
            "path": self.path,
            "hotspots": self.hotspots,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SummaryOutput:
    """Structured summary from observations."""

    id: UUID = field(default_factory=uuid4)
    text: str = ""
    key_points: list[str] = field(default_factory=list)
    modality_counts: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "text": self.text,
            "key_points": self.key_points,
            "modality_counts": self.modality_counts,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Theory:
    """Theory synthesized from hypotheses and evidence."""

    id: UUID = field(default_factory=uuid4)
    statement: str = ""
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    contradictions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "statement": self.statement,
            "supporting_evidence": self.supporting_evidence,
            "confidence": self.confidence,
            "contradictions": self.contradictions,
            "created_at": self.created_at.isoformat(),
        }


class GenerationModule:
    """Create maps, summaries, and theories from agent experience."""

    def __init__(self, agent_id: UUID, max_points: int = 500) -> None:
        self.agent_id = agent_id
        self.max_points = max_points
        self._map_history: list[GeneratedMap] = []
        self._summary_history: list[SummaryOutput] = []
        self._theory_history: list[Theory] = []

    async def generate_map(
        self,
        exploration_data: list[dict[str, Any]],
        map_type: str = "route",
    ) -> GeneratedMap:
        """
        Generate a conceptual map from exploration data.

        Map includes center, bounds, coverage, and hotspots.
        """
        if not exploration_data:
            return GeneratedMap(
                map_type=map_type,
                summary="No exploration data available to map.",
            )

        coords = []
        hotspots = []
        for item in exploration_data[: self.max_points]:
            lat, lon = self._extract_lat_lon(item)
            if lat is None or lon is None:
                continue
            coords.append((lat, lon))
            if item.get("findings") or item.get("observations"):
                hotspots.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": item.get("label") or item.get("type") or "finding",
                        "details": item.get("findings") or item.get("observations"),
                    }
                )

        if not coords:
            return GeneratedMap(
                map_type=map_type,
                summary="No valid coordinates found in exploration data.",
            )

        lats, lons = zip(*coords)
        center = (float(np.mean(lats)), float(np.mean(lons)))
        bounds = {
            "lat_min": float(min(lats)),
            "lat_max": float(max(lats)),
            "lon_min": float(min(lons)),
            "lon_max": float(max(lons)),
        }
        coverage_km2 = self._estimate_area_km2(bounds["lat_min"], bounds["lat_max"], bounds["lon_min"], bounds["lon_max"])

        # Build ordered path using provided timestamps where possible
        path = self._build_ordered_path(exploration_data)

        summary = (
            f"Mapped {len(coords)} points over ~{coverage_km2:.1f} km^2. "
            f"Center at ({center[0]:.3f}, {center[1]:.3f}). "
            f"{len(hotspots)} notable hotspots identified."
        )

        generated_map = GeneratedMap(
            map_type=map_type,
            center=center,
            bounds=bounds,
            coverage_km2=coverage_km2,
            path=path,
            hotspots=hotspots,
            summary=summary,
        )
        self._map_history.append(generated_map)
        return generated_map

    async def create_summary(
        self,
        observations: list[dict[str, Any]],
        max_length: int = 200,
    ) -> SummaryOutput:
        """Summarize observations into concise human-readable output."""
        if not observations:
            return SummaryOutput(text="No observations to summarize.")

        modality_counts = Counter(o.get("modality", "unknown") for o in observations)
        key_points: list[str] = []

        for obs in observations[:10]:
            modality = obs.get("modality", "observation")
            data = obs.get("data", {})
            if isinstance(data, dict) and data:
                # Extract up to two salient fields per observation
                items = list(data.items())[:2]
                excerpt = ", ".join(f"{k}: {v}" for k, v in items)
            else:
                excerpt = str(data)[:80]
            key_points.append(f"{modality}: {excerpt}")

        dominant_modalities = ", ".join(
            f"{modality} ({count})" for modality, count in modality_counts.most_common(3)
        )
        summary_text = (
            f"Observed {len(observations)} items. Dominant modalities: {dominant_modalities}. "
            f"Key details: {'; '.join(key_points)}"
        )

        if len(summary_text) > max_length:
            summary_text = summary_text[: max_length - 3] + "..."

        summary_output = SummaryOutput(
            text=summary_text,
            key_points=key_points,
            modality_counts=dict(modality_counts),
        )
        self._summary_history.append(summary_output)
        return summary_output

    async def formulate_theory(
        self,
        hypotheses: Iterable[Hypothesis] | None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> Theory:
        """
        Generate a theory by combining hypotheses and supporting evidence.

        Confidence blends hypothesis confidence, evidence volume, and contradictions.
        """
        hypotheses = list(hypotheses or [])
        evidence = evidence or []

        if not hypotheses:
            return Theory(statement="No hypotheses available to form a theory.")

        combined_statement = "; ".join(h.description for h in hypotheses if h.description)

        avg_confidence = float(np.mean([h.confidence for h in hypotheses])) if hypotheses else 0.5
        contradiction_text = [
            h.description for h in hypotheses if h.confirmed is False or len(h.contradicting_evidence) > 0
        ]
        evidence_strength = min(1.0, len(evidence) / 10)

        confidence = max(0.0, min(1.0, 0.6 * avg_confidence + 0.3 * evidence_strength - 0.1 * len(contradiction_text)))

        theory = Theory(
            statement=combined_statement or "Theory synthesized from available hypotheses.",
            supporting_evidence=evidence,
            confidence=confidence,
            contradictions=contradiction_text,
        )
        self._theory_history.append(theory)
        return theory

    def _extract_lat_lon(self, entry: dict[str, Any]) -> tuple[float | None, float | None]:
        """Pull latitude/longitude from varied exploration payloads."""
        location = entry.get("location") or entry.get("position") or {}
        lat = entry.get("lat") or entry.get("latitude") or location.get("lat") or location.get("latitude")
        lon = entry.get("lon") or entry.get("longitude") or location.get("lon") or location.get("longitude")
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None, None

    def _estimate_area_km2(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
        """Approximate coverage area from bounding box."""
        # Approximation: 1 degree lat ~ 111km, longitude scaled by latitude
        lat_km = abs(lat_max - lat_min) * 111.0
        avg_lat_rad = math.radians((lat_min + lat_max) / 2.0)
        lon_km = abs(lon_max - lon_min) * 111.0 * math.cos(avg_lat_rad)
        return max(0.0, lat_km * lon_km)

    def _build_ordered_path(self, exploration_data: list[dict[str, Any]]) -> list[tuple[float, float]]:
        """Order path using timestamps if present; fallback to input order."""
        def extract_time(item: dict[str, Any]) -> float:
            ts = item.get("timestamp") or item.get("observed_at")
            if isinstance(ts, datetime):
                return ts.timestamp()
            try:
                return float(ts)
            except (TypeError, ValueError):
                return 0.0

        sorted_items = sorted(exploration_data, key=extract_time)
        path: list[tuple[float, float]] = []
        for item in sorted_items[: self.max_points]:
            lat, lon = self._extract_lat_lon(item)
            if lat is None or lon is None:
                continue
            path.append((lat, lon))
        return path
