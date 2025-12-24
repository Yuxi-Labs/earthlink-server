"""API routes for Earthlink server."""

from fastapi import APIRouter

from src.api.agents import router as agents_router
from src.api.worlds import router as worlds_router
from src.api.simulation import router as simulation_router
from src.api.websocket import router as ws_router
from src.api.metrics import router as metrics_router
from src.api.social import router as social_router
from src.api.geo import router as geo_router

router = APIRouter()

router.include_router(agents_router, prefix="/agents", tags=["agents"])
router.include_router(worlds_router, prefix="/worlds", tags=["worlds"])
router.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
router.include_router(ws_router, prefix="/ws", tags=["websocket"])
router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
router.include_router(social_router, prefix="/social", tags=["social"])
router.include_router(geo_router, prefix="/geo", tags=["geo"])
