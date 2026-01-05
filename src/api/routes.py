"""API routes for Earthlink server."""

from fastapi import APIRouter

from src.api.agents import router as agents_router
from src.api.earth import router as earth_router
from src.api.simulation import router as simulation_router
from src.api.websocket import router as ws_router
from src.api.metrics import router as metrics_router
from src.api.social import router as social_router
from src.api.capabilities import router as capabilities_router
from src.api.voice import router as voice_router
from src.api.weather import router as weather_router

router = APIRouter()

router.include_router(agents_router, prefix="/agents", tags=["agents"])
router.include_router(earth_router, prefix="/earth", tags=["earth"])
router.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
router.include_router(ws_router, prefix="/ws", tags=["websocket"])
router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
router.include_router(social_router, prefix="/social", tags=["social"])
router.include_router(capabilities_router, prefix="/capabilities", tags=["capabilities"])
router.include_router(voice_router, prefix="/voice", tags=["voice"])
router.include_router(weather_router, prefix="/weather", tags=["weather"])
