"""API routes for Earthlink server."""

from fastapi import APIRouter

from src.api.agents import router as agents_router
from src.api.worlds import router as worlds_router
from src.api.websocket import router as ws_router

router = APIRouter()

router.include_router(agents_router, prefix="/agents", tags=["agents"])
router.include_router(worlds_router, prefix="/worlds", tags=["worlds"])
router.include_router(ws_router, prefix="/ws", tags=["websocket"])
