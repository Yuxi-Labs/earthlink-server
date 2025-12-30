"""Earthlink API Server - Main entry point."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.api.websocket import manager as ws_manager
from src.api.event_bridge import init_event_bridge, EventBridge
from src.api.messaging import router as messaging_router
from src.api.geo import router as geo_router
from src.api.analytics import router as analytics_router
from src.agents.core import MessageRouter, init_message_router, get_message_router
from src.config import settings
from src.db.database import init_db
from src.simulation import SimulationRunner, SimulationConfig


# Global instances
_simulation_runner: SimulationRunner | None = None
_event_bridge: EventBridge | None = None
_message_router: MessageRouter | None = None
_broadcast_task: asyncio.Task | None = None


def get_simulation_runner() -> SimulationRunner | None:
    """Get the global simulation runner instance."""
    return _simulation_runner


def get_event_bridge() -> EventBridge | None:
    """Get the global event bridge instance."""
    return _event_bridge


def get_message_router_instance() -> MessageRouter | None:
    """Get the global message router instance."""
    return _message_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _simulation_runner, _event_bridge, _message_router, _broadcast_task

    # Startup
    await init_db()

    # Initialize event bridge for WebSocket broadcasting
    _event_bridge = init_event_bridge(ws_manager)

    # Start WebSocket broadcast worker
    _broadcast_task = asyncio.create_task(ws_manager.start_broadcast_worker())

    # Initialize simulation runner
    config = SimulationConfig(
        steps_per_second=getattr(settings, 'simulation_steps_per_second', 10.0),
        checkpoint_dir=getattr(settings, 'checkpoint_dir', 'checkpoints'),
    )
    _simulation_runner = SimulationRunner(config=config)
    await _simulation_runner.initialize()

    # Register event bridge with simulation's event bus
    if _simulation_runner.event_bus and _event_bridge:
        _event_bridge.register_with_event_bus(_simulation_runner.event_bus)

    # Initialize message router with simulation's event bus
    _message_router = init_message_router(_simulation_runner.event_bus)

    # Auto-start simulation
    asyncio.create_task(_simulation_runner.run())

    yield

    # Shutdown
    # Stop broadcast worker
    ws_manager.stop_broadcast_worker()
    if _broadcast_task:
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass
        _broadcast_task = None

    # Shutdown simulation
    if _simulation_runner:
        if _event_bridge:
            _event_bridge.unregister_from_event_bus(_simulation_runner.event_bus)
        await _simulation_runner.shutdown()
        _simulation_runner = None

    _event_bridge = None
    _message_router = None


app = FastAPI(
    title="Earthlink API",
    description="Backend for training autonomous agents as cross-world explorers",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for Tauri client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
from src.api.commands import router as commands_router
from src.api.episodes import router as episodes_router
from src.api.snapshots import router as snapshots_router
from src.api.websocket_commands import router as websocket_commands_router
from src.api.collaboration import router as collaboration_router
from src.api.health import router as health_router

app.include_router(api_router, prefix="/api/v1")
app.include_router(messaging_router, prefix="/api/v1/messaging", tags=["messaging"])
app.include_router(geo_router, prefix="/geo", tags=["geo"])
app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])
app.include_router(commands_router, prefix="/api/v1/commands", tags=["commands"])
app.include_router(episodes_router, prefix="/api/v1/episodes", tags=["episodes"])
app.include_router(snapshots_router, prefix="/api/v1/snapshots", tags=["snapshots"])
app.include_router(websocket_commands_router, prefix="/ws", tags=["websocket-commands"])
app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["collaboration"])
app.include_router(health_router, tags=["health"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "earthlink-api"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "name": "Earthlink API",
        "version": "0.1.0",
        "description": "Autonomous agents exploring human and non-human realities",
        "docs": "/docs",
    }
