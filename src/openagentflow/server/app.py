"""FastAPI application factory for the OpenAgentFlow dashboard."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openagentflow.server.ws import manager

logger = logging.getLogger(__name__)

_start_time: float = 0.0


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global _start_time
    _start_time = time.time()

    app = FastAPI(
        title="OpenAgentFlow",
        description="Dashboard API for OpenAgentFlow -- autonomous AI agent framework",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    from openagentflow.server.routes import agents, health, memory, reasoning, tools, traces

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(agents.router, prefix="/api", tags=["agents"])
    app.include_router(tools.router, prefix="/api", tags=["tools"])
    app.include_router(reasoning.router, prefix="/api", tags=["reasoning"])
    app.include_router(traces.router, prefix="/api", tags=["traces"])
    app.include_router(memory.router, prefix="/api", tags=["memory"])

    # WebSocket endpoint
    @app.websocket("/ws/events")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, handle client messages if needed
                data = await websocket.receive_text()
                # Echo back or handle ping
                if data == "ping":
                    await websocket.send_text('{"type": "pong"}')
        except WebSocketDisconnect:
            await manager.disconnect(websocket)

    # Serve frontend static files (production build)
    frontend_dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        logger.info("Serving frontend from %s", frontend_dist)

    return app


def get_start_time() -> float:
    """Return the server start time for uptime calculation."""
    return _start_time
