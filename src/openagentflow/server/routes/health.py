"""Health check and system stats endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter

from openagentflow.server.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health and basic stats."""
    from openagentflow import __version__, get_all_agents, get_all_tools
    from openagentflow.server.app import get_start_time

    # Count reasoning engines
    from openagentflow.reasoning.base import ReasoningEngine
    engine_count = len(ReasoningEngine.__subclasses__())
    # Some engines might not be direct subclasses, count from __init__
    try:
        from openagentflow import reasoning
        engine_names = [name for name in reasoning.__all__ if name not in ("ReasoningEngine", "ReasoningStep", "ReasoningTrace")]
        engine_count = len(engine_names)
    except Exception:
        engine_count = 30

    return HealthResponse(
        status="ok",
        version=__version__,
        agents_count=len(get_all_agents()),
        tools_count=len(get_all_tools()),
        engines_count=engine_count,
        uptime_seconds=time.time() - get_start_time(),
    )
