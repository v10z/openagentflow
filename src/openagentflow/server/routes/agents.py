"""Agent registry and execution endpoints."""

from __future__ import annotations

import asyncio
import uuid
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from openagentflow.server.models import (
    AgentListResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentSummary,
)
from openagentflow.server.ws import manager

router = APIRouter()


@router.get("/agents", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """List all registered agents."""
    from openagentflow import get_all_agents

    agents = get_all_agents()
    summaries = []
    for name, spec in agents.items():
        summaries.append(AgentSummary(
            name=spec.name,
            description=spec.description or "",
            model_id=spec.model.model_id if spec.model else "unknown",
            provider=spec.model.provider.value if spec.model else "unknown",
            tools=[t.name for t in (spec.tools or [])],
            reasoning_strategy=spec.reasoning_strategy.value if hasattr(spec, 'reasoning_strategy') and spec.reasoning_strategy else "react",
        ))
    return AgentListResponse(agents=summaries, total=len(summaries))


@router.post("/agents/{name}/run", response_model=AgentRunResponse)
async def run_agent(name: str, request: AgentRunRequest) -> AgentRunResponse:
    """Execute a registered agent."""
    from openagentflow import get_agent

    spec = get_agent(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    run_id = str(uuid.uuid4())[:8]

    # Broadcast start event
    await manager.broadcast("agent.started", {
        "run_id": run_id,
        "agent_name": name,
        "input": request.input_data,
    })

    start = time.time()
    try:
        func = spec.func
        if hasattr(func, "_async_call"):
            result = await func._async_call(**request.input_data)
        else:
            result = await func(**request.input_data)

        duration_ms = (time.time() - start) * 1000

        output = str(result.output) if hasattr(result, "output") else str(result)
        status = result.status.name if hasattr(result, "status") else "SUCCEEDED"
        error = result.error if hasattr(result, "error") else None
        total_tokens = result.total_tokens if hasattr(result, "total_tokens") else 0

        await manager.broadcast("agent.completed", {
            "run_id": run_id,
            "agent_name": name,
            "status": status,
            "duration_ms": duration_ms,
        })

        return AgentRunResponse(
            run_id=run_id,
            agent_name=name,
            status=status,
            output=output,
            error=error,
            duration_ms=duration_ms,
            total_tokens=total_tokens,
        )
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        await manager.broadcast("agent.error", {
            "run_id": run_id,
            "agent_name": name,
            "error": str(e),
        })
        return AgentRunResponse(
            run_id=run_id,
            agent_name=name,
            status="FAILED",
            error=str(e),
            duration_ms=duration_ms,
        )
