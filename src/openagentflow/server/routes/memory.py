"""Memory system inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from openagentflow.server.models import (
    MemoryEntryResponse,
    MemoryTierResponse,
)

router = APIRouter()


@router.get("/memory/{tier}", response_model=MemoryTierResponse)
async def get_memory_tier(tier: str) -> MemoryTierResponse:
    """Inspect memory contents for a given tier (fleeting, short_term, long_term)."""
    valid_tiers = {"fleeting", "short_term", "long_term"}
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(valid_tiers))}",
        )

    # Memory is typically per-session; return empty for now since there's no
    # persistent global memory instance in the server yet.
    # In production, this would connect to the MemoryManager singleton.
    return MemoryTierResponse(
        tier=tier,
        entries=[],
        total=0,
    )
