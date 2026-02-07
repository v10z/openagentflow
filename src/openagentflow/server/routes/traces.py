"""Trace storage and retrieval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from openagentflow.server.models import (
    TraceDetailResponse,
    TraceEdge,
    TraceListResponse,
    TraceSummary,
    TraceVertex,
)

router = APIRouter()

# In-memory trace store (for demo; production would use SQLiteGraphBackend)
_trace_store: dict[str, dict] = {}


def store_trace(trace_id: str, trace_data: dict) -> None:
    """Store a trace for later retrieval."""
    _trace_store[trace_id] = trace_data


@router.get("/traces", response_model=TraceListResponse)
async def list_traces() -> TraceListResponse:
    """List all stored reasoning traces."""
    summaries = []
    for trace_id, data in _trace_store.items():
        final_output = data.get("final_output", "")
        preview = final_output[:200] + "..." if len(final_output) > 200 else final_output
        summaries.append(TraceSummary(
            trace_id=trace_id,
            strategy=data.get("strategy", "unknown"),
            steps_count=len(data.get("vertices", [])),
            total_llm_calls=data.get("total_llm_calls", 0),
            total_tokens=data.get("total_tokens", 0),
            duration_ms=data.get("duration_ms", 0),
            final_output_preview=preview,
            created_at=data.get("created_at", ""),
        ))
    return TraceListResponse(traces=summaries, total=len(summaries))


@router.get("/traces/{trace_id}", response_model=TraceDetailResponse)
async def get_trace(trace_id: str) -> TraceDetailResponse:
    """Get full trace detail with DAG vertices and edges."""
    data = _trace_store.get(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")

    vertices = [
        TraceVertex(
            id=v["id"],
            type=v.get("type", ""),
            content=v.get("content", ""),
            score=v.get("score", 0.0),
            metadata=v.get("metadata", {}),
            timestamp=v.get("timestamp", ""),
        )
        for v in data.get("vertices", [])
    ]

    edges = [
        TraceEdge(
            source=e["source"],
            target=e["target"],
            label=e.get("label", "LEADS_TO"),
        )
        for e in data.get("edges", [])
    ]

    return TraceDetailResponse(
        trace_id=trace_id,
        strategy=data.get("strategy", "unknown"),
        vertices=vertices,
        edges=edges,
        total_llm_calls=data.get("total_llm_calls", 0),
        total_tokens=data.get("total_tokens", 0),
        duration_ms=data.get("duration_ms", 0),
        final_output=data.get("final_output", ""),
    )
