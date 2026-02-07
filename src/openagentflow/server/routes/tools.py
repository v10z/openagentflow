"""Tool registry and execution endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from openagentflow.server.models import (
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolListResponse,
    ToolSummary,
)

router = APIRouter()

# Category mapping based on tool module paths
CATEGORY_MAP = {
    "text": "Text Processing",
    "code": "Code Analysis",
    "data": "Data Transform",
    "web": "Web/HTTP",
    "math": "Math/Science",
    "media": "Media",
    "datetime": "Date/Time",
    "ai": "AI/ML Helpers",
    "system": "System/File",
}


def _infer_category(tool_name: str) -> str:
    """Infer tool category from its module or name."""
    from openagentflow import get_tool
    spec = get_tool(tool_name)
    if spec and hasattr(spec, 'func'):
        module = getattr(spec.func, '__module__', '')
        for key, label in CATEGORY_MAP.items():
            if f".tools.{key}" in module:
                return label
    return "Other"


@router.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """List all registered tools."""
    from openagentflow import get_all_tools

    all_tools = get_all_tools()
    summaries = []
    categories_seen: set[str] = set()

    for name, spec in all_tools.items():
        category = _infer_category(name)
        categories_seen.add(category)
        summaries.append(ToolSummary(
            name=spec.name,
            description=spec.description or "",
            category=category,
            parameters=spec.parameters if hasattr(spec, 'parameters') and spec.parameters else {},
        ))

    return ToolListResponse(
        tools=summaries,
        total=len(summaries),
        categories=sorted(categories_seen),
    )


@router.post("/tools/{name}/execute", response_model=ToolExecuteResponse)
async def execute_tool(name: str, request: ToolExecuteRequest) -> ToolExecuteResponse:
    """Execute a tool by name with given arguments."""
    from openagentflow import get_tool, execute_tool as exec_tool

    spec = get_tool(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

    start = time.time()
    try:
        result = await exec_tool(name, request.arguments)
        duration_ms = (time.time() - start) * 1000
        return ToolExecuteResponse(
            tool_name=name,
            result=result,
            duration_ms=duration_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
