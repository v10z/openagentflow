"""Pydantic response models for the OpenAgentFlow API."""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    agents_count: int
    tools_count: int
    engines_count: int
    uptime_seconds: float


class AgentSummary(BaseModel):
    name: str
    description: str
    model_id: str
    provider: str
    tools: list[str]
    reasoning_strategy: str


class AgentListResponse(BaseModel):
    agents: list[AgentSummary]
    total: int


class AgentRunRequest(BaseModel):
    input_data: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 10


class AgentRunResponse(BaseModel):
    run_id: str
    agent_name: str
    status: str
    output: str | None = None
    error: str | None = None
    duration_ms: float = 0
    total_tokens: int = 0
    trace_id: str | None = None


class ToolSummary(BaseModel):
    name: str
    description: str
    category: str
    parameters: dict[str, Any]


class ToolListResponse(BaseModel):
    tools: list[ToolSummary]
    total: int
    categories: list[str]


class ToolExecuteRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecuteResponse(BaseModel):
    tool_name: str
    result: Any
    duration_ms: float


class EngineSummary(BaseModel):
    name: str
    description: str
    category: str  # "core", "neuroscience", "physics"


class EngineListResponse(BaseModel):
    engines: list[EngineSummary]
    total: int


class ReasoningRunRequest(BaseModel):
    engine_name: str
    query: str
    provider: str = "mock"
    max_iterations: int = 10


class ReasoningStepResponse(BaseModel):
    step_id: str
    step_type: str
    content: str
    score: float
    metadata: dict[str, Any]
    parent_step_id: str | None = None


class ReasoningRunResponse(BaseModel):
    engine_name: str
    query: str
    final_output: str
    steps: list[ReasoningStepResponse]
    total_llm_calls: int
    total_tokens: int
    duration_ms: float


class TraceVertex(BaseModel):
    id: str
    type: str
    content: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""


class TraceEdge(BaseModel):
    source: str
    target: str
    label: str = "LEADS_TO"


class TraceSummary(BaseModel):
    trace_id: str
    strategy: str
    steps_count: int
    total_llm_calls: int
    total_tokens: int
    duration_ms: float
    final_output_preview: str  # truncated to 200 chars
    created_at: str


class TraceDetailResponse(BaseModel):
    trace_id: str
    strategy: str
    vertices: list[TraceVertex]
    edges: list[TraceEdge]
    total_llm_calls: int
    total_tokens: int
    duration_ms: float
    final_output: str


class TraceListResponse(BaseModel):
    traces: list[TraceSummary]
    total: int


class MemoryEntryResponse(BaseModel):
    key: str
    content: str
    importance: float
    access_count: int
    created_at: str
    last_accessed: str


class MemoryTierResponse(BaseModel):
    tier: str
    entries: list[MemoryEntryResponse]
    total: int
