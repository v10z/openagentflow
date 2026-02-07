"""Reasoning engine registry and execution endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from openagentflow.server.models import (
    EngineListResponse,
    EngineSummary,
    ReasoningRunRequest,
    ReasoningRunResponse,
    ReasoningStepResponse,
)
from openagentflow.server.ws import manager

router = APIRouter()

# Engine class names grouped by category
_CORE_ENGINES = {
    "DialecticalSpiral", "DreamWakeCycle", "MetaCognitiveLoop",
    "AdversarialSelfPlay", "EvolutionaryThought", "FractalRecursion",
    "ResonanceNetwork", "TemporalRecursion", "SimulatedAnnealing",
    "SocraticInterrogation",
}
_NEURO_ENGINES = {
    "PredictiveCoding", "GlobalWorkspace", "HebbianAssociation",
    "DefaultModeNetwork", "HippocampalReplay", "AttractorNetwork",
    "NeuralOscillation", "LateralInhibition", "BasalGangliaGating",
    "NeuromodulatorySweep",
}
_PHYSICS_ENGINES = {
    "SuperpositionCollapse", "WaveInterference", "PhaseTransition",
    "EntropicFunnel", "RenormalizationGroup", "GaugeInvariance",
    "PerturbativeExpansion", "LeastActionPath", "BarrierPenetration",
    "EntangledThreads",
}


def _get_category(name: str) -> str:
    if name in _CORE_ENGINES:
        return "core"
    elif name in _NEURO_ENGINES:
        return "neuroscience"
    elif name in _PHYSICS_ENGINES:
        return "physics"
    return "custom"


def _get_engine_classes() -> dict[str, type]:
    """Get all engine classes by name."""
    import openagentflow.reasoning as reasoning_mod
    from openagentflow.reasoning.base import ReasoningEngine

    engines = {}
    for attr_name in reasoning_mod.__all__:
        if attr_name in ("ReasoningEngine", "ReasoningStep", "ReasoningTrace"):
            continue
        cls = getattr(reasoning_mod, attr_name, None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, ReasoningEngine):
            engines[attr_name] = cls
    return engines


@router.get("/reasoning/engines", response_model=EngineListResponse)
async def list_engines() -> EngineListResponse:
    """List all available reasoning engines."""
    engines = _get_engine_classes()
    summaries = []
    for name, cls in sorted(engines.items()):
        instance = cls.__new__(cls)
        desc = getattr(instance, 'description', '') or getattr(cls, 'description', '')
        summaries.append(EngineSummary(
            name=name,
            description=desc,
            category=_get_category(name),
        ))
    return EngineListResponse(engines=summaries, total=len(summaries))


@router.post("/reasoning/run", response_model=ReasoningRunResponse)
async def run_engine(request: ReasoningRunRequest) -> ReasoningRunResponse:
    """Run a reasoning engine with the given query."""
    engines = _get_engine_classes()
    cls = engines.get(request.engine_name)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"Engine '{request.engine_name}' not found")

    # Create provider
    if request.provider == "mock":
        from openagentflow.llm.mock import MockProvider
        provider = MockProvider()
    else:
        raise HTTPException(status_code=400, detail=f"Provider '{request.provider}' not supported via API yet. Use 'mock' for testing.")

    engine = cls()

    await manager.broadcast("agent.started", {
        "run_id": request.engine_name,
        "agent_name": f"reasoning:{request.engine_name}",
        "query": request.query,
    })

    start = time.time()
    try:
        trace = await engine.reason(
            query=request.query,
            llm_provider=provider,
            max_iterations=request.max_iterations,
        )
        duration_ms = (time.time() - start) * 1000

        steps = [
            ReasoningStepResponse(
                step_id=s.step_id,
                step_type=s.step_type,
                content=s.content,
                score=s.score,
                metadata=s.metadata,
                parent_step_id=s.parent_step_id,
            )
            for s in trace.steps
        ]

        await manager.broadcast("agent.completed", {
            "run_id": request.engine_name,
            "agent_name": f"reasoning:{request.engine_name}",
            "duration_ms": duration_ms,
        })

        return ReasoningRunResponse(
            engine_name=request.engine_name,
            query=request.query,
            final_output=trace.final_output,
            steps=steps,
            total_llm_calls=trace.total_llm_calls,
            total_tokens=trace.total_tokens,
            duration_ms=duration_ms,
        )
    except Exception as e:
        await manager.broadcast("agent.error", {
            "run_id": request.engine_name,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))
