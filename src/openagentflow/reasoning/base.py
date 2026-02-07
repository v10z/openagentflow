"""Base types and abstract interface for reasoning engines.

Provides:

- ``ReasoningStep`` -- a single unit of work inside a reasoning trace.
  Steps form a DAG via ``parent_step_id`` / ``children`` links.
- ``ReasoningTrace`` -- an ordered collection of steps plus bookkeeping.
  Offers graph-oriented helpers (``to_dag``, ``get_path``) so traces
  integrate naturally with OpenAgentFlow's graph backends.
- ``ReasoningEngine`` -- abstract base class that every concrete strategy
  must subclass.

All types are pure-stdlib dataclasses (no third-party deps).

Example::

    from openagentflow.reasoning import DialecticalSpiral

    engine = DialecticalSpiral(max_depth=3)
    trace = await engine.reason(
        query="What is the best approach to AI alignment?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
    print(f"Total LLM calls: {trace.total_llm_calls}")
    dag = trace.to_dag()
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ReasoningStep:
    """One discrete step in a reasoning trace.

    Each step captures one atomic unit of reasoning -- a thought, an
    observation after using a tool, a critique of a prior step, a synthesis
    of conflicting positions, etc.  Steps reference each other via
    ``parent_step_id`` and ``children`` to form a directed acyclic graph
    (DAG) that records the full structure of the reasoning process.

    Attributes:
        step_id: Short unique identifier for this step.
        step_type: Free-form label (e.g. ``"thought"``, ``"thesis"``,
            ``"antithesis"``, ``"synthesis"``, ``"dream"``,
            ``"evaluation"``, ``"critique"``, ``"judgment"``).
        content: The textual content produced during this step.
        metadata: Arbitrary key-value data attached to the step.
        timestamp: UTC timestamp of when the step was created.
        parent_step_id: Optional link to a parent step (tree/DAG structures).
        children: IDs of child steps, if any.
        score: Numeric score associated with this step (interpretation
            depends on the engine; typically 0.0--1.0).

    Example::

        step = ReasoningStep(
            step_type="thesis",
            content="AI alignment requires ...",
            metadata={"level": 0},
        )
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_type: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_step_id: str | None = None
    children: list[str] = field(default_factory=list)
    score: float = 0.0

    def add_child(self, child_step_id: str) -> None:
        """Register a child step id.

        Args:
            child_step_id: The ``step_id`` of the child to add.
        """
        if child_step_id not in self.children:
            self.children.append(child_step_id)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning session -- a DAG of ``ReasoningStep`` objects.

    Collects every ``ReasoningStep`` produced by an engine along with
    aggregate metrics (LLM calls, token usage, wall-clock time).

    The step DAG can be exported via :meth:`to_dag` for storage in
    OpenAgentFlow's graph backends or for visualisation.

    Attributes:
        steps: Ordered list of reasoning steps.
        strategy_name: Name of the reasoning strategy that produced this trace.
        total_llm_calls: Cumulative number of LLM invocations.
        total_tokens: Cumulative token consumption across all calls.
        duration_ms: Total wall-clock time in milliseconds.
        final_output: The final answer / solution text.

    Example::

        trace = ReasoningTrace(strategy_name="DialecticalSpiral")
        trace.add_step(my_step)
        dag = trace.to_dag()
        path = trace.get_path(some_step_id)
    """

    steps: list[ReasoningStep] = field(default_factory=list)
    strategy_name: str = ""
    total_llm_calls: int = 0
    total_tokens: int = 0
    duration_ms: float = 0
    final_output: str = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_index(self) -> dict[str, ReasoningStep]:
        """Build a look-up dict from ``step_id`` to ``ReasoningStep``."""
        return {s.step_id: s for s in self.steps}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(self, step: ReasoningStep) -> None:
        """Append a step to the trace, wiring parent/child links.

        If the step has a ``parent_step_id`` that corresponds to an
        existing step in the trace, the parent's ``children`` list is
        updated automatically.

        Args:
            step: The reasoning step to record.
        """
        self.steps.append(step)
        if step.parent_step_id is not None:
            idx = self._step_index()
            parent = idx.get(step.parent_step_id)
            if parent is not None:
                parent.add_child(step.step_id)

    def get_roots(self) -> list[ReasoningStep]:
        """Return all root steps (steps with no parent).

        Returns:
            List of ``ReasoningStep`` objects that have
            ``parent_step_id is None``.
        """
        return [s for s in self.steps if s.parent_step_id is None]

    def get_path(self, step_id: str) -> list[ReasoningStep]:
        """Get the path from a root to the step identified by *step_id*.

        Walks up the ``parent_step_id`` chain until a root step is reached,
        then returns the path in root-to-target order.

        Args:
            step_id: The ``step_id`` of the target step.

        Returns:
            Ordered list of ``ReasoningStep`` from root to target
            (inclusive).

        Raises:
            KeyError: If *step_id* is not found in the trace.

        Example::

            path = trace.get_path("abc123")
            for step in path:
                print(f"  [{step.step_type}] {step.content[:80]}")
        """
        idx = self._step_index()
        if step_id not in idx:
            raise KeyError(f"Step {step_id!r} not found in trace")

        path: list[ReasoningStep] = []
        current_id: str | None = step_id
        while current_id is not None:
            step = idx[current_id]
            path.append(step)
            current_id = step.parent_step_id
        path.reverse()
        return path

    def to_dag(self) -> dict[str, Any]:
        """Convert the trace to a DAG dictionary for graph storage.

        Returns a dict with two top-level keys:

        - ``"vertices"``: list of dicts, one per step, with fields
          ``id``, ``type``, ``content``, ``score``, ``metadata``,
          ``timestamp``.
        - ``"edges"``: list of dicts with ``source``, ``target``,
          ``label`` (always ``"LEADS_TO"``).

        Plus aggregate metadata: ``strategy``, ``total_llm_calls``,
        ``total_tokens``, ``duration_ms``, ``final_output``.

        This format integrates directly with OpenAgentFlow's
        ``GraphBackend.add_vertex`` / ``GraphBackend.add_edge``.

        Returns:
            A dictionary with ``"vertices"`` and ``"edges"`` lists.

        Example::

            dag = trace.to_dag()
            for v in dag["vertices"]:
                await backend.add_vertex(v["id"], v["type"], v)
            for e in dag["edges"]:
                await backend.add_edge(e["source"], e["target"], e["label"])
        """
        vertices: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for step in self.steps:
            vertices.append({
                "id": step.step_id,
                "type": step.step_type,
                "content": step.content,
                "score": step.score,
                "metadata": step.metadata,
                "timestamp": step.timestamp.isoformat(),
            })
            if step.parent_step_id is not None:
                edges.append({
                    "source": step.parent_step_id,
                    "target": step.step_id,
                    "label": "LEADS_TO",
                })

        return {
            "vertices": vertices,
            "edges": edges,
            "strategy": self.strategy_name,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "final_output": self.final_output,
        }

    def get_steps_by_type(self, step_type: str) -> list[ReasoningStep]:
        """Return all steps of a given type.

        Args:
            step_type: The ``step_type`` label to filter on (e.g.
                ``"thesis"``, ``"dream"``).

        Returns:
            List of matching ``ReasoningStep`` objects, in trace order.
        """
        return [s for s in self.steps if s.step_type == step_type]

    def to_dict(self) -> dict[str, Any]:
        """Return a lightweight dictionary summary of the trace."""
        return {
            "strategy": self.strategy_name,
            "steps": len(self.steps),
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "output": self.final_output,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the trace.

        Returns:
            A multi-line string with strategy name, step count, LLM calls,
            tokens, duration, and truncated final output.
        """
        output_preview = (
            self.final_output[:200] + "..."
            if len(self.final_output) > 200
            else self.final_output
        )
        return (
            f"ReasoningTrace(strategy={self.strategy_name!r}, "
            f"steps={len(self.steps)}, "
            f"llm_calls={self.total_llm_calls}, "
            f"tokens={self.total_tokens}, "
            f"duration_ms={self.duration_ms:.1f})\n"
            f"Output: {output_preview}"
        )


# ---------------------------------------------------------------------------
# Abstract engine
# ---------------------------------------------------------------------------


class ReasoningEngine(ABC):
    """Abstract base for all reasoning engines.

    Subclasses must implement :meth:`reason`.  The helper
    :meth:`_call_llm` handles the mechanics of calling a
    ``BaseLLMProvider`` and updating the trace metrics.

    Subclasses MUST set ``name`` and ``description`` (either as class
    attributes or in ``__init__``).

    Example (implementing a custom engine)::

        class MyEngine(ReasoningEngine):
            name = "MyEngine"
            description = "Does something clever."

            async def reason(self, query, llm_provider, **kwargs):
                trace = ReasoningTrace(strategy_name=self.name)
                response = await self._call_llm(
                    provider=llm_provider,
                    messages=[{"role": "user", "content": query}],
                    trace=trace,
                    system="You are a helpful assistant.",
                    temperature=0.7,
                )
                trace.final_output = response
                return trace
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: Any | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Run the reasoning strategy on *query* and return a trace.

        Args:
            query: The user question or problem statement.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Optional tool specs available to the engine.
            max_iterations: Safety cap on LLM round-trips.
            **kwargs: Engine-specific options.

        Returns:
            A ``ReasoningTrace`` containing all steps and the final output.
        """
        ...

    async def _call_llm(
        self,
        provider: Any,
        messages: list[dict[str, str]],
        trace: ReasoningTrace,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Convenience wrapper around a provider's ``generate`` method.

        Converts plain dicts into ``Message`` objects, builds a minimal
        ``ModelConfig``, calls the provider, and updates *trace* metrics.

        This is the **only** path through which reasoning engines should
        invoke the LLM.  It ensures that ``trace.total_llm_calls`` and
        ``trace.total_tokens`` are always accurate.

        Args:
            provider: A ``BaseLLMProvider`` instance.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            trace: The current reasoning trace (mutated in place).
            system: Optional system prompt override.
            temperature: Optional temperature override.

        Returns:
            The text content of the LLM response.
        """
        from openagentflow.core.types import LLMProvider, Message, ModelConfig

        # Determine the model id from the provider (best-effort).
        model_id: str = getattr(provider, "model", None) or getattr(
            provider, "default_model", "default"
        )

        # Determine the LLMProvider enum value (best-effort).
        provider_enum = LLMProvider.ANTHROPIC  # safe default
        pname = getattr(provider, "provider_name", "")
        for member in LLMProvider:
            if member.value == pname:
                provider_enum = member
                break

        config = ModelConfig(
            provider=provider_enum,
            model_id=model_id,
        )
        if temperature is not None:
            config.temperature = temperature

        msg_objects = [
            Message(role=m["role"], content=m["content"]) for m in messages
        ]

        result = await provider.generate(
            messages=msg_objects,
            config=config,
            system_prompt=system,
        )

        trace.total_llm_calls += 1
        trace.total_tokens += result.total_tokens

        return result.content
