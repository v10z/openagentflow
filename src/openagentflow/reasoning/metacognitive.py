"""Meta-Cognitive Loop reasoning engine.

Implements a three-level self-monitoring reasoning strategy:

1. **Object level** -- works on the actual problem using a configurable
   sub-strategy (step-by-step, decompose, analogize, contrast, or direct).
2. **Meta level** -- periodically evaluates whether the object-level
   reasoning is making progress, detecting loops or narrow/broad
   exploration, and switches strategy when warranted.
3. **Meta-meta level** -- periodically audits the meta-level decisions
   themselves to recalibrate monitoring frequency and thresholds.

The key capability is *dynamic strategy switching*: if chain-of-thought is
failing, the engine can switch to decomposition; if that is too slow, it
can switch to direct answering.  Every decision is recorded as a
:class:`~openagentflow.reasoning.base.ReasoningStep` inside a
:class:`~openagentflow.reasoning.base.ReasoningTrace`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

# ---------------------------------------------------------------------------
# Prompts used by the three cognitive levels
# ---------------------------------------------------------------------------

_OBJECT_SYSTEM = (
    "You are a careful reasoning assistant. You are working on a problem "
    "using a specific reasoning strategy. Follow the strategy faithfully "
    "and show your work."
)

_META_SYSTEM = (
    "You are a meta-cognitive monitor.  Your job is to evaluate the "
    "reasoning process of another agent (the 'object level').  You will "
    "receive the original query and the reasoning steps taken so far. "
    "You must respond ONLY with a JSON object (no markdown fences) with "
    "these exact keys:\n"
    '  "progress": <float 0.0-1.0>,\n'
    '  "stuck": <bool>,\n'
    '  "switch_to": <string strategy name or null>,\n'
    '  "reason": <string explanation>\n'
    "Available strategies: {strategies}\n"
    "Current strategy: {current_strategy}\n"
    "Progress threshold (below which we consider switching): {threshold}"
)

_META_META_SYSTEM = (
    "You are a meta-meta-cognitive auditor.  You review the decisions made "
    "by the meta-level monitor to determine whether its evaluations have "
    "been accurate and helpful.  Respond ONLY with a JSON object (no "
    "markdown fences):\n"
    '  "meta_accuracy": <float 0.0-1.0>,\n'
    '  "adjust_interval": <int or null>,\n'
    '  "notes": <string explanation>'
)

_SYNTHESIS_SYSTEM = (
    "You are a synthesis assistant.  Given a query and a sequence of "
    "reasoning steps (possibly using different strategies), produce a "
    "clear, concise final answer."
)


def _strategy_prompt(strategy: str) -> str:
    """Return an instruction snippet for the given object-level strategy."""
    prompts: dict[str, str] = {
        "direct": (
            "Answer the question directly and concisely.  Do not show "
            "intermediate work unless absolutely necessary."
        ),
        "step_by_step": (
            "Think step by step.  Number each step and clearly explain "
            "your reasoning before giving a conclusion."
        ),
        "decompose": (
            "Break the problem into smaller sub-problems.  Solve each "
            "sub-problem independently, then combine the results."
        ),
        "analogize": (
            "Think of an analogous, simpler problem first.  Solve the "
            "analogy, then map the solution back to the original problem."
        ),
        "contrast": (
            "Consider multiple competing answers or viewpoints.  Argue "
            "for and against each, then select the strongest."
        ),
    }
    return prompts.get(strategy, prompts["step_by_step"])


class MetaCognitiveLoop(ReasoningEngine):
    """A reasoning engine that monitors its own reasoning process and adapts.

    Three levels of cognition work together:

    * **Object level** -- Solves the actual problem using a configurable
      sub-strategy (e.g. ``"step_by_step"``, ``"decompose"``).
    * **Meta level** -- Every *monitor_interval* steps, evaluates progress.
      If progress is below *progress_threshold*, switches the object-level
      strategy.  Also detects reasoning loops.
    * **Meta-meta level** -- Every *meta_meta_interval* steps, audits the
      meta-level's decisions to recalibrate monitoring frequency.

    The key capability is **dynamic strategy switching**: if step-by-step
    reasoning is stuck, the engine switches to decomposition; if that is
    too slow, it can try analogize, contrast, or direct answering.

    Args:
        monitor_interval: How many object-level steps between meta checks.
        meta_meta_interval: How many object-level steps between meta-meta
            audits (should be a multiple of *monitor_interval*).
        progress_threshold: Progress score (0--1) below which the meta
            level will recommend a strategy switch.
        available_strategies: List of strategy names the engine can switch
            between.  Defaults to ``["direct", "step_by_step", "decompose",
            "analogize", "contrast"]``.

    Example::

        engine = MetaCognitiveLoop(monitor_interval=3, progress_threshold=0.3)
        trace = await engine.reason(
            query="Explain why the sky is blue",
            llm_provider=my_provider,
            max_iterations=12,
        )
        print(trace.final_output)
    """

    name: str = "meta_cognitive_loop"
    description: str = (
        "Self-monitoring reasoning with dynamic strategy switching across "
        "three cognitive levels: object, meta, and meta-meta."
    )

    def __init__(
        self,
        monitor_interval: int = 3,
        meta_meta_interval: int = 9,
        progress_threshold: float = 0.3,
        available_strategies: list[str] | None = None,
    ) -> None:
        self.monitor_interval = monitor_interval
        self.meta_meta_interval = meta_meta_interval
        self.progress_threshold = progress_threshold
        self.available_strategies: list[str] = available_strategies or [
            "direct",
            "step_by_step",
            "decompose",
            "analogize",
            "contrast",
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Run the meta-cognitive reasoning loop.

        Args:
            query: The question or task to reason about.
            llm_provider: A :class:`BaseLLMProvider` instance.
            tools: Optional tool specifications (reserved for future use).
            max_iterations: Maximum number of object-level steps.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A :class:`ReasoningTrace` with all object, meta, and meta-meta
            steps plus the final synthesized answer.
        """
        start = time.perf_counter()
        trace = ReasoningTrace(strategy_name=self.name)

        current_strategy = "step_by_step"
        object_history: list[str] = []  # collected object-level outputs
        meta_history: list[dict[str, Any]] = []  # collected meta evaluations
        strategy_switches: list[dict[str, str]] = []
        consecutive_stuck = 0

        for iteration in range(1, max_iterations + 1):
            # ----- 1. OBJECT LEVEL -----
            obj_output = await self._object_step(
                query=query,
                history=object_history,
                strategy=current_strategy,
                provider=llm_provider,
                trace=trace,
                iteration=iteration,
            )
            object_history.append(obj_output)

            # ----- 2. META LEVEL (every monitor_interval steps) -----
            if iteration % self.monitor_interval == 0:
                meta_result = await self._meta_evaluate(
                    query=query,
                    history=object_history,
                    strategy=current_strategy,
                    provider=llm_provider,
                    trace=trace,
                    iteration=iteration,
                )
                meta_history.append(meta_result)

                if meta_result.get("stuck", False):
                    consecutive_stuck += 1
                else:
                    consecutive_stuck = 0

                # Strategy switch decision
                progress = meta_result.get("progress", 0.5)
                suggested = meta_result.get("switch_to")
                if (
                    progress < self.progress_threshold
                    or consecutive_stuck >= 2
                ) and suggested and suggested != current_strategy:
                    if suggested in self.available_strategies:
                        old_strategy = current_strategy
                        current_strategy = suggested
                        strategy_switches.append(
                            {"from": old_strategy, "to": current_strategy}
                        )
                        switch_step = ReasoningStep(
                            step_type="strategy_switch",
                            content=(
                                f"Switching strategy from '{old_strategy}' to "
                                f"'{current_strategy}'. Reason: "
                                f"{meta_result.get('reason', 'low progress')}"
                            ),
                            metadata={
                                "from": old_strategy,
                                "to": current_strategy,
                                "progress": progress,
                                "iteration": iteration,
                            },
                            score=progress,
                        )
                        trace.add_step(switch_step)
                        consecutive_stuck = 0

                # Check if meta-level believes we have a good answer
                if progress >= 0.9 and not meta_result.get("stuck", False):
                    # High confidence -- synthesize early
                    break

            # ----- 3. META-META LEVEL (every meta_meta_interval steps) -----
            if iteration % self.meta_meta_interval == 0 and meta_history:
                mm_result = await self._meta_meta_evaluate(
                    query=query,
                    meta_history=meta_history,
                    provider=llm_provider,
                    trace=trace,
                    iteration=iteration,
                )
                # Adjust monitoring interval if recommended
                new_interval = mm_result.get("adjust_interval")
                if (
                    new_interval is not None
                    and isinstance(new_interval, int)
                    and 1 <= new_interval <= max_iterations
                ):
                    self.monitor_interval = new_interval
                    adjust_step = ReasoningStep(
                        step_type="interval_adjustment",
                        content=(
                            f"Meta-meta audit adjusted monitor interval to "
                            f"{new_interval}. Notes: {mm_result.get('notes', '')}"
                        ),
                        metadata={
                            "new_interval": new_interval,
                            "meta_accuracy": mm_result.get("meta_accuracy", 0.0),
                            "iteration": iteration,
                        },
                    )
                    trace.add_step(adjust_step)

        # ----- SYNTHESIS -----
        final_output = await self._synthesize(
            query=query,
            history=object_history,
            strategy_switches=strategy_switches,
            provider=llm_provider,
            trace=trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.perf_counter() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Object level
    # ------------------------------------------------------------------

    async def _object_step(
        self,
        query: str,
        history: list[str],
        strategy: str,
        provider: Any,
        trace: ReasoningTrace,
        iteration: int,
    ) -> str:
        """Execute one object-level reasoning step.

        Asks the LLM to continue reasoning about *query* according to the
        current *strategy*, given what has been produced so far (*history*).

        Args:
            query: The original question.
            history: Previous object-level outputs.
            strategy: The active strategy name.
            provider: LLM provider.
            trace: Active reasoning trace.
            iteration: Current iteration number.

        Returns:
            The LLM's output for this step.
        """
        strategy_instruction = _strategy_prompt(strategy)

        history_text = ""
        if history:
            numbered = [f"Step {i+1}: {h}" for i, h in enumerate(history)]
            history_text = (
                "Here is your reasoning so far:\n"
                + "\n".join(numbered)
                + "\n\nContinue reasoning. Produce the next step."
            )
        else:
            history_text = (
                "Begin reasoning about the query. Produce the first step."
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Strategy: {strategy_instruction}\n\n"
                    f"Query: {query}\n\n"
                    f"{history_text}"
                ),
            }
        ]

        output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_OBJECT_SYSTEM,
            temperature=0.7,
        )

        step = ReasoningStep(
            step_type="thought",
            content=output,
            metadata={"strategy": strategy, "iteration": iteration, "level": "object"},
        )
        trace.add_step(step)

        return output

    # ------------------------------------------------------------------
    # Meta level
    # ------------------------------------------------------------------

    async def _meta_evaluate(
        self,
        query: str,
        history: list[str],
        strategy: str,
        provider: Any,
        trace: ReasoningTrace,
        iteration: int,
    ) -> dict[str, Any]:
        """Evaluate reasoning progress and decide on strategy changes.

        The meta level answers questions like:

        * Am I stuck in a loop?
        * Am I exploring too narrowly or broadly?
        * Should I switch strategies?
        * Is my confidence calibrated?

        Args:
            query: The original question.
            history: Object-level outputs so far.
            strategy: Current active strategy.
            provider: LLM provider.
            trace: Active reasoning trace.
            iteration: Current iteration number.

        Returns:
            A dictionary with keys ``progress`` (float 0--1), ``stuck``
            (bool), ``switch_to`` (str or None), and ``reason`` (str).
        """
        system = _META_SYSTEM.format(
            strategies=", ".join(self.available_strategies),
            current_strategy=strategy,
            threshold=self.progress_threshold,
        )

        numbered = [f"Step {i+1}: {h}" for i, h in enumerate(history)]
        history_block = "\n".join(numbered)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Current strategy: {strategy}\n\n"
                    f"Reasoning steps so far:\n{history_block}\n\n"
                    "Evaluate the reasoning process.  Are we making progress? "
                    "Are we stuck in a loop?  Should we switch strategies?  "
                    "Respond ONLY with the required JSON object."
                ),
            }
        ]

        raw = await self._call_llm(
            provider,
            messages,
            trace,
            system=system,
            temperature=0.3,
        )

        result = self._parse_json_safe(raw, {
            "progress": 0.5,
            "stuck": False,
            "switch_to": None,
            "reason": raw,
        })

        step = ReasoningStep(
            step_type="meta",
            content=f"Meta evaluation: {json.dumps(result)}",
            metadata={
                "level": "meta",
                "iteration": iteration,
                "progress": result.get("progress", 0.5),
                "stuck": result.get("stuck", False),
                "switch_to": result.get("switch_to"),
            },
            score=result.get("progress", 0.5),
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Meta-meta level
    # ------------------------------------------------------------------

    async def _meta_meta_evaluate(
        self,
        query: str,
        meta_history: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        iteration: int,
    ) -> dict[str, Any]:
        """Evaluate the quality of meta-level monitoring itself.

        Answers questions like:

        * Is my self-monitoring helping or hurting?
        * Am I being too cautious or too reckless in switching?
        * Should the monitoring frequency change?

        Args:
            query: The original question.
            meta_history: List of meta-evaluation result dicts.
            provider: LLM provider.
            trace: Active reasoning trace.
            iteration: Current iteration number.

        Returns:
            A dictionary with keys ``meta_accuracy`` (float 0--1),
            ``adjust_interval`` (int or None), and ``notes`` (str).
        """
        meta_summary = json.dumps(meta_history, default=str)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Meta-level decisions so far:\n{meta_summary}\n\n"
                    "Review these meta-level decisions.  Were strategy switches "
                    "helpful?  Is the monitoring frequency (currently every "
                    f"{self.monitor_interval} steps) correct?  Should we trust "
                    "our own confidence assessments?  Respond ONLY with the "
                    "required JSON object."
                ),
            }
        ]

        raw = await self._call_llm(
            provider,
            messages,
            trace,
            system=_META_META_SYSTEM,
            temperature=0.3,
        )

        result = self._parse_json_safe(raw, {
            "meta_accuracy": 0.5,
            "adjust_interval": None,
            "notes": raw,
        })

        step = ReasoningStep(
            step_type="meta_meta",
            content=f"Meta-meta audit: {json.dumps(result)}",
            metadata={
                "level": "meta_meta",
                "iteration": iteration,
                "meta_accuracy": result.get("meta_accuracy", 0.5),
                "adjust_interval": result.get("adjust_interval"),
            },
            score=result.get("meta_accuracy", 0.5),
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        history: list[str],
        strategy_switches: list[dict[str, str]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Produce the final answer by synthesizing all reasoning steps.

        Args:
            query: The original question.
            history: All object-level outputs.
            strategy_switches: Record of any strategy changes that occurred.
            provider: LLM provider.
            trace: Active reasoning trace.

        Returns:
            The final synthesized answer string.
        """
        numbered = [f"Step {i+1}: {h}" for i, h in enumerate(history)]
        history_block = "\n".join(numbered)

        switch_info = ""
        if strategy_switches:
            switch_lines = [
                f"  - Switched from '{s['from']}' to '{s['to']}'"
                for s in strategy_switches
            ]
            switch_info = (
                "\n\nNote: the reasoning strategy was adjusted during the "
                "process:\n" + "\n".join(switch_lines)
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Reasoning steps:\n{history_block}"
                    f"{switch_info}\n\n"
                    "Based on all reasoning above, provide a clear, concise, "
                    "and well-structured final answer to the query."
                ),
            }
        ]

        output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.4,
        )

        step = ReasoningStep(
            step_type="synthesis",
            content=output,
            metadata={"level": "synthesis"},
            score=1.0,
        )
        trace.add_step(step)

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_safe(text: str, default: dict[str, Any]) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output.

        Handles common issues like markdown fences, trailing commas, and
        completely non-JSON responses.

        Args:
            text: Raw LLM output that should contain a JSON object.
            default: Fallback dict to return if parsing fails entirely.

        Returns:
            Parsed dictionary, or *default* on failure.
        """
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            # Remove opening fence (optionally with language tag)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3]
            cleaned = cleaned.strip()

        # Try to find a JSON object in the text
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Last resort: try the whole string
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return default
