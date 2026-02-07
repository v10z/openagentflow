"""Temporal Recursion reasoning engine.

Reason from your future self's perspective, then incorporate the insights
back into the present plan.  Each "future perspective" (1 week, 1 month,
1 year) catches a different class of failure -- tactical mistakes, design
flaws, and strategic blunders respectively.

Phase overview::

    PRESENT            -> Generate initial solution / plan
    FUTURE PROJECTION  -> Simulate hindsight from each time horizon
    TIME TRAVEL        -> Fold future warnings into the present plan
    RE-PROJECT         -> Ask future self to re-evaluate the revised plan
    CONVERGE           -> Stop when future self has no more warnings

This is a formalised pre-mortem reasoning loop: assume the plan failed,
identify *why*, then prevent the failure before it happens.

Example::

    from openagentflow.reasoning.temporal import TemporalRecursion

    engine = TemporalRecursion(max_projections=3)
    trace = await engine.reason(
        query="Design a migration strategy for our monolith to microservices.",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openagentflow.core.types import Message, ToolSpec
from openagentflow.llm.base import BaseLLMProvider
from openagentflow.reasoning.base import ReasoningEngine, ReasoningTrace

logger = logging.getLogger(__name__)

_DEFAULT_PERSPECTIVES = [
    "1 week later",
    "1 month later",
    "1 year later",
]


class TemporalRecursion(ReasoningEngine):
    """Reason from your future self's perspective, then incorporate insights.

    The engine operates as a recursive pre-mortem loop:

    1. **PRESENT** -- Generate an initial solution or plan for the query.
    2. **FUTURE PROJECTION** -- For each configured time horizon (e.g.
       "1 week later", "1 month later", "1 year later"), imagine that the
       plan has been implemented and ask: "What went wrong?  What do you
       wish you had considered?"
    3. **TIME TRAVEL** -- Incorporate the future-self's warnings and
       insights back into the present plan, producing a revised version.
    4. **RE-PROJECTION** -- Repeat the future projection on the revised
       plan.  If the future self has no new warnings (or the maximum
       number of projection rounds has been reached), the loop terminates.
    5. **CONVERGENCE** -- Produce the final refined plan.

    Different time horizons are designed to surface different failure modes:

    - *Short-term* (1 week): implementation bugs, missing edge cases,
      incorrect assumptions about APIs or data.
    - *Medium-term* (1 month): integration issues, performance bottlenecks,
      team coordination problems, scope creep.
    - *Long-term* (1 year): architectural flaws, scalability limits,
      maintainability debt, shifting requirements.

    Attributes:
        name: ``"TemporalRecursion"``
        description: Short human-readable summary.
        max_projections: Maximum number of future-present iteration rounds.
        future_perspectives: List of time-horizon labels used for
            perspective projection.
        convergence_phrases: Key phrases that signal the future self has
            no further warnings.
    """

    name: str = "TemporalRecursion"
    description: str = (
        "Pre-mortem reasoning: project into the future, identify failures, "
        "then travel back to prevent them."
    )

    def __init__(
        self,
        max_projections: int = 3,
        future_perspectives: list[str] | None = None,
        convergence_phrases: list[str] | None = None,
    ) -> None:
        """Initialise the Temporal Recursion engine.

        Args:
            max_projections: Maximum number of full projection-revision
                cycles to execute.
            future_perspectives: Time-horizon labels.  Defaults to
                ``["1 week later", "1 month later", "1 year later"]``.
            convergence_phrases: Phrases in the future-self's response
                that indicate there are no more warnings.  If any of
                these phrases appear, the engine considers that horizon
                converged.
        """
        self.max_projections = max(1, max_projections)
        self.future_perspectives = future_perspectives or list(_DEFAULT_PERSPECTIVES)
        self.convergence_phrases = convergence_phrases or [
            "no further concerns",
            "no additional warnings",
            "looks good",
            "no new issues",
            "nothing else to add",
            "no remaining concerns",
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: BaseLLMProvider,
        tools: list[ToolSpec] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the full Temporal Recursion reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: An LLM provider for generating plans and
                projections.
            tools: Optional tool specs (currently unused by this engine).
            max_iterations: Hard cap on total projection rounds (combined
                with ``max_projections``).
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing present-plan, projection,
            time-travel, and convergence steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_max = min(self.max_projections, max_iterations)

        # Phase 1 -- PRESENT: generate initial plan
        current_plan = await self._generate_initial_plan(query, llm_provider, trace)

        # Phases 2-4 -- iterative projection + revision
        for iteration in range(effective_max):
            # Collect warnings from all future perspectives
            all_warnings: list[dict[str, Any]] = []
            any_new_warnings = False

            for perspective in self.future_perspectives:
                warnings = await self._project_future(
                    query, current_plan, perspective, iteration, llm_provider, trace
                )
                has_new = not self._is_converged(warnings)
                all_warnings.append({
                    "perspective": perspective,
                    "warnings": warnings,
                    "has_new_warnings": has_new,
                })
                if has_new:
                    any_new_warnings = True

            # Check convergence -- if no perspective has new warnings, stop
            if not any_new_warnings:
                conv_step = self._make_step(
                    step_type="convergence",
                    content=(
                        f"All future perspectives report no further concerns "
                        f"after iteration {iteration + 1}. Plan is stable."
                    ),
                    score=1.0,
                    metadata={
                        "phase": "converge",
                        "iteration": iteration + 1,
                        "perspectives_checked": len(self.future_perspectives),
                    },
                )
                trace.add_step(conv_step)
                logger.debug(
                    "TemporalRecursion: converged at iteration %d", iteration + 1
                )
                break

            # Phase 3 -- TIME TRAVEL: revise the plan with all warnings
            current_plan = await self._incorporate_warnings(
                query, current_plan, all_warnings, iteration, llm_provider, trace
            )

        # Phase 5 -- produce final polished output
        final_output = await self._finalise(query, current_plan, llm_provider, trace)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- PRESENT
    # ------------------------------------------------------------------

    async def _generate_initial_plan(
        self,
        query: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the initial solution / plan.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The initial plan text.
        """
        prompt = (
            f"You are a thorough planner. Given the following query, produce "
            f"a detailed initial plan or solution. Be specific and concrete.\n\n"
            f"Query: {query}\n\n"
            f"Provide your detailed plan:"
        )

        plan = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a meticulous strategic planner.",
        )

        step = self._make_step(
            step_type="present_plan",
            content=plan,
            score=0.5,
            metadata={"phase": "present", "iteration": 0},
        )
        trace.add_step(step)
        return plan

    # ------------------------------------------------------------------
    # Phase 2 -- FUTURE PROJECTION
    # ------------------------------------------------------------------

    async def _project_future(
        self,
        query: str,
        current_plan: str,
        perspective: str,
        iteration: int,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Project into the future from a specific time horizon.

        The LLM is asked to imagine it is now *perspective* (e.g.
        "1 month later") and the plan has been implemented.  It must
        identify what went wrong, what was overlooked, and what it wishes
        had been done differently.

        Args:
            query: Original user query.
            current_plan: The plan being evaluated.
            perspective: Time-horizon label (e.g. "1 year later").
            iteration: Current projection-revision iteration number.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The future-self's warnings and insights.
        """
        prompt = (
            f"You are looking back from {perspective} after this plan was "
            f"implemented.\n\n"
            f"Original query: {query}\n\n"
            f"The plan that was executed:\n{current_plan}\n\n"
            f"It is now {perspective}. Looking back:\n"
            f"1. What went wrong or what problems emerged?\n"
            f"2. What do you wish had been considered at the time?\n"
            f"3. What assumptions turned out to be incorrect?\n"
            f"4. What would you change if you could go back?\n\n"
            f"If the plan is genuinely robust and you have no further "
            f"concerns for this time horizon, say 'no further concerns' "
            f"and explain briefly why the plan holds up.\n\n"
            f"Be specific and honest."
        )

        warnings = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt=(
                f"You are an experienced practitioner reviewing from "
                f"the vantage point of {perspective}. Be critical and honest."
            ),
        )

        is_converged = self._is_converged(warnings)
        step = self._make_step(
            step_type="future_projection",
            content=warnings,
            score=0.0 if is_converged else 0.7,
            metadata={
                "phase": "project",
                "perspective": perspective,
                "iteration": iteration + 1,
                "converged": is_converged,
            },
        )
        trace.add_step(step)
        return warnings

    # ------------------------------------------------------------------
    # Phase 3 -- TIME TRAVEL (incorporate warnings)
    # ------------------------------------------------------------------

    async def _incorporate_warnings(
        self,
        query: str,
        current_plan: str,
        all_warnings: list[dict[str, Any]],
        iteration: int,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Revise the current plan by incorporating future-self warnings.

        Only warnings that contain genuine new concerns (i.e. where
        ``has_new_warnings`` is True) are included in the revision prompt.

        Args:
            query: Original user query.
            current_plan: The plan to revise.
            all_warnings: Warnings from all future perspectives.
            iteration: Current iteration number.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The revised plan text.
        """
        # Filter to only perspectives with real warnings
        actionable = [
            w for w in all_warnings if w.get("has_new_warnings", False)
        ]
        if not actionable:
            return current_plan

        warning_text = ""
        for w in actionable:
            warning_text += (
                f"\n--- From {w['perspective']} ---\n"
                f"{w['warnings']}\n"
            )

        prompt = (
            f"You are revising a plan based on insights from future "
            f"perspectives. Incorporate the warnings below to strengthen "
            f"the plan while keeping its core intent.\n\n"
            f"Original query: {query}\n\n"
            f"Current plan:\n{current_plan}\n\n"
            f"Warnings from future perspectives:{warning_text}\n\n"
            f"Produce an improved, revised plan that addresses these "
            f"concerns. Be specific about what changed and why."
        )

        revised = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a meticulous planner incorporating hindsight.",
        )

        step = self._make_step(
            step_type="time_travel",
            content=revised,
            score=0.5 + (iteration + 1) * 0.1,
            metadata={
                "phase": "time_travel",
                "iteration": iteration + 1,
                "warnings_incorporated": len(actionable),
                "perspectives_with_warnings": [
                    w["perspective"] for w in actionable
                ],
            },
        )
        trace.add_step(step)
        return revised

    # ------------------------------------------------------------------
    # Phase 5 -- FINALISE
    # ------------------------------------------------------------------

    async def _finalise(
        self,
        query: str,
        final_plan: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Produce a polished final output from the refined plan.

        The LLM is given the fully refined plan and asked to present it
        clearly and comprehensively.

        Args:
            query: Original user query.
            final_plan: The plan after all revision rounds.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final polished answer.
        """
        prompt = (
            f"You are presenting a final, refined plan that has been stress-"
            f"tested through multiple rounds of future-perspective analysis.\n\n"
            f"Original query: {query}\n\n"
            f"Refined plan:\n{final_plan}\n\n"
            f"Present this as a clear, well-structured final answer. "
            f"Highlight key decisions and the reasoning behind them. "
            f"Note any remaining risks or trade-offs."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a clear, comprehensive communicator.",
        )

        step = self._make_step(
            step_type="final_synthesis",
            content=final,
            score=1.0,
            metadata={"phase": "finalise"},
        )
        trace.add_step(step)
        return final

    # ------------------------------------------------------------------
    # Convergence check
    # ------------------------------------------------------------------

    def _is_converged(self, future_response: str) -> bool:
        """Check whether a future projection indicates convergence.

        Convergence is detected when any of the ``convergence_phrases``
        appear in the response (case-insensitive).

        Args:
            future_response: The text from a future projection.

        Returns:
            True if the response contains a convergence indicator.
        """
        lower = future_response.lower()
        return any(phrase in lower for phrase in self.convergence_phrases)


__all__ = ["TemporalRecursion"]
