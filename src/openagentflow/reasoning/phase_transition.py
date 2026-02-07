"""Phase Transition reasoning engine.

Applies Landau theory of phase transitions and critical phenomena to the
reasoning process.  The engine deliberately drives reasoning through a
discontinuous transition from disordered brainstorming (high-entropy phase)
to crystallized, coherent framework (ordered phase).

Physics basis:

In Landau theory, a phase transition is characterised by an **order parameter**
that is zero in the disordered phase and nonzero in the ordered phase.  Near
the critical point ``T = T_c``, the free energy expands as::

    F(M) = a(T - T_c) * M^2 + b * M^4 + ...

Key phenomena modelled:

- **Disordered exploration**: At high temperature, fluctuations dominate and
  many incompatible viewpoints coexist.
- **Ordering pressure**: Constraints are gradually tightened (temperature
  lowered) until consensus themes emerge.
- **Critical threshold**: When the order parameter crosses a critical value,
  the system has undergone a phase transition from disorder to order.
- **Spontaneous symmetry breaking**: When multiple frameworks are equally
  valid, the engine must commit to one -- just as a ferromagnet must choose
  "up" or "down" below the Curie temperature.
- **Emergent order**: The coherent answer arises from disordered exploration
  without being imposed externally.

LLM call pattern::

    DISORDER(N) --> ORDER_MEASURE(1) -->
    [COOLING(N) --> CRITICAL_CHECK(1)] x R -->
    SYMMETRY_BREAK(1) --> REFINEMENT(1)

    Total: N + 1 + R*(N + 1) + 2,  default (N=5, R=2) = 20 calls

Example::

    from openagentflow.reasoning.phase_transition import PhaseTransition

    engine = PhaseTransition(num_perspectives=5, max_cooling_rounds=3)
    trace = await engine.reason(
        query="Design a governance framework for open-source AI models.",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

from openagentflow.reasoning.base import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
)

logger = logging.getLogger(__name__)

# Default personality prompts for the disordered phase.  Each perspective
# is deliberately different to maximise entropy in the initial exploration.
_DEFAULT_PERSONALITIES: list[str] = [
    "You are a deep skeptic.  Question every assumption, point out hidden "
    "risks, and argue for why the obvious approach will fail.",
    "You are a radical optimist.  Focus on the best-case scenario, highlight "
    "opportunities, and argue for ambitious, expansive solutions.",
    "You are a hard-nosed pragmatist.  Only care about what is implementable "
    "today with existing resources and constraints.",
    "You are a pure theorist.  Reason from first principles, seek elegance "
    "and generality, and ignore short-term practicality.",
    "You are a deliberate contrarian.  Take the opposite position to the "
    "mainstream view.  If everyone says X, argue for not-X.",
]


class PhaseTransition(ReasoningEngine):
    """Phase-transition-driven reasoning engine.

    The engine starts in a maximally disordered phase -- generating diverse,
    unconstrained perspectives at high temperature.  It then measures the
    **order parameter** (degree of coherence among perspectives) and applies
    increasing ordering pressure (tighter constraints, lower temperature)
    until the order parameter crosses a critical threshold.

    At the critical point, the engine performs **spontaneous symmetry
    breaking** -- committing to a specific coherent framework among equally
    valid candidates -- and then refines the ordered state into a polished
    answer.

    The approach is particularly effective for problems where the structure
    of the answer is unknown and must emerge naturally from brainstorming,
    rather than being imposed top-down.

    Attributes:
        name: ``"phase_transition"``
        description: Short human-readable summary.
        num_perspectives: Number of parallel disordered responses (``N``).
        max_cooling_rounds: Maximum number of cooling iterations before
            forcing transition.
        cooling_factor: Temperature multiplier per round (0 < factor < 1).
        critical_threshold: Order parameter value that triggers the
            phase transition (0.0--1.0).
        initial_temperature: Starting temperature for the disordered phase.
        ordered_temperature: Temperature used after the transition.
        personalities: System prompts for each perspective.
    """

    name: str = "phase_transition"
    description: str = (
        "Explores in maximum-entropy disorder, then cools until "
        "a phase transition spontaneously crystallizes coherent order."
    )

    def __init__(
        self,
        num_perspectives: int = 5,
        max_cooling_rounds: int = 3,
        cooling_factor: float = 0.7,
        critical_threshold: float = 0.7,
        initial_temperature: float = 1.0,
        ordered_temperature: float = 0.3,
        personalities: list[str] | None = None,
    ) -> None:
        """Initialise the PhaseTransition engine.

        Args:
            num_perspectives: Number of parallel disordered responses per
                phase.  More perspectives increase diversity but also LLM
                call count.
            max_cooling_rounds: Maximum cooling iterations before the
                engine forces a transition regardless of the order
                parameter.
            cooling_factor: Multiplicative factor applied to the
                temperature each cooling round.  Must be in ``(0, 1)``.
            critical_threshold: Order-parameter value (0.0--1.0) at which
                the phase transition is considered to have occurred.
            initial_temperature: Starting LLM temperature for the
                disordered phase.
            ordered_temperature: LLM temperature for the ordered
                (post-transition) phase.
            personalities: Optional list of system-prompt personalities.
                Must have length >= ``num_perspectives``.  If ``None``,
                sensible defaults are used.
        """
        if not 0 < cooling_factor < 1:
            raise ValueError(
                f"cooling_factor must be in (0, 1), got {cooling_factor}"
            )
        self.num_perspectives = max(2, num_perspectives)
        self.max_cooling_rounds = max(1, max_cooling_rounds)
        self.cooling_factor = cooling_factor
        self.critical_threshold = max(0.1, min(0.99, critical_threshold))
        self.initial_temperature = max(0.1, initial_temperature)
        self.ordered_temperature = max(0.05, ordered_temperature)

        # Build or validate personality list
        if personalities is not None:
            if len(personalities) < self.num_perspectives:
                raise ValueError(
                    f"personalities list has {len(personalities)} entries "
                    f"but num_perspectives is {self.num_perspectives}"
                )
            self.personalities = list(personalities)
        else:
            # Cycle defaults if we need more than 5
            base = _DEFAULT_PERSONALITIES
            self.personalities = [
                base[i % len(base)] for i in range(self.num_perspectives)
            ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: Any | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the phase-transition reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for all generation and evaluation.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on total cooling rounds, overriding
                ``max_cooling_rounds`` if smaller.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing disordered, cooling,
            symmetry-breaking, and ordered-state steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        effective_max_rounds = min(self.max_cooling_rounds, max_iterations)

        # ---------------------------------------------------------------
        # Phase 1: DISORDERED EXPLORATION (N calls)
        # ---------------------------------------------------------------
        temperature = self.initial_temperature
        responses = await self._generate_disordered(
            query, llm_provider, trace, temperature
        )

        # ---------------------------------------------------------------
        # Phase 2: ORDER PARAMETER MEASUREMENT (1 call)
        # ---------------------------------------------------------------
        order_result = await self._measure_order(
            query, responses, llm_provider, trace
        )
        order_param = order_result["order_parameter"]
        majority_themes = order_result.get("majority_themes", [])
        contradicted = order_result.get("contradicted_claims", [])

        # ---------------------------------------------------------------
        # Phase 3-4: COOLING LOOP until critical threshold or max rounds
        # ---------------------------------------------------------------
        cooling_round = 0
        while (
            order_param < self.critical_threshold
            and cooling_round < effective_max_rounds
        ):
            cooling_round += 1
            temperature *= self.cooling_factor

            # Re-generate with constraints (N calls)
            responses = await self._generate_cooled(
                query,
                majority_themes,
                contradicted,
                llm_provider,
                trace,
                temperature,
                cooling_round,
            )

            # Re-measure order parameter (1 call)
            order_result = await self._measure_order(
                query, responses, llm_provider, trace
            )
            order_param = order_result["order_parameter"]
            majority_themes = order_result.get("majority_themes", [])
            contradicted = order_result.get("contradicted_claims", [])

            # Record critical check
            crossed = order_param >= self.critical_threshold
            check_step = self._make_step(
                step_type="critical_check",
                content=(
                    f"Cooling round {cooling_round}: order_parameter="
                    f"{order_param:.3f}, threshold={self.critical_threshold}, "
                    f"crossed={'YES' if crossed else 'NO'}, "
                    f"temperature={temperature:.4f}"
                ),
                score=order_param,
                metadata={
                    "phase": "critical_check",
                    "cooling_round": cooling_round,
                    "order_parameter": round(order_param, 4),
                    "critical_threshold": self.critical_threshold,
                    "crossed": crossed,
                    "temperature": round(temperature, 4),
                    "majority_themes_count": len(majority_themes),
                },
            )
            trace.add_step(check_step)

            if crossed:
                logger.info(
                    "Phase transition detected at cooling round %d "
                    "(order_parameter=%.3f)",
                    cooling_round,
                    order_param,
                )
                break

        # ---------------------------------------------------------------
        # Phase 5: SYMMETRY BREAKING (1 call)
        # ---------------------------------------------------------------
        framework = await self._symmetry_breaking(
            query,
            responses,
            majority_themes,
            order_param,
            llm_provider,
            trace,
        )

        # ---------------------------------------------------------------
        # Phase 6: ORDERED STATE REFINEMENT (1 call)
        # ---------------------------------------------------------------
        final_output = await self._refine_ordered_state(
            query, framework, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Disordered exploration
    # ------------------------------------------------------------------

    async def _generate_disordered(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        temperature: float,
    ) -> list[str]:
        """Generate N maximally diverse, unconstrained perspectives.

        Each perspective uses a different personality system prompt and
        the highest temperature to maximise entropy.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.
            temperature: LLM temperature for generation.

        Returns:
            List of N response strings, one per perspective.
        """
        responses: list[str] = []

        for idx in range(self.num_perspectives):
            personality = self.personalities[idx]
            prompt = (
                f"Analyze the following problem from your unique perspective. "
                f"Be bold, unconstrained, and thorough.  Do not self-censor "
                f"or hedge.  Commit fully to your viewpoint.\n\n"
                f"Problem: {query}"
            )

            response = await self._call_llm(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                trace=trace,
                system=personality,
                temperature=min(temperature, 1.0),
            )

            step = self._make_step(
                step_type="disordered_state",
                content=response,
                score=0.0,
                metadata={
                    "phase": "disordered",
                    "perspective_index": idx,
                    "personality": personality[:80],
                    "temperature": round(temperature, 4),
                },
            )
            trace.add_step(step)
            responses.append(response)

        return responses

    # ------------------------------------------------------------------
    # Phase 2/4: Order parameter measurement
    # ------------------------------------------------------------------

    async def _measure_order(
        self,
        query: str,
        responses: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Measure the order parameter across all current responses.

        The order parameter quantifies how much coherence / consensus
        exists among the N perspectives.  A value near 0.0 means total
        chaos (no agreement); a value near 1.0 means near-perfect order.

        Args:
            query: Original user query.
            responses: Current list of perspective responses.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with keys ``order_parameter`` (float 0--1),
            ``majority_themes`` (list of str), and
            ``contradicted_claims`` (list of str).
        """
        formatted_responses = ""
        for idx, resp in enumerate(responses):
            formatted_responses += (
                f"\n--- Perspective {idx + 1} ---\n{resp}\n"
            )

        prompt = (
            f"You are measuring the ORDER PARAMETER of a system of "
            f"{len(responses)} independent analyses of the same problem.\n\n"
            f"Problem: {query}\n\n"
            f"Analyses:{formatted_responses}\n\n"
            f"Measure the degree of coherence among these analyses:\n"
            f"1. What themes or conclusions appear in a MAJORITY "
            f"(more than half) of the analyses?\n"
            f"2. What claims are explicitly CONTRADICTED by a majority?\n"
            f"3. On a scale of 0.0 (total chaos, no agreement) to 1.0 "
            f"(perfect consensus), what is the ORDER PARAMETER -- the "
            f"overall degree of coherence?\n\n"
            f"Respond with JSON only:\n"
            f'{{"order_parameter": 0.35, '
            f'"majority_themes": ["theme1", "theme2"], '
            f'"contradicted_claims": ["claim1"]}}'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are a precise measurement instrument.  Quantify the "
                "degree of order in a system of diverse responses.  Return "
                "valid JSON only -- no commentary."
            ),
            temperature=0.2,
        )

        result = self._parse_order_measurement(raw)

        step = self._make_step(
            step_type="order_measurement",
            content=(
                f"Order parameter: {result['order_parameter']:.3f}\n"
                f"Majority themes: {result['majority_themes']}\n"
                f"Contradicted claims: {result['contradicted_claims']}"
            ),
            score=result["order_parameter"],
            metadata={
                "phase": "order_measurement",
                "order_parameter": round(result["order_parameter"], 4),
                "majority_themes": result["majority_themes"],
                "contradicted_claims": result["contradicted_claims"],
                "num_responses": len(responses),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 3: Cooled re-generation
    # ------------------------------------------------------------------

    async def _generate_cooled(
        self,
        query: str,
        majority_themes: list[str],
        contradicted: list[str],
        provider: Any,
        trace: ReasoningTrace,
        temperature: float,
        cooling_round: int,
    ) -> list[str]:
        """Re-generate perspectives with constraints from majority themes.

        Each perspective must incorporate the established themes as fixed
        points but is free to disagree on everything else.  The LLM
        temperature is lowered to reduce entropy.

        Args:
            query: Original user query.
            majority_themes: Themes that appeared in the majority of
                previous responses (constraints).
            contradicted: Claims that were contradicted by the majority
                (anti-constraints).
            provider: LLM provider.
            trace: Reasoning trace.
            temperature: Current (cooled) LLM temperature.
            cooling_round: Which cooling iteration this is.

        Returns:
            List of N re-generated responses.
        """
        responses: list[str] = []

        themes_text = "\n".join(
            f"  - {t}" for t in majority_themes
        ) if majority_themes else "  (none established yet)"
        avoid_text = "\n".join(
            f"  - {c}" for c in contradicted
        ) if contradicted else "  (none identified)"

        for idx in range(self.num_perspectives):
            personality = self.personalities[idx]

            prompt = (
                f"Analyze this problem again, but this time you MUST "
                f"incorporate the following established themes as given:\n"
                f"{themes_text}\n\n"
                f"You should AVOID or explicitly address these "
                f"contradicted claims:\n{avoid_text}\n\n"
                f"Within these constraints, maintain your unique "
                f"perspective and contribute novel insights.\n\n"
                f"Problem: {query}"
            )

            response = await self._call_llm(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                trace=trace,
                system=personality,
                temperature=min(temperature, 1.0),
            )

            step = self._make_step(
                step_type="cooled_state",
                content=response,
                score=0.0,
                metadata={
                    "phase": "cooling",
                    "cooling_round": cooling_round,
                    "perspective_index": idx,
                    "temperature": round(temperature, 4),
                    "num_constraints": len(majority_themes),
                },
            )
            trace.add_step(step)
            responses.append(response)

        return responses

    # ------------------------------------------------------------------
    # Phase 5: Symmetry breaking
    # ------------------------------------------------------------------

    async def _symmetry_breaking(
        self,
        query: str,
        responses: list[str],
        majority_themes: list[str],
        order_param: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Perform spontaneous symmetry breaking.

        When multiple equally valid orderings exist (like a ferromagnet
        at the critical point), this phase forces a commitment to a
        specific direction.  The engine must choose one coherent framework
        even though the underlying problem may not uniquely prefer it.

        Args:
            query: Original user query.
            responses: Current perspective responses.
            majority_themes: Established themes.
            order_param: Current order parameter value.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The chosen framework as a string.
        """
        formatted_responses = ""
        for idx, resp in enumerate(responses):
            formatted_responses += (
                f"\n--- Perspective {idx + 1} ---\n{resp}\n"
            )

        themes_text = ", ".join(majority_themes) if majority_themes else "none"

        prompt = (
            f"The system has reached the critical point (order parameter = "
            f"{order_param:.3f}).  Multiple coherent frameworks are now "
            f"possible.\n\n"
            f"Problem: {query}\n\n"
            f"Established consensus themes: {themes_text}\n\n"
            f"Current perspectives:{formatted_responses}\n\n"
            f"SPONTANEOUS SYMMETRY BREAKING:\n"
            f"You must now commit to a SINGLE coherent framework that:\n"
            f"1. Incorporates all established consensus themes.\n"
            f"2. Resolves any remaining contradictions by choosing a "
            f"direction.\n"
            f"3. Forms a self-consistent, complete position.\n\n"
            f"This is like a ferromagnet choosing 'up' or 'down' -- "
            f"the underlying physics does not prefer one direction, but "
            f"the system must break symmetry and commit.  Choose the "
            f"direction that best serves the problem at hand.\n\n"
            f"Present the chosen framework as a unified, coherent analysis."
        )

        framework = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing spontaneous symmetry breaking.  There "
                "is no uniquely correct direction -- but you must choose one "
                "and commit fully.  Do not hedge or present alternatives.  "
                "Lock in a single coherent framework."
            ),
            temperature=self.ordered_temperature,
        )

        step = self._make_step(
            step_type="symmetry_breaking",
            content=framework,
            score=order_param,
            metadata={
                "phase": "symmetry_breaking",
                "order_parameter": round(order_param, 4),
                "majority_themes": majority_themes,
                "num_perspectives": len(responses),
            },
        )
        trace.add_step(step)

        return framework

    # ------------------------------------------------------------------
    # Phase 6: Ordered-state refinement
    # ------------------------------------------------------------------

    async def _refine_ordered_state(
        self,
        query: str,
        framework: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Refine the chosen framework into a polished final answer.

        The phase transition is complete.  This step polishes the
        crystallised order into a clear, well-structured response.

        Args:
            query: Original user query.
            framework: The framework chosen during symmetry breaking.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The polished final answer.
        """
        prompt = (
            f"The phase transition is complete.  A coherent framework has "
            f"crystallised from disordered brainstorming.  Refine this "
            f"framework into a polished, well-structured, and thorough "
            f"answer.\n\n"
            f"Original problem: {query}\n\n"
            f"Crystallised framework:\n{framework}\n\n"
            f"Polish this into a final answer.  The spontaneous order that "
            f"emerged represents the natural structure of the problem -- do "
            f"not fight it.  Improve clarity, structure, and completeness "
            f"without altering the fundamental direction."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are refining a crystallised framework into a polished "
                "answer.  Maintain the committed direction.  Improve "
                "presentation without changing substance."
            ),
            temperature=self.ordered_temperature,
        )

        ordered_step = self._make_step(
            step_type="ordered_state",
            content=final,
            score=1.0,
            metadata={"phase": "ordered_state"},
        )
        trace.add_step(ordered_step)

        output_step = self._make_step(
            step_type="final_output",
            content=final,
            score=1.0,
            metadata={"phase": "final"},
            parent_step_id=ordered_step.step_id,
        )
        trace.add_step(output_step)

        return final

    # ------------------------------------------------------------------
    # JSON parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order_measurement(raw: str) -> dict[str, Any]:
        """Extract order-parameter measurement from LLM JSON output.

        Parses a JSON object with ``order_parameter``,
        ``majority_themes``, and ``contradicted_claims``.  Falls back
        to sensible defaults on parse failure.

        Args:
            raw: Raw LLM output string.

        Returns:
            Dict with ``order_parameter`` (float), ``majority_themes``
            (list[str]), ``contradicted_claims`` (list[str]).
        """
        default: dict[str, Any] = {
            "order_parameter": 0.3,
            "majority_themes": [],
            "contradicted_claims": [],
        }

        text = raw.strip()
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return default

        try:
            parsed = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return default

        if not isinstance(parsed, dict):
            return default

        # Extract order parameter
        op = parsed.get("order_parameter", 0.3)
        try:
            op = max(0.0, min(1.0, float(op)))
        except (TypeError, ValueError):
            op = 0.3

        # Extract themes
        themes = parsed.get("majority_themes", [])
        if not isinstance(themes, list):
            themes = []
        themes = [str(t) for t in themes if t]

        # Extract contradicted claims
        contradicted = parsed.get("contradicted_claims", [])
        if not isinstance(contradicted, list):
            contradicted = []
        contradicted = [str(c) for c in contradicted if c]

        return {
            "order_parameter": op,
            "majority_themes": themes,
            "contradicted_claims": contradicted,
        }


__all__ = ["PhaseTransition"]
