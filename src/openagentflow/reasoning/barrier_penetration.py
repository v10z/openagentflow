"""Barrier Penetration reasoning engine.

Based on the physics of tunneling through classically forbidden barriers.
In classical mechanics, a particle with energy E encountering a potential
barrier of height V_0 > E is always reflected.  In wave mechanics, the
wavefunction decays exponentially inside the barrier but does not vanish --
if the barrier has finite width d, there is a nonzero transmission
probability::

    T ~ exp(-2 * kappa * d)

    where kappa = sqrt(2m(V_0 - E)) / hbar

This engine is designed for problems where reasoning gets stuck -- trapped
behind a seemingly insurmountable barrier (a false constraint, a conceptual
block, an apparent impossibility).  It identifies barriers, measures their
"width" and "height", and applies specific techniques to narrow or lower
the barrier before tunneling through.

Techniques for barrier narrowing:

- **Reframing**: Reformulate the problem to reduce the barrier's width
- **Decomposition**: Break the barrier into thinner sub-barriers
- **Relaxation**: Temporarily relax constraints to lower the barrier height
- **Transformation**: Transform the problem space so the barrier is thinner
- **Lateral thinking**: Approach from a direction where the barrier is narrower

Example::

    from openagentflow.reasoning.barrier_penetration import BarrierPenetration

    engine = BarrierPenetration(max_tunnels=3)
    trace = await engine.reason(
        query="How can we achieve zero-latency distributed consensus?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    for step in trace.get_steps_by_type("tunneling_attempt"):
        barrier = step.metadata.get("barrier_name")
        penetrated = step.metadata.get("barrier_was_real")
        status = "REAL barrier" if penetrated else "barrier penetrated"
        print(f"  {barrier}: {status}")
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)


class BarrierPenetration(ReasoningEngine):
    """Tunneling-inspired reasoning for escaping conceptual barriers.

    The engine works in six phases:

    1. **INITIAL APPROACH** -- Attempt to solve the problem directly.  Flag
       any point where reasoning gets stuck, blocked, or encounters an
       apparent impossibility.
    2. **BARRIER IDENTIFICATION** -- Analyse each barrier: what assumption or
       constraint creates it, how "wide" (specific vs. fundamental) and how
       "high" (minor vs. impossible) it is, and what solution lies beyond it.
    3. **TUNNELING PROBABILITY COMPUTATION** -- Compute a penetrability score
       for each barrier using the WKB-inspired formula
       ``T = exp(-scale * width * height)``.  Rank barriers from most to
       least penetrable.
    4. **TUNNELING ATTEMPTS** -- For each barrier (up to ``max_tunnels``),
       suspend the blocking assumption and solve as if the barrier does not
       exist.  Then check: is the solution on the other side valid?  Was the
       barrier an illusion or a genuine hard constraint?
    5. **POST-TUNNELING ASSESSMENT** -- Summarise which barriers were
       penetrated (false constraints) and which were real.
    6. **RECONSTRUCTED SOLUTION** -- Present the final answer incorporating
       insights from beyond any successfully penetrated barriers.

    Attributes:
        name: ``"BarrierPenetration"``
        description: Short human-readable summary.
        max_tunnels: Maximum number of barriers to attempt tunneling through.
        barrier_threshold: Minimum tunneling probability to attempt.
        tunneling_temperature: LLM temperature for tunneling attempts.
        initial_temperature: LLM temperature for the initial approach.
        wkb_scaling: Scaling factor in the tunneling probability formula.
        allow_multi_barrier: Whether to attempt multiple barriers.
    """

    name: str = "BarrierPenetration"
    description: str = (
        "Identifies barriers blocking the reasoning, computes "
        "tunneling probabilities, and attempts to penetrate "
        "barriers by suspending blocking assumptions."
    )

    def __init__(
        self,
        max_tunnels: int = 3,
        barrier_threshold: float = 0.3,
        tunneling_temperature: float = 0.6,
        initial_temperature: float = 0.5,
        wkb_scaling: float = 2.0,
        allow_multi_barrier: bool = True,
    ) -> None:
        """Initialise the Barrier Penetration engine.

        Args:
            max_tunnels: Maximum number of barriers to attempt tunneling
                through.  Each attempt costs one LLM call.
            barrier_threshold: Minimum tunneling probability (T value) to
                attempt.  Barriers below this threshold are considered
                impenetrable.
            tunneling_temperature: Temperature for the creative tunneling
                attempts (higher = more creative).
            initial_temperature: Temperature for the initial direct approach.
            wkb_scaling: Scaling factor in ``T = exp(-scale * width * height)``.
                Higher values make tunneling harder (more conservative).
            allow_multi_barrier: If ``True``, attempts tunneling through
                multiple barriers sequentially.  If ``False``, only the
                most penetrable barrier is attempted.
        """
        self.max_tunnels = max(1, max_tunnels)
        self.barrier_threshold = max(0.0, min(1.0, barrier_threshold))
        self.tunneling_temperature = tunneling_temperature
        self.initial_temperature = initial_temperature
        self.wkb_scaling = max(0.1, wkb_scaling)
        self.allow_multi_barrier = allow_multi_barrier

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
        """Execute the Barrier Penetration reasoning strategy.

        Args:
            query: The user question or problem to reason about.
            llm_provider: An LLM provider for generation and analysis.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Hard cap on total LLM calls.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with initial approach, barrier analysis,
            tunneling probability, tunneling attempt, post-tunneling, and
            reconstructed solution steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- INITIAL APPROACH
        initial_attempt = await self._initial_approach(query, llm_provider, trace)
        init_step = self._make_step(
            step_type="initial_approach",
            content=initial_attempt,
            score=0.5,
            metadata={"phase": "initial_approach"},
        )
        trace.add_step(init_step)

        # Phase 2 -- BARRIER IDENTIFICATION
        barriers = await self._identify_barriers(
            query, initial_attempt, llm_provider, trace
        )

        barrier_step = self._make_step(
            step_type="barrier_analysis",
            content=f"Identified {len(barriers)} barrier(s).",
            score=0.0,
            metadata={
                "phase": "barrier_analysis",
                "barriers": barriers,
                "count": len(barriers),
            },
        )
        trace.add_step(barrier_step)

        if not barriers:
            # No barriers found -- the initial approach is the answer
            trace.final_output = initial_attempt
            trace.duration_ms = (time.time() - start) * 1000
            return trace

        # Phase 3 -- TUNNELING PROBABILITY COMPUTATION
        for b in barriers:
            width = b.get("width", 0.5)
            height = b.get("height", 0.5)
            b["tunneling_probability"] = math.exp(
                -self.wkb_scaling * width * height
            )

        # Sort by tunneling probability descending (most penetrable first)
        barriers.sort(
            key=lambda b: b.get("tunneling_probability", 0.0), reverse=True
        )

        prob_summary_parts = []
        for idx, b in enumerate(barriers):
            prob_summary_parts.append(
                f"  {idx + 1}. {b.get('barrier', 'unknown')}: "
                f"T={b['tunneling_probability']:.4f} "
                f"(w={b.get('width', 0):.2f}, h={b.get('height', 0):.2f})"
            )
        prob_summary = "\n".join(prob_summary_parts)

        # LLM call to rank and assess penetrability
        prob_assessment = await self._assess_penetrability(
            query, barriers, llm_provider, trace
        )

        prob_step = self._make_step(
            step_type="tunneling_probability",
            content=f"Tunneling probabilities:\n{prob_summary}\n\n{prob_assessment}",
            score=barriers[0]["tunneling_probability"] if barriers else 0.0,
            metadata={
                "phase": "tunneling_probability",
                "ranked_barriers": [
                    {
                        "barrier": b.get("barrier", ""),
                        "tunneling_probability": round(
                            b["tunneling_probability"], 4
                        ),
                        "width": b.get("width", 0),
                        "height": b.get("height", 0),
                    }
                    for b in barriers
                ],
            },
        )
        trace.add_step(prob_step)

        # Phase 4 -- TUNNELING ATTEMPTS
        # Filter by threshold and cap to max_tunnels
        attemptable = [
            b for b in barriers
            if b.get("tunneling_probability", 0.0) >= self.barrier_threshold
        ]
        if not self.allow_multi_barrier:
            attemptable = attemptable[:1]
        else:
            attemptable = attemptable[: self.max_tunnels]

        tunnel_results: list[dict[str, Any]] = []
        for b in attemptable:
            if trace.total_llm_calls >= max_iterations:
                logger.debug(
                    "BarrierPenetration: max_iterations reached during tunneling"
                )
                break

            result = await self._attempt_tunneling(
                query, initial_attempt, b, llm_provider, trace
            )
            tunnel_results.append(result)

            t_step = self._make_step(
                step_type="tunneling_attempt",
                content=result.get("solution_beyond", ""),
                score=b["tunneling_probability"],
                metadata={
                    "phase": "tunneling_attempt",
                    "barrier_name": b.get("barrier", "unknown"),
                    "barrier_was_real": result.get("barrier_was_real", True),
                    "what_broke": result.get("what_broke"),
                    "tunneling_probability": round(
                        b["tunneling_probability"], 4
                    ),
                },
                parent_step_id=prob_step.step_id,
            )
            trace.add_step(t_step)

        # Phase 5 -- POST-TUNNELING ASSESSMENT
        if trace.total_llm_calls < max_iterations:
            post_assessment = await self._post_tunneling_assessment(
                query, tunnel_results, llm_provider, trace
            )
            post_step = self._make_step(
                step_type="post_tunneling",
                content=post_assessment,
                score=0.8,
                metadata={
                    "phase": "post_tunneling",
                    "penetrated_count": sum(
                        1
                        for r in tunnel_results
                        if not r.get("barrier_was_real", True)
                    ),
                    "real_count": sum(
                        1
                        for r in tunnel_results
                        if r.get("barrier_was_real", True)
                    ),
                    "total_attempted": len(tunnel_results),
                },
            )
            trace.add_step(post_step)
        else:
            post_assessment = ""

        # Phase 6 -- RECONSTRUCTED SOLUTION
        final_output = await self._reconstruct_solution(
            query,
            initial_attempt,
            tunnel_results,
            post_assessment,
            llm_provider,
            trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- INITIAL APPROACH
    # ------------------------------------------------------------------

    async def _initial_approach(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Attempt to solve the problem directly, flagging barriers.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The initial attempt text with barriers flagged.
        """
        prompt = (
            f"Attempt to solve this problem directly. If you encounter any "
            f"point where you feel stuck, blocked, or unsure how to proceed, "
            f"explicitly flag it as a BARRIER.\n\n"
            f"For each barrier, describe:\n"
            f"  - What is blocking progress\n"
            f"  - What assumption or constraint creates the barrier\n"
            f"  - What lies beyond it (if you can see past it)\n\n"
            f"Problem: {query}\n\n"
            f"Format barriers as: [BARRIER: description]\n"
            f"Proceed as far as you can, flagging barriers along the way."
        )
        return await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are a problem solver who explicitly identifies barriers. "
                "Do not give up when you hit a barrier -- flag it and continue."
            ),
            temperature=self.initial_temperature,
        )

    # ------------------------------------------------------------------
    # Phase 2 -- BARRIER IDENTIFICATION
    # ------------------------------------------------------------------

    async def _identify_barriers(
        self,
        query: str,
        initial_attempt: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Analyse barriers encountered in the initial approach.

        Returns a list of barrier dicts with width, height, assumption,
        and beyond fields.

        Args:
            query: Original user query.
            initial_attempt: The initial approach with flagged barriers.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of barrier dicts.
        """
        prompt = (
            f"Analyse the barriers encountered in this problem-solving "
            f"attempt. For each barrier identified (or that should have been "
            f"identified):\n\n"
            f"- 'barrier': A concise name for the barrier\n"
            f"- 'assumption': What assumption or constraint creates it?\n"
            f"- 'width': How wide is the barrier? (0.0 = very narrow/specific "
            f"technical issue, 1.0 = very wide/fundamental conceptual block)\n"
            f"- 'height': How high is the barrier? (0.0 = minor difficulty, "
            f"1.0 = seems completely impossible)\n"
            f"- 'beyond': What solution would be available if the barrier did "
            f"not exist?\n"
            f"- 'technique': Which narrowing technique is most promising? "
            f"(reframing, decomposition, relaxation, transformation, lateral)\n\n"
            f"Original problem: {query}\n\n"
            f"Initial attempt:\n{initial_attempt}\n\n"
            f"Return ONLY a JSON array of barrier objects.\n"
            f'Example: [{{"barrier": "speed of light limit", '
            f'"assumption": "information cannot travel faster than c", '
            f'"width": 0.9, "height": 0.95, '
            f'"beyond": "instant communication would enable...", '
            f'"technique": "decomposition"}}]'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are analysing reasoning barriers. Be precise about "
                "width (specificity) and height (severity). Return valid "
                "JSON only."
            ),
            temperature=0.4,
        )

        return self._parse_barrier_list(raw)

    # ------------------------------------------------------------------
    # Phase 3 -- TUNNELING PROBABILITY (LLM assessment)
    # ------------------------------------------------------------------

    async def _assess_penetrability(
        self,
        query: str,
        barriers: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Have the LLM assess which barriers are most likely penetrable.

        Args:
            query: Original user query.
            barriers: Sorted list of barriers with tunneling probabilities.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The LLM's penetrability assessment text.
        """
        barrier_listing = "\n".join(
            f"  {i + 1}. {b.get('barrier', '?')} "
            f"(T={b.get('tunneling_probability', 0):.4f}, "
            f"assumption: {b.get('assumption', '?')}, "
            f"technique: {b.get('technique', '?')})"
            for i, b in enumerate(barriers)
        )

        prompt = (
            f"These barriers have been ranked by penetrability "
            f"(tunneling probability T). For the most penetrable barriers: "
            f"what would happen if we simply SUSPENDED the assumption that "
            f"creates each one? Is the assumption actually necessary, or "
            f"is it a false constraint?\n\n"
            f"Problem: {query}\n\n"
            f"Barriers (ranked by penetrability):\n{barrier_listing}\n\n"
            f"For each barrier, briefly assess whether the blocking assumption "
            f"is truly necessary or potentially suspendable."
        )

        return await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are assessing barrier penetrability. Question assumptions "
                "rigorously but honestly."
            ),
            temperature=0.5,
        )

    # ------------------------------------------------------------------
    # Phase 4 -- TUNNELING ATTEMPTS
    # ------------------------------------------------------------------

    async def _attempt_tunneling(
        self,
        query: str,
        initial_attempt: str,
        barrier: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Attempt to tunnel through a single barrier.

        Suspends the blocking assumption and solves as if the barrier does
        not exist, then checks validity.

        Args:
            query: Original user query.
            initial_attempt: The initial approach for context.
            barrier: The barrier dict to tunnel through.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``solution_beyond``, ``barrier_was_real``, and
            ``what_broke``.
        """
        barrier_name = barrier.get("barrier", "unknown")
        assumption = barrier.get("assumption", "unknown assumption")
        beyond = barrier.get("beyond", "unknown")
        technique = barrier.get("technique", "reframing")

        prompt = (
            f"TUNNEL THROUGH THIS BARRIER.\n\n"
            f"The barrier is: {barrier_name}\n"
            f"The blocking assumption is: {assumption}\n"
            f"Suggested technique: {technique}\n"
            f"What lies beyond: {beyond}\n\n"
            f"For this phase, SUSPEND this assumption entirely. "
            f"Solve the problem as if the barrier does not exist.\n\n"
            f"Apply the '{technique}' technique:\n"
        )

        # Add technique-specific instructions
        technique_instructions = {
            "reframing": (
                "Reformulate the problem so that the blocking assumption "
                "becomes irrelevant. Find a framing where the barrier "
                "simply does not appear."
            ),
            "decomposition": (
                "Break the barrier into smaller sub-barriers. Each sub-barrier "
                "may be thin enough to penetrate individually, even if the "
                "combined barrier seems impenetrable."
            ),
            "relaxation": (
                "Temporarily relax the constraint that creates the barrier. "
                "Find the solution in the relaxed space, then check how much "
                "of it survives when the constraint is re-imposed."
            ),
            "transformation": (
                "Transform the problem space so that the barrier becomes "
                "thinner or lower. Use a change of variables, analogy, or "
                "abstraction that makes the barrier more tractable."
            ),
            "lateral": (
                "Approach from a completely different direction where the "
                "barrier is narrower. What adjacent or analogous problem "
                "does not have this barrier? Solve that, then map back."
            ),
        }

        prompt += technique_instructions.get(technique, technique_instructions["reframing"])
        prompt += (
            f"\n\nOriginal problem: {query}\n\n"
            f"After tunneling, answer these questions:\n"
            f"1. What is the solution on the other side of the barrier?\n"
            f"2. Is the solution actually valid? Did suspending the assumption "
            f"break anything fundamental?\n"
            f"3. Was the barrier real (genuine hard constraint) or an illusion "
            f"(false/unnecessary assumption)?\n\n"
            f"At the end of your response, provide a summary line:\n"
            f"BARRIER_REAL: true/false\n"
            f"WHAT_BROKE: <description or 'nothing'>"
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are tunneling through a reasoning barrier. Be creative "
                "and bold -- suspend the blocking assumption completely. Then "
                "honestly assess whether the solution is valid."
            ),
            temperature=self.tunneling_temperature,
        )

        # Parse the result
        barrier_was_real = self._extract_barrier_real(raw)
        what_broke = self._extract_what_broke(raw)
        solution_beyond = self._strip_summary_lines(raw)

        return {
            "barrier_name": barrier_name,
            "solution_beyond": solution_beyond,
            "barrier_was_real": barrier_was_real,
            "what_broke": what_broke,
            "technique": technique,
        }

    # ------------------------------------------------------------------
    # Phase 5 -- POST-TUNNELING ASSESSMENT
    # ------------------------------------------------------------------

    async def _post_tunneling_assessment(
        self,
        query: str,
        tunnel_results: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Summarise tunneling outcomes.

        Args:
            query: Original user query.
            tunnel_results: Results from all tunneling attempts.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Assessment text.
        """
        result_summaries = []
        for r in tunnel_results:
            status = "REAL (not penetrated)" if r.get("barrier_was_real", True) else "PENETRATED (assumption was false/unnecessary)"
            result_summaries.append(
                f"  - {r.get('barrier_name', '?')}: {status}\n"
                f"    Technique: {r.get('technique', '?')}\n"
                f"    What broke: {r.get('what_broke', 'N/A')}"
            )
        results_text = "\n".join(result_summaries)

        penetrated = sum(
            1 for r in tunnel_results if not r.get("barrier_was_real", True)
        )
        real = sum(
            1 for r in tunnel_results if r.get("barrier_was_real", True)
        )

        prompt = (
            f"Tunneling results summary:\n{results_text}\n\n"
            f"Penetrated: {penetrated}/{len(tunnel_results)}\n"
            f"Real barriers: {real}/{len(tunnel_results)}\n\n"
            f"Original problem: {query}\n\n"
            f"For barriers that were successfully penetrated (assumption was "
            f"false or unnecessary): describe the insight gained and how the "
            f"solution beyond the barrier should be integrated.\n\n"
            f"For barriers that were real (tunneling broke something "
            f"fundamental): acknowledge the genuine constraint and describe "
            f"how to work within it."
        )

        return await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing post-tunneling assessment. Distinguish "
                "between genuinely penetrated barriers and real constraints."
            ),
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # Phase 6 -- RECONSTRUCTED SOLUTION
    # ------------------------------------------------------------------

    async def _reconstruct_solution(
        self,
        query: str,
        initial_attempt: str,
        tunnel_results: list[dict[str, Any]],
        post_assessment: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Present the final answer incorporating tunneling insights.

        Args:
            query: Original user query.
            initial_attempt: The initial approach.
            tunnel_results: Results from all tunneling attempts.
            post_assessment: The post-tunneling assessment.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final answer text.
        """
        penetrated_insights = []
        real_constraints = []
        for r in tunnel_results:
            if not r.get("barrier_was_real", True):
                penetrated_insights.append(
                    f"  - Penetrated '{r.get('barrier_name', '?')}': "
                    f"{r.get('solution_beyond', '')[:300]}"
                )
            else:
                real_constraints.append(
                    f"  - Real barrier '{r.get('barrier_name', '?')}': "
                    f"must work within this constraint"
                )

        insights_text = "\n".join(penetrated_insights) if penetrated_insights else "  (none)"
        constraints_text = "\n".join(real_constraints) if real_constraints else "  (none)"

        prompt = (
            f"Present the final answer. It should incorporate insights from "
            f"beyond any successfully penetrated barriers while respecting "
            f"genuine constraints.\n\n"
            f"Original problem: {query}\n\n"
            f"Initial approach:\n{initial_attempt[:500]}\n\n"
            f"Insights from penetrated barriers:\n{insights_text}\n\n"
            f"Real constraints to respect:\n{constraints_text}\n\n"
            f"Post-tunneling assessment:\n{post_assessment[:500]}\n\n"
            f"Produce a comprehensive, polished final answer that integrates "
            f"all insights from the barrier penetration process."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are reconstructing the solution after barrier penetration. "
                "Integrate insights from beyond penetrated barriers while "
                "respecting genuine constraints."
            ),
            temperature=0.4,
        )

        recon_step = self._make_step(
            step_type="reconstructed_solution",
            content=final,
            score=1.0,
            metadata={
                "phase": "reconstructed_solution",
                "penetrated_barriers": [
                    r.get("barrier_name", "?")
                    for r in tunnel_results
                    if not r.get("barrier_was_real", True)
                ],
                "real_barriers": [
                    r.get("barrier_name", "?")
                    for r in tunnel_results
                    if r.get("barrier_was_real", True)
                ],
            },
        )
        trace.add_step(recon_step)

        return final

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_barrier_list(raw: str) -> list[dict[str, Any]]:
        """Parse a JSON array of barrier objects from LLM output.

        Falls back to a single generic barrier if parsing fails.

        Args:
            raw: Raw LLM output.

        Returns:
            List of barrier dicts.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, dict):
                            result.append({
                                "barrier": str(item.get("barrier", "unknown")),
                                "assumption": str(
                                    item.get("assumption", "unknown")
                                ),
                                "width": max(
                                    0.0,
                                    min(1.0, float(item.get("width", 0.5))),
                                ),
                                "height": max(
                                    0.0,
                                    min(1.0, float(item.get("height", 0.5))),
                                ),
                                "beyond": str(item.get("beyond", "")),
                                "technique": str(
                                    item.get("technique", "reframing")
                                ),
                            })
                    if result:
                        return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback
        return [
            {
                "barrier": "conceptual block",
                "assumption": "unstated assumption blocking progress",
                "width": 0.5,
                "height": 0.5,
                "beyond": "alternative solution space",
                "technique": "reframing",
            }
        ]

    @staticmethod
    def _extract_barrier_real(raw: str) -> bool:
        """Extract the BARRIER_REAL flag from the LLM response.

        Args:
            raw: Raw LLM response.

        Returns:
            ``True`` if the barrier was real, ``False`` if penetrated.
        """
        match = re.search(r"BARRIER_REAL:\s*(true|false)", raw, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        # Heuristic: look for keywords indicating the barrier was false
        lower = raw.lower()
        if any(
            phrase in lower
            for phrase in [
                "barrier was an illusion",
                "assumption was false",
                "assumption was unnecessary",
                "successfully tunnel",
                "barrier is not real",
                "constraint is not actually necessary",
            ]
        ):
            return False
        return True  # Assume real by default (conservative)

    @staticmethod
    def _extract_what_broke(raw: str) -> str | None:
        """Extract the WHAT_BROKE field from the LLM response.

        Args:
            raw: Raw LLM response.

        Returns:
            Description of what broke, or ``None`` if nothing.
        """
        match = re.search(r"WHAT_BROKE:\s*(.+)", raw, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value.lower() in ("nothing", "none", "n/a", "null"):
                return None
            return value
        return None

    @staticmethod
    def _strip_summary_lines(raw: str) -> str:
        """Remove BARRIER_REAL and WHAT_BROKE lines from the response.

        Args:
            raw: Raw LLM response.

        Returns:
            Cleaned response text.
        """
        lines = raw.splitlines()
        cleaned = [
            line
            for line in lines
            if not re.match(r"^\s*BARRIER_REAL:", line, re.IGNORECASE)
            and not re.match(r"^\s*WHAT_BROKE:", line, re.IGNORECASE)
        ]
        return "\n".join(cleaned).strip()


__all__ = ["BarrierPenetration"]
