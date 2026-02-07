"""Perturbative Expansion reasoning engine.

Applies perturbation theory -- the workhorse of theoretical physics -- to the
reasoning process.  The engine separates a problem into a solvable core
(zeroth-order solution) plus ordered corrections from identified complications.
Each correction should be smaller than the last; if corrections grow the engine
detects divergence and flags a non-perturbative regime.

Physics basis::

    H = H_0 + lambda * V

    E_n = E_n^(0) + lambda * E_n^(1) + lambda^2 * E_n^(2) + ...

Each order provides a correction to the base solution:

- **Zeroth order**: Simplified base solution ignoring complications.
- **First order**: Leading correction from the strongest perturbation.
- **Higher orders**: Successive corrections accounting for interactions
  (cross-terms) between perturbations.

Convergence is monitored at every order.  If ``|correction_N| >
|correction_{N-1}|`` the series is diverging, indicating the "perturbation" is
too large for incremental treatment (strong-coupling regime).

Example::

    from openagentflow.reasoning.perturbative_expansion import PerturbativeExpansion

    engine = PerturbativeExpansion(max_perturbation_order=4)
    trace = await engine.reason(
        query="Estimate the time to market for a mobile banking app.",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    for step in trace.get_steps_by_type("convergence_check"):
        order = step.metadata.get("order")
        converging = step.metadata.get("converging")
        print(f"  Order {order}: {'converging' if converging else 'DIVERGING'}")
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


class PerturbativeExpansion(ReasoningEngine):
    """Perturbation-theory reasoning: base solution plus ordered corrections.

    The engine works in six phases:

    1. **ZEROTH-ORDER SOLUTION** -- Solve a deliberately simplified version of
       the problem.  Ignore complications, edge cases, and second-order effects.
    2. **PERTURBATION IDENTIFICATION** -- Enumerate the complications that were
       ignored, ordered by estimated coupling strength (impact on the answer).
    3. **FIRST-ORDER CORRECTION** -- Apply the strongest perturbation to the
       base solution, computing the leading correction.
    4. **HIGHER-ORDER CORRECTIONS** -- For each additional perturbation (in
       decreasing coupling strength), apply a correction that also accounts for
       cross-terms with previously applied perturbations.
    5. **CONVERGENCE CHECK** -- After each correction, verify that the series
       is converging (each correction is smaller than the last).  If corrections
       grow, flag divergence.
    6. **RESUMMATION / SYNTHESIS** -- If converging, present the fully corrected
       answer.  If diverging, acknowledge the breakdown and warn which
       perturbation requires non-perturbative treatment.

    Attributes:
        name: ``"PerturbativeExpansion"``
        description: Short human-readable summary.
        max_perturbation_order: Maximum number of correction orders to compute.
        convergence_threshold: If the ratio of successive correction magnitudes
            is below this, the series is converging.
        coupling_cutoff: Perturbations with coupling strength below this are
            ignored entirely.
        zeroth_order_temperature: LLM temperature for the base solution.
        correction_temperature: LLM temperature for correction computations.
        warn_on_divergence: Whether to explicitly flag non-perturbative
            breakdowns in the output.
    """

    name: str = "PerturbativeExpansion"
    description: str = (
        "Starts with a simplified zeroth-order solution and applies "
        "successive perturbative corrections from identified complications."
    )

    def __init__(
        self,
        max_perturbation_order: int = 4,
        convergence_threshold: float = 0.5,
        coupling_cutoff: float = 0.1,
        zeroth_order_temperature: float = 0.3,
        correction_temperature: float = 0.4,
        warn_on_divergence: bool = True,
    ) -> None:
        """Initialise the Perturbative Expansion engine.

        Args:
            max_perturbation_order: Maximum number of correction orders to
                apply.  Each order costs one LLM call.
            convergence_threshold: Ratio ``|E^(N)| / |E^(N-1)|`` below which
                the series is considered converging.  Values below 1.0 mean
                corrections are shrinking.
            coupling_cutoff: Perturbations with estimated coupling strength
                below this value are discarded.
            zeroth_order_temperature: Temperature for the base solution LLM
                call (lower = more conservative).
            correction_temperature: Temperature for correction LLM calls.
            warn_on_divergence: If ``True``, the final output explicitly flags
                perturbations that caused divergence.
        """
        self.max_perturbation_order = max(1, max_perturbation_order)
        self.convergence_threshold = convergence_threshold
        self.coupling_cutoff = max(0.0, coupling_cutoff)
        self.zeroth_order_temperature = zeroth_order_temperature
        self.correction_temperature = correction_temperature
        self.warn_on_divergence = warn_on_divergence

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
        """Execute the Perturbative Expansion reasoning strategy.

        Args:
            query: The user question or problem to reason about.
            llm_provider: An LLM provider for generating and refining
                solutions.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Hard cap on total LLM calls.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with zeroth-order, correction,
            convergence-check, and resummation steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- ZEROTH-ORDER SOLUTION
        zeroth_solution = await self._generate_zeroth_order(query, llm_provider, trace)
        zeroth_step = self._make_step(
            step_type="zeroth_order",
            content=zeroth_solution,
            score=0.5,
            metadata={"phase": "zeroth_order", "order": 0},
        )
        trace.add_step(zeroth_step)

        # Phase 2 -- PERTURBATION IDENTIFICATION
        perturbations = await self._identify_perturbations(
            query, zeroth_solution, llm_provider, trace
        )
        pert_step = self._make_step(
            step_type="perturbation_identification",
            content=f"Identified {len(perturbations)} perturbation(s).",
            score=0.0,
            metadata={
                "phase": "perturbation_identification",
                "perturbations": perturbations,
                "count": len(perturbations),
            },
        )
        trace.add_step(pert_step)

        # Filter by coupling cutoff and cap to max_perturbation_order
        active_perts = [
            p for p in perturbations
            if p.get("coupling_strength", 0.0) >= self.coupling_cutoff
        ]
        active_perts = active_perts[: self.max_perturbation_order]

        if not active_perts:
            # No perturbations significant enough -- the zeroth order is final
            trace.final_output = zeroth_solution
            trace.duration_ms = (time.time() - start) * 1000
            return trace

        # Phase 3+4 -- CORRECTIONS (first-order + higher orders)
        current_solution = zeroth_solution
        correction_magnitudes: list[float] = []
        convergence_status: list[dict[str, Any]] = []
        diverged = False
        divergent_perturbation: str | None = None

        for order_idx, pert in enumerate(active_perts):
            order_num = order_idx + 1

            # Honour max_iterations safety cap
            if trace.total_llm_calls >= max_iterations:
                logger.debug(
                    "PerturbativeExpansion: max_iterations reached at order %d",
                    order_num,
                )
                break

            # Generate the correction
            correction_result = await self._apply_correction(
                query=query,
                current_solution=current_solution,
                perturbation=pert,
                order_num=order_num,
                previous_perturbations=active_perts[:order_idx],
                provider=llm_provider,
                trace=trace,
            )

            corrected_solution = correction_result["corrected_solution"]
            magnitude = correction_result["magnitude"]
            correction_magnitudes.append(magnitude)

            # Record the correction step
            step_type = "first_order" if order_num == 1 else f"order_{order_num}_correction"
            corr_step = self._make_step(
                step_type=step_type,
                content=corrected_solution,
                score=magnitude,
                metadata={
                    "phase": "correction",
                    "order": order_num,
                    "perturbation": pert.get("perturbation", ""),
                    "coupling_strength": pert.get("coupling_strength", 0.0),
                    "correction_magnitude": round(magnitude, 4),
                },
                parent_step_id=zeroth_step.step_id,
            )
            trace.add_step(corr_step)

            # Phase 5 -- CONVERGENCE CHECK (programmatic)
            converging = True
            ratio = 0.0
            if len(correction_magnitudes) >= 2:
                prev_mag = correction_magnitudes[-2]
                curr_mag = correction_magnitudes[-1]
                if prev_mag > 0:
                    ratio = curr_mag / prev_mag
                    converging = ratio < self.convergence_threshold
                else:
                    converging = True

            conv_info = {
                "order": order_num,
                "correction_magnitude": round(magnitude, 4),
                "ratio": round(ratio, 4),
                "converging": converging,
            }
            convergence_status.append(conv_info)

            conv_content = (
                f"Order {order_num}: magnitude={magnitude:.4f}"
            )
            if len(correction_magnitudes) >= 2:
                conv_content += (
                    f", ratio={ratio:.4f} "
                    f"({'converging' if converging else 'DIVERGING'})"
                )
            else:
                conv_content += " (first correction -- no ratio yet)"

            conv_step = self._make_step(
                step_type="convergence_check",
                content=conv_content,
                score=ratio if len(correction_magnitudes) >= 2 else 0.0,
                metadata={
                    "phase": "convergence_check",
                    **conv_info,
                },
                parent_step_id=corr_step.step_id,
            )
            trace.add_step(conv_step)

            current_solution = corrected_solution

            if not converging:
                diverged = True
                divergent_perturbation = pert.get("perturbation", "unknown")
                if self.warn_on_divergence:
                    logger.warning(
                        "PerturbativeExpansion: divergence detected at order %d "
                        "(perturbation: %s, ratio: %.4f)",
                        order_num,
                        divergent_perturbation,
                        ratio,
                    )
                # Continue applying remaining corrections -- the resummation
                # step will handle the divergence warning.

        # Phase 6 -- RESUMMATION / SYNTHESIS
        final_output = await self._resum_and_synthesize(
            query=query,
            zeroth_solution=zeroth_solution,
            corrected_solution=current_solution,
            correction_magnitudes=correction_magnitudes,
            convergence_status=convergence_status,
            diverged=diverged,
            divergent_perturbation=divergent_perturbation,
            provider=llm_provider,
            trace=trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- ZEROTH-ORDER SOLUTION
    # ------------------------------------------------------------------

    async def _generate_zeroth_order(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the simplified base solution (zeroth order).

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The zeroth-order solution text.
        """
        prompt = (
            f"Solve a SIMPLIFIED version of this problem. Ignore complications, "
            f"edge cases, and second-order effects. What is the straightforward, "
            f"first-principles answer if the world were simple?\n\n"
            f"Problem: {query}\n\n"
            f"Provide a clear, concrete answer to this simplified version. "
            f"Be explicit about what simplifications you are making."
        )
        return await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are computing the zeroth-order solution. Simplify "
                "aggressively. Get the essential answer right, even if incomplete."
            ),
            temperature=self.zeroth_order_temperature,
        )

    # ------------------------------------------------------------------
    # Phase 2 -- PERTURBATION IDENTIFICATION
    # ------------------------------------------------------------------

    async def _identify_perturbations(
        self,
        query: str,
        zeroth_solution: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Identify complications ignored in the zeroth-order solution.

        Returns a list of perturbation dicts sorted by coupling strength
        (strongest first).

        Args:
            query: Original user query.
            zeroth_solution: The base solution from phase 1.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of perturbation dicts with keys ``perturbation``,
            ``coupling_strength``, and ``description``.
        """
        prompt = (
            f"What complications, real-world factors, edge cases, and "
            f"second-order effects were ignored in the zeroth-order solution? "
            f"List them in order of importance (largest perturbation first). "
            f"For each, estimate its 'coupling strength' -- how much it would "
            f"change the answer on a scale of 0.0 (negligible) to 1.0 "
            f"(completely changes the answer).\n\n"
            f"Original problem: {query}\n\n"
            f"Zeroth-order solution:\n{zeroth_solution}\n\n"
            f"Return ONLY a JSON array of objects with keys: "
            f'"perturbation", "coupling_strength", "description".\n'
            f'Example: [{{"perturbation": "regulatory compliance", '
            f'"coupling_strength": 0.8, '
            f'"description": "Financial regulations add 3-6 months"}}]'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system="You are a rigorous analyst. Return valid JSON only.",
            temperature=self.correction_temperature,
        )

        perturbations = self._parse_perturbation_list(raw)

        # Sort by coupling strength descending
        perturbations.sort(
            key=lambda p: p.get("coupling_strength", 0.0), reverse=True
        )
        return perturbations

    # ------------------------------------------------------------------
    # Phases 3+4 -- CORRECTIONS
    # ------------------------------------------------------------------

    async def _apply_correction(
        self,
        query: str,
        current_solution: str,
        perturbation: dict[str, Any],
        order_num: int,
        previous_perturbations: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Apply a single perturbative correction.

        At first order, this applies the strongest perturbation to the zeroth
        solution.  At higher orders, it accounts for cross-terms with
        previously applied perturbations.

        Args:
            query: Original user query.
            current_solution: The solution corrected up to the previous order.
            perturbation: The perturbation dict to apply at this order.
            order_num: The correction order (1 = first order).
            previous_perturbations: Perturbations already applied.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``corrected_solution`` and ``magnitude``.
        """
        pert_name = perturbation.get("perturbation", "unknown complication")
        pert_desc = perturbation.get("description", "")
        coupling = perturbation.get("coupling_strength", 0.5)

        if order_num == 1:
            cross_term_note = ""
        else:
            prev_names = [
                p.get("perturbation", "?") for p in previous_perturbations
            ]
            cross_term_note = (
                f"\n\nNote: this perturbation may interact with previously "
                f"applied corrections ({', '.join(prev_names)}). Account for "
                f"these cross-terms."
            )

        prompt = (
            f"The current answer incorporates corrections up to order "
            f"{order_num - 1}. Now apply the next perturbation.\n\n"
            f"Original problem: {query}\n\n"
            f"Current corrected solution:\n{current_solution}\n\n"
            f"Perturbation to apply (order {order_num}):\n"
            f"  Name: {pert_name}\n"
            f"  Coupling strength: {coupling:.2f}\n"
            f"  Description: {pert_desc}\n"
            f"{cross_term_note}\n\n"
            f"How does accounting for '{pert_name}' modify the answer? "
            f"Produce the corrected answer that incorporates this perturbation.\n\n"
            f"Also, at the end of your response, on a new line, state the "
            f"CORRECTION MAGNITUDE: a float between 0.0 and 1.0 indicating "
            f"how much this correction changed the answer (0.0 = no change, "
            f"1.0 = complete rewrite). Format: MAGNITUDE: 0.XX"
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                f"You are computing a perturbative correction at order "
                f"{order_num}. Modify the existing answer to account for "
                f"the new complication. Preserve the structure from previous "
                f"orders where possible."
            ),
            temperature=self.correction_temperature,
        )

        # Extract the magnitude from the response
        magnitude = self._extract_magnitude(raw, coupling)

        # Strip the magnitude line from the solution text
        corrected_solution = self._strip_magnitude_line(raw)

        return {
            "corrected_solution": corrected_solution,
            "magnitude": magnitude,
        }

    # ------------------------------------------------------------------
    # Phase 6 -- RESUMMATION / SYNTHESIS
    # ------------------------------------------------------------------

    async def _resum_and_synthesize(
        self,
        query: str,
        zeroth_solution: str,
        corrected_solution: str,
        correction_magnitudes: list[float],
        convergence_status: list[dict[str, Any]],
        diverged: bool,
        divergent_perturbation: str | None,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Produce the final answer via resummation.

        If the series converged, presents the perturbatively corrected answer.
        If it diverged, acknowledges the breakdown.

        Args:
            query: Original user query.
            zeroth_solution: The base zeroth-order solution.
            corrected_solution: The solution after all corrections.
            correction_magnitudes: List of magnitude values per order.
            convergence_status: List of convergence dicts per order.
            diverged: Whether the series diverged at any point.
            divergent_perturbation: Name of the perturbation that caused
                divergence, if any.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final synthesised answer text.
        """
        max_order = len(correction_magnitudes)
        magnitudes_str = ", ".join(f"{m:.4f}" for m in correction_magnitudes)

        convergence_note = ""
        if diverged and self.warn_on_divergence:
            convergence_note = (
                f"\n\nWARNING: The perturbation series DIVERGED at the "
                f"correction for '{divergent_perturbation}'. This means that "
                f"complication is too large for incremental correction -- it "
                f"requires fundamental rethinking (non-perturbative treatment). "
                f"Acknowledge this limitation in your answer."
            )
        else:
            convergence_note = (
                f"\n\nThe perturbation series CONVERGED. All corrections "
                f"diminished in magnitude, indicating the answer has stabilised."
            )

        prompt = (
            f"The perturbative expansion has been computed to order {max_order}.\n\n"
            f"Original problem: {query}\n\n"
            f"Zeroth-order (simplified) solution:\n{zeroth_solution}\n\n"
            f"Fully corrected solution (after {max_order} orders):\n"
            f"{corrected_solution}\n\n"
            f"Correction magnitudes by order: [{magnitudes_str}]\n"
            f"{convergence_note}\n\n"
            f"Present the final, polished answer. Structure it clearly. "
            f"If the series converged, present the corrected answer with "
            f"confidence. If it diverged, acknowledge which aspect requires "
            f"deeper analysis and present the best answer you can given "
            f"the breakdown."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing resummation of a perturbation series. "
                "Synthesise the corrections into a coherent, polished answer."
            ),
            temperature=self.correction_temperature,
        )

        resum_step = self._make_step(
            step_type="resummation",
            content=final,
            score=1.0 if not diverged else 0.7,
            metadata={
                "phase": "resummation",
                "max_order": max_order,
                "correction_magnitudes": [round(m, 4) for m in correction_magnitudes],
                "converged": not diverged,
                "divergent_perturbation": divergent_perturbation,
            },
        )
        trace.add_step(resum_step)
        return final

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_perturbation_list(raw: str) -> list[dict[str, Any]]:
        """Parse a JSON array of perturbation objects from LLM output.

        Falls back to a single generic perturbation if parsing fails.

        Args:
            raw: Raw LLM output string.

        Returns:
            List of perturbation dicts.
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
                                "perturbation": str(
                                    item.get("perturbation", "unknown")
                                ),
                                "coupling_strength": float(
                                    item.get("coupling_strength", 0.5)
                                ),
                                "description": str(
                                    item.get("description", "")
                                ),
                            })
                    if result:
                        return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback -- return a generic perturbation
        return [
            {
                "perturbation": "real-world complications",
                "coupling_strength": 0.5,
                "description": "General complications not captured in simplified model.",
            }
        ]

    @staticmethod
    def _extract_magnitude(raw: str, default_coupling: float) -> float:
        """Extract the correction magnitude from the LLM response.

        Looks for a ``MAGNITUDE: X.XX`` pattern.  Falls back to a heuristic
        based on coupling strength.

        Args:
            raw: Raw LLM response.
            default_coupling: Coupling strength to use as fallback.

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        pattern = r"MAGNITUDE:\s*([0-9]*\.?[0-9]+)"
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            except ValueError:
                pass

        # Heuristic fallback: use coupling strength scaled down
        return max(0.0, min(1.0, default_coupling * 0.6))

    @staticmethod
    def _strip_magnitude_line(raw: str) -> str:
        """Remove the MAGNITUDE: line from the response.

        Args:
            raw: Raw LLM response.

        Returns:
            The response with the magnitude line stripped.
        """
        lines = raw.splitlines()
        cleaned = [
            line for line in lines
            if not re.match(r"^\s*MAGNITUDE:\s*[0-9]", line, re.IGNORECASE)
        ]
        return "\n".join(cleaned).strip()


__all__ = ["PerturbativeExpansion"]
