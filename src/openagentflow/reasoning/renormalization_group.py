"""Renormalization Group reasoning engine.

Applies Wilson's renormalization group (coarse-graining) procedure to
reasoning, progressing bottom-up from maximum microscopic detail to
scale-invariant macroscopic conclusions.

Physics basis:

Kenneth Wilson's renormalization group (Nobel Prize 1982) provides a
systematic method for understanding how physical systems behave across
different length scales.  The core procedure:

1. **Coarse-grain**: Replace a block of microscopic degrees of freedom
   with a single effective degree of freedom (e.g., replace a block of
   spins with a single spin representing the block average).
2. **Rescale**: Zoom out to restore the original resolution.
3. **Renormalize**: Adjust coupling constants so the coarse-grained system
   has the same macroscopic behaviour as the original.

This is applied iteratively.  At each step, irrelevant microscopic details
are integrated out, and only features that matter at the current scale are
retained.  The transformation defines a flow in parameter space::

    K' = R(K)  (RG transformation: parameters K flow to K')

**Fixed points** of the RG flow (``K* = R(K*)``) correspond to
scale-invariant systems.  Near a fixed point, parameters are classified as:

- **Relevant**: grow under RG flow (matter at large scales).
- **Marginal**: unchanged under RG flow (borderline important).
- **Irrelevant**: shrink under RG flow (wash out at large scales).

This classification explains **universality**: systems with different
microscopic details but the same relevant parameters exhibit identical
macroscopic behaviour.

The engine is the opposite of FractalRecursion (which goes top-down).
RenormalizationGroup goes bottom-up from maximum detail, progressively
abstracting until scale-invariant conclusions emerge.

LLM call pattern::

    MICROSCOPIC(1) --> COARSE_GRAIN(S) --> FIXED_POINT(1) -->
    RECONSTRUCTION(1) --> UNIVERSALITY(1)

    Total: 4 + S calls  (default S=3: 7 calls)

Example::

    from openagentflow.reasoning.renormalization_group import RenormalizationGroup

    engine = RenormalizationGroup(num_scales=3)
    trace = await engine.reason(
        query="What are the key principles for building reliable distributed systems?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
    fixed = trace.get_steps_by_type("fixed_point")
    print(fixed[0].metadata["relevant"])
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openagentflow.reasoning.base import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
)

logger = logging.getLogger(__name__)

# Default scale labels from fine to coarse.
_DEFAULT_SCALE_LABELS: list[str] = [
    "component-level",
    "system-level",
    "strategic",
]


class RenormalizationGroup(ReasoningEngine):
    """Renormalization-group coarse-graining applied to reasoning.

    The engine starts at the finest grain of detail (microscopic analysis),
    then iteratively coarse-grains: grouping related details into effective
    variables, replacing specific facts with general principles, and
    discarding details that are irrelevant at broader scales.

    After coarse-graining through all scales, the engine detects
    **fixed points** -- conclusions that remain invariant across scales --
    and classifies all findings as **relevant** (grow at larger scales),
    **marginal** (scale-independent), or **irrelevant** (wash out).

    The answer is then reconstructed top-down from the fixed points
    and relevant parameters, optionally enriched by a universality check
    that identifies analogous problems in the same universality class.

    Attributes:
        name: ``"renormalization_group"``
        description: Short human-readable summary.
        num_scales: Number of coarse-graining steps.
        scale_labels: Human-readable labels for each scale.
        microscopic_temperature: LLM temperature for the detailed phase.
        abstraction_temperature_base: Base temperature for abstraction.
        abstraction_temperature_step: Temperature increment per scale.
        enable_universality: Whether to perform the universality check.
    """

    name: str = "renormalization_group"
    description: str = (
        "Coarse-grains reasoning across scales, identifies "
        "scale-invariant fixed points, and classifies findings "
        "as relevant, marginal, or irrelevant."
    )

    def __init__(
        self,
        num_scales: int = 3,
        scale_labels: list[str] | None = None,
        microscopic_temperature: float = 0.4,
        abstraction_temperature_base: float = 0.3,
        abstraction_temperature_step: float = 0.1,
        enable_universality: bool = True,
    ) -> None:
        """Initialise the RenormalizationGroup engine.

        Args:
            num_scales: Number of coarse-graining steps (``S``).  Each
                step makes one LLM call.
            scale_labels: Human-readable labels for each coarse-graining
                level (e.g., ``["component-level", "system-level",
                "strategic"]``).  Must have length ``num_scales`` if
                provided.
            microscopic_temperature: LLM temperature for the initial
                microscopic analysis.
            abstraction_temperature_base: Base LLM temperature for
                coarse-graining calls.
            abstraction_temperature_step: Temperature increment added
                per scale level (allows slightly more creativity at
                higher abstraction).
            enable_universality: Whether to perform the final
                universality-class check (adds one LLM call).
        """
        self.num_scales = max(1, num_scales)
        self.microscopic_temperature = max(0.1, microscopic_temperature)
        self.abstraction_temperature_base = max(0.1, abstraction_temperature_base)
        self.abstraction_temperature_step = max(0.0, abstraction_temperature_step)
        self.enable_universality = enable_universality

        if scale_labels is not None:
            if len(scale_labels) < self.num_scales:
                raise ValueError(
                    f"scale_labels has {len(scale_labels)} entries but "
                    f"num_scales is {self.num_scales}"
                )
            self.scale_labels = list(scale_labels)
        else:
            base = _DEFAULT_SCALE_LABELS
            self.scale_labels = [
                base[i] if i < len(base) else f"scale-{i + 1}"
                for i in range(self.num_scales)
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
        """Execute the renormalization-group reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for all generation and evaluation.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on coarse-graining scales,
                overriding ``num_scales`` if smaller.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing microscopic,
            coarse-grain, fixed-point, reconstruction, and (optionally)
            universality steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        effective_scales = min(self.num_scales, max_iterations)

        # ---------------------------------------------------------------
        # Phase 1: MICROSCOPIC ANALYSIS (1 call)
        # ---------------------------------------------------------------
        microscopic = await self._microscopic_analysis(
            query, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 2: ITERATIVE COARSE-GRAINING (S calls)
        # ---------------------------------------------------------------
        previous_analysis = microscopic
        all_analyses: list[dict[str, Any]] = [
            {"scale": "microscopic", "content": microscopic}
        ]

        for scale_idx in range(effective_scales):
            scale_label = self.scale_labels[scale_idx]
            temperature = (
                self.abstraction_temperature_base
                + self.abstraction_temperature_step * (scale_idx + 1)
            )

            coarsened = await self._coarse_grain(
                query,
                previous_analysis,
                scale_label,
                scale_idx,
                temperature,
                llm_provider,
                trace,
            )

            all_analyses.append({
                "scale": scale_label,
                "content": coarsened,
            })
            previous_analysis = coarsened

        # ---------------------------------------------------------------
        # Phase 3: FIXED-POINT DETECTION (1 call)
        # ---------------------------------------------------------------
        fixed_point_result = await self._detect_fixed_points(
            query, all_analyses, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 4: TOP-DOWN RECONSTRUCTION (1 call)
        # ---------------------------------------------------------------
        reconstruction = await self._reconstruct(
            query, fixed_point_result, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 5: UNIVERSALITY CHECK (1 call, optional)
        # ---------------------------------------------------------------
        if self.enable_universality:
            final_output = await self._universality_check(
                query, reconstruction, fixed_point_result, llm_provider, trace
            )
        else:
            final_output = reconstruction
            output_step = self._make_step(
                step_type="final_output",
                content=final_output,
                score=1.0,
                metadata={"phase": "final", "universality_skipped": True},
            )
            trace.add_step(output_step)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Microscopic analysis
    # ------------------------------------------------------------------

    async def _microscopic_analysis(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Analyse the problem at the finest grain of detail.

        Captures every specific fact, constraint, variable, edge case,
        and nuance without simplification.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Detailed microscopic analysis string.
        """
        prompt = (
            f"Analyze the following problem at the FINEST grain of detail.  "
            f"List EVERY specific fact, constraint, variable, edge case, "
            f"exception, and nuance.  Do NOT simplify -- capture the full "
            f"microscopic complexity.  Include:\n"
            f"- All relevant variables and their interactions\n"
            f"- Edge cases and boundary conditions\n"
            f"- Implicit assumptions that should be made explicit\n"
            f"- Specific examples and concrete instances\n"
            f"- Technical details and implementation considerations\n\n"
            f"Problem: {query}"
        )

        analysis = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing microscopic analysis.  Capture every "
                "detail at the finest granularity.  Do not abstract or "
                "simplify.  Be exhaustive."
            ),
            temperature=self.microscopic_temperature,
        )

        step = self._make_step(
            step_type="microscopic",
            content=analysis,
            score=0.0,
            metadata={
                "phase": "microscopic",
                "scale": "microscopic",
                "detail_level": "maximum",
            },
        )
        trace.add_step(step)

        return analysis

    # ------------------------------------------------------------------
    # Phase 2: Coarse-graining
    # ------------------------------------------------------------------

    async def _coarse_grain(
        self,
        query: str,
        previous_analysis: str,
        scale_label: str,
        scale_index: int,
        temperature: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Perform one coarse-graining step.

        Takes the analysis from the previous (finer) scale and
        coarse-grains it: groups related details into effective
        variables, replaces specifics with general principles, and
        discards details that are irrelevant at the broader scale.

        Args:
            query: Original user query.
            previous_analysis: Analysis from the previous scale.
            scale_label: Human-readable label for this scale.
            scale_index: Zero-based index of this scale.
            temperature: LLM temperature for this call.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The coarse-grained analysis at this scale.
        """
        prompt = (
            f"You are performing a RENORMALIZATION GROUP TRANSFORMATION.\n\n"
            f"Take the analysis from the previous scale and COARSE-GRAIN it "
            f"to the '{scale_label}' level:\n\n"
            f"1. GROUP related details into effective variables (replace "
            f"many specific facts with one general principle).\n"
            f"2. REPLACE specific instances with the general patterns they "
            f"represent.\n"
            f"3. DISCARD details that are irrelevant at this broader scale "
            f"-- details that do not affect the behaviour when you zoom "
            f"out.\n"
            f"4. KEEP details that persist when you zoom out -- these are "
            f"the relevant operators.\n\n"
            f"For each item, explicitly classify it:\n"
            f"- KEPT (relevant at this scale)\n"
            f"- GROUPED (merged into an effective variable)\n"
            f"- DISCARDED (irrelevant at this scale)\n\n"
            f"Problem: {query}\n\n"
            f"Previous-scale analysis:\n{previous_analysis}\n\n"
            f"Produce the coarse-grained analysis at '{scale_label}' "
            f"granularity."
        )

        coarsened = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                f"You are performing renormalization at the "
                f"'{scale_label}' scale.  Integrate out fine-grained "
                f"degrees of freedom.  Keep only what matters at this "
                f"level of abstraction."
            ),
            temperature=min(temperature, 1.0),
        )

        step = self._make_step(
            step_type="coarse_grain",
            content=coarsened,
            score=float(scale_index + 1) / self.num_scales,
            metadata={
                "phase": "coarse_grain",
                "scale": scale_label,
                "scale_index": scale_index,
                "temperature": round(temperature, 4),
            },
        )
        trace.add_step(step)

        return coarsened

    # ------------------------------------------------------------------
    # Phase 3: Fixed-point detection
    # ------------------------------------------------------------------

    async def _detect_fixed_points(
        self,
        query: str,
        all_analyses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Compare analyses at all scales to find fixed points.

        Fixed points are conclusions or principles that remain true
        regardless of the scale of analysis.  All findings are also
        classified as relevant, marginal, or irrelevant.

        Args:
            query: Original user query.
            all_analyses: List of dicts with ``scale`` and ``content``
                for every analysis level (microscopic + all coarse-
                grained levels).
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``fixed_points`` (list of str), ``relevant``
            (list of str), ``marginal`` (list of str), and
            ``irrelevant`` (list of str).
        """
        analyses_text = ""
        for a in all_analyses:
            analyses_text += (
                f"\n--- Scale: {a['scale']} ---\n{a['content']}\n"
            )

        prompt = (
            f"Compare the analyses at ALL scales below.  Identify:\n\n"
            f"1. **FIXED POINTS**: Conclusions or principles that remain "
            f"true and UNCHANGED regardless of the scale of analysis.  "
            f"These are the scale-invariant truths -- the conclusions that "
            f"survive every level of coarse-graining.\n\n"
            f"2. **CLASSIFICATION** of every finding:\n"
            f"   - RELEVANT: Matters MORE at larger scales (strategic "
            f"importance, grows under coarse-graining).\n"
            f"   - MARGINAL: Equally important at ALL scales (borderline).\n"
            f"   - IRRELEVANT: Matters only at fine scales, washes out when "
            f"you zoom out (implementation detail).\n\n"
            f"Problem: {query}\n\n"
            f"Analyses at different scales:{analyses_text}\n\n"
            f"Respond with JSON:\n"
            f'{{"fixed_points": ["point1", "point2"], '
            f'"relevant": ["finding1", "finding2"], '
            f'"marginal": ["finding3"], '
            f'"irrelevant": ["detail1", "detail2"]}}'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are detecting renormalization-group fixed points.  "
                "A fixed point is a conclusion that is invariant under "
                "changes of scale.  Be rigorous -- only true invariants "
                "qualify.  Return valid JSON only."
            ),
            temperature=0.3,
        )

        result = self._parse_fixed_points(raw)

        step = self._make_step(
            step_type="fixed_point",
            content=(
                f"Fixed points: {result['fixed_points']}\n"
                f"Relevant: {result['relevant']}\n"
                f"Marginal: {result['marginal']}\n"
                f"Irrelevant: {result['irrelevant']}"
            ),
            score=len(result["fixed_points"]) / max(
                1,
                len(result["fixed_points"])
                + len(result["relevant"])
                + len(result["marginal"])
                + len(result["irrelevant"]),
            ),
            metadata={
                "phase": "fixed_point",
                "fixed_points": result["fixed_points"],
                "relevant": result["relevant"],
                "marginal": result["marginal"],
                "irrelevant": result["irrelevant"],
                "num_fixed_points": len(result["fixed_points"]),
                "num_relevant": len(result["relevant"]),
                "num_marginal": len(result["marginal"]),
                "num_irrelevant": len(result["irrelevant"]),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 4: Top-down reconstruction
    # ------------------------------------------------------------------

    async def _reconstruct(
        self,
        query: str,
        fixed_point_result: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Reconstruct the answer top-down from fixed points.

        The fixed points provide the skeleton, the relevant parameters
        determine the structure, the marginal parameters add nuance,
        and the irrelevant parameters provide optional detail.

        Args:
            query: Original user query.
            fixed_point_result: Output from fixed-point detection.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The reconstructed answer string.
        """
        fixed_text = "\n".join(
            f"  - {fp}" for fp in fixed_point_result["fixed_points"]
        ) or "  (none identified)"
        relevant_text = "\n".join(
            f"  - {r}" for r in fixed_point_result["relevant"]
        ) or "  (none identified)"
        marginal_text = "\n".join(
            f"  - {m}" for m in fixed_point_result["marginal"]
        ) or "  (none identified)"
        irrelevant_text = "\n".join(
            f"  - {i}" for i in fixed_point_result["irrelevant"]
        ) or "  (none identified)"

        prompt = (
            f"Starting from the scale-invariant fixed points and relevant "
            f"parameters, reconstruct the answer TOP-DOWN.\n\n"
            f"Problem: {query}\n\n"
            f"FIXED POINTS (scale-invariant skeleton):\n{fixed_text}\n\n"
            f"RELEVANT PARAMETERS (determine structure):\n{relevant_text}\n\n"
            f"MARGINAL PARAMETERS (add nuance):\n{marginal_text}\n\n"
            f"IRRELEVANT PARAMETERS (optional detail):\n{irrelevant_text}\n\n"
            f"Build the answer from the top down:\n"
            f"1. Start with the fixed points as the foundation.\n"
            f"2. Add the relevant parameters as the main structure.\n"
            f"3. Include marginal parameters as nuance and caveats.\n"
            f"4. Mention irrelevant parameters only for completeness.\n\n"
            f"The result should clearly distinguish what is universal "
            f"(true regardless of details) from what is contingent "
            f"(depends on specific circumstances)."
        )

        reconstruction = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are reconstructing an answer from renormalization-group "
                "results.  Build top-down from universal principles to "
                "specific details.  Clearly mark what is universal versus "
                "what is contingent."
            ),
            temperature=0.4,
        )

        step = self._make_step(
            step_type="reconstruction",
            content=reconstruction,
            score=0.8,
            metadata={
                "phase": "reconstruction",
                "num_fixed_points": len(fixed_point_result["fixed_points"]),
                "num_relevant": len(fixed_point_result["relevant"]),
            },
        )
        trace.add_step(step)

        return reconstruction

    # ------------------------------------------------------------------
    # Phase 5: Universality check
    # ------------------------------------------------------------------

    async def _universality_check(
        self,
        query: str,
        reconstruction: str,
        fixed_point_result: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Check for universality -- are there analogous problems?

        Identifies other problems that would flow to the same fixed
        point, and extracts lessons from those analogues.

        Args:
            query: Original user query.
            reconstruction: The reconstructed answer.
            fixed_point_result: Output from fixed-point detection.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final answer enriched with universality insights.
        """
        fixed_text = ", ".join(
            fixed_point_result["fixed_points"]
        ) if fixed_point_result["fixed_points"] else "(none)"

        prompt = (
            f"The renormalization analysis has identified the essential "
            f"structure of this problem.\n\n"
            f"Problem: {query}\n\n"
            f"Fixed points (scale-invariant truths): {fixed_text}\n\n"
            f"Reconstructed answer:\n{reconstruction}\n\n"
            f"UNIVERSALITY CHECK:\n"
            f"Are there other, apparently DIFFERENT problems that would "
            f"flow to the SAME fixed point -- i.e., exhibit the same "
            f"essential structure despite superficial differences?\n\n"
            f"If so:\n"
            f"1. Identify 2-3 analogous problems in different domains.\n"
            f"2. What lessons from those analogous problems apply here?\n"
            f"3. How does the universality class inform the answer?\n\n"
            f"Produce the final polished answer that incorporates these "
            f"universality insights where they strengthen the argument."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing a universality analysis.  Problems in "
                "the same universality class share the same essential "
                "structure despite different microscopic details.  Use "
                "this to strengthen and validate the answer."
            ),
            temperature=0.5,
        )

        univ_step = self._make_step(
            step_type="universality",
            content=final,
            score=1.0,
            metadata={
                "phase": "universality",
                "fixed_points": fixed_point_result["fixed_points"],
            },
        )
        trace.add_step(univ_step)

        output_step = self._make_step(
            step_type="final_output",
            content=final,
            score=1.0,
            metadata={"phase": "final"},
            parent_step_id=univ_step.step_id,
        )
        trace.add_step(output_step)

        return final

    # ------------------------------------------------------------------
    # JSON parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fixed_points(raw: str) -> dict[str, Any]:
        """Parse fixed-point detection output from the LLM.

        Expects a JSON object with ``fixed_points``, ``relevant``,
        ``marginal``, and ``irrelevant`` keys (each a list of strings).

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with four lists of strings.
        """
        default: dict[str, Any] = {
            "fixed_points": [],
            "relevant": [],
            "marginal": [],
            "irrelevant": [],
        }

        text = raw.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

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

        result: dict[str, Any] = {}
        for key in ("fixed_points", "relevant", "marginal", "irrelevant"):
            items = parsed.get(key, [])
            if not isinstance(items, list):
                items = [str(items)] if items else []
            result[key] = [str(item) for item in items if item]

        return result


__all__ = ["RenormalizationGroup"]
