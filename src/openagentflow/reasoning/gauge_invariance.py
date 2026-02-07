"""Gauge Invariance reasoning engine.

Applies the principle of gauge symmetry from electromagnetism and Yang-Mills
theory to extract frame-independent conclusions from multi-perspective
analysis.

Physics basis:

Gauge symmetry is the foundational principle of modern particle physics.
A **gauge transformation** is a local change of description that does not
change the physics.  In electromagnetism, the electric and magnetic fields
(the physical observables) are invariant under::

    A_mu -> A_mu + partial_mu * Lambda(x)

for any scalar function ``Lambda(x)``.  The potential ``A_mu`` is NOT physical
-- it is a convenient mathematical description.  Different gauges (different
choices of ``Lambda``) give different potentials but identical physical
predictions.

Key principles:

- **Gauge redundancy**: Multiple descriptions of the same physics.  The gauge
  freedom is not physical -- it is an artifact of the description.
- **Gauge invariants**: Quantities that are the same in ALL gauges are the
  physical observables.  Only gauge-invariant quantities correspond to
  measurable reality.
- **Gauge fixing**: To perform a concrete calculation, you must "fix a gauge"
  -- choose a specific description.  But the result must not depend on this
  choice.
- **Noether's theorem**: Every continuous symmetry implies a conservation law.
  Gauge symmetry implies conserved quantities that constrain all valid
  solutions.

The mapping to reasoning: different **framings**, **representations**, and
**perspectives** on a problem are like different gauges.  They are all valid
descriptions, but the **invariant** content -- the conclusions that hold
regardless of framing -- is the "physics."  The engine systematically
separates what changes across framings (gauge artifacts / framing effects)
from what stays the same (gauge invariants / robust conclusions).

LLM call pattern::

    MULTI_GAUGE(K) --> INVARIANT_EXTRACTION(1) --> CONSERVATION_LAWS(1) -->
    GAUGE_FIXED_SOLUTION(1) --> VERIFICATION(1)

    Total: K + 4 calls  (default K=5: 9 calls)

Example::

    from openagentflow.reasoning.gauge_invariance import GaugeInvariance

    engine = GaugeInvariance(num_gauges=5)
    trace = await engine.reason(
        query="Should our startup prioritise growth or profitability?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
    invariants = trace.get_steps_by_type("invariant_extraction")
    print(invariants[0].metadata["invariants"])
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

# Default gauge framings -- each represents a radically different way to
# represent the same problem.
_DEFAULT_GAUGE_FRAMINGS: list[dict[str, str]] = [
    {
        "name": "optimization",
        "system": (
            "You are an optimization theorist.  Frame the problem as an "
            "optimization problem with a clear objective function, decision "
            "variables, and constraints.  What is being maximised or "
            "minimised?  What are the trade-offs?"
        ),
    },
    {
        "name": "constraint_satisfaction",
        "system": (
            "You are a constraint-satisfaction specialist.  Frame the "
            "problem as a set of constraints that must all be satisfied "
            "simultaneously.  What are the hard constraints?  What are "
            "the soft constraints?  Is the feasible region empty?"
        ),
    },
    {
        "name": "narrative",
        "system": (
            "You are a narrative analyst.  Frame the problem as a story "
            "with actors, motivations, conflicts, and resolutions.  Who "
            "are the characters?  What are their goals?  What tensions "
            "drive the plot?"
        ),
    },
    {
        "name": "inverted",
        "system": (
            "You are a contrarian inversion specialist.  Invert the "
            "problem: instead of asking how to succeed, ask how to fail.  "
            "Instead of asking what to do, ask what to avoid.  Work "
            "backwards from the inverted perspective."
        ),
    },
    {
        "name": "abstract",
        "system": (
            "You are an abstract theorist.  Strip away all domain-specific "
            "details and analyse the problem in the most abstract terms "
            "possible.  What is the underlying mathematical or logical "
            "structure?  What class of problems does this belong to?"
        ),
    },
]


class GaugeInvariance(ReasoningEngine):
    """Gauge-invariance reasoning via multi-perspective invariant extraction.

    The engine analyses the same problem from ``num_gauges`` radically
    different perspectives (gauge choices), then systematically identifies:

    - **Gauge invariants**: conclusions that appear across ALL or MOST
      perspectives -- the "physical observables" that are robust to framing.
    - **Gauge artifacts**: claims that appear in only one perspective and
      are likely artifacts of that particular representation.
    - **Conservation laws**: inviolable principles implied by the symmetry
      across perspectives (via Noether's theorem).

    A concrete solution is then constructed in the most natural gauge,
    constrained to satisfy all conservation laws and invariants.  Finally,
    a gauge-independence verification checks that the core conclusions
    would survive a change of framing.

    Attributes:
        name: ``"gauge_invariance"``
        description: Short human-readable summary.
        num_gauges: Number of distinct framings to analyse.
        gauge_framings: List of gauge definitions (name + system prompt).
        robustness_threshold: Minimum fraction of gauges in which a
            conclusion must appear to qualify as "invariant".
        gauge_temperature: LLM temperature for gauge-specific analysis.
        solution_temperature: LLM temperature for solution construction.
    """

    name: str = "gauge_invariance"
    description: str = (
        "Analyzes the problem under multiple framings (gauges) "
        "and extracts the invariant conclusions that hold regardless "
        "of representation."
    )

    def __init__(
        self,
        num_gauges: int = 5,
        gauge_framings: list[dict[str, str]] | None = None,
        robustness_threshold: float = 0.6,
        gauge_temperature: float = 0.5,
        solution_temperature: float = 0.4,
    ) -> None:
        """Initialise the GaugeInvariance engine.

        Args:
            num_gauges: Number of distinct framings (gauges) to use.
            gauge_framings: Optional list of gauge definitions, each a
                dict with ``name`` (str) and ``system`` (str, the system
                prompt).  Must have length >= ``num_gauges`` if provided.
            robustness_threshold: Minimum fraction of gauges in which a
                conclusion must appear to be classified as a gauge
                invariant (0.0--1.0).
            gauge_temperature: LLM temperature for the gauge-specific
                analysis calls.
            solution_temperature: LLM temperature for the gauge-fixed
                solution and verification calls.
        """
        self.num_gauges = max(2, num_gauges)
        self.robustness_threshold = max(0.1, min(1.0, robustness_threshold))
        self.gauge_temperature = max(0.1, gauge_temperature)
        self.solution_temperature = max(0.1, solution_temperature)

        if gauge_framings is not None:
            if len(gauge_framings) < self.num_gauges:
                raise ValueError(
                    f"gauge_framings has {len(gauge_framings)} entries but "
                    f"num_gauges is {self.num_gauges}"
                )
            self.gauge_framings = list(gauge_framings)
        else:
            base = _DEFAULT_GAUGE_FRAMINGS
            self.gauge_framings = [
                base[i % len(base)] for i in range(self.num_gauges)
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
        """Execute the gauge-invariance reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for all generation and evaluation.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on the number of gauges, overriding
                ``num_gauges`` if smaller.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing gauge-representation,
            invariant-extraction, conservation-law, gauge-fixed-solution,
            and gauge-check steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        effective_gauges = min(self.num_gauges, max_iterations)

        # ---------------------------------------------------------------
        # Phase 1: MULTI-GAUGE GENERATION (K calls)
        # ---------------------------------------------------------------
        gauge_analyses = await self._generate_gauge_representations(
            query, effective_gauges, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 2: INVARIANT EXTRACTION (1 call)
        # ---------------------------------------------------------------
        invariant_result = await self._extract_invariants(
            query, gauge_analyses, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 3: CONSERVATION LAW DERIVATION (1 call)
        # ---------------------------------------------------------------
        conservation_laws = await self._derive_conservation_laws(
            query, invariant_result, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 4: GAUGE-FIXED SOLUTION (1 call)
        # ---------------------------------------------------------------
        solution = await self._gauge_fixed_solution(
            query,
            invariant_result,
            conservation_laws,
            gauge_analyses,
            llm_provider,
            trace,
        )

        # ---------------------------------------------------------------
        # Phase 5: GAUGE INDEPENDENCE VERIFICATION (1 call)
        # ---------------------------------------------------------------
        final_output = await self._verify_gauge_independence(
            query, solution, invariant_result, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Multi-gauge generation
    # ------------------------------------------------------------------

    async def _generate_gauge_representations(
        self,
        query: str,
        num_gauges: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Analyse the problem under each gauge (framing).

        Each gauge uses a unique system prompt that enforces a radically
        different way of looking at the problem.

        Args:
            query: Original user query.
            num_gauges: Number of gauges to generate.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of dicts with ``gauge_name``, ``system_prompt``,
            ``analysis``, and ``key_claims`` for each gauge.
        """
        gauge_analyses: list[dict[str, Any]] = []

        for idx in range(num_gauges):
            framing = self.gauge_framings[idx]
            gauge_name = framing["name"]
            system_prompt = framing["system"]

            prompt = (
                f"Analyze the following problem ENTIRELY through the lens "
                f"of your specific framing.  Do not break character.  "
                f"Commit fully to this perspective.\n\n"
                f"At the end, list 3-7 KEY CLAIMS -- the most important "
                f"conclusions from this framing.  Format them as a "
                f"numbered list.\n\n"
                f"Problem: {query}"
            )

            analysis = await self._call_llm(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                trace=trace,
                system=system_prompt,
                temperature=self.gauge_temperature,
            )

            # Extract key claims from the analysis
            key_claims = self._extract_claims_from_text(analysis)

            gauge_data = {
                "gauge_name": gauge_name,
                "gauge_index": idx,
                "system_prompt": system_prompt[:100],
                "analysis": analysis,
                "key_claims": key_claims,
            }
            gauge_analyses.append(gauge_data)

            step = self._make_step(
                step_type="gauge_representation",
                content=analysis,
                score=0.0,
                metadata={
                    "phase": "gauge_representation",
                    "gauge_name": gauge_name,
                    "gauge_index": idx,
                    "num_claims": len(key_claims),
                    "key_claims": key_claims,
                },
            )
            trace.add_step(step)

        return gauge_analyses

    # ------------------------------------------------------------------
    # Phase 2: Invariant extraction
    # ------------------------------------------------------------------

    async def _extract_invariants(
        self,
        query: str,
        gauge_analyses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Identify gauge invariants and gauge artifacts.

        Gauge invariants appear across most gauges (above the robustness
        threshold).  Gauge artifacts appear in only one gauge.

        Args:
            query: Original user query.
            gauge_analyses: All gauge-specific analyses.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``invariants`` (list of dicts with ``claim``,
            ``appears_in``, ``robustness``) and ``artifacts`` (list of
            dicts with ``claim``, ``only_in``, ``why_artifact``).
        """
        # Format all gauge analyses for the LLM
        gauges_text = ""
        for ga in gauge_analyses:
            claims_text = "\n".join(
                f"    {i+1}. {c}" for i, c in enumerate(ga["key_claims"])
            ) or "    (no specific claims extracted)"
            gauges_text += (
                f"\n--- Gauge: {ga['gauge_name']} ---\n"
                f"Analysis summary:\n{ga['analysis'][:500]}...\n"
                f"Key claims:\n{claims_text}\n"
            )

        num_gauges = len(gauge_analyses)
        min_appearances = max(
            2, int(num_gauges * self.robustness_threshold)
        )

        prompt = (
            f"You have seen the SAME problem analyzed under "
            f"{num_gauges} different framings (gauges).\n\n"
            f"Problem: {query}\n\n"
            f"Gauge analyses:{gauges_text}\n\n"
            f"Identify:\n\n"
            f"1. **GAUGE INVARIANTS**: Conclusions, insights, or claims "
            f"that appear in {min_appearances}+ of the {num_gauges} "
            f"framings, regardless of how the problem was represented.  "
            f"These are the 'physical observables' -- the robust truths.\n\n"
            f"2. **GAUGE ARTIFACTS**: Claims that appear in ONLY ONE "
            f"framing and are likely artifacts of that particular "
            f"representation -- things that seem true only because of how "
            f"you are looking at the problem.\n\n"
            f"3. **APPARENT CONTRADICTIONS**: Claims that seem contradictory "
            f"across gauges but may actually be gauge artifacts (the same "
            f"truth expressed differently in different frames).\n\n"
            f"Respond with JSON:\n"
            f'{{"invariants": ['
            f'{{"claim": "...", "appears_in": ["gauge1", "gauge2"], '
            f'"robustness": 0.8}}], '
            f'"artifacts": ['
            f'{{"claim": "...", "only_in": "gauge_name", '
            f'"why_artifact": "..."}}], '
            f'"apparent_contradictions": ['
            f'{{"claim_a": "...", "gauge_a": "...", '
            f'"claim_b": "...", "gauge_b": "...", '
            f'"resolution": "..."}}]}}'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are extracting gauge-invariant content.  Only "
                "conclusions that survive changes of framing are physical.  "
                "Everything else is a representation artifact.  Be rigorous.  "
                "Return valid JSON only."
            ),
            temperature=0.3,
        )

        result = self._parse_invariants(raw)

        step = self._make_step(
            step_type="invariant_extraction",
            content=(
                f"Invariants: {len(result['invariants'])} found\n"
                f"Artifacts: {len(result['artifacts'])} found\n"
                f"Apparent contradictions: "
                f"{len(result.get('apparent_contradictions', []))} found"
            ),
            score=len(result["invariants"]) / max(
                1,
                len(result["invariants"]) + len(result["artifacts"]),
            ),
            metadata={
                "phase": "invariant_extraction",
                "invariants": result["invariants"],
                "artifacts": result["artifacts"],
                "apparent_contradictions": result.get(
                    "apparent_contradictions", []
                ),
                "num_invariants": len(result["invariants"]),
                "num_artifacts": len(result["artifacts"]),
                "robustness_threshold": self.robustness_threshold,
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 3: Conservation law derivation
    # ------------------------------------------------------------------

    async def _derive_conservation_laws(
        self,
        query: str,
        invariant_result: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Derive conservation laws from the gauge symmetry.

        By Noether's theorem, every continuous symmetry implies a
        conservation law.  The gauge invariants imply that certain
        properties must be conserved across all valid framings.

        Args:
            query: Original user query.
            invariant_result: Output from invariant extraction.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The conservation laws as a string.
        """
        invariants_text = "\n".join(
            f"  - {inv['claim']} (robustness: {inv.get('robustness', 'N/A')})"
            for inv in invariant_result.get("invariants", [])
        ) or "  (none identified)"

        contradictions_text = "\n".join(
            f"  - {c.get('claim_a', '')} vs {c.get('claim_b', '')}: "
            f"{c.get('resolution', 'unresolved')}"
            for c in invariant_result.get("apparent_contradictions", [])
        ) or "  (none)"

        prompt = (
            f"Noether's theorem: every symmetry implies a conservation "
            f"law.  The gauge invariants identified below imply that "
            f"certain properties are CONSERVED across all framings.\n\n"
            f"Problem: {query}\n\n"
            f"Gauge invariants:\n{invariants_text}\n\n"
            f"Apparent contradictions resolved:\n{contradictions_text}\n\n"
            f"Derive the CONSERVATION LAWS:\n"
            f"1. What properties MUST be preserved regardless of how the "
            f"problem is approached?\n"
            f"2. What constraints do they impose on any valid solution?\n"
            f"3. What are the INVIOLABLE principles that any answer must "
            f"respect?\n\n"
            f"These are the non-negotiable requirements that survive every "
            f"change of perspective."
        )

        laws = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are deriving conservation laws from symmetry.  What "
                "must be preserved regardless of how the problem is "
                "approached?  Be rigorous -- these are inviolable "
                "constraints on any valid solution."
            ),
            temperature=0.3,
        )

        step = self._make_step(
            step_type="conservation_law",
            content=laws,
            score=0.0,
            metadata={
                "phase": "conservation_law",
                "num_invariants_used": len(
                    invariant_result.get("invariants", [])
                ),
            },
        )
        trace.add_step(step)

        return laws

    # ------------------------------------------------------------------
    # Phase 4: Gauge-fixed solution
    # ------------------------------------------------------------------

    async def _gauge_fixed_solution(
        self,
        query: str,
        invariant_result: dict[str, Any],
        conservation_laws: str,
        gauge_analyses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Construct a concrete solution in a chosen gauge.

        Picks the most natural and useful framing, then constructs a
        solution that satisfies all conservation laws and is consistent
        with all gauge invariants.

        Args:
            query: Original user query.
            invariant_result: Output from invariant extraction.
            conservation_laws: Derived conservation laws.
            gauge_analyses: All gauge-specific analyses.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The gauge-fixed solution string.
        """
        invariants_text = "\n".join(
            f"  - {inv['claim']}"
            for inv in invariant_result.get("invariants", [])
        ) or "  (none identified)"

        artifacts_text = "\n".join(
            f"  - {art['claim']} (only in: {art.get('only_in', 'unknown')})"
            for art in invariant_result.get("artifacts", [])
        ) or "  (none identified)"

        gauge_names = ", ".join(
            ga["gauge_name"] for ga in gauge_analyses
        )

        prompt = (
            f"Now FIX A GAUGE -- choose the single most natural and useful "
            f"framing for this problem.  Within that gauge, construct a "
            f"concrete solution.\n\n"
            f"Problem: {query}\n\n"
            f"Available gauges: {gauge_names}\n\n"
            f"GAUGE INVARIANTS (must be satisfied in ANY gauge):\n"
            f"{invariants_text}\n\n"
            f"GAUGE ARTIFACTS (framing-specific, may include or exclude):\n"
            f"{artifacts_text}\n\n"
            f"CONSERVATION LAWS (inviolable constraints):\n"
            f"{conservation_laws}\n\n"
            f"Your solution MUST:\n"
            f"1. Satisfy ALL conservation laws.\n"
            f"2. Be consistent with ALL gauge invariants.\n"
            f"3. Clearly distinguish which parts of the solution are "
            f"gauge-invariant (would be the same in any framing) versus "
            f"gauge-dependent (specific to your chosen framing).\n\n"
            f"State which gauge you are fixing, then present the solution."
        )

        solution = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are constructing a gauge-fixed solution.  Choose the "
                "most natural framing and build a concrete answer.  Clearly "
                "mark which conclusions are universal (gauge-invariant) and "
                "which are framing-specific (gauge-dependent)."
            ),
            temperature=self.solution_temperature,
        )

        step = self._make_step(
            step_type="gauge_fixed_solution",
            content=solution,
            score=0.8,
            metadata={
                "phase": "gauge_fixed_solution",
                "num_invariants": len(
                    invariant_result.get("invariants", [])
                ),
                "num_artifacts": len(
                    invariant_result.get("artifacts", [])
                ),
            },
        )
        trace.add_step(step)

        return solution

    # ------------------------------------------------------------------
    # Phase 5: Gauge independence verification
    # ------------------------------------------------------------------

    async def _verify_gauge_independence(
        self,
        query: str,
        solution: str,
        invariant_result: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Verify that the solution's core is gauge-independent.

        Checks whether someone approaching the problem from a completely
        different framing would reach the same essential conclusions.

        Args:
            query: Original user query.
            solution: The gauge-fixed solution.
            invariant_result: Output from invariant extraction.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The verified and polished final answer.
        """
        prompt = (
            f"GAUGE INDEPENDENCE VERIFICATION:\n\n"
            f"Problem: {query}\n\n"
            f"Proposed solution:\n{solution}\n\n"
            f"Verify that the solution's CORE CONCLUSIONS are "
            f"gauge-independent:\n\n"
            f"1. If someone approached this problem from a COMPLETELY "
            f"different framing, would they reach the same essential "
            f"conclusions?\n"
            f"2. Which parts of the solution are truly frame-independent "
            f"(robust truths)?\n"
            f"3. Which parts are framing artifacts that might change under "
            f"a different perspective?\n"
            f"4. Are there any remaining apparent contradictions that are "
            f"actually just gauge artifacts (the same truth expressed "
            f"differently)?\n\n"
            f"Produce the FINAL polished answer that:\n"
            f"- Leads with gauge-invariant conclusions (the robust truths)\n"
            f"- Clearly marks any gauge-dependent recommendations as "
            f"framing-specific\n"
            f"- Resolves any remaining apparent contradictions\n"
            f"- Is self-consistent and complete"
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing a final gauge-independence check.  "
                "Ensure the answer separates robust, frame-independent "
                "truths from representation-specific details.  The final "
                "answer should withstand any change of perspective."
            ),
            temperature=self.solution_temperature,
        )

        check_step = self._make_step(
            step_type="gauge_check",
            content=final,
            score=1.0,
            metadata={
                "phase": "gauge_check",
                "num_invariants": len(
                    invariant_result.get("invariants", [])
                ),
            },
        )
        trace.add_step(check_step)

        output_step = self._make_step(
            step_type="final_output",
            content=final,
            score=1.0,
            metadata={"phase": "final"},
            parent_step_id=check_step.step_id,
        )
        trace.add_step(output_step)

        return final

    # ------------------------------------------------------------------
    # Text / JSON parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_claims_from_text(text: str) -> list[str]:
        """Extract numbered claims from analysis text.

        Looks for numbered items (``1. ...``, ``1) ...``) in the text,
        typically from a "Key Claims" section.

        Args:
            text: Full analysis text from an LLM response.

        Returns:
            List of extracted claim strings.
        """
        # Look for a "key claims" section
        claims_section = text
        for marker in ("key claim", "KEY CLAIM", "Key Claim"):
            idx = text.lower().find(marker.lower())
            if idx != -1:
                claims_section = text[idx:]
                break

        # Extract numbered items
        pattern = r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)"
        matches = re.findall(pattern, claims_section, re.DOTALL)
        claims = [m.strip() for m in matches if m.strip()]

        if claims:
            return claims[:10]  # Cap at 10 claims

        # Fallback: split by newlines and take non-empty lines
        lines = [
            line.strip()
            for line in claims_section.split("\n")
            if line.strip() and len(line.strip()) > 20
        ]
        return lines[:5] if lines else ["(no specific claims extracted)"]

    @staticmethod
    def _parse_invariants(raw: str) -> dict[str, Any]:
        """Parse invariant-extraction output from the LLM.

        Expects a JSON object with ``invariants``, ``artifacts``, and
        optionally ``apparent_contradictions``.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``invariants`` (list of dicts), ``artifacts``
            (list of dicts), and ``apparent_contradictions`` (list of
            dicts).
        """
        default: dict[str, Any] = {
            "invariants": [],
            "artifacts": [],
            "apparent_contradictions": [],
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

        # Parse invariants
        invariants_raw = parsed.get("invariants", [])
        invariants: list[dict[str, Any]] = []
        if isinstance(invariants_raw, list):
            for inv in invariants_raw:
                if isinstance(inv, dict):
                    claim = inv.get("claim", "")
                    if claim:
                        appears_in = inv.get("appears_in", [])
                        if not isinstance(appears_in, list):
                            appears_in = [str(appears_in)]
                        robustness = inv.get("robustness", 0.5)
                        try:
                            robustness = max(0.0, min(1.0, float(robustness)))
                        except (TypeError, ValueError):
                            robustness = 0.5
                        invariants.append({
                            "claim": str(claim),
                            "appears_in": [str(a) for a in appears_in],
                            "robustness": robustness,
                        })
                elif isinstance(inv, str) and inv:
                    invariants.append({
                        "claim": inv,
                        "appears_in": [],
                        "robustness": 0.5,
                    })

        # Parse artifacts
        artifacts_raw = parsed.get("artifacts", [])
        artifacts: list[dict[str, Any]] = []
        if isinstance(artifacts_raw, list):
            for art in artifacts_raw:
                if isinstance(art, dict):
                    claim = art.get("claim", "")
                    if claim:
                        artifacts.append({
                            "claim": str(claim),
                            "only_in": str(art.get("only_in", "unknown")),
                            "why_artifact": str(
                                art.get("why_artifact", "")
                            ),
                        })
                elif isinstance(art, str) and art:
                    artifacts.append({
                        "claim": art,
                        "only_in": "unknown",
                        "why_artifact": "",
                    })

        # Parse apparent contradictions
        contradictions_raw = parsed.get("apparent_contradictions", [])
        contradictions: list[dict[str, Any]] = []
        if isinstance(contradictions_raw, list):
            for cont in contradictions_raw:
                if isinstance(cont, dict):
                    contradictions.append({
                        "claim_a": str(cont.get("claim_a", "")),
                        "gauge_a": str(cont.get("gauge_a", "")),
                        "claim_b": str(cont.get("claim_b", "")),
                        "gauge_b": str(cont.get("gauge_b", "")),
                        "resolution": str(cont.get("resolution", "")),
                    })

        return {
            "invariants": invariants,
            "artifacts": artifacts,
            "apparent_contradictions": contradictions,
        }


__all__ = ["GaugeInvariance"]
