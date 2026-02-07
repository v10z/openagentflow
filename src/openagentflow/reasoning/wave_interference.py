"""Wave Interference reasoning engine.

Models constructive and destructive wave interference, standing waves, and
boundary conditions applied to multi-perspective reasoning.  Hard constraints
act as boundary conditions that filter which argumentative modes can exist.
Destructive interference nodes are reported as areas of genuine uncertainty.

Physics basis:

When two or more waves overlap in space, the resulting amplitude at any point
is the algebraic sum of the individual amplitudes (the principle of
superposition for linear waves)::

    A_total(x, t) = sum_i  A_i * cos(k_i * x - omega_i * t + phi_i)

This produces interference patterns:

- **Constructive interference**: When waves arrive in phase
  (``phi_i - phi_j ~ 0``), amplitudes add, producing a peak.
- **Destructive interference**: When waves arrive out of phase
  (``phi_i - phi_j ~ pi``), amplitudes cancel, producing a node.
- **Beats**: Waves of slightly different frequencies superpose to produce
  a slowly modulating envelope -- periodically constructive and destructive.
- **Diffraction**: Waves bend around obstacles and spread through apertures,
  extending conclusions into adjacent unexplored regions.

Young's double-slit experiment (1801) demonstrated that light exhibits
interference.  The key insight for reasoning: ideas generated from different
perspectives produce an interference pattern when brought together.  Points
of constructive interference are where multiple perspectives agree and
reinforce.  Points of destructive interference reveal genuine contradictions
that must be addressed.

The engine uses approximately 5 LLM calls.

Example::

    from openagentflow.reasoning.wave_interference import WaveInterference

    engine = WaveInterference(num_perspectives=4, enable_diffraction=True)
    trace = await engine.reason(
        query="Should we adopt a four-day work week?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_WAVE_SOURCE_SYSTEM = (
    "You are generating coherent wave sources. Each perspective must be "
    "internally consistent (coherent) but may differ from others in "
    "assumptions, values, methodology, or domain. Produce perspectives "
    "that span genuinely different angles on the problem."
)

_PHASE_ALIGNMENT_SYSTEM = (
    "You are computing phase relationships between claims from different "
    "perspectives. Phase 0 means perfectly in phase (the claims agree and "
    "reinforce each other). Phase pi (3.14159) means perfectly out of phase "
    "(the claims directly contradict). Intermediate values represent partial "
    "agreement or tangential relationships. Also rate the coherence (0-1) "
    "of each pair -- how strongly they interact at all."
)

_INTERFERENCE_SYSTEM = (
    "You are analyzing the interference pattern that emerges when multiple "
    "reasoning perspectives are superposed. Constructive peaks represent "
    "strong consensus; destructive nodes represent genuine contradictions. "
    "For each destructive node, determine whether it reveals a real "
    "ambiguity or a resolvable error."
)

_DIFFRACTION_SYSTEM = (
    "You are computing diffracted reasoning -- extending established "
    "conclusions into adjacent territory. Like waves bending around "
    "obstacles and spreading through apertures, the existing strong "
    "conclusions can illuminate areas where no perspective explicitly "
    "reached. Extrapolate carefully."
)

_SYNTHESIS_SYSTEM = (
    "You are synthesizing a final answer from a complete interference "
    "pattern. Build the answer on constructive peaks (strong consensus). "
    "Address destructive nodes (contradictions) explicitly. Include "
    "diffracted insights (extended conclusions). Report areas of genuine "
    "uncertainty where perspectives cancel out."
)


class WaveInterference(ReasoningEngine):
    """Wave interference pattern analysis for multi-perspective reasoning.

    The engine generates N coherent reasoning perspectives (wave sources),
    computes the phase relationships between their claims, calculates the
    resulting interference pattern, optionally extends conclusions via
    diffraction, and synthesizes a final answer that explicitly identifies
    consensus peaks and contradiction nodes.

    This handles N-way superposition with continuous phase relationships
    and explicitly computes the full interference pattern -- not just
    pairwise conflicts.

    Attributes:
        name: ``"WaveInterference"``
        description: Short human-readable summary.
        num_perspectives: Number of wave source perspectives.
        coherence_threshold: Minimum coherence for a claim pair to count.
        constructive_threshold: Amplitude above which a claim is a peak.
        destructive_threshold: Amplitude below which a claim is a node.
        enable_diffraction: Whether to extend into gaps via diffraction.
        source_temperature: LLM temperature for wave source generation.
        synthesis_temperature: LLM temperature for the synthesis step.
    """

    name: str = "WaveInterference"
    description: str = (
        "Multiple reasoning perspectives interfere like waves. "
        "Constructive peaks reveal consensus; destructive nodes "
        "expose contradictions."
    )

    def __init__(
        self,
        num_perspectives: int = 4,
        coherence_threshold: float = 0.3,
        constructive_threshold: float = 0.6,
        destructive_threshold: float = 0.2,
        enable_diffraction: bool = True,
        source_temperature: float = 0.7,
        synthesis_temperature: float = 0.4,
    ) -> None:
        """Initialise the Wave Interference engine.

        Args:
            num_perspectives: Number of wave source perspectives to generate.
                More perspectives = richer interference pattern, more tokens.
            coherence_threshold: Minimum coherence (interaction strength) for
                a claim pair to be included in the interference calculation.
                Pairs below this threshold are treated as non-interacting.
            constructive_threshold: Resultant amplitude above which a claim
                is classified as a constructive peak (strong consensus).
            destructive_threshold: Resultant amplitude below which a claim
                is classified as a destructive node (contradiction).
            enable_diffraction: Whether to run the diffraction phase, which
                extends established conclusions into unexplored territory.
            source_temperature: LLM temperature for generating perspective
                analyses (higher = more diverse perspectives).
            synthesis_temperature: LLM temperature for the final synthesis.
        """
        self.num_perspectives = max(2, num_perspectives)
        self.coherence_threshold = max(0.0, min(1.0, coherence_threshold))
        self.constructive_threshold = max(0.0, min(1.0, constructive_threshold))
        self.destructive_threshold = max(0.0, min(1.0, destructive_threshold))
        self.enable_diffraction = enable_diffraction
        self.source_temperature = source_temperature
        self.synthesis_temperature = synthesis_temperature

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
        """Execute the Wave Interference reasoning strategy.

        Phases:

        1. Wave source generation -- N perspective analyses with claims
        2. Claim extraction and phase alignment -- phase matrix
        3. Interference pattern computation -- peaks and nodes
        4. Diffraction analysis (optional) -- extended conclusions
        5. Pattern synthesis -- final answer

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider.
            tools: Optional tool specs (currently unused).
            max_iterations: Safety cap (not typically binding for this engine).
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing wave source, phase alignment,
            interference pattern, diffraction, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- Generate wave sources (perspectives)
        perspectives = await self._generate_wave_sources(
            query, llm_provider, trace
        )

        # Phase 2 -- Extract claims and compute phase alignment
        phase_data = await self._compute_phase_alignment(
            query, perspectives, llm_provider, trace
        )

        # Phase 3 -- Compute interference pattern
        peaks, nodes, pattern_data = await self._compute_interference_pattern(
            query, perspectives, phase_data, llm_provider, trace
        )

        # Phase 4 -- Diffraction (optional)
        diffraction_insights = ""
        if self.enable_diffraction:
            diffraction_insights = await self._compute_diffraction(
                query, peaks, nodes, llm_provider, trace
            )

        # Phase 5 -- Pattern synthesis
        final_output = await self._synthesize_pattern(
            query, perspectives, peaks, nodes, diffraction_insights,
            llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Wave source generation
    # ------------------------------------------------------------------

    async def _generate_wave_sources(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate N coherent reasoning perspectives as wave sources.

        Each perspective is a self-consistent analysis from a specific angle,
        containing a description and a list of key claims.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of perspective dicts with ``perspective``, ``analysis``,
            and ``key_claims`` keys.
        """
        prompt = (
            f"Analyze the following problem from {self.num_perspectives} "
            f"distinct perspectives. Each perspective should be a coherent "
            f"wave of reasoning from a different angle: different domains, "
            f"assumptions, value systems, or methodologies.\n\n"
            f"Problem: {query}\n\n"
            f"For each perspective, provide:\n"
            f"- A name/label for the perspective\n"
            f"- A detailed analysis from that angle\n"
            f"- 3-5 key claims (specific, testable assertions)\n\n"
            f"Return a JSON object:\n"
            f'{{"perspectives": [\n'
            f'  {{"perspective": "name", "analysis": "detailed analysis", '
            f'"key_claims": ["claim 1", "claim 2", "claim 3"]}},\n'
            f"  ...\n"
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_WAVE_SOURCE_SYSTEM,
            temperature=self.source_temperature,
        )

        perspectives = self._parse_perspectives(raw)

        # Record wave source steps
        for i, persp in enumerate(perspectives):
            step = self._make_step(
                step_type="wave_source",
                content=(
                    f"Perspective: {persp.get('perspective', f'Source {i}')}\n\n"
                    f"{persp.get('analysis', 'N/A')}\n\n"
                    f"Key claims:\n"
                    + "\n".join(
                        f"  - {c}" for c in persp.get("key_claims", [])
                    )
                ),
                metadata={
                    "phase": "wave_sources",
                    "perspective_index": i,
                    "perspective_name": persp.get("perspective", f"Source {i}"),
                    "num_claims": len(persp.get("key_claims", [])),
                },
            )
            trace.add_step(step)

        return perspectives

    # ------------------------------------------------------------------
    # Phase 2: Phase alignment
    # ------------------------------------------------------------------

    async def _compute_phase_alignment(
        self,
        query: str,
        perspectives: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Compute phase relationships between claims across perspectives.

        For each pair of claims from different perspectives, determines the
        phase angle (0 = in phase / agreeing, pi = out of phase /
        contradicting) and coherence (interaction strength).

        Args:
            query: Original user query.
            perspectives: List of perspective dicts from Phase 1.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of claim-pair dicts with ``claim_i``, ``claim_j``,
            ``perspective_i``, ``perspective_j``, ``phase_angle``, and
            ``coherence`` keys.
        """
        # Collect all claims with source perspective labels
        all_claims: list[dict[str, str]] = []
        for p_idx, persp in enumerate(perspectives):
            p_name = persp.get("perspective", f"Source {p_idx}")
            for claim in persp.get("key_claims", []):
                all_claims.append({
                    "perspective": p_name,
                    "claim": claim,
                })

        # Build claim pairs for evaluation (cross-perspective only)
        claim_pairs: list[dict[str, Any]] = []
        for i in range(len(all_claims)):
            for j in range(i + 1, len(all_claims)):
                if all_claims[i]["perspective"] != all_claims[j]["perspective"]:
                    claim_pairs.append({
                        "idx_i": i,
                        "idx_j": j,
                        "claim_i": all_claims[i]["claim"],
                        "claim_j": all_claims[j]["claim"],
                        "perspective_i": all_claims[i]["perspective"],
                        "perspective_j": all_claims[j]["perspective"],
                    })

        # Limit pairs to avoid excessive prompt length
        max_pairs = 20
        if len(claim_pairs) > max_pairs:
            # Sample evenly across the list
            step_size = max(1, len(claim_pairs) // max_pairs)
            claim_pairs = claim_pairs[::step_size][:max_pairs]

        pairs_text = "\n\n".join(
            f"Pair {k + 1}:\n"
            f"  Claim A [{cp['perspective_i']}]: {cp['claim_i']}\n"
            f"  Claim B [{cp['perspective_j']}]: {cp['claim_j']}"
            for k, cp in enumerate(claim_pairs)
        )

        prompt = (
            f"PHASE ALIGNMENT COMPUTATION\n\n"
            f"Problem: {query}\n\n"
            f"Multiple perspectives have made specific claims. For each pair "
            f"of claims from different perspectives, determine:\n"
            f"- phase_angle: float in [0, {math.pi:.4f}] where 0 = perfectly "
            f"in phase (agree/reinforce), {math.pi:.4f} = perfectly out of "
            f"phase (contradict)\n"
            f"- coherence: float in [0, 1] -- how strongly these claims "
            f"interact (0 = unrelated, 1 = directly relevant to each other)\n\n"
            f"Claim pairs:\n{pairs_text}\n\n"
            f"Return a JSON object:\n"
            f'{{"alignments": [\n'
            f'  {{"pair_index": 1, "phase_angle": 0.5, "coherence": 0.8}},\n'
            f"  ...\n"
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_PHASE_ALIGNMENT_SYSTEM,
            temperature=0.3,
        )

        alignments = self._parse_alignments(raw, claim_pairs)

        # Merge alignment data back into claim pairs
        phase_data: list[dict[str, Any]] = []
        for k, cp in enumerate(claim_pairs):
            alignment = alignments.get(k, {})
            phase_data.append({
                "claim_i": cp["claim_i"],
                "claim_j": cp["claim_j"],
                "perspective_i": cp["perspective_i"],
                "perspective_j": cp["perspective_j"],
                "phase_angle": alignment.get("phase_angle", math.pi / 2),
                "coherence": alignment.get("coherence", 0.5),
            })

        step = self._make_step(
            step_type="phase_alignment",
            content=raw,
            metadata={
                "phase": "alignment",
                "num_claims": len(all_claims),
                "num_pairs_evaluated": len(claim_pairs),
                "mean_coherence": round(
                    sum(d["coherence"] for d in phase_data) / max(1, len(phase_data)),
                    4,
                ),
            },
        )
        trace.add_step(step)

        return phase_data

    # ------------------------------------------------------------------
    # Phase 3: Interference pattern
    # ------------------------------------------------------------------

    async def _compute_interference_pattern(
        self,
        query: str,
        perspectives: list[dict[str, Any]],
        phase_data: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """Compute the interference pattern from phase alignment data.

        For each claim, computes the resultant amplitude by summing
        contributions from all perspectives weighted by coherence::

            A_claim = sum_perspectives cos(phase_angle_p) * coherence_p

        Claims with high positive amplitude are constructive peaks.
        Claims with near-zero or negative amplitude are destructive nodes.

        Args:
            query: Original user query.
            perspectives: Perspective data from Phase 1.
            phase_data: Phase alignment data from Phase 2.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (constructive peaks list, destructive nodes list,
            raw LLM analysis text).
        """
        # Programmatic: compute resultant amplitude for each unique claim
        claim_amplitudes: dict[str, float] = {}
        claim_interactions: dict[str, list[tuple[str, float, float]]] = {}

        for pd in phase_data:
            ci = pd["claim_i"]
            cj = pd["claim_j"]
            phase = pd["phase_angle"]
            coh = pd["coherence"]

            if coh < self.coherence_threshold:
                continue

            contribution = math.cos(phase) * coh

            # Update claim_i amplitude
            claim_amplitudes.setdefault(ci, 0.0)
            claim_amplitudes[ci] += contribution
            claim_interactions.setdefault(ci, [])
            claim_interactions[ci].append(
                (cj, phase, contribution)
            )

            # Update claim_j amplitude (symmetric)
            claim_amplitudes.setdefault(cj, 0.0)
            claim_amplitudes[cj] += contribution
            claim_interactions.setdefault(cj, [])
            claim_interactions[cj].append(
                (ci, phase, contribution)
            )

        # Normalize amplitudes to [0, 1] range
        if claim_amplitudes:
            max_abs = max(abs(v) for v in claim_amplitudes.values()) or 1.0
            normalized = {
                k: (v / max_abs + 1.0) / 2.0
                for k, v in claim_amplitudes.items()
            }
        else:
            normalized = {}

        # Classify claims
        constructive_claims: list[str] = []
        destructive_claims: list[str] = []
        for claim, amp in normalized.items():
            if amp >= self.constructive_threshold:
                constructive_claims.append(claim)
            elif amp <= self.destructive_threshold:
                destructive_claims.append(claim)

        # LLM analysis of the interference pattern
        peaks_text = "\n".join(
            f"  - {c} (amplitude: {normalized.get(c, 0.0):.3f})"
            for c in constructive_claims
        ) or "  (none identified)"

        nodes_text = "\n".join(
            f"  - {c} (amplitude: {normalized.get(c, 0.0):.3f})"
            for c in destructive_claims
        ) or "  (none identified)"

        prompt = (
            f"INTERFERENCE PATTERN ANALYSIS\n\n"
            f"Problem: {query}\n\n"
            f"After computing phase alignments between claims from "
            f"{len(perspectives)} perspectives, the interference pattern "
            f"reveals:\n\n"
            f"CONSTRUCTIVE PEAKS (strong consensus):\n{peaks_text}\n\n"
            f"DESTRUCTIVE NODES (contradictions):\n{nodes_text}\n\n"
            f"For each constructive peak, explain why multiple perspectives "
            f"converge on this claim.\n"
            f"For each destructive node, explain:\n"
            f"1. Why the perspectives contradict\n"
            f"2. Whether the contradiction reveals a genuine ambiguity (real "
            f"uncertainty in the problem) or a resolvable error (one side is "
            f"clearly wrong)\n\n"
            f"Return a JSON object:\n"
            f'{{"peaks": [\n'
            f'  {{"claim": "...", "explanation": "why perspectives agree", '
            f'"strength": 0.0-1.0}}\n'
            f'], "nodes": [\n'
            f'  {{"claim": "...", "explanation": "why they contradict", '
            f'"is_genuine_ambiguity": true/false, "resolution": "if resolvable"}}\n'
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_INTERFERENCE_SYSTEM,
            temperature=0.4,
        )

        peaks, nodes = self._parse_pattern(raw)

        # Record the interference pattern step
        step = self._make_step(
            step_type="interference_pattern",
            content=raw,
            metadata={
                "phase": "interference",
                "num_constructive_peaks": len(peaks),
                "num_destructive_nodes": len(nodes),
                "total_claims_analyzed": len(claim_amplitudes),
                "mean_amplitude": round(
                    sum(normalized.values()) / max(1, len(normalized)),
                    4,
                ) if normalized else 0.0,
            },
        )
        trace.add_step(step)

        return peaks, nodes, raw

    # ------------------------------------------------------------------
    # Phase 4: Diffraction
    # ------------------------------------------------------------------

    async def _compute_diffraction(
        self,
        query: str,
        peaks: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Extend conclusions into unexplored territory via diffraction.

        Like waves bending around obstacles, the established constructive
        peaks can illuminate adjacent areas where no perspective explicitly
        reached.

        Args:
            query: Original user query.
            peaks: Constructive peaks from Phase 3.
            nodes: Destructive nodes from Phase 3.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Diffracted insights text.
        """
        peaks_text = "\n".join(
            f"  - {p.get('claim', 'N/A')}: {p.get('explanation', '')[:150]}"
            for p in peaks
        ) or "  (no strong peaks)"

        nodes_text = "\n".join(
            f"  - {n.get('claim', 'N/A')}: {n.get('explanation', '')[:150]}"
            for n in nodes
        ) or "  (no destructive nodes)"

        prompt = (
            f"DIFFRACTION ANALYSIS\n\n"
            f"Problem: {query}\n\n"
            f"The interference pattern has identified strong peaks (consensus) "
            f"and destructive nodes (contradictions). There are likely areas "
            f"the perspectives did not directly address.\n\n"
            f"CONSTRUCTIVE PEAKS:\n{peaks_text}\n\n"
            f"DESTRUCTIVE NODES:\n{nodes_text}\n\n"
            f"Like waves diffracting through an aperture, extend the strong "
            f"peaks into adjacent territory. What secondary conclusions can "
            f"be drawn by extrapolating from the established consensus? "
            f"What implications follow that no perspective stated explicitly?\n\n"
            f"Be careful not to over-extrapolate. Label each diffracted "
            f"insight with your confidence level."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_DIFFRACTION_SYSTEM,
            temperature=0.5,
        )

        step = self._make_step(
            step_type="diffraction",
            content=raw,
            metadata={
                "phase": "diffraction",
                "num_input_peaks": len(peaks),
                "num_input_nodes": len(nodes),
            },
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Phase 5: Pattern synthesis
    # ------------------------------------------------------------------

    async def _synthesize_pattern(
        self,
        query: str,
        perspectives: list[dict[str, Any]],
        peaks: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        diffraction_insights: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize the final answer from the interference pattern.

        Builds on constructive peaks, addresses destructive nodes, and
        incorporates diffracted insights.

        Args:
            query: Original user query.
            perspectives: Perspective data from Phase 1.
            peaks: Constructive peaks from Phase 3.
            nodes: Destructive nodes from Phase 3.
            diffraction_insights: Diffracted insights from Phase 4.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Final synthesized answer.
        """
        peaks_text = "\n".join(
            f"  PEAK: {p.get('claim', 'N/A')}\n"
            f"    Why: {p.get('explanation', '')[:200]}\n"
            f"    Strength: {p.get('strength', 0.0):.2f}"
            for p in peaks
        ) or "  (no strong consensus points)"

        nodes_text = "\n".join(
            f"  NODE: {n.get('claim', 'N/A')}\n"
            f"    Why: {n.get('explanation', '')[:200]}\n"
            f"    Genuine ambiguity: {n.get('is_genuine_ambiguity', 'unknown')}\n"
            f"    Resolution: {n.get('resolution', 'N/A')[:200]}"
            for n in nodes
        ) or "  (no contradictions found)"

        perspective_names = ", ".join(
            p.get("perspective", f"Source {i}")
            for i, p in enumerate(perspectives)
        )

        diffraction_section = ""
        if diffraction_insights:
            diffraction_section = (
                f"\n\nDIFFRACTED INSIGHTS (extended conclusions):\n"
                f"{diffraction_insights}"
            )

        prompt = (
            f"PATTERN SYNTHESIS\n\n"
            f"Problem: {query}\n\n"
            f"Perspectives analyzed: {perspective_names}\n\n"
            f"CONSTRUCTIVE PEAKS (build on these):\n{peaks_text}\n\n"
            f"DESTRUCTIVE NODES (address these):\n{nodes_text}"
            f"{diffraction_section}\n\n"
            f"Synthesize the final answer:\n"
            f"1. Build the core answer on the constructive peaks (these are "
            f"the strongest, most agreed-upon conclusions)\n"
            f"2. Address each destructive node explicitly -- either resolve "
            f"the contradiction or acknowledge genuine uncertainty\n"
            f"3. Incorporate diffracted insights where appropriate\n"
            f"4. Clearly label areas of genuine uncertainty where "
            f"perspectives irreconcilably disagree\n\n"
            f"Produce a comprehensive, well-structured final answer."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=self.synthesis_temperature,
        )

        # Record synthesis step
        synth_step = self._make_step(
            step_type="pattern_synthesis",
            content=raw,
            metadata={
                "phase": "synthesis",
                "num_perspectives": len(perspectives),
                "num_peaks": len(peaks),
                "num_nodes": len(nodes),
                "diffraction_enabled": self.enable_diffraction,
            },
        )
        trace.add_step(synth_step)

        # Record final output step
        final_step = self._make_step(
            step_type="final_output",
            content=raw,
            metadata={"phase": "final"},
            parent_step_id=synth_step.step_id,
        )
        trace.add_step(final_step)

        return raw

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_perspectives(raw: str) -> list[dict[str, Any]]:
        """Parse perspective data from LLM output.

        Args:
            raw: Raw LLM output.

        Returns:
            List of perspective dicts with ``perspective``, ``analysis``,
            and ``key_claims`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                persp_list = parsed.get("perspectives", [])
                if isinstance(persp_list, list):
                    result: list[dict[str, Any]] = []
                    for p in persp_list:
                        if isinstance(p, dict):
                            claims = p.get("key_claims", [])
                            if not isinstance(claims, list):
                                claims = [str(claims)]
                            result.append({
                                "perspective": str(
                                    p.get("perspective", "unnamed")
                                ),
                                "analysis": str(p.get("analysis", "")),
                                "key_claims": [str(c) for c in claims],
                            })
                    if result:
                        return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: single perspective from the raw text
        return [{
            "perspective": "General analysis",
            "analysis": raw,
            "key_claims": [raw[:200]],
        }]

    @staticmethod
    def _parse_alignments(
        raw: str,
        claim_pairs: list[dict[str, Any]],
    ) -> dict[int, dict[str, float]]:
        """Parse phase alignment results.

        Args:
            raw: Raw LLM output.
            claim_pairs: The claim pairs that were evaluated.

        Returns:
            Dict mapping pair index to dict with ``phase_angle`` and
            ``coherence`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        result: dict[int, dict[str, float]] = {}

        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                alignments = parsed.get("alignments", [])
                if isinstance(alignments, list):
                    for a in alignments:
                        if isinstance(a, dict):
                            # pair_index is 1-based in the prompt
                            idx = int(a.get("pair_index", 0)) - 1
                            if 0 <= idx < len(claim_pairs):
                                result[idx] = {
                                    "phase_angle": max(
                                        0.0,
                                        min(
                                            math.pi,
                                            float(a.get("phase_angle", math.pi / 2)),
                                        ),
                                    ),
                                    "coherence": max(
                                        0.0,
                                        min(
                                            1.0,
                                            float(a.get("coherence", 0.5)),
                                        ),
                                    ),
                                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return result

    @staticmethod
    def _parse_pattern(
        raw: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse the interference pattern analysis.

        Args:
            raw: Raw LLM output.

        Returns:
            Tuple of (constructive peaks list, destructive nodes list).
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        peaks: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []

        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])

                for p in parsed.get("peaks", []):
                    if isinstance(p, dict):
                        peaks.append({
                            "claim": str(p.get("claim", "")),
                            "explanation": str(p.get("explanation", "")),
                            "strength": max(
                                0.0,
                                min(1.0, float(p.get("strength", 0.5))),
                            ),
                        })

                for n in parsed.get("nodes", []):
                    if isinstance(n, dict):
                        nodes.append({
                            "claim": str(n.get("claim", "")),
                            "explanation": str(n.get("explanation", "")),
                            "is_genuine_ambiguity": bool(
                                n.get("is_genuine_ambiguity", True)
                            ),
                            "resolution": str(n.get("resolution", "")),
                        })

            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return peaks, nodes


__all__ = ["WaveInterference"]
