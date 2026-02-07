"""Neural Oscillation reasoning engine.

Implements theta-gamma cross-frequency coupling, a fundamental mechanism by
which the brain organizes information processing across temporal scales
(Lisman & Jensen 2013; Canolty & Knight 2010).

Neural oscillations are rhythmic fluctuations in neural excitability that
occur at multiple frequency bands simultaneously:

- **Theta (4-8 Hz)**: Associated with strategic navigation, memory encoding,
  and the hippocampal 'mode' of sequential processing (Buzsaki 2002).  In
  this engine, theta represents high-level strategic reasoning.
- **Alpha (8-13 Hz)**: Associated with attentional gating and inhibition of
  irrelevant information (Jensen & Mazaheri 2010).  Here it represents
  focused attention on the most relevant aspects of the problem.
- **Beta (13-30 Hz)**: Associated with motor planning, maintaining the
  status quo, and predictive coding of expected sequences (Engel & Fries
  2010).  Here it represents structured planning and step-by-step reasoning.
- **Gamma (30-100 Hz)**: Associated with local cortical computation, feature
  binding, and detailed analysis (Fries 2009).  Here it represents
  fine-grained detail work and specific computations.

The critical mechanism is *cross-frequency coupling* (CFC): gamma bursts are
nested within theta cycles, so that high-frequency detail processing is
organized by low-frequency strategic rhythm.  This allows the brain to
simultaneously maintain a strategic context (theta) while performing rapid
local computations (gamma).  When gamma-level processing discovers something
unexpected, it can modulate the theta-level strategy -- a bottom-up signal
that shifts the strategic frame.

Algorithm outline::

    1. THETA  -- Strategic overview: set high-level approach and goals
    2. ALPHA  -- Attentional filter: identify the most relevant aspects
    3. BETA   -- Planning: create a structured step-by-step plan
    4. GAMMA  -- Detail execution: execute each plan step in detail
    5. COUPLING -- Feed gamma discoveries back to theta; update strategy
    6. REPEAT -- Run another oscillation cycle with updated strategy
    7. SYNTHESIS -- Integrate across all frequency bands

Example::

    from openagentflow.reasoning.neural_oscillation import NeuralOscillation

    engine = NeuralOscillation(num_cycles=2, gamma_detail_level=3)
    trace = await engine.reason(
        query="Design a fault-tolerant distributed consensus protocol.",
        llm_provider=my_provider,
    )
    print(trace.final_output)
    print(trace.summary())
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


class NeuralOscillation(ReasoningEngine):
    """Cross-frequency coupling across theta, alpha, beta, and gamma bands.

    The engine processes the problem at four frequency bands -- each
    corresponding to a different level of abstraction and temporal
    resolution.  Theta (strategic) sets the overall approach; alpha
    (attentional) filters for relevance; beta (planning) creates a
    structured execution plan; gamma (detail) executes fine-grained
    analysis.

    The key mechanism is *cross-frequency coupling*: gamma-level
    discoveries feed back to theta-level strategy, potentially shifting
    the entire approach.  This bidirectional influence between fast
    local processing and slow global context mirrors the nested-oscillation
    architecture of the mammalian cortex.

    Parameters:
        num_cycles: Number of full oscillation cycles (theta -> alpha ->
            beta -> gamma -> coupling).  Each cycle refines the answer.
        gamma_detail_level: Number of gamma-band detail sub-steps per
            beta-level plan step.  Higher values produce more detailed
            analysis but cost more LLM calls.
        coupling_strength: How strongly gamma discoveries can modify
            theta strategy (0.0 = no coupling, 1.0 = full override).
        theta_temperature: Temperature for strategic-level thinking.
        gamma_temperature: Temperature for detail-level analysis.
        alpha_top_k: Number of aspects the alpha filter selects from the
            problem for focused attention.

    Example::

        engine = NeuralOscillation(num_cycles=3, gamma_detail_level=2)
        trace = await engine.reason(
            query="What are the trade-offs of event sourcing vs. state-based "
                  "persistence?",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "neural_oscillation"
    description: str = (
        "Theta-gamma cross-frequency coupling: strategic reasoning (theta) "
        "organizes detail processing (gamma), with bottom-up feedback that "
        "can shift strategy when unexpected findings emerge."
    )

    def __init__(
        self,
        num_cycles: int = 2,
        gamma_detail_level: int = 3,
        coupling_strength: float = 0.6,
        theta_temperature: float = 0.7,
        gamma_temperature: float = 0.3,
        alpha_top_k: int = 3,
    ) -> None:
        self.num_cycles = max(1, num_cycles)
        self.gamma_detail_level = max(1, gamma_detail_level)
        self.coupling_strength = max(0.0, min(1.0, coupling_strength))
        self.theta_temperature = max(0.0, min(1.0, theta_temperature))
        self.gamma_temperature = max(0.0, min(1.0, gamma_temperature))
        self.alpha_top_k = max(1, alpha_top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: Any | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the full neural oscillation reasoning cycle.

        Args:
            query: The problem statement to solve.
            llm_provider: A ``BaseLLMProvider`` instance for all LLM calls.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Soft cap -- ``num_cycles`` is capped at
                ``min(num_cycles, max_iterations)``.
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` integrates
            insights from all frequency bands and oscillation cycles.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_cycles = min(self.num_cycles, max_iterations)

        # State that persists across cycles
        theta_strategy = ""
        accumulated_gamma_discoveries: list[str] = []
        cycle_outputs: list[dict[str, Any]] = []

        for cycle_idx in range(effective_cycles):
            cycle_label = f"cycle-{cycle_idx + 1}/{effective_cycles}"

            # Record cycle start
            cycle_step = self._make_step(
                step_type="oscillation_cycle",
                content=f"Starting {cycle_label}",
                metadata={
                    "phase": "cycle_start",
                    "cycle": cycle_idx + 1,
                    "total_cycles": effective_cycles,
                },
            )
            trace.add_step(cycle_step)
            cycle_parent_id = cycle_step.step_id

            # --- THETA BAND: Strategic overview ---
            theta_strategy, theta_step_id = await self._theta_band(
                query, theta_strategy, accumulated_gamma_discoveries,
                cycle_idx, llm_provider, trace, cycle_parent_id,
            )

            # --- ALPHA BAND: Attentional filter ---
            focus_aspects, alpha_step_id = await self._alpha_band(
                query, theta_strategy, llm_provider, trace, theta_step_id,
            )

            # --- BETA BAND: Structured planning ---
            plan_steps, beta_step_id = await self._beta_band(
                query, theta_strategy, focus_aspects,
                llm_provider, trace, alpha_step_id,
            )

            # --- GAMMA BAND: Detail execution ---
            gamma_results, gamma_step_id = await self._gamma_band(
                query, plan_steps, llm_provider, trace, beta_step_id,
            )

            # --- COUPLING: Feed gamma back to theta ---
            discoveries, coupling_step_id = await self._cross_frequency_coupling(
                query, theta_strategy, gamma_results,
                llm_provider, trace, gamma_step_id,
            )
            accumulated_gamma_discoveries.extend(discoveries)

            cycle_outputs.append({
                "cycle": cycle_idx + 1,
                "theta_strategy": theta_strategy,
                "focus_aspects": focus_aspects,
                "plan_steps": plan_steps,
                "gamma_results": gamma_results,
                "discoveries": discoveries,
            })

        # --- SYNTHESIS: Integrate across all bands and cycles ---
        final_output = await self._synthesise(
            query, cycle_outputs, llm_provider, trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # THETA BAND -- Strategic overview
    # ------------------------------------------------------------------

    async def _theta_band(
        self,
        query: str,
        previous_strategy: str,
        gamma_discoveries: list[str],
        cycle_idx: int,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[str, str]:
        """Theta-band processing: set or update the strategic approach.

        On the first cycle, generates a fresh strategic overview.  On
        subsequent cycles, integrates gamma-level discoveries to refine
        the strategy -- this is the upward arm of cross-frequency coupling.

        Returns:
            Tuple of (strategy_text, step_id).
        """
        if cycle_idx == 0:
            system_prompt = (
                "You are a strategic reasoner operating at the THETA frequency "
                "band -- the slowest, most abstract level of cognitive processing. "
                "Think about the BIG PICTURE: overall approach, key goals, "
                "fundamental tradeoffs, and guiding principles. Do NOT get into "
                "details -- that is for faster frequency bands."
            )
            user_prompt = (
                f"Set the strategic direction for solving this problem.\n\n"
                f"PROBLEM:\n{query}\n\n"
                f"Provide:\n"
                f"1. The overarching goal\n"
                f"2. The fundamental approach or methodology\n"
                f"3. Key strategic tradeoffs to manage\n"
                f"4. Success criteria for a good answer\n\n"
                f"Stay at the strategic level -- no implementation details."
            )
        else:
            discovery_text = "\n".join(
                f"- {d}" for d in gamma_discoveries[-5:]  # Last 5 discoveries
            ) if gamma_discoveries else "None yet."

            system_prompt = (
                "You are updating the THETA-band strategy based on new "
                "discoveries from gamma-band detail processing. This is "
                "cross-frequency coupling: fast local processing has found "
                "something that should shift slow global strategy. Adjust "
                "the strategic direction accordingly."
            )
            user_prompt = (
                f"PROBLEM:\n{query}\n\n"
                f"CURRENT STRATEGY:\n{previous_strategy}\n\n"
                f"GAMMA-BAND DISCOVERIES (from detail-level analysis):\n"
                f"{discovery_text}\n\n"
                f"Coupling strength: {self.coupling_strength:.1f}\n\n"
                f"Update the strategy. If gamma discoveries require a "
                f"strategic shift, make it proportional to the coupling "
                f"strength ({self.coupling_strength:.0%} influence). Preserve "
                f"parts of the strategy that are still valid."
            )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=self.theta_temperature,
        )

        step = self._make_step(
            step_type="theta_band",
            content=raw,
            score=0.5,
            metadata={
                "phase": "theta",
                "cycle": cycle_idx + 1,
                "is_update": cycle_idx > 0,
                "discovery_count": len(gamma_discoveries),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)
        logger.debug("NeuralOscillation: theta band cycle %d complete", cycle_idx + 1)
        return raw, step.step_id

    # ------------------------------------------------------------------
    # ALPHA BAND -- Attentional filter
    # ------------------------------------------------------------------

    async def _alpha_band(
        self,
        query: str,
        theta_strategy: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[list[str], str]:
        """Alpha-band processing: attentional gating and relevance filtering.

        Given the theta-level strategy, identifies the top-K most relevant
        aspects of the problem to focus on, suppressing irrelevant details.
        This mirrors alpha oscillations' role in inhibiting task-irrelevant
        cortical areas (Jensen & Mazaheri 2010).

        Returns:
            Tuple of (list_of_focus_aspects, step_id).
        """
        system_prompt = (
            "You are an ALPHA-band attentional filter. Your job is to gate "
            "information: given the strategic direction, identify the most "
            "RELEVANT aspects of the problem and suppress the irrelevant "
            "ones. Focus is power -- narrow the beam."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"STRATEGIC DIRECTION (from theta band):\n{theta_strategy}\n\n"
            f"Identify exactly {self.alpha_top_k} aspects of this problem "
            f"that deserve FOCUSED ATTENTION given the strategy above. "
            f"These should be the highest-leverage areas where detailed "
            f"analysis will most improve the answer.\n\n"
            f"Return a JSON array of {self.alpha_top_k} strings, each "
            f"describing one focus area.\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        aspects = self._parse_json_string_list(raw, self.alpha_top_k)

        step = self._make_step(
            step_type="alpha_band",
            content="Attentional focus: " + "; ".join(aspects),
            score=0.5,
            metadata={
                "phase": "alpha",
                "focus_aspects": aspects,
                "top_k": self.alpha_top_k,
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)
        logger.debug("NeuralOscillation: alpha band selected %d aspects", len(aspects))
        return aspects, step.step_id

    # ------------------------------------------------------------------
    # BETA BAND -- Structured planning
    # ------------------------------------------------------------------

    async def _beta_band(
        self,
        query: str,
        theta_strategy: str,
        focus_aspects: list[str],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[list[str], str]:
        """Beta-band processing: create a structured execution plan.

        Translates the strategic direction (theta) and attentional focus
        (alpha) into a concrete, ordered plan of analysis steps.  This
        mirrors beta oscillations' role in maintaining and sequencing
        planned actions (Engel & Fries 2010).

        Returns:
            Tuple of (list_of_plan_steps, step_id).
        """
        focus_text = "\n".join(f"- {a}" for a in focus_aspects)

        system_prompt = (
            "You are a BETA-band planner. Given a strategy and focused "
            "attention areas, create a structured, sequential plan for "
            "detailed analysis. Each plan step should be concrete and "
            "actionable. Order them logically -- dependencies first."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"STRATEGY (theta):\n{theta_strategy}\n\n"
            f"FOCUS AREAS (alpha):\n{focus_text}\n\n"
            f"Create a structured plan with {self.gamma_detail_level} "
            f"concrete analysis steps. Each step should address one or "
            f"more focus areas and advance toward the strategic goals.\n\n"
            f"Return a JSON array of {self.gamma_detail_level} strings, "
            f"each describing one analysis step.\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.4,
        )

        plan_steps = self._parse_json_string_list(raw, self.gamma_detail_level)

        step = self._make_step(
            step_type="beta_band",
            content="Plan: " + " -> ".join(
                f"[{i + 1}] {s[:60]}" for i, s in enumerate(plan_steps)
            ),
            score=0.5,
            metadata={
                "phase": "beta",
                "plan_steps": plan_steps,
                "step_count": len(plan_steps),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)
        logger.debug("NeuralOscillation: beta band planned %d steps", len(plan_steps))
        return plan_steps, step.step_id

    # ------------------------------------------------------------------
    # GAMMA BAND -- Detail execution
    # ------------------------------------------------------------------

    async def _gamma_band(
        self,
        query: str,
        plan_steps: list[str],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Gamma-band processing: execute each plan step in fine detail.

        This is the fastest, most local level of processing -- analogous
        to gamma oscillations (30-100 Hz) that support feature binding
        and detailed computation within a cortical column.

        For each beta-level plan step, the LLM performs deep analysis and
        flags any unexpected findings that should be coupled back to the
        theta-level strategy.

        Returns:
            Tuple of (list_of_result_dicts, parent_step_id_for_coupling).
        """
        system_prompt = (
            "You are a GAMMA-band detail processor -- the fastest, most "
            "fine-grained level of analysis. Execute the assigned analysis "
            "step thoroughly. If you discover anything UNEXPECTED that "
            "challenges the overall strategy, flag it explicitly as "
            "'[DISCOVERY]: ...' on its own line."
        )

        gamma_parent = self._make_step(
            step_type="gamma_batch",
            content=f"Executing {len(plan_steps)} gamma-band detail steps",
            metadata={
                "phase": "gamma",
                "step_count": len(plan_steps),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(gamma_parent)

        results: list[dict[str, Any]] = []

        for idx, plan_step in enumerate(plan_steps):
            user_prompt = (
                f"PROBLEM: {query}\n\n"
                f"ANALYSIS TASK (step {idx + 1}/{len(plan_steps)}):\n"
                f"{plan_step}\n\n"
                f"Perform this analysis in detail. Be thorough and specific. "
                f"If you find anything UNEXPECTED that might change the "
                f"overall strategic approach, prefix it with '[DISCOVERY]: '."
            )

            raw = await self._call_llm(
                provider,
                [{"role": "user", "content": user_prompt}],
                trace,
                system=system_prompt,
                temperature=self.gamma_temperature,
            )

            # Extract any flagged discoveries
            discoveries = self._extract_discoveries(raw)

            result_entry = {
                "step_index": idx,
                "plan_step": plan_step,
                "analysis": raw,
                "discoveries": discoveries,
            }
            results.append(result_entry)

            gamma_step = self._make_step(
                step_type="gamma_detail",
                content=raw[:500] + "..." if len(raw) > 500 else raw,
                score=0.5 + 0.1 * len(discoveries),
                metadata={
                    "phase": "gamma",
                    "step_index": idx,
                    "plan_step": plan_step,
                    "discovery_count": len(discoveries),
                    "discoveries": discoveries,
                },
                parent_step_id=gamma_parent.step_id,
            )
            trace.add_step(gamma_step)

        logger.debug(
            "NeuralOscillation: gamma band completed %d steps, %d total discoveries",
            len(results),
            sum(len(r["discoveries"]) for r in results),
        )
        return results, gamma_parent.step_id

    # ------------------------------------------------------------------
    # COUPLING -- Feed gamma back to theta
    # ------------------------------------------------------------------

    async def _cross_frequency_coupling(
        self,
        query: str,
        theta_strategy: str,
        gamma_results: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[list[str], str]:
        """Cross-frequency coupling: gamma discoveries modulate theta strategy.

        Collects all discoveries from gamma-band processing and evaluates
        which ones are significant enough to warrant a strategic update.
        This is the bottom-up arm of cross-frequency coupling.

        Returns:
            Tuple of (list_of_significant_discoveries, step_id).
        """
        all_discoveries: list[str] = []
        for result in gamma_results:
            all_discoveries.extend(result.get("discoveries", []))

        if not all_discoveries:
            # No discoveries -- coupling is trivial
            step = self._make_step(
                step_type="coupling",
                content="No significant gamma-band discoveries to couple to theta.",
                score=0.0,
                metadata={
                    "phase": "coupling",
                    "discovery_count": 0,
                    "strategic_shift": False,
                },
                parent_step_id=parent_step_id,
            )
            trace.add_step(step)
            return [], step.step_id

        discovery_text = "\n".join(f"- {d}" for d in all_discoveries)

        system_prompt = (
            "You are performing cross-frequency coupling analysis. "
            "Gamma-band detail processing has flagged unexpected discoveries. "
            "Evaluate which ones are significant enough to warrant a shift "
            "in the theta-band (strategic) approach. Return only the "
            "discoveries that truly challenge or enrich the overall strategy."
        )
        user_prompt = (
            f"PROBLEM: {query}\n\n"
            f"CURRENT THETA STRATEGY:\n{theta_strategy}\n\n"
            f"GAMMA-BAND DISCOVERIES:\n{discovery_text}\n\n"
            f"Coupling strength: {self.coupling_strength:.1f}\n\n"
            f"Which of these discoveries are SIGNIFICANT enough to modify "
            f"the strategic approach? For each significant discovery, "
            f"explain briefly why it matters strategically.\n\n"
            f"Return a JSON array of strings -- only the significant "
            f"discoveries (rephrased as strategic implications).\n"
            f"If none are significant, return an empty array [].\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        significant = self._parse_json_string_list_flexible(raw)

        step = self._make_step(
            step_type="coupling",
            content=(
                f"Cross-frequency coupling: {len(significant)} of "
                f"{len(all_discoveries)} gamma discoveries are strategically "
                f"significant."
            ),
            score=len(significant) / max(len(all_discoveries), 1),
            metadata={
                "phase": "coupling",
                "total_discoveries": len(all_discoveries),
                "significant_discoveries": len(significant),
                "strategic_shift": len(significant) > 0,
                "coupling_strength": self.coupling_strength,
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)

        logger.debug(
            "NeuralOscillation: coupling found %d/%d significant discoveries",
            len(significant),
            len(all_discoveries),
        )
        return significant, step.step_id

    # ------------------------------------------------------------------
    # SYNTHESIS -- Final integration
    # ------------------------------------------------------------------

    async def _synthesise(
        self,
        query: str,
        cycle_outputs: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesise the final answer from all oscillation cycles.

        Integrates strategic-level (theta), focused (alpha), planned
        (beta), and detailed (gamma) insights across all cycles into a
        unified answer.

        Returns:
            The final synthesised answer text.
        """
        # Build a summary of each cycle
        cycle_summaries = []
        for co in cycle_outputs:
            gamma_summary = "\n".join(
                f"  - {r['analysis'][:200]}..."
                if len(r.get("analysis", "")) > 200
                else f"  - {r.get('analysis', 'N/A')}"
                for r in co.get("gamma_results", [])
            )
            discovery_summary = "\n".join(
                f"  * {d}" for d in co.get("discoveries", [])
            )
            cycle_summaries.append(
                f"CYCLE {co['cycle']}:\n"
                f"  Strategy: {co.get('theta_strategy', 'N/A')[:300]}\n"
                f"  Focus: {', '.join(co.get('focus_aspects', []))}\n"
                f"  Detailed analysis:\n{gamma_summary}\n"
                f"  Discoveries:\n{discovery_summary or '  (none)'}"
            )

        all_cycles_text = "\n\n".join(cycle_summaries)

        system_prompt = (
            "You are performing a FINAL SYNTHESIS across multiple oscillation "
            "cycles of neural processing. Each cycle operated at four "
            "frequency bands: theta (strategy), alpha (attention), beta "
            "(planning), and gamma (detail). Cross-frequency coupling allowed "
            "detail-level discoveries to refine strategy across cycles. "
            "Integrate all insights into a comprehensive final answer."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"OSCILLATION CYCLE RESULTS:\n{all_cycles_text}\n\n"
            f"Produce the final, integrated answer that:\n"
            f"1. Reflects the evolved strategic understanding (theta)\n"
            f"2. Focuses on the most relevant aspects (alpha)\n"
            f"3. Follows a clear structure (beta)\n"
            f"4. Includes critical details and specifics (gamma)\n"
            f"5. Incorporates all cross-frequency discoveries\n\n"
            f"Provide a complete, self-contained answer."
        )

        final = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.4,
        )

        step = self._make_step(
            step_type="synthesis",
            content=final,
            score=1.0,
            metadata={
                "phase": "synthesis",
                "total_cycles": len(cycle_outputs),
                "total_discoveries": sum(
                    len(co.get("discoveries", [])) for co in cycle_outputs
                ),
            },
        )
        trace.add_step(step)
        logger.debug("NeuralOscillation: synthesis complete")
        return final

    # ------------------------------------------------------------------
    # Parsing and utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_discoveries(text: str) -> list[str]:
        """Extract lines flagged as discoveries from gamma-band output.

        Looks for lines prefixed with ``[DISCOVERY]:`` or variations.

        Returns:
            List of discovery strings (without the prefix).
        """
        discoveries: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # Match [DISCOVERY]: ... or DISCOVERY: ... or **DISCOVERY**: ...
            match = re.match(
                r"(?:\*\*)?(?:\[)?DISCOVERY(?:\])?(?:\*\*)?[:\s]+(.+)",
                stripped,
                re.IGNORECASE,
            )
            if match:
                discoveries.append(match.group(1).strip())
        return discoveries

    @staticmethod
    def _parse_json_string_list(raw: str, expected: int) -> list[str]:
        """Parse a JSON array of strings, padding to ``expected`` length.

        Falls back to line splitting if JSON parsing fails.

        Returns:
            List of strings of length ``expected``.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, list):
                    result = [str(item) for item in parsed]
                    while len(result) < expected:
                        result.append(f"Additional aspect #{len(result) + 1}")
                    return result[:expected]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: line splitting
        lines = [
            ln.strip().lstrip("0123456789.-) ")
            for ln in cleaned.splitlines()
            if ln.strip()
        ]
        lines = [ln for ln in lines if ln]
        while len(lines) < expected:
            lines.append(f"Additional aspect #{len(lines) + 1}")
        return lines[:expected]

    @staticmethod
    def _parse_json_string_list_flexible(raw: str) -> list[str]:
        """Parse a JSON array of strings with no expected count.

        Returns whatever strings are found, or an empty list.

        Returns:
            List of strings (possibly empty).
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except (json.JSONDecodeError, ValueError):
                pass

        return []


__all__ = ["NeuralOscillation"]
