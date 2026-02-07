"""Basal Ganglia Gating reasoning engine.

Models the basal ganglia's Go/No-Go/Hyperdirect pathway architecture for
dopamine-modulated action selection applied to multi-step decision making.

Neuroscience basis:

The basal ganglia are a set of subcortical nuclei (striatum, globus pallidus,
subthalamic nucleus, substantia nigra) that implement action selection through
a competitive gating mechanism.  Three parallel pathways evaluate candidate
actions:

- **Direct (Go) pathway**: Striatal D1 neurons disinhibit the thalamus,
  facilitating the selected action.  This pathway *advocates* for an option.
- **Indirect (No-Go) pathway**: Striatal D2 neurons increase thalamic
  inhibition, suppressing the action.  This pathway *critiques* an option.
- **Hyperdirect pathway**: The subthalamic nucleus sends a fast, global
  inhibitory signal that can veto the entire decision -- an emergency brake
  that fires when the situation is ambiguous or dangerous.

Dopamine from the substantia nigra pars compacta (SNc) modulates the
balance between Go and No-Go pathways:

- High dopamine -> Go pathway dominance -> exploratory, reward-seeking
- Low dopamine -> No-Go pathway dominance -> cautious, loss-averse

The engine decomposes the problem into sequential decision points and at each
point runs parallel Go (advocate) and No-Go (critic) evaluations, with a
Hyperdirect veto check.  A dynamically-adjusted dopamine level controls the
explore/exploit balance across decisions.

Example::

    from openagentflow.reasoning.basal_ganglia_gating import BasalGangliaGating

    engine = BasalGangliaGating(dopamine_level=0.6, num_options=3)
    trace = await engine.reason(
        query="Which database should we migrate to for our growing SaaS?",
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

_DECOMPOSE_SYSTEM = (
    "You are an expert decision analyst. Decompose the given problem into "
    "a sequence of discrete decision points that must be resolved in order. "
    "Each decision point should be a concrete choice with identifiable options."
)

_GO_SYSTEM = (
    "You are the Go pathway -- an enthusiastic advocate. Your role is to find "
    "every reason why this option is excellent: benefits, opportunities, "
    "upside potential, synergies, and strategic advantages. Be specific and "
    "evidence-based, but your disposition is optimistic. Argue FOR this option."
)

_NOGO_SYSTEM = (
    "You are the No-Go pathway -- a rigorous critic. Your role is to find "
    "every reason why this option should be rejected: risks, costs, hidden "
    "downsides, failure modes, opportunity costs, and weaknesses. Be specific "
    "and evidence-based, but your disposition is cautious. Argue AGAINST this option."
)

_HYPERDIRECT_SYSTEM = (
    "You are the Hyperdirect pathway -- an emergency veto mechanism. Your job "
    "is NOT to evaluate individual options but to assess whether the entire "
    "decision framing is safe to proceed with. Look for: catastrophic risks "
    "across ALL options, missing critical information, flawed premises, "
    "ethical red flags, or situations where ANY choice is dangerous. "
    "Only trigger a veto if the situation genuinely warrants halting."
)

_GATE_SYSTEM = (
    "You are the thalamic gate integrating Go (advocacy) and No-Go (criticism) "
    "signals under the current dopamine level. Higher dopamine favors the Go "
    "pathway (risk-tolerant, opportunity-seeking); lower dopamine favors the "
    "No-Go pathway (risk-averse, loss-avoiding). Integrate both signals and "
    "select the best option for this decision point."
)

_DOPAMINE_UPDATE_SYSTEM = (
    "You are the dopaminergic reward prediction system. Based on the outcome "
    "of the most recent decision, update the dopamine level. If the decision "
    "went well (good option selected with high confidence), increase dopamine "
    "slightly (more exploration). If the decision was difficult or risky, "
    "decrease dopamine slightly (more caution)."
)

_SYNTHESIS_SYSTEM = (
    "You are a master synthesizer. Given a sequence of gated decisions, each "
    "informed by Go/No-Go evaluation and dopamine-modulated selection, produce "
    "a coherent final answer that integrates all decisions into a unified "
    "recommendation."
)


class BasalGangliaGating(ReasoningEngine):
    """Basal ganglia Go/No-Go/Hyperdirect gating for sequential decision making.

    The engine decomposes a problem into sequential decision points.  At each
    point it generates candidate options and evaluates them through three
    parallel pathways:

    1. **Go pathway** (D1 / direct): advocates for each option, highlighting
       benefits and opportunities.
    2. **No-Go pathway** (D2 / indirect): critiques each option, highlighting
       risks and costs.
    3. **Hyperdirect pathway** (STN): a fast global veto that can halt the
       entire decision if the framing is unsafe.

    A dopamine level (0.0--1.0) modulates the relative weight of Go vs No-Go
    signals.  High dopamine biases toward exploration and reward-seeking; low
    dopamine biases toward caution.  The dopamine level is updated after each
    decision based on outcome confidence.

    Attributes:
        name: ``"BasalGangliaGating"``
        description: Short human-readable summary.
        dopamine_level: Initial dopamine level (0.0--1.0).
        num_options: Number of candidate options to generate per decision point.
        veto_threshold: Confidence above which the Hyperdirect pathway triggers
            a veto, forcing reframing.
        dopamine_learning_rate: How quickly dopamine adjusts after each decision.
        go_temperature: LLM temperature for Go pathway calls.
        nogo_temperature: LLM temperature for No-Go pathway calls.
    """

    name: str = "BasalGangliaGating"
    description: str = (
        "Dopamine-modulated action selection via Go/No-Go/Hyperdirect "
        "pathways applied to sequential decision points."
    )

    def __init__(
        self,
        dopamine_level: float = 0.5,
        num_options: int = 3,
        veto_threshold: float = 0.7,
        dopamine_learning_rate: float = 0.1,
        go_temperature: float = 0.7,
        nogo_temperature: float = 0.3,
    ) -> None:
        """Initialise the Basal Ganglia Gating engine.

        Args:
            dopamine_level: Starting dopamine level in ``[0.0, 1.0]``.
                High values bias toward Go (exploration); low values bias
                toward No-Go (caution).
            num_options: Number of candidate options to generate and evaluate
                at each decision point.
            veto_threshold: Hyperdirect pathway veto confidence threshold.
                If the veto assessment exceeds this value the decision is
                reframed rather than selected.
            dopamine_learning_rate: Step size for dopamine updates after each
                gated decision.
            go_temperature: LLM temperature for Go pathway calls (higher =
                more creative advocacy).
            nogo_temperature: LLM temperature for No-Go pathway calls (lower =
                more rigorous criticism).
        """
        self.dopamine_level = max(0.0, min(1.0, dopamine_level))
        self.num_options = max(2, num_options)
        self.veto_threshold = max(0.0, min(1.0, veto_threshold))
        self.dopamine_learning_rate = max(0.01, min(0.5, dopamine_learning_rate))
        self.go_temperature = go_temperature
        self.nogo_temperature = nogo_temperature

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
        """Execute the Basal Ganglia Gating reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for generating and evaluating.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on the number of decision points to
                process, regardless of how many were decomposed.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing decomposition, Go/No-Go
            evaluations, veto checks, gating decisions, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        dopamine = self.dopamine_level

        # Phase 1 -- Decompose problem into decision points
        decision_points, options_map = await self._decompose(
            query, llm_provider, trace
        )

        # Cap decision points
        decision_points = decision_points[:max_iterations]

        # Track gated decisions for final synthesis
        gated_decisions: list[dict[str, Any]] = []

        # Phase 2 -- Process each decision point
        for dp_idx, dp in enumerate(decision_points):
            options = options_map.get(dp_idx, [])
            if not options:
                # Generate fallback options
                options = [f"Option {i + 1} for: {dp}" for i in range(self.num_options)]

            # Phase 2a -- Hyperdirect pathway veto check
            veto_triggered, veto_content, veto_score = await self._hyperdirect_check(
                query, dp, options, dp_idx, llm_provider, trace
            )

            if veto_triggered:
                # Reframe the decision point
                dp, options = await self._reframe_decision(
                    query, dp, options, veto_content, dp_idx, llm_provider, trace
                )

            # Phase 2b -- Go pathway evaluation for each option
            go_evaluations: list[dict[str, Any]] = []
            for opt_idx, option in enumerate(options):
                go_result = await self._go_pathway(
                    query, dp, option, opt_idx, dp_idx, dopamine,
                    llm_provider, trace
                )
                go_evaluations.append(go_result)

            # Phase 2c -- No-Go pathway evaluation for each option
            nogo_evaluations: list[dict[str, Any]] = []
            for opt_idx, option in enumerate(options):
                nogo_result = await self._nogo_pathway(
                    query, dp, option, opt_idx, dp_idx, dopamine,
                    llm_provider, trace
                )
                nogo_evaluations.append(nogo_result)

            # Phase 2d -- Thalamic gate: integrate and select
            gate_result = await self._thalamic_gate(
                query, dp, options, go_evaluations, nogo_evaluations,
                dopamine, dp_idx, llm_provider, trace
            )
            gated_decisions.append(gate_result)

            # Phase 2e -- Dopamine update
            dopamine = await self._update_dopamine(
                dopamine, gate_result, dp_idx, llm_provider, trace
            )

        # Phase 3 -- Final synthesis
        final_output = await self._synthesize(
            query, gated_decisions, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Decision decomposition
    # ------------------------------------------------------------------

    async def _decompose(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[list[str], dict[int, list[str]]]:
        """Decompose the problem into sequential decision points with options.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (list of decision point descriptions, dict mapping
            decision index to list of option strings).
        """
        prompt = (
            f"Decompose the following problem into a sequence of 2-5 key "
            f"decision points that must be resolved. For each decision point, "
            f"generate {self.num_options} candidate options.\n\n"
            f"Problem: {query}\n\n"
            f"Return a JSON object with this structure:\n"
            f'{{"decision_points": [\n'
            f'  {{"decision": "description of decision 1", '
            f'"options": ["option A", "option B", "option C"]}},\n'
            f'  {{"decision": "description of decision 2", '
            f'"options": ["option A", "option B", "option C"]}}\n'
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_DECOMPOSE_SYSTEM,
            temperature=0.4,
        )

        decision_points, options_map = self._parse_decomposition(raw)

        step = self._make_step(
            step_type="decomposition",
            content=raw,
            metadata={
                "phase": "decompose",
                "num_decision_points": len(decision_points),
                "num_options_per": self.num_options,
            },
        )
        trace.add_step(step)

        return decision_points, options_map

    # ------------------------------------------------------------------
    # Phase 2a: Hyperdirect pathway
    # ------------------------------------------------------------------

    async def _hyperdirect_check(
        self,
        query: str,
        decision_point: str,
        options: list[str],
        dp_idx: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[bool, str, float]:
        """Run the Hyperdirect pathway veto check on a decision point.

        The Hyperdirect pathway evaluates whether the entire decision framing
        is safe to proceed with, independent of which option is chosen.

        Args:
            query: Original user query.
            decision_point: The current decision point description.
            options: Candidate options for this decision point.
            dp_idx: Index of this decision point.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (veto_triggered, veto_content, veto_confidence).
        """
        options_text = "\n".join(
            f"  {i + 1}. {opt}" for i, opt in enumerate(options)
        )
        prompt = (
            f"HYPERDIRECT VETO ASSESSMENT\n\n"
            f"Original problem: {query}\n"
            f"Decision point: {decision_point}\n"
            f"Options under consideration:\n{options_text}\n\n"
            f"Assess whether it is safe to proceed with this decision. "
            f"Look for:\n"
            f"- Catastrophic risks present in ALL options\n"
            f"- Critical missing information that makes any choice premature\n"
            f"- Flawed premises in the decision framing\n"
            f"- Ethical red flags\n\n"
            f"Return a JSON object:\n"
            f'{{"veto": true/false, "confidence": 0.0-1.0, '
            f'"reasoning": "explanation"}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_HYPERDIRECT_SYSTEM,
            temperature=0.2,
        )

        veto, confidence, reasoning = self._parse_veto(raw)
        veto_triggered = veto and confidence >= self.veto_threshold

        step = self._make_step(
            step_type="hyperdirect_veto",
            content=reasoning,
            score=confidence,
            metadata={
                "phase": "hyperdirect",
                "decision_point_index": dp_idx,
                "veto_triggered": veto_triggered,
                "veto_confidence": round(confidence, 4),
            },
        )
        trace.add_step(step)

        return veto_triggered, reasoning, confidence

    # ------------------------------------------------------------------
    # Reframe after veto
    # ------------------------------------------------------------------

    async def _reframe_decision(
        self,
        query: str,
        decision_point: str,
        options: list[str],
        veto_reasoning: str,
        dp_idx: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[str, list[str]]:
        """Reframe a decision point after a Hyperdirect veto.

        Args:
            query: Original user query.
            decision_point: The vetoed decision point.
            options: The vetoed options.
            veto_reasoning: Explanation from the veto pathway.
            dp_idx: Decision point index.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (reframed decision description, new options list).
        """
        options_text = "\n".join(
            f"  {i + 1}. {opt}" for i, opt in enumerate(options)
        )
        prompt = (
            f"The Hyperdirect pathway has VETOED the following decision:\n\n"
            f"Decision: {decision_point}\n"
            f"Options:\n{options_text}\n\n"
            f"Veto reasoning: {veto_reasoning}\n\n"
            f"Reframe this decision to address the veto concerns. Provide a "
            f"new decision formulation and {self.num_options} new options that "
            f"avoid the identified problems.\n\n"
            f"Return a JSON object:\n"
            f'{{"reframed_decision": "...", '
            f'"new_options": ["option 1", "option 2", "option 3"]}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_DECOMPOSE_SYSTEM,
            temperature=0.5,
        )

        new_decision, new_options = self._parse_reframe(
            raw, decision_point, options
        )

        step = self._make_step(
            step_type="reframe",
            content=raw,
            metadata={
                "phase": "reframe",
                "decision_point_index": dp_idx,
                "original_decision": decision_point,
                "reframed_decision": new_decision,
            },
        )
        trace.add_step(step)

        return new_decision, new_options

    # ------------------------------------------------------------------
    # Phase 2b: Go pathway
    # ------------------------------------------------------------------

    async def _go_pathway(
        self,
        query: str,
        decision_point: str,
        option: str,
        opt_idx: int,
        dp_idx: int,
        dopamine: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Evaluate an option through the Go (advocacy) pathway.

        Args:
            query: Original user query.
            decision_point: The current decision point.
            option: The specific option to advocate for.
            opt_idx: Option index within the decision point.
            dp_idx: Decision point index.
            dopamine: Current dopamine level.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``option``, ``advocacy``, and ``go_score`` keys.
        """
        prompt = (
            f"GO PATHWAY EVALUATION (Dopamine level: {dopamine:.2f})\n\n"
            f"Original problem: {query}\n"
            f"Decision point: {decision_point}\n"
            f"Option to advocate for: {option}\n\n"
            f"Make the strongest possible case FOR this option. Identify:\n"
            f"- Direct benefits and advantages\n"
            f"- Strategic opportunities it enables\n"
            f"- Synergies with other aspects of the problem\n"
            f"- Upside potential and best-case scenarios\n\n"
            f"Return a JSON object:\n"
            f'{{"advocacy": "your argument for this option", '
            f'"go_score": 0.0-1.0}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_GO_SYSTEM,
            temperature=self.go_temperature,
        )

        result = self._parse_pathway_result(raw, option, "go")

        step = self._make_step(
            step_type="go_pathway",
            content=result.get("advocacy", raw),
            score=result.get("go_score", 0.5),
            metadata={
                "phase": "go",
                "decision_point_index": dp_idx,
                "option_index": opt_idx,
                "option": option,
                "dopamine": round(dopamine, 4),
                "go_score": round(result.get("go_score", 0.5), 4),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 2c: No-Go pathway
    # ------------------------------------------------------------------

    async def _nogo_pathway(
        self,
        query: str,
        decision_point: str,
        option: str,
        opt_idx: int,
        dp_idx: int,
        dopamine: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Evaluate an option through the No-Go (critic) pathway.

        Args:
            query: Original user query.
            decision_point: The current decision point.
            option: The specific option to critique.
            opt_idx: Option index within the decision point.
            dp_idx: Decision point index.
            dopamine: Current dopamine level.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``option``, ``critique``, and ``nogo_score`` keys.
        """
        prompt = (
            f"NO-GO PATHWAY EVALUATION (Dopamine level: {dopamine:.2f})\n\n"
            f"Original problem: {query}\n"
            f"Decision point: {decision_point}\n"
            f"Option to critique: {option}\n\n"
            f"Make the strongest possible case AGAINST this option. Identify:\n"
            f"- Risks and potential failure modes\n"
            f"- Hidden costs and downsides\n"
            f"- Opportunity costs (what you give up by choosing this)\n"
            f"- Worst-case scenarios and vulnerabilities\n\n"
            f"Return a JSON object:\n"
            f'{{"critique": "your argument against this option", '
            f'"nogo_score": 0.0-1.0}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_NOGO_SYSTEM,
            temperature=self.nogo_temperature,
        )

        result = self._parse_pathway_result(raw, option, "nogo")

        step = self._make_step(
            step_type="nogo_pathway",
            content=result.get("critique", raw),
            score=result.get("nogo_score", 0.5),
            metadata={
                "phase": "nogo",
                "decision_point_index": dp_idx,
                "option_index": opt_idx,
                "option": option,
                "dopamine": round(dopamine, 4),
                "nogo_score": round(result.get("nogo_score", 0.5), 4),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 2d: Thalamic gate
    # ------------------------------------------------------------------

    async def _thalamic_gate(
        self,
        query: str,
        decision_point: str,
        options: list[str],
        go_evaluations: list[dict[str, Any]],
        nogo_evaluations: list[dict[str, Any]],
        dopamine: float,
        dp_idx: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Integrate Go and No-Go signals to select the best option.

        The thalamic gate weights Go and No-Go scores based on the current
        dopamine level.  High dopamine increases Go weight (risk-tolerant);
        low dopamine increases No-Go weight (risk-averse).

        Gate score formula::

            gate_score_i = dopamine * go_score_i - (1 - dopamine) * nogo_score_i

        Args:
            query: Original user query.
            decision_point: Current decision point description.
            options: List of candidate options.
            go_evaluations: Go pathway results for each option.
            nogo_evaluations: No-Go pathway results for each option.
            dopamine: Current dopamine level.
            dp_idx: Decision point index.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``selected_option``, ``selected_index``, ``gate_scores``,
            ``confidence``, and ``reasoning`` keys.
        """
        # Compute gate scores programmatically
        gate_scores: list[float] = []
        for i in range(len(options)):
            go_score = go_evaluations[i].get("go_score", 0.5) if i < len(go_evaluations) else 0.5
            nogo_score = nogo_evaluations[i].get("nogo_score", 0.5) if i < len(nogo_evaluations) else 0.5
            gate = dopamine * go_score - (1.0 - dopamine) * nogo_score
            gate_scores.append(gate)

        # Build summary for LLM integration
        summary_parts: list[str] = []
        for i, option in enumerate(options):
            go_text = go_evaluations[i].get("advocacy", "N/A") if i < len(go_evaluations) else "N/A"
            nogo_text = nogo_evaluations[i].get("critique", "N/A") if i < len(nogo_evaluations) else "N/A"
            gs = gate_scores[i] if i < len(gate_scores) else 0.0
            summary_parts.append(
                f"Option {i + 1}: {option}\n"
                f"  Go (advocacy, score={go_evaluations[i].get('go_score', 0.5):.2f}): "
                f"{go_text[:200]}\n"
                f"  No-Go (critique, score={nogo_evaluations[i].get('nogo_score', 0.5):.2f}): "
                f"{nogo_text[:200]}\n"
                f"  Gate score: {gs:.4f}"
            )

        options_summary = "\n\n".join(summary_parts)

        prompt = (
            f"THALAMIC GATE DECISION\n\n"
            f"Original problem: {query}\n"
            f"Decision point: {decision_point}\n"
            f"Dopamine level: {dopamine:.2f} "
            f"({'exploratory/reward-seeking' if dopamine > 0.5 else 'cautious/loss-averse'})\n\n"
            f"Pathway evaluations:\n{options_summary}\n\n"
            f"Integrate the Go and No-Go signals under the current dopamine "
            f"modulation and select the best option. Explain your reasoning.\n\n"
            f"Return a JSON object:\n"
            f'{{"selected_index": 0, "selected_option": "...", '
            f'"confidence": 0.0-1.0, '
            f'"reasoning": "why this option wins the gating competition"}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_GATE_SYSTEM,
            temperature=0.3,
        )

        result = self._parse_gate_result(raw, options, gate_scores)

        step = self._make_step(
            step_type="thalamic_gate",
            content=result.get("reasoning", raw),
            score=result.get("confidence", 0.5),
            metadata={
                "phase": "gate",
                "decision_point_index": dp_idx,
                "decision_point": decision_point,
                "dopamine": round(dopamine, 4),
                "gate_scores": [round(g, 4) for g in gate_scores],
                "selected_index": result.get("selected_index", 0),
                "selected_option": result.get("selected_option", ""),
                "confidence": round(result.get("confidence", 0.5), 4),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 2e: Dopamine update
    # ------------------------------------------------------------------

    async def _update_dopamine(
        self,
        current_dopamine: float,
        gate_result: dict[str, Any],
        dp_idx: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> float:
        """Update dopamine level based on decision outcome.

        Uses a simple reward-prediction-error rule: if the gating was
        confident and clear, dopamine increases (reward signal); if the
        decision was ambiguous or forced, dopamine decreases (caution signal).

        Args:
            current_dopamine: Current dopamine level.
            gate_result: Result from the thalamic gate.
            dp_idx: Decision point index.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Updated dopamine level in ``[0.0, 1.0]``.
        """
        confidence = gate_result.get("confidence", 0.5)
        # Reward prediction error: confidence above 0.5 is a positive RPE
        rpe = confidence - 0.5
        new_dopamine = current_dopamine + self.dopamine_learning_rate * rpe
        new_dopamine = max(0.05, min(0.95, new_dopamine))

        step = self._make_step(
            step_type="dopamine_update",
            content=(
                f"Dopamine update: {current_dopamine:.4f} -> {new_dopamine:.4f} "
                f"(RPE={rpe:.4f}, confidence={confidence:.4f})"
            ),
            score=new_dopamine,
            metadata={
                "phase": "dopamine",
                "decision_point_index": dp_idx,
                "old_dopamine": round(current_dopamine, 4),
                "new_dopamine": round(new_dopamine, 4),
                "reward_prediction_error": round(rpe, 4),
                "confidence": round(confidence, 4),
            },
        )
        trace.add_step(step)

        return new_dopamine

    # ------------------------------------------------------------------
    # Phase 3: Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        gated_decisions: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize all gated decisions into a final answer.

        Args:
            query: Original user query.
            gated_decisions: List of gate results from all decision points.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final synthesized answer.
        """
        decisions_text = "\n\n".join(
            f"Decision {i + 1}: Selected '{d.get('selected_option', 'N/A')}' "
            f"(confidence: {d.get('confidence', 0.0):.2f})\n"
            f"Reasoning: {d.get('reasoning', 'N/A')}"
            for i, d in enumerate(gated_decisions)
        )

        prompt = (
            f"FINAL SYNTHESIS\n\n"
            f"Original problem: {query}\n\n"
            f"Gated decisions (each evaluated through Go/No-Go pathways with "
            f"dopamine-modulated selection):\n\n{decisions_text}\n\n"
            f"Synthesize these decisions into a coherent, comprehensive final "
            f"answer. Address how the decisions connect, any tensions between "
            f"them, and provide actionable recommendations."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.4,
        )

        step = self._make_step(
            step_type="synthesis",
            content=raw,
            metadata={"phase": "synthesis"},
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decomposition(
        raw: str,
    ) -> tuple[list[str], dict[int, list[str]]]:
        """Parse the decomposition LLM output into decision points and options.

        Args:
            raw: Raw LLM output (expected JSON).

        Returns:
            Tuple of (decision point strings, map of index to option lists).
        """
        decision_points: list[str] = []
        options_map: dict[int, list[str]] = {}

        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                dps = parsed.get("decision_points", [])
                if isinstance(dps, list):
                    for i, dp in enumerate(dps):
                        if isinstance(dp, dict):
                            decision_points.append(
                                dp.get("decision", f"Decision {i + 1}")
                            )
                            opts = dp.get("options", [])
                            if isinstance(opts, list):
                                options_map[i] = [str(o) for o in opts]
                        elif isinstance(dp, str):
                            decision_points.append(dp)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: treat each line as a decision point
        if not decision_points:
            for line in raw.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line:
                    decision_points.append(line)

        return decision_points, options_map

    @staticmethod
    def _parse_veto(raw: str) -> tuple[bool, float, str]:
        """Parse the Hyperdirect pathway veto response.

        Args:
            raw: Raw LLM output.

        Returns:
            Tuple of (veto_flag, confidence, reasoning).
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                veto = bool(parsed.get("veto", False))
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
                reasoning = str(parsed.get("reasoning", raw))
                return veto, confidence, reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return False, 0.0, raw

    @staticmethod
    def _parse_pathway_result(
        raw: str, option: str, pathway: str
    ) -> dict[str, Any]:
        """Parse Go or No-Go pathway result.

        Args:
            raw: Raw LLM output.
            option: The option being evaluated.
            pathway: ``"go"`` or ``"nogo"``.

        Returns:
            Dict with pathway-specific keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                result: dict[str, Any] = {"option": option}
                if pathway == "go":
                    result["advocacy"] = str(parsed.get("advocacy", raw))
                    result["go_score"] = max(
                        0.0, min(1.0, float(parsed.get("go_score", 0.5)))
                    )
                else:
                    result["critique"] = str(parsed.get("critique", raw))
                    result["nogo_score"] = max(
                        0.0, min(1.0, float(parsed.get("nogo_score", 0.5)))
                    )
                return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback
        result = {"option": option}
        if pathway == "go":
            result["advocacy"] = raw
            result["go_score"] = 0.5
        else:
            result["critique"] = raw
            result["nogo_score"] = 0.5
        return result

    @staticmethod
    def _parse_gate_result(
        raw: str,
        options: list[str],
        gate_scores: list[float],
    ) -> dict[str, Any]:
        """Parse the thalamic gate selection result.

        Args:
            raw: Raw LLM output.
            options: Available options.
            gate_scores: Pre-computed gate scores.

        Returns:
            Dict with ``selected_option``, ``selected_index``, ``confidence``,
            ``reasoning``, and ``gate_scores`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                idx = int(parsed.get("selected_index", 0))
                idx = max(0, min(len(options) - 1, idx))
                return {
                    "selected_index": idx,
                    "selected_option": parsed.get(
                        "selected_option", options[idx] if options else ""
                    ),
                    "confidence": max(
                        0.0, min(1.0, float(parsed.get("confidence", 0.5)))
                    ),
                    "reasoning": str(parsed.get("reasoning", raw)),
                    "gate_scores": gate_scores,
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: pick highest gate score
        if gate_scores:
            best_idx = gate_scores.index(max(gate_scores))
        else:
            best_idx = 0

        return {
            "selected_index": best_idx,
            "selected_option": options[best_idx] if options else "",
            "confidence": 0.5,
            "reasoning": raw,
            "gate_scores": gate_scores,
        }

    @staticmethod
    def _parse_reframe(
        raw: str,
        original_decision: str,
        original_options: list[str],
    ) -> tuple[str, list[str]]:
        """Parse the reframed decision after a veto.

        Args:
            raw: Raw LLM output.
            original_decision: The original (vetoed) decision.
            original_options: The original (vetoed) options.

        Returns:
            Tuple of (reframed decision, new options list).
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                new_decision = str(
                    parsed.get("reframed_decision", original_decision)
                )
                new_options = parsed.get("new_options", original_options)
                if isinstance(new_options, list) and new_options:
                    return new_decision, [str(o) for o in new_options]
                return new_decision, original_options
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return original_decision, original_options


__all__ = ["BasalGangliaGating"]
