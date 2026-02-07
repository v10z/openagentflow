"""Neuromodulatory Sweep reasoning engine.

Analyzes a problem under eight distinct neurochemical regimes defined by
high/low states of four neuromodulatory systems, then synthesizes across all
regimes to produce a robust, multi-perspective answer.

Neuroscience basis:

The brain's cognitive style is not fixed -- it is dynamically tuned by four
major neuromodulatory systems originating in brainstem and basal forebrain
nuclei.  Each system broadcasts diffusely across the cortex, modulating
neural computation in characteristic ways:

1. **Dopamine** (ventral tegmental area, substantia nigra): Controls
   reward sensitivity, motivation, and the explore/exploit trade-off.
   High dopamine -> optimistic, reward-seeking, exploratory.
   Low dopamine -> conservative, risk-averse, routine-following.

2. **Norepinephrine** (locus coeruleus): Controls arousal, gain, and
   signal-to-noise ratio.  Modeled by the Aston-Jones & Cohen (2005)
   tonic/phasic theory.
   High NE -> focused, high-gain, decisive (phasic mode).
   Low NE -> diffuse attention, broad exploration (tonic mode).

3. **Serotonin** (dorsal raphe nucleus): Controls temporal discounting,
   patience, and the time horizon of planning.  Daw et al. (2002)
   showed serotonin shifts the discount factor.
   High serotonin -> patient, long time horizon, strategic.
   Low serotonin -> impulsive, short time horizon, tactical.

4. **Acetylcholine** (nucleus basalis of Meynert): Controls the balance
   between top-down models and bottom-up sensory data.  Yu & Dayan
   (2005) formalized this as expected vs. unexpected uncertainty.
   High ACh -> data-driven, empirical, distrusts priors.
   Low ACh -> model-driven, theoretical, trusts existing frameworks.

With two levels (high/low) across four modulators, there are 2^4 = 16
possible regimes.  The engine samples the 8 most informative regimes
(vertices of the hypercube that maximize coverage) and uses each to
analyze the problem from a qualitatively different cognitive stance.

Example::

    from openagentflow.reasoning.neuromodulatory_sweep import NeuromodulatorySweep

    engine = NeuromodulatorySweep()
    trace = await engine.reason(
        query="How should our startup allocate its limited runway?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neurochemical regime definitions
# ---------------------------------------------------------------------------

_REGIMES: list[dict[str, Any]] = [
    {
        "name": "explorer",
        "dopamine": "high",
        "norepinephrine": "low",
        "serotonin": "high",
        "acetylcholine": "high",
        "description": (
            "Optimistic, broadly exploring, patient, data-driven. "
            "Seeks novel long-term opportunities grounded in evidence."
        ),
        "system_prompt": (
            "You are in an exploratory, evidence-driven state. You are "
            "optimistic about possibilities, casting a wide net with diffuse "
            "attention, thinking long-term, and grounding all reasoning in "
            "concrete data rather than theory. Prioritize discovering new "
            "opportunities and novel approaches."
        ),
        "temperature": 0.8,
    },
    {
        "name": "executor",
        "dopamine": "high",
        "norepinephrine": "high",
        "serotonin": "low",
        "acetylcholine": "low",
        "description": (
            "Reward-driven, laser-focused, impatient, model-trusting. "
            "Wants immediate high-impact action using existing frameworks."
        ),
        "system_prompt": (
            "You are in a focused, action-oriented state. You are motivated "
            "by immediate rewards, highly focused with sharp attention, "
            "impatient for results, and trusting your existing mental models. "
            "Prioritize speed and decisive action using proven approaches."
        ),
        "temperature": 0.5,
    },
    {
        "name": "strategist",
        "dopamine": "low",
        "norepinephrine": "low",
        "serotonin": "high",
        "acetylcholine": "low",
        "description": (
            "Conservative, diffuse-attention, patient, theory-driven. "
            "Careful long-range planning using established frameworks."
        ),
        "system_prompt": (
            "You are in a strategic, conservative state. You are cautious "
            "about risk, maintaining broad awareness rather than narrow "
            "focus, thinking on long time horizons, and trusting established "
            "theoretical frameworks. Prioritize careful planning and risk "
            "mitigation."
        ),
        "temperature": 0.4,
    },
    {
        "name": "vigilant",
        "dopamine": "low",
        "norepinephrine": "high",
        "serotonin": "low",
        "acetylcholine": "high",
        "description": (
            "Risk-averse, hyper-focused, short-horizon, data-driven. "
            "Defensive threat detection grounded in immediate evidence."
        ),
        "system_prompt": (
            "You are in a vigilant, threat-detecting state. You are "
            "highly sensitive to risks and downsides, intensely focused "
            "on specific details, thinking about immediate consequences, "
            "and demanding empirical evidence for every claim. Prioritize "
            "identifying dangers, weaknesses, and failure modes."
        ),
        "temperature": 0.3,
    },
    {
        "name": "creative",
        "dopamine": "high",
        "norepinephrine": "low",
        "serotonin": "low",
        "acetylcholine": "high",
        "description": (
            "Reward-seeking, diffuse, impulsive, data-driven. "
            "Rapid creative ideation seeking novel empirical patterns."
        ),
        "system_prompt": (
            "You are in a creative, free-associating state. You are "
            "excited by novel ideas and rewards, letting your attention "
            "wander freely, moving quickly without overthinking, and "
            "drawing inspiration from raw data and observations rather "
            "than theory. Prioritize originality and unexpected connections."
        ),
        "temperature": 0.9,
    },
    {
        "name": "analytical",
        "dopamine": "low",
        "norepinephrine": "high",
        "serotonin": "high",
        "acetylcholine": "low",
        "description": (
            "Conservative, focused, patient, model-driven. "
            "Systematic deep analysis using theoretical rigor."
        ),
        "system_prompt": (
            "You are in a systematic, analytical state. You are skeptical "
            "of easy wins, precisely focused on logical structure, willing "
            "to think deeply over long time horizons, and reasoning from "
            "established theoretical principles. Prioritize logical rigor, "
            "internal consistency, and formal analysis."
        ),
        "temperature": 0.3,
    },
    {
        "name": "pragmatist",
        "dopamine": "high",
        "norepinephrine": "high",
        "serotonin": "high",
        "acetylcholine": "high",
        "description": (
            "All-high: motivated, focused, patient, and data-driven. "
            "Peak performance state balancing all modalities."
        ),
        "system_prompt": (
            "You are in a peak-performance state with all cognitive systems "
            "fully engaged. You are motivated and optimistic but also "
            "focused and precise. You think long-term but remain grounded "
            "in evidence. Balance ambition with rigor. Produce the most "
            "complete, well-rounded analysis possible."
        ),
        "temperature": 0.5,
    },
    {
        "name": "contemplative",
        "dopamine": "low",
        "norepinephrine": "low",
        "serotonin": "low",
        "acetylcholine": "low",
        "description": (
            "All-low: unmotivated, diffuse, impatient, model-reliant. "
            "Default/resting state that surfaces implicit assumptions."
        ),
        "system_prompt": (
            "You are in a resting, contemplative state. You lack strong "
            "motivation, your attention is unfocused, you prefer quick "
            "answers over deep analysis, and you rely heavily on gut "
            "instinct and existing beliefs. Give your honest, instinctive "
            "first reaction. Do not overthink. What does your intuition say?"
        ),
        "temperature": 0.7,
    },
]

_SYNTHESIS_SYSTEM = (
    "You are a master integrator synthesizing insights from eight distinct "
    "cognitive regimes. Each regime represents a different neurochemical "
    "state of the brain, producing qualitatively different reasoning. "
    "Your job is to find the signal in the noise: where do multiple "
    "regimes converge? Where do they uniquely contribute? What would be "
    "missed by any single perspective?"
)


class NeuromodulatorySweep(ReasoningEngine):
    """Neuromodulatory sweep across eight neurochemical regimes.

    Analyzes a problem from eight distinct cognitive stances, each defined
    by the high/low state of four neuromodulatory systems (dopamine,
    norepinephrine, serotonin, acetylcholine).  The regimes span the
    neurochemical space to ensure qualitatively diverse perspectives.

    After all eight regime analyses are complete, a synthesis step
    integrates convergent findings, unique contributions, and tensions
    across regimes into a robust final answer.

    Attributes:
        name: ``"NeuromodulatorySweep"``
        description: Short human-readable summary.
        regimes: List of regime configuration dicts.
        synthesis_temperature: LLM temperature for the synthesis step.
        regime_analysis_max_tokens: Advisory maximum for each regime's
            analysis length.
    """

    name: str = "NeuromodulatorySweep"
    description: str = (
        "Analyzes a problem under eight neurochemical regimes (dopamine, "
        "norepinephrine, serotonin, acetylcholine high/low) and synthesizes "
        "across all perspectives."
    )

    def __init__(
        self,
        regimes: list[dict[str, Any]] | None = None,
        synthesis_temperature: float = 0.4,
        regime_analysis_max_tokens: int = 800,
    ) -> None:
        """Initialise the Neuromodulatory Sweep engine.

        Args:
            regimes: Optional custom list of regime dicts.  Each must have
                ``name``, ``description``, ``system_prompt``, and
                ``temperature`` keys.  If ``None`` the default 8 regimes
                are used.
            synthesis_temperature: LLM temperature for the final synthesis
                step.
            regime_analysis_max_tokens: Advisory maximum number of tokens
                for each regime analysis (used as a prompt hint, not
                enforced).
        """
        self.regimes = regimes if regimes is not None else list(_REGIMES)
        self.synthesis_temperature = synthesis_temperature
        self.regime_analysis_max_tokens = max(200, regime_analysis_max_tokens)

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
        """Execute the Neuromodulatory Sweep reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for generating regime analyses.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on the number of regimes to evaluate
                (defaults to all configured regimes).
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing regime analyses and a
            cross-regime synthesis.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- Problem framing
        problem_frame = await self._frame_problem(query, llm_provider, trace)

        # Phase 2 -- Regime analyses
        regimes_to_run = self.regimes[:max_iterations]
        regime_results: list[dict[str, Any]] = []

        for regime in regimes_to_run:
            result = await self._analyze_under_regime(
                query, problem_frame, regime, llm_provider, trace
            )
            regime_results.append(result)

        # Phase 3 -- Cross-regime comparison
        comparison = await self._compare_regimes(
            query, regime_results, llm_provider, trace
        )

        # Phase 4 -- Final synthesis
        final_output = await self._synthesize(
            query, regime_results, comparison, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Problem framing
    # ------------------------------------------------------------------

    async def _frame_problem(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Frame the problem for multi-regime analysis.

        Identifies the key dimensions, stakeholders, constraints, and
        success criteria that each regime should address.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            A structured problem frame string.
        """
        prompt = (
            f"Before analyzing this problem from multiple cognitive angles, "
            f"create a structured problem frame that identifies:\n"
            f"1. The core question or decision\n"
            f"2. Key stakeholders and their interests\n"
            f"3. Known constraints and boundaries\n"
            f"4. Success criteria and how we would evaluate solutions\n"
            f"5. Key uncertainties or unknowns\n\n"
            f"Problem: {query}\n\n"
            f"Provide a concise but thorough problem frame."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are a problem structuring expert. Create a clear, "
                "neutral frame for multi-perspective analysis."
            ),
            temperature=0.3,
        )

        step = self._make_step(
            step_type="problem_frame",
            content=raw,
            metadata={"phase": "framing"},
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Phase 2: Regime analysis
    # ------------------------------------------------------------------

    async def _analyze_under_regime(
        self,
        query: str,
        problem_frame: str,
        regime: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Analyze the problem under a specific neurochemical regime.

        Each regime has its own system prompt and temperature that shifts
        the LLM's cognitive stance.

        Args:
            query: Original user query.
            problem_frame: Structured problem frame from Phase 1.
            regime: Regime configuration dict.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``regime_name``, ``analysis``, ``key_insights``,
            and ``recommended_action`` keys.
        """
        regime_name = regime.get("name", "unnamed")
        regime_desc = regime.get("description", "")
        system_prompt = regime.get("system_prompt", "You are a helpful assistant.")
        temperature = regime.get("temperature", 0.5)

        neuro_state = (
            f"Dopamine: {regime.get('dopamine', 'unknown')} | "
            f"Norepinephrine: {regime.get('norepinephrine', 'unknown')} | "
            f"Serotonin: {regime.get('serotonin', 'unknown')} | "
            f"Acetylcholine: {regime.get('acetylcholine', 'unknown')}"
        )

        prompt = (
            f"Analyze the following problem from your current cognitive state.\n\n"
            f"PROBLEM: {query}\n\n"
            f"PROBLEM FRAME:\n{problem_frame}\n\n"
            f"YOUR COGNITIVE STATE: {regime_desc}\n"
            f"Neurochemical levels: {neuro_state}\n\n"
            f"Provide your analysis including:\n"
            f"1. Your perspective on the core problem\n"
            f"2. Key insights unique to your cognitive stance\n"
            f"3. Risks or opportunities you see that others might miss\n"
            f"4. Your recommended course of action\n\n"
            f"Return a JSON object:\n"
            f'{{"analysis": "your detailed analysis", '
            f'"key_insights": ["insight 1", "insight 2", "insight 3"], '
            f'"recommended_action": "what you recommend", '
            f'"confidence": 0.0-1.0}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=system_prompt,
            temperature=temperature,
        )

        result = self._parse_regime_result(raw, regime_name)

        step = self._make_step(
            step_type="regime_analysis",
            content=result.get("analysis", raw),
            score=result.get("confidence", 0.5),
            metadata={
                "phase": "regime",
                "regime_name": regime_name,
                "regime_description": regime_desc,
                "dopamine": regime.get("dopamine", "unknown"),
                "norepinephrine": regime.get("norepinephrine", "unknown"),
                "serotonin": regime.get("serotonin", "unknown"),
                "acetylcholine": regime.get("acetylcholine", "unknown"),
                "temperature": temperature,
                "confidence": round(result.get("confidence", 0.5), 4),
                "num_insights": len(result.get("key_insights", [])),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 3: Cross-regime comparison
    # ------------------------------------------------------------------

    async def _compare_regimes(
        self,
        query: str,
        regime_results: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Compare and contrast analyses across all regimes.

        Identifies convergent themes, unique contributions, and tensions.

        Args:
            query: Original user query.
            regime_results: List of regime analysis results.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Cross-regime comparison text.
        """
        regime_summaries = "\n\n".join(
            f"--- {r.get('regime_name', 'unnamed').upper()} REGIME ---\n"
            f"Key insights: {', '.join(r.get('key_insights', ['N/A']))}\n"
            f"Recommended action: {r.get('recommended_action', 'N/A')}\n"
            f"Confidence: {r.get('confidence', 0.0):.2f}"
            for r in regime_results
        )

        prompt = (
            f"CROSS-REGIME COMPARISON\n\n"
            f"Problem: {query}\n\n"
            f"Eight distinct cognitive regimes have analyzed this problem. "
            f"Here are their summaries:\n\n{regime_summaries}\n\n"
            f"Identify:\n"
            f"1. CONVERGENT THEMES: What do most regimes agree on? "
            f"(These are high-confidence conclusions.)\n"
            f"2. UNIQUE CONTRIBUTIONS: What does each regime see that "
            f"others miss? (These are perspective-dependent insights.)\n"
            f"3. TENSIONS: Where do regimes contradict each other? "
            f"(These reveal genuine trade-offs or uncertainties.)\n"
            f"4. BLIND SPOTS: What might ALL regimes be missing?\n\n"
            f"Return a JSON object:\n"
            f'{{"convergent_themes": ["theme 1", "theme 2"], '
            f'"unique_contributions": {{"regime_name": "unique insight"}}, '
            f'"tensions": ["tension 1", "tension 2"], '
            f'"blind_spots": ["blind spot 1"]}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are a meta-analytical expert comparing perspectives "
                "across distinct cognitive states. Be precise about where "
                "perspectives converge, diverge, and what each uniquely sees."
            ),
            temperature=0.4,
        )

        comparison = self._parse_comparison(raw)

        step = self._make_step(
            step_type="regime_comparison",
            content=raw,
            metadata={
                "phase": "comparison",
                "num_regimes_compared": len(regime_results),
                "convergent_themes": comparison.get("convergent_themes", []),
                "num_tensions": len(comparison.get("tensions", [])),
                "num_blind_spots": len(comparison.get("blind_spots", [])),
            },
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Phase 4: Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        regime_results: list[dict[str, Any]],
        comparison: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize all regime analyses into a final answer.

        Args:
            query: Original user query.
            regime_results: List of regime analysis results.
            comparison: Cross-regime comparison text.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Final synthesized answer.
        """
        all_insights: list[str] = []
        all_actions: list[str] = []
        for r in regime_results:
            insights = r.get("key_insights", [])
            all_insights.extend(insights)
            action = r.get("recommended_action", "")
            if action:
                all_actions.append(f"[{r.get('regime_name', '?')}] {action}")

        insights_text = "\n".join(f"- {ins}" for ins in all_insights)
        actions_text = "\n".join(f"- {act}" for act in all_actions)

        prompt = (
            f"NEUROMODULATORY SYNTHESIS\n\n"
            f"Problem: {query}\n\n"
            f"After analyzing the problem under eight distinct neurochemical "
            f"regimes, here is the collected intelligence:\n\n"
            f"ALL KEY INSIGHTS:\n{insights_text}\n\n"
            f"REGIME-SPECIFIC RECOMMENDATIONS:\n{actions_text}\n\n"
            f"CROSS-REGIME COMPARISON:\n{comparison}\n\n"
            f"Synthesize a final answer that:\n"
            f"1. Builds on convergent themes (high confidence)\n"
            f"2. Incorporates unique insights from specialist regimes\n"
            f"3. Explicitly addresses tensions and trade-offs\n"
            f"4. Acknowledges blind spots and residual uncertainties\n"
            f"5. Provides a concrete, actionable recommendation\n\n"
            f"Produce a comprehensive, well-structured final answer."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=self.synthesis_temperature,
        )

        step = self._make_step(
            step_type="synthesis",
            content=raw,
            metadata={
                "phase": "synthesis",
                "num_regimes": len(regime_results),
                "total_insights": len(all_insights),
            },
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_regime_result(raw: str, regime_name: str) -> dict[str, Any]:
        """Parse a regime analysis result from LLM output.

        Args:
            raw: Raw LLM output.
            regime_name: Name of the regime.

        Returns:
            Dict with ``regime_name``, ``analysis``, ``key_insights``,
            ``recommended_action``, and ``confidence`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return {
                    "regime_name": regime_name,
                    "analysis": str(parsed.get("analysis", raw)),
                    "key_insights": [
                        str(i)
                        for i in parsed.get("key_insights", [])
                        if isinstance(i, str)
                    ] or [raw[:200]],
                    "recommended_action": str(
                        parsed.get("recommended_action", "")
                    ),
                    "confidence": max(
                        0.0, min(1.0, float(parsed.get("confidence", 0.5)))
                    ),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback
        return {
            "regime_name": regime_name,
            "analysis": raw,
            "key_insights": [raw[:200]],
            "recommended_action": "",
            "confidence": 0.5,
        }

    @staticmethod
    def _parse_comparison(raw: str) -> dict[str, Any]:
        """Parse the cross-regime comparison result.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``convergent_themes``, ``unique_contributions``,
            ``tensions``, and ``blind_spots`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return {
                    "convergent_themes": parsed.get("convergent_themes", []),
                    "unique_contributions": parsed.get(
                        "unique_contributions", {}
                    ),
                    "tensions": parsed.get("tensions", []),
                    "blind_spots": parsed.get("blind_spots", []),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return {
            "convergent_themes": [],
            "unique_contributions": {},
            "tensions": [],
            "blind_spots": [],
        }


__all__ = ["NeuromodulatorySweep"]
