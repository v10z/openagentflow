"""Dream-Wake Cycle reasoning engine for OpenAgentFlow.

Implements a creativity-oriented reasoning pattern that oscillates between
unconstrained divergent thinking (DREAM phase) and rigorous convergent
validation (WAKE phase), with an integrative LUCID phase that fuses the
best elements into a coherent solution.

The metaphor: just as the human mind benefits from the unconstrained
associative leaps of dreaming followed by the grounded reality-testing of
waking consciousness, this engine deliberately alternates between modes to
produce solutions that are both creative and feasible.

Cycle structure::

    DREAM  -->  WAKE  -->  LUCID  -->  (quality check)
      ^                                     |
      +--- feed lucid insights back --------+
           (if quality < threshold)

Example::

    from openagentflow.reasoning import DreamWakeCycle

    engine = DreamWakeCycle(num_dreams=5, max_cycles=3, quality_threshold=0.8)
    trace = await engine.reason(
        query="Design a novel user interface for blind programmers.",
        llm_provider=my_provider,
        tools=[web_search_tool, code_tool],
    )
    print(trace.final_output)

    # Inspect dream ideas
    for step in trace.get_steps_by_type("dream"):
        print(f"Dream (score={step.score:.2f}): {step.content[:100]}...")

Trace structure (DAG)::

    query
      +-- cycle_0
      |     +-- dream_0_idea_0
      |     +-- dream_0_idea_1
      |     +-- ...
      |     +-- wake_0_eval_0  (child of dream_0_idea_0)
      |     +-- wake_0_eval_1  (child of dream_0_idea_1)
      |     +-- ...
      |     +-- lucid_0  (child of cycle_0)
      |     +-- quality_check_0
      +-- cycle_1  (if quality < threshold)
      |     +-- ...
      +-- final_output
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DREAM_SYSTEM = (
    "You are in DREAM mode. Your mind is completely unconstrained. "
    "Think wildly, creatively, across domains. Draw on metaphors from nature, "
    "art, science, history, and fiction. Ask 'What if?' without worrying "
    "about feasibility. Let ideas collide. Combine concepts that have never "
    "been combined. Be surprising. Be bold. Be weird.\n\n"
    "Generate ideas as a numbered list. Each idea should be a distinct, "
    "self-contained approach -- not a variation of the same theme. Push for "
    "genuine diversity of thought."
)

_WAKE_SYSTEM = (
    "You are in WAKE mode. You are a rigorous, clear-eyed analyst. "
    "Evaluate the following idea with intellectual honesty.\n\n"
    "Assess each of these dimensions on a scale from 0.0 to 1.0:\n"
    "- FEASIBILITY: Can this actually be built/implemented with current "
    "technology and reasonable resources?\n"
    "- NOVELTY: How genuinely new and different is this? (0 = completely "
    "standard, 1 = never been done)\n"
    "- COMPLETENESS: How well does this address the full scope of the "
    "original question?\n"
    "- RISK: What could go wrong? (higher = more risk, so LOWER is better "
    "for the overall score)\n\n"
    "Respond in EXACTLY this format (with actual decimal numbers):\n"
    "FEASIBILITY: <number>\n"
    "NOVELTY: <number>\n"
    "COMPLETENESS: <number>\n"
    "RISK: <number>\n"
    "ANALYSIS: <your detailed reasoning>\n\n"
    "Be honest. A wild idea that cannot work should score low on feasibility "
    "even if it is novel."
)

_LUCID_SYSTEM = (
    "You are in LUCID mode -- the bridge between dreaming and waking. "
    "You have access to the most promising ideas from the dream phase, each "
    "with detailed feasibility analysis. Your task:\n\n"
    "1. Identify the strongest elements from each top idea.\n"
    "2. Combine them into a single, coherent, actionable solution.\n"
    "3. Preserve the creative spark -- do not sand down the interesting "
    "edges just for safety.\n"
    "4. But ground everything in reality -- every claim should be plausible.\n"
    "5. Fill in gaps: where the dreams were vague, be specific.\n\n"
    "Produce a well-structured, detailed solution that someone could "
    "actually act on."
)

_QUALITY_SYSTEM = (
    "You are a quality assessor. Evaluate the following solution on a scale "
    "from 0.0 to 1.0 across these dimensions:\n"
    "- CREATIVITY: How creative and novel is the solution?\n"
    "- PRACTICALITY: How actionable and implementable is it?\n"
    "- DEPTH: How thorough and detailed is the analysis?\n"
    "- COHERENCE: How well do the parts fit together?\n\n"
    "Respond in EXACTLY this format:\n"
    "CREATIVITY: <number>\n"
    "PRACTICALITY: <number>\n"
    "DEPTH: <number>\n"
    "COHERENCE: <number>\n"
    "OVERALL: <number>\n"
    "ASSESSMENT: <brief explanation>\n\n"
    "Be calibrated: 0.8+ should be genuinely excellent work."
)


# ---------------------------------------------------------------------------
# Scored dream helper
# ---------------------------------------------------------------------------


@dataclass
class _ScoredDream:
    """Internal: a dream idea paired with its wake-phase evaluation."""

    idea: str = ""
    feasibility: float = 0.0
    novelty: float = 0.0
    completeness: float = 0.0
    risk: float = 0.0
    analysis: str = ""
    composite_score: float = 0.0

    def compute_composite(self) -> float:
        """Compute composite score: novelty * feasibility * completeness * (1 - risk).

        The composite rewards ideas that are simultaneously novel, feasible,
        and complete, while penalising high-risk ideas.  The multiplicative
        form means a zero in *any* dimension collapses the score, which is
        the desired behaviour -- we want ideas that are strong across all
        axes, not just one.

        Returns:
            The composite score (0.0 -- 1.0).
        """
        risk_factor = max(0.0, 1.0 - self.risk)
        self.composite_score = (
            self.novelty * self.feasibility * self.completeness * risk_factor
        )
        return self.composite_score


# ---------------------------------------------------------------------------
# DreamWakeCycle
# ---------------------------------------------------------------------------


class DreamWakeCycle(ReasoningEngine):
    """Creativity through oscillation between divergent and convergent thinking.

    Each cycle consists of three phases:

    **DREAM** (high temperature)
        Unconstrained ideation.  The LLM is prompted to think wildly,
        drawing on cross-domain analogies, metaphors, and "What if?"
        scenarios.  It generates ``num_dreams`` distinct ideas.

    **WAKE** (low temperature, tools enabled)
        Rigorous validation.  Each dream idea is independently evaluated
        for feasibility, novelty, completeness, and risk.  Tools (if
        provided) can be invoked to fact-check claims.  Each idea receives
        a composite score = novelty * feasibility * completeness * (1-risk).

    **LUCID** (synthesis)
        The top-scoring dreams are combined into a single coherent
        solution that preserves creative spark while being grounded in
        reality.

    After the LUCID phase, a quality check determines whether the output
    meets the ``quality_threshold``.  If not, the lucid output is fed back
    as context for the next DREAM cycle, enriching subsequent ideation with
    the insights gained so far.

    Args:
        num_dreams: Number of dream ideas to generate per cycle (default 5).
        max_cycles: Maximum number of dream-wake-lucid cycles (default 3).
        dream_temperature: Temperature for dream-phase generation
            (default 1.0 -- high creativity).
        wake_temperature: Temperature for wake-phase evaluation
            (default 0.2 -- analytical precision).
        quality_threshold: Overall quality score (0.0 -- 1.0) above which
            the engine stops early (default 0.8).
        top_k_dreams: Number of top dreams to pass into the lucid phase
            (default 3).

    Attributes:
        name: ``"dream_wake_cycle"``
        description: Short description of the engine.

    Example::

        engine = DreamWakeCycle(
            num_dreams=7,
            max_cycles=4,
            dream_temperature=1.2,
            wake_temperature=0.1,
            quality_threshold=0.85,
        )
        trace = await engine.reason(
            query="Invent a new programming paradigm.",
            llm_provider=provider,
        )
        print(trace.final_output)

        # Inspect dreams and their scores
        for step in trace.get_steps_by_type("dream"):
            print(f"Dream: {step.content[:80]}...")
        for step in trace.get_steps_by_type("evaluation"):
            print(f"Score: {step.score:.3f}  {step.content[:80]}...")

        # Export trace as graph
        dag = trace.to_dag()
    """

    name: str = "dream_wake_cycle"
    description: str = (
        "Oscillation between unconstrained divergent ideation (dream) and "
        "rigorous convergent validation (wake), fused in a lucid synthesis."
    )

    def __init__(
        self,
        num_dreams: int = 5,
        max_cycles: int = 3,
        dream_temperature: float = 1.0,
        wake_temperature: float = 0.2,
        quality_threshold: float = 0.8,
        top_k_dreams: int = 3,
    ) -> None:
        self.num_dreams = num_dreams
        self.max_cycles = max_cycles
        self.dream_temperature = dream_temperature
        self.wake_temperature = wake_temperature
        self.quality_threshold = quality_threshold
        self.top_k_dreams = top_k_dreams

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute dream-wake cycles and return a full reasoning trace.

        Args:
            query: The user question or creative challenge.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Optional tools available during the WAKE phase for
                reality-checking dream ideas.  Passed through to
                ``_call_llm`` during evaluation steps.
            max_iterations: Ignored -- ``max_cycles`` controls iteration.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with dream / evaluation / lucid /
            quality-check steps and the final output.
        """
        start = time.perf_counter()
        trace = ReasoningTrace(strategy_name=self.name)

        # Root step
        query_step = ReasoningStep(
            step_type="query",
            content=query,
            metadata={"role": "user_query"},
        )
        trace.add_step(query_step)

        accumulated_insights: str = ""
        best_lucid_output: str = ""
        best_quality_score: float = 0.0
        final_cycle_idx: int = 0

        for cycle_idx in range(self.max_cycles):
            final_cycle_idx = cycle_idx

            # -- Cycle marker --
            cycle_step = ReasoningStep(
                step_type="cycle",
                content=f"Dream-Wake-Lucid cycle {cycle_idx}",
                parent_step_id=query_step.step_id,
                metadata={"cycle": cycle_idx},
            )
            trace.add_step(cycle_step)

            # ---- DREAM PHASE ----
            dream_ideas = await self._dream_phase(
                query=query,
                context=accumulated_insights,
                provider=llm_provider,
                trace=trace,
                cycle_idx=cycle_idx,
                parent_step_id=cycle_step.step_id,
            )

            # ---- WAKE PHASE ----
            scored_dreams = await self._wake_phase(
                dreams=dream_ideas,
                query=query,
                provider=llm_provider,
                tools=tools,
                trace=trace,
                cycle_idx=cycle_idx,
                parent_step_id=cycle_step.step_id,
            )

            # Sort by composite score and pick top-k
            scored_dreams.sort(key=lambda d: d.composite_score, reverse=True)
            top_dreams = scored_dreams[: self.top_k_dreams]

            # ---- LUCID PHASE ----
            lucid_output = await self._lucid_phase(
                top_dreams=top_dreams,
                query=query,
                provider=llm_provider,
                trace=trace,
                cycle_idx=cycle_idx,
                parent_step_id=cycle_step.step_id,
            )

            # ---- QUALITY CHECK ----
            quality_score = await self._quality_check(
                solution=lucid_output,
                query=query,
                provider=llm_provider,
                trace=trace,
                cycle_idx=cycle_idx,
                parent_step_id=cycle_step.step_id,
            )

            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_lucid_output = lucid_output

            if quality_score >= self.quality_threshold:
                break

            # Feed insights back for next cycle
            accumulated_insights = (
                f"Previous cycle insights (quality score "
                f"{quality_score:.2f}):\n{lucid_output}\n\n"
                f"Top dream ideas and their scores:\n"
            )
            for i, sd in enumerate(top_dreams):
                accumulated_insights += (
                    f"  {i + 1}. (score={sd.composite_score:.3f}) "
                    f"{sd.idea[:200]}\n"
                    f"     Analysis: {sd.analysis[:150]}\n"
                )

        # ---- FINAL OUTPUT ----
        final_step = ReasoningStep(
            step_type="final_output",
            content=best_lucid_output,
            parent_step_id=query_step.step_id,
            score=best_quality_score,
            metadata={
                "total_cycles": final_cycle_idx + 1,
                "best_quality_score": best_quality_score,
            },
        )
        trace.add_step(final_step)

        trace.final_output = best_lucid_output
        trace.duration_ms = (time.perf_counter() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # DREAM phase
    # ------------------------------------------------------------------

    async def _dream_phase(
        self,
        query: str,
        context: str,
        provider: Any,
        trace: ReasoningTrace,
        cycle_idx: int,
        parent_step_id: str,
    ) -> list[str]:
        """Generate wild, unconstrained ideas at high temperature.

        The LLM is prompted to produce ``num_dreams`` numbered ideas.
        Each idea is extracted from the numbered list and recorded as a
        separate ``"dream"`` step in the trace.

        Args:
            query: Original user query.
            context: Accumulated insights from prior cycles (empty string
                for cycle 0).
            provider: LLM provider.
            trace: Current trace for metric tracking.
            cycle_idx: Current cycle index.
            parent_step_id: Step ID to use as parent for dream steps.

        Returns:
            List of idea strings, one per dream.
        """
        context_block = ""
        if context:
            context_block = (
                "\n\nYou have insights from a previous round of thinking. "
                "Use them as springboards -- go FURTHER, not backwards:\n"
                f"{context}"
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Generate {self.num_dreams} radically different and "
                    f"creative approaches to the following challenge. Each "
                    f"idea should be a genuinely distinct direction, not a "
                    f"variation of the same theme. Number them 1 through "
                    f"{self.num_dreams}.\n\n"
                    f"Challenge: {query}"
                    f"{context_block}"
                ),
            },
        ]

        raw_response = await self._call_llm(
            provider,
            messages,
            trace,
            system=_DREAM_SYSTEM,
            temperature=self.dream_temperature,
        )

        # Parse numbered list
        ideas = self._parse_numbered_list(raw_response, self.num_dreams)

        # Record each idea as a step
        for i, idea in enumerate(ideas):
            dream_step = ReasoningStep(
                step_type="dream",
                content=idea,
                parent_step_id=parent_step_id,
                metadata={"cycle": cycle_idx, "dream_index": i},
            )
            trace.add_step(dream_step)

        return ideas

    # ------------------------------------------------------------------
    # WAKE phase
    # ------------------------------------------------------------------

    async def _wake_phase(
        self,
        dreams: list[str],
        query: str,
        provider: Any,
        tools: list[Any] | None,
        trace: ReasoningTrace,
        cycle_idx: int,
        parent_step_id: str,
    ) -> list[_ScoredDream]:
        """Evaluate and score each dream idea at low temperature.

        Each dream is independently assessed for feasibility, novelty,
        completeness, and risk.  The LLM responds with structured scores
        that are parsed to produce a composite score per idea.

        Args:
            dreams: List of dream idea strings.
            query: Original user query.
            provider: LLM provider.
            tools: Optional tools for reality-checking (reserved).
            trace: Current trace.
            cycle_idx: Current cycle index.
            parent_step_id: Parent step ID.

        Returns:
            List of :class:`_ScoredDream` objects with composite scores.
        """
        # Find dream steps for this cycle to link evaluations
        dream_steps = [
            s for s in trace.steps
            if s.step_type == "dream"
            and s.metadata.get("cycle") == cycle_idx
        ]

        scored: list[_ScoredDream] = []

        for i, idea in enumerate(dreams):
            dream_step_id = (
                dream_steps[i].step_id
                if i < len(dream_steps)
                else parent_step_id
            )

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Original question: {query}\n\n"
                        f"Evaluate this idea:\n\n{idea}"
                    ),
                },
            ]

            evaluation_text = await self._call_llm(
                provider,
                messages,
                trace,
                system=_WAKE_SYSTEM,
                temperature=self.wake_temperature,
            )

            # Parse scores from evaluation
            sd = _ScoredDream(idea=idea)
            sd.feasibility = self._parse_score(evaluation_text, "FEASIBILITY")
            sd.novelty = self._parse_score(evaluation_text, "NOVELTY")
            sd.completeness = self._parse_score(
                evaluation_text, "COMPLETENESS"
            )
            sd.risk = self._parse_score(evaluation_text, "RISK")
            sd.analysis = self._parse_field(evaluation_text, "ANALYSIS")
            sd.compute_composite()

            # Record evaluation step
            eval_step = ReasoningStep(
                step_type="evaluation",
                content=evaluation_text,
                parent_step_id=dream_step_id,
                score=sd.composite_score,
                metadata={
                    "cycle": cycle_idx,
                    "dream_index": i,
                    "feasibility": sd.feasibility,
                    "novelty": sd.novelty,
                    "completeness": sd.completeness,
                    "risk": sd.risk,
                    "composite_score": sd.composite_score,
                },
            )
            trace.add_step(eval_step)
            scored.append(sd)

        return scored

    # ------------------------------------------------------------------
    # LUCID phase
    # ------------------------------------------------------------------

    async def _lucid_phase(
        self,
        top_dreams: list[_ScoredDream],
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        cycle_idx: int,
        parent_step_id: str,
    ) -> str:
        """Synthesize the best dream elements into a coherent solution.

        The top-scoring dreams (with their evaluation analyses) are
        presented to the LLM, which is asked to combine the strongest
        elements while maintaining creative spark and grounding the
        result in reality.

        Args:
            top_dreams: Top-k scored dreams.
            query: Original user query.
            provider: LLM provider.
            trace: Current trace.
            cycle_idx: Current cycle index.
            parent_step_id: Parent step ID.

        Returns:
            The synthesized lucid output text.
        """
        dreams_block = ""
        for i, sd in enumerate(top_dreams):
            dreams_block += (
                f"\n--- IDEA {i + 1} (composite score: "
                f"{sd.composite_score:.3f}) ---\n"
                f"Feasibility: {sd.feasibility:.2f} | "
                f"Novelty: {sd.novelty:.2f} | "
                f"Completeness: {sd.completeness:.2f} | "
                f"Risk: {sd.risk:.2f}\n\n"
                f"Idea: {sd.idea}\n\n"
                f"Analysis: {sd.analysis}\n"
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original challenge: {query}\n\n"
                    f"The following top ideas have been generated and "
                    f"evaluated:"
                    f"{dreams_block}\n\n"
                    f"Combine the best elements into a single, coherent, "
                    f"actionable solution. Preserve the creative strengths "
                    f"while addressing the identified weaknesses."
                ),
            },
        ]

        lucid_output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_LUCID_SYSTEM,
            temperature=0.5,  # Balanced between creative and analytical
        )

        lucid_step = ReasoningStep(
            step_type="lucid",
            content=lucid_output,
            parent_step_id=parent_step_id,
            metadata={
                "cycle": cycle_idx,
                "num_input_dreams": len(top_dreams),
                "top_scores": [sd.composite_score for sd in top_dreams],
            },
        )
        trace.add_step(lucid_step)

        return lucid_output

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------

    async def _quality_check(
        self,
        solution: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        cycle_idx: int,
        parent_step_id: str,
    ) -> float:
        """Assess the overall quality of the lucid output.

        The LLM evaluates the solution on creativity, practicality, depth,
        and coherence, producing an OVERALL score that determines whether
        the engine should continue cycling.

        Args:
            solution: The lucid-phase output to assess.
            query: Original user query.
            provider: LLM provider.
            trace: Current trace.
            cycle_idx: Current cycle index.
            parent_step_id: Parent step ID.

        Returns:
            Overall quality score (0.0 -- 1.0).
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original challenge: {query}\n\n"
                    f"Proposed solution:\n\n{solution}"
                ),
            },
        ]

        quality_text = await self._call_llm(
            provider,
            messages,
            trace,
            system=_QUALITY_SYSTEM,
            temperature=0.1,
        )

        overall = self._parse_score(quality_text, "OVERALL")
        creativity = self._parse_score(quality_text, "CREATIVITY")
        practicality = self._parse_score(quality_text, "PRACTICALITY")
        depth = self._parse_score(quality_text, "DEPTH")
        coherence = self._parse_score(quality_text, "COHERENCE")

        quality_step = ReasoningStep(
            step_type="quality_check",
            content=quality_text,
            parent_step_id=parent_step_id,
            score=overall,
            metadata={
                "cycle": cycle_idx,
                "overall": overall,
                "creativity": creativity,
                "practicality": practicality,
                "depth": depth,
                "coherence": coherence,
                "threshold": self.quality_threshold,
                "passed": overall >= self.quality_threshold,
            },
        )
        trace.add_step(quality_step)

        return overall

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numbered_list(text: str, expected_count: int) -> list[str]:
        """Extract items from a numbered list in the LLM's response.

        Handles formats like ``1. ...``, ``1) ...``, ``1: ...``.  Falls
        back to splitting by double-newline if numbered parsing finds fewer
        items than expected.

        Args:
            text: Raw LLM response text.
            expected_count: How many items were requested.

        Returns:
            List of extracted idea strings.  May be shorter than
            ``expected_count`` if the LLM produced fewer items, or longer
            if it produced more.

        Example::

            ideas = DreamWakeCycle._parse_numbered_list(
                "1. Build a flying car\\n2. Use telepathy\\n3. Train dolphins",
                expected_count=3,
            )
            # ideas == ["Build a flying car", "Use telepathy", "Train dolphins"]
        """
        # Try numbered patterns: "1. ...", "1) ...", "1: ..."
        pattern = r'(?:^|\n)\s*\d+[\.\)\:]\s*(.*?)(?=\n\s*\d+[\.\)\:]|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches and len(matches) >= 2:
            # Clean up whitespace
            return [m.strip() for m in matches if m.strip()]

        # Fallback: split by double-newline
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if len(chunks) >= 2:
            return chunks

        # Last resort: treat the whole text as one idea
        return [text.strip()] if text.strip() else []

    @staticmethod
    def _parse_score(text: str, label: str) -> float:
        """Extract a numerical score from a labelled line.

        Searches for a line like ``LABEL: 0.75`` and returns the float
        value.  Returns 0.5 as a neutral default if the label is not
        found or the value cannot be parsed.

        Args:
            text: The evaluation text to search.
            label: The label to look for (e.g. ``"FEASIBILITY"``).

        Returns:
            Extracted float, clamped to [0.0, 1.0], or 0.5 if not found.

        Example::

            score = DreamWakeCycle._parse_score(
                "FEASIBILITY: 0.85\\nNOVELTY: 0.6", "FEASIBILITY"
            )
            # score == 0.85
        """
        pattern = rf'{re.escape(label)}\s*:\s*([0-9]*\.?[0-9]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            except ValueError:
                return 0.5
        return 0.5

    @staticmethod
    def _parse_field(text: str, label: str) -> str:
        """Extract the text content following a labelled line.

        Searches for ``LABEL:`` and returns everything after it until the
        next labelled line (``UPPERCASE_WORD:``) or end of text.

        Args:
            text: The text to search.
            label: The label to look for.

        Returns:
            Extracted text, or empty string if not found.

        Example::

            analysis = DreamWakeCycle._parse_field(
                "RISK: 0.3\\nANALYSIS: This is a good idea because...",
                "ANALYSIS",
            )
            # analysis == "This is a good idea because..."
        """
        pattern = rf'{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
