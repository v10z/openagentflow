"""Dialectical Spiral reasoning engine for OpenAgentFlow.

Implements Hegelian dialectical reasoning -- Thesis, Antithesis, Synthesis --
applied recursively at increasing levels of abstraction.  Each synthesis
becomes the thesis for the next level, creating an ascending spiral of
increasingly refined understanding.

This engine produces deeper reasoning than Tree-of-Thought because it does not
merely explore alternatives; it forces genuine intellectual confrontation
between opposing positions and resolves them into higher-order insights.

The spiral terminates either when the synthesis converges (measured by word-
overlap similarity between consecutive syntheses exceeding
``convergence_threshold``) or when ``max_depth`` levels have been completed.

Example::

    from openagentflow.reasoning import DialecticalSpiral

    engine = DialecticalSpiral(max_depth=4, convergence_threshold=0.85)
    trace = await engine.reason(
        query="Should AI systems be open-sourced or kept proprietary?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # Inspect the dialectical levels
    for step in trace.get_steps_by_type("synthesis"):
        level = step.metadata.get("level", "?")
        print(f"Level {level} synthesis: {step.content[:120]}...")

Trace structure (DAG)::

    query
      +-- thesis_L0
      |     +-- antithesis_L0
      |           +-- synthesis_L0
      |                 +-- thesis_L1  (== synthesis_L0 reframed)
      |                       +-- antithesis_L1
      |                             +-- synthesis_L1
      |                                   +-- ...
      +-- convergence_check (metadata only)
      +-- final_output
"""

from __future__ import annotations

import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_THESIS_SYSTEM = (
    "You are a rigorous analytical thinker. Given a question or topic, "
    "develop a clear, well-reasoned position. Present your argument with "
    "supporting evidence, logical structure, and concrete examples. Be "
    "specific rather than vague. If you are building on a prior synthesis, "
    "take it as your starting point and deepen or extend the argument."
)

_ANTITHESIS_SYSTEM = (
    "You are a brilliant devil's advocate. Your job is to find the strongest "
    "possible objections, counter-examples, and flaws in the following "
    "position. Be intellectually honest -- do NOT strawman. Find REAL "
    "weaknesses: hidden assumptions, logical gaps, ignored perspectives, "
    "empirical counter-evidence, and edge cases where the position breaks "
    "down. Your goal is to make the strongest possible case AGAINST the "
    "thesis, even if you personally agree with it."
)

_SYNTHESIS_SYSTEM = (
    "You are a master synthesizer. Given two opposing positions on the same "
    "question, find the truth that transcends both. Do NOT simply compromise "
    "or split the difference. Instead, find the deeper principle that "
    "explains WHY both sides have merit, and construct a more nuanced "
    "position that preserves the genuine insights of each while resolving "
    "their contradictions. Your synthesis should be a genuine intellectual "
    "advance over either position taken alone."
)


# ---------------------------------------------------------------------------
# DialecticalSpiral
# ---------------------------------------------------------------------------


class DialecticalSpiral(ReasoningEngine):
    """Reasoning through contradiction and synthesis at increasing abstraction.

    At each level of the spiral:

    1. **THESIS** -- Generate an initial position or solution.  At level 0
       this is derived directly from the user query; at subsequent levels it
       is the prior level's synthesis, reframed as a thesis.
    2. **ANTITHESIS** -- Argue against the thesis from the strongest possible
       opposing perspective.  The LLM is prompted to be an intellectually
       honest devil's advocate.
    3. **SYNTHESIS** -- Reconcile the two positions into a higher-order
       understanding that preserves genuine insights from both while
       resolving their contradictions.

    The synthesis at level *N* becomes the thesis for level *N + 1*, creating
    an ascending spiral.  The engine stops when:

    - The synthesis converges (word-overlap Jaccard similarity exceeds
      ``convergence_threshold``), **or**
    - ``max_depth`` levels have been completed.

    The final synthesis is returned as the trace's ``final_output``.

    Args:
        max_depth: Maximum number of dialectical levels (default 3).
        convergence_threshold: Word-overlap Jaccard similarity threshold
            (0.0 -- 1.0) above which two consecutive syntheses are considered
            converged (default 0.85).

    Attributes:
        name: ``"dialectical_spiral"``
        description: Short description of the engine.
        max_depth: Maximum spiral depth.
        convergence_threshold: Convergence detection threshold.

    Example::

        engine = DialecticalSpiral(max_depth=3, convergence_threshold=0.85)
        trace = await engine.reason(
            query="Is democracy the best form of government?",
            llm_provider=provider,
        )
        # trace.steps contains thesis/antithesis/synthesis for each level
        for s in trace.steps:
            print(f"[{s.step_type}] (level {s.metadata.get('level')})")
            print(s.content[:200])
            print()

        # Inspect convergence
        for s in trace.get_steps_by_type("convergence_check"):
            print(f"Similarity: {s.metadata['similarity']:.3f}")

        # Export as graph
        dag = trace.to_dag()
    """

    name: str = "dialectical_spiral"
    description: str = (
        "Thesis-Antithesis-Synthesis at increasing abstraction levels. "
        "Forces genuine intellectual confrontation to refine understanding."
    )

    def __init__(
        self,
        max_depth: int = 3,
        convergence_threshold: float = 0.85,
    ) -> None:
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold

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
        """Execute the dialectical spiral and return a full reasoning trace.

        The engine runs up to ``max_depth`` levels of thesis-antithesis-
        synthesis.  At each level after the first, it checks whether the
        new synthesis has converged with the previous one (Jaccard
        similarity on significant words).  If so, it exits early.

        The ``max_iterations`` and ``tools`` parameters are accepted for
        interface compatibility but are not used by this engine.

        Args:
            query: The user question or task.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Ignored by this engine.
            max_iterations: Ignored -- ``max_depth`` controls depth.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with thesis / antithesis / synthesis
            steps, convergence-check metadata, and the final output.
        """
        start = time.perf_counter()
        trace = ReasoningTrace(strategy_name=self.name)

        # Record the initial query as the root step
        query_step = ReasoningStep(
            step_type="query",
            content=query,
            metadata={"role": "user_query"},
        )
        trace.add_step(query_step)

        previous_synthesis: str | None = None
        synthesis_content: str = ""
        last_synthesis_step_id: str = query_step.step_id
        completed_levels: int = 0

        for level in range(self.max_depth):
            # ---- THESIS ----
            if level == 0:
                thesis_context = query
            else:
                thesis_context = (
                    f"Building on the following prior synthesis, develop it "
                    f"further as a refined thesis.\n\n"
                    f"Original question: {query}\n\n"
                    f"Prior synthesis (level {level - 1}):\n{previous_synthesis}"
                )

            thesis_content = await self._generate_thesis(
                query=query,
                context=thesis_context,
                provider=llm_provider,
                trace=trace,
            )
            thesis_step = ReasoningStep(
                step_type="thesis",
                content=thesis_content,
                parent_step_id=last_synthesis_step_id,
                metadata={"level": level},
            )
            trace.add_step(thesis_step)

            # ---- ANTITHESIS ----
            antithesis_content = await self._generate_antithesis(
                thesis=thesis_content,
                query=query,
                provider=llm_provider,
                trace=trace,
            )
            antithesis_step = ReasoningStep(
                step_type="antithesis",
                content=antithesis_content,
                parent_step_id=thesis_step.step_id,
                metadata={"level": level},
            )
            trace.add_step(antithesis_step)

            # ---- SYNTHESIS ----
            synthesis_content = await self._generate_synthesis(
                thesis=thesis_content,
                antithesis=antithesis_content,
                query=query,
                provider=llm_provider,
                trace=trace,
                level=level,
            )
            synthesis_step = ReasoningStep(
                step_type="synthesis",
                content=synthesis_content,
                parent_step_id=antithesis_step.step_id,
                metadata={"level": level},
            )
            trace.add_step(synthesis_step)

            completed_levels = level + 1

            # ---- CONVERGENCE CHECK ----
            if previous_synthesis is not None:
                similarity = self._check_convergence(
                    synthesis_content, previous_synthesis
                )
                convergence_step = ReasoningStep(
                    step_type="convergence_check",
                    content=(
                        f"Similarity between level {level - 1} and level "
                        f"{level} syntheses: {similarity:.4f} "
                        f"(threshold: {self.convergence_threshold})"
                    ),
                    parent_step_id=synthesis_step.step_id,
                    score=similarity,
                    metadata={
                        "level": level,
                        "similarity": similarity,
                        "threshold": self.convergence_threshold,
                        "converged": similarity >= self.convergence_threshold,
                    },
                )
                trace.add_step(convergence_step)

                if similarity >= self.convergence_threshold:
                    # Reasoning has converged -- exit the spiral
                    break

            previous_synthesis = synthesis_content
            last_synthesis_step_id = synthesis_step.step_id

        # ---- FINAL OUTPUT ----
        final_content = synthesis_content

        final_step = ReasoningStep(
            step_type="final_output",
            content=final_content,
            parent_step_id=last_synthesis_step_id,
            metadata={"total_levels": completed_levels},
            score=1.0,
        )
        trace.add_step(final_step)

        trace.final_output = final_content
        trace.duration_ms = (time.perf_counter() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _generate_thesis(
        self,
        query: str,
        context: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the thesis for a dialectical level.

        Args:
            query: Original user query.
            context: Either the raw query (level 0) or the reframed prior
                synthesis (subsequent levels).
            provider: LLM provider.
            trace: Current trace for metric tracking.

        Returns:
            The thesis text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Develop a clear, well-reasoned position on the "
                    f"following.\n\n{context}"
                ),
            },
        ]
        return await self._call_llm(
            provider,
            messages,
            trace,
            system=_THESIS_SYSTEM,
            temperature=0.7,
        )

    async def _generate_antithesis(
        self,
        thesis: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the strongest possible counter-argument to the thesis.

        The LLM is instructed to act as an intellectually honest devil's
        advocate -- finding real weaknesses, not strawmen.

        Args:
            thesis: The thesis text to argue against.
            query: Original user query for context.
            provider: LLM provider.
            trace: Current trace for metric tracking.

        Returns:
            The antithesis text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"The following thesis has been proposed:\n\n"
                    f"---THESIS---\n{thesis}\n---END THESIS---\n\n"
                    f"Find the strongest possible objections, counter-"
                    f"examples, hidden assumptions, and logical flaws. "
                    f"Argue against this position as forcefully and "
                    f"honestly as you can."
                ),
            },
        ]
        return await self._call_llm(
            provider,
            messages,
            trace,
            system=_ANTITHESIS_SYSTEM,
            temperature=0.8,
        )

    async def _generate_synthesis(
        self,
        thesis: str,
        antithesis: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        level: int,
    ) -> str:
        """Synthesize thesis and antithesis into a higher-order understanding.

        The LLM is instructed to find the deeper principle that explains why
        both sides have merit, rather than merely splitting the difference.

        Args:
            thesis: The thesis text.
            antithesis: The antithesis text.
            query: Original user query.
            provider: LLM provider.
            trace: Current trace for metric tracking.
            level: Current dialectical level (for context in the prompt).

        Returns:
            The synthesis text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Two opposing positions have been developed "
                    f"(dialectical level {level}):\n\n"
                    f"---THESIS---\n{thesis}\n---END THESIS---\n\n"
                    f"---ANTITHESIS---\n{antithesis}\n---END ANTITHESIS---"
                    f"\n\n"
                    f"Synthesize these into a higher-order understanding "
                    f"that preserves the genuine insights of both while "
                    f"resolving their contradictions. Do not merely "
                    f"compromise -- find the deeper truth."
                ),
            },
        ]
        return await self._call_llm(
            provider,
            messages,
            trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.6,
        )

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_convergence(
        current_synthesis: str, previous_synthesis: str
    ) -> float:
        """Check whether two successive syntheses have converged.

        Uses Jaccard similarity on word sets (case-insensitive, stripped of
        common stop-words) as a fast, dependency-free approximation.  A
        value of 1.0 means the two texts share exactly the same significant
        vocabulary; 0.0 means they share none.

        This is intentionally a simple heuristic.  For production use, a
        semantic-similarity model would be more appropriate, but this keeps
        the engine dependency-free (pure stdlib).

        Args:
            current_synthesis: The synthesis from the current level.
            previous_synthesis: The synthesis from the previous level.

        Returns:
            Jaccard similarity coefficient (0.0 -- 1.0).

        Example::

            sim = DialecticalSpiral._check_convergence(
                "AI alignment requires both technical and social solutions.",
                "Aligning AI requires technical safety AND social governance.",
            )
            # sim will be high because the core vocabulary overlaps
        """
        # Minimal English stop-words to reduce noise without external deps
        _STOP_WORDS = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "nor", "not", "so",
            "yet", "both", "either", "neither", "each", "every", "all",
            "any", "few", "more", "most", "other", "some", "such", "no",
            "only", "same", "than", "too", "very", "just", "because",
            "if", "when", "where", "while", "how", "what", "which", "who",
            "whom", "this", "that", "these", "those", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their",
        })

        def _tokenize(text: str) -> set[str]:
            words: set[str] = set()
            for raw_word in text.lower().split():
                # Strip punctuation from edges
                cleaned = raw_word.strip(".,;:!?\"'()[]{}/-")
                if cleaned and cleaned not in _STOP_WORDS and len(cleaned) > 1:
                    words.add(cleaned)
            return words

        words_current = _tokenize(current_synthesis)
        words_previous = _tokenize(previous_synthesis)

        if not words_current and not words_previous:
            return 1.0
        if not words_current or not words_previous:
            return 0.0

        intersection = words_current & words_previous
        union = words_current | words_previous
        return len(intersection) / len(union)
