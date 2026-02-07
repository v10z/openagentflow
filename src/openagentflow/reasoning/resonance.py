"""Resonance Network reasoning engine.

Thoughts form a network where consistent ideas amplify and contradictions
suppress. The engine iteratively propagates activation through the network
until it reaches a stable attractor state. Surviving high-activation
thoughts form the final answer -- analogous to a Boltzmann machine for
reasoning.

Phase overview::

    SEED       -> Generate N independent thoughts about the problem
    CONNECT    -> Evaluate pairwise reinforcement / contradiction
    PROPAGATE  -> Update scores via the resonance matrix
    PRUNE      -> Remove thoughts below activation threshold
    STABILIZE  -> Repeat propagation until equilibrium
    SYNTHESIZE -> High-activation thoughts form the answer

Example::

    from openagentflow.reasoning.resonance import ResonanceNetwork

    engine = ResonanceNetwork(num_seeds=8, propagation_rounds=3)
    trace = await engine.reason(
        query="What are the trade-offs of microservice architectures?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openagentflow.core.types import Message, ToolSpec
from openagentflow.llm.base import BaseLLMProvider
from openagentflow.reasoning.base import ReasoningEngine, ReasoningTrace

logger = logging.getLogger(__name__)


class ResonanceNetwork(ReasoningEngine):
    """Thoughts form a network where consistent ideas amplify and contradictions suppress.

    The engine works in six phases:

    1. **SEED** -- Generate ``num_seeds`` independent thoughts / observations
       about the problem.  Each thought is evaluated in isolation so that
       diversity is maximised.
    2. **CONNECT** -- For every pair of thoughts, ask the LLM whether they
       *reinforce* or *contradict* each other and how strongly (a float in
       ``[-1, 1]``).  The result is a resonance matrix.
    3. **PROPAGATE** -- Update each thought's activation score by summing
       weighted influences from connected thoughts.  Reinforced thoughts
       gain energy; contradicted ones lose it.
    4. **PRUNE** -- Remove thoughts whose activation has fallen below
       ``activation_threshold``.
    5. **STABILIZE** -- Repeat PROPAGATE + PRUNE for ``propagation_rounds``
       iterations or until the network has converged (scores change less
       than ``convergence_epsilon``).
    6. **SYNTHESIZE** -- Ask the LLM to produce a final answer from the
       surviving high-activation thoughts.

    The metaphor is a Boltzmann machine: the stable attractor state of the
    network *is* the coherent answer.

    Attributes:
        name: ``"ResonanceNetwork"``
        description: Short human-readable summary.
        num_seeds: Number of initial thoughts to generate.
        propagation_rounds: Maximum number of propagation iterations.
        activation_threshold: Minimum score for a thought to survive pruning.
        reinforcement_strength: How strongly reinforcement boosts a score.
        contradiction_strength: How strongly contradiction diminishes a score.
        convergence_epsilon: If the max score change in a round is below this
            value the network is considered stable.
    """

    name: str = "ResonanceNetwork"
    description: str = (
        "Thoughts amplify/attenuate each other in a resonance network "
        "until a stable attractor forms."
    )

    def __init__(
        self,
        num_seeds: int = 8,
        propagation_rounds: int = 3,
        activation_threshold: float = 0.3,
        reinforcement_strength: float = 0.2,
        contradiction_strength: float = 0.3,
        convergence_epsilon: float = 0.02,
    ) -> None:
        """Initialise the Resonance Network engine.

        Args:
            num_seeds: Number of independent seed thoughts to generate.
                Higher values increase diversity but cost more LLM calls.
            propagation_rounds: Maximum propagation iterations before
                forced synthesis.
            activation_threshold: Thoughts with a score below this after
                propagation are pruned from the network.
            reinforcement_strength: Multiplier for positive (reinforcing)
                resonance between two thoughts.
            contradiction_strength: Multiplier for negative (contradicting)
                resonance between two thoughts.
            convergence_epsilon: If the largest absolute score change in a
                propagation round is below this, the network is deemed
                stable and propagation stops early.
        """
        self.num_seeds = max(2, num_seeds)
        self.propagation_rounds = max(1, propagation_rounds)
        self.activation_threshold = activation_threshold
        self.reinforcement_strength = reinforcement_strength
        self.contradiction_strength = contradiction_strength
        self.convergence_epsilon = convergence_epsilon

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: BaseLLMProvider,
        tools: list[ToolSpec] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the full Resonance Network reasoning strategy.

        Args:
            query: The user question or problem to reason about.
            llm_provider: An LLM provider for generating and evaluating
                thoughts.
            tools: Optional tool specs (currently unused by this engine).
            max_iterations: Soft cap -- the total propagation rounds are
                limited by ``min(propagation_rounds, max_iterations)``.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing seed, connection,
            propagation, pruning, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_rounds = min(self.propagation_rounds, max_iterations)

        # Phase 1 -- SEED
        thoughts = await self._seed_thoughts(query, llm_provider, trace)

        # Phase 2 -- CONNECT
        resonance_matrix = await self._build_resonance_matrix(
            query, thoughts, llm_provider, trace
        )

        # Phases 3-5 -- PROPAGATE + PRUNE + STABILIZE
        thoughts = await self._propagate_and_prune(
            thoughts, resonance_matrix, effective_rounds, trace
        )

        # Phase 6 -- SYNTHESIZE
        final_output = await self._synthesize(query, thoughts, llm_provider, trace)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- SEED
    # ------------------------------------------------------------------

    async def _seed_thoughts(
        self,
        query: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate ``num_seeds`` independent seed thoughts.

        Each seed is an independent observation, hypothesis, or angle on
        the query.  Thoughts are returned as dicts with keys ``id``,
        ``content``, and ``score`` (initialised to 0.5).

        Args:
            query: The original user query.
            provider: LLM provider for generation.
            trace: Reasoning trace to record steps into.

        Returns:
            List of thought dicts.
        """
        prompt = (
            f"You are generating independent thoughts about a problem. "
            f"Generate exactly {self.num_seeds} distinct observations, "
            f"hypotheses, or angles about the following query. Each thought "
            f"should be independent and explore a DIFFERENT aspect.\n\n"
            f"Query: {query}\n\n"
            f"Return your response as a JSON array of strings, one per thought. "
            f"Example: [\"thought 1\", \"thought 2\", ...]\n"
            f"Return ONLY the JSON array, no other text."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system="You are a precise analytical reasoner. Always return valid JSON.",
        )

        thought_texts = self._parse_json_list(raw, self.num_seeds)

        thoughts: list[dict[str, Any]] = []
        for idx, text in enumerate(thought_texts):
            thought_id = f"seed-{idx}"
            step = self._make_step(
                step_type="seed",
                content=text,
                score=0.5,
                metadata={"phase": "seed", "index": idx},
            )
            trace.add_step(step)
            thoughts.append({
                "id": thought_id,
                "step_id": step.step_id,
                "content": text,
                "score": 0.5,
            })

        logger.debug("ResonanceNetwork: seeded %d thoughts", len(thoughts))
        return thoughts

    # ------------------------------------------------------------------
    # Phase 2 -- CONNECT
    # ------------------------------------------------------------------

    async def _build_resonance_matrix(
        self,
        query: str,
        thoughts: list[dict[str, Any]],
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> dict[tuple[int, int], float]:
        """Evaluate pairwise relationships between all thoughts.

        For each unique pair ``(i, j)`` the LLM is asked whether the two
        thoughts reinforce or contradict each other.  The result is a
        resonance value in ``[-1.0, 1.0]`` stored in a symmetric matrix.

        To reduce LLM calls, pairs are batched into a single prompt
        whenever possible.

        Args:
            query: The original query for context.
            thoughts: The seed thoughts from phase 1.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict mapping ``(i, j)`` to a resonance float in ``[-1, 1]``.
        """
        n = len(thoughts)
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        if not pairs:
            return {}

        # Build a batch prompt listing all pairs
        pair_descriptions = []
        for idx, (i, j) in enumerate(pairs):
            pair_descriptions.append(
                f"Pair {idx}: Thought A: \"{thoughts[i]['content']}\" | "
                f"Thought B: \"{thoughts[j]['content']}\""
            )

        prompt = (
            f"You are evaluating relationships between thoughts about this query:\n"
            f"\"{query}\"\n\n"
            f"For each pair below, determine if the two thoughts REINFORCE "
            f"(support / complement) or CONTRADICT (conflict / undermine) each "
            f"other. Rate the relationship strength from -1.0 (strong "
            f"contradiction) to +1.0 (strong reinforcement). 0.0 means no "
            f"meaningful relationship.\n\n"
            + "\n".join(pair_descriptions)
            + "\n\nReturn a JSON array of numbers, one per pair, in the same "
            f"order. Example for {len(pairs)} pairs: [0.5, -0.3, ...]\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system=(
                "You are a precise analytical reasoner. Return valid JSON only."
            ),
        )

        scores = self._parse_json_float_list(raw, len(pairs))

        resonance: dict[tuple[int, int], float] = {}
        for idx, (i, j) in enumerate(pairs):
            val = max(-1.0, min(1.0, scores[idx]))
            resonance[(i, j)] = val
            resonance[(j, i)] = val

        # Record a single connection step summarising the matrix
        step = self._make_step(
            step_type="connection",
            content=(
                f"Evaluated {len(pairs)} thought pairs. "
                f"Reinforcements: {sum(1 for v in scores if v > 0)}, "
                f"Contradictions: {sum(1 for v in scores if v < 0)}, "
                f"Neutral: {sum(1 for v in scores if v == 0.0)}"
            ),
            metadata={
                "phase": "connect",
                "pair_count": len(pairs),
                "scores": {f"{i}-{j}": v for (i, j), v in resonance.items() if i < j},
            },
        )
        trace.add_step(step)
        return resonance

    # ------------------------------------------------------------------
    # Phases 3-5 -- PROPAGATE + PRUNE + STABILIZE
    # ------------------------------------------------------------------

    async def _propagate_and_prune(
        self,
        thoughts: list[dict[str, Any]],
        resonance: dict[tuple[int, int], float],
        rounds: int,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Iteratively propagate activation and prune weak thoughts.

        In each round, every surviving thought's score is updated based on
        the weighted influence of all other surviving thoughts.  Thoughts
        whose score falls below ``activation_threshold`` are removed.

        Propagation stops early if the maximum absolute score change in a
        round is below ``convergence_epsilon``.

        Args:
            thoughts: Current thought list with scores.
            resonance: Pairwise resonance matrix.
            rounds: Maximum propagation rounds.
            trace: Reasoning trace.

        Returns:
            The surviving thoughts after all propagation rounds.
        """
        # Build index map from thought id to position
        alive = list(range(len(thoughts)))

        for rnd in range(rounds):
            if len(alive) < 2:
                break

            max_delta = 0.0
            new_scores: dict[int, float] = {}

            for i in alive:
                influence = 0.0
                for j in alive:
                    if i == j:
                        continue
                    key = (i, j)
                    if key not in resonance:
                        continue
                    r = resonance[key]
                    if r > 0:
                        influence += r * self.reinforcement_strength * thoughts[j]["score"]
                    else:
                        influence += r * self.contradiction_strength * thoughts[j]["score"]

                new_score = max(0.0, min(1.0, thoughts[i]["score"] + influence))
                delta = abs(new_score - thoughts[i]["score"])
                if delta > max_delta:
                    max_delta = delta
                new_scores[i] = new_score

            # Apply new scores
            for i, s in new_scores.items():
                thoughts[i]["score"] = s

            # Prune
            pre_prune_count = len(alive)
            alive = [
                i for i in alive
                if thoughts[i]["score"] >= self.activation_threshold
            ]
            pruned_count = pre_prune_count - len(alive)

            # Record propagation step
            step = self._make_step(
                step_type="propagation",
                content=(
                    f"Round {rnd + 1}: max_delta={max_delta:.4f}, "
                    f"pruned={pruned_count}, surviving={len(alive)}"
                ),
                score=max_delta,
                metadata={
                    "phase": "propagate",
                    "round": rnd + 1,
                    "max_delta": max_delta,
                    "pruned": pruned_count,
                    "surviving": len(alive),
                    "scores": {thoughts[i]["id"]: thoughts[i]["score"] for i in alive},
                },
            )
            trace.add_step(step)

            # Check convergence
            if max_delta < self.convergence_epsilon:
                conv_step = self._make_step(
                    step_type="stabilization",
                    content=(
                        f"Network converged at round {rnd + 1} "
                        f"(max_delta={max_delta:.4f} < epsilon={self.convergence_epsilon})"
                    ),
                    metadata={"phase": "stabilize", "converged_at_round": rnd + 1},
                )
                trace.add_step(conv_step)
                break

        # Return surviving thoughts sorted by score descending
        surviving = [thoughts[i] for i in alive]
        surviving.sort(key=lambda t: t["score"], reverse=True)
        return surviving

    # ------------------------------------------------------------------
    # Phase 6 -- SYNTHESIZE
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        surviving_thoughts: list[dict[str, Any]],
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesise the final answer from surviving high-activation thoughts.

        The LLM is given the original query and the surviving thoughts
        (ranked by activation score) and asked to weave them into a
        coherent, comprehensive answer.

        Args:
            query: The original user query.
            surviving_thoughts: Thoughts that survived propagation and
                pruning, sorted by score descending.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The synthesised final answer text.
        """
        if not surviving_thoughts:
            step = self._make_step(
                step_type="synthesis",
                content="No thoughts survived propagation. Generating direct answer.",
                metadata={"phase": "synthesize", "surviving_count": 0},
            )
            trace.add_step(step)
            return await self._call_llm(
                provider=provider,
                messages=[Message.user(query)],
                trace=trace,
            )

        thought_listing = "\n".join(
            f"  [{t['score']:.2f}] {t['content']}" for t in surviving_thoughts
        )

        prompt = (
            f"You are synthesising a final answer from a resonance network "
            f"of thoughts. The thoughts below survived iterative amplification "
            f"and pruning -- they represent the most coherent, mutually "
            f"reinforcing ideas about the query.\n\n"
            f"Original query: {query}\n\n"
            f"Surviving thoughts (score = activation level):\n{thought_listing}\n\n"
            f"Synthesise these into a comprehensive, well-structured answer. "
            f"Give more weight to higher-scoring thoughts. Resolve any remaining "
            f"tensions and present a unified response."
        )

        answer = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system="You are a precise analytical synthesiser.",
        )

        step = self._make_step(
            step_type="synthesis",
            content=answer,
            score=1.0,
            metadata={
                "phase": "synthesize",
                "surviving_count": len(surviving_thoughts),
                "top_score": surviving_thoughts[0]["score"] if surviving_thoughts else 0,
            },
        )
        trace.add_step(step)
        return answer

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_list(raw: str, expected: int) -> list[str]:
        """Parse a JSON array of strings from LLM output.

        Falls back to line splitting if JSON parsing fails.

        Args:
            raw: Raw LLM output string.
            expected: Expected number of items.

        Returns:
            List of strings, padded or trimmed to *expected* length.
        """
        text = raw.strip()
        # Try to extract JSON array from the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    result = [str(item) for item in parsed]
                    # Pad or trim
                    while len(result) < expected:
                        result.append(f"Additional perspective #{len(result) + 1}")
                    return result[:expected]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: split by newlines, strip numbering
        lines = [
            ln.strip().lstrip("0123456789.-) ") for ln in text.splitlines() if ln.strip()
        ]
        lines = [ln for ln in lines if ln]
        while len(lines) < expected:
            lines.append(f"Additional perspective #{len(lines) + 1}")
        return lines[:expected]

    @staticmethod
    def _parse_json_float_list(raw: str, expected: int) -> list[float]:
        """Parse a JSON array of floats from LLM output.

        Falls back to zeros if parsing fails.

        Args:
            raw: Raw LLM output string.
            expected: Expected number of items.

        Returns:
            List of floats, padded with 0.0 if needed.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    result = [float(v) for v in parsed]
                    while len(result) < expected:
                        result.append(0.0)
                    return result[:expected]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return [0.0] * expected


__all__ = ["ResonanceNetwork"]
