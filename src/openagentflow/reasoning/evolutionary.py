"""Evolutionary Thought reasoning engine.

Applies Darwinian selection to solution populations: an initial cohort of
diverse candidate solutions is iteratively improved through fitness
evaluation, tournament selection, crossover (recombination), and mutation
over multiple generations.  The best individual from the final generation
is returned as the answer.

Key insight compared to Tree-of-Thought: ToT *explores* branches
independently, whereas Evolutionary Thought *recombines* partial solutions
so that two mediocre ideas can cross-pollinate to produce a superior one.

Algorithm outline::

    1. INITIALIZATION  -- generate N diverse solution candidates
    2. For each generation G:
       a. FITNESS      -- score every candidate on multiple criteria
       b. SELECTION    -- tournament selection (random subset, keep best)
       c. ELITISM      -- carry top-E candidates unchanged into next gen
       d. CROSSOVER    -- pair selected parents, ask LLM to fuse strengths
       e. MUTATION     -- with probability p, randomly vary one aspect
    3. Return the highest-fitness individual from the final generation.

Pure stdlib -- no third-party dependencies beyond the framework itself.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from openagentflow.reasoning.base import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
)


class EvolutionaryThought(ReasoningEngine):
    """Apply evolutionary algorithms to LLM reasoning.

    The engine maintains a *population* of candidate solutions.  Each
    generation subjects the population to fitness evaluation, tournament
    selection, crossover, and mutation -- mirroring a real genetic
    algorithm but using LLM calls in place of numerical operators.

    Parameters:
        population_size: Number of candidate solutions per generation.
            Larger populations increase diversity but cost more LLM calls.
        generations: How many evolutionary cycles to run.
        mutation_rate: Probability (0-1) that an offspring is mutated.
        crossover_rate: Probability (0-1) that crossover produces a child
            (as opposed to cloning a parent).
        tournament_size: Number of candidates sampled in each tournament
            round.  Larger values increase selection pressure.
        elite_count: Number of top candidates passed unchanged into the
            next generation (elitism prevents regression).

    Example::

        engine = EvolutionaryThought(
            population_size=6,
            generations=3,
            mutation_rate=0.3,
        )
        trace = await engine.reason(
            query="Design a caching strategy for a social-media feed.",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "evolutionary_thought"
    description: str = (
        "Darwinian selection on a population of LLM-generated solution "
        "candidates with crossover, mutation, and elitism."
    )

    def __init__(
        self,
        population_size: int = 6,
        generations: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elite_count: int = 1,
    ) -> None:
        self.population_size = max(population_size, 2)
        self.generations = max(generations, 1)
        self.mutation_rate = max(0.0, min(mutation_rate, 1.0))
        self.crossover_rate = max(0.0, min(crossover_rate, 1.0))
        self.tournament_size = max(2, min(tournament_size, self.population_size))
        self.elite_count = max(0, min(elite_count, self.population_size - 1))

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
        """Run the full evolutionary cycle and return the best solution.

        Args:
            query: The problem statement to solve.
            llm_provider: A ``BaseLLMProvider`` instance used for all LLM
                calls (population generation, fitness scoring, crossover,
                and mutation).
            tools: Unused -- present for interface compatibility.
            max_iterations: Ignored (generations controls iteration count).
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` is the best
            solution from the final generation.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # --- Generation 0: seed the population ---
        population = await self._initialize_population(
            query, llm_provider, trace
        )

        # --- Evolutionary loop ---
        for gen_idx in range(self.generations):
            gen_step = ReasoningStep(
                step_type="generation",
                content=f"Generation {gen_idx + 1}/{self.generations} "
                        f"-- population size {len(population)}",
                metadata={"generation": gen_idx + 1},
            )
            trace.add_step(gen_step)

            # 1. FITNESS evaluation
            population = await self._evaluate_fitness(
                population, query, llm_provider, trace, gen_step.step_id
            )

            # Sort descending by fitness
            population.sort(key=lambda c: c["fitness"], reverse=True)

            # Record top fitness
            gen_step.score = population[0]["fitness"]
            gen_step.metadata["best_fitness"] = population[0]["fitness"]
            gen_step.metadata["avg_fitness"] = (
                sum(c["fitness"] for c in population) / len(population)
            )

            # 2. ELITISM -- carry best individuals forward unchanged
            elites = [dict(c) for c in population[: self.elite_count]]

            # 3. SELECTION + CROSSOVER + MUTATION to fill remaining slots
            offspring: list[dict[str, Any]] = []
            needed = self.population_size - len(elites)

            while len(offspring) < needed:
                parent_a = self._tournament_select(population, self.tournament_size)
                parent_b = self._tournament_select(population, self.tournament_size)

                # Crossover
                if random.random() < self.crossover_rate:
                    child_content = await self._crossover(
                        parent_a, parent_b, query, llm_provider, trace,
                        gen_step.step_id,
                    )
                else:
                    # Clone the fitter parent
                    child_content = (
                        parent_a["content"]
                        if parent_a["fitness"] >= parent_b["fitness"]
                        else parent_b["content"]
                    )

                # Mutation
                if random.random() < self.mutation_rate:
                    child_content = await self._mutate(
                        child_content, query, llm_provider, trace,
                        gen_step.step_id,
                    )

                offspring.append({
                    "content": child_content,
                    "fitness": 0.0,
                    "scores": {},
                })

            population = elites + offspring

        # --- Final evaluation of last generation ---
        population = await self._evaluate_fitness(
            population, query, llm_provider, trace, parent_step_id=None
        )
        population.sort(key=lambda c: c["fitness"], reverse=True)

        best = population[0]

        # Record champion
        champion_step = ReasoningStep(
            step_type="champion",
            content=best["content"],
            score=best["fitness"],
            metadata={"scores": best.get("scores", {}), "generation": "final"},
        )
        trace.add_step(champion_step)

        trace.final_output = best["content"]
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _initialize_population(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate a diverse initial population of candidate solutions.

        Asks the LLM to produce ``population_size`` distinct approaches in
        a single call, then parses them into individual candidates.

        Returns:
            A list of dicts with keys ``content`` (str), ``fitness``
            (float, initially 0), and ``scores`` (dict, initially empty).
        """
        n = self.population_size
        system_prompt = (
            "You are a creative problem-solver tasked with generating multiple "
            "RADICALLY DIFFERENT approaches to a problem. Each approach must "
            "be substantively unique -- vary the methodology, perspective, or "
            "core assumption. Do NOT produce minor variations of the same idea."
        )
        user_prompt = (
            f"Generate exactly {n} diverse and complete solution candidates "
            f"for the following problem. Number them 1 through {n}. Each "
            f"candidate should be a self-contained solution of roughly equal "
            f"detail.\n\nPROBLEM:\n{query}\n\nProvide your {n} solutions now, "
            f"each clearly numbered (e.g., '1.', '2.', etc.)."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.9,
        )

        candidates = self._parse_numbered_items(raw, n)

        population: list[dict[str, Any]] = []
        init_step = ReasoningStep(
            step_type="initialization",
            content=f"Generated {len(candidates)} initial candidates",
            metadata={"raw_response_length": len(raw)},
        )
        trace.add_step(init_step)

        for idx, text in enumerate(candidates):
            cand_step = ReasoningStep(
                step_type="candidate_init",
                content=text,
                parent_step_id=init_step.step_id,
                metadata={"candidate_index": idx},
            )
            trace.add_step(cand_step)
            init_step.children.append(cand_step.step_id)
            population.append({
                "content": text,
                "fitness": 0.0,
                "scores": {},
            })

        # If parsing produced fewer than needed, pad with extra LLM calls
        while len(population) < self.population_size:
            extra_prompt = (
                f"Propose ONE more completely different solution for:\n{query}\n\n"
                f"Avoid repeating any of these existing approaches:\n"
                + "\n".join(
                    f"- {p['content'][:120]}..." for p in population
                )
            )
            extra = await self._call_llm(
                provider,
                [{"role": "user", "content": extra_prompt}],
                trace,
                system=system_prompt,
                temperature=0.95,
            )
            extra_step = ReasoningStep(
                step_type="candidate_init",
                content=extra,
                parent_step_id=init_step.step_id,
                metadata={"candidate_index": len(population), "extra": True},
            )
            trace.add_step(extra_step)
            init_step.children.append(extra_step.step_id)
            population.append({
                "content": extra,
                "fitness": 0.0,
                "scores": {},
            })

        return population

    async def _evaluate_fitness(
        self,
        candidates: list[dict[str, Any]],
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Score every candidate on multiple quality criteria.

        Each candidate is evaluated by the LLM on four axes (each 0-10):

        * **Correctness** -- factual accuracy and logical soundness.
        * **Completeness** -- whether the solution addresses all aspects.
        * **Elegance** -- clarity, simplicity, and aesthetic quality.
        * **Novelty** -- originality compared to obvious approaches.

        The overall fitness is the weighted sum of these scores.

        Returns:
            The same list with ``fitness`` and ``scores`` updated in-place.
        """
        system_prompt = (
            "You are a rigorous evaluator. Score the given solution on the "
            "four criteria below, each on a 0-10 integer scale. Respond "
            "ONLY with valid JSON -- no markdown fences, no commentary.\n\n"
            "Criteria:\n"
            "  correctness  -- factual accuracy and logical soundness\n"
            "  completeness -- whether the solution addresses all aspects\n"
            "  elegance     -- clarity, simplicity, and quality of expression\n"
            "  novelty      -- originality relative to obvious approaches\n\n"
            'Format: {"correctness": N, "completeness": N, "elegance": N, "novelty": N}'
        )

        fitness_step = ReasoningStep(
            step_type="fitness_evaluation",
            content=f"Evaluating {len(candidates)} candidates",
            parent_step_id=parent_step_id,
            metadata={"candidate_count": len(candidates)},
        )
        trace.add_step(fitness_step)

        for idx, candidate in enumerate(candidates):
            user_prompt = (
                f"PROBLEM:\n{query}\n\n"
                f"SOLUTION:\n{candidate['content']}\n\n"
                f"Score this solution. Respond with JSON only."
            )

            raw = await self._call_llm(
                provider,
                [{"role": "user", "content": user_prompt}],
                trace,
                system=system_prompt,
                temperature=0.2,
            )

            scores = self._parse_scores(raw)
            # Weighted fitness: correctness counts most, novelty least
            fitness = (
                scores.get("correctness", 5) * 0.35
                + scores.get("completeness", 5) * 0.30
                + scores.get("elegance", 5) * 0.20
                + scores.get("novelty", 5) * 0.15
            )

            candidate["scores"] = scores
            candidate["fitness"] = round(fitness, 3)

            score_step = ReasoningStep(
                step_type="fitness_score",
                content=(
                    f"Candidate {idx + 1}: fitness={fitness:.3f} "
                    f"scores={json.dumps(scores)}"
                ),
                parent_step_id=fitness_step.step_id,
                score=fitness,
                metadata={
                    "candidate_index": idx,
                    "scores": scores,
                    "fitness": fitness,
                },
            )
            trace.add_step(score_step)
            fitness_step.children.append(score_step.step_id)

        return candidates

    async def _crossover(
        self,
        parent_a: dict[str, Any],
        parent_b: dict[str, Any],
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str | None = None,
    ) -> str:
        """Combine two parent solutions into a single offspring.

        The LLM is instructed to identify the strongest aspects of each
        parent and merge them into a coherent, improved solution.

        Returns:
            The text of the offspring solution.
        """
        system_prompt = (
            "You are a solution architect. You will be given two different "
            "solutions to the same problem. Your job is to create a SINGLE "
            "improved solution that combines the best elements of both. "
            "Identify each solution's unique strengths and weave them "
            "together into a coherent whole. The result must be a complete, "
            "self-contained solution -- not a summary of differences."
        )

        scores_a = parent_a.get("scores", {})
        scores_b = parent_b.get("scores", {})
        strength_hint_a = max(scores_a, key=scores_a.get) if scores_a else "overall quality"
        strength_hint_b = max(scores_b, key=scores_b.get) if scores_b else "overall quality"

        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"SOLUTION A (fitness {parent_a['fitness']:.2f}, "
            f"strongest in {strength_hint_a}):\n{parent_a['content']}\n\n"
            f"SOLUTION B (fitness {parent_b['fitness']:.2f}, "
            f"strongest in {strength_hint_b}):\n{parent_b['content']}\n\n"
            f"Create a single improved solution that takes A's strength in "
            f"{strength_hint_a} and B's strength in {strength_hint_b}, "
            f"combining the best aspects of both into one coherent answer."
        )

        child_content = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.7,
        )

        crossover_step = ReasoningStep(
            step_type="crossover",
            content=child_content,
            parent_step_id=parent_step_id,
            metadata={
                "parent_a_fitness": parent_a["fitness"],
                "parent_b_fitness": parent_b["fitness"],
                "strength_a": strength_hint_a,
                "strength_b": strength_hint_b,
            },
        )
        trace.add_step(crossover_step)

        return child_content

    async def _mutate(
        self,
        solution: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str | None = None,
    ) -> str:
        """Randomly vary one significant aspect of a solution.

        Uses a high temperature to encourage creative divergence.  The
        LLM is asked to change exactly one major element while keeping
        the rest intact.

        Returns:
            The mutated solution text.
        """
        system_prompt = (
            "You are a creative disruptor. Take the solution provided and "
            "change ONE significant aspect of it in an unexpected, creative "
            "way. The change should be meaningful -- not just rewording. "
            "Alter an assumption, swap a technique, change the architecture, "
            "or introduce a novel constraint. Keep the rest of the solution "
            "intact so it remains coherent and complete."
        )

        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"CURRENT SOLUTION:\n{solution}\n\n"
            f"Mutate this solution by changing ONE significant aspect. "
            f"Be creative and unexpected. Output the full revised solution."
        )

        mutated = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=1.0,
        )

        mutation_step = ReasoningStep(
            step_type="mutation",
            content=mutated,
            parent_step_id=parent_step_id,
            metadata={"original_length": len(solution), "mutated_length": len(mutated)},
        )
        trace.add_step(mutation_step)

        return mutated

    def _tournament_select(
        self,
        population: list[dict[str, Any]],
        tournament_size: int,
    ) -> dict[str, Any]:
        """Select a candidate via tournament selection.

        Randomly samples ``tournament_size`` individuals from the
        population and returns the one with the highest fitness.

        Args:
            population: The current generation.
            tournament_size: How many candidates to include in the
                tournament.

        Returns:
            The winning candidate dict.
        """
        pool_size = min(tournament_size, len(population))
        contenders = random.sample(population, pool_size)
        winner = max(contenders, key=lambda c: c["fitness"])
        return winner

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numbered_items(text: str, expected: int) -> list[str]:
        """Split LLM output that contains numbered items (``1. ...``, etc.).

        Falls back to splitting on double-newlines if numbering is not
        detected.

        Returns:
            A list of individual solution texts.
        """
        # Try numbered pattern first: "1.", "2.", etc. at line start
        # Handles both "1." and "1)" formats
        parts = re.split(r"\n(?=\d+[\.\)]\s)", text)
        # Clean leading number from each part
        cleaned: list[str] = []
        for part in parts:
            stripped = re.sub(r"^\d+[\.\)]\s*", "", part.strip())
            if stripped:
                cleaned.append(stripped)

        if len(cleaned) >= 2:
            return cleaned

        # Fallback: split on double newlines
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if len(chunks) >= 2:
            return chunks

        # Last resort: treat entire text as a single candidate
        return [text.strip()] if text.strip() else []

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, float]:
        """Extract a JSON scores dict from LLM output.

        Tolerates markdown fences and surrounding text.  Returns default
        mid-range scores on failure.

        Returns:
            Dict with keys ``correctness``, ``completeness``, ``elegance``,
            ``novelty`` -- each a float in [0, 10].
        """
        default_scores = {
            "correctness": 5.0,
            "completeness": 5.0,
            "elegance": 5.0,
            "novelty": 5.0,
        }

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Try to find a JSON object in the text
        match = re.search(r"\{[^{}]+\}", cleaned)
        if not match:
            return default_scores

        try:
            parsed = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return default_scores

        scores: dict[str, float] = {}
        for key in ("correctness", "completeness", "elegance", "novelty"):
            val = parsed.get(key, 5)
            try:
                scores[key] = float(max(0, min(10, float(val))))
            except (TypeError, ValueError):
                scores[key] = 5.0

        return scores
