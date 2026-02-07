"""Simulated Annealing reasoning engine.

Applies the simulated annealing metaheuristic to the reasoning process.
At high temperatures the engine explores wildly different solutions; as
the temperature cools it focuses on refining the best solution found.
The Metropolis acceptance criterion allows occasional acceptance of worse
solutions, enabling escape from local optima.

Temperature schedule::

    T(t) = initial_temperature * cooling_rate ^ t

Acceptance probability for a worse solution::

    P(accept) = exp((score_new - score_current) / T)

Unlike greedy search this can escape local optima.  Unlike random search
this converges to good solutions.

Example::

    from openagentflow.reasoning.annealing import SimulatedAnnealing

    engine = SimulatedAnnealing(initial_temperature=1.0, cooling_rate=0.7)
    trace = await engine.reason(
        query="Design an optimal caching strategy for a read-heavy API.",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from typing import Any

from openagentflow.core.types import Message, ToolSpec
from openagentflow.llm.base import BaseLLMProvider
from openagentflow.reasoning.base import ReasoningEngine, ReasoningTrace

logger = logging.getLogger(__name__)


class SimulatedAnnealing(ReasoningEngine):
    """Simulated annealing applied to reasoning.

    The engine starts at a high temperature, generating diverse and
    potentially unconventional solutions.  As the temperature drops
    according to an exponential cooling schedule, the engine progressively
    refines and narrows its focus to the best solution found so far.

    At each temperature level the engine:

    1. Generates ``iterations_per_temp`` *neighbour* solutions -- small
       perturbations or variations of the current best solution.
    2. Evaluates each neighbour's quality via LLM scoring.
    3. Applies the **Metropolis criterion**: if the neighbour is better,
       accept it immediately; if worse, accept it with probability
       ``exp((score_new - score_current) / T)``.  This stochastic
       acceptance allows the search to escape local optima.
    4. Cools the temperature: ``T = T * cooling_rate``.
    5. Repeats until ``T < min_temperature`` (system is "frozen").

    The best solution encountered across *all* temperatures is returned.

    Attributes:
        name: ``"SimulatedAnnealing"``
        description: Short human-readable summary.
        initial_temperature: Starting temperature (higher = more random).
        cooling_rate: Multiplicative cooling factor per step (0 < rate < 1).
        min_temperature: Temperature below which the system is "frozen".
        iterations_per_temp: Number of neighbour solutions to try at each
            temperature level.
    """

    name: str = "SimulatedAnnealing"
    description: str = (
        "Temperature-driven reasoning: explore wildly at high T, "
        "converge as the system cools."
    )

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.7,
        min_temperature: float = 0.05,
        iterations_per_temp: int = 2,
    ) -> None:
        """Initialise the Simulated Annealing engine.

        Args:
            initial_temperature: Starting temperature.  Values above 1.0
                encourage very wild exploration; values below 0.5 start
                more conservatively.
            cooling_rate: Factor by which temperature is multiplied each
                step.  Must be in ``(0, 1)``.  Lower values cool faster.
            min_temperature: When ``T`` falls below this the system is
                frozen and the best solution is returned.
            iterations_per_temp: How many neighbour solutions to generate
                and evaluate at each temperature level.
        """
        if not 0 < cooling_rate < 1:
            raise ValueError(f"cooling_rate must be in (0, 1), got {cooling_rate}")
        self.initial_temperature = max(0.01, initial_temperature)
        self.cooling_rate = cooling_rate
        self.min_temperature = max(0.001, min_temperature)
        self.iterations_per_temp = max(1, iterations_per_temp)

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
        """Execute the Simulated Annealing reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for generating and scoring
                solutions.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on the total number of temperature
                steps, regardless of the cooling schedule.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing initial, neighbour,
            acceptance, and refinement steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- generate and score initial solution
        current_solution = await self._generate_initial(query, llm_provider, trace)
        current_score = await self._evaluate_solution(
            query, current_solution, llm_provider, trace
        )

        best_solution = current_solution
        best_score = current_score

        # Record initial state
        init_step = self._make_step(
            step_type="initial",
            content=current_solution,
            score=current_score,
            metadata={
                "phase": "initial",
                "temperature": self.initial_temperature,
                "score": current_score,
            },
        )
        trace.add_step(init_step)

        # Phase 2 -- annealing loop
        temperature = self.initial_temperature
        temp_step = 0

        while temperature > self.min_temperature and temp_step < max_iterations:
            temp_step += 1

            for sub_iter in range(self.iterations_per_temp):
                # Generate a neighbour solution
                neighbour = await self._generate_neighbour(
                    query, current_solution, temperature, llm_provider, trace
                )
                neighbour_score = await self._evaluate_solution(
                    query, neighbour, llm_provider, trace
                )

                # Metropolis criterion
                accepted, accept_reason = self._metropolis_criterion(
                    current_score, neighbour_score, temperature
                )

                # Record the neighbour step
                nb_step = self._make_step(
                    step_type="neighbour",
                    content=neighbour,
                    score=neighbour_score,
                    metadata={
                        "phase": "explore",
                        "temp_step": temp_step,
                        "sub_iteration": sub_iter + 1,
                        "temperature": round(temperature, 6),
                        "current_score": round(current_score, 4),
                        "neighbour_score": round(neighbour_score, 4),
                        "accepted": accepted,
                        "reason": accept_reason,
                    },
                )
                trace.add_step(nb_step)

                if accepted:
                    current_solution = neighbour
                    current_score = neighbour_score

                    # Track global best
                    if current_score > best_score:
                        best_solution = current_solution
                        best_score = current_score

                        best_step = self._make_step(
                            step_type="new_best",
                            content=f"New best solution found (score={best_score:.4f})",
                            score=best_score,
                            metadata={
                                "phase": "new_best",
                                "temp_step": temp_step,
                                "temperature": round(temperature, 6),
                            },
                        )
                        trace.add_step(best_step)

            # Record cooling step
            old_temp = temperature
            temperature *= self.cooling_rate
            cool_step = self._make_step(
                step_type="cooling",
                content=(
                    f"Temperature cooled: {old_temp:.4f} -> {temperature:.4f} "
                    f"(current_score={current_score:.4f}, "
                    f"best_score={best_score:.4f})"
                ),
                score=current_score,
                metadata={
                    "phase": "cool",
                    "temp_step": temp_step,
                    "old_temperature": round(old_temp, 6),
                    "new_temperature": round(temperature, 6),
                    "current_score": round(current_score, 4),
                    "best_score": round(best_score, 4),
                },
            )
            trace.add_step(cool_step)

        # Phase 3 -- final refinement of the best solution
        final_output = await self._refine_best(
            query, best_solution, best_score, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Solution generation
    # ------------------------------------------------------------------

    async def _generate_initial(
        self,
        query: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the initial candidate solution.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The initial solution text.
        """
        prompt = (
            f"Generate a thorough solution or answer for the following query. "
            f"Be specific, concrete, and detailed.\n\n"
            f"Query: {query}"
        )
        return await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a creative problem solver.",
        )

    async def _generate_neighbour(
        self,
        query: str,
        current_solution: str,
        temperature: float,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Generate a neighbour solution by perturbing the current one.

        At high temperatures the perturbation is radical (rethink major
        assumptions); at low temperatures it is a minor refinement.

        Args:
            query: Original user query.
            current_solution: The solution to perturb.
            temperature: Current temperature -- controls perturbation magnitude.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            A new neighbour solution.
        """
        if temperature > 0.7:
            perturbation_instruction = (
                "RADICAL CHANGE: Rethink the fundamental approach. Challenge "
                "core assumptions. Explore a significantly different strategy "
                "or angle. The new solution can be very different from the "
                "current one."
            )
        elif temperature > 0.3:
            perturbation_instruction = (
                "MODERATE CHANGE: Keep the overall structure but modify key "
                "components. Try alternative methods for specific parts. "
                "Explore different trade-offs."
            )
        else:
            perturbation_instruction = (
                "MINOR REFINEMENT: Fine-tune the existing solution. Fix small "
                "issues, clarify ambiguities, strengthen weak points. The "
                "overall approach should remain the same."
            )

        prompt = (
            f"You are exploring variations of a solution. The current "
            f"temperature is {temperature:.3f} (1.0 = very exploratory, "
            f"0.0 = frozen).\n\n"
            f"Original query: {query}\n\n"
            f"Current solution:\n{current_solution}\n\n"
            f"Instruction: {perturbation_instruction}\n\n"
            f"Produce an alternative solution that follows the instruction above."
        )

        return await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a creative problem solver exploring the solution space.",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def _evaluate_solution(
        self,
        query: str,
        solution: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> float:
        """Score a solution on a 0.0-1.0 scale.

        The LLM evaluates the solution against multiple quality criteria
        (correctness, completeness, clarity, practicality, creativity)
        and returns a single aggregate score.

        Args:
            query: Original user query.
            solution: The solution to evaluate.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            A float score in ``[0.0, 1.0]``.
        """
        prompt = (
            f"Evaluate the following solution on a scale of 0.0 to 1.0.\n\n"
            f"Query: {query}\n\n"
            f"Solution:\n{solution}\n\n"
            f"Score on these criteria (0.0 to 1.0 each):\n"
            f"- Correctness: Is the solution accurate and free of errors?\n"
            f"- Completeness: Does it address all aspects of the query?\n"
            f"- Clarity: Is it well-structured and easy to understand?\n"
            f"- Practicality: Is it actionable and implementable?\n"
            f"- Creativity: Does it show insight or novel thinking?\n\n"
            f"Return ONLY a JSON object with the scores and an overall score:\n"
            f'{{"correctness": 0.8, "completeness": 0.7, "clarity": 0.9, '
            f'"practicality": 0.8, "creativity": 0.6, "overall": 0.76}}\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a rigorous evaluator. Return valid JSON only.",
        )

        return self._parse_score(raw)

    # ------------------------------------------------------------------
    # Metropolis criterion
    # ------------------------------------------------------------------

    def _metropolis_criterion(
        self,
        current_score: float,
        new_score: float,
        temperature: float,
    ) -> tuple[bool, str]:
        """Apply the Metropolis acceptance criterion.

        If the new solution is better (higher score), accept it
        unconditionally.  If worse, accept it with probability
        ``exp((new_score - current_score) / temperature)``.

        Args:
            current_score: Score of the current solution.
            new_score: Score of the candidate neighbour.
            temperature: Current temperature.

        Returns:
            Tuple of ``(accepted, reason_string)``.
        """
        delta = new_score - current_score

        if delta >= 0:
            return True, f"improvement (delta=+{delta:.4f})"

        # Worse solution -- stochastic acceptance
        if temperature <= 0:
            return False, f"frozen (delta={delta:.4f}, T=0)"

        accept_probability = math.exp(delta / temperature)
        roll = random.random()
        accepted = roll < accept_probability

        reason = (
            f"{'stochastic accept' if accepted else 'rejected'} "
            f"(delta={delta:.4f}, T={temperature:.4f}, "
            f"P={accept_probability:.4f}, roll={roll:.4f})"
        )
        return accepted, reason

    # ------------------------------------------------------------------
    # Final refinement
    # ------------------------------------------------------------------

    async def _refine_best(
        self,
        query: str,
        best_solution: str,
        best_score: float,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Polish the best solution found during annealing.

        The LLM is given the best solution and asked to refine it for
        clarity, completeness, and presentation without changing its core
        substance.

        Args:
            query: Original user query.
            best_solution: The highest-scoring solution from annealing.
            best_score: The score of the best solution.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The refined final answer.
        """
        prompt = (
            f"You are polishing the best solution found through an extensive "
            f"search process (score: {best_score:.4f}/1.0).\n\n"
            f"Original query: {query}\n\n"
            f"Best solution found:\n{best_solution}\n\n"
            f"Refine this solution for maximum clarity, completeness, and "
            f"quality. Fix any rough edges, improve structure, and ensure "
            f"all aspects of the query are addressed. Do not change the "
            f"core approach -- just polish it."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a meticulous editor and communicator.",
        )

        step = self._make_step(
            step_type="final_refinement",
            content=final,
            score=best_score,
            metadata={
                "phase": "refine",
                "best_score": round(best_score, 4),
            },
        )
        trace.add_step(step)
        return final

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract a float score from LLM evaluation output.

        Tries to parse JSON with an ``overall`` key first, then falls back
        to extracting any float from the text.

        Args:
            raw: Raw LLM output.

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        text = raw.strip()

        # Try JSON parsing
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    if "overall" in parsed:
                        return max(0.0, min(1.0, float(parsed["overall"])))
                    # Average all numeric values
                    nums = [float(v) for v in parsed.values() if isinstance(v, (int, float))]
                    if nums:
                        return max(0.0, min(1.0, sum(nums) / len(nums)))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: look for a float pattern
        import re
        matches = re.findall(r"\b0?\.\d+\b|\b1\.0\b|\b[01]\b", text)
        if matches:
            try:
                return max(0.0, min(1.0, float(matches[-1])))
            except ValueError:
                pass

        # Default middle-of-the-road score
        return 0.5


__all__ = ["SimulatedAnnealing"]
