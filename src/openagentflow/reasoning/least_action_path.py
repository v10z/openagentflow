"""Least Action Path reasoning engine.

Based on Hamilton's principle of least action and Feynman's path integral
formulation.  Instead of reasoning forward step-by-step, the engine defines
boundary conditions (start state and desired end state), generates multiple
complete reasoning paths connecting them, evaluates each with a global cost
functional (the "action"), and variationally optimises the path of least
action.

Physics basis::

    S[q(t)] = integral_{t1}^{t2} L(q, dq/dt, t) dt

    where L = T - V  (kinetic energy minus potential energy)

The actual physical path satisfies the Euler-Lagrange equation::

    d/dt (dL/dq_dot) - dL/dq = 0

Key insight: the correct path is the one that makes the action *stationary*
(delta S = 0).  Nature "explores all paths" and selects the optimal one.

The mapping to reasoning: the "action" balances reasoning effort (kinetic
cost -- how much the reasoning changes between steps) against quality loss
(potential cost -- how far the current state is from optimal).

Example::

    from openagentflow.reasoning.least_action_path import LeastActionPath

    engine = LeastActionPath(num_paths=4)
    trace = await engine.reason(
        query="Design the most elegant API for a configuration library.",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# Default path strategy configurations
_DEFAULT_PATH_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "shortest",
        "instruction": (
            "Take the SHORTEST path: minimum number of reasoning steps, "
            "direct approach. Go straight from the initial state to the "
            "target with no detours."
        ),
        "temperature": 0.3,
    },
    {
        "name": "scenic",
        "instruction": (
            "Take the SCENIC path: explore broadly before converging. "
            "Consider multiple angles, draw analogies, explore related "
            "ideas, then synthesise them into the target state."
        ),
        "temperature": 0.8,
    },
    {
        "name": "conservative",
        "instruction": (
            "Take the CONSERVATIVE path: smallest changes at each step, "
            "safest reasoning. Each step should follow directly and "
            "obviously from the previous one. Minimise leaps of logic."
        ),
        "temperature": 0.3,
    },
    {
        "name": "bold",
        "instruction": (
            "Take the BOLD path: large leaps of reasoning, maximum "
            "creativity. Make daring connections, challenge assumptions, "
            "and reach the target through unexpected routes."
        ),
        "temperature": 0.8,
    },
]


class LeastActionPath(ReasoningEngine):
    """Variational optimisation of reasoning paths via the action principle.

    The engine works in five phases:

    1. **BOUNDARY CONDITIONS** -- Define the initial state (what we know) and
       the target state (what a complete answer looks like), plus quality
       criteria the path must optimise.
    2. **PATH GENERATION** -- Generate M candidate reasoning paths, each
       following a different strategy (shortest, scenic, conservative, bold).
       Each path is a complete trajectory from start to target.
    3. **ACTION EVALUATION** -- Score each path's "action" by balancing
       kinetic cost (reasoning effort, number of direction changes) against
       potential cost (quality of the final state relative to the target).
       The path with lowest action is optimal.
    4. **EULER-LAGRANGE REFINEMENT** -- Refine the optimal path using the
       variational principle: at each step, ask whether a local modification
       could reduce the total action.
    5. **STATIONARY POINT VERIFICATION** -- Verify that no small perturbation
       to any step reduces the action; present the final answer.

    Attributes:
        name: ``"LeastActionPath"``
        description: Short human-readable summary.
        num_paths: Number of candidate reasoning paths to generate.
        path_strategies: List of strategy configs for path generation.
        kinetic_weight: Weight of reasoning effort in the action functional.
        potential_weight: Weight of solution quality in the action functional.
        refinement_temperature: Temperature for Euler-Lagrange refinement.
        enable_verification: Whether to perform stationary-point verification.
    """

    name: str = "LeastActionPath"
    description: str = (
        "Generates multiple reasoning paths, evaluates their "
        "'action' (effort vs quality), and variationally "
        "optimises the path of least action."
    )

    def __init__(
        self,
        num_paths: int = 4,
        path_strategies: list[dict[str, Any]] | None = None,
        kinetic_weight: float = 0.4,
        potential_weight: float = 0.6,
        refinement_temperature: float = 0.4,
        enable_verification: bool = True,
    ) -> None:
        """Initialise the Least Action Path engine.

        Args:
            num_paths: Number of candidate reasoning paths to generate.
                Each path costs one LLM call.
            path_strategies: Optional list of strategy dicts, each with keys
                ``name``, ``instruction``, and ``temperature``.  Defaults to
                shortest / scenic / conservative / bold.
            kinetic_weight: Weight of effort (kinetic cost) in the action
                functional.  Higher values favour simpler reasoning paths.
            potential_weight: Weight of quality (potential cost) in the action
                functional.  Higher values favour better answers even if
                reasoning is complex.
            refinement_temperature: Temperature for the Euler-Lagrange
                refinement LLM call.
            enable_verification: If ``True``, a final perturbation test
                verifies the path is truly stationary.
        """
        self.num_paths = max(2, num_paths)
        self.path_strategies = path_strategies or _DEFAULT_PATH_STRATEGIES
        # Ensure we have enough strategies for num_paths
        while len(self.path_strategies) < self.num_paths:
            extra_idx = len(self.path_strategies) % len(_DEFAULT_PATH_STRATEGIES)
            self.path_strategies.append(_DEFAULT_PATH_STRATEGIES[extra_idx])
        self.kinetic_weight = max(0.0, kinetic_weight)
        self.potential_weight = max(0.0, potential_weight)
        self.refinement_temperature = refinement_temperature
        self.enable_verification = enable_verification

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
        """Execute the Least Action Path reasoning strategy.

        Args:
            query: The user question or problem to reason about.
            llm_provider: An LLM provider for generation and evaluation.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Hard cap on total LLM calls.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with boundary, path, action-evaluation,
            refinement, and verification steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- BOUNDARY CONDITIONS
        boundaries = await self._define_boundaries(query, llm_provider, trace)
        boundary_step = self._make_step(
            step_type="boundary_conditions",
            content=(
                f"Initial state: {boundaries.get('initial_state', 'N/A')}\n\n"
                f"Target state: {boundaries.get('target_state', 'N/A')}\n\n"
                f"Quality criteria: "
                f"{', '.join(boundaries.get('quality_criteria', []))}"
            ),
            score=0.0,
            metadata={"phase": "boundary_conditions", **boundaries},
        )
        trace.add_step(boundary_step)

        # Phase 2 -- PATH GENERATION
        effective_paths = min(self.num_paths, max_iterations - 1)
        paths = await self._generate_paths(
            query, boundaries, effective_paths, llm_provider, trace
        )

        # Phase 3 -- ACTION EVALUATION
        action_results = await self._evaluate_actions(
            query, boundaries, paths, llm_provider, trace
        )

        # Find the path with lowest action
        best_path_idx = 0
        best_action = float("inf")
        for idx, result in enumerate(action_results):
            if result.get("total_action", float("inf")) < best_action:
                best_action = result["total_action"]
                best_path_idx = idx

        best_path = paths[best_path_idx] if paths else {"content": ""}
        best_result = (
            action_results[best_path_idx] if action_results else {}
        )

        logger.debug(
            "LeastActionPath: best path is '%s' (action=%.4f)",
            best_path.get("strategy", "unknown"),
            best_action,
        )

        # Phase 4 -- EULER-LAGRANGE REFINEMENT
        refined = await self._euler_lagrange_refine(
            query, boundaries, best_path, best_result, llm_provider, trace
        )

        # Phase 5 -- STATIONARY POINT VERIFICATION (optional)
        if self.enable_verification and trace.total_llm_calls < max_iterations:
            final_output = await self._verify_stationary(
                query, boundaries, refined, llm_provider, trace
            )
        else:
            final_output = refined

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- BOUNDARY CONDITIONS
    # ------------------------------------------------------------------

    async def _define_boundaries(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Define initial state, target state, and quality criteria.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``initial_state``, ``target_state``, and
            ``quality_criteria`` keys.
        """
        prompt = (
            f"Define the boundary conditions for reasoning about this problem.\n\n"
            f"Problem: {query}\n\n"
            f"Return a JSON object with exactly these keys:\n"
            f'- "initial_state": What we know at the start (given information, '
            f"constraints, context)\n"
            f'- "target_state": What a complete, high-quality answer looks like '
            f"(desired properties, format, depth)\n"
            f'- "quality_criteria": An array of 3-5 specific criteria the answer '
            f"must optimise\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are defining the variational boundary conditions for "
                "a reasoning problem. Be precise and concrete."
            ),
            temperature=0.4,
        )

        return self._parse_boundaries(raw)

    # ------------------------------------------------------------------
    # Phase 2 -- PATH GENERATION
    # ------------------------------------------------------------------

    async def _generate_paths(
        self,
        query: str,
        boundaries: dict[str, Any],
        num_paths: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate candidate reasoning paths.

        Each path follows a different strategy and produces a complete
        reasoning trajectory from initial to target state.

        Args:
            query: Original user query.
            boundaries: Boundary conditions from phase 1.
            num_paths: Number of paths to generate.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of path dicts with ``strategy``, ``content``, and
            ``step_id`` keys.
        """
        initial = boundaries.get("initial_state", "the problem as given")
        target = boundaries.get("target_state", "a complete answer")
        criteria = boundaries.get("quality_criteria", [])
        criteria_str = "\n".join(f"  - {c}" for c in criteria)

        paths: list[dict[str, Any]] = []

        for idx in range(num_paths):
            strategy = self.path_strategies[idx]
            strategy_name = strategy.get("name", f"path_{idx}")
            instruction = strategy.get("instruction", "Reason freely.")
            temperature = strategy.get("temperature", 0.5)

            prompt = (
                f"You are generating a complete reasoning path from initial "
                f"state to target state.\n\n"
                f"Problem: {query}\n\n"
                f"Initial state: {initial}\n\n"
                f"Target state: {target}\n\n"
                f"Quality criteria:\n{criteria_str}\n\n"
                f"Path strategy: {strategy_name.upper()}\n"
                f"{instruction}\n\n"
                f"Present your reasoning as a numbered sequence of steps, "
                f"from the initial state to the final answer. Each step "
                f"should clearly follow from the previous one. End with "
                f"the complete answer that satisfies the target state."
            )

            raw = await self._call_llm(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                trace=trace,
                system=(
                    f"You are following the {strategy_name} reasoning path. "
                    f"Produce a complete trajectory with explicit steps."
                ),
                temperature=temperature,
            )

            path_step = self._make_step(
                step_type="candidate_path",
                content=raw,
                score=0.0,
                metadata={
                    "phase": "path_generation",
                    "strategy": strategy_name,
                    "path_index": idx,
                    "temperature": temperature,
                },
            )
            trace.add_step(path_step)

            paths.append({
                "strategy": strategy_name,
                "content": raw,
                "step_id": path_step.step_id,
                "index": idx,
            })

        return paths

    # ------------------------------------------------------------------
    # Phase 3 -- ACTION EVALUATION
    # ------------------------------------------------------------------

    async def _evaluate_actions(
        self,
        query: str,
        boundaries: dict[str, Any],
        paths: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Evaluate the action (cost functional) of each path.

        The action balances kinetic cost (reasoning effort) against potential
        cost (quality of the final state).

        Args:
            query: Original user query.
            boundaries: Boundary conditions.
            paths: Candidate paths from phase 2.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of action evaluation dicts per path.
        """
        target = boundaries.get("target_state", "a complete answer")
        criteria = boundaries.get("quality_criteria", [])
        criteria_str = "\n".join(f"  - {c}" for c in criteria)

        path_summaries = []
        for p in paths:
            # Truncate very long paths for the evaluation prompt
            content_preview = p["content"][:1500]
            if len(p["content"]) > 1500:
                content_preview += "\n... [truncated]"
            path_summaries.append(
                f"--- PATH {p['index']}: {p['strategy'].upper()} ---\n"
                f"{content_preview}"
            )

        paths_text = "\n\n".join(path_summaries)

        prompt = (
            f"Evaluate the 'action' of each reasoning path. The action "
            f"balances two costs:\n\n"
            f"1. KINETIC COST (weight={self.kinetic_weight:.2f}): How much "
            f"reasoning effort was expended? Count direction changes, "
            f"unnecessary detours, and complexity. 0.0 = effortless, "
            f"1.0 = extremely laborious.\n\n"
            f"2. POTENTIAL COST (weight={self.potential_weight:.2f}): How far "
            f"is the final state from the target? Does it satisfy the quality "
            f"criteria? 0.0 = perfect match, 1.0 = completely misses the "
            f"target.\n\n"
            f"TOTAL ACTION = {self.kinetic_weight:.2f} * kinetic_cost + "
            f"{self.potential_weight:.2f} * potential_cost\n"
            f"(lower action is better)\n\n"
            f"Problem: {query}\n"
            f"Target state: {target}\n"
            f"Quality criteria:\n{criteria_str}\n\n"
            f"Paths to evaluate:\n\n{paths_text}\n\n"
            f"Return a JSON array with one object per path, in order:\n"
            f'[{{"path_index": 0, "kinetic_cost": 0.X, "potential_cost": 0.X, '
            f'"total_action": 0.X, "reasoning": "..."}}, ...]\n'
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are evaluating the action functional of reasoning paths. "
                "Be rigorous and quantitative. Return valid JSON only."
            ),
            temperature=0.3,
        )

        results = self._parse_action_results(raw, len(paths))

        # Record the evaluation step
        eval_step = self._make_step(
            step_type="action_evaluation",
            content=(
                "Action evaluation:\n"
                + "\n".join(
                    f"  Path {r.get('path_index', i)} ({paths[i]['strategy']}): "
                    f"action={r.get('total_action', 'N/A'):.4f} "
                    f"(K={r.get('kinetic_cost', 'N/A'):.3f}, "
                    f"V={r.get('potential_cost', 'N/A'):.3f})"
                    for i, r in enumerate(results)
                )
            ),
            score=1.0,
            metadata={
                "phase": "action_evaluation",
                "results": results,
                "best_path_index": min(
                    range(len(results)),
                    key=lambda i: results[i].get("total_action", float("inf")),
                ) if results else 0,
            },
        )
        trace.add_step(eval_step)

        return results

    # ------------------------------------------------------------------
    # Phase 4 -- EULER-LAGRANGE REFINEMENT
    # ------------------------------------------------------------------

    async def _euler_lagrange_refine(
        self,
        query: str,
        boundaries: dict[str, Any],
        best_path: dict[str, Any],
        action_result: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Refine the optimal path using variational optimisation.

        At each step along the path, asks whether a local modification
        could reduce the total action.

        Args:
            query: Original user query.
            boundaries: Boundary conditions.
            best_path: The lowest-action path from phase 3.
            action_result: Action evaluation of the best path.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The refined answer text.
        """
        target = boundaries.get("target_state", "a complete answer")
        criteria = boundaries.get("quality_criteria", [])
        criteria_str = "\n".join(f"  - {c}" for c in criteria)
        strategy = best_path.get("strategy", "unknown")
        kinetic = action_result.get("kinetic_cost", 0.5)
        potential = action_result.get("potential_cost", 0.5)
        action = action_result.get("total_action", 1.0)

        prompt = (
            f"Take the lowest-action reasoning path and refine it using the "
            f"Euler-Lagrange variational principle.\n\n"
            f"Problem: {query}\n"
            f"Target state: {target}\n"
            f"Quality criteria:\n{criteria_str}\n\n"
            f"Best path (strategy: {strategy}, action={action:.4f}, "
            f"kinetic={kinetic:.3f}, potential={potential:.3f}):\n"
            f"{best_path.get('content', '')}\n\n"
            f"For each step along the path, ask:\n"
            f"  - Could this step be improved to reduce the total action?\n"
            f"  - Is there unnecessary effort (kinetic cost) that can be "
            f"removed?\n"
            f"  - Is there quality loss (potential cost) that can be "
            f"recovered?\n\n"
            f"Produce the refined, optimised answer. The optimal path is the "
            f"one where no local modification can reduce the total action."
        )

        refined = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are applying Euler-Lagrange variational optimisation. "
                "Refine the reasoning path to minimise total action: "
                "remove unnecessary complexity, recover quality where possible."
            ),
            temperature=self.refinement_temperature,
        )

        el_step = self._make_step(
            step_type="euler_lagrange",
            content=refined,
            score=action,
            metadata={
                "phase": "euler_lagrange",
                "source_strategy": strategy,
                "original_action": round(action, 4),
            },
        )
        trace.add_step(el_step)

        return refined

    # ------------------------------------------------------------------
    # Phase 5 -- STATIONARY POINT VERIFICATION
    # ------------------------------------------------------------------

    async def _verify_stationary(
        self,
        query: str,
        boundaries: dict[str, Any],
        refined_answer: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Verify the refined path is a true stationary point of the action.

        Tests whether small perturbations to any reasoning step increase the
        action.  If a reduction is found, applies it.

        Args:
            query: Original user query.
            boundaries: Boundary conditions.
            refined_answer: The Euler-Lagrange refined answer.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The verified (and possibly adjusted) final answer.
        """
        target = boundaries.get("target_state", "a complete answer")
        criteria = boundaries.get("quality_criteria", [])
        criteria_str = "\n".join(f"  - {c}" for c in criteria)

        prompt = (
            f"Verify that the following answer represents a true stationary "
            f"point of the action functional.\n\n"
            f"Problem: {query}\n"
            f"Target state: {target}\n"
            f"Quality criteria:\n{criteria_str}\n\n"
            f"Refined answer:\n{refined_answer}\n\n"
            f"Perturbation test: For each major reasoning step in the answer, "
            f"consider a small modification. Does the modification:\n"
            f"  (a) Reduce effort without sacrificing quality? If so, apply it.\n"
            f"  (b) Improve quality without excessive effort? If so, apply it.\n"
            f"  (c) Neither? Then the step is already optimal.\n\n"
            f"Present the final, fully optimised answer. If no perturbation "
            f"improves the action, confirm the answer is at a true stationary "
            f"point and present it as-is."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are performing a stationary-point verification. "
                "The optimal answer is one where no small change improves it."
            ),
            temperature=0.3,
        )

        ver_step = self._make_step(
            step_type="stationary_verification",
            content=final,
            score=1.0,
            metadata={"phase": "stationary_verification"},
        )
        trace.add_step(ver_step)

        return final

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_boundaries(raw: str) -> dict[str, Any]:
        """Parse boundary conditions JSON from LLM output.

        Falls back to sensible defaults if JSON parsing fails.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``initial_state``, ``target_state``, and
            ``quality_criteria``.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return {
                        "initial_state": str(
                            parsed.get("initial_state", "problem as given")
                        ),
                        "target_state": str(
                            parsed.get("target_state", "a complete answer")
                        ),
                        "quality_criteria": [
                            str(c) for c in parsed.get("quality_criteria", [])
                        ] or ["correctness", "clarity", "completeness"],
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: extract from text heuristically
        return {
            "initial_state": "the problem as stated",
            "target_state": "a comprehensive, well-structured answer",
            "quality_criteria": [
                "correctness",
                "clarity",
                "completeness",
                "practicality",
            ],
        }

    @staticmethod
    def _parse_action_results(
        raw: str, expected: int
    ) -> list[dict[str, Any]]:
        """Parse action evaluation JSON from LLM output.

        Falls back to equal-action defaults if parsing fails.

        Args:
            raw: Raw LLM output.
            expected: Expected number of path evaluations.

        Returns:
            List of action result dicts.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    results = []
                    for idx, item in enumerate(parsed):
                        if isinstance(item, dict):
                            kinetic = max(
                                0.0,
                                min(1.0, float(item.get("kinetic_cost", 0.5))),
                            )
                            potential = max(
                                0.0,
                                min(
                                    1.0, float(item.get("potential_cost", 0.5))
                                ),
                            )
                            total = float(
                                item.get(
                                    "total_action", kinetic + potential
                                )
                            )
                            results.append({
                                "path_index": int(
                                    item.get("path_index", idx)
                                ),
                                "kinetic_cost": kinetic,
                                "potential_cost": potential,
                                "total_action": total,
                                "reasoning": str(
                                    item.get("reasoning", "")
                                ),
                            })
                    # Pad if needed
                    while len(results) < expected:
                        results.append({
                            "path_index": len(results),
                            "kinetic_cost": 0.5,
                            "potential_cost": 0.5,
                            "total_action": 0.5,
                            "reasoning": "default evaluation",
                        })
                    return results[:expected]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: assign decreasing action to give preference to first path
        return [
            {
                "path_index": i,
                "kinetic_cost": 0.5,
                "potential_cost": 0.5,
                "total_action": 0.5 + i * 0.05,
                "reasoning": "fallback evaluation",
            }
            for i in range(expected)
        ]


__all__ = ["LeastActionPath"]
