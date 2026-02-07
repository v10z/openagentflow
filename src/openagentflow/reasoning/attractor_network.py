"""Attractor Network reasoning engine.

Implements attractor dynamics inspired by Hopfield networks (Hopfield 1982)
and attractor models of working memory and decision-making (Amit 1989;
Wang 2002; Machens, Romo & Brody 2005).

In recurrent neural circuits, activity patterns naturally evolve toward
stable configurations called *attractors*.  From any starting state within
an attractor's *basin of attraction*, the network dynamics will converge to
the same fixed point.  This has profound implications for decision-making:
the brain does not compute a single answer from a single starting point --
it launches noisy, variable trajectories from different initial conditions,
and the attractor that captures the most trajectories is the most robust
decision.

Key properties of attractor networks:
- **Multiple attractors** coexist in the same network, representing
  competing interpretations or solutions.
- **Basins of attraction** define how many initial conditions lead to a
  given attractor.  Larger basins indicate more robust solutions.
- **Convergence** is iterative: each step moves the state closer to the
  nearest attractor until a fixed point is reached.
- **Noise tolerance**: attractors are robust to perturbation -- small
  differences in initial conditions still converge to the same attractor.

Algorithm outline::

    1. INITIALIZATION  -- Launch N trajectories from different starting framings
    2. ITERATION       -- Each trajectory iteratively refines toward stability
    3. CONVERGENCE     -- Detect when trajectories have reached fixed points
    4. CLUSTERING      -- Group trajectories that converged to the same attractor
    5. BASIN ANALYSIS  -- The attractor with the largest basin wins
    6. SYNTHESIS       -- Extract and polish the winning attractor as the answer

Example::

    from openagentflow.reasoning.attractor_network import AttractorNetwork

    engine = AttractorNetwork(num_trajectories=5, max_refinement_steps=4)
    trace = await engine.reason(
        query="What is the best architecture for a real-time collaboration tool?",
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


class AttractorNetwork(ReasoningEngine):
    """Attractor dynamics for robust decision-making via trajectory convergence.

    The engine launches multiple reasoning trajectories from different
    starting framings of the problem.  Each trajectory iteratively refines
    toward a stable state (attractor).  Trajectories that converge to the
    same attractor are grouped, and the attractor with the largest basin
    of attraction (most convergent trajectories) is selected as the most
    robust answer.

    This mirrors the dynamics of Hopfield networks and cortical attractor
    circuits used for working memory and perceptual decision-making
    (Hopfield 1982; Wang 2002).

    Parameters:
        num_trajectories: Number of independent reasoning trajectories to
            launch.  More trajectories improve basin estimation but cost
            more LLM calls.
        max_refinement_steps: Maximum iterative refinement steps per
            trajectory before forced convergence check.
        convergence_threshold: Similarity score (0-1) above which two
            consecutive trajectory states are considered converged.
        similarity_threshold: Similarity score (0-1) above which two
            final trajectory states are considered to have reached the
            same attractor (for clustering).
        framing_temperature: Temperature for generating diverse initial
            framings.  Higher values yield more diverse starting points.
        refinement_temperature: Temperature for iterative refinement
            steps.  Lower values encourage convergence.

    Example::

        engine = AttractorNetwork(
            num_trajectories=6,
            max_refinement_steps=3,
            similarity_threshold=0.7,
        )
        trace = await engine.reason(
            query="How should we handle schema evolution in an event-sourced system?",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "attractor_network"
    description: str = (
        "Launches multiple reasoning trajectories from diverse framings, "
        "iteratively refines each toward a stable attractor, clusters "
        "converged trajectories, and selects the attractor with the "
        "largest basin of attraction."
    )

    def __init__(
        self,
        num_trajectories: int = 5,
        max_refinement_steps: int = 4,
        convergence_threshold: float = 0.85,
        similarity_threshold: float = 0.7,
        framing_temperature: float = 0.9,
        refinement_temperature: float = 0.3,
    ) -> None:
        self.num_trajectories = max(2, num_trajectories)
        self.max_refinement_steps = max(1, max_refinement_steps)
        self.convergence_threshold = max(0.0, min(1.0, convergence_threshold))
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.framing_temperature = max(0.0, min(1.0, framing_temperature))
        self.refinement_temperature = max(0.0, min(1.0, refinement_temperature))

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
        """Execute the full attractor network reasoning cycle.

        Args:
            query: The problem statement to solve.
            llm_provider: A ``BaseLLMProvider`` instance for all LLM calls.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Soft cap -- ``max_refinement_steps`` is capped
                at ``min(max_refinement_steps, max_iterations)``.
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` is the polished
            attractor with the largest basin of attraction.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_refine = min(self.max_refinement_steps, max_iterations)

        # Phase 1 -- INITIALIZATION: generate diverse starting framings
        framings = await self._generate_framings(query, llm_provider, trace)

        # Phase 2 -- ITERATION: refine each trajectory toward stability
        trajectories = await self._run_trajectories(
            query, framings, effective_refine, llm_provider, trace
        )

        # Phase 3 -- CLUSTERING: group trajectories by attractor similarity
        clusters = await self._cluster_attractors(
            query, trajectories, llm_provider, trace
        )

        # Phase 4 -- BASIN ANALYSIS: select the largest-basin attractor
        winning_cluster = self._select_largest_basin(clusters, trace)

        # Phase 5 -- SYNTHESIS: polish the winning attractor
        final_output = await self._synthesise_attractor(
            query, winning_cluster, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- INITIALIZATION
    # ------------------------------------------------------------------

    async def _generate_framings(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate diverse initial framings of the problem.

        Each framing represents a different 'initial condition' for the
        attractor dynamics -- a distinct perspective, methodology, or set
        of assumptions from which to approach the problem.

        Returns:
            List of dicts with ``id``, ``framing``, and ``step_id``.
        """
        n = self.num_trajectories
        system_prompt = (
            "You are generating diverse starting framings for a problem. "
            "Each framing should approach the problem from a RADICALLY "
            "different angle -- vary the methodology, core assumptions, "
            "perspective, or priority ordering. These are initial conditions "
            "for parallel reasoning trajectories."
        )
        user_prompt = (
            f"Generate exactly {n} diverse framings of the following problem. "
            f"Each framing should be a 2-3 sentence description of how to "
            f"approach the problem from a specific angle.\n\n"
            f"PROBLEM:\n{query}\n\n"
            f"Return a JSON array of {n} strings, each describing one "
            f"distinct framing.\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=self.framing_temperature,
        )

        framing_texts = self._parse_json_string_list(raw, n)

        init_step = self._make_step(
            step_type="initialization",
            content=f"Generated {len(framing_texts)} initial framings",
            metadata={
                "phase": "initialization",
                "num_trajectories": len(framing_texts),
            },
        )
        trace.add_step(init_step)

        framings: list[dict[str, Any]] = []
        for idx, text in enumerate(framing_texts):
            step = self._make_step(
                step_type="framing",
                content=text,
                score=0.0,
                metadata={
                    "phase": "initialization",
                    "trajectory_index": idx,
                },
                parent_step_id=init_step.step_id,
            )
            trace.add_step(step)
            framings.append({
                "id": f"traj-{idx}",
                "framing": text,
                "step_id": step.step_id,
            })

        logger.debug("AttractorNetwork: generated %d framings", len(framings))
        return framings

    # ------------------------------------------------------------------
    # Phase 2 -- ITERATION (trajectory refinement)
    # ------------------------------------------------------------------

    async def _run_trajectories(
        self,
        query: str,
        framings: list[dict[str, Any]],
        max_steps: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Run each trajectory through iterative refinement toward stability.

        Each trajectory starts from its framing and iteratively refines
        its answer.  At each step, the LLM is asked to improve the
        answer, and a self-similarity check determines whether the
        trajectory has converged (reached a fixed point / attractor).

        Returns:
            List of trajectory dicts with ``id``, ``framing``,
            ``final_state``, ``converged``, ``steps_taken``, and
            ``step_id``.
        """
        trajectories: list[dict[str, Any]] = []

        for framing in framings:
            current_state = ""
            converged = False
            steps_taken = 0
            last_step_id = framing["step_id"]

            # Generate initial solution from this framing
            init_prompt = (
                f"Solve the following problem using this specific approach:\n\n"
                f"PROBLEM: {query}\n\n"
                f"APPROACH: {framing['framing']}\n\n"
                f"Provide a complete solution following this approach."
            )

            current_state = await self._call_llm(
                provider,
                [{"role": "user", "content": init_prompt}],
                trace,
                system=(
                    "You are a focused problem solver. Follow the given "
                    "approach precisely."
                ),
                temperature=self.framing_temperature * 0.7,
            )

            traj_init_step = self._make_step(
                step_type="trajectory_init",
                content=current_state,
                score=0.0,
                metadata={
                    "phase": "iteration",
                    "trajectory": framing["id"],
                    "refinement_step": 0,
                },
                parent_step_id=last_step_id,
            )
            trace.add_step(traj_init_step)
            last_step_id = traj_init_step.step_id

            # Iterative refinement
            for step_idx in range(max_steps):
                refine_prompt = (
                    f"You are iteratively refining a solution toward stability. "
                    f"Review your current answer and improve it. Focus on making "
                    f"it more internally consistent, complete, and robust. Do NOT "
                    f"change the fundamental approach -- refine and strengthen it.\n\n"
                    f"PROBLEM: {query}\n\n"
                    f"CURRENT ANSWER:\n{current_state}\n\n"
                    f"Provide your refined answer. If the answer is already "
                    f"optimal and stable, reproduce it with minimal changes."
                )

                refined = await self._call_llm(
                    provider,
                    [{"role": "user", "content": refine_prompt}],
                    trace,
                    system=(
                        "You are refining a solution toward a stable fixed "
                        "point. Make improvements but preserve the core "
                        "approach. When the answer is stable, minimize changes."
                    ),
                    temperature=self.refinement_temperature,
                )

                steps_taken = step_idx + 1

                # Check convergence via self-similarity
                similarity = await self._check_similarity(
                    current_state, refined, provider, trace
                )

                refine_step = self._make_step(
                    step_type="trajectory_refine",
                    content=refined[:500] + "..." if len(refined) > 500 else refined,
                    score=similarity,
                    metadata={
                        "phase": "iteration",
                        "trajectory": framing["id"],
                        "refinement_step": step_idx + 1,
                        "similarity_to_previous": similarity,
                    },
                    parent_step_id=last_step_id,
                )
                trace.add_step(refine_step)
                last_step_id = refine_step.step_id

                current_state = refined

                if similarity >= self.convergence_threshold:
                    converged = True
                    conv_step = self._make_step(
                        step_type="convergence",
                        content=(
                            f"Trajectory {framing['id']} converged at step "
                            f"{step_idx + 1} (similarity={similarity:.3f} >= "
                            f"threshold={self.convergence_threshold})"
                        ),
                        score=similarity,
                        metadata={
                            "phase": "convergence",
                            "trajectory": framing["id"],
                            "step": step_idx + 1,
                        },
                        parent_step_id=last_step_id,
                    )
                    trace.add_step(conv_step)
                    last_step_id = conv_step.step_id
                    break

            trajectories.append({
                "id": framing["id"],
                "framing": framing["framing"],
                "final_state": current_state,
                "converged": converged,
                "steps_taken": steps_taken,
                "step_id": last_step_id,
            })

        logger.debug(
            "AttractorNetwork: %d/%d trajectories converged",
            sum(1 for t in trajectories if t["converged"]),
            len(trajectories),
        )
        return trajectories

    # ------------------------------------------------------------------
    # Phase 3 -- CLUSTERING
    # ------------------------------------------------------------------

    async def _cluster_attractors(
        self,
        query: str,
        trajectories: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Group trajectories that converged to the same attractor.

        The LLM evaluates pairwise similarity between all final
        trajectory states.  Trajectories above ``similarity_threshold``
        are grouped into the same cluster (attractor basin).

        Returns:
            List of cluster dicts, each with ``attractor_id``,
            ``trajectories`` (list of trajectory ids), ``representative``
            (the final state of the trajectory with highest convergence),
            and ``basin_size``.
        """
        n = len(trajectories)
        if n == 0:
            return []

        if n == 1:
            return [{
                "attractor_id": "attractor-0",
                "trajectories": [trajectories[0]["id"]],
                "representative": trajectories[0]["final_state"],
                "basin_size": 1,
            }]

        # Build pairwise similarity prompt
        summaries = []
        for t in trajectories:
            # Truncate for prompt efficiency
            summary = t["final_state"][:400]
            summaries.append(f"[{t['id']}]: {summary}")

        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        pair_descriptions = "\n".join(
            f"Pair ({trajectories[i]['id']}, {trajectories[j]['id']})"
            for i, j in pairs
        )

        system_prompt = (
            "You are evaluating similarity between solution proposals. "
            "Two solutions are 'similar' if they reach essentially the same "
            "conclusion via compatible reasoning, even if the wording differs. "
            "Return only valid JSON."
        )
        user_prompt = (
            f"Problem: \"{query}\"\n\n"
            f"Solution trajectories:\n" + "\n".join(summaries) + "\n\n"
            f"For each pair below, rate the similarity of their CONCLUSIONS "
            f"(not wording) on a scale of 0.0 (completely different solutions) "
            f"to 1.0 (essentially the same solution).\n\n"
            f"{pair_descriptions}\n\n"
            f"Return a JSON array of floats, one per pair, in the same order. "
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.2,
        )

        sim_scores = self._parse_json_float_list(raw, len(pairs))

        # Build similarity matrix
        sim_matrix: dict[tuple[int, int], float] = {}
        for idx, (i, j) in enumerate(pairs):
            score = max(0.0, min(1.0, sim_scores[idx]))
            sim_matrix[(i, j)] = score
            sim_matrix[(j, i)] = score

        # Simple greedy clustering: union-find style
        cluster_id = list(range(n))

        def find(x: int) -> int:
            while cluster_id[x] != x:
                cluster_id[x] = cluster_id[cluster_id[x]]
                x = cluster_id[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                cluster_id[ry] = rx

        for (i, j), score in sim_matrix.items():
            if i < j and score >= self.similarity_threshold:
                union(i, j)

        # Build cluster groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        clusters: list[dict[str, Any]] = []
        for cluster_idx, (root, members) in enumerate(groups.items()):
            # Pick the member with the best convergence as representative
            best_member = max(
                members,
                key=lambda m: (
                    1.0 if trajectories[m]["converged"] else 0.0,
                    trajectories[m]["steps_taken"],
                ),
            )
            clusters.append({
                "attractor_id": f"attractor-{cluster_idx}",
                "trajectories": [trajectories[m]["id"] for m in members],
                "representative": trajectories[best_member]["final_state"],
                "basin_size": len(members),
            })

        # Record clustering step
        cluster_step = self._make_step(
            step_type="clustering",
            content=(
                f"Clustered {n} trajectories into {len(clusters)} attractors. "
                f"Basin sizes: {[c['basin_size'] for c in clusters]}"
            ),
            metadata={
                "phase": "clustering",
                "num_trajectories": n,
                "num_attractors": len(clusters),
                "basin_sizes": {
                    c["attractor_id"]: c["basin_size"] for c in clusters
                },
                "similarity_threshold": self.similarity_threshold,
            },
        )
        trace.add_step(cluster_step)

        # Sort clusters by basin size (largest first)
        clusters.sort(key=lambda c: c["basin_size"], reverse=True)

        logger.debug(
            "AttractorNetwork: %d attractors, largest basin = %d",
            len(clusters),
            clusters[0]["basin_size"] if clusters else 0,
        )
        return clusters

    # ------------------------------------------------------------------
    # Phase 4 -- BASIN ANALYSIS
    # ------------------------------------------------------------------

    def _select_largest_basin(
        self,
        clusters: list[dict[str, Any]],
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Select the attractor with the largest basin of attraction.

        If multiple attractors have the same basin size, the first one
        (which was the first to form) is selected.

        Returns:
            The winning cluster dict.
        """
        if not clusters:
            return {
                "attractor_id": "attractor-fallback",
                "trajectories": [],
                "representative": "",
                "basin_size": 0,
            }

        winner = clusters[0]  # Already sorted by basin_size descending

        basin_step = self._make_step(
            step_type="basin_analysis",
            content=(
                f"Attractor '{winner['attractor_id']}' has the largest "
                f"basin of attraction with {winner['basin_size']} converging "
                f"trajectories ({', '.join(winner['trajectories'])}). "
                f"Total attractors: {len(clusters)}."
            ),
            score=winner["basin_size"] / max(self.num_trajectories, 1),
            metadata={
                "phase": "basin_analysis",
                "winning_attractor": winner["attractor_id"],
                "basin_size": winner["basin_size"],
                "total_attractors": len(clusters),
                "all_basins": {
                    c["attractor_id"]: c["basin_size"] for c in clusters
                },
            },
        )
        trace.add_step(basin_step)
        return winner

    # ------------------------------------------------------------------
    # Phase 5 -- SYNTHESIS
    # ------------------------------------------------------------------

    async def _synthesise_attractor(
        self,
        query: str,
        winning_cluster: dict[str, Any],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Polish and present the winning attractor as the final answer.

        The representative trajectory from the winning basin is refined
        into a clean, comprehensive answer.  Metadata about the basin
        size and convergence dynamics is included for transparency.

        Returns:
            The final polished answer text.
        """
        system_prompt = (
            "You are producing a final, polished answer from an attractor "
            "network analysis. The answer below represents the most robust "
            "solution -- the one that multiple independent reasoning "
            "trajectories converged to from different starting points. "
            "Polish it into a clear, comprehensive, well-structured response."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"WINNING ATTRACTOR (basin size: {winning_cluster['basin_size']} "
            f"out of {self.num_trajectories} trajectories):\n"
            f"{winning_cluster['representative']}\n\n"
            f"Polish this into a final answer. The fact that {winning_cluster['basin_size']} "
            f"independent reasoning trajectories converged to this solution "
            f"from different starting framings indicates it is robust. "
            f"Preserve the core conclusions while improving clarity and "
            f"completeness."
        )

        final = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        synth_step = self._make_step(
            step_type="synthesis",
            content=final,
            score=1.0,
            metadata={
                "phase": "synthesis",
                "winning_attractor": winning_cluster["attractor_id"],
                "basin_size": winning_cluster["basin_size"],
                "total_trajectories": self.num_trajectories,
            },
        )
        trace.add_step(synth_step)
        logger.debug("AttractorNetwork: synthesis complete")
        return final

    # ------------------------------------------------------------------
    # Utility: similarity checking
    # ------------------------------------------------------------------

    async def _check_similarity(
        self,
        state_a: str,
        state_b: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> float:
        """Ask the LLM to rate the similarity between two solution states.

        Used both for convergence detection (is a trajectory stable?) and
        for attractor clustering (did two trajectories reach the same
        attractor?).

        Returns:
            A float in [0.0, 1.0] where 1.0 means identical conclusions.
        """
        system_prompt = (
            "You are a similarity evaluator. Compare two solution states and "
            "rate their similarity. Focus on whether they reach the SAME "
            "conclusion, not whether the wording is identical. Respond with "
            "ONLY a JSON object."
        )
        user_prompt = (
            f"STATE A:\n{state_a[:600]}\n\n"
            f"STATE B:\n{state_b[:600]}\n\n"
            f"Rate the similarity of the conclusions on a 0.0-1.0 scale.\n"
            f"Return ONLY: {{\"similarity\": <float>}}"
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.1,
        )

        return self._extract_similarity_score(raw)

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_string_list(raw: str, expected: int) -> list[str]:
        """Parse a JSON array of strings from LLM output.

        Falls back to line-based splitting if JSON parsing fails.
        Pads with generic entries if fewer than expected items are found.

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
                        result.append(
                            f"Alternative framing #{len(result) + 1}: "
                            f"approach the problem from a different angle"
                        )
                    return result[:expected]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: split by newlines
        lines = [
            ln.strip().lstrip("0123456789.-) ")
            for ln in cleaned.splitlines()
            if ln.strip()
        ]
        lines = [ln for ln in lines if ln]
        while len(lines) < expected:
            lines.append(
                f"Alternative framing #{len(lines) + 1}: "
                f"approach from a novel perspective"
            )
        return lines[:expected]

    @staticmethod
    def _parse_json_float_list(raw: str, expected: int) -> list[float]:
        """Parse a JSON array of floats from LLM output.

        Falls back to 0.5 (uncertain) if parsing fails.

        Returns:
            List of floats of length ``expected``.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, list):
                    result: list[float] = []
                    for v in parsed:
                        try:
                            result.append(float(v))
                        except (TypeError, ValueError):
                            result.append(0.5)
                    while len(result) < expected:
                        result.append(0.5)
                    return result[:expected]
            except (json.JSONDecodeError, ValueError):
                pass

        return [0.5] * expected

    @staticmethod
    def _extract_similarity_score(raw: str) -> float:
        """Extract a similarity float from LLM output.

        Tries to parse a JSON object with a ``similarity`` key.  Falls
        back to regex extraction of any float, then to 0.5 as default.

        Returns:
            Float in [0.0, 1.0].
        """
        # Try JSON parse
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, dict) and "similarity" in parsed:
                    return max(0.0, min(1.0, float(parsed["similarity"])))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: find any float in the text
        match = re.search(r"(\d+\.?\d*)", cleaned)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    return val
                if 0 <= val <= 100:
                    return val / 100.0
            except (ValueError, TypeError):
                pass

        return 0.5


__all__ = ["AttractorNetwork"]
