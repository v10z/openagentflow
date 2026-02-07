"""Hippocampal Replay reasoning engine.

Implements memory consolidation through replay, inspired by the hippocampal
sharp-wave ripple (SWR) replay phenomenon discovered by Wilson & McNaughton
(1994).  During sleep and quiet wakefulness, hippocampal place cells reactivate
in compressed temporal sequences that recapitulate recent experience -- both in
the original (forward) order and, critically, in *reverse* (Foster & Wilson
2006).  This bidirectional replay is believed to serve memory consolidation,
credit assignment (backward replay links outcomes to earlier decisions), and
counterfactual evaluation (Olafsdottir, Bush & Barry 2018).

The engine translates this into a reasoning strategy:

1. **EXPERIENCE** -- Generate an initial solution (the "waking experience").
2. **FORWARD REPLAY** -- Compress and replay the reasoning chain forward,
   identifying the logical flow and key decision points.
3. **REVERSE REPLAY** -- Replay backward from the conclusion to the premises,
   performing credit assignment: which earlier decisions most influenced the
   final answer?
4. **PATTERN EXTRACTION** -- Distill recurring themes and structural patterns
   from both replay directions.
5. **COUNTERFACTUAL REPLAY** -- At the most influential decision points,
   replay with alternative choices to explore what would have changed.
6. **RECONSOLIDATION** -- Synthesise the original answer, replay insights,
   and counterfactual discoveries into a strengthened final answer.

This approach is especially effective for complex multi-step problems where
the reasoning chain is long and early decisions have non-obvious downstream
effects.

Example::

    from openagentflow.reasoning.hippocampal_replay import HippocampalReplay

    engine = HippocampalReplay(
        num_counterfactuals=3,
        replay_compression=0.5,
    )
    trace = await engine.reason(
        query="Design a migration strategy for a monolith to microservices.",
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


class HippocampalReplay(ReasoningEngine):
    """Bidirectional replay and counterfactual evaluation for memory consolidation.

    The engine generates an initial solution, then "replays" the reasoning
    both forward (compressed temporal sequence) and backward (credit
    assignment from conclusion to premises).  Patterns extracted from both
    directions seed counterfactual replays at critical decision points.
    The final answer reconsolidates insights from all replay passes.

    This mirrors the hippocampal sharp-wave ripple replay observed during
    sleep and quiet wakefulness (Wilson & McNaughton 1994; Foster & Wilson
    2006), which the brain uses for memory consolidation, credit
    assignment, and planning.

    Parameters:
        num_counterfactuals: Number of critical decision points to replay
            with alternative choices.  Higher values explore more
            alternatives but cost more LLM calls.
        replay_compression: How aggressively to compress the reasoning
            chain during forward replay (0.0 = no compression, 1.0 =
            maximum compression).  Moderate values (0.3-0.6) balance
            detail retention with pattern visibility.
        consolidation_temperature: Temperature for the final reconsolidation
            synthesis.  Lower values produce more conservative answers.
        experience_temperature: Temperature for the initial solution
            generation.  Higher values encourage more detailed reasoning
            chains (more material to replay).
        max_decision_points: Cap on how many decision points the reverse
            replay can identify, preventing runaway analysis.
        pattern_threshold: Minimum relevance score (0-1) for a pattern
            extracted from replays to be included in reconsolidation.

    Example::

        engine = HippocampalReplay(num_counterfactuals=2)
        trace = await engine.reason(
            query="What is the optimal data partitioning strategy for a "
                  "globally distributed database?",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "hippocampal_replay"
    description: str = (
        "Bidirectional replay (forward + reverse) with counterfactual "
        "evaluation at critical decision points, inspired by hippocampal "
        "sharp-wave ripple replay for memory consolidation."
    )

    def __init__(
        self,
        num_counterfactuals: int = 3,
        replay_compression: float = 0.5,
        consolidation_temperature: float = 0.4,
        experience_temperature: float = 0.6,
        max_decision_points: int = 5,
        pattern_threshold: float = 0.3,
    ) -> None:
        self.num_counterfactuals = max(1, num_counterfactuals)
        self.replay_compression = max(0.0, min(1.0, replay_compression))
        self.consolidation_temperature = max(0.0, min(1.0, consolidation_temperature))
        self.experience_temperature = max(0.0, min(1.0, experience_temperature))
        self.max_decision_points = max(1, max_decision_points)
        self.pattern_threshold = max(0.0, min(1.0, pattern_threshold))

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
        """Execute the full hippocampal replay reasoning cycle.

        Args:
            query: The problem statement to solve.
            llm_provider: A ``BaseLLMProvider`` instance for all LLM calls.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Soft cap on total replay iterations.
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` is the
            reconsolidated answer enriched by bidirectional replay and
            counterfactual analysis.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- EXPERIENCE: generate an initial detailed solution
        experience_step_id, experience_content = await self._generate_experience(
            query, llm_provider, trace
        )

        # Phase 2 -- FORWARD REPLAY: compressed forward pass
        forward_step_id, forward_content = await self._forward_replay(
            query, experience_content, llm_provider, trace, experience_step_id
        )

        # Phase 3 -- REVERSE REPLAY: backward credit assignment
        reverse_step_id, decision_points = await self._reverse_replay(
            query, experience_content, llm_provider, trace, experience_step_id
        )

        # Phase 4 -- PATTERN EXTRACTION from both replays
        patterns_step_id, patterns = await self._extract_patterns(
            query, forward_content, decision_points,
            llm_provider, trace, forward_step_id, reverse_step_id,
        )

        # Phase 5 -- COUNTERFACTUAL REPLAY at critical decision points
        counterfactual_insights = await self._counterfactual_replays(
            query, experience_content, decision_points,
            llm_provider, trace, reverse_step_id,
        )

        # Phase 6 -- RECONSOLIDATION: synthesise everything
        final_output = await self._reconsolidate(
            query, experience_content, patterns, counterfactual_insights,
            llm_provider, trace, patterns_step_id,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- EXPERIENCE
    # ------------------------------------------------------------------

    async def _generate_experience(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[str, str]:
        """Generate a detailed initial solution -- the 'waking experience'.

        The LLM is asked to solve the problem step-by-step, making its
        reasoning chain explicit so that subsequent replay phases have
        rich material to work with.

        Returns:
            Tuple of (step_id, solution_text).
        """
        system_prompt = (
            "You are a thorough analytical reasoner. Solve the problem below "
            "step by step, making every decision point and intermediate "
            "conclusion explicit. Number your reasoning steps so the chain "
            "is easy to follow. Be detailed -- this reasoning chain will "
            "be replayed and analysed."
        )
        user_prompt = (
            f"Solve the following problem with detailed, step-by-step reasoning. "
            f"Number each step clearly (Step 1, Step 2, etc.) and explain your "
            f"reasoning at each decision point.\n\n"
            f"PROBLEM:\n{query}"
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=self.experience_temperature,
        )

        step = self._make_step(
            step_type="experience",
            content=raw,
            score=0.5,
            metadata={
                "phase": "experience",
                "content_length": len(raw),
            },
        )
        trace.add_step(step)
        logger.debug("HippocampalReplay: experience generated (%d chars)", len(raw))
        return step.step_id, raw

    # ------------------------------------------------------------------
    # Phase 2 -- FORWARD REPLAY
    # ------------------------------------------------------------------

    async def _forward_replay(
        self,
        query: str,
        experience: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[str, str]:
        """Compress and replay the reasoning chain forward.

        Sharp-wave ripple replay compresses long experiences into brief
        bursts.  Here the LLM is asked to compress the full reasoning
        chain according to ``replay_compression``, preserving the logical
        flow while surfacing the critical transitions.

        Returns:
            Tuple of (step_id, compressed_replay_text).
        """
        compression_pct = int(self.replay_compression * 100)
        system_prompt = (
            "You are performing a compressed forward replay of a reasoning "
            "chain -- analogous to hippocampal sharp-wave ripple replay that "
            "replays experience in compressed temporal sequences. Identify "
            "the key logical transitions and decision points. Preserve the "
            "causal chain while removing redundancy."
        )
        user_prompt = (
            f"Below is a detailed reasoning chain for the problem: \"{query}\"\n\n"
            f"REASONING CHAIN:\n{experience}\n\n"
            f"Perform a FORWARD REPLAY with ~{compression_pct}% compression. "
            f"Compress the chain into a rapid, sequential summary that:\n"
            f"1. Lists each KEY decision point in order\n"
            f"2. Notes the logical transition between consecutive steps\n"
            f"3. Marks any step where the reasoning could have gone differently "
            f"   with [BRANCH POINT]\n"
            f"4. Highlights the strongest and weakest links in the chain\n\n"
            f"Format each item as: 'Step N -> Step N+1: [transition description]'"
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        step = self._make_step(
            step_type="forward_replay",
            content=raw,
            score=0.5,
            metadata={
                "phase": "forward_replay",
                "compression": self.replay_compression,
                "content_length": len(raw),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)
        logger.debug("HippocampalReplay: forward replay complete")
        return step.step_id, raw

    # ------------------------------------------------------------------
    # Phase 3 -- REVERSE REPLAY
    # ------------------------------------------------------------------

    async def _reverse_replay(
        self,
        query: str,
        experience: str,
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Replay the reasoning chain backward for credit assignment.

        Reverse replay (Foster & Wilson 2006) traces from the conclusion
        back to the premises, asking at each step: 'How much did this
        earlier decision contribute to the final answer?'

        Returns:
            Tuple of (step_id, list_of_decision_point_dicts).  Each dict
            has keys ``step_label``, ``description``, ``influence_score``,
            and ``alternative``.
        """
        system_prompt = (
            "You are performing a REVERSE replay of a reasoning chain -- "
            "tracing backward from the conclusion to the premises. This is "
            "analogous to reverse hippocampal replay used for credit "
            "assignment: determining which earlier decisions most influenced "
            "the final outcome. Start from the conclusion and work backward."
        )
        user_prompt = (
            f"Below is a detailed reasoning chain for: \"{query}\"\n\n"
            f"REASONING CHAIN:\n{experience}\n\n"
            f"Perform a REVERSE REPLAY. Start from the FINAL CONCLUSION and "
            f"trace backward through the chain. For each decision point you "
            f"encounter (up to {self.max_decision_points}), provide:\n\n"
            f"Return a JSON array of objects, each with:\n"
            f"- \"step_label\": short label for this decision point\n"
            f"- \"description\": what was decided at this point\n"
            f"- \"influence_score\": 0.0-1.0 how much this decision shaped "
            f"the final answer (1.0 = deterministic influence)\n"
            f"- \"alternative\": what the reasoning would have chosen "
            f"differently if this decision had gone another way\n\n"
            f"Order the array from the conclusion backward to the earliest "
            f"premise. Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        decision_points = self._parse_decision_points(raw)

        step = self._make_step(
            step_type="reverse_replay",
            content=raw,
            score=max((dp["influence_score"] for dp in decision_points), default=0.0),
            metadata={
                "phase": "reverse_replay",
                "decision_point_count": len(decision_points),
                "top_influence": (
                    decision_points[0]["step_label"] if decision_points else "none"
                ),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)

        # Record individual decision-point sub-steps
        for idx, dp in enumerate(decision_points):
            dp_step = self._make_step(
                step_type="decision_point",
                content=(
                    f"[{dp['step_label']}] {dp['description']} "
                    f"(influence: {dp['influence_score']:.2f})"
                ),
                score=dp["influence_score"],
                metadata={
                    "phase": "reverse_replay",
                    "index": idx,
                    "alternative": dp["alternative"],
                },
                parent_step_id=step.step_id,
            )
            trace.add_step(dp_step)

        logger.debug(
            "HippocampalReplay: reverse replay found %d decision points",
            len(decision_points),
        )
        return step.step_id, decision_points

    # ------------------------------------------------------------------
    # Phase 4 -- PATTERN EXTRACTION
    # ------------------------------------------------------------------

    async def _extract_patterns(
        self,
        query: str,
        forward_content: str,
        decision_points: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        forward_step_id: str,
        reverse_step_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Distill recurring patterns from both replay directions.

        Cross-referencing forward and reverse replays reveals structural
        patterns: recurring themes, bottlenecks, and leverage points that
        are visible only when the reasoning chain is examined from both
        directions.

        Returns:
            Tuple of (step_id, list_of_pattern_dicts) where each dict has
            keys ``pattern``, ``evidence``, and ``relevance``.
        """
        dp_summary = "\n".join(
            f"- [{dp['step_label']}] influence={dp['influence_score']:.2f}: "
            f"{dp['description']}"
            for dp in decision_points
        )

        system_prompt = (
            "You are a pattern recognition system analysing the results of "
            "bidirectional replay. Cross-reference forward and reverse passes "
            "to find recurring themes, structural bottlenecks, and leverage "
            "points in the reasoning."
        )
        user_prompt = (
            f"Problem: \"{query}\"\n\n"
            f"FORWARD REPLAY (compressed chain):\n{forward_content}\n\n"
            f"REVERSE REPLAY (decision points with influence scores):\n"
            f"{dp_summary}\n\n"
            f"Extract the key patterns that emerge from BOTH replay directions. "
            f"For each pattern, provide:\n"
            f"- \"pattern\": a concise name for the pattern\n"
            f"- \"evidence\": which forward/reverse observations support it\n"
            f"- \"relevance\": 0.0-1.0 how important this pattern is for "
            f"improving the answer\n\n"
            f"Return a JSON array of pattern objects. Only include patterns "
            f"with relevance >= {self.pattern_threshold:.1f}. "
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        patterns = self._parse_patterns(raw)

        # Filter by threshold
        patterns = [
            p for p in patterns if p["relevance"] >= self.pattern_threshold
        ]

        step = self._make_step(
            step_type="pattern_extraction",
            content=raw,
            score=max((p["relevance"] for p in patterns), default=0.0),
            metadata={
                "phase": "pattern_extraction",
                "pattern_count": len(patterns),
                "pattern_names": [p["pattern"] for p in patterns],
            },
            parent_step_id=forward_step_id,
        )
        trace.add_step(step)
        logger.debug(
            "HippocampalReplay: extracted %d patterns above threshold",
            len(patterns),
        )
        return step.step_id, patterns

    # ------------------------------------------------------------------
    # Phase 5 -- COUNTERFACTUAL REPLAY
    # ------------------------------------------------------------------

    async def _counterfactual_replays(
        self,
        query: str,
        experience: str,
        decision_points: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> list[dict[str, Any]]:
        """Replay with alternative choices at the most influential decision points.

        Selects the top-N most influential decision points (by
        ``influence_score``) and asks the LLM to replay the reasoning
        chain from that point forward, but making the alternative choice.
        This mirrors the brain's ability to simulate 'what if' scenarios
        during hippocampal replay (Olafsdottir, Bush & Barry 2018).

        Returns:
            List of counterfactual insight dicts with keys ``decision_point``,
            ``alternative_path``, ``outcome_difference``, and
            ``insight_score``.
        """
        if not decision_points:
            return []

        # Sort by influence and take top N
        sorted_dps = sorted(
            decision_points, key=lambda d: d["influence_score"], reverse=True
        )
        top_dps = sorted_dps[: self.num_counterfactuals]

        system_prompt = (
            "You are performing a COUNTERFACTUAL replay -- replaying a "
            "reasoning chain from a critical decision point, but making "
            "the ALTERNATIVE choice instead of the original one. Trace "
            "the consequences of this alternative path all the way to a "
            "new conclusion. Compare the original and counterfactual "
            "outcomes."
        )

        insights: list[dict[str, Any]] = []

        cf_parent_step = self._make_step(
            step_type="counterfactual_batch",
            content=(
                f"Running {len(top_dps)} counterfactual replays at the most "
                f"influential decision points."
            ),
            metadata={
                "phase": "counterfactual",
                "num_counterfactuals": len(top_dps),
                "decision_points": [dp["step_label"] for dp in top_dps],
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(cf_parent_step)

        for idx, dp in enumerate(top_dps):
            user_prompt = (
                f"Original problem: \"{query}\"\n\n"
                f"Original reasoning chain:\n{experience}\n\n"
                f"CRITICAL DECISION POINT: [{dp['step_label']}]\n"
                f"Original decision: {dp['description']}\n"
                f"Alternative choice: {dp['alternative']}\n\n"
                f"Replay the reasoning from this point forward, but take the "
                f"ALTERNATIVE path. Trace the consequences to a new conclusion. "
                f"Then compare:\n\n"
                f"Respond with a JSON object:\n"
                f"- \"alternative_path\": brief description of how the "
                f"reasoning unfolds differently\n"
                f"- \"outcome_difference\": how the final answer changes\n"
                f"- \"insight_score\": 0.0-1.0 how valuable this "
                f"counterfactual insight is for improving the original answer\n"
                f"- \"lesson\": one-sentence lesson learned from this "
                f"counterfactual\n\n"
                f"Return ONLY the JSON object."
            )

            raw = await self._call_llm(
                provider,
                [{"role": "user", "content": user_prompt}],
                trace,
                system=system_prompt,
                temperature=0.5,
            )

            cf_result = self._parse_counterfactual(raw, dp["step_label"])

            insight_entry = {
                "decision_point": dp["step_label"],
                "original_decision": dp["description"],
                "alternative_path": cf_result.get("alternative_path", ""),
                "outcome_difference": cf_result.get("outcome_difference", ""),
                "insight_score": cf_result.get("insight_score", 0.5),
                "lesson": cf_result.get("lesson", ""),
            }
            insights.append(insight_entry)

            cf_step = self._make_step(
                step_type="counterfactual_replay",
                content=(
                    f"Counterfactual at [{dp['step_label']}]: "
                    f"{cf_result.get('lesson', 'No lesson extracted.')}"
                ),
                score=cf_result.get("insight_score", 0.5),
                metadata={
                    "phase": "counterfactual",
                    "index": idx,
                    "decision_point": dp["step_label"],
                    "outcome_difference": cf_result.get("outcome_difference", ""),
                },
                parent_step_id=cf_parent_step.step_id,
            )
            trace.add_step(cf_step)

        logger.debug(
            "HippocampalReplay: completed %d counterfactual replays",
            len(insights),
        )
        return insights

    # ------------------------------------------------------------------
    # Phase 6 -- RECONSOLIDATION
    # ------------------------------------------------------------------

    async def _reconsolidate(
        self,
        query: str,
        experience: str,
        patterns: list[dict[str, Any]],
        counterfactual_insights: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> str:
        """Synthesise the final answer from all replay passes.

        Memory reconsolidation in the brain updates a memory each time
        it is retrieved.  Here the original experience is reconsolidated
        with insights from forward replay (structural patterns), reverse
        replay (credit assignment), and counterfactual replay (alternative
        outcomes).

        Returns:
            The final reconsolidated answer text.
        """
        pattern_text = "\n".join(
            f"- [{p['pattern']}] (relevance {p['relevance']:.2f}): {p['evidence']}"
            for p in patterns
        ) or "No strong patterns identified."

        cf_text = "\n".join(
            f"- At [{ci['decision_point']}]: {ci['lesson']} "
            f"(insight score: {ci['insight_score']:.2f})"
            for ci in counterfactual_insights
        ) or "No counterfactual insights."

        system_prompt = (
            "You are performing MEMORY RECONSOLIDATION -- updating and "
            "strengthening an answer by integrating insights from forward "
            "replay, reverse credit assignment, pattern extraction, and "
            "counterfactual analysis. Produce a final answer that is "
            "superior to the original because it incorporates lessons "
            "learned from replaying the reasoning from multiple angles."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"ORIGINAL SOLUTION (the 'experience'):\n{experience}\n\n"
            f"PATTERNS EXTRACTED FROM REPLAY:\n{pattern_text}\n\n"
            f"COUNTERFACTUAL INSIGHTS:\n{cf_text}\n\n"
            f"RECONSOLIDATE: Produce an improved final answer that:\n"
            f"1. Preserves the strengths of the original solution\n"
            f"2. Incorporates the structural patterns discovered\n"
            f"3. Addresses vulnerabilities revealed by counterfactual analysis\n"
            f"4. Strengthens the weakest links identified during replay\n\n"
            f"Provide a complete, self-contained answer."
        )

        final = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=self.consolidation_temperature,
        )

        step = self._make_step(
            step_type="reconsolidation",
            content=final,
            score=1.0,
            metadata={
                "phase": "reconsolidation",
                "patterns_used": len(patterns),
                "counterfactuals_used": len(counterfactual_insights),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(step)
        logger.debug("HippocampalReplay: reconsolidation complete")
        return final

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _parse_decision_points(self, raw: str) -> list[dict[str, Any]]:
        """Parse decision points from LLM output.

        Expects a JSON array of objects with keys ``step_label``,
        ``description``, ``influence_score``, and ``alternative``.
        Falls back to sensible defaults on parse failure.

        Returns:
            List of decision-point dicts, capped at ``max_decision_points``.
        """
        parsed = self._safe_parse_json_array(raw)

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(parsed[: self.max_decision_points]):
            if not isinstance(item, dict):
                continue
            results.append({
                "step_label": str(item.get("step_label", f"decision-{idx}")),
                "description": str(item.get("description", "Unspecified decision")),
                "influence_score": self._clamp_float(
                    item.get("influence_score", 0.5), 0.0, 1.0
                ),
                "alternative": str(
                    item.get("alternative", "No alternative specified")
                ),
            })

        # If nothing parsed, create a single placeholder
        if not results:
            results.append({
                "step_label": "initial-approach",
                "description": "The overall approach chosen for the problem",
                "influence_score": 0.8,
                "alternative": "A fundamentally different methodology",
            })

        return results

    def _parse_patterns(self, raw: str) -> list[dict[str, Any]]:
        """Parse pattern objects from LLM output.

        Expects a JSON array of objects with keys ``pattern``,
        ``evidence``, and ``relevance``.

        Returns:
            List of pattern dicts.
        """
        parsed = self._safe_parse_json_array(raw)

        results: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            results.append({
                "pattern": str(item.get("pattern", "unnamed pattern")),
                "evidence": str(item.get("evidence", "no evidence cited")),
                "relevance": self._clamp_float(
                    item.get("relevance", 0.5), 0.0, 1.0
                ),
            })

        if not results:
            results.append({
                "pattern": "general coherence",
                "evidence": "Overall reasoning chain consistency",
                "relevance": 0.5,
            })

        return results

    def _parse_counterfactual(
        self, raw: str, decision_label: str
    ) -> dict[str, Any]:
        """Parse a single counterfactual result from LLM output.

        Expects a JSON object with keys ``alternative_path``,
        ``outcome_difference``, ``insight_score``, and ``lesson``.

        Returns:
            A dict with the parsed fields and sensible defaults.
        """
        obj = self._safe_parse_json_object(raw)
        return {
            "alternative_path": str(
                obj.get("alternative_path", "Alternative path not specified")
            ),
            "outcome_difference": str(
                obj.get("outcome_difference", "Difference not specified")
            ),
            "insight_score": self._clamp_float(
                obj.get("insight_score", 0.5), 0.0, 1.0
            ),
            "lesson": str(
                obj.get(
                    "lesson",
                    f"No specific lesson from counterfactual at {decision_label}",
                )
            ),
        }

    # ------------------------------------------------------------------
    # Low-level parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json_array(raw: str) -> list[Any]:
        """Extract a JSON array from possibly noisy LLM output.

        Strips markdown fences, finds the outermost ``[...]``, and
        attempts to parse it.  Returns an empty list on failure.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(cleaned[start : end + 1])
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    @staticmethod
    def _safe_parse_json_object(raw: str) -> dict[str, Any]:
        """Extract a JSON object from possibly noisy LLM output.

        Strips markdown fences, finds the outermost ``{...}``, and
        attempts to parse it.  Returns an empty dict on failure.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(cleaned[start : end + 1])
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        return {}

    @staticmethod
    def _clamp_float(value: Any, low: float, high: float) -> float:
        """Safely convert a value to a float and clamp it to [low, high]."""
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return (low + high) / 2.0


__all__ = ["HippocampalReplay"]
