"""Lateral Inhibition reasoning engine.

Implements winner-take-all competition inspired by Hartline, Wagner &
Ratliff's (1956) discovery of lateral inhibition in the horseshoe crab
(*Limulus*) retina, and its generalisation to cortical winner-take-all
circuits (Douglas & Martin 2004; Fukai & Tanaka 1997).

Lateral inhibition is a fundamental neural circuit motif in which active
neurons suppress the activity of their neighbours.  The computational
effect is *contrast enhancement*: the most active neuron becomes more
active while its competitors are driven below threshold and silenced.
This transforms a noisy, distributed representation into a sharp,
decisive selection of one winner.

Key properties:
- **Competition**: neurons (hypotheses) actively inhibit each other --
  the stronger a hypothesis, the more it suppresses alternatives.
- **Self-excitation**: each hypothesis also reinforces itself, creating
  positive feedback that amplifies small initial differences.
- **Normalization**: total activity is bounded, so strengthening one
  hypothesis necessarily weakens others (a conserved resource).
- **Thresholding**: hypotheses that fall below a threshold are eliminated
  entirely, sharpening the competition.
- **Contrast enhancement**: the final surviving hypothesis is not just
  the winner but is *more distinct* from the alternatives than it was
  initially.

Algorithm outline::

    1. GENERATION       -- Generate N competing hypotheses
    2. INITIAL SCORING  -- Score each hypothesis independently
    3. INHIBITION ROUNDS (iterate):
       a. ATTACK        -- Each hypothesis generates critiques of competitors
       b. SCORE UPDATE  -- Apply inhibition (critiques reduce scores) and
                           self-excitation (successful defence boosts score)
       c. NORMALIZE     -- Rescale scores to sum to 1.0
       d. THRESHOLD     -- Eliminate hypotheses below the threshold
    4. CONTRAST ENHANCEMENT -- Sharpen the distinction between survivors
    5. SELECTION         -- The highest-scoring survivor is the winner

Example::

    from openagentflow.reasoning.lateral_inhibition import LateralInhibition

    engine = LateralInhibition(
        num_hypotheses=4,
        inhibition_rounds=3,
        elimination_threshold=0.15,
    )
    trace = await engine.reason(
        query="What is the best database choice for a write-heavy IoT platform?",
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


class LateralInhibition(ReasoningEngine):
    """Winner-take-all competition among hypotheses via lateral inhibition.

    The engine generates competing hypotheses, then runs iterative
    rounds of lateral inhibition: each hypothesis attacks its competitors,
    scores are updated with inhibition and self-excitation, weak
    hypotheses are eliminated, and contrast enhancement sharpens the
    final distinction between survivors.

    This mirrors the lateral-inhibition circuits in sensory cortex that
    transform distributed, ambiguous neural responses into sharp,
    decisive selections (Hartline, Wagner & Ratliff 1956; Douglas &
    Martin 2004).

    Parameters:
        num_hypotheses: Number of competing hypotheses to generate.
        inhibition_rounds: Number of attack/defence/elimination rounds.
        elimination_threshold: Score below which a hypothesis is eliminated.
            Higher thresholds produce more aggressive elimination.
        self_excitation: Boost applied to a hypothesis that successfully
            defends against attacks (0-1).
        inhibition_strength: How strongly a successful attack reduces the
            target's score (0-1).
        initial_score: Starting score for all hypotheses before the first
            inhibition round.

    Example::

        engine = LateralInhibition(
            num_hypotheses=5,
            inhibition_rounds=3,
        )
        trace = await engine.reason(
            query="Which concurrency model is best for a high-throughput "
                  "message broker?",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "lateral_inhibition"
    description: str = (
        "Winner-take-all competition among hypotheses via iterative "
        "lateral inhibition, self-excitation, normalization, and "
        "threshold elimination."
    )

    def __init__(
        self,
        num_hypotheses: int = 4,
        inhibition_rounds: int = 3,
        elimination_threshold: float = 0.15,
        self_excitation: float = 0.15,
        inhibition_strength: float = 0.2,
        initial_score: float = 0.5,
    ) -> None:
        self.num_hypotheses = max(2, num_hypotheses)
        self.inhibition_rounds = max(1, inhibition_rounds)
        self.elimination_threshold = max(0.0, min(1.0, elimination_threshold))
        self.self_excitation = max(0.0, min(1.0, self_excitation))
        self.inhibition_strength = max(0.0, min(1.0, inhibition_strength))
        self.initial_score = max(0.0, min(1.0, initial_score))

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
        """Execute the full lateral inhibition reasoning cycle.

        Args:
            query: The problem statement to solve.
            llm_provider: A ``BaseLLMProvider`` instance for all LLM calls.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Soft cap -- ``inhibition_rounds`` is capped at
                ``min(inhibition_rounds, max_iterations)``.
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` is the winning
            hypothesis after competition, contrast-enhanced and polished.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_rounds = min(self.inhibition_rounds, max_iterations)

        # Phase 1 -- GENERATION: create competing hypotheses
        hypotheses = await self._generate_hypotheses(query, llm_provider, trace)

        # Phase 2 -- INITIAL SCORING
        hypotheses = await self._score_hypotheses(
            query, hypotheses, llm_provider, trace
        )

        # Phase 3 -- INHIBITION ROUNDS
        for round_idx in range(effective_rounds):
            if len(hypotheses) < 2:
                # Only one survivor -- competition is over
                break

            round_label = f"round-{round_idx + 1}/{effective_rounds}"
            round_step = self._make_step(
                step_type="inhibition_round",
                content=(
                    f"Inhibition {round_label} with "
                    f"{len(hypotheses)} active hypotheses"
                ),
                metadata={
                    "phase": "inhibition",
                    "round": round_idx + 1,
                    "active_count": len(hypotheses),
                },
            )
            trace.add_step(round_step)

            # 3a. ATTACK: each hypothesis attacks its competitors
            attack_results = await self._attack_phase(
                query, hypotheses, llm_provider, trace, round_step.step_id
            )

            # 3b. SCORE UPDATE: apply inhibition and self-excitation
            hypotheses = self._update_scores(
                hypotheses, attack_results, trace, round_step.step_id
            )

            # 3c. NORMALIZE: rescale scores to sum to 1.0
            hypotheses = self._normalize_scores(hypotheses)

            # 3d. THRESHOLD: eliminate weak hypotheses
            hypotheses, eliminated = self._apply_threshold(
                hypotheses, trace, round_step.step_id
            )

            # Record round summary
            round_step.score = max(
                (h["score"] for h in hypotheses), default=0.0
            )
            round_step.metadata["eliminated"] = eliminated
            round_step.metadata["surviving"] = len(hypotheses)
            round_step.metadata["scores"] = {
                h["label"]: round(h["score"], 4) for h in hypotheses
            }

        # Phase 4 -- CONTRAST ENHANCEMENT
        if len(hypotheses) > 1:
            hypotheses = await self._contrast_enhancement(
                query, hypotheses, llm_provider, trace
            )

        # Phase 5 -- SELECTION: pick and polish the winner
        final_output = await self._select_winner(
            query, hypotheses, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- GENERATION
    # ------------------------------------------------------------------

    async def _generate_hypotheses(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate N competing hypotheses for the problem.

        Each hypothesis is a distinct proposed answer that will compete
        in the lateral inhibition tournament.

        Returns:
            List of hypothesis dicts with ``label``, ``content``, and
            ``score`` (initially ``initial_score``).
        """
        n = self.num_hypotheses
        system_prompt = (
            "You are generating competing hypotheses for a problem. Each "
            "hypothesis should propose a DISTINCT answer or approach. They "
            "should genuinely compete -- not be trivially compatible. "
            "Think of them as rival theories that cannot all be correct."
        )
        user_prompt = (
            f"Generate exactly {n} competing hypotheses (proposed answers) "
            f"for the following problem. Each hypothesis should be "
            f"substantively different from the others.\n\n"
            f"PROBLEM:\n{query}\n\n"
            f"For each hypothesis, provide a short label (2-5 words) and "
            f"a detailed explanation.\n\n"
            f"Return a JSON array of objects, each with:\n"
            f"- \"label\": short name for this hypothesis\n"
            f"- \"content\": detailed explanation (2-4 sentences)\n\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.8,
        )

        parsed = self._parse_hypothesis_list(raw, n)

        gen_step = self._make_step(
            step_type="generation",
            content=f"Generated {len(parsed)} competing hypotheses",
            metadata={
                "phase": "generation",
                "hypothesis_count": len(parsed),
                "labels": [h["label"] for h in parsed],
            },
        )
        trace.add_step(gen_step)

        hypotheses: list[dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            h = {
                "label": item["label"],
                "content": item["content"],
                "score": self.initial_score,
                "attacks_received": [],
                "defences": [],
            }
            h_step = self._make_step(
                step_type="hypothesis",
                content=f"[{item['label']}] {item['content']}",
                score=self.initial_score,
                metadata={
                    "phase": "generation",
                    "hypothesis_index": idx,
                    "label": item["label"],
                },
                parent_step_id=gen_step.step_id,
            )
            trace.add_step(h_step)
            h["step_id"] = h_step.step_id
            hypotheses.append(h)

        logger.debug(
            "LateralInhibition: generated %d hypotheses", len(hypotheses)
        )
        return hypotheses

    # ------------------------------------------------------------------
    # Phase 2 -- INITIAL SCORING
    # ------------------------------------------------------------------

    async def _score_hypotheses(
        self,
        query: str,
        hypotheses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Score each hypothesis independently before competition begins.

        Returns:
            The hypotheses list with ``score`` updated.
        """
        hyp_listing = "\n".join(
            f"  {idx + 1}. [{h['label']}]: {h['content']}"
            for idx, h in enumerate(hypotheses)
        )

        system_prompt = (
            "You are an impartial evaluator. Score each hypothesis on its "
            "initial plausibility, completeness, and strength of reasoning. "
            "Each score should be 0.0-1.0. Return only valid JSON."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"HYPOTHESES:\n{hyp_listing}\n\n"
            f"Score each hypothesis (0.0-1.0) on initial quality.\n"
            f"Return a JSON array of numbers, one per hypothesis, in order.\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.2,
        )

        scores = self._parse_float_list(raw, len(hypotheses))

        for idx, h in enumerate(hypotheses):
            h["score"] = max(0.01, min(1.0, scores[idx]))

        # Normalize after initial scoring
        hypotheses = self._normalize_scores(hypotheses)

        score_step = self._make_step(
            step_type="initial_scoring",
            content="Initial scores: " + ", ".join(
                f"[{h['label']}]={h['score']:.3f}" for h in hypotheses
            ),
            metadata={
                "phase": "initial_scoring",
                "scores": {h["label"]: round(h["score"], 4) for h in hypotheses},
            },
        )
        trace.add_step(score_step)

        logger.debug("LateralInhibition: initial scoring complete")
        return hypotheses

    # ------------------------------------------------------------------
    # Phase 3a -- ATTACK
    # ------------------------------------------------------------------

    async def _attack_phase(
        self,
        query: str,
        hypotheses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> list[dict[str, Any]]:
        """Each hypothesis attacks its competitors.

        For each hypothesis, the LLM generates targeted critiques of
        every other hypothesis.  The effectiveness of each attack is
        scored (0-1).

        Returns:
            List of attack dicts with ``attacker``, ``target``,
            ``critique``, and ``effectiveness``.
        """
        system_prompt = (
            "You are evaluating a hypothesis's ability to undermine a "
            "competitor. The attacker hypothesis tries to show why the "
            "target hypothesis is wrong, incomplete, or inferior. Rate "
            "the attack effectiveness: 0.0 = attack fails completely, "
            "1.0 = devastating critique. Be fair and rigorous."
        )

        attacks: list[dict[str, Any]] = []

        for i, attacker in enumerate(hypotheses):
            for j, target in enumerate(hypotheses):
                if i == j:
                    continue

                user_prompt = (
                    f"PROBLEM: {query}\n\n"
                    f"ATTACKER HYPOTHESIS [{attacker['label']}]:\n"
                    f"{attacker['content']}\n\n"
                    f"TARGET HYPOTHESIS [{target['label']}]:\n"
                    f"{target['content']}\n\n"
                    f"From the perspective of [{attacker['label']}], "
                    f"generate a critique of [{target['label']}]. "
                    f"What is wrong, missing, or weaker about the target?\n\n"
                    f"Return a JSON object with:\n"
                    f"- \"critique\": the specific critique (1-2 sentences)\n"
                    f"- \"effectiveness\": 0.0-1.0 how damaging this critique is\n\n"
                    f"Return ONLY the JSON object."
                )

                raw = await self._call_llm(
                    provider,
                    [{"role": "user", "content": user_prompt}],
                    trace,
                    system=system_prompt,
                    temperature=0.3,
                )

                attack_result = self._parse_attack_result(raw)

                attack_entry = {
                    "attacker": attacker["label"],
                    "attacker_idx": i,
                    "target": target["label"],
                    "target_idx": j,
                    "critique": attack_result.get("critique", "No critique."),
                    "effectiveness": attack_result.get("effectiveness", 0.3),
                }
                attacks.append(attack_entry)

        # Record attack summary
        attack_step = self._make_step(
            step_type="attack_phase",
            content=(
                f"Executed {len(attacks)} attacks across "
                f"{len(hypotheses)} hypotheses. Mean effectiveness: "
                f"{sum(a['effectiveness'] for a in attacks) / max(len(attacks), 1):.3f}"
            ),
            metadata={
                "phase": "attack",
                "attack_count": len(attacks),
                "mean_effectiveness": (
                    sum(a["effectiveness"] for a in attacks)
                    / max(len(attacks), 1)
                ),
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(attack_step)

        logger.debug("LateralInhibition: attack phase produced %d attacks", len(attacks))
        return attacks

    # ------------------------------------------------------------------
    # Phase 3b -- SCORE UPDATE
    # ------------------------------------------------------------------

    def _update_scores(
        self,
        hypotheses: list[dict[str, Any]],
        attacks: list[dict[str, Any]],
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> list[dict[str, Any]]:
        """Apply inhibition (from attacks) and self-excitation to scores.

        For each hypothesis:
        - Incoming attacks reduce its score proportional to attack
          effectiveness and ``inhibition_strength``.
        - Outgoing effective attacks (effectiveness > 0.5) boost its
          score via ``self_excitation`` (successful offence reinforces
          the attacker).

        Returns:
            Updated hypotheses list.
        """
        for h in hypotheses:
            incoming_inhibition = 0.0
            outgoing_excitation = 0.0

            for attack in attacks:
                if attack["target"] == h["label"]:
                    # This hypothesis was attacked
                    incoming_inhibition += (
                        attack["effectiveness"] * self.inhibition_strength
                    )
                if attack["attacker"] == h["label"]:
                    # This hypothesis attacked someone effectively
                    if attack["effectiveness"] > 0.5:
                        outgoing_excitation += (
                            (attack["effectiveness"] - 0.5)
                            * self.self_excitation
                        )

            old_score = h["score"]
            # Apply inhibition (reduce score) and excitation (boost score)
            new_score = old_score - incoming_inhibition + outgoing_excitation
            h["score"] = max(0.0, min(1.0, new_score))

        # Record the update
        update_step = self._make_step(
            step_type="score_update",
            content="Scores after inhibition + excitation: " + ", ".join(
                f"[{h['label']}]={h['score']:.4f}" for h in hypotheses
            ),
            metadata={
                "phase": "score_update",
                "scores": {h["label"]: round(h["score"], 4) for h in hypotheses},
            },
            parent_step_id=parent_step_id,
        )
        trace.add_step(update_step)

        return hypotheses

    # ------------------------------------------------------------------
    # Phase 3c -- NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_scores(
        self,
        hypotheses: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rescale scores to sum to 1.0.

        This enforces the conserved-resource constraint: total 'neural
        activity' is bounded, so strengthening one hypothesis necessarily
        weakens others relative to it.

        Returns:
            Hypotheses with normalized scores.
        """
        total = sum(h["score"] for h in hypotheses)
        if total > 0:
            for h in hypotheses:
                h["score"] = h["score"] / total
        else:
            # All zeros -- distribute equally
            equal_share = 1.0 / max(len(hypotheses), 1)
            for h in hypotheses:
                h["score"] = equal_share
        return hypotheses

    # ------------------------------------------------------------------
    # Phase 3d -- THRESHOLDING
    # ------------------------------------------------------------------

    def _apply_threshold(
        self,
        hypotheses: list[dict[str, Any]],
        trace: ReasoningTrace,
        parent_step_id: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Eliminate hypotheses below the elimination threshold.

        Always preserves at least one hypothesis (the highest scorer).

        Returns:
            Tuple of (surviving_hypotheses, count_eliminated).
        """
        if not hypotheses:
            return hypotheses, 0

        # Sort by score descending to guarantee the best always survives
        hypotheses.sort(key=lambda h: h["score"], reverse=True)

        survivors: list[dict[str, Any]] = []
        eliminated_labels: list[str] = []

        for idx, h in enumerate(hypotheses):
            if h["score"] >= self.elimination_threshold or idx == 0:
                survivors.append(h)
            else:
                eliminated_labels.append(h["label"])

        eliminated_count = len(hypotheses) - len(survivors)

        if eliminated_count > 0:
            elim_step = self._make_step(
                step_type="elimination",
                content=(
                    f"Eliminated {eliminated_count} hypothesis(es) below "
                    f"threshold {self.elimination_threshold}: "
                    f"{', '.join(eliminated_labels)}"
                ),
                metadata={
                    "phase": "threshold",
                    "eliminated": eliminated_labels,
                    "surviving": [h["label"] for h in survivors],
                    "threshold": self.elimination_threshold,
                },
                parent_step_id=parent_step_id,
            )
            trace.add_step(elim_step)

        # Re-normalize survivors
        survivors = self._normalize_scores(survivors)

        logger.debug(
            "LateralInhibition: threshold eliminated %d, %d survive",
            eliminated_count,
            len(survivors),
        )
        return survivors, eliminated_count

    # ------------------------------------------------------------------
    # Phase 4 -- CONTRAST ENHANCEMENT
    # ------------------------------------------------------------------

    async def _contrast_enhancement(
        self,
        query: str,
        hypotheses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Sharpen the distinction between surviving hypotheses.

        After lateral inhibition has eliminated weak competitors, the
        remaining survivors are compared to clarify exactly where they
        differ and why the leader is superior.

        Returns:
            Hypotheses with refined content and potentially adjusted scores.
        """
        survivor_listing = "\n".join(
            f"  [{h['label']}] (score={h['score']:.3f}): {h['content']}"
            for h in hypotheses
        )

        system_prompt = (
            "You are performing contrast enhancement -- sharpening the "
            "distinction between surviving hypotheses after a competition. "
            "Clarify exactly what makes the leading hypothesis superior "
            "and what specific weaknesses the runners-up have."
        )
        user_prompt = (
            f"PROBLEM: {query}\n\n"
            f"SURVIVING HYPOTHESES (after lateral inhibition):\n"
            f"{survivor_listing}\n\n"
            f"For each surviving hypothesis, provide:\n"
            f"- A sharpened, more precise version of the hypothesis\n"
            f"- Its key advantage over the others\n"
            f"- Its remaining weakness\n"
            f"- An adjusted score (0.0-1.0)\n\n"
            f"Return a JSON array of objects, each with:\n"
            f"- \"label\": the hypothesis label\n"
            f"- \"refined_content\": sharpened version\n"
            f"- \"advantage\": key advantage\n"
            f"- \"weakness\": remaining weakness\n"
            f"- \"score\": adjusted score\n\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        enhanced = self._parse_enhanced_hypotheses(raw, hypotheses)

        # Update hypotheses with enhanced content and scores
        label_map = {h["label"]: h for h in hypotheses}
        for item in enhanced:
            if item["label"] in label_map:
                h = label_map[item["label"]]
                h["content"] = item.get("refined_content", h["content"])
                h["score"] = item.get("score", h["score"])
                h["advantage"] = item.get("advantage", "")
                h["weakness"] = item.get("weakness", "")

        # Re-normalize
        hypotheses = self._normalize_scores(hypotheses)

        contrast_step = self._make_step(
            step_type="contrast_enhancement",
            content="Contrast-enhanced scores: " + ", ".join(
                f"[{h['label']}]={h['score']:.3f}" for h in hypotheses
            ),
            metadata={
                "phase": "contrast_enhancement",
                "scores": {h["label"]: round(h["score"], 4) for h in hypotheses},
            },
        )
        trace.add_step(contrast_step)

        logger.debug("LateralInhibition: contrast enhancement complete")
        return hypotheses

    # ------------------------------------------------------------------
    # Phase 5 -- SELECTION
    # ------------------------------------------------------------------

    async def _select_winner(
        self,
        query: str,
        hypotheses: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Select and polish the winning hypothesis.

        The highest-scoring hypothesis is presented as the final answer,
        enriched with insights gained during the competition (what it
        survived, what attacks it withstood).

        Returns:
            The final polished answer text.
        """
        if not hypotheses:
            return "No hypotheses survived the lateral inhibition process."

        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        winner = hypotheses[0]

        runner_up_text = ""
        if len(hypotheses) > 1:
            runner_up_text = (
                "\n\nThe winning hypothesis prevailed over these alternatives:\n"
                + "\n".join(
                    f"- [{h['label']}] (score={h['score']:.3f}): "
                    f"{h.get('weakness', 'eliminated during competition')}"
                    for h in hypotheses[1:]
                )
            )

        system_prompt = (
            "You are producing the final answer from a lateral inhibition "
            "competition. The winning hypothesis has survived multiple rounds "
            "of attack and defence against competitors. Present it as a "
            "polished, comprehensive answer that incorporates the strength "
            "that allowed it to win."
        )
        user_prompt = (
            f"PROBLEM:\n{query}\n\n"
            f"WINNING HYPOTHESIS [{winner['label']}] "
            f"(final score: {winner['score']:.3f}):\n"
            f"{winner['content']}\n"
            f"{runner_up_text}\n\n"
            f"Produce the final answer. The winning hypothesis prevailed through "
            f"competition -- present it confidently while acknowledging the "
            f"key considerations from the alternatives it defeated. "
            f"Provide a complete, self-contained answer."
        )

        final = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.3,
        )

        winner_step = self._make_step(
            step_type="winner_selection",
            content=final,
            score=1.0,
            metadata={
                "phase": "selection",
                "winner": winner["label"],
                "winner_score": winner["score"],
                "runner_up_count": len(hypotheses) - 1,
                "total_rounds": self.inhibition_rounds,
            },
        )
        trace.add_step(winner_step)
        logger.debug(
            "LateralInhibition: winner selected [%s] with score %.3f",
            winner["label"],
            winner["score"],
        )
        return final

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _parse_hypothesis_list(
        self, raw: str, expected: int
    ) -> list[dict[str, Any]]:
        """Parse hypothesis objects from LLM output.

        Expects a JSON array of objects with ``label`` and ``content``.
        Falls back to generating placeholder hypotheses.

        Returns:
            List of hypothesis dicts of length ``expected``.
        """
        parsed = self._safe_parse_json_array(raw)

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            if isinstance(item, dict):
                results.append({
                    "label": str(item.get("label", f"Hypothesis-{idx + 1}")),
                    "content": str(item.get("content", "No content provided.")),
                })
            elif isinstance(item, str):
                results.append({
                    "label": f"Hypothesis-{idx + 1}",
                    "content": item,
                })

        # Pad if needed
        while len(results) < expected:
            results.append({
                "label": f"Hypothesis-{len(results) + 1}",
                "content": f"Alternative approach #{len(results) + 1}",
            })

        return results[:expected]

    def _parse_attack_result(self, raw: str) -> dict[str, Any]:
        """Parse an attack result (critique + effectiveness) from LLM output.

        Returns:
            Dict with ``critique`` and ``effectiveness``.
        """
        obj = self._safe_parse_json_object(raw)
        return {
            "critique": str(obj.get("critique", "No specific critique.")),
            "effectiveness": self._clamp_float(
                obj.get("effectiveness", 0.3), 0.0, 1.0
            ),
        }

    def _parse_enhanced_hypotheses(
        self, raw: str, original: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse contrast-enhanced hypothesis objects from LLM output.

        Returns:
            List of enhanced hypothesis dicts.
        """
        parsed = self._safe_parse_json_array(raw)

        results: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            results.append({
                "label": str(item.get("label", "")),
                "refined_content": str(item.get("refined_content", "")),
                "advantage": str(item.get("advantage", "")),
                "weakness": str(item.get("weakness", "")),
                "score": self._clamp_float(item.get("score", 0.5), 0.0, 1.0),
            })

        return results

    def _parse_float_list(self, raw: str, expected: int) -> list[float]:
        """Parse a JSON array of floats from LLM output.

        Falls back to equal scores if parsing fails.

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
                            result.append(max(0.0, min(1.0, float(v))))
                        except (TypeError, ValueError):
                            result.append(0.5)
                    while len(result) < expected:
                        result.append(0.5)
                    return result[:expected]
            except (json.JSONDecodeError, ValueError):
                pass

        return [0.5] * expected

    # ------------------------------------------------------------------
    # Low-level parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json_array(raw: str) -> list[Any]:
        """Extract a JSON array from possibly noisy LLM output.

        Strips markdown fences, finds the outermost ``[...]``, and
        attempts to parse.  Returns an empty list on failure.
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
        attempts to parse.  Returns an empty dict on failure.
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
        """Safely convert a value to float and clamp to [low, high]."""
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return (low + high) / 2.0


__all__ = ["LateralInhibition"]
