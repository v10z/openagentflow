"""Adversarial Self-Play reasoning engine.

Implements a Red/Blue/Judge tribunal for producing robust outputs through
structured adversarial debate:

1. **Blue (Defender)** generates an initial solution.
2. **Red (Attacker)** finds flaws, edge cases, and counterexamples.
3. **Blue** defends, patches, and improves the solution.
4. **Judge** evaluates each round, scoring attack and defense strength.
5. Repeat until Blue wins *N* consecutive rounds (solution is robust)
   or the maximum number of rounds is reached.

The result is not merely a plausible answer but one that has been
stress-tested against adversarial pressure.  Every exchange is recorded
as a :class:`~openagentflow.reasoning.base.ReasoningStep`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

# ---------------------------------------------------------------------------
# System prompts for the three personas
# ---------------------------------------------------------------------------

_BLUE_INITIAL_SYSTEM = (
    "You are Blue, the problem solver.  Your job is to produce the best "
    "possible solution to the user's query.  Be thorough, precise, and "
    "consider edge cases proactively."
)

_RED_SYSTEM = (
    "You are Red, the adversarial critic.  Your sole purpose is to find "
    "the strongest flaw, edge case, logical error, or counterexample in "
    "the proposed solution.  Be specific and devastating.  Do NOT "
    "congratulate the solution or suggest minor improvements -- find a "
    "real weakness.  If you truly cannot find any flaw, say "
    '"NO_FLAW_FOUND" and explain why the solution is airtight.'
)

_BLUE_DEFEND_SYSTEM = (
    "You are Blue, the defender.  An adversary (Red) has attacked your "
    "solution.  You must do one of the following:\n"
    "1. Fix the flaw Red identified and present an IMPROVED solution.\n"
    "2. Demonstrate convincingly that Red's attack is invalid and the "
    "original solution stands.\n"
    "Always present your COMPLETE updated solution at the end."
)

_JUDGE_SYSTEM = (
    "You are the Judge.  You have seen the original query, the current "
    "solution, an attack from Red, and a defense from Blue.  You must "
    "evaluate this round of debate.\n\n"
    "Respond ONLY with a JSON object (no markdown fences) with these "
    "exact keys:\n"
    '  "winner": "red" or "blue",\n'
    '  "attack_strength": <float 0.0-1.0>,\n'
    '  "defense_strength": <float 0.0-1.0>,\n'
    '  "robust_enough": <bool>,\n'
    '  "reasoning": <string explanation>\n\n'
    "Rules:\n"
    "- Red wins if the attack exposed a genuine, unaddressed flaw.\n"
    "- Blue wins if the defense adequately fixed or refuted the attack.\n"
    '- Set "robust_enough" to true if the solution is now strong enough '
    "that further debate is unlikely to yield material improvements."
)


class AdversarialSelfPlay(ReasoningEngine):
    """Robust reasoning through adversarial Red/Blue/Judge debate.

    Three personas compete:

    * **Red (Attacker)** -- Tries to find flaws, edge cases, and
      counterexamples in the current solution.
    * **Blue (Defender)** -- Defends the solution, patches flaws, and
      presents an improved version.
    * **Judge** -- Evaluates each round, declares a winner, and decides
      whether the solution is robust enough to accept.

    A solution is accepted when Blue wins *blue_wins_needed* consecutive
    rounds, or when *max_rounds* is reached.  The final output is the
    hardened Blue solution.

    Args:
        max_rounds: Maximum number of Red/Blue debate rounds.
        blue_wins_needed: How many consecutive Blue wins are required
            before the solution is deemed robust.

    Example::

        engine = AdversarialSelfPlay(max_rounds=5, blue_wins_needed=2)
        trace = await engine.reason(
            query="Design a thread-safe singleton in Python",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    name: str = "adversarial_self_play"
    description: str = (
        "Red/Blue/Judge adversarial debate for stress-tested, robust outputs."
    )

    def __init__(
        self,
        max_rounds: int = 5,
        blue_wins_needed: int = 2,
    ) -> None:
        self.max_rounds = max_rounds
        self.blue_wins_needed = blue_wins_needed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Run adversarial self-play on *query*.

        The ``max_iterations`` parameter is respected as an upper bound on
        total LLM calls; the engine may also terminate early based on
        *max_rounds* and *blue_wins_needed*.

        Args:
            query: The question or task to reason about.
            llm_provider: A :class:`BaseLLMProvider` instance.
            tools: Optional tool specifications (reserved for future use).
            max_iterations: Maximum total LLM calls (safety limit).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A :class:`ReasoningTrace` containing all debate steps plus
            the final hardened answer.
        """
        start = time.perf_counter()
        trace = ReasoningTrace(strategy_name=self.name)

        # --- 1. Initial Blue solution ---
        solution = await self._initial_solution(query, llm_provider, trace)

        consecutive_blue_wins = 0
        attack_history: list[str] = []

        # --- 2. Debate rounds ---
        for round_num in range(1, self.max_rounds + 1):
            # Safety: respect max_iterations on total LLM calls
            if trace.total_llm_calls >= max_iterations:
                exhaustion_step = ReasoningStep(
                    step_type="observation",
                    content=(
                        f"LLM call limit reached ({max_iterations}). "
                        "Terminating debate."
                    ),
                    metadata={"round": round_num, "reason": "max_iterations"},
                )
                trace.add_step(exhaustion_step)
                break

            # --- 2a. RED ATTACK ---
            attack = await self._red_attack(
                solution=solution,
                query=query,
                attack_history=attack_history,
                provider=llm_provider,
                trace=trace,
                round_num=round_num,
            )
            attack_history.append(attack)

            # If Red couldn't find a flaw, solution is robust
            if "NO_FLAW_FOUND" in attack.upper():
                no_flaw_step = ReasoningStep(
                    step_type="observation",
                    content=(
                        f"Round {round_num}: Red could not find any flaw. "
                        "Solution deemed robust."
                    ),
                    metadata={"round": round_num, "event": "no_flaw_found"},
                    score=1.0,
                )
                trace.add_step(no_flaw_step)
                break

            # --- 2b. BLUE DEFEND ---
            defense = await self._blue_defend(
                solution=solution,
                attack=attack,
                query=query,
                provider=llm_provider,
                trace=trace,
                round_num=round_num,
            )

            # --- 2c. JUDGE VERDICT ---
            verdict = await self._judge_verdict(
                solution=solution,
                attack=attack,
                defense=defense,
                query=query,
                provider=llm_provider,
                trace=trace,
                round_num=round_num,
            )

            # Update solution to Blue's latest (whether Judge agreed or not,
            # the defense always contains Blue's improved attempt).
            solution = defense

            # Track consecutive Blue wins
            winner = verdict.get("winner", "").lower()
            if winner == "blue":
                consecutive_blue_wins += 1
            else:
                consecutive_blue_wins = 0

            # Early termination conditions
            if consecutive_blue_wins >= self.blue_wins_needed:
                robust_step = ReasoningStep(
                    step_type="observation",
                    content=(
                        f"Blue won {consecutive_blue_wins} consecutive rounds. "
                        "Solution is robust."
                    ),
                    metadata={
                        "round": round_num,
                        "consecutive_blue_wins": consecutive_blue_wins,
                    },
                    score=1.0,
                )
                trace.add_step(robust_step)
                break

            if verdict.get("robust_enough", False):
                robust_step = ReasoningStep(
                    step_type="observation",
                    content=(
                        f"Round {round_num}: Judge declared solution robust "
                        f"enough. Reasoning: {verdict.get('reasoning', '')}"
                    ),
                    metadata={"round": round_num, "event": "robust_enough"},
                    score=1.0,
                )
                trace.add_step(robust_step)
                break

        # --- 3. Finalize ---
        trace.final_output = solution
        trace.duration_ms = (time.perf_counter() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Blue: Initial Solution
    # ------------------------------------------------------------------

    async def _initial_solution(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Blue generates the initial solution.

        Args:
            query: The user's question or task.
            provider: LLM provider.
            trace: Active reasoning trace.

        Returns:
            Blue's initial solution text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Produce the best possible solution to this query. "
                    f"Be thorough and consider edge cases.\n\nQuery: {query}"
                ),
            }
        ]

        output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_BLUE_INITIAL_SYSTEM,
            temperature=0.7,
        )

        step = ReasoningStep(
            step_type="thought",
            content=output,
            metadata={"persona": "blue", "phase": "initial_solution"},
        )
        trace.add_step(step)

        return output

    # ------------------------------------------------------------------
    # Red: Attack
    # ------------------------------------------------------------------

    async def _red_attack(
        self,
        solution: str,
        query: str,
        attack_history: list[str],
        provider: Any,
        trace: ReasoningTrace,
        round_num: int,
    ) -> str:
        """Red finds the strongest flaw in the current solution.

        Red is instructed to avoid repeating previous attacks and to
        focus on the most impactful weakness.

        Args:
            solution: The current Blue solution to attack.
            query: The original query for context.
            attack_history: Previous attacks (to avoid repetition).
            provider: LLM provider.
            trace: Active reasoning trace.
            round_num: Current debate round number.

        Returns:
            Red's attack text.
        """
        prior_attacks = ""
        if attack_history:
            numbered = [
                f"  Attack {i+1}: {a}" for i, a in enumerate(attack_history)
            ]
            prior_attacks = (
                "\n\nPrevious attacks (do NOT repeat these -- find a NEW "
                "weakness):\n" + "\n".join(numbered)
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Current solution:\n{solution}\n"
                    f"{prior_attacks}\n\n"
                    "Find the strongest flaw, edge case, logical error, or "
                    "counterexample in this solution.  Be specific and "
                    "devastating.  If the solution is truly airtight, say "
                    '"NO_FLAW_FOUND" and explain why.'
                ),
            }
        ]

        output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_RED_SYSTEM,
            temperature=0.8,
        )

        step = ReasoningStep(
            step_type="critique",
            content=output,
            metadata={
                "persona": "red",
                "phase": "attack",
                "round": round_num,
            },
        )
        trace.add_step(step)

        return output

    # ------------------------------------------------------------------
    # Blue: Defend
    # ------------------------------------------------------------------

    async def _blue_defend(
        self,
        solution: str,
        attack: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        round_num: int,
    ) -> str:
        """Blue addresses the attack and improves the solution.

        Blue must either fix the identified flaw or convincingly refute
        it, and always present the complete updated solution.

        Args:
            solution: The current solution being attacked.
            attack: Red's attack text.
            query: The original query for context.
            provider: LLM provider.
            trace: Active reasoning trace.
            round_num: Current debate round number.

        Returns:
            Blue's defense and updated solution text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Your current solution:\n{solution}\n\n"
                    f"Red's attack:\n{attack}\n\n"
                    "Address this attack.  Either fix the flaw and present an "
                    "IMPROVED solution, or explain convincingly why Red's "
                    "attack is invalid.  In either case, present your "
                    "COMPLETE updated solution at the end."
                ),
            }
        ]

        output = await self._call_llm(
            provider,
            messages,
            trace,
            system=_BLUE_DEFEND_SYSTEM,
            temperature=0.6,
        )

        step = ReasoningStep(
            step_type="thought",
            content=output,
            metadata={
                "persona": "blue",
                "phase": "defense",
                "round": round_num,
            },
        )
        trace.add_step(step)

        return output

    # ------------------------------------------------------------------
    # Judge: Verdict
    # ------------------------------------------------------------------

    async def _judge_verdict(
        self,
        solution: str,
        attack: str,
        defense: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        round_num: int,
    ) -> dict[str, Any]:
        """Judge evaluates a round of debate.

        The Judge scores both the attack strength and defense strength,
        declares a winner, and decides whether the solution is now robust
        enough that further debate would not yield material improvements.

        Args:
            solution: The solution before this round's defense.
            attack: Red's attack text.
            defense: Blue's defense/improved solution.
            query: The original query for context.
            provider: LLM provider.
            trace: Active reasoning trace.
            round_num: Current debate round number.

        Returns:
            A dictionary with keys ``winner`` (``"red"`` or ``"blue"``),
            ``attack_strength`` (float 0--1), ``defense_strength``
            (float 0--1), ``robust_enough`` (bool), and ``reasoning``
            (str).
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Solution before this round:\n{solution}\n\n"
                    f"Red's attack:\n{attack}\n\n"
                    f"Blue's defense:\n{defense}\n\n"
                    "Evaluate this round.  Who won?  Rate attack strength "
                    "(0-1) and defense strength (0-1).  Is the solution now "
                    "robust enough?  Respond ONLY with the required JSON "
                    "object."
                ),
            }
        ]

        raw = await self._call_llm(
            provider,
            messages,
            trace,
            system=_JUDGE_SYSTEM,
            temperature=0.3,
        )

        default: dict[str, Any] = {
            "winner": "blue",
            "attack_strength": 0.5,
            "defense_strength": 0.5,
            "robust_enough": False,
            "reasoning": raw,
        }
        result = self._parse_json_safe(raw, default)

        # Normalize winner to lowercase
        if "winner" in result:
            result["winner"] = str(result["winner"]).lower().strip()
            if result["winner"] not in ("red", "blue"):
                result["winner"] = "blue"

        step = ReasoningStep(
            step_type="judgment",
            content=f"Judge verdict: {json.dumps(result)}",
            metadata={
                "persona": "judge",
                "phase": "verdict",
                "round": round_num,
                "winner": result.get("winner", "blue"),
                "attack_strength": result.get("attack_strength", 0.5),
                "defense_strength": result.get("defense_strength", 0.5),
                "robust_enough": result.get("robust_enough", False),
            },
            score=result.get("defense_strength", 0.5),
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_safe(text: str, default: dict[str, Any]) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output.

        Handles common issues like markdown fences, trailing text after
        the JSON block, and completely non-JSON responses.

        Args:
            text: Raw LLM output expected to contain a JSON object.
            default: Fallback dict to return if parsing fails.

        Returns:
            Parsed dictionary, or *default* on failure.
        """
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Try to find a JSON object in the text
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Last resort: try the whole string
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return default
