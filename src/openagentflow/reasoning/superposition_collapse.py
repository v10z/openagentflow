"""Superposition Collapse reasoning engine.

Models wavefunction superposition, interference, and basis-dependent collapse
applied to reasoning.  The engine treats the evaluation criterion (measurement
basis) as a variable, generating multiple solution states that interfere to
modify amplitudes before collapsing under multiple measurement bases.  The
solution that is robust across framings survives.

Physics basis:

In wave mechanics a system exists in a superposition of all possible
eigenstates until a measurement is performed.  The state vector is::

    |psi> = sum_i  c_i |phi_i>

where ``c_i`` are complex probability amplitudes satisfying
``sum |c_i|^2 = 1``.  Upon measurement (the Born rule), the wavefunction
collapses to a single eigenstate ``|phi_k>`` with probability ``|c_k|^2``.

Key insights mapped to reasoning:

1. **Interference before collapse**: States modify each other's amplitudes
   via constructive/destructive interference before any selection occurs.
   This is not voting -- it is mutual modification.
2. **Measurement basis matters**: The same superposition collapses to
   different outcomes depending on the measurement basis.  Measuring
   position versus momentum on the same state yields fundamentally
   different results.  The *framing* of the evaluation criterion determines
   which solution emerges.
3. **Decoherence robustness**: The solution that survives collapse under
   the most measurement bases is the decoherence-robust answer -- it is
   not an artefact of any single framing.

The engine uses approximately 7 LLM calls (default: 5 states, 3 bases).

Example::

    from openagentflow.reasoning.superposition_collapse import SuperpositionCollapse

    engine = SuperpositionCollapse(num_states=5, num_bases=3)
    trace = await engine.reason(
        query="Should we build or buy our ML infrastructure?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_STATE_PREP_SYSTEM = (
    "You are generating a superposition of solution states. Hold all "
    "possibilities simultaneously without premature judgment. Each state "
    "must be a fundamentally different approach -- not just variations on "
    "the same theme. Include at least one unconventional approach."
)

_INTERFERENCE_SYSTEM = (
    "You are evaluating interference between solution states. For each "
    "pair, determine whether considering both together makes each stronger "
    "(constructive interference, positive score) or whether they undermine "
    "each other (destructive interference, negative score). Also note any "
    "emergent insight that neither state contains alone."
)

_BASIS_SYSTEM = (
    "You are selecting measurement bases. A measurement basis is not a "
    "preference -- it is a frame of observation that determines what you "
    "see. Different bases applied to the same superposition yield different "
    "outcomes. Each basis should represent a genuinely different evaluation "
    "criterion."
)

_MEASUREMENT_SYSTEM = (
    "You are performing a measurement on a superposition of solution states. "
    "Given the measurement basis (evaluation criterion), determine which "
    "state this basis selects. Assign probabilities to all states under "
    "this specific frame of evaluation."
)

_DECOHERENCE_SYSTEM = (
    "You are synthesizing a decoherence-robust answer. The solution that "
    "survives collapse under the most measurement bases is the most robust. "
    "Incorporate interference patterns (emergent insights from state "
    "interactions) into the final answer."
)


class SuperpositionCollapse(ReasoningEngine):
    """Superposition and basis-dependent collapse for multi-framing reasoning.

    The engine explores *what question to ask* (which measurement basis to
    use) rather than just *what answer to give*.  It generates a superposition
    of solution states, lets them interfere to modify amplitudes, then
    collapses the superposition under multiple measurement bases.  The
    solution that is robust across framings is selected.

    This differs from evolutionary selection (fitness over generations) and
    resonance networks (pairwise reinforcement to equilibrium).  Here the
    novelty is that the measurement basis itself is a variable.

    Attributes:
        name: ``"SuperpositionCollapse"``
        description: Short human-readable summary.
        num_states: Number of solution states in the superposition.
        num_bases: Number of measurement bases to collapse under.
        interference_strength: Coupling constant for cross-state interference.
        new_state_threshold: Minimum amplitude for emergent pattern states.
        state_temperature: LLM temperature for state generation.
        measurement_temperature: LLM temperature for collapse measurements.
    """

    name: str = "SuperpositionCollapse"
    description: str = (
        "Parallel solution states interfere and collapse under "
        "basis-dependent measurement. The evaluation framing "
        "determines the outcome."
    )

    def __init__(
        self,
        num_states: int = 5,
        num_bases: int = 3,
        interference_strength: float = 0.15,
        new_state_threshold: float = 0.3,
        state_temperature: float = 0.8,
        measurement_temperature: float = 0.4,
    ) -> None:
        """Initialise the Superposition Collapse engine.

        Args:
            num_states: Number of solution states to generate in the initial
                superposition.  More states = broader exploration, more LLM
                calls.
            num_bases: Number of measurement bases (evaluation criteria) to
                collapse under.  More bases = higher confidence that the
                selected solution is genuinely robust.
            interference_strength: Coupling constant ``alpha`` controlling
                how strongly states influence each other's amplitudes.
                Higher = more radical amplitude redistribution.
            new_state_threshold: Minimum amplitude for emergent interference
                pattern states to be included in the superposition.
            state_temperature: LLM temperature for generating solution states
                (higher = more creative / divergent).
            measurement_temperature: LLM temperature for collapse measurements
                (lower = more deterministic evaluation).
        """
        self.num_states = max(2, num_states)
        self.num_bases = max(1, num_bases)
        self.interference_strength = max(0.01, min(1.0, interference_strength))
        self.new_state_threshold = max(0.05, min(1.0, new_state_threshold))
        self.state_temperature = state_temperature
        self.measurement_temperature = measurement_temperature

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
        """Execute the Superposition Collapse reasoning strategy.

        Phases:

        1. State preparation -- generate N solution states
        2. Interference -- pairwise interaction modifies amplitudes
        3. Basis selection -- propose K measurement bases
        4. Measurement / collapse -- collapse under each basis
        5. Decoherence synthesis -- select the robust solution

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for all generation and evaluation.
            tools: Optional tool specs (currently unused).
            max_iterations: Safety cap (not typically binding for this engine).
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing state preparation,
            interference, basis selection, measurement, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- State preparation
        states, amplitudes = await self._prepare_states(
            query, llm_provider, trace
        )

        # Phase 2 -- Interference
        states, amplitudes, patterns = await self._compute_interference(
            query, states, amplitudes, llm_provider, trace
        )

        # Phase 3 -- Basis selection
        bases = await self._select_bases(query, states, llm_provider, trace)

        # Phase 4 -- Measurement / collapse (one per basis)
        measurement_results: list[dict[str, Any]] = []
        for basis_idx, basis in enumerate(bases):
            result = await self._measure(
                query, states, amplitudes, basis, basis_idx, llm_provider, trace
            )
            measurement_results.append(result)

        # Phase 5 -- Decoherence synthesis
        final_output = await self._decoherence_synthesis(
            query, states, amplitudes, patterns,
            measurement_results, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: State preparation
    # ------------------------------------------------------------------

    async def _prepare_states(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[list[str], list[float]]:
        """Generate the initial superposition of solution states.

        Each state begins with equal amplitude ``1 / sqrt(N)``.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (list of state description strings, list of amplitudes).
        """
        prompt = (
            f"Generate {self.num_states} fundamentally different approaches "
            f"to the following problem. Do NOT evaluate them -- simply "
            f"enumerate all plausible states that could be the answer. "
            f"Include at least one unconventional approach.\n\n"
            f"Problem: {query}\n\n"
            f"Return a JSON object:\n"
            f'{{"states": [\n'
            f'  "description of approach 1",\n'
            f'  "description of approach 2",\n'
            f"  ...\n"
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_STATE_PREP_SYSTEM,
            temperature=self.state_temperature,
        )

        states = self._parse_states(raw)

        # Ensure we have at least 2 states
        while len(states) < 2:
            states.append(f"Alternative approach {len(states) + 1} to: {query}")

        # Initialise equal amplitudes: c_i = 1/sqrt(N)
        n = len(states)
        amplitudes = [1.0 / math.sqrt(n)] * n

        # Record preparation step
        prep_step = self._make_step(
            step_type="state_preparation",
            content=raw,
            metadata={
                "phase": "preparation",
                "num_states": n,
                "initial_amplitude": round(amplitudes[0], 6),
            },
        )
        trace.add_step(prep_step)

        # Record individual state steps
        for i, state in enumerate(states):
            state_step = self._make_step(
                step_type="state",
                content=state,
                score=amplitudes[i],
                metadata={
                    "phase": "preparation",
                    "state_index": i,
                    "amplitude": round(amplitudes[i], 6),
                },
                parent_step_id=prep_step.step_id,
            )
            trace.add_step(state_step)

        return states, amplitudes

    # ------------------------------------------------------------------
    # Phase 2: Interference
    # ------------------------------------------------------------------

    async def _compute_interference(
        self,
        query: str,
        states: list[str],
        amplitudes: list[float],
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[list[str], list[float], list[str]]:
        """Compute pairwise interference between states.

        Evaluates constructive (positive) and destructive (negative)
        interference for all unique state pairs, then updates amplitudes
        according to the path-integral-inspired rule::

            c_i' = c_i + sum_j interference(i,j) * c_j * alpha

        Amplitudes are then renormalized so ``sum |c_k'|^2 = 1``.

        Emergent interference patterns (new insights from considering pairs
        together) are added as new states with initial amplitude ``beta``.

        Args:
            query: Original user query.
            states: Current list of state descriptions.
            amplitudes: Current amplitude vector.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (updated states, updated amplitudes, list of emergent
            pattern strings).
        """
        # Build pairs list for the prompt
        pairs_desc: list[str] = []
        pair_labels: list[tuple[int, int]] = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                pairs_desc.append(
                    f"Pair ({i},{j}):\n"
                    f"  State {i}: {states[i][:200]}\n"
                    f"  State {j}: {states[j][:200]}"
                )
                pair_labels.append((i, j))

        pairs_text = "\n\n".join(pairs_desc)

        prompt = (
            f"INTERFERENCE EVALUATION\n\n"
            f"Problem: {query}\n\n"
            f"States in the superposition:\n"
            + "\n".join(f"  State {i}: {s[:150]}" for i, s in enumerate(states))
            + f"\n\nFor each pair of states, determine:\n"
            f"- interference: float in [-1, 1] where +1 = perfectly "
            f"constructive (they strengthen each other), -1 = perfectly "
            f"destructive (they undermine each other)\n"
            f"- pattern: a new insight that emerges from considering both "
            f"states together (null if no emergent insight)\n\n"
            f"Pairs to evaluate:\n{pairs_text}\n\n"
            f"Return a JSON object:\n"
            f'{{"pairs": [\n'
            f'  {{"i": 0, "j": 1, "interference": 0.5, '
            f'"pattern": "emergent insight or null"}},\n'
            f"  ...\n"
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_INTERFERENCE_SYSTEM,
            temperature=0.5,
        )

        interference_data = self._parse_interference(raw, len(states))

        # Apply amplitude update rule
        n = len(states)
        new_amplitudes = list(amplitudes)
        alpha = self.interference_strength

        for pair_info in interference_data:
            i = pair_info.get("i", 0)
            j = pair_info.get("j", 0)
            if 0 <= i < n and 0 <= j < n and i != j:
                interf = pair_info.get("interference", 0.0)
                new_amplitudes[i] += interf * amplitudes[j] * alpha
                new_amplitudes[j] += interf * amplitudes[i] * alpha

        # Collect emergent patterns
        patterns: list[str] = []
        for pair_info in interference_data:
            pattern = pair_info.get("pattern")
            if pattern and isinstance(pattern, str) and pattern.lower() != "null":
                patterns.append(pattern)

        # Add emergent patterns as new states
        for pattern in patterns:
            states.append(pattern)
            new_amplitudes.append(self.new_state_threshold)

        # Renormalize: sum |c_k|^2 = 1
        new_amplitudes = self._renormalize(new_amplitudes)

        # Record interference step
        interf_step = self._make_step(
            step_type="interference",
            content=raw,
            metadata={
                "phase": "interference",
                "num_pairs_evaluated": len(interference_data),
                "num_emergent_patterns": len(patterns),
                "amplitude_update_alpha": alpha,
            },
        )
        trace.add_step(interf_step)

        # Record emergent pattern steps
        for p_idx, pattern in enumerate(patterns):
            pattern_step = self._make_step(
                step_type="interference_pattern",
                content=pattern,
                score=new_amplitudes[n + p_idx] if (n + p_idx) < len(new_amplitudes) else 0.0,
                metadata={
                    "phase": "interference",
                    "pattern_index": p_idx,
                    "source_pair": (
                        interference_data[p_idx].get("i", -1),
                        interference_data[p_idx].get("j", -1),
                    ) if p_idx < len(interference_data) else (-1, -1),
                },
                parent_step_id=interf_step.step_id,
            )
            trace.add_step(pattern_step)

        # Record amplitude update step
        amp_step = self._make_step(
            step_type="amplitude_update",
            content=(
                "Updated amplitudes after interference:\n"
                + "\n".join(
                    f"  State {i}: {a:.6f}"
                    for i, a in enumerate(new_amplitudes)
                )
            ),
            metadata={
                "phase": "interference",
                "amplitudes": [round(a, 6) for a in new_amplitudes],
                "num_states": len(states),
            },
            parent_step_id=interf_step.step_id,
        )
        trace.add_step(amp_step)

        return states, new_amplitudes, patterns

    # ------------------------------------------------------------------
    # Phase 3: Basis selection
    # ------------------------------------------------------------------

    async def _select_bases(
        self,
        query: str,
        states: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[str]:
        """Propose measurement bases (evaluation criteria).

        Each basis represents a different framing or evaluation criterion
        that may cause the superposition to collapse to a different state.

        Args:
            query: Original user query.
            states: Current superposition states.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of measurement basis description strings.
        """
        states_text = "\n".join(
            f"  State {i}: {s[:150]}" for i, s in enumerate(states)
        )

        prompt = (
            f"MEASUREMENT BASIS SELECTION\n\n"
            f"Problem: {query}\n\n"
            f"Current superposition states:\n{states_text}\n\n"
            f"Propose {self.num_bases} different measurement bases -- "
            f"concrete evaluation criteria or framings -- that could collapse "
            f"this superposition. Each basis should represent a genuinely "
            f"different way of deciding which state is best. Different bases "
            f"may yield different winners.\n\n"
            f"Examples of distinct bases:\n"
            f"- Practical feasibility (can it be implemented?)\n"
            f"- Long-term sustainability (will it endure?)\n"
            f"- Stakeholder satisfaction (who benefits?)\n"
            f"- Innovation potential (does it open new possibilities?)\n"
            f"- Risk minimization (what could go wrong?)\n\n"
            f"Return a JSON object:\n"
            f'{{"bases": [\n'
            f'  "description of measurement basis 1",\n'
            f'  "description of measurement basis 2",\n'
            f"  ...\n"
            f"]}}\n\n"
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_BASIS_SYSTEM,
            temperature=0.5,
        )

        bases = self._parse_bases(raw)

        # Ensure we have at least 1 basis
        if not bases:
            bases = ["Overall quality and fitness for purpose"]

        step = self._make_step(
            step_type="basis_selection",
            content=raw,
            metadata={
                "phase": "basis_selection",
                "num_bases": len(bases),
                "bases": bases,
            },
        )
        trace.add_step(step)

        return bases

    # ------------------------------------------------------------------
    # Phase 4: Measurement / collapse
    # ------------------------------------------------------------------

    async def _measure(
        self,
        query: str,
        states: list[str],
        amplitudes: list[float],
        basis: str,
        basis_idx: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Collapse the superposition under a specific measurement basis.

        Args:
            query: Original user query.
            states: Current superposition states.
            amplitudes: Current amplitude vector.
            basis: The measurement basis (evaluation criterion).
            basis_idx: Index of this basis.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``basis``, ``selected_index``, ``probability``,
            ``reasoning``, ``collapsed_answer``, and ``state_probs`` keys.
        """
        states_text = "\n".join(
            f"  State {i} (amplitude={amplitudes[i]:.4f}): {s[:200]}"
            for i, s in enumerate(states)
        )

        prompt = (
            f"MEASUREMENT / COLLAPSE\n\n"
            f"Problem: {query}\n\n"
            f"Measurement basis (evaluation criterion): {basis}\n\n"
            f"Superposition states with current amplitudes:\n{states_text}\n\n"
            f"Apply this measurement basis to collapse the superposition. "
            f"Which state does this evaluation criterion select? Assign a "
            f"probability (0.0-1.0) to each state under this basis, then "
            f"identify the winner and explain why.\n\n"
            f"Return a JSON object:\n"
            f'{{"selected_index": 0, '
            f'"state_probabilities": [0.4, 0.3, 0.2, 0.1], '
            f'"reasoning": "why this state wins under this basis", '
            f'"collapsed_answer": "the answer according to this basis"}}\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_MEASUREMENT_SYSTEM,
            temperature=self.measurement_temperature,
        )

        result = self._parse_measurement(raw, len(states), basis)

        step = self._make_step(
            step_type="measurement",
            content=result.get("reasoning", raw),
            score=result.get("probability", 0.0),
            metadata={
                "phase": "measurement",
                "basis_index": basis_idx,
                "basis": basis,
                "selected_index": result.get("selected_index", 0),
                "state_probabilities": result.get("state_probs", []),
                "probability": round(result.get("probability", 0.0), 4),
            },
        )
        trace.add_step(step)

        return result

    # ------------------------------------------------------------------
    # Phase 5: Decoherence synthesis
    # ------------------------------------------------------------------

    async def _decoherence_synthesis(
        self,
        query: str,
        states: list[str],
        amplitudes: list[float],
        patterns: list[str],
        measurement_results: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize the decoherence-robust answer.

        Identifies the state that survives the most measurement bases and
        builds the final answer incorporating interference patterns.

        Args:
            query: Original user query.
            states: Final superposition states.
            amplitudes: Final amplitude vector.
            patterns: Emergent interference pattern strings.
            measurement_results: Results from all basis collapses.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Final synthesized answer.
        """
        # Compute cross-basis robustness: average probability across all bases
        n = len(states)
        robustness = [0.0] * n
        for result in measurement_results:
            probs = result.get("state_probs", [])
            for i in range(min(n, len(probs))):
                robustness[i] += probs[i]

        if measurement_results:
            robustness = [r / len(measurement_results) for r in robustness]

        # Find the most robust state
        if robustness:
            robust_idx = robustness.index(max(robustness))
        else:
            robust_idx = 0

        # Build measurement summary
        measurement_text = "\n\n".join(
            f"Basis {i + 1}: {r.get('basis', 'unknown')}\n"
            f"  Selected state: {r.get('selected_index', '?')}\n"
            f"  Reasoning: {r.get('reasoning', 'N/A')[:200]}\n"
            f"  Collapsed answer: {r.get('collapsed_answer', 'N/A')[:200]}"
            for i, r in enumerate(measurement_results)
        )

        patterns_text = (
            "\n".join(f"  - {p}" for p in patterns)
            if patterns
            else "  (none)"
        )

        robustness_text = "\n".join(
            f"  State {i}: robustness={r:.4f} "
            f"{'<-- MOST ROBUST' if i == robust_idx else ''}"
            for i, r in enumerate(robustness)
        )

        prompt = (
            f"DECOHERENCE SYNTHESIS\n\n"
            f"Problem: {query}\n\n"
            f"Multiple measurements have been performed on the superposition "
            f"from different bases. The most robust state is the one that "
            f"survives the most measurement bases.\n\n"
            f"MEASUREMENT RESULTS:\n{measurement_text}\n\n"
            f"CROSS-BASIS ROBUSTNESS SCORES:\n{robustness_text}\n\n"
            f"MOST ROBUST STATE (index {robust_idx}):\n"
            f"  {states[robust_idx] if robust_idx < len(states) else 'N/A'}\n\n"
            f"EMERGENT INTERFERENCE PATTERNS:\n{patterns_text}\n\n"
            f"Synthesize the final answer. Build primarily on the most robust "
            f"state but incorporate insights from other bases and interference "
            f"patterns. Explain why this solution is robust across framings."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=_DECOHERENCE_SYSTEM,
            temperature=0.4,
        )

        # Record decoherence step
        dec_step = self._make_step(
            step_type="decoherence",
            content=raw,
            score=max(robustness) if robustness else 0.0,
            metadata={
                "phase": "decoherence",
                "robust_state_index": robust_idx,
                "robustness_scores": [round(r, 4) for r in robustness],
                "num_measurement_bases": len(measurement_results),
                "num_interference_patterns": len(patterns),
            },
        )
        trace.add_step(dec_step)

        # Record final output step
        final_step = self._make_step(
            step_type="final_output",
            content=raw,
            metadata={"phase": "final"},
            parent_step_id=dec_step.step_id,
        )
        trace.add_step(final_step)

        return raw

    # ------------------------------------------------------------------
    # Amplitude utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _renormalize(amplitudes: list[float]) -> list[float]:
        """Renormalize amplitudes so that sum |c_k|^2 = 1.

        Negative amplitudes are clamped to a small positive floor to
        preserve the state (destructive interference reduces but does not
        eliminate a state from consideration).

        Args:
            amplitudes: Amplitude vector to normalize.

        Returns:
            Normalized amplitude vector.
        """
        # Floor negative amplitudes
        floored = [max(0.01, a) for a in amplitudes]
        norm = math.sqrt(sum(a * a for a in floored))
        if norm < 1e-12:
            n = len(floored)
            return [1.0 / math.sqrt(n)] * n
        return [a / norm for a in floored]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_states(raw: str) -> list[str]:
        """Parse state descriptions from LLM output.

        Args:
            raw: Raw LLM output.

        Returns:
            List of state description strings.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                states_raw = parsed.get("states", [])
                if isinstance(states_raw, list):
                    return [str(s) for s in states_raw if s]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try parsing as JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                arr = json.loads(text[start : end + 1])
                if isinstance(arr, list):
                    return [str(s) for s in arr if s]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Last resort: split by numbered lines
        states: list[str] = []
        for line in raw.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and len(line) > 10:
                states.append(line)
        return states

    @staticmethod
    def _parse_interference(
        raw: str, num_states: int
    ) -> list[dict[str, Any]]:
        """Parse interference evaluation results.

        Args:
            raw: Raw LLM output.
            num_states: Number of states in the superposition.

        Returns:
            List of dicts with ``i``, ``j``, ``interference``, and
            ``pattern`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                pairs = parsed.get("pairs", [])
                if isinstance(pairs, list):
                    result: list[dict[str, Any]] = []
                    for p in pairs:
                        if isinstance(p, dict):
                            result.append({
                                "i": int(p.get("i", 0)),
                                "j": int(p.get("j", 0)),
                                "interference": max(
                                    -1.0,
                                    min(1.0, float(p.get("interference", 0.0))),
                                ),
                                "pattern": p.get("pattern"),
                            })
                    return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return []

    @staticmethod
    def _parse_bases(raw: str) -> list[str]:
        """Parse measurement bases from LLM output.

        Args:
            raw: Raw LLM output.

        Returns:
            List of basis description strings.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                bases_raw = parsed.get("bases", [])
                if isinstance(bases_raw, list):
                    return [str(b) for b in bases_raw if b]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                arr = json.loads(text[start : end + 1])
                if isinstance(arr, list):
                    return [str(b) for b in arr if b]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return []

    @staticmethod
    def _parse_measurement(
        raw: str, num_states: int, basis: str
    ) -> dict[str, Any]:
        """Parse measurement collapse result.

        Args:
            raw: Raw LLM output.
            num_states: Number of states in the superposition.
            basis: The measurement basis used.

        Returns:
            Dict with ``basis``, ``selected_index``, ``probability``,
            ``reasoning``, ``collapsed_answer``, and ``state_probs`` keys.
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                idx = int(parsed.get("selected_index", 0))
                idx = max(0, min(num_states - 1, idx))

                probs_raw = parsed.get("state_probabilities", [])
                state_probs: list[float] = []
                if isinstance(probs_raw, list):
                    for p in probs_raw:
                        try:
                            state_probs.append(max(0.0, min(1.0, float(p))))
                        except (ValueError, TypeError):
                            state_probs.append(0.0)

                # Pad or trim to num_states
                while len(state_probs) < num_states:
                    state_probs.append(0.0)
                state_probs = state_probs[:num_states]

                # Normalize probabilities
                total = sum(state_probs)
                if total > 0:
                    state_probs = [p / total for p in state_probs]
                else:
                    state_probs = [1.0 / num_states] * num_states

                return {
                    "basis": basis,
                    "selected_index": idx,
                    "probability": state_probs[idx] if idx < len(state_probs) else 0.0,
                    "reasoning": str(parsed.get("reasoning", raw)),
                    "collapsed_answer": str(
                        parsed.get("collapsed_answer", "")
                    ),
                    "state_probs": [round(p, 4) for p in state_probs],
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: equal probabilities
        equal = [1.0 / num_states] * num_states
        return {
            "basis": basis,
            "selected_index": 0,
            "probability": equal[0],
            "reasoning": raw,
            "collapsed_answer": "",
            "state_probs": [round(p, 4) for p in equal],
        }


__all__ = ["SuperpositionCollapse"]
