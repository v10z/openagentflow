"""Entropic Funnel reasoning engine.

Applies Shannon entropy, the maximum-entropy principle, and the information
bottleneck to guide reasoning from maximum uncertainty to a sharp conclusion.

Physics / information-theory basis:

Shannon (1948) defined entropy as a measure of uncertainty::

    H(X) = -sum_i p(x_i) * log_2(p(x_i))

Maximum entropy occurs when all outcomes are equally likely.  Minimum entropy
(``H = 0``) occurs when the outcome is certain.  Learning can be understood as
entropy reduction: each piece of evidence eliminates possibilities and lowers
the posterior entropy.

Jaynes' **Maximum Entropy Principle** (1957) says that the least biased
distribution consistent with known constraints is the one with maximum entropy.
As constraints are added, the maximum-entropy distribution narrows.

The **Information Bottleneck** (Tishby et al., 1999) formalises compression:
at each stage, irrelevant information is discarded while information about the
target variable is preserved.  The system passes through stages of decreasing
entropy, forming a funnel shape.

This engine operationalises the funnel: start with every possibility open
(maximum entropy), identify hard constraints, apply them one-by-one to
reduce entropy, then compress through the information bottleneck to obtain
a minimal representation from which the final answer is synthesised.

LLM call pattern::

    MAX_ENTROPY(1) --> CONSTRAINTS(1) --> REDUCTION(C) -->
    BOTTLENECK(1) --> SYNTHESIS(1)

    Total: 4 + C calls  (default C=4: 8 calls)

Example::

    from openagentflow.reasoning.entropic_funnel import EntropicFunnel

    engine = EntropicFunnel(max_constraint_groups=4)
    trace = await engine.reason(
        query="What programming language should we use for our new backend?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
    for step in trace.get_steps_by_type("entropy_reduction"):
        print(step.metadata["entropy_before"], "->", step.metadata["entropy_after"])
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

from openagentflow.reasoning.base import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
)

logger = logging.getLogger(__name__)


class EntropicFunnel(ReasoningEngine):
    """Information-theoretic reasoning via progressive entropy reduction.

    The engine operates like optimal 20-Questions or optimal experimental
    design.  It begins by enumerating the widest possible solution space
    (maximum entropy), then systematically applies constraints to eliminate
    possibilities.  At each step it selects the most informative constraint
    -- the one that maximises expected information gain -- and applies it,
    measuring the resulting entropy reduction.

    Once entropy is sufficiently low, the remaining possibilities are
    compressed through an information bottleneck to extract only what is
    relevant to the answer, and the final synthesis produces a sharp,
    committed conclusion.

    The engine explicitly tracks entropy (in bits) throughout, providing
    a quantitative audit trail of how uncertainty was reduced.

    Attributes:
        name: ``"entropic_funnel"``
        description: Short human-readable summary.
        max_constraint_groups: Number of constraint-application rounds.
        target_entropy: Entropy (in bits) below which to stop reduction.
        initial_temperature: LLM temperature for maximum-entropy generation.
        bottleneck_temperature: LLM temperature for the bottleneck phase.
        synthesis_temperature: LLM temperature for final synthesis.
        entropy_base: Logarithm base for entropy computation (``"log2"``
            or ``"natural"``).
    """

    name: str = "entropic_funnel"
    description: str = (
        "Starts at maximum entropy (all possibilities open), "
        "applies constraints to reduce entropy through an information "
        "funnel until a sharp answer crystallizes."
    )

    def __init__(
        self,
        max_constraint_groups: int = 4,
        target_entropy: float = 1.0,
        initial_temperature: float = 1.0,
        bottleneck_temperature: float = 0.3,
        synthesis_temperature: float = 0.2,
        entropy_computation: str = "log2",
    ) -> None:
        """Initialise the EntropicFunnel engine.

        Args:
            max_constraint_groups: Maximum number of constraint-application
                rounds.  Each round makes one LLM call.
            target_entropy: When the estimated entropy (in bits if using
                ``log2``) drops below this value, stop reducing and move
                to the bottleneck.
            initial_temperature: LLM temperature for the maximum-entropy
                generation phase.
            bottleneck_temperature: LLM temperature for the information
                bottleneck compression.
            synthesis_temperature: LLM temperature for the final
                low-entropy synthesis.
            entropy_computation: ``"log2"`` for bits, ``"natural"`` for
                nats.
        """
        self.max_constraint_groups = max(1, max_constraint_groups)
        self.target_entropy = max(0.0, target_entropy)
        self.initial_temperature = max(0.1, initial_temperature)
        self.bottleneck_temperature = max(0.05, bottleneck_temperature)
        self.synthesis_temperature = max(0.05, synthesis_temperature)

        if entropy_computation not in ("log2", "natural"):
            raise ValueError(
                f"entropy_computation must be 'log2' or 'natural', "
                f"got {entropy_computation!r}"
            )
        self.entropy_computation = entropy_computation

    # ------------------------------------------------------------------
    # Entropy helpers
    # ------------------------------------------------------------------

    def _compute_entropy(self, num_options: int) -> float:
        """Compute the entropy of a uniform distribution over *num_options*.

        Args:
            num_options: Number of equally likely outcomes.

        Returns:
            Entropy in bits (log2) or nats (natural log).
        """
        if num_options <= 1:
            return 0.0
        if self.entropy_computation == "log2":
            return math.log2(num_options)
        return math.log(num_options)

    @property
    def _entropy_unit(self) -> str:
        """Return the unit string for the current entropy base."""
        return "bits" if self.entropy_computation == "log2" else "nats"

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
        """Execute the entropic-funnel reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: LLM provider for all generation and evaluation.
            tools: Optional tool specs (currently unused).
            max_iterations: Hard cap on constraint-application rounds,
                overriding ``max_constraint_groups`` if smaller.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing maximum-entropy,
            constraint, reduction, bottleneck, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        effective_max = min(self.max_constraint_groups, max_iterations)

        # ---------------------------------------------------------------
        # Phase 1: MAXIMUM ENTROPY STATE (1 call)
        # ---------------------------------------------------------------
        max_ent_result = await self._generate_max_entropy(
            query, llm_provider, trace
        )
        dimensions = max_ent_result["dimensions"]
        total_options = max_ent_result["total_options"]
        current_entropy = self._compute_entropy(total_options)

        # ---------------------------------------------------------------
        # Phase 2: CONSTRAINT IDENTIFICATION (1 call)
        # ---------------------------------------------------------------
        constraints = await self._identify_constraints(
            query, dimensions, llm_provider, trace
        )

        # ---------------------------------------------------------------
        # Phase 3: ITERATIVE ENTROPY REDUCTION (C calls)
        # ---------------------------------------------------------------
        remaining_options = total_options
        remaining_dims = dimensions
        reduction_summaries: list[str] = []

        for group_idx in range(min(len(constraints), effective_max)):
            if current_entropy <= self.target_entropy:
                logger.info(
                    "Entropy %.2f below target %.2f at group %d; stopping.",
                    current_entropy,
                    self.target_entropy,
                    group_idx,
                )
                break

            constraint_group = constraints[group_idx]
            result = await self._apply_constraint(
                query,
                constraint_group,
                remaining_dims,
                current_entropy,
                remaining_options,
                reduction_summaries,
                llm_provider,
                trace,
                group_idx,
            )

            remaining_options = result["remaining_options"]
            remaining_dims = result["remaining_dimensions"]
            new_entropy = self._compute_entropy(remaining_options)
            info_gain = current_entropy - new_entropy

            # Record entropy reduction step
            step = self._make_step(
                step_type="entropy_reduction",
                content=(
                    f"Applied constraint: {constraint_group.get('constraint', 'unknown')}\n"
                    f"Entropy: {current_entropy:.2f} -> {new_entropy:.2f} "
                    f"{self._entropy_unit} "
                    f"(gain: {info_gain:.2f} {self._entropy_unit})\n"
                    f"Options: {remaining_options} remaining\n"
                    f"Analysis: {result.get('analysis', '')}"
                ),
                score=info_gain,
                metadata={
                    "phase": "entropy_reduction",
                    "group_index": group_idx,
                    "constraint": constraint_group.get("constraint", ""),
                    "entropy_before": round(current_entropy, 4),
                    "entropy_after": round(new_entropy, 4),
                    "information_gain": round(info_gain, 4),
                    "remaining_options": remaining_options,
                    "entropy_unit": self._entropy_unit,
                },
            )
            trace.add_step(step)

            reduction_summaries.append(
                f"Constraint '{constraint_group.get('constraint', '')}': "
                f"H {current_entropy:.2f}->{new_entropy:.2f}"
            )
            current_entropy = new_entropy

        # ---------------------------------------------------------------
        # Phase 4: INFORMATION BOTTLENECK (1 call)
        # ---------------------------------------------------------------
        minimal_repr = await self._information_bottleneck(
            query,
            remaining_dims,
            current_entropy,
            reduction_summaries,
            llm_provider,
            trace,
        )

        # ---------------------------------------------------------------
        # Phase 5: LOW-ENTROPY SYNTHESIS (1 call)
        # ---------------------------------------------------------------
        final_output = await self._low_entropy_synthesis(
            query,
            minimal_repr,
            current_entropy,
            llm_provider,
            trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1: Maximum entropy generation
    # ------------------------------------------------------------------

    async def _generate_max_entropy(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Generate the broadest possible analysis -- maximum entropy.

        Enumerates all potentially relevant dimensions and options
        without prioritising any of them.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Dict with ``dimensions`` (list of dimension dicts) and
            ``total_options`` (int).
        """
        prompt = (
            f"Generate the BROADEST possible analysis of this problem.  "
            f"List ALL potentially relevant dimensions, factors, approaches, "
            f"and considerations.  Do NOT prioritise -- include everything "
            f"that has any plausibility.  The goal is maximum coverage.\n\n"
            f"Problem: {query}\n\n"
            f"For each dimension, list the possible options or values it "
            f"could take.\n\n"
            f"Respond with JSON only:\n"
            f'{{"dimensions": ['
            f'{{"dimension": "name", "options": ["opt1", "opt2", ...], '
            f'"relevance_uncertain": true}},'
            f" ..."
            f"]}}"
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are a maximum-entropy generator.  Your goal is to be "
                "maximally uncertain -- include every possibility without "
                "preference.  Return valid JSON only."
            ),
            temperature=self.initial_temperature,
        )

        result = self._parse_dimensions(raw)
        dimensions = result["dimensions"]
        total_options = max(1, sum(
            len(d.get("options", [])) for d in dimensions
        ))

        initial_entropy = self._compute_entropy(total_options)

        step = self._make_step(
            step_type="maximum_entropy",
            content=(
                f"Identified {len(dimensions)} dimensions with "
                f"{total_options} total options.\n"
                f"Initial entropy: H = {initial_entropy:.2f} "
                f"{self._entropy_unit}"
            ),
            score=initial_entropy,
            metadata={
                "phase": "maximum_entropy",
                "num_dimensions": len(dimensions),
                "total_options": total_options,
                "entropy": round(initial_entropy, 4),
                "entropy_unit": self._entropy_unit,
            },
        )
        trace.add_step(step)

        return {"dimensions": dimensions, "total_options": total_options}

    # ------------------------------------------------------------------
    # Phase 2: Constraint identification
    # ------------------------------------------------------------------

    async def _identify_constraints(
        self,
        query: str,
        dimensions: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Identify hard constraints and order them by restrictiveness.

        Each constraint eliminates some options, reducing entropy.  The
        constraints are returned in decreasing order of restrictiveness
        (most options eliminated first).

        Args:
            query: Original user query.
            dimensions: The dimensions from the max-entropy phase.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of constraint dicts, each with ``constraint`` (str),
            ``eliminates`` (list of str), and ``entropy_reduction``
            (float estimate).
        """
        dims_summary = json.dumps(dimensions, indent=2, default=str)

        prompt = (
            f"Given this problem and its full option space, identify the "
            f"HARD CONSTRAINTS -- facts, requirements, and logical "
            f"necessities that MUST be satisfied.  Each constraint "
            f"eliminates some options and reduces entropy.\n\n"
            f"Problem: {query}\n\n"
            f"Option space:\n{dims_summary}\n\n"
            f"Order constraints from MOST restrictive (eliminates the most "
            f"options) to LEAST restrictive.  For each, estimate how many "
            f"options it eliminates and the approximate entropy reduction.\n\n"
            f"Respond with JSON only:\n"
            f'{{"constraints": ['
            f'{{"constraint": "description", '
            f'"eliminates": ["option1", "option2"], '
            f'"entropy_reduction": 1.5}},'
            f" ..."
            f"]}}"
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are identifying constraints that narrow a solution "
                "space.  Be rigorous -- only include genuine hard constraints, "
                "not preferences.  Return valid JSON only."
            ),
            temperature=0.3,
        )

        constraints = self._parse_constraints(raw)

        step = self._make_step(
            step_type="constraint_identification",
            content=(
                f"Identified {len(constraints)} constraint groups, "
                f"ordered by restrictiveness."
            ),
            score=0.0,
            metadata={
                "phase": "constraint_identification",
                "num_constraints": len(constraints),
                "constraints": [c.get("constraint", "") for c in constraints],
            },
        )
        trace.add_step(step)

        return constraints

    # ------------------------------------------------------------------
    # Phase 3: Iterative entropy reduction
    # ------------------------------------------------------------------

    async def _apply_constraint(
        self,
        query: str,
        constraint: dict[str, Any],
        remaining_dims: list[dict[str, Any]],
        current_entropy: float,
        remaining_options: int,
        prior_reductions: list[str],
        provider: Any,
        trace: ReasoningTrace,
        group_index: int,
    ) -> dict[str, Any]:
        """Apply a single constraint group and measure entropy reduction.

        After applying the constraint, the LLM identifies the most
        informative analysis to further reduce entropy.

        Args:
            query: Original user query.
            constraint: The constraint to apply.
            remaining_dims: Current surviving dimensions/options.
            current_entropy: Entropy before this constraint.
            remaining_options: Number of options before this constraint.
            prior_reductions: Summary strings of prior reductions.
            provider: LLM provider.
            trace: Reasoning trace.
            group_index: Index of this constraint group.

        Returns:
            Dict with ``remaining_options`` (int),
            ``remaining_dimensions`` (list), and ``analysis`` (str).
        """
        constraint_text = constraint.get("constraint", "unknown constraint")
        eliminates = constraint.get("eliminates", [])
        dims_summary = json.dumps(remaining_dims, indent=2, default=str)
        history = "\n".join(prior_reductions) if prior_reductions else "(none)"

        prompt = (
            f"The constraint '{constraint_text}' has been applied.\n\n"
            f"This eliminates the following options: {eliminates}\n\n"
            f"Problem: {query}\n\n"
            f"Current option space:\n{dims_summary}\n\n"
            f"Prior reductions:\n{history}\n\n"
            f"Current entropy: {current_entropy:.2f} {self._entropy_unit}\n\n"
            f"Tasks:\n"
            f"1. Remove the eliminated options from the option space.\n"
            f"2. Identify any SECONDARY eliminations -- options that become "
            f"impossible as a consequence of the constraint.\n"
            f"3. Determine the most informative question or analysis that "
            f"would further reduce entropy in the remaining space.\n"
            f"4. Perform that analysis.\n\n"
            f"Respond with JSON:\n"
            f'{{"remaining_options": 15, '
            f'"remaining_dimensions": [same format as input], '
            f'"secondary_eliminations": ["opt1", "opt2"], '
            f'"most_informative_question": "...", '
            f'"analysis": "..."}}'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are applying constraints to reduce entropy in a "
                "solution space.  Be precise about what is eliminated and "
                "what remains.  Return valid JSON."
            ),
            temperature=0.3,
        )

        return self._parse_reduction_result(raw, remaining_options, remaining_dims)

    # ------------------------------------------------------------------
    # Phase 4: Information bottleneck
    # ------------------------------------------------------------------

    async def _information_bottleneck(
        self,
        query: str,
        remaining_dims: list[dict[str, Any]],
        current_entropy: float,
        reduction_history: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Compress through the information bottleneck.

        Extracts the minimal representation that preserves all
        information relevant to the answer while discarding everything
        irrelevant.

        Args:
            query: Original user query.
            remaining_dims: Surviving dimensions and options.
            current_entropy: Current entropy level.
            reduction_history: Summary of all prior reductions.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The minimal representation as a string.
        """
        dims_summary = json.dumps(remaining_dims, indent=2, default=str)
        history = "\n".join(reduction_history) if reduction_history else "(none)"

        prompt = (
            f"We have narrowed from maximum entropy to "
            f"{current_entropy:.2f} {self._entropy_unit} of uncertainty.\n\n"
            f"Problem: {query}\n\n"
            f"Remaining option space:\n{dims_summary}\n\n"
            f"Reduction history:\n{history}\n\n"
            f"Apply the INFORMATION BOTTLENECK: What is the MINIMAL "
            f"representation of the problem that preserves ALL information "
            f"relevant to the answer while discarding everything "
            f"irrelevant?\n\n"
            f"Compress the remaining analysis to its essential core.  "
            f"Every word in your response should carry maximum information.  "
            f"Discard redundancy, tangential considerations, and details "
            f"that do not affect the conclusion."
        )

        minimal = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are an information bottleneck.  Discard all irrelevant "
                "detail.  Preserve only what matters for the answer.  "
                "Maximum compression, zero information loss about the target."
            ),
            temperature=self.bottleneck_temperature,
        )

        step = self._make_step(
            step_type="information_bottleneck",
            content=minimal,
            score=current_entropy,
            metadata={
                "phase": "information_bottleneck",
                "entropy_at_bottleneck": round(current_entropy, 4),
                "entropy_unit": self._entropy_unit,
                "compression_ratio": round(
                    len(dims_summary) / max(1, len(minimal)), 2
                ),
            },
        )
        trace.add_step(step)

        return minimal

    # ------------------------------------------------------------------
    # Phase 5: Low-entropy synthesis
    # ------------------------------------------------------------------

    async def _low_entropy_synthesis(
        self,
        query: str,
        minimal_repr: str,
        final_entropy: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesise the final answer from the post-bottleneck state.

        Entropy is now low, so the answer should be sharp and specific
        rather than hedging.

        Args:
            query: Original user query.
            minimal_repr: The minimal representation from the bottleneck.
            final_entropy: Entropy at the point of synthesis.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final answer string.
        """
        prompt = (
            f"Starting from this minimal, post-bottleneck representation, "
            f"construct the final answer.\n\n"
            f"Original problem: {query}\n\n"
            f"Minimal representation (all irrelevant detail removed):\n"
            f"{minimal_repr}\n\n"
            f"Current entropy: {final_entropy:.2f} {self._entropy_unit} "
            f"(low uncertainty).\n\n"
            f"The entropy is now low -- the answer should be SHARP and "
            f"SPECIFIC, not hedging or uncertain.  Commit to the conclusion "
            f"that the information funnel converged toward.  Be decisive."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are synthesising a low-entropy answer.  The hard work "
                "of narrowing has been done.  Be sharp, specific, and "
                "committed.  Do not reintroduce uncertainty."
            ),
            temperature=self.synthesis_temperature,
        )

        synth_step = self._make_step(
            step_type="low_entropy_synthesis",
            content=final,
            score=1.0 - min(1.0, final_entropy / 10.0),
            metadata={
                "phase": "low_entropy_synthesis",
                "final_entropy": round(final_entropy, 4),
                "entropy_unit": self._entropy_unit,
            },
        )
        trace.add_step(synth_step)

        output_step = self._make_step(
            step_type="final_output",
            content=final,
            score=1.0,
            metadata={"phase": "final"},
            parent_step_id=synth_step.step_id,
        )
        trace.add_step(output_step)

        return final

    # ------------------------------------------------------------------
    # JSON parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        """Attempt to extract a JSON object from raw LLM output.

        Strips markdown fences and surrounding text.

        Args:
            raw: Raw LLM output.

        Returns:
            Parsed dict, or ``None`` on failure.
        """
        text = raw.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    @classmethod
    def _parse_dimensions(cls, raw: str) -> dict[str, Any]:
        """Parse the maximum-entropy dimensions output.

        Args:
            raw: Raw LLM output containing a JSON object with a
                ``dimensions`` key.

        Returns:
            Dict with ``dimensions`` (list of dimension dicts).
        """
        default_dim: dict[str, Any] = {
            "dimension": "general",
            "options": ["option_a", "option_b", "option_c"],
            "relevance_uncertain": True,
        }

        parsed = cls._extract_json(raw)
        if parsed is None:
            return {"dimensions": [default_dim]}

        dimensions = parsed.get("dimensions", [])
        if not isinstance(dimensions, list) or not dimensions:
            return {"dimensions": [default_dim]}

        # Validate each dimension
        cleaned: list[dict[str, Any]] = []
        for d in dimensions:
            if not isinstance(d, dict):
                continue
            dim_name = d.get("dimension", d.get("name", "unnamed"))
            options = d.get("options", [])
            if not isinstance(options, list):
                options = [str(options)]
            options = [str(o) for o in options if o]
            if not options:
                options = ["unspecified"]
            cleaned.append({
                "dimension": str(dim_name),
                "options": options,
                "relevance_uncertain": bool(
                    d.get("relevance_uncertain", True)
                ),
            })

        return {"dimensions": cleaned if cleaned else [default_dim]}

    @classmethod
    def _parse_constraints(cls, raw: str) -> list[dict[str, Any]]:
        """Parse the constraint-identification output.

        Args:
            raw: Raw LLM output containing a JSON object with a
                ``constraints`` key.

        Returns:
            List of constraint dicts with ``constraint``, ``eliminates``,
            and ``entropy_reduction`` keys.
        """
        default_constraint: dict[str, Any] = {
            "constraint": "basic feasibility",
            "eliminates": [],
            "entropy_reduction": 1.0,
        }

        parsed = cls._extract_json(raw)
        if parsed is None:
            return [default_constraint]

        constraints = parsed.get("constraints", [])
        if not isinstance(constraints, list) or not constraints:
            return [default_constraint]

        cleaned: list[dict[str, Any]] = []
        for c in constraints:
            if not isinstance(c, dict):
                continue
            constraint_text = c.get("constraint", c.get("description", ""))
            if not constraint_text:
                continue
            eliminates = c.get("eliminates", [])
            if not isinstance(eliminates, list):
                eliminates = [str(eliminates)]
            eliminates = [str(e) for e in eliminates]

            er = c.get("entropy_reduction", 1.0)
            try:
                er = max(0.0, float(er))
            except (TypeError, ValueError):
                er = 1.0

            cleaned.append({
                "constraint": str(constraint_text),
                "eliminates": eliminates,
                "entropy_reduction": er,
            })

        return cleaned if cleaned else [default_constraint]

    @classmethod
    def _parse_reduction_result(
        cls,
        raw: str,
        fallback_options: int,
        fallback_dims: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Parse the result of a constraint-application step.

        Args:
            raw: Raw LLM output.
            fallback_options: Fallback remaining options count.
            fallback_dims: Fallback remaining dimensions.

        Returns:
            Dict with ``remaining_options``, ``remaining_dimensions``,
            and ``analysis``.
        """
        parsed = cls._extract_json(raw)

        if parsed is not None:
            remaining = parsed.get("remaining_options", fallback_options)
            try:
                remaining = max(1, int(remaining))
            except (TypeError, ValueError):
                remaining = max(1, fallback_options - 1)

            dims = parsed.get("remaining_dimensions", fallback_dims)
            if not isinstance(dims, list):
                dims = fallback_dims

            analysis = parsed.get("analysis", "")
            if not isinstance(analysis, str):
                analysis = str(analysis)

            return {
                "remaining_options": remaining,
                "remaining_dimensions": dims,
                "analysis": analysis,
            }

        # Fallback: reduce options by roughly 30%
        reduced = max(1, int(fallback_options * 0.7))
        return {
            "remaining_options": reduced,
            "remaining_dimensions": fallback_dims,
            "analysis": raw[:500] if raw else "",
        }


__all__ = ["EntropicFunnel"]
