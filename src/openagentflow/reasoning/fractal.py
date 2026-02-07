"""Fractal Recursion reasoning engine.

Applies the same reasoning pattern recursively at multiple scales of
abstraction:

    MACRO  (Strategy)   -- "What approach should we take?"
      MESO  (Tactics)   -- "How do we implement this approach?"
        MICRO (Steps)   -- "What is the exact action / code?"
          NANO (Details) -- "What is the precise value / parameter?"

Key insight: most agent failures are *strategic* (wrong approach) but
retries typically happen at the *micro* level (re-running the same step).
Fractal Recursion ensures that failures at any level trigger
re-reasoning at **that** level, and repeated failures *escalate* upward
so that higher-level plans can be revised.

Each level uses the same decompose-then-execute pattern, creating a
self-similar structure.  Results bubble upward: nano informs micro, micro
informs meso, meso informs macro.  Failures also bubble upward: if micro
fails enough times it triggers meso re-evaluation, and so on.

Pure stdlib -- no third-party dependencies beyond the framework itself.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openagentflow.reasoning.base import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
)


class FractalRecursion(ReasoningEngine):
    """Self-similar recursive reasoning across multiple abstraction scales.

    The engine decomposes problems from macro (strategy) down to nano
    (exact parameters), executing the same reason-decompose-execute
    pattern at every level.  Failures are retried at their originating
    scale, and persistent failures escalate to the next higher scale for
    re-planning.

    Parameters:
        max_depth: How many scale levels to recurse through (1 = macro
            only, 2 = macro + meso, 3 = macro + meso + micro,
            4 = all four levels).
        retries_before_escalation: How many consecutive failures at one
            scale before the engine escalates to the level above.
        inner_strategy: A label describing the reasoning style used at
            each level (informational; does not select a sub-engine).

    Example::

        engine = FractalRecursion(max_depth=3, retries_before_escalation=2)
        trace = await engine.reason(
            query="Build a rate limiter for a REST API.",
            llm_provider=my_provider,
        )
        print(trace.final_output)
    """

    SCALES: list[str] = ["macro", "meso", "micro", "nano"]

    _SCALE_LABELS: dict[str, str] = {
        "macro": "Strategy",
        "meso": "Tactics",
        "micro": "Steps",
        "nano": "Details",
    }

    _SCALE_PROMPTS: dict[str, str] = {
        "macro": (
            "You are a strategic planner. Given the problem below, determine "
            "the high-level strategy. Describe the overall approach, then "
            "list the distinct tactical sub-problems that must be solved. "
            "Output your strategy description first, then list the "
            "sub-problems as a numbered list (1., 2., ...).\n\n"
            "PROBLEM:\n{query}"
        ),
        "meso": (
            "You are a tactical planner. Given the strategy context and the "
            "specific sub-problem below, describe a concrete tactical plan. "
            "Then list the specific executable steps needed as a numbered "
            "list (1., 2., ...).\n\n"
            "STRATEGY CONTEXT:\n{context}\n\n"
            "SUB-PROBLEM:\n{query}"
        ),
        "micro": (
            "You are a precise executor. Given the tactical plan and the "
            "specific step below, provide the exact action, code, or "
            "concrete output required. Be specific and complete -- this "
            "should be directly usable. If this step has sub-details that "
            "need pinning down, list them as a numbered list (1., 2., ...).\n\n"
            "TACTICAL PLAN:\n{context}\n\n"
            "STEP TO EXECUTE:\n{query}"
        ),
        "nano": (
            "You are a detail specialist. Provide the precise value, "
            "parameter, constant, or micro-decision needed for the item "
            "below. Be exact -- give the specific number, string, config "
            "value, or choice, along with a brief justification.\n\n"
            "EXECUTION CONTEXT:\n{context}\n\n"
            "DETAIL NEEDED:\n{query}"
        ),
    }

    name: str = "fractal_recursion"
    description: str = (
        "Self-similar recursive reasoning that decomposes problems across "
        "macro / meso / micro / nano scales with failure escalation."
    )

    def __init__(
        self,
        max_depth: int = 3,
        retries_before_escalation: int = 2,
        inner_strategy: str = "step_by_step",
    ) -> None:
        self.max_depth = max(1, min(max_depth, len(self.SCALES)))
        self.retries_before_escalation = max(1, retries_before_escalation)
        self.inner_strategy = inner_strategy

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
        """Run fractal recursive reasoning on *query*.

        Starts at the macro scale and recurses downward through meso,
        micro, and nano (up to ``max_depth`` levels).  At each level the
        LLM is asked to solve / decompose the current sub-problem, and
        sub-problems are processed recursively at the next finer scale.

        Args:
            query: The problem statement.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Unused -- present for interface compatibility.
            max_iterations: Maximum total LLM calls across all scales
                (safety cap to prevent runaway recursion).
            **kwargs: Reserved for future use.

        Returns:
            A ``ReasoningTrace`` whose ``final_output`` is the
            synthesized result aggregated across all scales.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Shared mutable counter to enforce max_iterations globally
        call_budget = {"remaining": max(max_iterations, 5)}

        result = await self._reason_at_scale(
            query=query,
            scale_index=0,
            context="",
            provider=llm_provider,
            trace=trace,
            call_budget=call_budget,
            depth=0,
            parent_step_id=None,
        )

        trace.final_output = result
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Core recursive method
    # ------------------------------------------------------------------

    async def _reason_at_scale(
        self,
        query: str,
        scale_index: int,
        context: str,
        provider: Any,
        trace: ReasoningTrace,
        call_budget: dict[str, int],
        depth: int,
        parent_step_id: str | None,
    ) -> str:
        """Recursively reason at a given abstraction scale.

        1. Ask the LLM to address *query* at the current scale.
        2. Parse the response to extract sub-problems (if any).
        3. If we have not reached ``max_depth``, recurse into each
           sub-problem at the next finer scale.
        4. Synthesize all sub-results back into a coherent answer at
           the current scale.

        Failures at a given scale are retried up to
        ``retries_before_escalation`` times.  If retries are exhausted
        the failure is escalated (returned to the caller for re-planning).

        Args:
            query: The (sub-)problem to solve at this scale.
            scale_index: Index into ``SCALES`` (0=macro, 1=meso, ...).
            context: Output from the parent scale, used as context.
            provider: LLM provider.
            trace: The reasoning trace being built.
            call_budget: Mutable dict with key ``"remaining"``; decremented
                on each LLM call.
            depth: Current recursion depth (for logging / safety).
            parent_step_id: Step ID of the parent, for trace linkage.

        Returns:
            The synthesized textual result for this scale.
        """
        if scale_index >= self.max_depth or scale_index >= len(self.SCALES):
            # Beyond configured depth -- return context as-is or query
            return context if context else query

        scale = self.SCALES[scale_index]
        label = self._SCALE_LABELS[scale]

        # -- Step 1: Reason at this scale --
        scale_step = ReasoningStep(
            step_type=f"scale_{scale}",
            content=f"[{label}] Reasoning about: {query[:200]}...",
            parent_step_id=parent_step_id,
            metadata={
                "scale": scale,
                "scale_index": scale_index,
                "depth": depth,
            },
        )
        trace.add_step(scale_step)

        retries = 0
        response_text = ""

        while retries <= self.retries_before_escalation:
            if call_budget["remaining"] <= 0:
                scale_step.metadata["budget_exhausted"] = True
                break

            call_budget["remaining"] -= 1

            prompt_template = self._SCALE_PROMPTS[scale]
            prompt = prompt_template.format(
                query=query,
                context=context or "(no prior context)",
            )

            try:
                response_text = await self._call_llm(
                    provider,
                    [{"role": "user", "content": prompt}],
                    trace,
                    system=(
                        f"You are reasoning at the {label} level. "
                        f"Inner strategy: {self.inner_strategy}. "
                        f"Be thorough and precise at this level of abstraction."
                    ),
                    temperature=0.5,
                )
                break  # success
            except Exception as exc:
                retries += 1
                retry_step = ReasoningStep(
                    step_type="retry",
                    content=(
                        f"[{label}] Attempt {retries} failed: {exc!s}. "
                        f"{'Retrying...' if retries <= self.retries_before_escalation else 'Escalating.'}"
                    ),
                    parent_step_id=scale_step.step_id,
                    metadata={
                        "scale": scale,
                        "retry": retries,
                        "error": str(exc),
                    },
                )
                trace.add_step(retry_step)
                scale_step.children.append(retry_step.step_id)

                if retries > self.retries_before_escalation:
                    # Escalate failure
                    escalation_result = await self._escalate(
                        failure_info=(
                            f"Failed {retries} times at {label} level. "
                            f"Last error: {exc!s}. Query: {query[:300]}"
                        ),
                        current_scale=scale,
                        query=query,
                        provider=provider,
                        trace=trace,
                        call_budget=call_budget,
                        parent_step_id=scale_step.step_id,
                    )
                    response_text = escalation_result
                    break

        if not response_text:
            response_text = f"[{label}] Unable to produce a result for: {query[:200]}"

        # Record the response
        response_step = ReasoningStep(
            step_type=f"{scale}_response",
            content=response_text,
            parent_step_id=scale_step.step_id,
            metadata={"scale": scale, "depth": depth},
        )
        trace.add_step(response_step)
        scale_step.children.append(response_step.step_id)

        # -- Step 2: Extract sub-problems --
        sub_problems = self._extract_sub_problems(response_text)

        # If no sub-problems or at max depth, return this level's result
        if not sub_problems or scale_index + 1 >= self.max_depth:
            scale_step.content = (
                f"[{label}] Completed with {len(sub_problems)} sub-items "
                f"(terminal level)"
            )
            return response_text

        # -- Step 3: Recurse into sub-problems --
        decomposition_step = ReasoningStep(
            step_type="decomposition",
            content=(
                f"[{label}] Decomposed into {len(sub_problems)} sub-problems "
                f"for {self._SCALE_LABELS.get(self.SCALES[scale_index + 1], 'next')} level"
            ),
            parent_step_id=scale_step.step_id,
            metadata={
                "scale": scale,
                "sub_problem_count": len(sub_problems),
                "next_scale": self.SCALES[scale_index + 1] if scale_index + 1 < len(self.SCALES) else None,
            },
        )
        trace.add_step(decomposition_step)
        scale_step.children.append(decomposition_step.step_id)

        sub_results: list[str] = []
        for sp_idx, sub_problem in enumerate(sub_problems):
            if call_budget["remaining"] <= 0:
                sub_results.append(
                    f"(Budget exhausted -- sub-problem not processed: "
                    f"{sub_problem[:100]})"
                )
                continue

            sub_result = await self._reason_at_scale(
                query=sub_problem,
                scale_index=scale_index + 1,
                context=response_text,
                provider=provider,
                trace=trace,
                call_budget=call_budget,
                depth=depth + 1,
                parent_step_id=decomposition_step.step_id,
            )
            sub_results.append(sub_result)

        # -- Step 4: Synthesize sub-results back up --
        synthesized = await self._synthesize(
            scale=scale,
            original_response=response_text,
            sub_results=sub_results,
            query=query,
            provider=provider,
            trace=trace,
            call_budget=call_budget,
            parent_step_id=scale_step.step_id,
        )

        scale_step.content = (
            f"[{label}] Completed: {len(sub_problems)} sub-problems solved "
            f"and synthesized"
        )
        return synthesized

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    async def _escalate(
        self,
        failure_info: str,
        current_scale: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        call_budget: dict[str, int],
        parent_step_id: str | None = None,
    ) -> str:
        """Escalate a failure to a higher reasoning level.

        When a scale exhausts its retries, this method asks the LLM to
        re-think the problem from a higher vantage point and propose an
        alternative path.

        Args:
            failure_info: Description of what went wrong and how many
                times it failed.
            current_scale: The scale label where the failure occurred.
            query: The original (sub-)problem.
            provider: LLM provider.
            trace: The reasoning trace being built.
            call_budget: Mutable remaining-call counter.
            parent_step_id: For trace linkage.

        Returns:
            An alternative response produced after re-evaluation.
        """
        label = self._SCALE_LABELS.get(current_scale, current_scale)

        # Find the scale one level above
        try:
            current_idx = self.SCALES.index(current_scale)
        except ValueError:
            current_idx = 0
        higher_idx = max(0, current_idx - 1)
        higher_scale = self.SCALES[higher_idx]
        higher_label = self._SCALE_LABELS.get(higher_scale, higher_scale)

        escalation_step = ReasoningStep(
            step_type="escalation",
            content=(
                f"Escalating failure from {label} to {higher_label}. "
                f"Failure: {failure_info[:300]}"
            ),
            parent_step_id=parent_step_id,
            metadata={
                "from_scale": current_scale,
                "to_scale": higher_scale,
                "failure_info": failure_info,
            },
        )
        trace.add_step(escalation_step)

        if call_budget["remaining"] <= 0:
            return f"[Escalation] Budget exhausted. Last failure: {failure_info[:200]}"

        call_budget["remaining"] -= 1

        system_prompt = (
            f"You are re-evaluating a problem at the {higher_label} level "
            f"because the {label}-level reasoning failed repeatedly. "
            f"Propose an alternative approach that avoids the failure mode."
        )
        user_prompt = (
            f"The following attempt at the {label} level has failed "
            f"multiple times:\n\n"
            f"FAILURE DETAILS:\n{failure_info}\n\n"
            f"ORIGINAL PROBLEM:\n{query}\n\n"
            f"Please re-think this from the {higher_label} level. Propose "
            f"a completely different approach or path that might succeed "
            f"where the previous one failed. Provide a complete answer."
        )

        alternative = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.7,
        )

        alt_step = ReasoningStep(
            step_type="escalation_result",
            content=alternative,
            parent_step_id=escalation_step.step_id,
            metadata={"higher_scale": higher_scale},
        )
        trace.add_step(alt_step)
        escalation_step.children.append(alt_step.step_id)

        return alternative

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        scale: str,
        original_response: str,
        sub_results: list[str],
        query: str,
        provider: Any,
        trace: ReasoningTrace,
        call_budget: dict[str, int],
        parent_step_id: str | None = None,
    ) -> str:
        """Combine sub-results back into a coherent answer at *scale*.

        After child scales have processed their sub-problems, this method
        asks the LLM to weave the pieces together into a unified result
        that is informed by the deeper analysis.

        Args:
            scale: The current scale label (e.g. ``"macro"``).
            original_response: This scale's original LLM output (the plan).
            sub_results: Results produced by the child scale for each
                sub-problem.
            query: The original query at this scale.
            provider: LLM provider.
            trace: The reasoning trace.
            call_budget: Mutable remaining-call counter.
            parent_step_id: For trace linkage.

        Returns:
            The synthesized text.
        """
        label = self._SCALE_LABELS.get(scale, scale)

        if call_budget["remaining"] <= 0:
            # Cannot make another LLM call -- concatenate manually
            combined = (
                f"[{label} synthesis -- budget exhausted]\n\n"
                f"Original plan:\n{original_response}\n\n"
                + "\n\n".join(
                    f"Sub-result {i + 1}:\n{sr}" for i, sr in enumerate(sub_results)
                )
            )
            synth_step = ReasoningStep(
                step_type="synthesis",
                content=combined,
                parent_step_id=parent_step_id,
                metadata={"scale": scale, "budget_exhausted": True},
            )
            trace.add_step(synth_step)
            return combined

        call_budget["remaining"] -= 1

        sub_result_text = "\n\n".join(
            f"--- Sub-result {i + 1} ---\n{sr}"
            for i, sr in enumerate(sub_results)
        )

        system_prompt = (
            f"You are synthesizing results at the {label} level. You have "
            f"a high-level plan and detailed results from a finer analysis. "
            f"Combine them into a single, coherent, and complete answer that "
            f"reflects the depth of the sub-analyses while maintaining the "
            f"strategic clarity of the plan."
        )
        user_prompt = (
            f"ORIGINAL PROBLEM:\n{query}\n\n"
            f"YOUR {label.upper()}-LEVEL PLAN:\n{original_response}\n\n"
            f"DETAILED SUB-RESULTS:\n{sub_result_text}\n\n"
            f"Synthesize these into a single, complete answer. Incorporate "
            f"the insights from the sub-results to make the plan more "
            f"concrete and well-supported. Output the final synthesized "
            f"answer only."
        )

        synthesized = await self._call_llm(
            provider,
            [{"role": "user", "content": user_prompt}],
            trace,
            system=system_prompt,
            temperature=0.4,
        )

        synth_step = ReasoningStep(
            step_type="synthesis",
            content=synthesized,
            parent_step_id=parent_step_id,
            metadata={
                "scale": scale,
                "sub_result_count": len(sub_results),
            },
        )
        trace.add_step(synth_step)

        return synthesized

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sub_problems(text: str) -> list[str]:
        """Parse numbered sub-problems from an LLM response.

        Looks for lines starting with ``1.``, ``2.``, etc.  Falls back
        to splitting on double-newlines if no numbering is found.

        Returns:
            A list of sub-problem strings.  Empty if the response
            does not decompose into sub-items.
        """
        import re

        lines = text.strip().splitlines()
        numbered: list[str] = []
        current_item: list[str] = []

        for line in lines:
            # Detect start of a numbered item
            if re.match(r"^\s*\d+[\.\)]\s", line):
                if current_item:
                    numbered.append("\n".join(current_item).strip())
                # Strip the number prefix
                cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
                current_item = [cleaned]
            elif current_item:
                # Continuation of the current numbered item
                current_item.append(line)

        # Flush last item
        if current_item:
            numbered.append("\n".join(current_item).strip())

        # Filter out very short items (likely not real sub-problems)
        numbered = [item for item in numbered if len(item) > 10]

        if len(numbered) >= 2:
            return numbered

        # If fewer than 2 numbered items, this level probably doesn't
        # decompose further -- return empty to signal "terminal".
        return []
