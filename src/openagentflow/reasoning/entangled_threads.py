"""Entangled Threads reasoning engine.

Based on entanglement -- the phenomenon where coupled sub-systems cannot be
described independently.  Measuring one particle instantaneously constrains
the state of its entangled partner, regardless of distance.

Physics basis::

    |psi> = (1/sqrt(2)) * (|0>_A |1>_B - |1>_A |0>_B)   (Bell singlet state)

Key properties mapped to reasoning:

1. **Non-local correlations**: A conclusion in one thread instantaneously
   constrains conclusions in all entangled threads -- without explicit
   message passing.
2. **Bell inequality violation**: The correlations between threads are
   stronger than any "local" (independent) reasoning could produce.
3. **Monogamy of entanglement**: If thread A is maximally entangled with B,
   it cannot be entangled with C.  Entanglement is a finite resource.
4. **Decoherence**: Interaction with complexity (the "environment") can
   destroy entanglement, revealing that two threads are not as coupled as
   initially thought.

The engine establishes explicit entanglement maps (correlation constraints)
between threads, co-evolves them with constraint propagation at every step,
detects decoherence when thread states become inconsistent, and measures
threads sequentially so that measuring one constrains the next.

Example::

    from openagentflow.reasoning.entangled_threads import EntangledThreads

    engine = EntangledThreads(num_threads=4)
    trace = await engine.reason(
        query="Design a microservices architecture balancing performance, "
              "security, and developer experience.",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # Inspect entanglement structure
    for step in trace.get_steps_by_type("entanglement_creation"):
        print(f"Entanglements: {step.metadata.get('entanglements')}")
    for step in trace.get_steps_by_type("decoherence_check"):
        print(f"Survived: {step.metadata.get('survived')}")
        print(f"Broken: {step.metadata.get('broken')}")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# Default focus areas when none are specified
_DEFAULT_FOCI = [
    "architectural structure and patterns",
    "performance and scalability",
    "security and reliability",
    "developer experience and maintainability",
]


class EntangledThreads(ReasoningEngine):
    """Entanglement-based reasoning with coupled sub-problem threads.

    The engine works in seven phases:

    1. **THREAD INITIALIZATION** -- Generate N independent reasoning threads,
       each tackling the problem from a different focus area.
    2. **ENTANGLEMENT CREATION** -- Identify hard logical dependencies between
       threads: "if thread A concludes X, then thread B must conclude Y."
       Entanglements can be correlated or anti-correlated.
    3. **MEASUREMENT OF THREAD 1** -- Force the first thread to commit to a
       definite conclusion.
    4. **ENTANGLED COLLAPSE** -- For each thread entangled with thread 1,
       propagate the measurement: the entanglement constraint determines what
       the correlated thread must now conclude.
    5. **DECOHERENCE CHECK** -- Verify that entanglement constraints are still
       consistent after collapse.  Detect broken entanglements caused by
       interaction with the problem's complexity.
    6. **REMAINING THREADS** -- Measure threads that were not entangled with
       thread 1, incorporating context from already-collapsed threads.
    7. **ENTANGLED SYNTHESIS** -- Synthesise the final answer from all
       collapsed thread states, respecting surviving entanglement structure.

    Attributes:
        name: ``"EntangledThreads"``
        description: Short human-readable summary.
        num_threads: Number of reasoning threads.
        thread_foci: Optional pre-specified focus areas for threads.
        entanglement_temperature: Temperature for entanglement creation.
        measurement_temperature: Temperature for thread measurements.
        allow_decoherence: Whether to check for broken entanglements.
        max_entanglement_depth: Max chain length for transitive entanglement.
    """

    name: str = "EntangledThreads"
    description: str = (
        "Parallel reasoning threads with hard entanglement constraints. "
        "Measuring one thread instantaneously collapses correlated threads."
    )

    def __init__(
        self,
        num_threads: int = 4,
        thread_foci: list[str] | None = None,
        entanglement_temperature: float = 0.4,
        measurement_temperature: float = 0.3,
        allow_decoherence: bool = True,
        max_entanglement_depth: int = 2,
    ) -> None:
        """Initialise the Entangled Threads engine.

        Args:
            num_threads: Number of reasoning threads.  Each thread represents
                a different angle or sub-problem.
            thread_foci: Optional list of focus-area strings for threads.
                If ``None`` or shorter than ``num_threads``, defaults are
                supplied.
            entanglement_temperature: Temperature for the entanglement
                creation LLM call (lower = more precise constraints).
            measurement_temperature: Temperature for thread measurement
                LLM calls (lower = more decisive commitments).
            allow_decoherence: If ``True``, performs a decoherence check
                after the initial collapse to detect broken entanglements.
            max_entanglement_depth: Maximum chain length for transitive
                entanglement propagation.
        """
        self.num_threads = max(2, num_threads)
        self.thread_foci = list(thread_foci) if thread_foci else []
        # Pad foci to match num_threads
        while len(self.thread_foci) < self.num_threads:
            default_idx = len(self.thread_foci) % len(_DEFAULT_FOCI)
            self.thread_foci.append(_DEFAULT_FOCI[default_idx])
        self.thread_foci = self.thread_foci[: self.num_threads]
        self.entanglement_temperature = entanglement_temperature
        self.measurement_temperature = measurement_temperature
        self.allow_decoherence = allow_decoherence
        self.max_entanglement_depth = max(1, max_entanglement_depth)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: Any | None = None,
        max_iterations: int = 15,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the Entangled Threads reasoning strategy.

        Args:
            query: The user question or problem to reason about.
            llm_provider: An LLM provider for generation and evaluation.
            tools: Optional tool specs (unused by this engine).
            max_iterations: Hard cap on total LLM calls.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with thread-init, entanglement,
            measurement, collapse, decoherence, and synthesis steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Phase 1 -- THREAD INITIALIZATION
        threads = await self._initialize_threads(query, llm_provider, trace)

        # Phase 2 -- ENTANGLEMENT CREATION
        entanglements = await self._create_entanglements(
            query, threads, llm_provider, trace
        )

        # Build lookup: which threads are entangled with thread 0?
        first_thread_id = 0
        entangled_with_first: list[int] = []
        remaining_threads: list[int] = []

        for e in entanglements:
            thread_ids = e.get("threads", [])
            if first_thread_id in thread_ids:
                for tid in thread_ids:
                    if tid != first_thread_id and tid not in entangled_with_first:
                        entangled_with_first.append(tid)

        for i in range(1, self.num_threads):
            if i not in entangled_with_first:
                remaining_threads.append(i)

        # Phase 3 -- MEASUREMENT OF THREAD 1
        if trace.total_llm_calls < max_iterations:
            thread_0_conclusion = await self._measure_thread(
                query=query,
                thread=threads[first_thread_id],
                thread_index=first_thread_id,
                context="This is the first thread to be measured. "
                        "Make a definite commitment.",
                provider=llm_provider,
                trace=trace,
            )
            threads[first_thread_id]["conclusion"] = thread_0_conclusion
            threads[first_thread_id]["measured"] = True

        # Phase 4 -- ENTANGLED COLLAPSE
        for tid in entangled_with_first:
            if trace.total_llm_calls >= max_iterations:
                break

            # Find the entanglement constraint(s) linking tid to thread 0
            relevant_constraints = []
            for e in entanglements:
                if first_thread_id in e.get("threads", []) and tid in e.get("threads", []):
                    relevant_constraints.append(e)

            collapsed_conclusion = await self._entangled_collapse(
                query=query,
                measured_thread=threads[first_thread_id],
                target_thread=threads[tid],
                target_index=tid,
                constraints=relevant_constraints,
                provider=llm_provider,
                trace=trace,
            )
            threads[tid]["conclusion"] = collapsed_conclusion
            threads[tid]["measured"] = True
            threads[tid]["collapsed_via_entanglement"] = True

        # Phase 5 -- DECOHERENCE CHECK
        surviving_entanglements: list[dict[str, Any]] = list(entanglements)
        broken_entanglements: list[dict[str, Any]] = []

        if self.allow_decoherence and trace.total_llm_calls < max_iterations:
            surviving_entanglements, broken_entanglements = (
                await self._decoherence_check(
                    query, threads, entanglements, llm_provider, trace
                )
            )

        # After decoherence, propagate any newly revealed entanglements
        # between remaining threads and already-collapsed threads
        newly_entangled: list[int] = []
        for e in surviving_entanglements:
            thread_ids = e.get("threads", [])
            for tid in thread_ids:
                if (
                    tid in remaining_threads
                    and tid not in newly_entangled
                    and any(
                        threads[other_tid].get("measured", False)
                        for other_tid in thread_ids
                        if other_tid != tid
                    )
                ):
                    newly_entangled.append(tid)

        for tid in newly_entangled:
            if trace.total_llm_calls >= max_iterations:
                break

            # Find constraints linking this thread to any measured thread
            relevant_constraints = []
            measured_partner = None
            for e in surviving_entanglements:
                if tid in e.get("threads", []):
                    for other_tid in e.get("threads", []):
                        if other_tid != tid and threads[other_tid].get("measured", False):
                            relevant_constraints.append(e)
                            measured_partner = other_tid
                            break

            if measured_partner is not None and relevant_constraints:
                collapsed_conclusion = await self._entangled_collapse(
                    query=query,
                    measured_thread=threads[measured_partner],
                    target_thread=threads[tid],
                    target_index=tid,
                    constraints=relevant_constraints,
                    provider=llm_provider,
                    trace=trace,
                )
                threads[tid]["conclusion"] = collapsed_conclusion
                threads[tid]["measured"] = True
                threads[tid]["collapsed_via_entanglement"] = True
                remaining_threads.remove(tid)

        # Phase 6 -- REMAINING THREADS (independent measurement)
        for tid in remaining_threads:
            if trace.total_llm_calls >= max_iterations:
                break
            if threads[tid].get("measured", False):
                continue

            # Build context from already-measured threads
            context_parts = []
            for other_tid in range(self.num_threads):
                if threads[other_tid].get("measured", False):
                    context_parts.append(
                        f"Thread {other_tid} ({threads[other_tid].get('focus', '?')}): "
                        f"{threads[other_tid].get('conclusion', 'N/A')[:200]}"
                    )
            context = (
                "Other threads have already committed to these conclusions. "
                "Your measurement is independent but should be consistent "
                "with the overall problem context.\n\n"
                + "\n".join(context_parts)
            )

            conclusion = await self._measure_thread(
                query=query,
                thread=threads[tid],
                thread_index=tid,
                context=context,
                provider=llm_provider,
                trace=trace,
            )
            threads[tid]["conclusion"] = conclusion
            threads[tid]["measured"] = True

        # Phase 7 -- ENTANGLED SYNTHESIS
        final_output = await self._synthesize(
            query=query,
            threads=threads,
            surviving_entanglements=surviving_entanglements,
            broken_entanglements=broken_entanglements,
            provider=llm_provider,
            trace=trace,
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- THREAD INITIALIZATION
    # ------------------------------------------------------------------

    async def _initialize_threads(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Generate N independent reasoning threads.

        Each thread analyses the problem from a different focus area and
        produces initial claims, tentative conclusions, and open questions.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of thread dicts with ``focus``, ``content``, ``step_id``,
            and ``measured`` keys.
        """
        threads: list[dict[str, Any]] = []

        for idx in range(self.num_threads):
            focus = self.thread_foci[idx]

            prompt = (
                f"You are reasoning thread {idx}. Your specific focus is: "
                f"{focus}.\n\n"
                f"Analyse the following problem from your perspective. "
                f"Produce your initial state:\n"
                f"- Key claims about the problem from your focus area\n"
                f"- Tentative conclusions\n"
                f"- Open questions that affect your focus area\n"
                f"- Dependencies: what decisions in OTHER areas would "
                f"constrain your conclusions?\n\n"
                f"Problem: {query}"
            )

            raw = await self._call_llm(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                trace=trace,
                system=(
                    f"You are reasoning thread {idx}, focused on: {focus}. "
                    f"Analyse the problem exclusively from this perspective."
                ),
                temperature=0.6,
            )

            step = self._make_step(
                step_type="thread_init",
                content=raw,
                score=0.5,
                metadata={
                    "phase": "thread_init",
                    "thread_index": idx,
                    "focus": focus,
                },
            )
            trace.add_step(step)

            threads.append({
                "index": idx,
                "focus": focus,
                "content": raw,
                "step_id": step.step_id,
                "measured": False,
                "conclusion": None,
                "collapsed_via_entanglement": False,
            })

        return threads

    # ------------------------------------------------------------------
    # Phase 2 -- ENTANGLEMENT CREATION
    # ------------------------------------------------------------------

    async def _create_entanglements(
        self,
        query: str,
        threads: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> list[dict[str, Any]]:
        """Identify hard logical dependencies between threads.

        For each entangled pair, specifies the constraint and whether the
        entanglement is correlated or anti-correlated.

        Args:
            query: Original user query.
            threads: Initialised thread dicts.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of entanglement dicts.
        """
        thread_summaries = []
        for t in threads:
            # Truncate for prompt efficiency
            content_preview = t["content"][:400]
            thread_summaries.append(
                f"Thread {t['index']} (focus: {t['focus']}):\n{content_preview}"
            )
        threads_text = "\n\n".join(thread_summaries)

        prompt = (
            f"Examine the N={self.num_threads} reasoning threads below. "
            f"Identify pairs (or groups) of threads that are logically "
            f"ENTANGLED -- where a conclusion in one thread necessarily "
            f"constrains or determines conclusions in another.\n\n"
            f"These are NOT mere similarities; they are hard logical "
            f"dependencies. For each entangled pair, specify:\n"
            f'- "threads": [i, j] -- the thread indices\n'
            f'- "constraint": A precise statement of the dependency '
            f'("If thread A concludes X, then thread B must conclude Y")\n'
            f'- "type": "correlated" (both must agree/align) or '
            f'"anti_correlated" (if one says X, the other must say NOT-X)\n\n'
            f"Problem: {query}\n\n"
            f"Threads:\n{threads_text}\n\n"
            f"Return ONLY a JSON array of entanglement objects.\n"
            f'Example: [{{"threads": [0, 1], '
            f'"constraint": "If stateless architecture, then must use caching", '
            f'"type": "correlated"}}]'
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are identifying entanglement constraints between "
                "reasoning threads. Only flag TRUE logical dependencies, "
                "not mere topical overlap. Return valid JSON only."
            ),
            temperature=self.entanglement_temperature,
        )

        entanglements = self._parse_entanglement_list(raw)

        # Record entanglement step
        step = self._make_step(
            step_type="entanglement_creation",
            content=(
                f"Identified {len(entanglements)} entanglement(s):\n"
                + "\n".join(
                    f"  threads {e.get('threads', [])}: {e.get('constraint', '?')} "
                    f"({e.get('type', '?')})"
                    for e in entanglements
                )
            ),
            score=0.0,
            metadata={
                "phase": "entanglement_creation",
                "entanglements": entanglements,
                "count": len(entanglements),
            },
        )
        trace.add_step(step)

        return entanglements

    # ------------------------------------------------------------------
    # Phase 3 -- THREAD MEASUREMENT
    # ------------------------------------------------------------------

    async def _measure_thread(
        self,
        query: str,
        thread: dict[str, Any],
        thread_index: int,
        context: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Force a thread to commit to a definite conclusion.

        Args:
            query: Original user query.
            thread: The thread dict to measure.
            thread_index: Index of the thread.
            context: Additional context for the measurement.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The committed conclusion text.
        """
        focus = thread.get("focus", "general")
        initial_analysis = thread.get("content", "")

        prompt = (
            f"Thread {thread_index} (focus: {focus}) is being 'measured' -- "
            f"forced to make a definite commitment.\n\n"
            f"Problem: {query}\n\n"
            f"Your initial analysis:\n{initial_analysis[:600]}\n\n"
            f"{context}\n\n"
            f"Given your analysis and the problem constraints, what is your "
            f"FINAL, COMMITTED conclusion? Be specific and decisive. Do not "
            f"hedge -- you must choose a definite position."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                f"You are reasoning thread {thread_index}, focused on "
                f"{focus}. You must commit to a definite conclusion now. "
                f"No hedging, no ambiguity."
            ),
            temperature=self.measurement_temperature,
        )

        step_type = "measurement" if not thread.get("collapsed_via_entanglement") else "independent_measurement"
        step = self._make_step(
            step_type=step_type,
            content=raw,
            score=0.8,
            metadata={
                "phase": "measurement",
                "thread_index": thread_index,
                "focus": focus,
            },
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Phase 4 -- ENTANGLED COLLAPSE
    # ------------------------------------------------------------------

    async def _entangled_collapse(
        self,
        query: str,
        measured_thread: dict[str, Any],
        target_thread: dict[str, Any],
        target_index: int,
        constraints: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Collapse an entangled thread based on a measured thread's state.

        The entanglement constraint determines what the target thread must
        conclude given the measured thread's commitment.

        Args:
            query: Original user query.
            measured_thread: The already-measured thread.
            target_thread: The thread to collapse.
            target_index: Index of the target thread.
            constraints: Entanglement constraints linking them.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The collapsed conclusion text.
        """
        measured_focus = measured_thread.get("focus", "?")
        measured_conclusion = measured_thread.get("conclusion", "N/A")
        target_focus = target_thread.get("focus", "?")
        target_analysis = target_thread.get("content", "")

        constraints_text = "\n".join(
            f"  - {c.get('constraint', '?')} ({c.get('type', '?')})"
            for c in constraints
        )

        prompt = (
            f"Thread {target_index} (focus: {target_focus}) is ENTANGLED "
            f"with a thread that has already committed.\n\n"
            f"Problem: {query}\n\n"
            f"Measured thread ({measured_focus}) committed to:\n"
            f"{measured_conclusion[:500]}\n\n"
            f"Entanglement constraints:\n{constraints_text}\n\n"
            f"Your initial analysis:\n{target_analysis[:400]}\n\n"
            f"Given the entanglement constraint(s), what MUST thread "
            f"{target_index} now conclude? This is not optional -- the "
            f"entanglement constraint is inviolable. Propagate the collapse.\n\n"
            f"Provide your committed conclusion that satisfies the "
            f"entanglement constraint while staying true to your focus area."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                f"You are reasoning thread {target_index}, focused on "
                f"{target_focus}. You are entangled with another thread that "
                f"has already committed. The entanglement constraint is "
                f"inviolable -- you must respect it."
            ),
            temperature=self.measurement_temperature,
        )

        step = self._make_step(
            step_type="entangled_collapse",
            content=raw,
            score=0.8,
            metadata={
                "phase": "entangled_collapse",
                "thread_index": target_index,
                "focus": target_focus,
                "measured_thread_focus": measured_focus,
                "constraints": [c.get("constraint", "") for c in constraints],
            },
        )
        trace.add_step(step)

        return raw

    # ------------------------------------------------------------------
    # Phase 5 -- DECOHERENCE CHECK
    # ------------------------------------------------------------------

    async def _decoherence_check(
        self,
        query: str,
        threads: list[dict[str, Any]],
        entanglements: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Check whether entanglement constraints survived collapse.

        Detects "decoherence" -- when the problem's complexity causes
        entanglement constraints to break.

        Args:
            query: Original user query.
            threads: All threads (some measured, some not).
            entanglements: Original entanglement list.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            Tuple of (surviving_entanglements, broken_entanglements).
        """
        # Build summary of measured thread conclusions
        measured_summaries = []
        for t in threads:
            if t.get("measured", False) and t.get("conclusion"):
                measured_summaries.append(
                    f"Thread {t['index']} ({t['focus']}): "
                    f"{t['conclusion'][:300]}"
                )
        measured_text = "\n".join(measured_summaries)

        entanglement_text = "\n".join(
            f"  {idx + 1}. threads {e.get('threads', [])}: "
            f"{e.get('constraint', '?')} ({e.get('type', '?')})"
            for idx, e in enumerate(entanglements)
        )

        prompt = (
            f"Check for DECOHERENCE -- are any of the entanglement "
            f"constraints violated by the collapsed thread states?\n\n"
            f"Problem: {query}\n\n"
            f"Committed thread states:\n{measured_text}\n\n"
            f"Entanglement constraints:\n{entanglement_text}\n\n"
            f"For each entanglement constraint, assess:\n"
            f"- Is the constraint satisfied by the current thread states?\n"
            f"- If violated, why? Has interaction with the problem's "
            f"complexity destroyed this entanglement?\n"
            f"- What does a broken entanglement reveal about the problem?\n\n"
            f"Return a JSON object with:\n"
            f'- "survived": array of constraint indices (0-based) that are intact\n'
            f'- "broken": array of objects with "index", "reason"\n\n'
            f"Return ONLY the JSON object."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are checking for decoherence in entangled reasoning "
                "threads. Be precise about which constraints survived and "
                "which broke. Return valid JSON only."
            ),
            temperature=0.3,
        )

        # Parse the decoherence result
        survived_indices, broken_info = self._parse_decoherence(
            raw, len(entanglements)
        )

        surviving = [entanglements[i] for i in survived_indices if i < len(entanglements)]
        broken = []
        for info in broken_info:
            idx = info.get("index", -1)
            if 0 <= idx < len(entanglements):
                entry = dict(entanglements[idx])
                entry["break_reason"] = info.get("reason", "unknown")
                broken.append(entry)

        # Record the step
        step = self._make_step(
            step_type="decoherence_check",
            content=(
                f"Decoherence check: {len(surviving)} survived, "
                f"{len(broken)} broken.\n"
                + (
                    "\n".join(
                        f"  BROKEN: {b.get('constraint', '?')} -- "
                        f"{b.get('break_reason', '?')}"
                        for b in broken
                    )
                    if broken
                    else "  All entanglements intact."
                )
            ),
            score=len(surviving) / max(1, len(entanglements)),
            metadata={
                "phase": "decoherence_check",
                "survived": [e.get("constraint", "") for e in surviving],
                "broken": [
                    {
                        "constraint": b.get("constraint", ""),
                        "reason": b.get("break_reason", ""),
                    }
                    for b in broken
                ],
                "survived_count": len(surviving),
                "broken_count": len(broken),
            },
        )
        trace.add_step(step)

        return surviving, broken

    # ------------------------------------------------------------------
    # Phase 7 -- ENTANGLED SYNTHESIS
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        threads: list[dict[str, Any]],
        surviving_entanglements: list[dict[str, Any]],
        broken_entanglements: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesise the final answer from all collapsed threads.

        Respects surviving entanglement structure and incorporates lessons
        from broken entanglements.

        Args:
            query: Original user query.
            threads: All threads with conclusions.
            surviving_entanglements: Entanglements that survived decoherence.
            broken_entanglements: Entanglements that broke.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final synthesised answer.
        """
        thread_conclusions = []
        for t in threads:
            conclusion = t.get("conclusion", t.get("content", "N/A"))
            conclusion_preview = conclusion[:400] if conclusion else "N/A"
            via = " (collapsed via entanglement)" if t.get("collapsed_via_entanglement") else ""
            thread_conclusions.append(
                f"Thread {t['index']} ({t['focus']}){via}:\n{conclusion_preview}"
            )
        threads_text = "\n\n".join(thread_conclusions)

        surviving_text = "\n".join(
            f"  - threads {e.get('threads', [])}: {e.get('constraint', '?')}"
            for e in surviving_entanglements
        ) or "  (none)"

        broken_text = "\n".join(
            f"  - {e.get('constraint', '?')} -- BROKEN: {e.get('break_reason', '?')}"
            for e in broken_entanglements
        ) or "  (none)"

        prompt = (
            f"All reasoning threads have collapsed to definite states. "
            f"The entanglement constraints that survived decoherence "
            f"represent deep structural relationships in the problem.\n\n"
            f"Problem: {query}\n\n"
            f"Thread conclusions:\n{threads_text}\n\n"
            f"Surviving entanglements (deep structural relationships):\n"
            f"{surviving_text}\n\n"
            f"Broken entanglements (revealed false assumptions):\n"
            f"{broken_text}\n\n"
            f"Synthesise the final answer from all thread conclusions. "
            f"Give particular weight to conclusions linked by surviving "
            f"entanglements -- these represent the strongest structural "
            f"relationships. Note what was learned from broken entanglements."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            trace=trace,
            system=(
                "You are synthesising entangled reasoning threads into a "
                "coherent final answer. Respect the surviving entanglement "
                "structure -- it represents genuine logical dependencies."
            ),
            temperature=0.4,
        )

        step = self._make_step(
            step_type="entangled_synthesis",
            content=final,
            score=1.0,
            metadata={
                "phase": "entangled_synthesis",
                "thread_count": self.num_threads,
                "measured_count": sum(
                    1 for t in threads if t.get("measured", False)
                ),
                "surviving_entanglements": len(surviving_entanglements),
                "broken_entanglements": len(broken_entanglements),
            },
        )
        trace.add_step(step)

        return final

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entanglement_list(raw: str) -> list[dict[str, Any]]:
        """Parse entanglement JSON from LLM output.

        Falls back to a generic entanglement if parsing fails.

        Args:
            raw: Raw LLM output.

        Returns:
            List of entanglement dicts.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, dict):
                            threads = item.get("threads", [])
                            if isinstance(threads, list) and len(threads) >= 2:
                                result.append({
                                    "threads": [int(t) for t in threads[:2]],
                                    "constraint": str(
                                        item.get("constraint", "logical dependency")
                                    ),
                                    "type": str(
                                        item.get("type", "correlated")
                                    ),
                                })
                    if result:
                        return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: assume first two threads are entangled
        return [
            {
                "threads": [0, 1],
                "constraint": "These threads have a logical dependency.",
                "type": "correlated",
            }
        ]

    @staticmethod
    def _parse_decoherence(
        raw: str, total_entanglements: int
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """Parse decoherence check JSON from LLM output.

        Args:
            raw: Raw LLM output.
            total_entanglements: Total number of entanglements to check.

        Returns:
            Tuple of (survived_indices, broken_info_list).
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    survived = [
                        int(i) for i in parsed.get("survived", [])
                    ]
                    broken = []
                    for b in parsed.get("broken", []):
                        if isinstance(b, dict):
                            broken.append({
                                "index": int(b.get("index", -1)),
                                "reason": str(b.get("reason", "unknown")),
                            })
                    return survived, broken
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: assume all survived
        return list(range(total_entanglements)), []


__all__ = ["EntangledThreads"]
