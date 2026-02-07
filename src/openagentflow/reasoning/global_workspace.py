"""Global Workspace Theory reasoning engine for OpenAgentFlow.

Based on Bernard Baars' Global Workspace Theory (1988, 2005) and Stanislas
Dehaene's neuronal workspace model (Dehaene & Naccache 2001; Dehaene,
Changeux, Naccache et al. 2006).

Consciousness, in this framework, is a *global broadcast* mechanism.  The
brain contains many specialised, unconscious processors running in parallel
-- visual cortex, auditory cortex, language areas, motor planning circuits,
emotional circuits, memory systems.  Each processor operates independently on
its own domain, but only one (or a small coalition) can "win" access to the
global workspace at any given moment.

The global workspace acts as a shared blackboard with limited capacity.
When a processor's output is selected for broadcast (through an attentional
bottleneck), its content becomes available to *all* other processors
simultaneously.  This broadcast-then-integrate cycle is what we experience
as conscious thought.  Key properties:

- **Parallel unconscious processing**: Many specialists work simultaneously.
- **Competitive selection**: An attentional bottleneck selects the most
  relevant/salient specialist output.
- **Global broadcast**: The winner's content is broadcast to all processors.
- **Integration and ignition**: Other processors integrate the broadcast
  content with their own processing, potentially triggering a cascade.

The engine implements 5 specialist processors (analytical, creative,
critical, integrative, experiential) that process the query in parallel.
An attentional bottleneck selects the most relevant contribution.  The
winner's insight is broadcast, and all specialists re-process in light of
the broadcast.  Multiple broadcast cycles refine the collective output.

Example::

    from openagentflow.reasoning.global_workspace import GlobalWorkspace

    engine = GlobalWorkspace(broadcast_cycles=3)
    trace = await engine.reason(
        query="How should we approach the ethical implications of AGI?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # See which specialist won each broadcast cycle
    for step in trace.get_steps_by_type("broadcast"):
        winner = step.metadata.get("winner", "?")
        print(f"Broadcast winner: {winner}")

Trace structure (DAG)::

    query
      +-- specialist_analytical_0
      +-- specialist_creative_0
      +-- specialist_critical_0
      +-- specialist_integrative_0
      +-- specialist_experiential_0
      +-- attention_selection_0
      +-- broadcast_0
      +-- specialist_analytical_1 (parent: broadcast_0)
      +-- specialist_creative_1 (parent: broadcast_0)
      +-- ... (repeat for each cycle)
      +-- final_integration
      +-- final_output
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Specialist definitions
# ---------------------------------------------------------------------------

_SPECIALISTS: dict[str, dict[str, str]] = {
    "analytical": {
        "label": "Analytical Processor",
        "system": (
            "You are the ANALYTICAL specialist processor.  Your domain is "
            "logic, structure, formal reasoning, and systematic decomposition. "
            "Analyse the problem using rigorous logical frameworks, identify "
            "causal relationships, break down complex structures, and apply "
            "deductive and inductive reasoning.  Be precise and methodical."
        ),
    },
    "creative": {
        "label": "Creative Processor",
        "system": (
            "You are the CREATIVE specialist processor.  Your domain is "
            "divergent thinking, novel associations, metaphors, analogies, "
            "and unconventional perspectives.  Generate surprising connections, "
            "reframe the problem in unexpected ways, propose imaginative "
            "solutions, and think laterally.  Prioritise originality over "
            "convention."
        ),
    },
    "critical": {
        "label": "Critical Processor",
        "system": (
            "You are the CRITICAL specialist processor.  Your domain is "
            "identifying flaws, hidden assumptions, logical fallacies, risks, "
            "failure modes, and blind spots.  Challenge every claim, stress-"
            "test every assertion, and find the weakest links.  Be "
            "constructively skeptical -- your role is to ensure quality by "
            "finding problems before they matter."
        ),
    },
    "integrative": {
        "label": "Integrative Processor",
        "system": (
            "You are the INTEGRATIVE specialist processor.  Your domain is "
            "synthesising diverse perspectives, finding common threads across "
            "different viewpoints, building coherent wholes from disparate "
            "parts, and reconciling apparent contradictions.  Seek the "
            "higher-order pattern that unifies different analyses."
        ),
    },
    "experiential": {
        "label": "Experiential Processor",
        "system": (
            "You are the EXPERIENTIAL specialist processor.  Your domain is "
            "practical experience, real-world implications, human factors, "
            "emotional dimensions, stakeholder perspectives, and embodied "
            "knowledge.  Consider how this plays out in practice, what people "
            "actually experience, and what the lived reality looks like "
            "beyond abstract analysis."
        ),
    },
}

_ATTENTION_SYSTEM = (
    "You are the attentional bottleneck of a global workspace.  Multiple "
    "specialist processors have analysed a problem in parallel.  Your job "
    "is to evaluate which specialist's contribution is MOST relevant, "
    "insightful, and valuable for the current stage of reasoning.  Consider "
    "salience, novelty, depth, and relevance to the query.\n\n"
    "Return a JSON object:\n"
    '{"winner": "specialist_name", "salience_scores": '
    '{"analytical": 0.7, "creative": 0.8, ...}, '
    '"rationale": "The creative processor won because..."}'
)

_BROADCAST_SYSTEM = (
    "You are the global broadcast mechanism.  The winning specialist's "
    "insight has been selected for conscious broadcast.  Frame this insight "
    "as a clear, impactful broadcast message that will be received by ALL "
    "other specialist processors.  The broadcast should be concise but "
    "complete -- it must convey the core insight in a way that can influence "
    "diverse forms of subsequent processing."
)

_FINAL_INTEGRATION_SYSTEM = (
    "You are producing the final integrated output of a global workspace "
    "reasoning process.  Multiple specialist processors (analytical, "
    "creative, critical, integrative, experiential) have processed the "
    "problem through several broadcast cycles.  Synthesize their collective "
    "insights into a comprehensive, well-structured final answer.  Draw on "
    "the best contributions from each specialist while maintaining coherence."
)


# ---------------------------------------------------------------------------
# GlobalWorkspace
# ---------------------------------------------------------------------------


class GlobalWorkspace(ReasoningEngine):
    """Global Workspace Theory: parallel specialists competing for broadcast.

    The engine simulates conscious cognition as described by Baars and
    Dehaene.  Five specialist processors work in parallel on the query,
    each bringing a different cognitive lens.  An attentional bottleneck
    selects the most salient contribution, which is then *broadcast*
    globally to all specialists.  Each specialist integrates the broadcast
    content with its own prior output and produces an updated analysis.

    This broadcast-integrate cycle repeats for ``broadcast_cycles``
    iterations, producing progressively richer and more integrated reasoning.

    Args:
        broadcast_cycles: Number of broadcast-integrate cycles (default 3).
        specialists: Optional list of specialist names to activate.  Must
            be a subset of ``{"analytical", "creative", "critical",
            "integrative", "experiential"}``.  Defaults to all five.

    Attributes:
        name: ``"global_workspace"``
        description: Short description of the engine.

    Example::

        engine = GlobalWorkspace(broadcast_cycles=2)
        trace = await engine.reason(
            query="Design a fault-tolerant notification system.",
            llm_provider=provider,
        )
        for step in trace.get_steps_by_type("attention_selection"):
            print(f"Winner: {step.metadata.get('winner')}")
    """

    name: str = "global_workspace"
    description: str = (
        "Parallel specialist processors compete for attentional broadcast "
        "in a global workspace, inspired by Baars/Dehaene conscious access theory."
    )

    def __init__(
        self,
        broadcast_cycles: int = 3,
        specialists: list[str] | None = None,
    ) -> None:
        self.broadcast_cycles = max(1, broadcast_cycles)
        valid = set(_SPECIALISTS.keys())
        if specialists is not None:
            unknown = set(specialists) - valid
            if unknown:
                raise ValueError(
                    f"Unknown specialists: {unknown}. Valid: {valid}"
                )
            self.specialists = list(specialists)
        else:
            self.specialists = list(_SPECIALISTS.keys())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the Global Workspace reasoning process.

        Args:
            query: The user question or task.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Ignored by this engine.
            max_iterations: Ignored -- ``broadcast_cycles`` controls depth.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with specialist, attention_selection,
            broadcast, and final_integration steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Root step
        query_step = self._make_step(
            step_type="query",
            content=query,
            metadata={"role": "user_query"},
        )
        trace.add_step(query_step)

        # State: latest output from each specialist
        specialist_outputs: dict[str, str] = {}
        broadcast_history: list[str] = []
        last_broadcast_step_id: str | None = None

        for cycle in range(self.broadcast_cycles):
            parent_id = last_broadcast_step_id or query_step.step_id

            # Phase 1: Parallel specialist processing
            specialist_step_ids: dict[str, str] = {}
            for spec_name in self.specialists:
                spec_output = await self._run_specialist(
                    spec_name=spec_name,
                    query=query,
                    cycle=cycle,
                    prior_output=specialist_outputs.get(spec_name),
                    broadcast_content=(
                        broadcast_history[-1] if broadcast_history else None
                    ),
                    provider=llm_provider,
                    trace=trace,
                )

                spec_step = self._make_step(
                    step_type=f"specialist_{spec_name}",
                    content=spec_output,
                    metadata={
                        "cycle": cycle,
                        "specialist": spec_name,
                        "label": _SPECIALISTS[spec_name]["label"],
                    },
                    parent_step_id=parent_id,
                )
                trace.add_step(spec_step)
                specialist_outputs[spec_name] = spec_output
                specialist_step_ids[spec_name] = spec_step.step_id

            # Phase 2: Attentional bottleneck -- select winner
            attention_result = await self._attention_selection(
                query=query,
                specialist_outputs=specialist_outputs,
                cycle=cycle,
                provider=llm_provider,
                trace=trace,
            )
            winner_name = attention_result.get("winner", self.specialists[0])
            salience_scores = attention_result.get("salience_scores", {})
            rationale = attention_result.get("rationale", "")

            # Ensure winner is valid
            if winner_name not in self.specialists:
                winner_name = self.specialists[0]

            winner_score = float(salience_scores.get(winner_name, 0.7))

            attn_step = self._make_step(
                step_type="attention_selection",
                content=(
                    f"Cycle {cycle}: {winner_name} wins the attentional "
                    f"competition.  {rationale}"
                ),
                score=winner_score,
                metadata={
                    "cycle": cycle,
                    "winner": winner_name,
                    "salience_scores": salience_scores,
                    "rationale": rationale,
                },
                parent_step_id=specialist_step_ids.get(
                    winner_name, parent_id,
                ),
            )
            trace.add_step(attn_step)

            # Phase 3: Global broadcast
            broadcast_content = await self._broadcast(
                query=query,
                winner_name=winner_name,
                winner_output=specialist_outputs[winner_name],
                cycle=cycle,
                provider=llm_provider,
                trace=trace,
            )
            broadcast_history.append(broadcast_content)

            broadcast_step = self._make_step(
                step_type="broadcast",
                content=broadcast_content,
                score=winner_score,
                metadata={
                    "cycle": cycle,
                    "winner": winner_name,
                    "broadcast_number": len(broadcast_history),
                },
                parent_step_id=attn_step.step_id,
            )
            trace.add_step(broadcast_step)
            last_broadcast_step_id = broadcast_step.step_id

        # Phase 4: Final integration across all specialists
        final_output = await self._final_integration(
            query=query,
            specialist_outputs=specialist_outputs,
            broadcast_history=broadcast_history,
            provider=llm_provider,
            trace=trace,
        )

        integration_step = self._make_step(
            step_type="final_integration",
            content=final_output,
            score=1.0,
            metadata={
                "total_cycles": self.broadcast_cycles,
                "broadcast_winners": [
                    step.metadata.get("winner", "?")
                    for step in trace.get_steps_by_type("attention_selection")
                ],
            },
            parent_step_id=last_broadcast_step_id or query_step.step_id,
        )
        trace.add_step(integration_step)

        final_step = self._make_step(
            step_type="final_output",
            content=final_output,
            score=1.0,
            metadata={"total_cycles": self.broadcast_cycles},
            parent_step_id=integration_step.step_id,
        )
        trace.add_step(final_step)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _run_specialist(
        self,
        spec_name: str,
        query: str,
        cycle: int,
        prior_output: str | None,
        broadcast_content: str | None,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Run a single specialist processor on the query.

        On cycle 0, the specialist processes the raw query.  On subsequent
        cycles, it integrates the latest broadcast content with its own
        prior analysis.

        Args:
            spec_name: Name of the specialist (e.g. ``"analytical"``).
            query: Original user query.
            cycle: Current broadcast cycle.
            prior_output: This specialist's output from the previous cycle.
            broadcast_content: The most recent global broadcast content.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            The specialist's analysis text.
        """
        spec_info = _SPECIALISTS[spec_name]

        if cycle == 0:
            user_content = (
                f"Process the following query from your specialist "
                f"perspective.\n\n"
                f"Query: {query}\n\n"
                f"Provide your analysis from the {spec_name} perspective. "
                f"Be thorough but focused on your domain of expertise."
            )
        else:
            user_content = (
                f"A new insight has been broadcast to the global workspace. "
                f"Integrate it with your prior analysis and update your "
                f"contribution.\n\n"
                f"Query: {query}\n\n"
                f"---YOUR PRIOR ANALYSIS---\n{prior_output}\n"
                f"---END PRIOR ANALYSIS---\n\n"
                f"---GLOBAL BROADCAST (cycle {cycle - 1})---\n"
                f"{broadcast_content}\n"
                f"---END BROADCAST---\n\n"
                f"How does this broadcast insight affect your {spec_name} "
                f"analysis?  Update and deepen your contribution in light "
                f"of this new information."
            )

        # Higher cycles get slightly lower temperature as reasoning converges
        temperature = max(0.3, 0.7 - 0.1 * cycle)

        messages = [{"role": "user", "content": user_content}]
        return await self._call_llm(
            provider, messages, trace,
            system=spec_info["system"],
            temperature=temperature,
        )

    async def _attention_selection(
        self,
        query: str,
        specialist_outputs: dict[str, str],
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Run the attentional bottleneck to select a broadcast winner.

        Evaluates all specialist outputs for salience and selects the one
        most relevant for the current reasoning stage.

        Args:
            query: Original query.
            specialist_outputs: Current outputs from all specialists.
            cycle: Current broadcast cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Dict with ``winner``, ``salience_scores``, and ``rationale``.
        """
        outputs_text = "\n\n".join(
            f"---{name.upper()} SPECIALIST---\n{output}\n"
            f"---END {name.upper()}---"
            for name, output in specialist_outputs.items()
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Evaluate which specialist's contribution should win "
                    f"access to the global workspace broadcast in cycle "
                    f"{cycle}.\n\n"
                    f"Query: {query}\n\n"
                    f"{outputs_text}\n\n"
                    f"Score each specialist on salience (0.0-1.0) and select "
                    f"a winner.  Return JSON with: winner (string), "
                    f"salience_scores (dict of name->float), rationale (string)."
                ),
            },
        ]
        raw = await self._call_llm(
            provider, messages, trace,
            system=_ATTENTION_SYSTEM,
            temperature=0.3,
        )
        return self._parse_attention_json(raw)

    async def _broadcast(
        self,
        query: str,
        winner_name: str,
        winner_output: str,
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Create the global broadcast message from the winning specialist.

        Args:
            query: Original query.
            winner_name: Name of the winning specialist.
            winner_output: The winning specialist's output.
            cycle: Current broadcast cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            The broadcast message text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"The {winner_name} specialist has won the attentional "
                    f"competition in cycle {cycle}.  Frame their key insight "
                    f"as a global broadcast message.\n\n"
                    f"Query: {query}\n\n"
                    f"---WINNING ANALYSIS ({winner_name})---\n{winner_output}\n"
                    f"---END ANALYSIS---\n\n"
                    f"Create a concise but complete broadcast that conveys the "
                    f"core insight to all other specialist processors."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_BROADCAST_SYSTEM,
            temperature=0.4,
        )

    async def _final_integration(
        self,
        query: str,
        specialist_outputs: dict[str, str],
        broadcast_history: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Produce the final integrated output.

        Args:
            query: Original query.
            specialist_outputs: Final outputs from all specialists.
            broadcast_history: All broadcast messages across cycles.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            The final comprehensive answer.
        """
        outputs_text = "\n\n".join(
            f"---{name.upper()} (FINAL)---\n{output}\n---END {name.upper()}---"
            for name, output in specialist_outputs.items()
        )
        broadcast_text = "\n\n".join(
            f"Broadcast {i}: {bc}" for i, bc in enumerate(broadcast_history)
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Produce the final integrated answer by synthesizing "
                    f"all specialist contributions after {len(broadcast_history)} "
                    f"broadcast cycles.\n\n"
                    f"Query: {query}\n\n"
                    f"---SPECIALIST OUTPUTS---\n{outputs_text}\n"
                    f"---END SPECIALISTS---\n\n"
                    f"---BROADCAST HISTORY---\n{broadcast_text}\n"
                    f"---END BROADCASTS---\n\n"
                    f"Synthesize a comprehensive, well-structured answer that "
                    f"integrates the best insights from all specialists."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_FINAL_INTEGRATION_SYSTEM,
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_attention_json(self, raw: str) -> dict[str, Any]:
        """Parse the attention selection JSON from LLM output.

        Tries ``json.loads`` first, falls back to regex extraction.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``winner``, ``salience_scores``, and ``rationale``.
        """
        text = raw.strip()

        # Try JSON
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict) and "winner" in parsed:
                    parsed.setdefault("salience_scores", {})
                    parsed.setdefault("rationale", "")
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try to extract winner name from text
        winner = self.specialists[0]  # default
        for spec_name in self.specialists:
            pattern = rf"winner[\"'\s:]*[\"']?{re.escape(spec_name)}[\"']?"
            if re.search(pattern, text, re.I):
                winner = spec_name
                break

        return {
            "winner": winner,
            "salience_scores": {s: 0.5 for s in self.specialists},
            "rationale": raw[:200],
        }


__all__ = ["GlobalWorkspace"]
