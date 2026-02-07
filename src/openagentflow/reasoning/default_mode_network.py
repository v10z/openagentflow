"""Default Mode Network reasoning engine for OpenAgentFlow.

Based on Marcus Raichle's discovery of the Default Mode Network (Raichle et
al. 2001; Buckner, Andrews-Hanna & Schacter 2008) and subsequent research on
spontaneous thought, mind-wandering, and creativity (Christoff et al. 2016;
Beaty et al. 2016).

The Default Mode Network (DMN) is a large-scale brain network that becomes
*more* active when a person is not focused on external tasks -- during
daydreaming, mind-wandering, autobiographical memory recall, imagining the
future, and thinking about others' mental states.  It was initially dismissed
as mere "noise," but Raichle's work revealed it as a structured, metabolically
expensive network performing crucial cognitive functions.

The DMN comprises several interacting subsystems:

- **Medial prefrontal cortex (mPFC)**: Self-referential processing, social
  cognition, theory of mind.
- **Posterior cingulate cortex (PCC) / precuneus**: Autobiographical memory,
  scene construction, integrating information across time.
- **Medial temporal lobe (MTL)**: Episodic memory, imagining novel scenarios,
  recombining past experiences into future simulations.
- **Temporal parietal junction (TPJ)**: Perspective-taking, mentalizing,
  understanding others' beliefs and intentions.
- **Lateral temporal cortex**: Semantic memory, conceptual knowledge.

Crucially, the DMN and the Task-Positive Network (TPN -- dorsal attention
network, frontoparietal control network) are *anti-correlated*: when one is
active, the other is suppressed.  However, creative cognition requires
dynamic *cooperation* between the DMN and TPN.  Beaty et al. (2016)
showed that highly creative individuals exhibit stronger coupling between
DMN and executive control regions.

The engine alternates between:

- **TPN mode** (focused analytical work) -- systematic, structured reasoning
  directed at the problem.
- **DMN mode** (deliberate mind-wandering) -- unconstrained, associative
  thought with three sub-phases:
  1. **Free association**: Unconstrained associative wandering.
  2. **Scenario simulation**: Imagining concrete future scenarios.
  3. **Perspective-taking**: Considering the problem from other viewpoints.

After each DMN phase, a *harvest* step extracts creative insights and
integrates them back into the focused analysis.

Example::

    from openagentflow.reasoning.default_mode_network import DefaultModeNetwork

    engine = DefaultModeNetwork(oscillation_cycles=3)
    trace = await engine.reason(
        query="How should we redesign the onboarding experience?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # See insights harvested from mind-wandering
    for step in trace.get_steps_by_type("harvest"):
        cycle = step.metadata.get("cycle", "?")
        print(f"Cycle {cycle} harvest: {step.content[:120]}...")

Trace structure (DAG)::

    query
      +-- tpn_analysis_0
      |     +-- dmn_free_association_0
      |     +-- dmn_scenario_simulation_0
      |     +-- dmn_perspective_taking_0
      |     +-- harvest_0
      |           +-- tpn_analysis_1
      |                 +-- dmn_free_association_1
      |                 +-- dmn_scenario_simulation_1
      |                 +-- dmn_perspective_taking_1
      |                 +-- harvest_1
      |                       +-- ...
      +-- final_integration
      +-- final_output
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TPN_SYSTEM = (
    "You are operating in TASK-POSITIVE NETWORK mode -- focused, analytical, "
    "systematic reasoning.  Approach the problem with structured thinking: "
    "decompose it, identify key variables, apply logical frameworks, and "
    "build a coherent analysis.  Be rigorous, precise, and directed.  "
    "This is the focused-attention counterpart to mind-wandering."
)

_DMN_FREE_ASSOCIATION_SYSTEM = (
    "You are operating in DEFAULT MODE NETWORK mode -- specifically the FREE "
    "ASSOCIATION sub-phase.  Let your mind wander freely.  Do not try to be "
    "structured or logical.  Instead, follow chains of association wherever "
    "they lead.  Make unexpected connections.  Think of analogies from "
    "completely different domains.  Let memories, images, and fragments of "
    "ideas surface spontaneously.  Embrace tangents and digressions.  The "
    "goal is NOT to solve the problem directly but to generate raw creative "
    "material that might contain seeds of insight."
)

_DMN_SCENARIO_SIMULATION_SYSTEM = (
    "You are operating in DEFAULT MODE NETWORK mode -- specifically the "
    "SCENARIO SIMULATION sub-phase (medial temporal lobe function).  "
    "Imagine vivid, concrete scenarios related to the problem.  What would "
    "it actually look like if different solutions were implemented?  "
    "Simulate future situations in detail: who is there, what happens, "
    "what goes wrong, what unexpected things occur?  Think of this as "
    "mental time travel -- projecting yourself into possible futures and "
    "exploring them experientially, not analytically."
)

_DMN_PERSPECTIVE_TAKING_SYSTEM = (
    "You are operating in DEFAULT MODE NETWORK mode -- specifically the "
    "PERSPECTIVE-TAKING sub-phase (temporal parietal junction function).  "
    "Consider the problem from the viewpoints of different stakeholders, "
    "personas, and even inanimate systems.  What would a novice think?  "
    "An expert in a completely different field?  A future user?  A critic?  "
    "Someone from a different culture or background?  Try to genuinely "
    "inhabit each perspective, not just list them."
)

_HARVEST_SYSTEM = (
    "You are harvesting creative insights from a mind-wandering session.  "
    "The Default Mode Network has produced free associations, scenario "
    "simulations, and perspective-taking explorations.  Your job is to sift "
    "through this raw creative material and extract the genuinely valuable "
    "insights -- ideas, connections, reframings, or concerns that the "
    "focused analytical mind might have missed.  Be selective: not "
    "everything from mind-wandering is useful, but the gems are often "
    "transformative."
)

_FINAL_INTEGRATION_SYSTEM = (
    "You are producing the final integrated output after alternating between "
    "focused analytical work (Task-Positive Network) and deliberate "
    "mind-wandering (Default Mode Network).  The oscillation between these "
    "modes has produced both rigorous analysis AND creative insights.  "
    "Synthesize everything into a comprehensive answer that benefits from "
    "both analytical depth and creative breadth.  The best answers emerge "
    "from the dynamic interplay between focus and wandering."
)


# ---------------------------------------------------------------------------
# DefaultModeNetwork
# ---------------------------------------------------------------------------


class DefaultModeNetwork(ReasoningEngine):
    """Default Mode Network: oscillating between focused work and mind-wandering.

    Alternates between Task-Positive Network (TPN) mode for structured
    analytical reasoning and Default Mode Network (DMN) mode for creative
    mind-wandering.  The DMN phase includes three sub-phases inspired by
    distinct DMN subsystems:

    1. **Free Association** -- Unconstrained associative thought
       (lateral temporal cortex / semantic memory).
    2. **Scenario Simulation** -- Vivid mental simulation of concrete
       future scenarios (medial temporal lobe / hippocampus).
    3. **Perspective-Taking** -- Adopting diverse viewpoints
       (temporal parietal junction / mentalizing).

    After each DMN phase, a *harvest* step extracts useful creative
    insights.  These are fed back into the next TPN analytical cycle,
    creating a productive oscillation between focus and creativity.

    Args:
        oscillation_cycles: Number of TPN-DMN oscillation cycles
            (default 3).
        dmn_temperature: LLM temperature for DMN (mind-wandering) phases
            (default 0.9).  Higher values produce more divergent thought.
        tpn_temperature: LLM temperature for TPN (focused) phases
            (default 0.4).  Lower values produce more structured analysis.
        enable_free_association: Whether to include the free association
            DMN sub-phase (default True).
        enable_scenario_simulation: Whether to include scenario simulation
            (default True).
        enable_perspective_taking: Whether to include perspective-taking
            (default True).

    Attributes:
        name: ``"default_mode_network"``
        description: Short description of the engine.

    Example::

        engine = DefaultModeNetwork(
            oscillation_cycles=3,
            dmn_temperature=1.0,
        )
        trace = await engine.reason(
            query="Design a new approach to code review.",
            llm_provider=provider,
        )
        print(trace.final_output)
    """

    name: str = "default_mode_network"
    description: str = (
        "Alternates between focused analytical work (TPN) and deliberate "
        "mind-wandering (DMN) with free association, scenario simulation, "
        "and perspective-taking sub-phases."
    )

    def __init__(
        self,
        oscillation_cycles: int = 3,
        dmn_temperature: float = 0.9,
        tpn_temperature: float = 0.4,
        enable_free_association: bool = True,
        enable_scenario_simulation: bool = True,
        enable_perspective_taking: bool = True,
    ) -> None:
        self.oscillation_cycles = max(1, oscillation_cycles)
        self.dmn_temperature = max(0.3, min(1.5, dmn_temperature))
        self.tpn_temperature = max(0.1, min(0.8, tpn_temperature))
        self.enable_free_association = enable_free_association
        self.enable_scenario_simulation = enable_scenario_simulation
        self.enable_perspective_taking = enable_perspective_taking

        # At least one DMN sub-phase must be enabled
        if not any([
            enable_free_association,
            enable_scenario_simulation,
            enable_perspective_taking,
        ]):
            self.enable_free_association = True

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
        """Execute the TPN-DMN oscillation reasoning process.

        Args:
            query: The user question or task.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Ignored by this engine.
            max_iterations: Ignored -- ``oscillation_cycles`` controls depth.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with tpn_analysis, dmn_free_association,
            dmn_scenario_simulation, dmn_perspective_taking, harvest, and
            final_integration steps.
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

        # Track accumulated analysis and insights
        tpn_analyses: list[str] = []
        harvested_insights: list[str] = []
        last_step_id = query_step.step_id

        for cycle in range(self.oscillation_cycles):
            # ---- TPN MODE: Focused analytical work ----
            tpn_output = await self._tpn_analysis(
                query=query,
                cycle=cycle,
                prior_analyses=tpn_analyses,
                prior_insights=harvested_insights,
                provider=llm_provider,
                trace=trace,
            )
            tpn_analyses.append(tpn_output)

            tpn_step = self._make_step(
                step_type="tpn_analysis",
                content=tpn_output,
                metadata={
                    "cycle": cycle,
                    "mode": "task_positive_network",
                    "temperature": self.tpn_temperature,
                },
                parent_step_id=last_step_id,
            )
            trace.add_step(tpn_step)

            # ---- DMN MODE: Mind-wandering with 3 sub-phases ----
            dmn_outputs: dict[str, str] = {}
            dmn_step_ids: list[str] = []

            # Sub-phase 1: Free Association
            if self.enable_free_association:
                fa_output = await self._dmn_free_association(
                    query=query,
                    tpn_analysis=tpn_output,
                    cycle=cycle,
                    provider=llm_provider,
                    trace=trace,
                )
                dmn_outputs["free_association"] = fa_output

                fa_step = self._make_step(
                    step_type="dmn_free_association",
                    content=fa_output,
                    metadata={
                        "cycle": cycle,
                        "mode": "default_mode_network",
                        "sub_phase": "free_association",
                        "temperature": self.dmn_temperature,
                    },
                    parent_step_id=tpn_step.step_id,
                )
                trace.add_step(fa_step)
                dmn_step_ids.append(fa_step.step_id)

            # Sub-phase 2: Scenario Simulation
            if self.enable_scenario_simulation:
                ss_output = await self._dmn_scenario_simulation(
                    query=query,
                    tpn_analysis=tpn_output,
                    cycle=cycle,
                    provider=llm_provider,
                    trace=trace,
                )
                dmn_outputs["scenario_simulation"] = ss_output

                ss_step = self._make_step(
                    step_type="dmn_scenario_simulation",
                    content=ss_output,
                    metadata={
                        "cycle": cycle,
                        "mode": "default_mode_network",
                        "sub_phase": "scenario_simulation",
                        "temperature": self.dmn_temperature,
                    },
                    parent_step_id=tpn_step.step_id,
                )
                trace.add_step(ss_step)
                dmn_step_ids.append(ss_step.step_id)

            # Sub-phase 3: Perspective-Taking
            if self.enable_perspective_taking:
                pt_output = await self._dmn_perspective_taking(
                    query=query,
                    tpn_analysis=tpn_output,
                    cycle=cycle,
                    provider=llm_provider,
                    trace=trace,
                )
                dmn_outputs["perspective_taking"] = pt_output

                pt_step = self._make_step(
                    step_type="dmn_perspective_taking",
                    content=pt_output,
                    metadata={
                        "cycle": cycle,
                        "mode": "default_mode_network",
                        "sub_phase": "perspective_taking",
                        "temperature": self.dmn_temperature,
                    },
                    parent_step_id=tpn_step.step_id,
                )
                trace.add_step(pt_step)
                dmn_step_ids.append(pt_step.step_id)

            # ---- HARVEST: Extract creative insights from DMN ----
            harvest_output = await self._harvest_insights(
                query=query,
                tpn_analysis=tpn_output,
                dmn_outputs=dmn_outputs,
                cycle=cycle,
                provider=llm_provider,
                trace=trace,
            )
            harvested_insights.append(harvest_output)

            # Parent the harvest step to the last DMN sub-phase
            harvest_parent = dmn_step_ids[-1] if dmn_step_ids else tpn_step.step_id
            harvest_step = self._make_step(
                step_type="harvest",
                content=harvest_output,
                metadata={
                    "cycle": cycle,
                    "dmn_sub_phases": list(dmn_outputs.keys()),
                    "num_insights_total": len(harvested_insights),
                },
                parent_step_id=harvest_parent,
            )
            trace.add_step(harvest_step)
            last_step_id = harvest_step.step_id

        # ---- FINAL INTEGRATION ----
        final_output = await self._final_integration(
            query=query,
            tpn_analyses=tpn_analyses,
            harvested_insights=harvested_insights,
            provider=llm_provider,
            trace=trace,
        )

        integration_step = self._make_step(
            step_type="final_integration",
            content=final_output,
            score=1.0,
            metadata={
                "total_cycles": self.oscillation_cycles,
                "total_tpn_analyses": len(tpn_analyses),
                "total_harvested_insights": len(harvested_insights),
            },
            parent_step_id=last_step_id,
        )
        trace.add_step(integration_step)

        final_step = self._make_step(
            step_type="final_output",
            content=final_output,
            score=1.0,
            metadata={"total_cycles": self.oscillation_cycles},
            parent_step_id=integration_step.step_id,
        )
        trace.add_step(final_step)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # TPN (focused) phases
    # ------------------------------------------------------------------

    async def _tpn_analysis(
        self,
        query: str,
        cycle: int,
        prior_analyses: list[str],
        prior_insights: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Run focused analytical reasoning (Task-Positive Network mode).

        On cycle 0 this is a fresh analysis of the query.  On subsequent
        cycles it builds on prior analyses and incorporates creative
        insights harvested from DMN phases.

        Args:
            query: Original user query.
            cycle: Current oscillation cycle.
            prior_analyses: TPN analyses from previous cycles.
            prior_insights: Harvested insights from previous DMN phases.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Focused analytical output.
        """
        if cycle == 0:
            user_content = (
                f"Analyse the following query with focused, systematic "
                f"reasoning.  Decompose the problem, identify key factors, "
                f"and build a structured analysis.\n\n"
                f"Query: {query}"
            )
        else:
            latest_analysis = prior_analyses[-1] if prior_analyses else ""
            latest_insight = prior_insights[-1] if prior_insights else ""
            user_content = (
                f"Continue your focused analysis, now enriched by creative "
                f"insights from mind-wandering.\n\n"
                f"Query: {query}\n\n"
                f"---YOUR PRIOR ANALYSIS (cycle {cycle - 1})---\n"
                f"{latest_analysis}\n---END PRIOR ANALYSIS---\n\n"
                f"---CREATIVE INSIGHTS FROM MIND-WANDERING---\n"
                f"{latest_insight}\n---END INSIGHTS---\n\n"
                f"Integrate any valuable insights from the mind-wandering "
                f"phase into your analysis.  Deepen, refine, and extend "
                f"your reasoning.  Be open to reframing if the creative "
                f"insights suggest a better angle."
            )

        messages = [{"role": "user", "content": user_content}]
        return await self._call_llm(
            provider, messages, trace,
            system=_TPN_SYSTEM,
            temperature=self.tpn_temperature,
        )

    # ------------------------------------------------------------------
    # DMN (mind-wandering) phases
    # ------------------------------------------------------------------

    async def _dmn_free_association(
        self,
        query: str,
        tpn_analysis: str,
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Run the free association sub-phase of DMN.

        The LLM is prompted to wander freely, making unexpected
        associations and following chains of thought without constraint.

        Args:
            query: Original query.
            tpn_analysis: The most recent focused analysis.
            cycle: Current oscillation cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Free association output.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Let your mind wander freely about this topic.  Do not "
                    f"try to be structured or solve the problem.  Instead, "
                    f"follow associations wherever they lead.\n\n"
                    f"Query: {query}\n\n"
                    f"The focused analysis found:\n{tpn_analysis[:500]}...\n\n"
                    f"Now let go of that structure.  What comes to mind?  "
                    f"What analogies from completely different fields?  What "
                    f"memories or images?  What surprising connections?  "
                    f"Follow the thread of association freely."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_DMN_FREE_ASSOCIATION_SYSTEM,
            temperature=self.dmn_temperature,
        )

    async def _dmn_scenario_simulation(
        self,
        query: str,
        tpn_analysis: str,
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Run the scenario simulation sub-phase of DMN.

        The LLM is prompted to vividly imagine concrete future scenarios,
        simulating what different outcomes would actually look and feel like.

        Args:
            query: Original query.
            tpn_analysis: The most recent focused analysis.
            cycle: Current oscillation cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Scenario simulation output.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Imagine vivid, concrete scenarios related to this "
                    f"problem.  Do not analyse -- simulate.  What would it "
                    f"actually look like?\n\n"
                    f"Query: {query}\n\n"
                    f"Current understanding:\n{tpn_analysis[:500]}...\n\n"
                    f"Simulate 2-3 concrete future scenarios.  For each: "
                    f"set the scene, describe what happens step by step, "
                    f"note what goes right and what goes wrong, and capture "
                    f"any unexpected twists.  Make them vivid and specific, "
                    f"like watching a film."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_DMN_SCENARIO_SIMULATION_SYSTEM,
            temperature=self.dmn_temperature,
        )

    async def _dmn_perspective_taking(
        self,
        query: str,
        tpn_analysis: str,
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Run the perspective-taking sub-phase of DMN.

        The LLM is prompted to genuinely inhabit different perspectives
        and viewpoints on the problem.

        Args:
            query: Original query.
            tpn_analysis: The most recent focused analysis.
            cycle: Current oscillation cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Perspective-taking output.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Consider this problem from radically different "
                    f"perspectives.  Genuinely inhabit each viewpoint.\n\n"
                    f"Query: {query}\n\n"
                    f"Current understanding:\n{tpn_analysis[:500]}...\n\n"
                    f"Consider at least 3 very different perspectives:\n"
                    f"- Someone who would be most affected by this\n"
                    f"- An expert from a completely unrelated field\n"
                    f"- A skeptic who thinks the premise is wrong\n"
                    f"- A future person looking back at this decision\n\n"
                    f"For each perspective, genuinely try to see the world "
                    f"through their eyes.  What do they notice that you "
                    f"missed?  What do they care about that you overlooked?"
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_DMN_PERSPECTIVE_TAKING_SYSTEM,
            temperature=self.dmn_temperature,
        )

    # ------------------------------------------------------------------
    # Harvest and integration
    # ------------------------------------------------------------------

    async def _harvest_insights(
        self,
        query: str,
        tpn_analysis: str,
        dmn_outputs: dict[str, str],
        cycle: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Harvest creative insights from DMN mind-wandering outputs.

        Sifts through the raw creative material from all DMN sub-phases
        and extracts genuinely valuable insights, connections, and
        reframings.

        Args:
            query: Original query.
            tpn_analysis: The focused analysis that preceded this DMN phase.
            dmn_outputs: Dict mapping sub-phase name to output text.
            cycle: Current oscillation cycle.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Harvested insights text.
        """
        dmn_text_parts = []
        for phase_name, output in dmn_outputs.items():
            label = phase_name.replace("_", " ").title()
            dmn_text_parts.append(
                f"---{label.upper()}---\n{output}\n---END {label.upper()}---"
            )
        dmn_text = "\n\n".join(dmn_text_parts)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Harvest the valuable creative insights from the "
                    f"mind-wandering session below.\n\n"
                    f"Query: {query}\n\n"
                    f"---FOCUSED ANALYSIS (pre-wandering)---\n"
                    f"{tpn_analysis}\n---END ANALYSIS---\n\n"
                    f"---MIND-WANDERING OUTPUTS---\n{dmn_text}\n"
                    f"---END WANDERING---\n\n"
                    f"Extract the genuinely valuable insights -- ideas, "
                    f"connections, reframings, warnings, or creative "
                    f"approaches that the focused analysis missed.  Be "
                    f"selective: not all wandering is useful, but the gems "
                    f"can be transformative.  For each insight, explain WHY "
                    f"it matters for the query."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_HARVEST_SYSTEM,
            temperature=0.5,
        )

    async def _final_integration(
        self,
        query: str,
        tpn_analyses: list[str],
        harvested_insights: list[str],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Produce the final integrated answer.

        Synthesizes all focused analyses and creative insights from the
        entire oscillation process.

        Args:
            query: Original query.
            tpn_analyses: All TPN analysis outputs across cycles.
            harvested_insights: All harvested insight outputs across cycles.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Final comprehensive answer.
        """
        # Include the latest analysis and all insights
        latest_analysis = tpn_analyses[-1] if tpn_analyses else ""
        all_insights = "\n\n".join(
            f"--- Cycle {i} insights ---\n{ins}"
            for i, ins in enumerate(harvested_insights)
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Produce the final integrated answer after "
                    f"{len(tpn_analyses)} cycles of alternating between "
                    f"focused analysis and creative mind-wandering.\n\n"
                    f"Query: {query}\n\n"
                    f"---LATEST FOCUSED ANALYSIS---\n{latest_analysis}\n"
                    f"---END ANALYSIS---\n\n"
                    f"---ALL HARVESTED CREATIVE INSIGHTS---\n{all_insights}\n"
                    f"---END INSIGHTS---\n\n"
                    f"Synthesize a comprehensive, well-structured final "
                    f"answer that draws on both the rigorous analytical work "
                    f"AND the creative insights from mind-wandering.  The "
                    f"best answers emerge from the interplay between "
                    f"disciplined focus and unconstrained creativity."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_FINAL_INTEGRATION_SYSTEM,
            temperature=0.4,
        )


__all__ = ["DefaultModeNetwork"]
