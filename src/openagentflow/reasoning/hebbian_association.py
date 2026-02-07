"""Hebbian Association reasoning engine for OpenAgentFlow.

Based on Donald Hebb's learning rule (1949) -- "neurons that fire together
wire together" -- and the spreading activation model of semantic memory
(Collins & Loftus 1975; Anderson 1983).

Hebb's postulate states that when neuron A repeatedly participates in
firing neuron B, the synaptic connection from A to B is strengthened.
This is the foundational principle of associative learning and underpins
long-term potentiation (LTP) -- the primary cellular mechanism of memory
formation in the hippocampus and cortex.

The complementary process, long-term depression (LTD), weakens connections
between neurons that fire out of synchrony or in opposition.  Together, LTP
and LTD sculpt the associative network, strengthening productive pathways
and pruning contradictory or irrelevant ones.

Spreading activation describes how activating one concept in a semantic
network causes activation to propagate along associative links to related
concepts, with activation decaying over distance.  This explains priming
effects, creative associations, and the structure of semantic memory.

The engine implements this as follows:

1. **Concept Extraction** -- Extract key concepts from the query and build
   an initial associative network with weighted connections.
2. **Spreading Activation** -- Activate the most central concepts and
   propagate activation through the network.
3. **Hebbian Learning (LTP)** -- Strengthen connections between
   co-activated concepts that produce coherent, useful reasoning.
4. **Long-Term Depression (LTD)** -- Weaken connections between concepts
   that produce contradictions or incoherent reasoning.
5. **Cluster Identification** -- Identify the dominant concept cluster
   (most strongly interconnected, highest total activation).
6. **Synthesis** -- Synthesise the final answer from the dominant cluster.

Example::

    from openagentflow.reasoning.hebbian_association import HebbianAssociation

    engine = HebbianAssociation(max_iterations=4, activation_decay=0.3)
    trace = await engine.reason(
        query="What makes a programming language successful?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # Inspect the concept network evolution
    for step in trace.get_steps_by_type("hebbian_update"):
        it = step.metadata.get("iteration", "?")
        strengthened = step.metadata.get("connections_strengthened", 0)
        weakened = step.metadata.get("connections_weakened", 0)
        print(f"Iteration {it}: +{strengthened} / -{weakened} connections")

Trace structure (DAG)::

    query
      +-- concept_extraction
      +-- activation_spread_0
      |     +-- reasoning_probe_0
      |           +-- hebbian_update_0
      |                 +-- activation_spread_1
      |                       +-- reasoning_probe_1
      |                             +-- hebbian_update_1
      |                                   +-- ...
      +-- cluster_identification
      +-- synthesis
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
# Prompts
# ---------------------------------------------------------------------------

_CONCEPT_EXTRACTION_SYSTEM = (
    "You are a semantic network builder.  Given a query, extract the key "
    "concepts and build an associative network.  For each concept, assess "
    "its relevance (0.0-1.0) to the query.  For each pair of related "
    "concepts, estimate the connection strength (0.0-1.0) and whether the "
    "relationship is synergistic (+) or antagonistic (-).\n\n"
    "Return a JSON object:\n"
    '{"concepts": [{"name": "concept_name", "relevance": 0.8, '
    '"description": "..."}], '
    '"connections": [{"from": "A", "to": "B", "weight": 0.7, '
    '"type": "synergistic", "description": "..."}]}'
)

_ACTIVATION_SPREAD_SYSTEM = (
    "You are simulating spreading activation through a semantic network.  "
    "Given the current network state with concept activations and connection "
    "weights, determine which concepts become activated and to what degree.  "
    "Activation spreads from highly active concepts along strong connections, "
    "decaying with distance.  Report the new activation levels.\n\n"
    "Return a JSON object:\n"
    '{"activations": [{"concept": "name", "activation": 0.8, '
    '"activated_by": "source_concept", "reasoning": "..."}], '
    '"narrative": "..."}'
)

_REASONING_PROBE_SYSTEM = (
    "You are probing a cluster of co-activated concepts to generate "
    "reasoning.  Given a set of simultaneously active concepts, explore "
    "what insights emerge from their co-activation.  What does it mean "
    "that these concepts are active together?  What implications, "
    "connections, and conclusions follow from their conjunction?"
)

_HEBBIAN_UPDATE_SYSTEM = (
    "You are applying Hebbian learning to an associative network.  Given "
    "the reasoning produced by co-activated concepts, determine which "
    "connections should be STRENGTHENED (LTP -- the co-activation produced "
    "useful, coherent insight) and which should be WEAKENED (LTD -- the "
    "co-activation produced contradictions or irrelevant noise).\n\n"
    "Return a JSON object:\n"
    '{"strengthen": [{"from": "A", "to": "B", "delta": 0.15, '
    '"reason": "produced coherent insight about..."}], '
    '"weaken": [{"from": "C", "to": "D", "delta": -0.1, '
    '"reason": "produced contradiction..."}], '
    '"narrative": "..."}'
)

_CLUSTER_IDENTIFICATION_SYSTEM = (
    "You are identifying the dominant concept cluster in an associative "
    "network after multiple rounds of Hebbian learning.  Given the final "
    "network state, identify the cluster of most strongly interconnected, "
    "most highly activated concepts.  This cluster represents the core "
    "conceptual structure for answering the query.\n\n"
    "Return a JSON object:\n"
    '{"dominant_cluster": ["concept1", "concept2", ...], '
    '"cluster_coherence": 0.85, '
    '"cluster_narrative": "This cluster represents...", '
    '"secondary_clusters": [["other1", "other2"]]}'
)

_SYNTHESIS_SYSTEM = (
    "You are synthesizing a final answer from the dominant concept cluster "
    "in an associative network.  The network has been shaped by multiple "
    "rounds of spreading activation and Hebbian learning.  The dominant "
    "cluster represents the core conceptual structure.  Build a "
    "comprehensive, well-structured answer that weaves together the "
    "cluster concepts and their relationships."
)


# ---------------------------------------------------------------------------
# HebbianAssociation
# ---------------------------------------------------------------------------


class HebbianAssociation(ReasoningEngine):
    """Hebbian associative reasoning via spreading activation and LTP/LTD.

    Builds a concept network from the query, propagates activation through
    it, strengthens productive associations (LTP), weakens contradictory
    ones (LTD), and synthesises from the dominant cluster.

    At each iteration the engine:

    1. **Spreads activation** from high-activation concepts along weighted
       connections, with configurable decay.
    2. **Probes** the co-activated cluster for reasoning insights.
    3. **Applies Hebbian learning** -- strengthening connections that produce
       coherent reasoning (LTP) and weakening those that produce
       contradictions (LTD).

    After all iterations, the dominant concept cluster is identified and
    used as the scaffold for final synthesis.

    Args:
        max_iterations: Number of activation-learning cycles (default 3).
        activation_decay: How much activation decays per hop (default 0.3).
            Higher values mean activation spreads less far.
        learning_rate: Scaling factor for connection weight updates
            (default 1.0).
        min_concepts: Minimum number of concepts to extract (default 6).
        max_concepts: Maximum number of concepts to extract (default 12).

    Attributes:
        name: ``"hebbian_association"``
        description: Short description of the engine.

    Example::

        engine = HebbianAssociation(max_iterations=4, activation_decay=0.25)
        trace = await engine.reason(
            query="What are the key principles of good API design?",
            llm_provider=provider,
        )
        print(trace.final_output)
    """

    name: str = "hebbian_association"
    description: str = (
        "Builds an associative concept network, propagates spreading "
        "activation, and applies Hebbian learning (LTP/LTD) to sculpt "
        "the network toward the strongest conceptual cluster."
    )

    def __init__(
        self,
        max_iterations: int = 3,
        activation_decay: float = 0.3,
        learning_rate: float = 1.0,
        min_concepts: int = 6,
        max_concepts: int = 12,
    ) -> None:
        self.max_iterations = max(1, max_iterations)
        self.activation_decay = max(0.0, min(1.0, activation_decay))
        self.learning_rate = max(0.1, learning_rate)
        self.min_concepts = max(3, min_concepts)
        self.max_concepts = max(self.min_concepts, max_concepts)

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
        """Execute the Hebbian association reasoning process.

        Args:
            query: The user question or task.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Ignored by this engine.
            max_iterations: Ignored -- ``self.max_iterations`` controls depth.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with concept_extraction,
            activation_spread, reasoning_probe, hebbian_update,
            cluster_identification, and synthesis steps.
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

        # Phase 1: Extract concepts and build initial network
        network_raw = await self._extract_concepts(
            query, llm_provider, trace,
        )
        network = self._parse_network_json(network_raw)

        concept_step = self._make_step(
            step_type="concept_extraction",
            content=network_raw,
            metadata={
                "num_concepts": len(network.get("concepts", [])),
                "num_connections": len(network.get("connections", [])),
            },
            parent_step_id=query_step.step_id,
        )
        trace.add_step(concept_step)

        # Initialise activation levels from relevance scores
        activations: dict[str, float] = {}
        for concept in network.get("concepts", []):
            name = concept.get("name", "")
            if name:
                activations[name] = float(concept.get("relevance", 0.5))

        # Maintain mutable connection weights
        connections: list[dict[str, Any]] = list(network.get("connections", []))

        # Phase 2: Iterative activation-learning cycles
        last_step_id = concept_step.step_id

        for iteration in range(self.max_iterations):
            # 2a: Spreading activation
            activation_raw = await self._spread_activation(
                query=query,
                activations=activations,
                connections=connections,
                iteration=iteration,
                provider=llm_provider,
                trace=trace,
            )
            activation_data = self._parse_activation_json(activation_raw)

            # Update activations
            for act_entry in activation_data.get("activations", []):
                concept_name = act_entry.get("concept", "")
                act_level = float(act_entry.get("activation", 0.5))
                if concept_name:
                    activations[concept_name] = max(
                        0.0, min(1.0, act_level),
                    )

            act_step = self._make_step(
                step_type="activation_spread",
                content=activation_data.get("narrative", activation_raw),
                metadata={
                    "iteration": iteration,
                    "num_active": sum(
                        1 for v in activations.values() if v > 0.3
                    ),
                    "top_concepts": sorted(
                        activations.items(), key=lambda x: x[1], reverse=True,
                    )[:5],
                },
                parent_step_id=last_step_id,
            )
            trace.add_step(act_step)

            # 2b: Probe co-activated concepts for reasoning
            active_concepts = [
                name for name, act in sorted(
                    activations.items(), key=lambda x: x[1], reverse=True,
                )
                if act > 0.3
            ][:8]

            probe_output = await self._reasoning_probe(
                query=query,
                active_concepts=active_concepts,
                activations=activations,
                iteration=iteration,
                provider=llm_provider,
                trace=trace,
            )

            probe_step = self._make_step(
                step_type="reasoning_probe",
                content=probe_output,
                metadata={
                    "iteration": iteration,
                    "active_concepts": active_concepts,
                },
                parent_step_id=act_step.step_id,
            )
            trace.add_step(probe_step)

            # 2c: Hebbian learning -- update connection weights
            hebbian_raw = await self._hebbian_update(
                query=query,
                probe_output=probe_output,
                active_concepts=active_concepts,
                connections=connections,
                iteration=iteration,
                provider=llm_provider,
                trace=trace,
            )
            hebbian_data = self._parse_hebbian_json(hebbian_raw)

            # Apply weight updates
            strengthened_count = 0
            weakened_count = 0
            connections = self._apply_weight_updates(
                connections, hebbian_data,
            )
            strengthened_count = len(hebbian_data.get("strengthen", []))
            weakened_count = len(hebbian_data.get("weaken", []))

            hebb_step = self._make_step(
                step_type="hebbian_update",
                content=hebbian_data.get("narrative", hebbian_raw),
                metadata={
                    "iteration": iteration,
                    "connections_strengthened": strengthened_count,
                    "connections_weakened": weakened_count,
                },
                parent_step_id=probe_step.step_id,
            )
            trace.add_step(hebb_step)
            last_step_id = hebb_step.step_id

        # Phase 3: Identify dominant cluster
        cluster_raw = await self._identify_clusters(
            query=query,
            activations=activations,
            connections=connections,
            provider=llm_provider,
            trace=trace,
        )
        cluster_data = self._parse_cluster_json(cluster_raw)

        cluster_step = self._make_step(
            step_type="cluster_identification",
            content=cluster_data.get("cluster_narrative", cluster_raw),
            score=float(cluster_data.get("cluster_coherence", 0.7)),
            metadata={
                "dominant_cluster": cluster_data.get("dominant_cluster", []),
                "cluster_coherence": cluster_data.get("cluster_coherence", 0.7),
                "secondary_clusters": cluster_data.get("secondary_clusters", []),
            },
            parent_step_id=last_step_id,
        )
        trace.add_step(cluster_step)

        # Phase 4: Final synthesis from dominant cluster
        final_output = await self._synthesize(
            query=query,
            dominant_cluster=cluster_data.get("dominant_cluster", list(activations.keys())),
            cluster_narrative=cluster_data.get("cluster_narrative", ""),
            activations=activations,
            connections=connections,
            provider=llm_provider,
            trace=trace,
        )

        synthesis_step = self._make_step(
            step_type="synthesis",
            content=final_output,
            score=1.0,
            metadata={
                "total_iterations": self.max_iterations,
                "final_network_size": len(activations),
            },
            parent_step_id=cluster_step.step_id,
        )
        trace.add_step(synthesis_step)

        final_step = self._make_step(
            step_type="final_output",
            content=final_output,
            score=1.0,
            metadata={"total_iterations": self.max_iterations},
            parent_step_id=synthesis_step.step_id,
        )
        trace.add_step(final_step)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _extract_concepts(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Extract concepts and build the initial associative network.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with network JSON.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Extract the key concepts from the following query and "
                    f"build an associative network.\n\n"
                    f"Query: {query}\n\n"
                    f"Extract {self.min_concepts}-{self.max_concepts} concepts. "
                    f"For each, provide name, relevance (0.0-1.0), and a brief "
                    f"description.  For each pair of related concepts, provide "
                    f"the connection weight (0.0-1.0) and type (synergistic or "
                    f"antagonistic).\n\n"
                    f"Return the result as a JSON object."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_CONCEPT_EXTRACTION_SYSTEM,
            temperature=0.5,
        )

    async def _spread_activation(
        self,
        query: str,
        activations: dict[str, float],
        connections: list[dict[str, Any]],
        iteration: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Simulate spreading activation through the network.

        Args:
            query: Original query.
            activations: Current activation levels per concept.
            connections: Current connections with weights.
            iteration: Current iteration.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with updated activations.
        """
        # Format current state for the LLM
        state_text = self._format_network_state(activations, connections)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Simulate spreading activation through this semantic "
                    f"network for the query below.  Activation decay per hop: "
                    f"{self.activation_decay}.\n\n"
                    f"Query: {query}\n\n"
                    f"---NETWORK STATE (iteration {iteration})---\n"
                    f"{state_text}\n---END STATE---\n\n"
                    f"Determine new activation levels for each concept.  "
                    f"Activation spreads from highly active concepts along "
                    f"strong connections, decaying by {self.activation_decay} "
                    f"per hop.  Synergistic connections propagate positive "
                    f"activation; antagonistic connections propagate inhibition.\n\n"
                    f"Return JSON with activations (list of concept, activation, "
                    f"activated_by, reasoning) and a narrative summary."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_ACTIVATION_SPREAD_SYSTEM,
            temperature=0.4,
        )

    async def _reasoning_probe(
        self,
        query: str,
        active_concepts: list[str],
        activations: dict[str, float],
        iteration: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Probe the co-activated concept cluster for reasoning.

        Args:
            query: Original query.
            active_concepts: Names of currently active concepts.
            activations: Current activation levels.
            iteration: Current iteration.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Reasoning text from probing the active cluster.
        """
        concept_list = ", ".join(
            f"{c} ({activations.get(c, 0):.2f})" for c in active_concepts
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"The following concepts are co-activated in the semantic "
                    f"network for this query.  Explore what insights emerge "
                    f"from their simultaneous activation.\n\n"
                    f"Query: {query}\n\n"
                    f"Co-activated concepts (with activation levels):\n"
                    f"{concept_list}\n\n"
                    f"This is iteration {iteration}.  What does the "
                    f"co-activation of these concepts reveal?  What "
                    f"connections, implications, and conclusions emerge?"
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_REASONING_PROBE_SYSTEM,
            temperature=0.5,
        )

    async def _hebbian_update(
        self,
        query: str,
        probe_output: str,
        active_concepts: list[str],
        connections: list[dict[str, Any]],
        iteration: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Apply Hebbian learning to update connection weights.

        Args:
            query: Original query.
            probe_output: Reasoning from the current activation probe.
            active_concepts: Currently active concepts.
            connections: Current connections.
            iteration: Current iteration.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with weight update instructions.
        """
        conn_text = "\n".join(
            f"  {c.get('from', '?')} -> {c.get('to', '?')}: "
            f"weight={c.get('weight', 0.5):.2f}, type={c.get('type', '?')}"
            for c in connections[:20]  # cap to avoid overly long prompts
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Apply Hebbian learning based on the reasoning produced "
                    f"by co-activated concepts.\n\n"
                    f"Query: {query}\n\n"
                    f"---REASONING FROM CO-ACTIVATION---\n{probe_output}\n"
                    f"---END REASONING---\n\n"
                    f"Active concepts: {', '.join(active_concepts)}\n\n"
                    f"Current connections:\n{conn_text}\n\n"
                    f"Determine which connections should be STRENGTHENED (LTP) "
                    f"because the co-activation produced useful insight, and "
                    f"which should be WEAKENED (LTD) because the co-activation "
                    f"produced contradictions or noise.\n\n"
                    f"Learning rate: {self.learning_rate}\n\n"
                    f"Return JSON with: strengthen (list), weaken (list), "
                    f"and narrative."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_HEBBIAN_UPDATE_SYSTEM,
            temperature=0.3,
        )

    async def _identify_clusters(
        self,
        query: str,
        activations: dict[str, float],
        connections: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Identify the dominant concept cluster in the final network.

        Args:
            query: Original query.
            activations: Final activation levels.
            connections: Final connections with updated weights.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with cluster identification.
        """
        state_text = self._format_network_state(activations, connections)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Identify the dominant concept cluster in the final "
                    f"associative network after Hebbian learning.\n\n"
                    f"Query: {query}\n\n"
                    f"---FINAL NETWORK STATE---\n{state_text}\n"
                    f"---END STATE---\n\n"
                    f"Which cluster of concepts is most strongly "
                    f"interconnected and most highly activated?  This "
                    f"cluster will form the core of the final answer.\n\n"
                    f"Return JSON with: dominant_cluster (list of names), "
                    f"cluster_coherence (0.0-1.0), cluster_narrative, "
                    f"secondary_clusters."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_CLUSTER_IDENTIFICATION_SYSTEM,
            temperature=0.3,
        )

    async def _synthesize(
        self,
        query: str,
        dominant_cluster: list[str],
        cluster_narrative: str,
        activations: dict[str, float],
        connections: list[dict[str, Any]],
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize the final answer from the dominant cluster.

        Args:
            query: Original query.
            dominant_cluster: Concepts in the dominant cluster.
            cluster_narrative: Narrative about the cluster.
            activations: Final activations.
            connections: Final connections.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Final answer text.
        """
        # Filter connections to those within or touching the dominant cluster
        cluster_set = set(dominant_cluster)
        relevant_connections = [
            c for c in connections
            if c.get("from") in cluster_set or c.get("to") in cluster_set
        ]
        conn_text = "\n".join(
            f"  {c.get('from', '?')} -> {c.get('to', '?')}: "
            f"weight={c.get('weight', 0.5):.2f}"
            for c in relevant_connections[:15]
        )

        cluster_with_activations = ", ".join(
            f"{c} ({activations.get(c, 0):.2f})" for c in dominant_cluster
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Synthesize a comprehensive final answer from the "
                    f"dominant concept cluster.\n\n"
                    f"Query: {query}\n\n"
                    f"Dominant cluster: {cluster_with_activations}\n\n"
                    f"Cluster narrative: {cluster_narrative}\n\n"
                    f"Key connections:\n{conn_text}\n\n"
                    f"Build a thorough, well-structured answer that weaves "
                    f"together the cluster concepts and their relationships. "
                    f"The answer should reflect the associative structure "
                    f"discovered through Hebbian learning."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_network_state(
        activations: dict[str, float],
        connections: list[dict[str, Any]],
    ) -> str:
        """Format the network state as readable text for LLM prompts.

        Args:
            activations: Current activation levels per concept.
            connections: Current connections.

        Returns:
            Human-readable network state string.
        """
        sorted_concepts = sorted(
            activations.items(), key=lambda x: x[1], reverse=True,
        )
        concept_text = "\n".join(
            f"  {name}: activation={act:.3f}" for name, act in sorted_concepts
        )
        conn_text = "\n".join(
            f"  {c.get('from', '?')} -> {c.get('to', '?')}: "
            f"weight={float(c.get('weight', 0.5)):.3f}, "
            f"type={c.get('type', 'synergistic')}"
            for c in connections[:25]
        )
        return f"Concepts:\n{concept_text}\n\nConnections:\n{conn_text}"

    def _apply_weight_updates(
        self,
        connections: list[dict[str, Any]],
        hebbian_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply LTP/LTD weight changes to the connection list.

        Args:
            connections: Current connections.
            hebbian_data: Parsed Hebbian update data.

        Returns:
            Updated connections list.
        """
        # Build a lookup for quick access
        conn_map: dict[tuple[str, str], dict[str, Any]] = {}
        for c in connections:
            key = (c.get("from", ""), c.get("to", ""))
            conn_map[key] = c

        # Apply LTP (strengthening)
        for entry in hebbian_data.get("strengthen", []):
            key = (entry.get("from", ""), entry.get("to", ""))
            delta = abs(float(entry.get("delta", 0.1))) * self.learning_rate
            if key in conn_map:
                old_w = float(conn_map[key].get("weight", 0.5))
                conn_map[key]["weight"] = min(1.0, old_w + delta)
            else:
                # Create new connection
                connections.append({
                    "from": entry.get("from", ""),
                    "to": entry.get("to", ""),
                    "weight": min(1.0, delta),
                    "type": "synergistic",
                    "description": entry.get("reason", "Hebbian LTP"),
                })

        # Apply LTD (weakening)
        for entry in hebbian_data.get("weaken", []):
            key = (entry.get("from", ""), entry.get("to", ""))
            delta = abs(float(entry.get("delta", 0.1))) * self.learning_rate
            if key in conn_map:
                old_w = float(conn_map[key].get("weight", 0.5))
                conn_map[key]["weight"] = max(0.0, old_w - delta)

        return connections

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_network_json(raw: str) -> dict[str, Any]:
        """Parse the concept network JSON.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``concepts`` and ``connections`` lists.
        """
        text = raw.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("concepts", [])
                    parsed.setdefault("connections", [])
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {"concepts": [], "connections": []}

    @staticmethod
    def _parse_activation_json(raw: str) -> dict[str, Any]:
        """Parse activation spread results.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``activations`` list and ``narrative``.
        """
        text = raw.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("activations", [])
                    parsed.setdefault("narrative", raw)
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {"activations": [], "narrative": raw}

    @staticmethod
    def _parse_hebbian_json(raw: str) -> dict[str, Any]:
        """Parse Hebbian update instructions.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``strengthen``, ``weaken``, and ``narrative``.
        """
        text = raw.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("strengthen", [])
                    parsed.setdefault("weaken", [])
                    parsed.setdefault("narrative", raw)
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {"strengthen": [], "weaken": [], "narrative": raw}

    @staticmethod
    def _parse_cluster_json(raw: str) -> dict[str, Any]:
        """Parse cluster identification results.

        Args:
            raw: Raw LLM output.

        Returns:
            Dict with ``dominant_cluster``, ``cluster_coherence``,
            ``cluster_narrative``, ``secondary_clusters``.
        """
        text = raw.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("dominant_cluster", [])
                    parsed.setdefault("cluster_coherence", 0.7)
                    parsed.setdefault("cluster_narrative", raw)
                    parsed.setdefault("secondary_clusters", [])
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {
            "dominant_cluster": [],
            "cluster_coherence": 0.5,
            "cluster_narrative": raw,
            "secondary_clusters": [],
        }


__all__ = ["HebbianAssociation"]
