"""Tests for swarm execution."""

from __future__ import annotations

import pytest

from openagentflow import agent
from openagentflow.core.swarm import swarm, SwarmResult, get_swarm
from openagentflow.core.types import AgentStatus


class TestSwarmDecorator:
    """Tests for the @swarm decorator."""

    def test_swarm_registration(self):
        """Test that swarms are registered."""
        @agent(model="mock")
        async def swarm_worker1(x: str) -> str:
            pass

        @agent(model="mock")
        async def swarm_worker2(x: str) -> str:
            pass

        @swarm(agents=["swarm_worker1", "swarm_worker2"], strategy="voting")
        async def my_swarm(x: str) -> str:
            pass

        spec = get_swarm("my_swarm")
        assert spec is not None
        assert spec.name == "my_swarm"
        assert spec.agents == ["swarm_worker1", "swarm_worker2"]
        assert spec.consensus_strategy == "voting"

    def test_swarm_custom_name(self):
        """Test swarm with custom name."""
        @swarm(agents=[], name="custom_swarm")
        async def internal_swarm(x: str) -> str:
            pass

        spec = get_swarm("custom_swarm")
        assert spec is not None

    @pytest.mark.asyncio
    async def test_swarm_parallel_execution(self):
        """Test swarm executes agents in parallel."""
        @agent(model="mock")
        async def parallel_agent1(x: str) -> str:
            pass

        @agent(model="mock")
        async def parallel_agent2(x: str) -> str:
            pass

        @swarm(agents=["parallel_agent1", "parallel_agent2"])
        async def parallel_swarm(x: str) -> str:
            pass

        result = await parallel_swarm("input")

        assert isinstance(result, SwarmResult)
        assert result.status == AgentStatus.SUCCEEDED
        assert len(result.agent_results) == 2

    @pytest.mark.asyncio
    async def test_swarm_voting_consensus(self):
        """Test voting consensus strategy."""
        @agent(model="mock")
        async def voter1(x: str) -> str:
            pass

        @agent(model="mock")
        async def voter2(x: str) -> str:
            pass

        @agent(model="mock")
        async def voter3(x: str) -> str:
            pass

        @swarm(agents=["voter1", "voter2", "voter3"], strategy="voting")
        async def voting_swarm(x: str) -> str:
            pass

        result = await voting_swarm("vote on this")
        assert result.status == AgentStatus.SUCCEEDED
        assert result.consensus_method == "voting"
        assert result.agreement_score > 0

    @pytest.mark.asyncio
    async def test_swarm_synthesis_strategy(self):
        """Test synthesis strategy returns all outputs."""
        @agent(model="mock")
        async def synth_agent1(x: str) -> str:
            pass

        @agent(model="mock")
        async def synth_agent2(x: str) -> str:
            pass

        @swarm(agents=["synth_agent1", "synth_agent2"], strategy="synthesis")
        async def synthesis_swarm(x: str) -> str:
            pass

        result = await synthesis_swarm("synthesize")
        assert result.status == AgentStatus.SUCCEEDED
        assert result.consensus_method == "synthesis"
        assert isinstance(result.output, list)
        assert len(result.output) == 2

    @pytest.mark.asyncio
    async def test_swarm_first_strategy(self):
        """Test first strategy returns first result."""
        @agent(model="mock")
        async def first_agent1(x: str) -> str:
            pass

        @agent(model="mock")
        async def first_agent2(x: str) -> str:
            pass

        @swarm(agents=["first_agent1", "first_agent2"], strategy="first")
        async def first_swarm(x: str) -> str:
            pass

        result = await first_swarm("first wins")
        assert result.status == AgentStatus.SUCCEEDED
        assert result.consensus_method == "first"

    @pytest.mark.asyncio
    async def test_swarm_missing_agents(self):
        """Test swarm fails when no agents exist."""
        @swarm(agents=["nonexistent1", "nonexistent2"])
        async def empty_swarm(x: str) -> str:
            pass

        result = await empty_swarm("input")
        assert result.status == AgentStatus.FAILED
        assert "No valid agents" in result.error

    @pytest.mark.asyncio
    async def test_swarm_lineage_tracking(self):
        """Test that execution hash lineage is tracked."""
        @agent(model="mock")
        async def lineage_swarm_agent1(x: str) -> str:
            pass

        @agent(model="mock")
        async def lineage_swarm_agent2(x: str) -> str:
            pass

        @swarm(agents=["lineage_swarm_agent1", "lineage_swarm_agent2"])
        async def lineage_swarm(x: str) -> str:
            pass

        result = await lineage_swarm("test", parent_hash="swarm_parent")
        assert result.execution_hash is not None
        assert "swarm_parent" in result.execution_hash.parents
