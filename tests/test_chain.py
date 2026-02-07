"""Tests for chain execution."""

from __future__ import annotations

import pytest

from openagentflow import agent, tool
from openagentflow.core.chain import chain, ChainResult, get_chain
from openagentflow.core.types import AgentStatus


class TestChainDecorator:
    """Tests for the @chain decorator."""

    def test_chain_registration(self):
        """Test that chains are registered."""
        @agent(model="mock")
        async def step1(x: str) -> str:
            pass

        @agent(model="mock")
        async def step2(x: str) -> str:
            pass

        @chain(agents=["step1", "step2"])
        async def my_chain(x: str) -> str:
            pass

        spec = get_chain("my_chain")
        assert spec is not None
        assert spec.name == "my_chain"
        assert spec.agents == ["step1", "step2"]

    def test_chain_custom_name(self):
        """Test chain with custom name."""
        @chain(agents=[], name="custom_pipeline")
        async def internal_name(x: str) -> str:
            pass

        spec = get_chain("custom_pipeline")
        assert spec is not None

    @pytest.mark.asyncio
    async def test_chain_execution(self):
        """Test chain executes agents in sequence."""
        calls = []

        @agent(model="mock")
        async def chain_agent1(x: str) -> str:
            calls.append("agent1")
            return f"processed1:{x}"

        @agent(model="mock")
        async def chain_agent2(x: str) -> str:
            calls.append("agent2")
            return f"processed2:{x}"

        @chain(agents=["chain_agent1", "chain_agent2"])
        async def test_chain(x: str) -> str:
            pass

        result = await test_chain("input")

        assert isinstance(result, ChainResult)
        assert result.status == AgentStatus.SUCCEEDED
        assert len(result.agent_results) == 2

    @pytest.mark.asyncio
    async def test_chain_passes_output(self):
        """Test that output is passed between agents."""
        @agent(model="mock")
        async def transform1(x: str) -> str:
            pass

        @agent(model="mock")
        async def transform2(x: str) -> str:
            pass

        @chain(agents=["transform1", "transform2"], pass_output=True)
        async def transform_chain(x: str) -> str:
            pass

        result = await transform_chain("start")
        assert result.status == AgentStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_chain_missing_agent(self):
        """Test chain fails when agent is missing."""
        @chain(agents=["nonexistent_agent"], stop_on_failure=True)
        async def broken_chain(x: str) -> str:
            pass

        result = await broken_chain("input")
        assert result.status == AgentStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_chain_lineage_tracking(self):
        """Test that execution hash lineage is tracked."""
        @agent(model="mock")
        async def lineage_agent1(x: str) -> str:
            pass

        @agent(model="mock")
        async def lineage_agent2(x: str) -> str:
            pass

        @chain(agents=["lineage_agent1", "lineage_agent2"])
        async def lineage_chain(x: str) -> str:
            pass

        result = await lineage_chain("test", parent_hash="parent123")
        assert result.execution_hash is not None
        assert "parent123" in result.execution_hash.parents
