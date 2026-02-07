"""
Open Agent Flow - A distributed agentic AI workflow framework.

Graph-native reasoning traces, multi-LLM support, and production-ready infrastructure.

Quick Start:
    import openagentflow
    from openagentflow import agent, tool

    # Option 1: Configure with API key directly
    openagentflow.configure(anthropic_api_key="sk-ant-...")

    # Option 2: Use environment variable (ANTHROPIC_API_KEY)
    # Option 3: Use .env file
    openagentflow.load_dotenv()

    @tool
    def search(query: str) -> list[dict]:
        '''Search the web.'''
        return [{"title": "Result", "url": "https://..."}]

    @agent(model="claude-sonnet-4-20250514", tools=[search])
    async def researcher(query: str) -> str:
        '''Research agent.'''
        pass

    # Run the agent
    result = await researcher("AI trends 2024")
    print(result.output)  # Agent's response
    print(result.status)  # AgentStatus.SUCCEEDED
    print(result.total_tokens)  # Token usage

Installation:
    pip install openagentflow
    pip install openagentflow[anthropic]  # For Claude
    pip install openagentflow[openai]     # For GPT
    pip install openagentflow[all]        # All features
"""

from openagentflow.config.settings import (
    configure,
    get_api_key,
    get_settings,
    is_configured,
    load_dotenv,
)
from openagentflow.core.agent import agent, get_agent, get_all_agents, run_agent
from openagentflow.core.chain import chain, get_chain
from openagentflow.core.swarm import get_swarm, swarm
from openagentflow.core.tool import (
    execute_tool,
    get_all_tools,
    get_tool,
    tool,
    tools_to_anthropic_format,
    tools_to_openai_format,
)
from openagentflow.core.types import (
    AgentResult,
    AgentSpec,
    AgentState,
    AgentStatus,
    ChainSpec,
    ExecutionHash,
    ExecutorType,
    LLMProvider,
    MemoryConfig,
    MemoryType,
    Message,
    ModelConfig,
    ReasoningStrategy,
    SwarmSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from openagentflow.graph import (
    GraphBackend,
    SQLiteGraphBackend,
    get_default_backend,
)
from openagentflow.memory import (
    FleetingMemory,
    LongTermMemory,
    MemoryEntry,
    MemoryGarbageCollector,
    MemoryManager,
    MemoryTier,
    ShortTermMemory,
)
from openagentflow.meta import (
    Sandbox,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
    ToolFactory,
    ToolFactoryError,
)
from openagentflow.reasoning import (
    AdversarialSelfPlay,
    AttractorNetwork,
    BarrierPenetration,
    BasalGangliaGating,
    DefaultModeNetwork,
    DialecticalSpiral,
    DreamWakeCycle,
    EntangledThreads,
    EntropicFunnel,
    EvolutionaryThought,
    FractalRecursion,
    GaugeInvariance,
    GlobalWorkspace,
    HebbianAssociation,
    HippocampalReplay,
    LateralInhibition,
    LeastActionPath,
    MetaCognitiveLoop,
    NeuralOscillation,
    NeuromodulatorySweep,
    PerturbativeExpansion,
    PhaseTransition,
    PredictiveCoding,
    ReasoningEngine,
    ReasoningStep,
    ReasoningTrace,
    RenormalizationGroup,
    ResonanceNetwork,
    SimulatedAnnealing,
    SocraticInterrogation,
    SuperpositionCollapse,
    TemporalRecursion,
    WaveInterference,
)
from openagentflow.distributed import (
    ComputeBackend,
    ComputeCluster,
    ComputeNode,
    DistributedOllamaProvider,
    DockerBackend,
    KubernetesBackend,
    LeastLoadBalancer,
    LoadBalancer,
    ModelAffinityBalancer,
    RoundRobinBalancer,
    SSHBackend,
)

__version__ = "0.2.0"
__all__ = [
    # Configuration
    "configure",
    "load_dotenv",
    "get_settings",
    "get_api_key",
    "is_configured",
    # Decorators
    "agent",
    "tool",
    "chain",
    "swarm",
    # Registry functions
    "get_agent",
    "get_all_agents",
    "run_agent",
    "get_tool",
    "get_all_tools",
    "execute_tool",
    "get_chain",
    "get_swarm",
    # Tool format converters
    "tools_to_anthropic_format",
    "tools_to_openai_format",
    # Types
    "AgentSpec",
    "AgentState",
    "AgentStatus",
    "AgentResult",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "Message",
    "ModelConfig",
    "MemoryConfig",
    "MemoryType",
    "LLMProvider",
    "ReasoningStrategy",
    "ChainSpec",
    "SwarmSpec",
    "ExecutionHash",
    "ExecutorType",
    # Graph tracing
    "GraphBackend",
    "SQLiteGraphBackend",
    "get_default_backend",
    # Memory hierarchy
    "MemoryManager",
    "MemoryTier",
    "MemoryEntry",
    "FleetingMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryGarbageCollector",
    # Meta / JIT tool creation
    "Sandbox",
    "SandboxValidationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "ToolFactory",
    "ToolFactoryError",
    # Reasoning engines
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningTrace",
    "AdversarialSelfPlay",
    "AttractorNetwork",
    "BarrierPenetration",
    "BasalGangliaGating",
    "DefaultModeNetwork",
    "DialecticalSpiral",
    "DreamWakeCycle",
    "EntangledThreads",
    "EntropicFunnel",
    "EvolutionaryThought",
    "FractalRecursion",
    "GaugeInvariance",
    "GlobalWorkspace",
    "HebbianAssociation",
    "HippocampalReplay",
    "LateralInhibition",
    "LeastActionPath",
    "MetaCognitiveLoop",
    "NeuralOscillation",
    "NeuromodulatorySweep",
    "PerturbativeExpansion",
    "PhaseTransition",
    "PredictiveCoding",
    "RenormalizationGroup",
    "ResonanceNetwork",
    "SimulatedAnnealing",
    "SocraticInterrogation",
    "SuperpositionCollapse",
    "TemporalRecursion",
    "WaveInterference",
    # Distributed compute
    "ComputeBackend",
    "ComputeCluster",
    "ComputeNode",
    "LoadBalancer",
    "RoundRobinBalancer",
    "LeastLoadBalancer",
    "ModelAffinityBalancer",
    "KubernetesBackend",
    "DockerBackend",
    "SSHBackend",
    "DistributedOllamaProvider",
]
