"""
OpenAgentFlow Reasoning Engines -- Advanced reasoning hierarchies beyond ReAct/CoT/ToT.

This package provides pluggable reasoning engines that implement structured,
multi-step cognitive strategies.  Each engine accepts a query and an LLM
provider, executes a domain-specific reasoning loop, and returns a fully
populated :class:`ReasoningTrace` capturing every intermediate step.

Available engines:
- DialecticalSpiral: Thesis-Antithesis-Synthesis at increasing abstraction levels
- DreamWakeCycle: Divergent ideation + convergent validation oscillation
- MetaCognitiveLoop: Reasoning about reasoning with strategy switching
- AdversarialSelfPlay: Red/Blue/Judge tribunal for robust outputs
- EvolutionaryThought: Darwinian selection on reasoning populations
- FractalRecursion: Self-similar reasoning at macro/meso/micro/nano scales
- ResonanceNetwork: Thoughts amplify/attenuate each other like neural activation
- TemporalRecursion: Pre-mortem reasoning from future-self perspective
- SimulatedAnnealing: High-temperature exploration cooling to convergent solution
- SocraticInterrogation: Progressive questioning to expose and fix assumptions

All engines produce a ReasoningTrace with detailed step-by-step records,
DAG export (``to_dag``), path queries (``get_path``), and step filtering
(``get_steps_by_type``).
"""

from openagentflow.reasoning.adversarial import AdversarialSelfPlay
from openagentflow.reasoning.annealing import SimulatedAnnealing
from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace
from openagentflow.reasoning.dialectical import DialecticalSpiral
from openagentflow.reasoning.dreamwake import DreamWakeCycle
from openagentflow.reasoning.evolutionary import EvolutionaryThought
from openagentflow.reasoning.fractal import FractalRecursion
from openagentflow.reasoning.metacognitive import MetaCognitiveLoop
from openagentflow.reasoning.resonance import ResonanceNetwork
from openagentflow.reasoning.socratic import SocraticInterrogation
from openagentflow.reasoning.temporal import TemporalRecursion

__all__ = [
    # Base classes
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningTrace",
    # Engines (alphabetical)
    "AdversarialSelfPlay",
    "DialecticalSpiral",
    "DreamWakeCycle",
    "EvolutionaryThought",
    "FractalRecursion",
    "MetaCognitiveLoop",
    "ResonanceNetwork",
    "SimulatedAnnealing",
    "SocraticInterrogation",
    "TemporalRecursion",
]
