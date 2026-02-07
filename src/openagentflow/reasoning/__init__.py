"""
OpenAgentFlow Reasoning Engines -- Advanced reasoning hierarchies beyond ReAct/CoT/ToT.

This package provides pluggable reasoning engines that implement structured,
multi-step cognitive strategies.  Each engine accepts a query and an LLM
provider, executes a domain-specific reasoning loop, and returns a fully
populated :class:`ReasoningTrace` capturing every intermediate step.

**Original engines (10):**

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

**Neuroscience-inspired engines (10):**

- AttractorNetwork: Hopfield-style attractor dynamics for robust answer convergence
- BasalGangliaGating: Go/No-Go/Hyperdirect pathway decision gating
- DefaultModeNetwork: Focused work alternating with creative mind-wandering
- GlobalWorkspace: Parallel specialist processors competing for conscious broadcast
- HebbianAssociation: Associative concept networks with spreading activation
- HippocampalReplay: Forward/backward replay with counterfactual reasoning
- LateralInhibition: Winner-take-all competitive hypothesis elimination
- NeuralOscillation: Multi-frequency processing with cross-frequency coupling
- NeuromodulatorySweep: Problem analysis under 8 neurochemical regimes
- PredictiveCoding: Prediction-error minimisation via precision-weighted updates

**Physics-inspired engines (10):**

- BarrierPenetration: Tunnelling through reasoning barriers and impossibilities
- EntangledThreads: Co-evolving coupled sub-problems with constraint propagation
- EntropicFunnel: Information-theoretic optimal hypothesis narrowing
- GaugeInvariance: Extracting perspective-invariant truths across framings
- LeastActionPath: Variational optimisation of complete reasoning paths
- PerturbativeExpansion: Core solution plus ordered corrections with convergence monitoring
- PhaseTransition: Disorder-to-order crystallisation under increasing pressure
- RenormalizationGroup: Bottom-up coarse-graining to find what survives abstraction
- SuperpositionCollapse: Multi-basis measurement for framing-robust solutions
- WaveInterference: Constructive/destructive argument interference with boundary filtering

All engines produce a ReasoningTrace with detailed step-by-step records,
DAG export (``to_dag``), path queries (``get_path``), and step filtering
(``get_steps_by_type``).
"""

# Base classes
from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

# Original engines
from openagentflow.reasoning.adversarial import AdversarialSelfPlay
from openagentflow.reasoning.annealing import SimulatedAnnealing
from openagentflow.reasoning.dialectical import DialecticalSpiral
from openagentflow.reasoning.dreamwake import DreamWakeCycle
from openagentflow.reasoning.evolutionary import EvolutionaryThought
from openagentflow.reasoning.fractal import FractalRecursion
from openagentflow.reasoning.metacognitive import MetaCognitiveLoop
from openagentflow.reasoning.resonance import ResonanceNetwork
from openagentflow.reasoning.socratic import SocraticInterrogation
from openagentflow.reasoning.temporal import TemporalRecursion

# Neuroscience-inspired engines
from openagentflow.reasoning.attractor_network import AttractorNetwork
from openagentflow.reasoning.basal_ganglia_gating import BasalGangliaGating
from openagentflow.reasoning.default_mode_network import DefaultModeNetwork
from openagentflow.reasoning.global_workspace import GlobalWorkspace
from openagentflow.reasoning.hebbian_association import HebbianAssociation
from openagentflow.reasoning.hippocampal_replay import HippocampalReplay
from openagentflow.reasoning.lateral_inhibition import LateralInhibition
from openagentflow.reasoning.neural_oscillation import NeuralOscillation
from openagentflow.reasoning.neuromodulatory_sweep import NeuromodulatorySweep
from openagentflow.reasoning.predictive_coding import PredictiveCoding

# Physics-inspired engines
from openagentflow.reasoning.barrier_penetration import BarrierPenetration
from openagentflow.reasoning.entangled_threads import EntangledThreads
from openagentflow.reasoning.entropic_funnel import EntropicFunnel
from openagentflow.reasoning.gauge_invariance import GaugeInvariance
from openagentflow.reasoning.least_action_path import LeastActionPath
from openagentflow.reasoning.perturbative_expansion import PerturbativeExpansion
from openagentflow.reasoning.phase_transition import PhaseTransition
from openagentflow.reasoning.renormalization_group import RenormalizationGroup
from openagentflow.reasoning.superposition_collapse import SuperpositionCollapse
from openagentflow.reasoning.wave_interference import WaveInterference

__all__ = [
    # Base classes
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningTrace",
    # Engines (alphabetical)
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
]
