# OpenAgentFlow Reasoning Engines

Full documentation: [docs/reasoning.md](../../../docs/reasoning.md)

## Quick Reference

### Core Cognitive Engines (10)

| Engine | Pattern | Best For |
|--------|---------|----------|
| `DialecticalSpiral` | Thesis / Antithesis / Synthesis | Deep analysis |
| `DreamWakeCycle` | Divergent / Convergent oscillation | Creative solutions |
| `MetaCognitiveLoop` | Reasoning about reasoning | Complex planning |
| `AdversarialSelfPlay` | Red / Blue / Judge tribunal | Robust validation |
| `EvolutionaryThought` | Darwinian selection on ideas | Optimization |
| `FractalRecursion` | Self-similar at every scale | Hierarchical tasks |
| `ResonanceNetwork` | Thought amplification network | Coherent synthesis |
| `TemporalRecursion` | Future-self pre-mortem | Risk planning |
| `SimulatedAnnealing` | Temperature-based exploration | Escaping local optima |
| `SocraticInterrogation` | Progressive questioning | Critical thinking |

### Neuroscience-Inspired Engines (10)

| Engine | Pattern | Best For |
|--------|---------|----------|
| `PredictiveCoding` | Prediction-error minimisation | Iterative refinement |
| `GlobalWorkspace` | Specialist competition + broadcast | Multi-domain synthesis |
| `HebbianAssociation` | Spreading activation networks | Concept discovery |
| `DefaultModeNetwork` | Focus / mind-wander alternation | Creative breakthroughs |
| `HippocampalReplay` | Forward/backward mental replay | Learning from experience |
| `AttractorNetwork` | Basin-of-attraction convergence | Robust consensus |
| `NeuralOscillation` | Multi-frequency processing | Complex integration |
| `LateralInhibition` | Winner-take-all competition | Decisive selection |
| `BasalGangliaGating` | Go/No-Go/Hyperdirect gating | Action selection |
| `NeuromodulatorySweep` | Multi-neurochemical analysis | Comprehensive coverage |

### Physics-Inspired Engines (10)

| Engine | Pattern | Best For |
|--------|---------|----------|
| `SuperpositionCollapse` | Multi-basis measurement | Framing-robust solutions |
| `WaveInterference` | Constructive/destructive overlay | Argument synthesis |
| `PhaseTransition` | Order from disorder | Breakthrough insights |
| `EntropicFunnel` | Information-theoretic narrowing | Hypothesis elimination |
| `RenormalizationGroup` | Scale-invariant coarse-graining | Essential extraction |
| `GaugeInvariance` | Perspective-invariant truths | Bias elimination |
| `PerturbativeExpansion` | Core + ordered corrections | Systematic improvement |
| `LeastActionPath` | Variational path optimisation | Efficient reasoning |
| `BarrierPenetration` | Tunnelling through barriers | Impossible problems |
| `EntangledThreads` | Coupled co-evolving threads | Interdependent problems |

## Basic Usage

```python
from openagentflow.reasoning import AdversarialSelfPlay

engine = AdversarialSelfPlay(max_rounds=5)
trace = await engine.reason("Design a secure auth system", provider)
print(trace.final_output)
```
