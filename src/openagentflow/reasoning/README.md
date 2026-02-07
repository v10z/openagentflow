# OpenAgentFlow Reasoning Engines

Full documentation: [docs/reasoning.md](../../../docs/reasoning.md)

## Quick Reference

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

## Basic Usage

```python
from openagentflow.reasoning import AdversarialSelfPlay

engine = AdversarialSelfPlay(max_rounds=5)
trace = await engine.reason("Design a secure auth system", provider)
print(trace.final_output)
```
