# Physics-Inspired Reasoning Engines: Design Document

## Overview

This document specifies 10 new reasoning engines for OpenAgentFlow, each grounded in
a real physics phenomenon. Each engine subclasses `ReasoningEngine` from
`openagentflow.reasoning.base`, implements `async def reason(query, llm_provider,
...) -> ReasoningTrace`, and uses `_call_llm()` as the sole path for LLM
invocations.

These engines are distinct from the existing 10 engines (DialecticalSpiral,
DreamWakeCycle, MetaCognitiveLoop, AdversarialSelfPlay, EvolutionaryThought,
FractalRecursion, ResonanceNetwork, TemporalRecursion, SimulatedAnnealing,
SocraticInterrogation). While SimulatedAnnealing already draws from statistical
mechanics, the 10 engines described here draw from deeper physics -- quantum
mechanics, field theory, general relativity, information theory, and classical
mechanics -- and each maps a specific physical mechanism onto a novel LLM
orchestration pattern.

### Conventions Followed

- Each engine is a single Python file in `openagentflow/reasoning/`.
- Engine names are `CamelCase` class names; file names are `snake_case.py`.
- Each engine sets `name` (a `snake_case` string) and `description` class attributes.
- All LLM calls go through `self._call_llm(provider, messages, trace, system=...,
  temperature=...)`.
- Steps are recorded via `trace.add_step(ReasoningStep(...))` with typed step names,
  parent linkage, metadata, and scores.
- Convergence/termination uses pure-stdlib heuristics (no external dependencies).
- Engines are stateless across calls; all state lives in the `ReasoningTrace`.

### Physics Concepts Mapped

| Physics Concept | Engine | Core Mapping |
|----------------|--------|-------------|
| Quantum superposition / measurement / collapse | QuantumSuperposition | Parallel states + basis-dependent collapse |
| Wave interference (constructive/destructive) | WaveInterference | Idea amplification and cancellation |
| Phase transitions / critical phenomena | PhaseTransition | Order parameter emergence at criticality |
| Entropy and information theory | EntropicFunnel | Shannon entropy reduction through refinement |
| Renormalization group (scale invariance) | RenormalizationGroup | Coarse-graining at multiple scales |
| Gauge symmetry / invariance | GaugeInvariance | Perspective-independent invariant extraction |
| Perturbation theory (incremental corrections) | PerturbativeExpansion | Successive order corrections to a base solution |
| Lagrangian mechanics (principle of least action) | LeastActionPath | Variational optimization of reasoning paths |
| Quantum tunneling (barrier penetration) | QuantumTunneling | Escaping local traps via barrier penetration |
| Entanglement (correlated threads) | EntangledThreads | Non-local correlations between reasoning branches |

---

## Engine 1: QuantumSuperposition

### File: `quantum_superposition.py`

### Physics Basis

**Quantum Superposition, Measurement, and Wavefunction Collapse**

In quantum mechanics, a system exists in a superposition of all possible
eigenstates until a measurement is performed. The state vector is:

```
|psi> = sum_i c_i |phi_i>
```

where `c_i` are complex probability amplitudes satisfying `sum |c_i|^2 = 1`. Upon
measurement (the Born rule), the wavefunction collapses to a single eigenstate
`|phi_k>` with probability `|c_k|^2`. Crucially, before measurement the system
genuinely does not have a definite value -- all possibilities coexist and
**interfere** with each other.

The key insight that distinguishes this from "generate many options and pick one":

1. **Interference before collapse**: States modify each other's amplitudes via
   constructive/destructive interference before any selection occurs. This is not
   voting or scoring -- it is mutual modification.
2. **Measurement basis matters**: The same superposition collapses to different
   outcomes depending on the measurement basis (the observable being measured).
   Measuring position versus momentum on the same quantum state yields
   fundamentally different results. This maps to: the **framing** of the
   evaluation criterion determines which solution emerges.
3. **No hidden variables (Copenhagen)**: The outcome is genuinely indeterminate
   until measurement. Premature commitment to a solution before interference has
   completed is like measuring a system before it has evolved -- you get a less
   interesting answer.

This engine is fundamentally different from EvolutionaryThought (fitness selection
over generations) and ResonanceNetwork (pairwise reinforcement to equilibrium).
Here the novelty is that the measurement basis itself is a variable -- the engine
explores *what question to ask* to collapse the superposition.

### Algorithm

**Phase 1: STATE PREPARATION** (1 LLM call)
1. LLM call: "Generate N fundamentally different approaches to this problem. Do
   NOT evaluate them -- simply enumerate all plausible states that could be the
   answer. Include at least one unconventional approach."
   - System prompt: "You are generating a quantum superposition of solution states.
     Hold all possibilities simultaneously without premature judgment."
   - Temperature: 0.8
   - Output: JSON array of N state strings
   - Each state starts with amplitude `1/sqrt(N)`
   - Records: `state_preparation` step, plus one `state` child step per state

**Phase 2: INTERFERENCE** (1-2 LLM calls, batched)
2. For all unique pairs `(i, j)`, evaluate interference:
   - LLM call: "For each pair of solution states, determine whether they are
     constructively interfering (considering both together makes each stronger) or
     destructively interfering (they undermine each other). Also note any
     'interference pattern' -- a new insight that neither state contains alone."
   - Output: JSON array of `{pair, interference: float in [-1,1], pattern: str|null}`
   - Amplitude update rule (inspired by the path integral formulation):
     ```
     c_i' = c_i + sum_j interference(i,j) * c_j * alpha
     ```
     where `alpha` is `interference_strength`. Then re-normalize so `sum |c_k'|^2 = 1`.
   - New interference patterns (emergent insights from considering pairs together)
     are added as new states with initial amplitude `beta`.
   - Records: `interference` step with pair scores; `interference_pattern` steps for
     new emergent insights; `amplitude_update` step with updated amplitudes

**Phase 3: MEASUREMENT BASIS SELECTION** (1 LLM call)
3. LLM call: "Propose K different measurement bases -- concrete evaluation
   criteria or framings -- that could collapse this superposition. Each basis
   should represent a genuinely different way of deciding which state is best.
   Different bases may yield different winners."
   - System prompt: "You are selecting measurement bases. A measurement basis is
     not a preference -- it is a frame of observation that determines what you see."
   - Temperature: 0.5
   - Output: JSON array of K basis strings
   - Records: `basis_selection` step

**Phase 4: MEASUREMENT / COLLAPSE** (K LLM calls, one per basis)
4. For each measurement basis:
   - LLM call: "Given this evaluation criterion, collapse the superposition.
     Which state does this measurement select? Assign final probabilities to all
     states under this basis."
   - Output: JSON with `selected_state_index`, `probability`, `reasoning`,
     `collapsed_answer`
   - Accumulate: the state with the highest average selection probability across
     ALL bases is the "decoherence-robust" collapse (survives in the most bases).
   - Records: `measurement` step per basis

**Phase 5: DECOHERENCE SYNTHESIS** (1 LLM call)
5. LLM call: "Multiple measurements have been performed on the superposition from
   different bases. The robustly selected state is the one that survives the most
   measurement bases. Synthesize the final answer, incorporating any interference
   patterns that emerged."
   - Records: `decoherence` step, `final_output` step

### LLM Call Pattern

```
STATE_PREP(1) --> INTERFERENCE(1-2) --> BASIS_SELECTION(1) -->
MEASUREMENT(K) --> SYNTHESIS(1)

Total: 4 + K calls (default K=3: 7 calls)
Dependencies: strictly sequential phases
Parallelism: Phase 4 measurements are independent and parallelizable
```

### When to Use

- Problems where the **framing** of the question determines the answer (policy
  decisions, design trade-offs, philosophical questions)
- Problems where you suspect premature commitment to an approach
- Multi-stakeholder problems where different evaluation criteria lead to different
  optimal solutions
- Situations where the "right answer" depends on unstated assumptions
- NOT for: well-defined factual questions with a single unambiguous correct answer

### Step Types

| Step Type | Description |
|-----------|-------------|
| `state_preparation` | Initial superposition generation prompt |
| `state` | Individual state in the superposition |
| `interference` | Pairwise interference evaluation with scores |
| `interference_pattern` | New insight emerging from state interaction |
| `amplitude_update` | Updated amplitudes after interference |
| `basis_selection` | Proposed measurement bases (evaluation criteria) |
| `measurement` | Collapse result under a specific basis |
| `decoherence` | Cross-basis robust selection |
| `final_output` | Polished final answer |

### Parameters

```python
class QuantumSuperposition(ReasoningEngine):
    name = "quantum_superposition"
    description = (
        "Parallel solution states interfere and collapse under "
        "basis-dependent measurement. The evaluation framing "
        "determines the outcome."
    )

    def __init__(
        self,
        num_states: int = 5,              # Superposition size
        num_bases: int = 3,               # Measurement bases to try
        interference_strength: float = 0.15,  # Cross-state interference coupling
        new_state_threshold: float = 0.3, # Min amplitude for emergent pattern states
        state_temperature: float = 0.8,   # Temperature for state generation
        measurement_temperature: float = 0.4,  # Temperature for collapse
    ) -> None: ...
```

### DAG Structure

```
query
  +-- state_preparation
  |     +-- state_0 (c_0 = 0.447)
  |     +-- state_1 (c_1 = 0.447)
  |     +-- state_2 (c_2 = 0.447)
  |     +-- state_3 (c_3 = 0.447)
  |     +-- state_4 (c_4 = 0.447)
  +-- interference
  |     +-- interference_pattern_0 (emergent insight)
  |     +-- amplitude_update (new amplitudes)
  +-- basis_selection
  |     +-- measurement_basis_0
  |     +-- measurement_basis_1
  |     +-- measurement_basis_2
  +-- decoherence (robust selection)
  +-- final_output
```

---

## Engine 2: WaveInterference

### File: `wave_interference.py`

### Physics Basis

**Wave Superposition, Interference, and Diffraction**

When two or more waves overlap in space, the resulting amplitude at any point is
the algebraic sum of the individual amplitudes (the principle of superposition for
linear waves):

```
A_total(x, t) = sum_i A_i * cos(k_i * x - omega_i * t + phi_i)
```

This produces **interference patterns**:

- **Constructive interference**: When waves arrive in phase (`phi_i - phi_j ~= 0`),
  amplitudes add, producing a peak. The signal is amplified.
- **Destructive interference**: When waves arrive out of phase
  (`phi_i - phi_j ~= pi`), amplitudes cancel, producing a node. The signal is
  suppressed.
- **Beats**: When waves of slightly different frequencies superpose, they produce
  a slowly modulating envelope -- periodically constructive and destructive. This
  is the temporal analog of spatial interference fringes.
- **Diffraction**: Waves bend around obstacles and spread through apertures,
  producing intensity patterns that depend on the geometry. Huygen's principle: every
  point on a wavefront acts as a source of secondary wavelets.

Young's double-slit experiment (1801) demonstrated that light exhibits interference.
The intensity pattern on the screen:

```
I(theta) = I_0 * cos^2(pi * d * sin(theta) / lambda)
```

The key insight for reasoning: ideas generated from **different perspectives** (the
"slits") produce an interference pattern when brought together. Points of
constructive interference are where multiple perspectives agree and reinforce --
these are the strongest conclusions. Points of destructive interference reveal
genuine contradictions that must be resolved.

This differs from QuantumSuperposition (which focuses on measurement basis
selection) and from DialecticalSpiral (which is limited to thesis/antithesis pairs).
WaveInterference handles N-way superposition with continuous phase relationships
and explicitly computes the full interference pattern, not just pairwise conflicts.

### Algorithm

**Phase 1: WAVE SOURCE GENERATION** (1 LLM call)
1. LLM call: "Analyze this problem from N distinct perspectives/angles. Each
   perspective is a coherent 'wave' of reasoning. Perspectives should span
   different domains, assumptions, value systems, or methodologies."
   - System prompt: "You are generating coherent wave sources. Each perspective
     must be internally consistent (coherent) but may differ from others."
   - Temperature: 0.7
   - Output: JSON array of `{perspective, analysis, key_claims: [str]}`
   - Records: `wave_source` step per perspective

**Phase 2: CLAIM EXTRACTION AND ALIGNMENT** (1 LLM call)
2. LLM call: "For each claim made by each perspective, determine its 'phase' --
   how it relates to claims from other perspectives. Are they in phase (agreeing),
   out of phase (contradicting), or at some intermediate angle?"
   - Output: Matrix of claim-pair phase relationships: `{claim_i, claim_j,
     phase_angle: float in [0, pi], coherence: float in [0, 1]}`
   - Phase 0 = perfectly in phase (constructive), phase pi = perfectly out of
     phase (destructive)
   - Records: `phase_alignment` step with full phase matrix

**Phase 3: INTERFERENCE PATTERN COMPUTATION** (programmatic + 1 LLM call)
3. For each claim, compute the resultant amplitude by summing contributions from
   all perspectives, weighted by coherence:
   ```
   A_claim = sum_perspectives cos(phase_angle_p) * coherence_p
   ```
   - Claims with high positive resultant amplitude: constructive interference
     (strong consensus points)
   - Claims with near-zero amplitude: destructive interference (contradictions
     that cancel)
   - Claims with moderate amplitude: partial interference (nuanced areas)
4. LLM call: "The interference pattern reveals these constructive peaks (strong
   consensus claims) and these destructive nodes (contradictions). For each
   destructive node, explain why the perspectives contradict and whether the
   contradiction reveals a genuine ambiguity or a resolvable error."
   - Records: `interference_pattern` step with peaks, nodes, and explanations

**Phase 4: DIFFRACTION ANALYSIS** (1 LLM call)
5. LLM call: "The interference pattern has gaps -- areas where no perspective's
   wave reached. Like diffraction through an aperture, the existing wave sources
   can illuminate adjacent regions. What secondary conclusions can be drawn by
   extending the strong interference peaks into these gaps?"
   - System prompt: "You are computing diffracted reasoning -- extending
     established conclusions into adjacent territory."
   - Temperature: 0.5
   - Records: `diffraction` step

**Phase 5: PATTERN SYNTHESIS** (1 LLM call)
6. LLM call: "Synthesize the final answer from the full interference and
   diffraction pattern. Build the answer on the constructive peaks (strong
   consensus). Address the destructive nodes (contradictions) explicitly. Include
   diffracted insights (extended conclusions)."
   - Records: `pattern_synthesis` step, `final_output` step

### LLM Call Pattern

```
WAVE_SOURCES(1) --> PHASE_ALIGNMENT(1) --> INTERFERENCE_PATTERN(1) -->
DIFFRACTION(1) --> SYNTHESIS(1)

Total: 5 calls
Dependencies: strictly sequential
```

### When to Use

- Problems where multiple valid perspectives exist and you need to find consensus
- Interdisciplinary analysis where insights from different fields must be combined
- Situations where contradictions between viewpoints are as informative as agreements
- Complex policy or design questions with multiple stakeholders
- Any problem where you want to explicitly identify where perspectives agree (peaks)
  and disagree (nodes)

### Step Types

| Step Type | Description |
|-----------|-------------|
| `wave_source` | A coherent perspective/analysis with key claims |
| `phase_alignment` | Phase relationships between claims across perspectives |
| `interference_pattern` | Computed constructive peaks and destructive nodes |
| `diffraction` | Extended conclusions in unexplored regions |
| `pattern_synthesis` | Final answer from the interference pattern |
| `final_output` | Polished output |

### Parameters

```python
class WaveInterference(ReasoningEngine):
    name = "wave_interference"
    description = (
        "Multiple reasoning perspectives interfere like waves. "
        "Constructive peaks reveal consensus; destructive nodes "
        "expose contradictions."
    )

    def __init__(
        self,
        num_perspectives: int = 4,          # Number of wave sources
        coherence_threshold: float = 0.3,   # Min coherence for a claim to count
        constructive_threshold: float = 0.6, # Amplitude above which = constructive peak
        destructive_threshold: float = 0.2,  # Amplitude below which = destructive node
        enable_diffraction: bool = True,     # Whether to extend into gaps
        source_temperature: float = 0.7,
        synthesis_temperature: float = 0.4,
    ) -> None: ...
```

### DAG Structure

```
query
  +-- wave_source_0 (perspective A)
  +-- wave_source_1 (perspective B)
  +-- wave_source_2 (perspective C)
  +-- wave_source_3 (perspective D)
  +-- phase_alignment (claim-pair phase matrix)
  |     +-- interference_pattern
  |           +-- constructive_peak_0 ... N
  |           +-- destructive_node_0 ... M
  +-- diffraction (extended conclusions)
  +-- pattern_synthesis
  +-- final_output
```

---

## Engine 3: PhaseTransition

### File: `phase_transition.py`

### Physics Basis

**Phase Transitions and Critical Phenomena**

A phase transition occurs when a system undergoes a qualitative change in its
macroscopic behavior as an external parameter (temperature, pressure, magnetic
field) crosses a critical value. Classic examples:

- Water: liquid to solid at 0C, liquid to gas at 100C
- Ferromagnetism: disordered spins spontaneously align below the Curie temperature
- Superconductivity: resistance drops to zero below T_c

The Landau theory of phase transitions introduces an **order parameter** --
a macroscopic quantity that is zero in the disordered phase and nonzero in the
ordered phase. For a ferromagnet, this is the magnetization M. The free energy
near the critical point:

```
F(M) = a(T - T_c) * M^2 + b * M^4 + ...
```

At the critical point `T = T_c`, the system exhibits:

- **Universality**: Microscopically different systems show identical critical
  behavior (same critical exponents)
- **Divergent correlation length**: Fluctuations span all length scales
- **Critical slowing down**: The system takes infinitely long to reach equilibrium
- **Spontaneous symmetry breaking**: Below T_c, the system must "choose" a
  direction even though the underlying physics is symmetric
- **Emergent order**: Macroscopic organization arises from microscopic disorder
  without any external organizer

The key insight for reasoning: complex problems have a "phase transition" where
disorganized exploration of ideas suddenly crystallizes into a coherent solution.
The engine deliberately operates near this critical point -- maintaining maximum
diversity of ideas (disordered phase) while gradually lowering the "temperature"
(tightening constraints) until spontaneous order emerges. The order parameter
measures how coherent the current thinking is.

### Algorithm

**Phase 1: DISORDERED EXPLORATION** (N LLM calls, parallelizable)
1. Generate N maximally diverse, loosely constrained responses to the problem.
   Each uses a different system prompt personality and high temperature.
   - System prompts vary: "You are a skeptic," "You are an optimist," "You are
     a pragmatist," "You are a theorist," "You are a contrarian"
   - Temperature: 1.0 (maximum entropy / disordered phase)
   - Records: `disordered_state` step per response

**Phase 2: ORDER PARAMETER MEASUREMENT** (1 LLM call)
2. LLM call: "Examine these N responses. Measure the 'order parameter' -- how
   much coherence/consensus exists among them? Specifically: (a) What themes
   appear in a majority of responses? (b) What claims are contradicted by a
   majority? (c) On a scale of 0.0 (total chaos) to 1.0 (perfect order), rate
   the current degree of coherence."
   - Output: JSON with `order_parameter: float`, `majority_themes: [str]`,
     `contradicted_claims: [str]`
   - Records: `order_measurement` step

**Phase 3: CONSTRAINED RE-GENERATION (cooling toward T_c)** (N LLM calls)
3. Re-generate all N responses, but now constrain them with the majority themes
   as fixed points. Temperature is lowered to `T = T_initial * cooling_factor`.
   - "You must incorporate these established themes: [majority_themes]. You are
     free to disagree on everything else."
   - Records: `cooled_state` step per response

**Phase 4: CRITICAL POINT DETECTION** (1 LLM call)
4. Re-measure the order parameter. If it has crossed the critical threshold
   (`order_parameter >= critical_threshold`), the phase transition has occurred --
   proceed to Phase 5. If not, repeat Phase 3 with further cooling.
   - Records: `critical_check` step
   - If `order_parameter < critical_threshold` and cooling cycles remain: loop
     to Phase 3 with tighter constraints

**Phase 5: SYMMETRY BREAKING** (1 LLM call)
5. LLM call: "The system has reached the critical point. Multiple equally valid
   orderings are possible (like a ferromagnet choosing 'up' or 'down'). Choose
   the specific ordering that best fits the problem. This is spontaneous symmetry
   breaking -- you must commit to a direction even though the physics does not
   prefer one."
   - System prompt: "You are performing spontaneous symmetry breaking. There is
     no unique correct direction -- but you must choose one and commit fully."
   - Temperature: 0.3 (low T, ordered phase)
   - Records: `symmetry_breaking` step

**Phase 6: ORDERED STATE REFINEMENT** (1 LLM call)
6. LLM call: "The phase transition is complete. Refine the ordered state into a
   polished answer. The spontaneous order that emerged represents the natural
   structure of the problem -- do not fight it."
   - Records: `ordered_state` step, `final_output` step

### LLM Call Pattern

```
DISORDER(N) --> ORDER_MEASURE(1) --> [COOLING(N) --> CRITICAL_CHECK(1)] x R -->
SYMMETRY_BREAK(1) --> REFINEMENT(1)

Total: N + 1 + R * (N + 1) + 2, where R = cooling rounds until phase transition
With defaults (N=5, R=2): 5 + 1 + 2*(5+1) + 2 = 20 calls
Parallelism: All N calls within each phase are independent
```

### When to Use

- Problems where you need emergent structure to arise from brainstorming
- Situations where premature organization kills creativity
- Problems where the "shape" of the answer is unknown and must emerge naturally
- Complex systems design where coherence must emerge from many independently
  conceived components
- Strategic planning where many stakeholders must spontaneously align

### Step Types

| Step Type | Description |
|-----------|-------------|
| `disordered_state` | High-entropy unconstrained response |
| `order_measurement` | Coherence/consensus quantification |
| `cooled_state` | Constrained re-generation at lower temperature |
| `critical_check` | Has the order parameter crossed the threshold? |
| `symmetry_breaking` | Committing to a specific ordered direction |
| `ordered_state` | Refined solution in the ordered phase |
| `final_output` | Polished final answer |

### Parameters

```python
class PhaseTransition(ReasoningEngine):
    name = "phase_transition"
    description = (
        "Explores in maximum-entropy disorder, then cools until "
        "a phase transition spontaneously crystallizes coherent order."
    )

    def __init__(
        self,
        num_perspectives: int = 5,        # N parallel disordered responses
        max_cooling_rounds: int = 3,      # Maximum cooling iterations
        cooling_factor: float = 0.7,      # Temperature multiplier per round
        critical_threshold: float = 0.7,  # Order parameter value triggering transition
        initial_temperature: float = 1.0,
        ordered_temperature: float = 0.3,
        personalities: list[str] | None = None,
            # Default: skeptic, optimist, pragmatist, theorist, contrarian
    ) -> None: ...
```

---

## Engine 4: EntropicFunnel

### File: `entropic_funnel.py`

### Physics Basis

**Shannon Entropy, Information Theory, and the Maximum Entropy Principle**

Claude Shannon (1948) defined the entropy of a discrete probability distribution
as a measure of uncertainty or information content:

```
H(X) = -sum_i p(x_i) * log_2(p(x_i))
```

Maximum entropy occurs when all outcomes are equally likely (uniform distribution).
Minimum entropy (H=0) occurs when the outcome is certain (one probability = 1).
The process of **learning** can be understood as entropy reduction: each piece of
evidence eliminates possibilities and lowers the entropy of the posterior
distribution.

Jaynes' **Maximum Entropy Principle** (1957) states: given incomplete information,
the least biased probability distribution to assign is the one with maximum entropy
subject to known constraints. As constraints are added, the maximum-entropy
distribution narrows (entropy decreases), eventually converging to a sharp
prediction.

The **information funnel** (or information bottleneck, Tishby et al. 1999)
formalizes this as a compression process: at each stage, information about
irrelevant details is discarded while information about the target variable is
preserved. The system passes through stages of decreasing entropy, forming a
funnel shape.

Related: Boltzmann's entropy formula `S = k_B * ln(W)` where `W` is the number of
microstates, connects thermodynamic entropy to the number of ways a macrostate can
be realized. Reducing entropy means reducing the number of viable microstates
(solutions).

### Algorithm

**Phase 1: MAXIMUM ENTROPY STATE** (1 LLM call)
1. LLM call: "Generate the broadest possible analysis of this problem. List ALL
   potentially relevant dimensions, factors, approaches, and considerations.
   Do not prioritize -- include everything that has any plausibility. The goal is
   maximum coverage (maximum entropy)."
   - System prompt: "You are a maximum-entropy generator. Your goal is to be
     maximally uncertain -- include every possibility without preference."
   - Temperature: 1.0
   - Output: JSON array of `{dimension, options: [str], relevance_uncertain: bool}`
   - Compute initial entropy: H = log_2(total_options)
   - Records: `maximum_entropy` step with H value

**Phase 2: CONSTRAINT IDENTIFICATION** (1 LLM call)
2. LLM call: "Identify the hard constraints -- facts, requirements, and logical
   necessities that MUST be satisfied. Each constraint eliminates some options and
   reduces entropy. Order constraints from most restrictive (eliminates the most
   options) to least restrictive."
   - Output: JSON array of `{constraint, eliminates: [option_ids], entropy_reduction: float}`
   - Records: `constraint_identification` step

**Phase 3: ITERATIVE ENTROPY REDUCTION** (C LLM calls, one per constraint group)
3. Apply constraints in order of restrictiveness. After each constraint group:
   - LLM call: "The constraint [X] has been applied. The following options are
     eliminated: [eliminated]. Given the remaining option space, what is the most
     informative question or analysis that would further reduce entropy? Perform
     that analysis."
   - Compute new entropy: `H_new = log_2(remaining_options)`
   - Records: `entropy_reduction` step per constraint group with H_before, H_after,
     and the information gain `Delta_H = H_before - H_after`

**Phase 4: INFORMATION BOTTLENECK** (1 LLM call)
4. LLM call: "We have narrowed from H_initial to H_current bits of uncertainty.
   The remaining options are [remaining]. Apply the information bottleneck: what is
   the MINIMAL representation of the problem that preserves ALL information relevant
   to the answer while discarding everything irrelevant? Compress."
   - System prompt: "You are an information bottleneck. Discard all irrelevant
     detail. Preserve only what matters for the answer."
   - Temperature: 0.3
   - Records: `information_bottleneck` step

**Phase 5: LOW-ENTROPY SYNTHESIS** (1 LLM call)
5. LLM call: "Starting from the minimal representation (post-bottleneck), construct
   the answer. The entropy is now low -- the answer should be sharp and specific,
   not hedging or uncertain. Commit to the conclusion that the information funnel
   converged toward."
   - Temperature: 0.2
   - Records: `low_entropy_synthesis` step, `final_output` step

### LLM Call Pattern

```
MAX_ENTROPY(1) --> CONSTRAINTS(1) --> REDUCTION(C) --> BOTTLENECK(1) --> SYNTHESIS(1)

Total: 4 + C calls, where C = number of constraint groups
With defaults (C=4 constraint groups): 8 calls
Dependencies: strictly sequential (each reduction depends on the prior state)
```

### When to Use

- Problems with a large solution space that must be narrowed systematically
- Analysis tasks where the problem is "too open" and needs progressive focusing
- Requirements engineering and specification refinement
- Any problem where the key challenge is "there are too many possibilities"
- Research questions that need systematic elimination of hypotheses

### Step Types

| Step Type | Description |
|-----------|-------------|
| `maximum_entropy` | Broadest possible analysis, all options enumerated |
| `constraint_identification` | Hard constraints ordered by restrictiveness |
| `entropy_reduction` | Application of constraint group with entropy measurement |
| `information_bottleneck` | Minimal representation preserving relevant information |
| `low_entropy_synthesis` | Sharp, committed answer from low-entropy state |
| `final_output` | Polished final answer |

### Parameters

```python
class EntropicFunnel(ReasoningEngine):
    name = "entropic_funnel"
    description = (
        "Starts at maximum entropy (all possibilities open), "
        "applies constraints to reduce entropy through an information "
        "funnel until a sharp answer crystallizes."
    )

    def __init__(
        self,
        max_constraint_groups: int = 4,     # Number of constraint-application steps
        target_entropy: float = 1.0,        # Stop when entropy drops below this (bits)
        initial_temperature: float = 1.0,
        bottleneck_temperature: float = 0.3,
        synthesis_temperature: float = 0.2,
        entropy_computation: str = "log2",  # "log2" or "natural"
    ) -> None: ...
```

### DAG Structure

```
query
  +-- maximum_entropy (H = 5.7 bits)
  +-- constraint_identification
  |     +-- entropy_reduction_0 (H: 5.7 -> 4.1)
  |     +-- entropy_reduction_1 (H: 4.1 -> 2.8)
  |     +-- entropy_reduction_2 (H: 2.8 -> 1.5)
  |     +-- entropy_reduction_3 (H: 1.5 -> 0.8)
  +-- information_bottleneck (minimal representation)
  +-- low_entropy_synthesis
  +-- final_output
```

---

## Engine 5: RenormalizationGroup

### File: `renormalization_group.py`

### Physics Basis

**The Renormalization Group (RG) and Scale Invariance**

The renormalization group, developed by Kenneth Wilson (Nobel Prize 1982), is one
of the most powerful ideas in theoretical physics. It provides a systematic method
for understanding how physical systems behave across different length scales.

The core procedure:

1. **Coarse-grain**: Replace a block of microscopic degrees of freedom with a single
   effective degree of freedom (e.g., replace a block of spins with a single spin
   representing the block average).
2. **Rescale**: Zoom out to restore the original resolution.
3. **Renormalize**: Adjust the coupling constants (parameters) so that the
   coarse-grained, rescaled system has the same macroscopic behavior as the original.

This is applied iteratively. At each step, irrelevant microscopic details are
integrated out, and only the features that matter at the current scale are retained.
The transformation defines a flow in parameter space:

```
K' = R(K)  (RG transformation: parameters K flow to new parameters K')
```

Fixed points of the RG flow (`K* = R(K*)`) correspond to scale-invariant systems --
they look the same at every scale. Near a fixed point, the flow can be linearized,
and the eigenvalues of the linearized transformation classify parameters as:

- **Relevant**: grow under RG flow (matter at large scales)
- **Marginal**: unchanged under RG flow (borderline important)
- **Irrelevant**: shrink under RG flow (wash out at large scales)

This classification explains universality: systems with different microscopic
details but the same relevant parameters flow to the same fixed point and exhibit
identical macroscopic behavior.

### Algorithm

**Phase 1: MICROSCOPIC ANALYSIS** (1 LLM call)
1. LLM call: "Analyze this problem at the finest grain of detail. List every
   specific fact, constraint, variable, edge case, and nuance. Do not simplify --
   capture the full microscopic complexity."
   - Temperature: 0.4
   - Records: `microscopic` step with detailed analysis

**Phase 2: COARSE-GRAINING (iterative, S scales)** (S LLM calls)
2. For each scale s from 1 to S:
   - LLM call: "You are performing a renormalization group transformation.
     Take the analysis from the previous scale and coarse-grain it: group related
     details into effective variables. Replace specific facts with the general
     principles they represent. Discard details that are irrelevant at this broader
     scale. The analysis should now be at [scale_label] granularity."
   - Scale labels: "detailed" -> "component-level" -> "system-level" -> "strategic"
   - Temperature: 0.3 + 0.1 * s (slightly increasing with abstraction)
   - Records: `coarse_grain_s{s}` step, tracking which details were kept vs discarded

**Phase 3: FIXED-POINT DETECTION** (1 LLM call)
3. LLM call: "Compare the analyses at all scales. Identify the 'fixed points' --
   conclusions or principles that remain true regardless of the scale of analysis.
   These are the scale-invariant truths. Also classify each finding as:
   - RELEVANT: matters more at larger scales (strategic importance)
   - MARGINAL: equally important at all scales
   - IRRELEVANT: matters only at fine scales (implementation details)"
   - Output: JSON with `fixed_points`, `relevant`, `marginal`, `irrelevant`
   - Records: `fixed_point` step with classification

**Phase 4: TOP-DOWN RECONSTRUCTION** (1 LLM call)
4. LLM call: "Starting from the scale-invariant fixed points and relevant
   parameters, reconstruct the answer top-down. The fixed points provide the
   skeleton; the relevant parameters determine the structure; the marginal
   parameters add nuance; the irrelevant parameters provide optional detail for
   completeness."
   - Temperature: 0.4
   - Records: `reconstruction` step

**Phase 5: UNIVERSALITY CHECK** (1 LLM call)
5. LLM call: "The renormalization analysis has identified the essential structure
   of this problem. Are there other, apparently different problems that would flow
   to the same fixed point (exhibit the same essential structure)? If so, what
   lessons from those analogous problems apply here? This is the universality class
   of this problem."
   - Records: `universality` step, `final_output` step

### LLM Call Pattern

```
MICROSCOPIC(1) --> COARSE_GRAIN(S) --> FIXED_POINT(1) -->
RECONSTRUCTION(1) --> UNIVERSALITY(1)

Total: 4 + S calls
With defaults (S=3 scales): 7 calls
Dependencies: strictly sequential
```

### When to Use

- Problems with multiple levels of abstraction (code architecture, organizational
  design, complex systems)
- Situations where you need to distinguish "what matters at scale" from "what is
  just a detail"
- Engineering problems where the same principle applies at component, module, and
  system levels
- Strategic analysis where zooming in and out reveals different priorities
- Any problem where finding the scale-invariant truth is the goal

### Step Types

| Step Type | Description |
|-----------|-------------|
| `microscopic` | Finest-grain detailed analysis |
| `coarse_grain` | Coarse-grained analysis at a specific scale |
| `fixed_point` | Scale-invariant conclusions with RG classification |
| `reconstruction` | Top-down answer from fixed points |
| `universality` | Analogous problems in the same universality class |
| `final_output` | Polished final answer |

### Parameters

```python
class RenormalizationGroup(ReasoningEngine):
    name = "renormalization_group"
    description = (
        "Coarse-grains reasoning across scales, identifies "
        "scale-invariant fixed points, and classifies findings "
        "as relevant, marginal, or irrelevant."
    )

    def __init__(
        self,
        num_scales: int = 3,                # Number of coarse-graining steps
        scale_labels: list[str] | None = None,
            # Default: ["component-level", "system-level", "strategic"]
        microscopic_temperature: float = 0.4,
        abstraction_temperature_base: float = 0.3,
        abstraction_temperature_step: float = 0.1,
        enable_universality: bool = True,   # Whether to perform universality check
    ) -> None: ...
```

---

## Engine 6: GaugeInvariance

### File: `gauge_invariance.py`

### Physics Basis

**Gauge Symmetry and Gauge Invariance**

Gauge symmetry is the foundational principle of modern particle physics. A gauge
transformation is a local change of description that does not change the physics.
For example, in electromagnetism, the electric and magnetic fields (the physical
observables) are invariant under the gauge transformation of the vector potential:

```
A_mu -> A_mu + partial_mu * Lambda(x)
```

for any scalar function `Lambda(x)`. The potential `A_mu` is not physical -- it is
a convenient mathematical description. Different gauges (different choices of
`Lambda`) give different potentials but identical physical predictions. The
requirement that physics be gauge-invariant constrains the allowed interactions
and, remarkably, **determines** the structure of the electromagnetic, weak, and
strong forces (Yang-Mills theory).

Key principles:
- **Gauge redundancy**: Multiple descriptions of the same physics. The "gauge
  freedom" is NOT physical -- it is an artifact of the description.
- **Gauge invariants**: Quantities that are the same in all gauges are the physical
  observables. Only gauge-invariant quantities correspond to measurable reality.
- **Gauge fixing**: To perform a concrete calculation, you must "fix a gauge" --
  choose a specific description. But the result must not depend on this choice.
- **Noether's theorem**: Every continuous symmetry implies a conservation law.
  Gauge symmetry (local symmetry) implies the existence of force-carrying gauge
  bosons (photon, W/Z, gluons).

The mapping to reasoning: different **framings**, **representations**, and
**perspectives** on a problem are like different gauges. They are all valid
descriptions, but the **invariant** content -- the conclusions that hold regardless
of framing -- is the "physics." The engine systematically identifies what changes
across framings (gauge artifacts / framing effects) versus what stays the same
(gauge invariants / robust conclusions).

### Algorithm

**Phase 1: MULTI-GAUGE GENERATION** (K LLM calls, parallelizable)
1. For each of K "gauges" (framings/representations), make one LLM call:
   - Gauge 1: Frame the problem as an optimization problem
   - Gauge 2: Frame the problem as a constraint satisfaction problem
   - Gauge 3: Frame the problem as a narrative/story
   - Gauge 4: Frame the problem from the opposite perspective (invert the question)
   - Gauge 5: Frame the problem in the most abstract possible terms
   - Each call has a unique system prompt that enforces the framing
   - Temperature: 0.5
   - Records: `gauge_representation` step per gauge

**Phase 2: INVARIANT EXTRACTION** (1 LLM call)
2. LLM call: "You have seen the same problem analyzed under K different framings.
   Identify the GAUGE INVARIANTS -- conclusions, insights, or claims that appear
   in ALL or MOST framings, regardless of how the problem was represented. These
   are the 'physical observables' -- the robust truths. Also identify GAUGE
   ARTIFACTS -- claims that appear in only one framing and are likely artifacts
   of that particular representation."
   - Output: JSON with `invariants: [{claim, appears_in: [gauge_ids],
     robustness: float}]`, `artifacts: [{claim, only_in: gauge_id, why_artifact: str}]`
   - Records: `invariant_extraction` step

**Phase 3: CONSERVATION LAW DERIVATION** (1 LLM call)
3. LLM call: "Noether's theorem: every symmetry implies a conservation law.
   The gauge invariants you identified imply that certain properties are CONSERVED
   across all framings. What are these conserved quantities? What constraints do
   they impose on any valid solution? These are the inviolable principles."
   - System prompt: "You are deriving conservation laws from symmetry. What
     must be preserved regardless of how the problem is approached?"
   - Records: `conservation_law` step

**Phase 4: GAUGE-FIXED SOLUTION** (1 LLM call)
4. LLM call: "Now fix a gauge -- choose the single most natural and useful framing
   for this problem. Within that gauge, construct a concrete solution that satisfies
   all conservation laws (inviolable constraints) and is consistent with all gauge
   invariants (robust conclusions). The solution may include framing-specific details
   (gauge-dependent quantities), but clearly distinguish them from the invariant
   content."
   - Temperature: 0.4
   - Records: `gauge_fixed_solution` step

**Phase 5: GAUGE INDEPENDENCE VERIFICATION** (1 LLM call)
5. LLM call: "Verify that the solution's core conclusions are gauge-independent.
   If someone approached this problem from a completely different framing, would
   they reach the same essential conclusions? If not, what part of your solution
   is a framing artifact?"
   - Records: `gauge_check` step, `final_output` step

### LLM Call Pattern

```
MULTI_GAUGE(K) --> INVARIANT_EXTRACTION(1) --> CONSERVATION_LAWS(1) -->
GAUGE_FIXED_SOLUTION(1) --> VERIFICATION(1)

Total: K + 4 calls
With defaults (K=5): 9 calls
Parallelism: Phase 1 is fully parallel (K independent calls)
```

### When to Use

- Problems where the answer might depend on how the question is framed (avoid
  framing bias)
- Separating robust conclusions from representation artifacts
- Cross-functional analysis where different teams use different frameworks
- Any situation where you want to know "what is actually true regardless of
  perspective?"
- Philosophy, policy, and strategy where different ideological framings produce
  different conclusions -- the invariants are the non-ideological truths

### Step Types

| Step Type | Description |
|-----------|-------------|
| `gauge_representation` | Analysis under a specific framing/gauge |
| `invariant_extraction` | Identification of frame-independent conclusions |
| `conservation_law` | Inviolable principles derived from symmetry |
| `gauge_fixed_solution` | Concrete solution in a chosen framing |
| `gauge_check` | Verification of gauge independence |
| `final_output` | Polished final answer |

### Parameters

```python
class GaugeInvariance(ReasoningEngine):
    name = "gauge_invariance"
    description = (
        "Analyzes the problem under multiple framings (gauges) "
        "and extracts the invariant conclusions that hold regardless "
        "of representation."
    )

    def __init__(
        self,
        num_gauges: int = 5,                # Number of distinct framings
        gauge_framings: list[str] | None = None,
            # Default: optimization, constraint-satisfaction, narrative,
            #          inverted, abstract
        robustness_threshold: float = 0.6,  # Min fraction of gauges for "invariant"
        gauge_temperature: float = 0.5,
        solution_temperature: float = 0.4,
    ) -> None: ...
```

---

## Engine 7: PerturbativeExpansion

### File: `perturbative_expansion.py`

### Physics Basis

**Perturbation Theory**

Perturbation theory is the workhorse method of theoretical physics for solving
problems that are "close to" a solvable problem. The idea: if you cannot solve
the full Hamiltonian `H = H_0 + lambda * V`, where `H_0` is solvable and `V` is
a "perturbation," you can expand the solution as a power series in the coupling
constant `lambda`:

```
E_n = E_n^(0) + lambda * E_n^(1) + lambda^2 * E_n^(2) + ...

|psi_n> = |psi_n^(0)> + lambda * |psi_n^(1)> + lambda^2 * |psi_n^(2)> + ...
```

Each order provides a correction:
- **Zeroth order** `E^(0)`: The unperturbed solution (ignoring the complication)
- **First order** `E^(1)`: The leading correction (easiest improvement)
- **Second order** `E^(2)`: Accounts for indirect effects (perturbation modifying
  the perturbation)
- **Higher orders**: Increasingly subtle corrections

The series may converge (each correction is smaller than the last) or diverge
(corrections grow -- the perturbation is too large). Convergence is assessed by
comparing the magnitude of successive corrections.

Feynman diagrams provide a visual representation: each order in perturbation theory
corresponds to diagrams with more interaction vertices, representing more complex
physical processes (virtual particle loops, scattering events).

The mapping to reasoning: start with a simple, tractable answer to the problem
(zeroth order). Then systematically add corrections for complications that were
initially ignored. Each correction refines the answer, and the series of corrections
is checked for convergence (diminishing returns indicate the answer has stabilized).

### Algorithm

**Phase 1: ZEROTH-ORDER SOLUTION** (1 LLM call)
1. LLM call: "Solve a SIMPLIFIED version of this problem. Ignore complications,
   edge cases, and second-order effects. What is the straightforward, first-
   principles answer if the world were simple?"
   - System prompt: "You are computing the zeroth-order solution. Simplify
     aggressively. Get the essential answer right, even if incomplete."
   - Temperature: 0.3
   - Records: `zeroth_order` step

**Phase 2: PERTURBATION IDENTIFICATION** (1 LLM call)
2. LLM call: "What complications, real-world factors, edge cases, and second-order
   effects were ignored in the zeroth-order solution? List them in order of
   importance (largest perturbation first). For each, estimate its 'coupling
   strength' (0.0 = negligible, 1.0 = completely changes the answer)."
   - Output: JSON array of `{perturbation, coupling_strength: float, description}`
   - Records: `perturbation_identification` step

**Phase 3: FIRST-ORDER CORRECTION** (1 LLM call)
3. LLM call: "Apply the strongest perturbation to the zeroth-order solution. How
   does accounting for [perturbation] modify the answer? Compute the first-order
   correction. The corrected answer is: zeroth-order + first-order correction."
   - Temperature: 0.4
   - Records: `first_order` step with `correction_magnitude`

**Phase 4: HIGHER-ORDER CORRECTIONS** (P-1 LLM calls)
4. For each additional perturbation (in order of coupling strength):
   - LLM call: "The current answer incorporates corrections up to order [N-1].
     Now apply the next perturbation [P]. How does it modify the current answer?
     Note: this perturbation may interact with previously applied perturbations
     (cross-terms). Account for these interactions."
   - Compute correction magnitude: how much did this order change the answer?
   - Records: `order_N_correction` step with `correction_magnitude`

**Phase 5: CONVERGENCE CHECK** (programmatic)
5. After each correction, check whether the series is converging:
   - If `|correction_N| < convergence_threshold * |correction_{N-1}|`: converging
   - If `|correction_N| > |correction_{N-1}|`: diverging -- the perturbation is too
     strong. Flag this and switch to a "non-perturbative" warning.
   - Records: `convergence_check` step per order

**Phase 6: RESUMMATION / SYNTHESIS** (1 LLM call)
6. LLM call: "The perturbative expansion has been computed to order [max_order].
   Convergence status: [converging/diverging]. If converging: present the
   perturbatively corrected answer. If diverging: acknowledge that the
   perturbative approach has broken down for [specific perturbation], which
   indicates this aspect requires non-perturbative treatment (fundamental
   rethinking)."
   - Records: `resummation` step, `final_output` step

### LLM Call Pattern

```
ZEROTH(1) --> PERTURBATIONS(1) --> FIRST_ORDER(1) --> HIGHER_ORDERS(P-1) -->
RESUMMATION(1)

Total: 3 + P calls, where P = number of perturbation orders applied
With defaults (P=4): 7 calls
Dependencies: strictly sequential (each order depends on all previous)
```

### When to Use

- Problems that have a "simple version" with known complications
- Engineering problems with an ideal-case solution plus real-world corrections
- Estimations where you need progressively better accuracy
- Any problem where a good first approximation exists and systematic improvement
  is possible
- Risk assessment (each perturbation is a risk factor modifying the baseline)
- NOT for: problems where the "perturbation" is larger than the "base" (need
  a non-perturbative approach instead)

### Step Types

| Step Type | Description |
|-----------|-------------|
| `zeroth_order` | Simplified base solution |
| `perturbation_identification` | Complications ranked by coupling strength |
| `first_order` | Leading correction from strongest perturbation |
| `order_N_correction` | Higher-order correction at order N |
| `convergence_check` | Is the correction series converging? |
| `resummation` | Final answer incorporating all corrections |
| `final_output` | Polished final answer |

### Parameters

```python
class PerturbativeExpansion(ReasoningEngine):
    name = "perturbative_expansion"
    description = (
        "Starts with a simplified zeroth-order solution and "
        "applies successive perturbative corrections from "
        "identified complications."
    )

    def __init__(
        self,
        max_perturbation_order: int = 4,   # Maximum correction orders
        convergence_threshold: float = 0.5, # Ratio threshold for convergence
        coupling_cutoff: float = 0.1,       # Ignore perturbations below this strength
        zeroth_order_temperature: float = 0.3,
        correction_temperature: float = 0.4,
        warn_on_divergence: bool = True,    # Flag non-perturbative breakdowns
    ) -> None: ...
```

### DAG Structure

```
query
  +-- zeroth_order (E^(0): base solution)
  +-- perturbation_identification
  |     +-- perturbation_0 (lambda_0 = 0.8, strongest)
  |     +-- perturbation_1 (lambda_1 = 0.5)
  |     +-- perturbation_2 (lambda_2 = 0.3)
  |     +-- perturbation_3 (lambda_3 = 0.1)
  +-- first_order (E^(1): correction from perturbation_0)
  |     +-- convergence_check_1
  +-- order_2_correction (E^(2): correction from perturbation_1 + cross-terms)
  |     +-- convergence_check_2
  +-- order_3_correction (E^(3))
  |     +-- convergence_check_3
  +-- order_4_correction (E^(4))
  |     +-- convergence_check_4
  +-- resummation
  +-- final_output
```

---

## Engine 8: LeastActionPath

### File: `least_action_path.py`

### Physics Basis

**Lagrangian Mechanics and the Principle of Least Action**

The principle of least action (Hamilton's principle) is the most fundamental
formulation of classical mechanics. Instead of computing forces and accelerations
at each instant (Newtonian mechanics), it considers all possible paths a system
could take between two states and selects the one that extremizes the action
functional:

```
S[q(t)] = integral_{t_1}^{t_2} L(q, dq/dt, t) dt
```

where `L = T - V` is the Lagrangian (kinetic energy minus potential energy). The
actual physical path satisfies the Euler-Lagrange equation:

```
d/dt (partial L / partial q_dot) - partial L / partial q = 0
```

Key insights:
- **Variational principle**: The correct path is the one that makes the action
  stationary (delta S = 0). Nature "explores all paths" and selects the optimal one.
- **Path integral formulation** (Feynman): In quantum mechanics, ALL paths
  contribute to the amplitude, weighted by `exp(i * S / hbar)`. Paths near the
  classical path interfere constructively; paths far from it interfere
  destructively. The classical path dominates because it is the stationary-phase
  point.
- **Configuration space**: The problem is defined in terms of generalized
  coordinates `q`, which may be abstract (not just positions). The "path" is
  through this abstract configuration space.

The mapping to reasoning: define a "configuration space" of reasoning states. The
"action" is a cost functional that balances effort (kinetic energy -- how much
the reasoning changes between steps) against quality loss (potential energy -- how
far the current state is from optimal). The engine generates multiple candidate
reasoning paths, evaluates their action, and selects the path of least action --
the one that achieves the best result with the most elegant, economical reasoning.

### Algorithm

**Phase 1: BOUNDARY CONDITIONS** (1 LLM call)
1. LLM call: "Define the initial state (what we know at the start) and the target
   state (what a complete answer looks like). These are the boundary conditions of
   the variational problem. Also define the quality criteria that the path must
   optimize."
   - Output: JSON with `initial_state`, `target_state`, `quality_criteria: [str]`
   - Records: `boundary_conditions` step

**Phase 2: PATH GENERATION** (M LLM calls, parallelizable)
2. For each of M candidate paths, generate a complete reasoning trajectory from
   initial state to target state:
   - Path 1: "Shortest path" -- minimum number of reasoning steps, direct approach
   - Path 2: "Scenic path" -- explore broadly before converging
   - Path 3: "Conservative path" -- smallest changes at each step, safest reasoning
   - Path 4: "Bold path" -- large leaps of reasoning, maximum creativity
   - Each path is a sequence of reasoning steps with explicit state at each point
   - Temperature: varies by path type (0.3 for shortest/conservative, 0.8 for
     scenic/bold)
   - Records: `candidate_path` step per path, with child steps for each reasoning
     stage within the path

**Phase 3: ACTION EVALUATION** (1 LLM call)
3. LLM call: "Evaluate the 'action' of each reasoning path. The action balances:
   - KINETIC COST: How much reasoning effort was expended? How many large jumps
     or direction changes? (Lower is better)
   - POTENTIAL COST: How close does the final state come to the target state?
     How well does it satisfy the quality criteria? (Lower is better -- potential
     energy at the target should be zero)
   - TOTAL ACTION: S = integral of (kinetic_cost - quality_achieved) along the path
   Rate each path's action as a single float. The path with the LOWEST action is
   the optimal one."
   - Output: JSON array of `{path_id, kinetic_cost, potential_cost, total_action,
     reasoning}`
   - Records: `action_evaluation` step

**Phase 4: EULER-LAGRANGE REFINEMENT** (1 LLM call)
4. LLM call: "Take the lowest-action path and refine it using the Euler-Lagrange
   principle. At each step along the path, ask: could this step be improved to
   further reduce the action? Is there unnecessary effort (kinetic cost) that can
   be removed? Is there quality loss (potential cost) that can be recovered? Produce
   the refined optimal path."
   - System prompt: "You are applying variational optimization. The optimal path
     is the one where no local modification can reduce the total action."
   - Temperature: 0.4
   - Records: `euler_lagrange` step

**Phase 5: STATIONARY POINT VERIFICATION** (1 LLM call)
5. LLM call: "Verify that the refined path is truly a stationary point of the
   action. Test: if you perturb any step slightly, does the action increase?
   If you find a step where perturbation reduces the action, adjust it. Present
   the final optimized reasoning path as the answer."
   - Records: `stationary_verification` step, `final_output` step

### LLM Call Pattern

```
BOUNDARY(1) --> PATH_GEN(M) --> ACTION_EVAL(1) --> EULER_LAGRANGE(1) -->
VERIFICATION(1)

Total: M + 4 calls
With defaults (M=4 paths): 8 calls
Parallelism: Phase 2 is fully parallel (M independent path generations)
```

### When to Use

- Problems where multiple solution strategies exist and you need the most efficient
  one
- Optimization problems where both solution quality and reasoning economy matter
- Engineering design where elegance/simplicity is valued alongside correctness
- Any problem where the "how you get there" matters as much as "where you end up"
- Planning and process design (finding the optimal process, not just the optimal
  outcome)

### Step Types

| Step Type | Description |
|-----------|-------------|
| `boundary_conditions` | Initial state, target state, quality criteria |
| `candidate_path` | Complete reasoning trajectory (with sub-steps) |
| `action_evaluation` | Kinetic + potential cost assessment per path |
| `euler_lagrange` | Variational refinement of the optimal path |
| `stationary_verification` | Perturbation test confirming optimality |
| `final_output` | Polished final answer |

### Parameters

```python
class LeastActionPath(ReasoningEngine):
    name = "least_action_path"
    description = (
        "Generates multiple reasoning paths, evaluates their "
        "'action' (effort vs quality), and variationally "
        "optimizes the path of least action."
    )

    def __init__(
        self,
        num_paths: int = 4,                # Candidate reasoning paths
        path_strategies: list[str] | None = None,
            # Default: shortest, scenic, conservative, bold
        kinetic_weight: float = 0.4,       # Weight of effort in action functional
        potential_weight: float = 0.6,     # Weight of quality in action functional
        refinement_temperature: float = 0.4,
        enable_verification: bool = True,
    ) -> None: ...
```

---

## Engine 9: QuantumTunneling

### File: `quantum_tunneling.py`

### Physics Basis

**Quantum Tunneling (Barrier Penetration)**

In classical mechanics, a particle with energy `E` encountering a potential barrier
of height `V_0 > E` is always reflected -- it cannot cross the barrier. In quantum
mechanics, the wavefunction does not abruptly go to zero at the barrier. Instead,
it decays exponentially inside the barrier:

```
psi(x) ~ exp(-kappa * x),  where kappa = sqrt(2m(V_0 - E)) / hbar
```

If the barrier has finite width `d`, there is a nonzero probability of the
wavefunction emerging on the other side:

```
T ~ exp(-2 * kappa * d)  (transmission probability)
```

This is quantum tunneling. The particle "tunnels through" a classically forbidden
region. The tunneling probability depends exponentially on barrier width and height,
so thin or low barriers are penetrated more easily.

Real-world manifestations:
- **Alpha decay**: Nucleons tunnel through the nuclear potential barrier
- **Scanning tunneling microscope**: Electron tunneling current maps surface
  topography
- **Josephson junctions**: Cooper pairs tunnel across superconductor gaps
- **Tunnel diodes**: Exploit quantum tunneling for ultrafast switching

The mapping to reasoning: when a reasoning process is "stuck" -- trapped in a local
minimum by a barrier (a seemingly insurmountable objection, a false assumption, or
a conceptual block) -- the engine attempts to "tunnel through" by:
1. Identifying the barrier (what is blocking progress)
2. Computing a "tunneling probability" (how likely is it that the barrier is
   actually penetrable?)
3. Attempting passage by suspending the assumption that creates the barrier
4. Checking whether the other side of the barrier contains a viable solution

### Algorithm

**Phase 1: INITIAL APPROACH** (1 LLM call)
1. LLM call: "Attempt to solve this problem directly. If you encounter any point
   where you feel stuck, blocked, or unsure how to proceed, explicitly flag it as
   a BARRIER. Describe the barrier and what lies beyond it (if you can see past it)."
   - Temperature: 0.5
   - Records: `initial_approach` step

**Phase 2: BARRIER IDENTIFICATION** (1 LLM call)
2. LLM call: "Analyze the barriers encountered. For each barrier:
   - What assumption or constraint creates it?
   - How 'wide' is the barrier? (narrow = specific technical issue;
     wide = fundamental conceptual block)
   - How 'high' is the barrier? (low = minor difficulty; high = seems impossible)
   - What is on the other side? (what solution would be available if the barrier
     did not exist?)
   Return JSON: [{barrier, assumption, width: float, height: float,
   beyond: str}]"
   - Records: `barrier_analysis` step with barrier properties

**Phase 3: TUNNELING PROBABILITY COMPUTATION** (programmatic + 1 LLM call)
3. Compute tunneling probability for each barrier:
   ```
   T_i = exp(-2 * width_i * height_i)  (analogous to WKB approximation)
   ```
   Barriers with higher tunneling probability are more likely to be penetrable.
4. LLM call: "Rank these barriers by tunneling probability (penetrability). For the
   most penetrable barrier: what would happen if we simply SUSPENDED the assumption
   that creates it? Is the assumption actually necessary, or is it a false
   constraint?"
   - Records: `tunneling_probability` step

**Phase 4: TUNNELING ATTEMPT** (B LLM calls, one per barrier attempted)
5. For each barrier (in order of tunneling probability, up to max_tunnels):
   - LLM call: "TUNNEL THROUGH THIS BARRIER. The barrier is: [barrier].
     The blocking assumption is: [assumption]. For this phase, SUSPEND this
     assumption entirely. Solve the problem as if the barrier does not exist.
     Then check: is the solution on the other side actually valid? Did
     suspending the assumption break anything fundamental, or was the barrier
     an illusion?"
   - Temperature: 0.6 (creative enough to escape the trap)
   - Output includes: `solution_beyond`, `barrier_was_real: bool`,
     `what_broke: str|null`
   - Records: `tunneling_attempt` step per barrier

**Phase 5: POST-TUNNELING ASSESSMENT** (1 LLM call)
6. LLM call: "Tunneling results: [summary of which barriers were penetrated,
   which were real]. For barriers that were successfully tunneled (assumption
   was false or unnecessary): integrate the solution found beyond the barrier.
   For barriers that were real (tunneling broke something fundamental):
   acknowledge the genuine constraint and work within it."
   - Records: `post_tunneling` step

**Phase 6: RECONSTRUCTED SOLUTION** (1 LLM call)
7. LLM call: "Present the final answer. It should incorporate insights from both
   sides of any successfully tunneled barriers. If all barriers were real, present
   the best solution within the constraints."
   - Records: `reconstructed_solution` step, `final_output` step

### LLM Call Pattern

```
INITIAL(1) --> BARRIER_ANALYSIS(1) --> TUNNELING_PROB(1) -->
TUNNELING_ATTEMPTS(B) --> POST_TUNNELING(1) --> RECONSTRUCTION(1)

Total: 5 + B calls, where B = number of barriers attempted
With defaults (B=3): 8 calls
Dependencies: sequential (each tunnel depends on barrier analysis)
```

### When to Use

- Problems where the reasoning gets "stuck" at a seemingly impossible constraint
- Situations where conventional wisdom says "you cannot do X" but you want to
  test that assumption
- Creative problem solving where false assumptions block novel solutions
- Engineering problems with apparent hard limits that might be soft limits
- Any problem where "thinking outside the box" means questioning the box itself
- Particularly good as a second pass when another engine has reached a dead end

### Step Types

| Step Type | Description |
|-----------|-------------|
| `initial_approach` | Direct problem-solving attempt with barrier flags |
| `barrier_analysis` | Barrier properties (width, height, assumption) |
| `tunneling_probability` | Penetrability ranking of barriers |
| `tunneling_attempt` | Attempt to solve with suspended assumption |
| `post_tunneling` | Assessment of which barriers were penetrated |
| `reconstructed_solution` | Final answer incorporating tunnel insights |
| `final_output` | Polished final answer |

### Parameters

```python
class QuantumTunneling(ReasoningEngine):
    name = "quantum_tunneling"
    description = (
        "Identifies barriers blocking the reasoning, computes "
        "tunneling probabilities, and attempts to penetrate "
        "barriers by suspending blocking assumptions."
    )

    def __init__(
        self,
        max_tunnels: int = 3,              # Max barriers to attempt tunneling through
        barrier_threshold: float = 0.3,    # Min tunneling probability to attempt
        tunneling_temperature: float = 0.6,
        initial_temperature: float = 0.5,
        wkb_scaling: float = 2.0,          # Scaling factor in T = exp(-scale*w*h)
        allow_multi_barrier: bool = True,   # Attempt sequential barriers
    ) -> None: ...
```

### DAG Structure

```
query
  +-- initial_approach
  |     (barriers flagged inline)
  +-- barrier_analysis
  |     +-- barrier_0 (width=0.3, height=0.7, T=0.66)
  |     +-- barrier_1 (width=0.8, height=0.9, T=0.24)
  |     +-- barrier_2 (width=0.5, height=0.5, T=0.61)
  +-- tunneling_probability (ranked: barrier_0 > barrier_2 > barrier_1)
  +-- tunneling_attempt_0 (barrier_0: succeeded -- assumption was false)
  +-- tunneling_attempt_1 (barrier_2: succeeded -- constraint was soft)
  +-- tunneling_attempt_2 (barrier_1: failed -- genuine hard constraint)
  +-- post_tunneling (2/3 barriers penetrated)
  +-- reconstructed_solution
  +-- final_output
```

---

## Engine 10: EntangledThreads

### File: `entangled_threads.py`

### Physics Basis

**Quantum Entanglement**

Quantum entanglement occurs when two or more particles become correlated such
that the quantum state of the composite system cannot be written as a product of
individual states. For a pair of entangled qubits:

```
|psi> = (1/sqrt(2)) * (|0>_A |1>_B - |1>_A |0>_B)
```

This is a Bell state (specifically, the singlet state). The key properties:

1. **Non-local correlations**: Measuring particle A instantaneously determines
   the state of particle B, regardless of the distance between them. This is not
   communication (no information transfer) but correlation.
2. **Bell inequality violation**: Entangled particles exhibit correlations that
   cannot be explained by any local hidden variable theory (Bell 1964, tested by
   Aspect et al. 1982). The correlations are genuinely quantum.
3. **Monogamy of entanglement**: If A is maximally entangled with B, it cannot
   be entangled with C at all. Entanglement is a finite resource.
4. **Entanglement as a resource**: Quantum teleportation, superdense coding, and
   quantum key distribution all use entanglement as a computational resource.
5. **Decoherence**: Interaction with the environment destroys entanglement.
   Maintaining entanglement requires isolation.

The mapping to reasoning: two or more reasoning threads are "entangled" when they
are deeply correlated -- a conclusion in one thread instantaneously constrains the
conclusions in all entangled threads. This goes beyond simple information sharing
(that would be classical communication). Entanglement means the threads are in a
joint state that cannot be decomposed into independent parts. When one thread
makes a commitment, the entangled threads must immediately update their state
to maintain consistency -- without explicit message passing between them.

This differs from ResonanceNetwork (which uses pairwise similarity scores and
gradual propagation) and from AdversarialSelfPlay (which uses explicit debate).
Entangled threads have instantaneous, inviolable correlations enforced as hard
constraints.

### Algorithm

**Phase 1: THREAD INITIALIZATION** (N LLM calls, parallelizable)
1. Generate N independent reasoning threads, each tackling the problem from a
   different angle or sub-problem:
   - LLM call per thread: "You are reasoning thread [i]. Your specific focus is
     [focus_area]. Analyze the problem from your perspective. Produce your initial
     state: key claims, tentative conclusions, and open questions."
   - Temperature: 0.6
   - Records: `thread_init` step per thread

**Phase 2: ENTANGLEMENT CREATION** (1 LLM call)
2. LLM call: "Examine the N reasoning threads. Identify pairs (or groups) of
   threads that are logically ENTANGLED -- where a conclusion in one thread
   necessarily constrains or determines conclusions in another. These are not
   mere similarities; they are hard logical dependencies. For each entangled pair,
   specify the CONSTRAINT: 'If thread A concludes X, then thread B must conclude Y.'
   Also specify the type: CORRELATED (both must agree) or ANTI-CORRELATED
   (if one says X, the other must say NOT-X)."
   - Output: JSON array of `{threads: [i, j], constraint, type: "correlated"|
     "anti_correlated"}`
   - Records: `entanglement_creation` step

**Phase 3: MEASUREMENT OF THREAD 1** (1 LLM call)
3. LLM call: "Thread 1 is being 'measured' -- forced to make a definite commitment.
   Given its initial state and the problem constraints, what is its final,
   committed conclusion?"
   - Temperature: 0.3 (low, decisive)
   - Records: `measurement` step for thread 1

**Phase 4: ENTANGLED COLLAPSE** (E LLM calls, one per entangled thread)
4. For each thread entangled with thread 1:
   - LLM call: "Thread [j] is entangled with thread 1 via the constraint: [constraint].
     Thread 1 has committed to: [conclusion]. Given this entanglement constraint,
     what must thread [j] now conclude? This is not optional -- the entanglement
     constraint is inviolable. Propagate the collapse."
   - Records: `entangled_collapse` step per thread

**Phase 5: DECOHERENCE CHECK** (1 LLM call)
5. LLM call: "Check for decoherence -- are any of the entanglement constraints
   violated by the collapsed states? Has interaction with the problem's complexity
   (the 'environment') destroyed any entanglements? For each violated constraint,
   explain why it broke and what this reveals about the problem's structure."
   - Records: `decoherence_check` step

**Phase 6: REMAINING THREADS** (R LLM calls)
6. For threads not yet measured or entanglement-collapsed, measure them
   independently, incorporating the context of already-collapsed threads.
   - Records: `independent_measurement` step per thread

**Phase 7: ENTANGLED SYNTHESIS** (1 LLM call)
7. LLM call: "All threads have collapsed to definite states. The entanglement
   constraints that survived decoherence represent deep structural relationships
   in the problem. Synthesize the final answer from all thread conclusions,
   respecting the surviving entanglement structure."
   - Records: `entangled_synthesis` step, `final_output` step

### LLM Call Pattern

```
THREAD_INIT(N) --> ENTANGLEMENT(1) --> MEASURE_FIRST(1) -->
ENTANGLED_COLLAPSE(E) --> DECOHERENCE(1) --> REMAINING(R) --> SYNTHESIS(1)

Total: N + E + R + 4 (where E + R <= N - 1)
With defaults (N=4, E=2, R=1): 4 + 2 + 1 + 4 = 11 calls
Parallelism: Phase 1 is fully parallel; Phase 4 may be partially parallel
```

### When to Use

- Problems with deeply coupled sub-problems where decisions in one area constrain
  another
- Systems design where component choices are interdependent
- Multi-variable optimization where variables are correlated
- Any problem where "you cannot decide X without deciding Y" is a core challenge
- Concurrent engineering decisions that must remain consistent
- Contract or policy design where clauses are logically entangled

### Step Types

| Step Type | Description |
|-----------|-------------|
| `thread_init` | Independent reasoning thread initialization |
| `entanglement_creation` | Identification of hard logical dependencies |
| `measurement` | First thread forced to commit |
| `entangled_collapse` | Correlated threads forced to consistent state |
| `decoherence_check` | Verification of entanglement constraint survival |
| `independent_measurement` | Measurement of non-entangled remaining threads |
| `entangled_synthesis` | Final answer respecting entanglement structure |
| `final_output` | Polished final answer |

### Parameters

```python
class EntangledThreads(ReasoningEngine):
    name = "entangled_threads"
    description = (
        "Parallel reasoning threads with hard entanglement "
        "constraints. Measuring one thread instantaneously "
        "collapses correlated threads."
    )

    def __init__(
        self,
        num_threads: int = 4,              # Number of reasoning threads
        thread_foci: list[str] | None = None,  # Optional: pre-specified focus areas
        entanglement_temperature: float = 0.4,
        measurement_temperature: float = 0.3,
        allow_decoherence: bool = True,     # Whether to check for broken entanglements
        max_entanglement_depth: int = 2,    # Max chain length for transitive entanglement
    ) -> None: ...
```

### DAG Structure

```
query
  +-- thread_init_0 (focus: architecture)
  +-- thread_init_1 (focus: performance)
  +-- thread_init_2 (focus: security)
  +-- thread_init_3 (focus: usability)
  +-- entanglement_creation
  |     entangle(0,1): "If stateless arch, then must use caching for perf"
  |     entangle(1,2): "If caching used, then cache invalidation is security risk"
  |     entangle(0,3): anti-correlated -- "microservices = harder UX"
  +-- measurement_thread_0 (committed: microservices)
  |     +-- entangled_collapse_thread_1 (must use distributed cache)
  |     +-- entangled_collapse_thread_3 (must address UX complexity)
  +-- decoherence_check (entangle(1,2) survived; all intact)
  |     +-- entangled_collapse_thread_2 (must mitigate cache attacks)
  +-- entangled_synthesis
  +-- final_output
```

---

## Summary Table

| # | Engine | Physics | File | LLM Calls (typical) |
|---|--------|---------|------|---------------------|
| 1 | `QuantumSuperposition` | Superposition / measurement / collapse | `quantum_superposition.py` | 7 |
| 2 | `WaveInterference` | Constructive/destructive wave interference | `wave_interference.py` | 5 |
| 3 | `PhaseTransition` | Critical phenomena / spontaneous order | `phase_transition.py` | 20 |
| 4 | `EntropicFunnel` | Shannon entropy / information bottleneck | `entropic_funnel.py` | 8 |
| 5 | `RenormalizationGroup` | Scale-invariant coarse-graining | `renormalization_group.py` | 7 |
| 6 | `GaugeInvariance` | Gauge symmetry / frame-independent truth | `gauge_invariance.py` | 9 |
| 7 | `PerturbativeExpansion` | Perturbation theory / successive corrections | `perturbative_expansion.py` | 7 |
| 8 | `LeastActionPath` | Lagrangian mechanics / variational optimization | `least_action_path.py` | 8 |
| 9 | `QuantumTunneling` | Barrier penetration / assumption suspension | `quantum_tunneling.py` | 8 |
| 10 | `EntangledThreads` | Quantum entanglement / correlated collapse | `entangled_threads.py` | 11 |

---

## Decision Guide: When to Use Which Engine

```
Problem Type                                    Recommended Engine
------------                                    ------------------
"The framing determines the answer"          -> QuantumSuperposition
"Need consensus across perspectives"         -> WaveInterference
"Need emergent order from chaos"             -> PhaseTransition
"Too many possibilities, need to narrow"     -> EntropicFunnel
"What matters depends on scale"              -> RenormalizationGroup
"Is this conclusion real or a framing effect" -> GaugeInvariance
"Good first approximation, need refinement"  -> PerturbativeExpansion
"Need the most elegant/efficient solution"   -> LeastActionPath
"Stuck on a seemingly impossible constraint" -> QuantumTunneling
"Coupled sub-problems with dependencies"     -> EntangledThreads
```

### Quality vs. Speed Tradeoff

```
Quality
  ^
  |  *PhaseTransition
  |       *EntangledThreads
  |    *GaugeInvariance  *QuantumSuperposition
  |        *LeastActionPath     *PerturbativeExpansion
  |   *WaveInterference
  |          *EntropicFunnel
  |     *RenormalizationGroup
  |              *QuantumTunneling
  |
  +-------------------------------------------> Speed
  Fast                                    Slow
```

### Comparison with Existing Engines

| Physics Engine | Most Similar Existing Engine | Key Difference |
|---------------|----------------------------|----------------|
| QuantumSuperposition | EvolutionaryThought | QS explores WHAT QUESTION to ask, not just what answers exist |
| WaveInterference | ResonanceNetwork | WI computes full N-way phase relationships, not just pairwise reinforcement |
| PhaseTransition | SimulatedAnnealing | PT seeks emergent order through cooling; SA optimizes a single solution |
| EntropicFunnel | SocraticInterrogation | EF uses information-theoretic entropy measurement; SI uses questioning |
| RenormalizationGroup | FractalRecursion | RG explicitly classifies relevant/irrelevant at each scale; FR just decomposes |
| GaugeInvariance | AdversarialSelfPlay | GI varies the framing, not the role; extracts invariants, not verdicts |
| PerturbativeExpansion | (none) | Systematic correction series with convergence checking |
| LeastActionPath | DreamWakeCycle | LAP variationally optimizes the reasoning process itself |
| QuantumTunneling | (none) | Explicitly targets false constraints and blocking assumptions |
| EntangledThreads | ResonanceNetwork | ET enforces hard logical constraints; RN uses soft amplification |

---

## Implementation Notes

### Shared Utilities

Several patterns recur across engines and should use shared utilities:

1. `_parse_json_safe(text, default)` -- JSON extraction from LLM output
2. `_parse_json_list(raw, expected)` -- JSON array parsing (exists on ResonanceNetwork)
3. `_parse_score(text)` -- Float score extraction (exists on SimulatedAnnealing)
4. `_make_step(step_type, content, score, metadata)` -- Step factory (exists on
   newer engines)

### _call_llm API Pattern

New engines should use **Pattern A** (dict messages with `system=`) for consistency
with the base class `_call_llm` signature:

```python
await self._call_llm(
    provider,
    [{"role": "user", "content": prompt}],
    trace,
    system=SYSTEM_PROMPT,
    temperature=0.7,
)
```

### Parallelism

Several engines have phases with independent LLM calls that should use
`asyncio.gather()`:

- **QuantumSuperposition**: Phase 4 measurements are independent
- **WaveInterference**: Phase 1 wave sources (if generated separately)
- **PhaseTransition**: All N disordered/cooled states per phase
- **GaugeInvariance**: Phase 1 gauge representations are independent
- **LeastActionPath**: Phase 2 candidate paths are independent
- **EntangledThreads**: Phase 1 thread initializations are independent

```python
results = await asyncio.gather(*[
    self._call_llm(provider, [{"role": "user", "content": prompt_i}], trace,
                   system=system_i, temperature=temp_i)
    for prompt_i, system_i, temp_i in configs
])
```

### Token Budget Management

Engines with potentially high call counts (PhaseTransition=20, EntangledThreads=11)
should implement graceful degradation:

- Track cumulative tokens via `trace.total_tokens`
- Accept an optional `max_tokens_budget` parameter
- Reduce iterations/threads/states if approaching budget

### Registration

All 10 new engines must be added to `openagentflow/reasoning/__init__.py`:

```python
from openagentflow.reasoning.quantum_superposition import QuantumSuperposition
from openagentflow.reasoning.wave_interference import WaveInterference
from openagentflow.reasoning.phase_transition import PhaseTransition
from openagentflow.reasoning.entropic_funnel import EntropicFunnel
from openagentflow.reasoning.renormalization_group import RenormalizationGroup
from openagentflow.reasoning.gauge_invariance import GaugeInvariance
from openagentflow.reasoning.perturbative_expansion import PerturbativeExpansion
from openagentflow.reasoning.least_action_path import LeastActionPath
from openagentflow.reasoning.quantum_tunneling import QuantumTunneling
from openagentflow.reasoning.entangled_threads import EntangledThreads

__all__ = [
    # ... existing engines ...
    "EntangledThreads",
    "EntropicFunnel",
    "GaugeInvariance",
    "LeastActionPath",
    "PerturbativeExpansion",
    "PhaseTransition",
    "QuantumSuperposition",
    "QuantumTunneling",
    "RenormalizationGroup",
    "WaveInterference",
]
```

---

## Usage Examples

### QuantumSuperposition

```python
from openagentflow.reasoning import QuantumSuperposition

engine = QuantumSuperposition(num_states=5, num_bases=3)
trace = await engine.reason(
    "Should our startup pivot from B2B to B2C?",
    llm_provider,
)

# Inspect which states survived which measurement bases
for step in trace.get_steps_by_type("measurement"):
    basis = step.metadata.get("basis")
    winner = step.metadata.get("selected_state_index")
    print(f"  Basis '{basis}': selected state {winner}")

print(f"\nRobust answer: {trace.final_output}")
```

### PerturbativeExpansion

```python
from openagentflow.reasoning import PerturbativeExpansion

engine = PerturbativeExpansion(max_perturbation_order=4)
trace = await engine.reason(
    "Estimate the time to market for a mobile banking app",
    llm_provider,
)

# Check convergence of the perturbation series
for step in trace.get_steps_by_type("convergence_check"):
    order = step.metadata.get("order")
    converging = step.metadata.get("converging")
    print(f"  Order {order}: {'converging' if converging else 'DIVERGING'}")

print(f"\nFinal estimate: {trace.final_output}")
```

### QuantumTunneling

```python
from openagentflow.reasoning import QuantumTunneling

engine = QuantumTunneling(max_tunnels=3)
trace = await engine.reason(
    "How can we achieve sub-10ms latency on a globally distributed database?",
    llm_provider,
)

# See which barriers were real and which were illusions
for step in trace.get_steps_by_type("tunneling_attempt"):
    barrier = step.metadata.get("barrier")
    penetrated = step.metadata.get("barrier_was_real") == False
    status = "PENETRATED (false constraint)" if penetrated else "REAL BARRIER"
    print(f"  {barrier}: {status}")

print(f"\nSolution: {trace.final_output}")
```

### EntangledThreads

```python
from openagentflow.reasoning import EntangledThreads

engine = EntangledThreads(
    num_threads=4,
    thread_foci=["architecture", "performance", "security", "usability"],
)
trace = await engine.reason(
    "Design a multi-tenant SaaS platform",
    llm_provider,
)

# Inspect entanglement constraints
for step in trace.get_steps_by_type("entanglement_creation"):
    print(f"Entanglement constraints:")
    for c in step.metadata.get("constraints", []):
        print(f"  {c['threads']}: {c['constraint']} ({c['type']})")

# See which constraints survived decoherence
for step in trace.get_steps_by_type("decoherence_check"):
    print(f"\nSurviving entanglements: {step.metadata.get('surviving')}")
    print(f"Broken entanglements: {step.metadata.get('broken')}")

print(f"\nFinal design: {trace.final_output}")
```

### GaugeInvariance

```python
from openagentflow.reasoning import GaugeInvariance

engine = GaugeInvariance(num_gauges=5)
trace = await engine.reason(
    "Is remote work better for productivity?",
    llm_provider,
)

# See which conclusions are invariant vs. artifacts
for step in trace.get_steps_by_type("invariant_extraction"):
    invariants = step.metadata.get("invariants", [])
    artifacts = step.metadata.get("artifacts", [])
    print(f"Gauge-invariant conclusions ({len(invariants)}):")
    for inv in invariants:
        print(f"  [{inv['robustness']:.0%}] {inv['claim']}")
    print(f"\nFraming artifacts ({len(artifacts)}):")
    for art in artifacts:
        print(f"  [only in {art['only_in']}] {art['claim']}")

print(f"\nFrame-independent answer: {trace.final_output}")
```

---

## Appendix: Physics-to-Reasoning Mapping Reference

| Physics Concept | Reasoning Analog | Engine |
|----------------|-----------------|--------|
| Wavefunction `\|psi>` | Set of possible solutions held simultaneously | QuantumSuperposition |
| Probability amplitude `c_i` | Solution viability score | QuantumSuperposition |
| Measurement basis | Evaluation criterion / framing | QuantumSuperposition |
| Wavefunction collapse | Committing to a single answer | QuantumSuperposition |
| Wave phase `phi` | Perspective alignment angle | WaveInterference |
| Constructive interference | Perspectives that reinforce | WaveInterference |
| Destructive interference | Perspectives that contradict | WaveInterference |
| Diffraction | Extending conclusions to adjacent regions | WaveInterference |
| Order parameter | Coherence/consensus measure | PhaseTransition |
| Critical temperature | Constraint threshold for ordering | PhaseTransition |
| Spontaneous symmetry breaking | Committing to a direction | PhaseTransition |
| Shannon entropy `H` | Solution space uncertainty (bits) | EntropicFunnel |
| Information bottleneck | Minimal relevant representation | EntropicFunnel |
| Constraint application | Entropy reduction (learning) | EntropicFunnel |
| Coarse-graining | Abstraction (detail to principle) | RenormalizationGroup |
| RG fixed point | Scale-invariant conclusion | RenormalizationGroup |
| Relevant/irrelevant operators | Strategically important vs. detail-level findings | RenormalizationGroup |
| Gauge transformation | Re-framing the problem | GaugeInvariance |
| Gauge invariant | Frame-independent conclusion | GaugeInvariance |
| Conservation law | Inviolable constraint from symmetry | GaugeInvariance |
| Zeroth-order solution | Simplified base answer | PerturbativeExpansion |
| Perturbation `lambda * V` | Real-world complication | PerturbativeExpansion |
| Coupling constant `lambda` | Complication severity | PerturbativeExpansion |
| Series convergence | Diminishing corrections (answer stabilizing) | PerturbativeExpansion |
| Action functional `S[q(t)]` | Effort-vs-quality cost of a reasoning path | LeastActionPath |
| Euler-Lagrange equation | Local optimality condition for each step | LeastActionPath |
| Stationary point | Path where no local change improves the action | LeastActionPath |
| Potential barrier | Blocking assumption or false constraint | QuantumTunneling |
| Tunneling probability | Likelihood that a barrier is penetrable | QuantumTunneling |
| WKB approximation | Exponential decay with barrier width and height | QuantumTunneling |
| Entangled state | Logically coupled sub-problem conclusions | EntangledThreads |
| Bell correlation | Hard constraint between thread outcomes | EntangledThreads |
| Decoherence | Entanglement constraint breaking under complexity | EntangledThreads |
| Measurement | Forcing a thread to commit to a definite conclusion | EntangledThreads |
