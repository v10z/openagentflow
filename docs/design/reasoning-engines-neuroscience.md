# Neuroscience-Inspired Reasoning Engines: Second Wave Design Document

## Overview

This document specifies 10 new reasoning engines for OpenAgentFlow, each grounded
in a distinct neuroscience mechanism not previously covered by the existing engine
catalog. These engines extend the framework's reasoning capabilities by translating
well-characterized neural computations into structured LLM orchestration patterns.

### Relationship to Existing Engines

OpenAgentFlow already ships with 10 core engines (DialecticalSpiral, DreamWakeCycle,
MetaCognitiveLoop, AdversarialSelfPlay, EvolutionaryThought, FractalRecursion,
ResonanceNetwork, TemporalRecursion, SimulatedAnnealing, SocraticInterrogation) and
a first wave of 10 neuroscience-inspired engines (PredictiveCoding, GlobalWorkspace,
HebbianAssociation, DefaultModeNetwork, HippocampalReplay, AttractorNetwork,
NeuralOscillation, LateralInhibition, BasalGangliaGating, NeuromodulatorySweep).

The 10 engines in this document draw on different neural mechanisms and address
different reasoning failure modes than any prior engine.

### Conventions

- Each engine subclasses `ReasoningEngine` from `openagentflow.reasoning.base`.
- Each engine implements `async def reason(query, llm_provider, ...) -> ReasoningTrace`.
- All LLM calls go through `self._call_llm(provider, messages, trace, system=..., temperature=...)`.
- Steps are recorded via `trace.add_step(ReasoningStep(...))` with typed step names,
  parent linkage, metadata, and scores.
- Engines are stateless across calls; all state lives in the `ReasoningTrace`.
- File names are `snake_case.py`; class names are `CamelCase`.
- Dict-style messages with `system=` keyword (Pattern A) are used for consistency
  with the base class `_call_llm` signature.

---

## Engine 1: CorticalColumnHierarchy

### File: `cortical_column.py`

### Python Class Name

```python
class CorticalColumnHierarchy(ReasoningEngine):
```

### Neuroscience Basis

**Cortical Columns and Hierarchical Predictive Processing (Mountcastle 1957, 1978; Hawkins & Ahmad 2016)**

Vernon Mountcastle's discovery that the neocortex is organized into repeating
functional units -- cortical columns roughly 300-600 micrometers in diameter -- is one of
the foundational insights of systems neuroscience. Each column contains a
stereotyped microcircuit spanning six layers (L1-L6) with distinct cell types and
connectivity patterns:

- **Layer 4**: Receives feedforward (bottom-up) input from thalamus or lower cortical areas.
- **Layers 2/3**: Produce feedforward output to higher areas and lateral connections
  to neighboring columns. These layers carry prediction errors in the predictive
  coding framework (Bastos et al. 2012).
- **Layer 5**: Produces feedback (top-down) output to lower areas and subcortical
  structures. Carries predictions downward.
- **Layer 6**: Modulates thalamic relay and gates information flow.
- **Layer 1**: Receives long-range feedback connections from distant cortical areas,
  providing contextual modulation.

Jeff Hawkins' Thousand Brains Theory (Hawkins 2021) proposes that each cortical
column learns a complete model of an object or concept, and columns "vote"
among themselves to reach a consensus representation. Columns at different
hierarchical levels represent information at different levels of abstraction, with
bidirectional information flow (feedforward data, feedback predictions) between levels.

The critical insight is that the cortex does not just process information
bottom-up; it simultaneously processes top-down expectations at every level, and
the interaction between levels -- not any single level alone -- produces understanding.

### Algorithm

The engine instantiates a hierarchy of "cortical columns" at multiple levels of
abstraction. Each column independently processes the problem at its assigned level.
Columns at adjacent levels exchange feedforward signals (data/evidence rising up)
and feedback signals (predictions/expectations flowing down). The hierarchy
iteratively refines until inter-level predictions align with inter-level evidence
(prediction error minimized at every level).

**Phase 1: HIERARCHY INSTANTIATION**
1. LLM call: "Decompose this problem into a hierarchy of abstraction levels.
   Level 0 is the raw, concrete, surface-level details. Each higher level is
   more abstract and principled. Identify 3-4 levels, describe what each level
   represents, and what aspects of the problem belong at each level."
   - System prompt: "You are a hierarchical analyst decomposing a problem across
     abstraction levels."
   - Temperature: 0.4
   - Records: `hierarchy_plan` step

**Phase 2: COLUMN PROCESSING (parallel across levels)**
2. For each level L (0 to N-1): LLM call with level-specific system prompt:
   "You are cortical column at abstraction level L. Process the problem at YOUR
   level only. Level 0: focus on concrete details, data, specifics. Level 1: focus
   on patterns, categories, relationships. Level 2: focus on principles, theories,
   frameworks. Level 3: focus on meta-principles, worldview implications."
   - Temperature: 0.3 + 0.15 * level (higher levels get more creative latitude)
   - Records: `column_output` step per level

**Phase 3: FEEDFORWARD PASS (bottom-up evidence)**
3. For each level L (1 to N-1): LLM call: "You are at abstraction level L. The level
   below (L-1) has produced the following analysis: [level L-1 output]. What
   evidence, patterns, or data points from this lower-level analysis are relevant
   to your level? Distill the bottom-up signal -- what does the concrete evidence
   tell the abstract framework?"
   - Temperature: 0.3
   - Records: `feedforward_signal` step per level

**Phase 4: FEEDBACK PASS (top-down predictions)**
4. For each level L (N-2 down to 0): LLM call: "You are at abstraction level L.
   The level above (L+1) has the following framework: [level L+1 output]. What
   predictions does this higher-level framework make about what you should find at
   your level? Generate specific expectations: 'If the abstract principle is X, then
   at the concrete level we should see Y.'"
   - Temperature: 0.3
   - Records: `feedback_prediction` step per level

**Phase 5: PREDICTION ERROR COMPUTATION**
5. For each level: LLM call: "Compare the feedforward evidence with the feedback
   predictions at your level. Where do they match? Where do they diverge? For each
   divergence, rate the surprise (0.0-1.0) and specify whether the error suggests
   the lower level is wrong or the higher level is wrong."
   - Temperature: 0.2
   - Records: `prediction_error` step per level

**Phase 6: INTER-COLUMN VOTING (Thousand Brains)**
6. LLM call: "Multiple columns at each level have now processed prediction errors.
   At each level, which interpretation should win? Columns that made accurate
   predictions (low error) should be weighted more heavily. Produce a consensus
   at each level."
   - Records: `column_vote` step

**Phase 7: REPEAT (phases 3-6 for max_iterations or until errors converge)**

**Phase 8: HIERARCHICAL SYNTHESIS**
7. LLM call: "Synthesize the final answer by weaving together the consensus at
   every level of the hierarchy -- from concrete details (L0) through abstract
   principles (L_max). The answer should be vertically coherent: every concrete
   claim should be justified by an abstract principle, and every abstract principle
   should be grounded in concrete evidence."
   - Records: `hierarchical_synthesis` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: N parallel calls (one per level)
- Phase 3: N-1 sequential calls (bottom to top)
- Phase 4: N-1 sequential calls (top to bottom)
- Phase 5: N parallel calls
- Phase 6: 1 call
- Phases 3-6 repeat for `max_iterations`
- Phase 8: 1 call
- Total: 2 + N + max_iterations * (2*(N-1) + N + 1) + 1
- With defaults (N=4, max_iterations=2): 2 + 4 + 2*(6+4+1) + 1 = 29 calls

### When to Use

- Problems that span multiple levels of abstraction (e.g., "How should we design
  our system architecture?" requires reasoning about user needs, component design,
  implementation details, and deployment simultaneously)
- Situations where surface-level details and deep principles must be reconciled
- Any problem where "zooming in" and "zooming out" reveals different things
- Complex engineering problems, policy analysis, scientific research questions
- Tasks where vertical coherence (details consistent with principles) matters

### Step Types

| Step Type | Description |
|-----------|-------------|
| `hierarchy_plan` | Decomposition into abstraction levels |
| `column_output` | Initial processing at a given level |
| `feedforward_signal` | Bottom-up evidence distillation |
| `feedback_prediction` | Top-down prediction generation |
| `prediction_error` | Mismatch between evidence and predictions |
| `column_vote` | Consensus formation across columns |
| `hierarchical_synthesis` | Final vertically-coherent answer |

### Parameters

```python
class CorticalColumnHierarchy(ReasoningEngine):
    def __init__(
        self,
        num_levels: int = 4,              # Abstraction levels (cortical hierarchy depth)
        max_iterations: int = 2,           # Feedforward-feedback-error cycles
        convergence_threshold: float = 0.2, # Mean prediction error below which to stop
        base_temperature: float = 0.3,     # Temperature at lowest level
        temperature_increment: float = 0.15, # Additional temperature per level
        level_labels: list[str] | None = None,
            # Default: ["concrete_details", "patterns_and_relationships",
            #  "principles_and_frameworks", "meta_principles"]
    ) -> None: ...
```

### DAG Structure

```
query
  +-- hierarchy_plan
  +-- column_output_L0
  +-- column_output_L1
  +-- column_output_L2
  +-- column_output_L3
  +-- feedforward_signal_L1 (parent: column_output_L0)
  +-- feedforward_signal_L2 (parent: feedforward_signal_L1)
  +-- feedforward_signal_L3 (parent: feedforward_signal_L2)
  +-- feedback_prediction_L2 (parent: column_output_L3)
  +-- feedback_prediction_L1 (parent: feedback_prediction_L2)
  +-- feedback_prediction_L0 (parent: feedback_prediction_L1)
  +-- prediction_error_L0 ... prediction_error_L3
  +-- column_vote
  +-- [repeat feedforward/feedback/error/vote for iteration 2]
  +-- hierarchical_synthesis
```

---

## Engine 2: PlaceFieldNavigation

### File: `place_field.py`

### Python Class Name

```python
class PlaceFieldNavigation(ReasoningEngine):
```

### Neuroscience Basis

**Hippocampal Place Cells, Grid Cells, and Cognitive Maps (O'Keefe & Nadel 1978; Moser, Kropff & Moser 2008)**

John O'Keefe's discovery that hippocampal CA1 neurons fire selectively when an
animal occupies a specific location in its environment -- "place cells" --
earned the 2014 Nobel Prize. The Mosers subsequently discovered "grid cells"
in the medial entorhinal cortex (MEC) that fire in a regular hexagonal lattice
pattern across the environment, providing a metric coordinate system for navigation.

Together, these systems implement a **cognitive map** -- an internal representation
of the spatial relationships in an environment that supports flexible navigation:
- **Place cells** (CA1, CA3): encode specific locations ("you are here")
- **Grid cells** (MEC): provide a coordinate scaffold at multiple spatial scales
- **Head direction cells** (postsubiculum): encode current heading
- **Border cells** (MEC): encode distances to boundaries
- **Goal cells** (orbitofrontal cortex, subiculum): encode the target location

Tolman (1948) first proposed that animals build cognitive maps of their
environments, enabling them to take novel shortcuts rather than merely repeating
learned routes. The hippocampal system implements this: given a start location
and a goal, the system can compute trajectories through never-before-traversed
paths by consulting the internal map.

Critically, Behrens et al. (2018) showed that this same hippocampal machinery is
repurposed for navigating *abstract conceptual spaces* -- not just physical
environments. The brain represents relationships between concepts as distances
in a "cognitive space," and uses the same place/grid cell machinery to navigate
between abstract ideas.

### Algorithm

The engine maps the problem onto a "solution space" with explicit coordinates
(dimensions representing key tradeoffs or parameters). It identifies the current
position (the initial understanding), the goal position (the desired solution),
and any boundaries/obstacles. It then navigates through the space using
path-planning, exploring intermediate positions, detecting obstacles, and
finding routes around them.

**Phase 1: SPACE CONSTRUCTION**
1. LLM call: "Map this problem onto a navigable conceptual space. Identify 2-4
   key dimensions (tradeoffs, parameters, or axes) that define the solution space.
   For each dimension, describe what low and high values mean. Identify boundaries
   (hard constraints) and obstacles (known failure modes or contradictions)."
   - Records: `space_construction` step

**Phase 2: POSITION FIXING (place cell activation)**
2. LLM call: "Given the current understanding of the problem, where are we in
   the solution space? Plot our current position on each dimension. What is our
   current heading (direction of reasoning)? What is the goal position (the ideal
   solution's coordinates)?"
   - Records: `position_fix` step

**Phase 3: PATH PLANNING (grid cell trajectory)**
3. LLM call: "Plan a trajectory from the current position to the goal position
   through the solution space. Identify 3-5 waypoints -- intermediate positions
   where the reasoning should stop and reassess. For each waypoint, describe what
   that region of the solution space looks like (what combination of dimension
   values it represents). Note any obstacles that require detours."
   - Records: `path_plan` step

**Phase 4: WAYPOINT NAVIGATION (per waypoint)**
4. For each waypoint: LLM call: "Navigate to waypoint [W]. Develop the
   analysis/solution at this specific point in the solution space, where
   [dimension X = value, dimension Y = value, ...]. What does a solution look
   like here? What are its properties? Are there any unexpected obstacles?"
   - Records: `waypoint_analysis` step per waypoint

**Phase 5: OBSTACLE DETECTION AND REPLANNING**
5. After each waypoint, LLM call: "Having reached waypoint [W], are there any
   obstacles (contradictions, infeasibilities, dead ends) that require a detour?
   If so, propose an alternative route to the goal that avoids this obstacle.
   If not, proceed to the next waypoint."
   - Records: `obstacle_check` step (with `obstacle_detected` boolean in metadata)
   - If obstacle detected: LLM call for `replan` step, then continue

**Phase 6: GOAL ARRIVAL AND VERIFICATION**
6. LLM call: "We have navigated through the solution space and arrived at the
   goal region. Verify: does this position actually satisfy the original problem
   requirements? Are we at the goal, or merely nearby? What is the distance
   (residual gap) between our current solution and the ideal?"
   - Records: `goal_verification` step

**Phase 7: SOLUTION EXTRACTION**
7. LLM call: "Extract the final solution from the goal position. Describe the
   solution in terms of the original problem (not the abstract space coordinates).
   Note which dimensions required the most difficult navigation and where
   detours were necessary."
   - Records: `solution_extraction` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: 1 call
- Phase 3: 1 call
- Phase 4: W calls (one per waypoint, sequential -- each depends on obstacle check)
- Phase 5: W calls (one per waypoint, interleaved with Phase 4)
  + up to R replan calls if obstacles detected
- Phase 6: 1 call
- Phase 7: 1 call
- Total: 5 + 2W + R
- With defaults (W=4 waypoints, R=1 replan): 14 calls

### When to Use

- Problems with clear tradeoffs or parameter spaces to navigate
- Optimization problems where the solution is a point in a multi-dimensional space
- Design decisions where multiple competing constraints must be balanced
- Any problem where "mapping the landscape" of options is a natural metaphor
- Strategic planning with multiple axes of concern
- Problems where dead ends and detours are expected

### Step Types

| Step Type | Description |
|-----------|-------------|
| `space_construction` | Defining the dimensions and boundaries of the solution space |
| `position_fix` | Locating current and goal positions |
| `path_plan` | Trajectory planning with waypoints |
| `waypoint_analysis` | Analysis at a specific point in solution space |
| `obstacle_check` | Detecting contradictions or infeasibilities |
| `replan` | Alternative route around an obstacle |
| `goal_verification` | Checking arrival at the target solution |
| `solution_extraction` | Translating abstract coordinates to concrete answer |

### Parameters

```python
class PlaceFieldNavigation(ReasoningEngine):
    def __init__(
        self,
        num_dimensions: int = 3,          # Axes of the solution space
        num_waypoints: int = 4,            # Intermediate stops on the path
        max_replans: int = 2,              # Maximum detours for obstacles
        planning_temperature: float = 0.5,
        navigation_temperature: float = 0.4,
        obstacle_sensitivity: float = 0.6, # How aggressively to detect obstacles (0-1)
    ) -> None: ...
```

### DAG Structure

```
query
  +-- space_construction
  +-- position_fix (parent: space_construction)
  +-- path_plan (parent: position_fix)
  +-- waypoint_analysis_0 (parent: path_plan)
  |     +-- obstacle_check_0
  +-- waypoint_analysis_1 (parent: obstacle_check_0)
  |     +-- obstacle_check_1
  |           +-- replan_0 (if obstacle detected)
  +-- waypoint_analysis_2 (parent: replan_0 or obstacle_check_1)
  |     +-- obstacle_check_2
  +-- waypoint_analysis_3 (parent: obstacle_check_2)
  |     +-- obstacle_check_3
  +-- goal_verification (parent: last obstacle_check)
  +-- solution_extraction (parent: goal_verification)
```

---

## Engine 3: CerebellumErrorCorrection

### File: `cerebellar_correction.py`

### Python Class Name

```python
class CerebellumErrorCorrection(ReasoningEngine):
```

### Neuroscience Basis

**Cerebellar Forward Models and Error-Driven Calibration (Wolpert, Miall & Kawato 1998; Ito 2008)**

The cerebellum, despite containing more than half of the brain's neurons, was long
considered a "mere" motor coordination structure. Modern neuroscience recognizes it
as a universal error-correction and calibration engine. The cerebellum implements
**forward models** (also called internal models): given a planned action and the
current state, it predicts the sensory consequences of that action BEFORE the
action is executed.

The core computation relies on two distinctive cell types:
- **Purkinje cells**: The sole output neurons of the cerebellar cortex. Each Purkinje
  cell receives ~200,000 parallel fiber inputs (from granule cells) encoding context
  and state, and ONE climbing fiber input (from the inferior olive) encoding the
  error signal. This architecture is a biological implementation of supervised
  learning (Albus 1971; Marr 1969).
- **Climbing fibers**: Carry the "teaching signal" -- the difference between the
  predicted outcome and the actual outcome. This error signal triggers long-term
  depression (LTD) at the parallel fiber-Purkinje cell synapse, adjusting the
  forward model.

The computational loop is:
1. Motor cortex generates a motor command (plan)
2. A copy ("efference copy") is sent to the cerebellum
3. The cerebellum predicts the sensory outcome
4. When the actual outcome arrives, the prediction error (via climbing fibers) updates the model
5. Over trials, predictions become more accurate and can be used for rapid online correction

Ramnani (2006) and Stoodley & Schmahmann (2009) showed that cerebellar-prefrontal
loops contribute to cognitive functions including language, working memory, and
planning -- the cerebellum calibrates thought just as it calibrates movement.

### Algorithm

The engine generates an initial solution (the "motor command"), predicts what
problems or objections it will encounter (the "forward model prediction"), then
actually stress-tests the solution, computes the discrepancy between predicted
and actual problems (the "climbing fiber error"), and uses the error to
calibrate both the solution and the prediction model across multiple trials.

**Phase 1: INITIAL COMMAND (motor plan)**
1. LLM call: "Generate a complete solution to this problem. This is your first
   attempt -- produce the best answer you can."
   - Temperature: 0.5
   - Records: `motor_command` step

**Phase 2: FORWARD MODEL PREDICTION**
2. LLM call: "Before testing this solution, predict what will go wrong. What
   objections, edge cases, failure modes, and weaknesses will emerge? For each
   predicted problem, rate your confidence that it will actually arise (0.0-1.0).
   List exactly 5 predicted problems."
   - Temperature: 0.4
   - Records: `forward_prediction` step

**Phase 3: ACTUAL STRESS TEST**
3. LLM call: "Now rigorously stress-test the solution. Test it against edge cases,
   adversarial inputs, scale, and real-world constraints. What problems ACTUALLY
   arise? List every genuine problem you find, whether or not it was predicted."
   - Temperature: 0.3
   - Records: `actual_outcome` step

**Phase 4: CLIMBING FIBER ERROR COMPUTATION**
4. LLM call: "Compare the predicted problems with the actual problems found during
   stress testing. Classify each into: (a) correctly predicted (prediction matched
   reality), (b) false positive (predicted a problem that did not materialize),
   (c) false negative (a real problem that was NOT predicted -- a surprise).
   False negatives are the most important: they reveal blind spots in the forward
   model. Rate the overall calibration quality (0.0-1.0)."
   - Temperature: 0.2
   - Records: `climbing_fiber_error` step with calibration score

**Phase 5: MODEL UPDATE (Purkinje cell LTD)**
5. LLM call: "Update the solution to fix the actual problems found. For false
   negatives (surprise problems), explain WHY the forward model missed them -- what
   assumption or blind spot caused the miss? For false positives, explain why the
   predicted problem did not materialize. Then produce an improved solution."
   - Temperature: 0.4
   - Records: `model_update` step

**Phase 6: REPEAT (phases 2-5 for calibration_rounds)**
- Each subsequent round's forward model should be better calibrated because it
  has learned from prior errors. Track calibration improvement across rounds.

**Phase 7: CALIBRATION REPORT AND FINAL OUTPUT**
6. LLM call: "Present the final calibrated solution. Include a brief calibration
   report: how did the forward model improve across rounds? What were the biggest
   surprise errors in round 1 that were correctly predicted in later rounds?
   What residual blind spots remain?"
   - Records: `calibration_report` step

### LLM Call Pattern

- Phase 1: 1 call
- Phases 2-5: 4 calls per calibration round (sequential)
- Phase 7: 1 call
- Total: 2 + 4 * calibration_rounds
- With defaults (calibration_rounds=3): 14 calls

### When to Use

- Problems where the first solution is likely to have non-obvious flaws
- Engineering and design tasks that need systematic stress-testing
- Any problem where metacognitive calibration (knowing what you don't know) matters
- Quality assurance and review processes
- Situations where iterative refinement based on testing is the natural approach
- Problems where understanding WHY errors were missed is as important as fixing them

### Step Types

| Step Type | Description |
|-----------|-------------|
| `motor_command` | Initial solution (the "action plan") |
| `forward_prediction` | Predicted problems before testing |
| `actual_outcome` | Problems found during stress testing |
| `climbing_fiber_error` | Prediction-vs-reality comparison and calibration |
| `model_update` | Improved solution and updated forward model |
| `calibration_report` | Final solution with calibration analysis |

### Parameters

```python
class CerebellumErrorCorrection(ReasoningEngine):
    def __init__(
        self,
        calibration_rounds: int = 3,       # Predict-test-correct cycles
        num_predictions: int = 5,          # Predicted problems per round
        initial_temperature: float = 0.5,
        test_temperature: float = 0.3,
        error_temperature: float = 0.2,
        update_temperature: float = 0.4,
        calibration_target: float = 0.8,   # Target calibration score to stop early
    ) -> None: ...
```

### DAG Structure

```
query
  +-- motor_command
  +-- forward_prediction_0 (parent: motor_command)
  |     +-- actual_outcome_0
  |           +-- climbing_fiber_error_0
  |                 +-- model_update_0
  |                       +-- forward_prediction_1
  |                             +-- actual_outcome_1
  |                                   +-- climbing_fiber_error_1
  |                                         +-- model_update_1
  |                                               +-- ...
  +-- calibration_report
```

---

## Engine 4: ThalamicRelay

### File: `thalamic_relay.py`

### Python Class Name

```python
class ThalamicRelay(ReasoningEngine):
```

### Neuroscience Basis

**The Thalamus as Information Gateway and Routing Hub (Sherman & Guillery 2006; Halassa & Kastner 2017)**

The thalamus is not merely a passive relay station -- it is the brain's central
information routing hub, determining WHAT information reaches cortical processing
and in WHAT form. Every sensory modality (except olfaction) passes through
thalamic nuclei before reaching cortex, and critically, cortico-thalamo-cortical
loops are how different cortical areas communicate with each other.

Key thalamic nuclei and their functions:
- **Lateral geniculate nucleus (LGN)**: Visual relay -- but 80% of its input comes
  from cortex (feedback), not from the retina. The thalamus is ALREADY filtering.
- **Pulvinar**: The largest thalamic nucleus in humans. Mediates cortico-cortical
  communication, especially attentional routing between visual areas (Saalmann et al. 2012).
- **Mediodorsal nucleus (MD)**: Connects to prefrontal cortex; critical for
  working memory and cognitive flexibility (Mitchell & Chakraborty 2013).
- **Reticular nucleus (TRN)**: A shell of GABAergic inhibitory neurons surrounding
  the thalamus. The TRN implements a "searchlight of attention" (Crick 1984) --
  it selectively inhibits thalamic relay cells to gate which information streams
  reach cortex. This is one of the brain's primary attention mechanisms.

The thalamus operates in two modes:
1. **Tonic mode**: Faithful relay of information (high-fidelity, linear transmission).
   Active during attentive waking. Neurons fire in a regular, sustained pattern.
2. **Burst mode**: Brief, high-frequency bursts that serve as "wake-up calls" or
   salience signals. Active during drowsiness or when a novel/unexpected stimulus
   arrives. Bursts are highly detectable but carry less information content.

Sherman (2016) proposed the distinction between "first-order" relays (carrying new
information to cortex for the first time) and "higher-order" relays (routing
information between cortical areas). Higher-order thalamic relays are how the
prefrontal cortex communicates with temporal or parietal cortex -- the thalamus
is the switchboard.

### Algorithm

The engine implements an explicit information routing system. Multiple analytical
perspectives process the problem, but instead of all perspectives reaching the
final synthesis directly, a "thalamic gating" mechanism selectively routes the
most relevant information streams, filters noise, and identifies which
perspective-to-perspective communications would be most productive. The Reticular
Nucleus acts as an attentional gate, and burst-mode detection flags genuinely
novel insights for priority processing.

**Phase 1: MULTI-STREAM INPUT GENERATION**
1. For each of K input streams (configurable analytical lenses), LLM call in
   parallel: "Analyze this problem through the lens of [stream: e.g., technical
   feasibility, user impact, cost-benefit, risk, precedent/analogy]. Produce a
   focused analysis from this ONE perspective only."
   - Records: `input_stream` step per stream

**Phase 2: RETICULAR NUCLEUS GATING (attention)**
2. LLM call: "You are the thalamic reticular nucleus -- the brain's attentional
   gate. You have received K analytical streams. For each, rate its relevance
   to the core problem (0.0-1.0) and its novelty -- does it contain any truly
   surprising or non-obvious insight (0.0-1.0)? Streams with high relevance
   should pass through in tonic (faithful) mode. Streams with high novelty
   should be flagged as burst signals. Streams with low relevance AND low novelty
   should be gated out (suppressed). Return JSON: [{stream_id, relevance, novelty,
   mode: 'tonic'|'burst'|'suppressed', reason}]"
   - Temperature: 0.2
   - Records: `reticular_gate` step

**Phase 3: TONIC RELAY (high-fidelity transmission)**
3. For each tonic-mode stream: LLM call: "This stream passed attentional gating
   in tonic mode (high relevance, reliable information). Relay its content
   faithfully, but organize and structure it for integration with other streams.
   Preserve all detail."
   - Records: `tonic_relay` step per tonic stream

**Phase 4: BURST DETECTION (novelty/salience relay)**
4. For each burst-mode stream: LLM call: "This stream was flagged for burst-mode
   processing (contains a novel or surprising insight). Extract the core novel
   insight and amplify it -- explain WHY it is surprising and what implications
   it has for the overall problem. This burst signal should interrupt and redirect
   ongoing processing."
   - Records: `burst_relay` step per burst stream

**Phase 5: HIGHER-ORDER ROUTING (cortico-thalamo-cortical)**
5. LLM call: "Given the relayed tonic streams and burst signals, identify which
   pairs of streams should be connected -- where would cross-stream integration
   produce insight that neither stream alone could provide? Route the most productive
   cross-stream connections. For each connection, specify what information should
   flow from stream A to stream B and vice versa."
   - Records: `higher_order_routing` step

**Phase 6: CROSS-STREAM INTEGRATION**
6. For each routed connection: LLM call: "Integrate information from stream A and
   stream B. Stream A contributes: [X]. Stream B contributes: [Y]. What emerges
   from combining these perspectives that neither contained alone?"
   - Records: `cross_stream_integration` step per connection

**Phase 7: UNIFIED SYNTHESIS**
7. LLM call: "Produce the final answer by synthesizing all relayed streams, burst
   insights, and cross-stream integrations. Give priority to burst signals (novel
   insights) and cross-stream integrations (emergent understanding)."
   - Records: `unified_synthesis` step

### LLM Call Pattern

- Phase 1: K parallel calls
- Phase 2: 1 call
- Phase 3: T calls (T = tonic streams, T <= K)
- Phase 4: B calls (B = burst streams, B <= K, T+B <= K)
- Phase 5: 1 call
- Phase 6: C calls (C = routed connections, typically 2-4)
- Phase 7: 1 call
- Total: K + 3 + T + B + C
- With defaults (K=5, ~3 tonic, ~1 burst, ~3 connections): 5+3+3+1+3 = 15 calls

### When to Use

- Problems requiring integration of multiple analytical perspectives
- Situations with information overload where filtering is essential
- Tasks where some inputs are routine and others contain genuinely novel insights
- Cross-functional decision making (routing between engineering, business, user perspectives)
- Any problem where "what to pay attention to" is itself a key question
- Analysis tasks where connections between different viewpoints produce emergent insight

### Step Types

| Step Type | Description |
|-----------|-------------|
| `input_stream` | Raw analysis from one perspective |
| `reticular_gate` | Attentional filtering and mode assignment |
| `tonic_relay` | High-fidelity transmission of relevant information |
| `burst_relay` | Amplified novel/surprising insight |
| `higher_order_routing` | Identifying productive cross-stream connections |
| `cross_stream_integration` | Emergent insight from combining streams |
| `unified_synthesis` | Final answer integrating all relayed information |

### Parameters

```python
class ThalamicRelay(ReasoningEngine):
    def __init__(
        self,
        input_streams: list[str] | None = None,
            # Default: ["technical_feasibility", "user_impact",
            #  "cost_benefit", "risk_assessment", "precedent_analogy"]
        relevance_threshold: float = 0.3,  # Below this, stream is suppressed
        novelty_threshold: float = 0.7,    # Above this, stream gets burst mode
        max_cross_connections: int = 4,    # Maximum routed cross-stream pairs
        stream_temperature: float = 0.5,
        gate_temperature: float = 0.2,
        integration_temperature: float = 0.5,
    ) -> None: ...
```

---

## Engine 5: SynapticConsolidation

### File: `synaptic_consolidation.py`

### Python Class Name

```python
class SynapticConsolidation(ReasoningEngine):
```

### Neuroscience Basis

**Synaptic Consolidation and the Complementary Learning Systems Theory (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016)**

The brain uses two complementary learning systems with fundamentally different
properties:

1. **Hippocampal system (fast learning)**: Rapidly encodes specific episodes and
   experiences with high fidelity. Uses sparse, pattern-separated representations
   to minimize interference between memories. Can learn in one shot. But these
   memories are fragile and context-dependent.

2. **Neocortical system (slow learning)**: Gradually extracts statistical
   regularities and general knowledge across many experiences. Uses overlapping,
   distributed representations that capture shared structure. Learns slowly
   (via interleaved training) to avoid catastrophic interference. But these
   representations are robust and generalizable.

Memory consolidation is the process by which hippocampal fast-learned memories are
gradually transferred to the neocortical slow system. This happens through:
- **Synaptic consolidation** (hours): Molecular cascades (protein synthesis,
  CREB-mediated gene expression) stabilize synaptic changes at the cellular level
  (Dudai 2004).
- **Systems consolidation** (days to years): Repeated hippocampal replay gradually
  trains neocortical networks to store the memory independently (Frankland &
  Bontempi 2005).

The crucial principle is that the fast system captures the specific instance, but
the slow system extracts the generalizable principle. Neither system alone is
sufficient -- you need both and the transfer process between them.

### Algorithm

The engine implements the two-system architecture: a "fast system" generates
specific, context-rich analyses (like hippocampal encoding), and a "slow system"
extracts generalizable principles (like neocortical consolidation). The engine
then interleaves specific instances and general principles, checking for
catastrophic interference (where learning one thing destroys another).

**Phase 1: RAPID ENCODING (hippocampal -- specific instances)**
1. For each of N different "episodes" (framings/instantiations of the problem),
   LLM call: "Analyze this specific instance of the problem: [framing]. Be
   concrete, specific, and detailed. Capture the unique aspects of THIS particular
   case. Do not generalize."
   - Temperature: 0.5
   - Records: `rapid_encoding` step per episode

**Phase 2: PATTERN SEPARATION CHECK**
2. LLM call: "You have N specific analyses. Are any of them too similar (risking
   confusion/interference between memories)? For overlapping analyses, sharpen the
   distinctions -- what makes each truly unique? Return JSON: [{pair: [i,j],
   overlap_score, distinguishing_features}]"
   - Records: `pattern_separation` step

**Phase 3: SLOW EXTRACTION (neocortical -- general principles)**
3. LLM call: "Across all N specific episodes, what general principles, patterns,
   rules, or regularities emerge? These should be abstract enough to apply to novel
   instances you have not seen, but grounded enough to be actionable. Extract 3-5
   generalizable principles. For each, cite which specific episodes support it."
   - Temperature: 0.4
   - Records: `slow_extraction` step

**Phase 4: INTERLEAVED TRAINING (consolidation)**
4. For each general principle: LLM call: "Test this general principle against each
   specific episode. Does the principle hold for all episodes? Where does it break
   down? Does applying the principle to one episode produce an answer that
   contradicts another episode (catastrophic interference)? If so, refine the
   principle to avoid the interference."
   - Records: `consolidation_test` step per principle

**Phase 5: INTERFERENCE DETECTION AND RESOLUTION**
5. LLM call: "Review all consolidation tests. Were there any cases of catastrophic
   interference -- where a general principle that works for episodes A and B fails
   for episode C? For each conflict, propose either: (a) a refined principle that
   accommodates all episodes, or (b) a scope boundary (this principle applies in
   context X but not context Y)."
   - Records: `interference_resolution` step

**Phase 6: CONSOLIDATED SYNTHESIS**
6. LLM call: "Produce the final answer using BOTH the consolidated general
   principles AND the specific episode analyses. The general principles provide
   the framework; the specific episodes provide the evidence and nuance. Mark which
   parts of the answer are 'general knowledge' (robust, transferable) and which
   are 'episodic' (context-specific, possibly fragile)."
   - Records: `consolidated_synthesis` step

### LLM Call Pattern

- Phase 1: N parallel calls
- Phase 2: 1 call
- Phase 3: 1 call
- Phase 4: P calls (P = number of principles, sequential)
- Phase 5: 1 call
- Phase 6: 1 call
- Total: N + 4 + P
- With defaults (N=4 episodes, P=4 principles): 12 calls

### When to Use

- Problems where both specific examples and general principles matter
- Learning from multiple case studies or precedents
- Situations where generalizing too aggressively leads to errors
- Any problem where "it depends on context" is part of the answer
- Tasks requiring both concrete recommendations and abstract frameworks
- Knowledge transfer problems (applying lessons from one domain to another)

### Step Types

| Step Type | Description |
|-----------|-------------|
| `rapid_encoding` | Specific, context-rich analysis of one instance |
| `pattern_separation` | Ensuring distinct episodes do not blur together |
| `slow_extraction` | Abstracting general principles across episodes |
| `consolidation_test` | Testing principles against specific episodes |
| `interference_resolution` | Resolving conflicts between general and specific |
| `consolidated_synthesis` | Final answer with both general and specific knowledge |

### Parameters

```python
class SynapticConsolidation(ReasoningEngine):
    def __init__(
        self,
        num_episodes: int = 4,             # Specific instances to encode
        max_principles: int = 5,           # Maximum general principles to extract
        interference_threshold: float = 0.3, # Above this, consolidation is flagged
        episode_temperature: float = 0.5,
        extraction_temperature: float = 0.4,
        episode_framings: list[str] | None = None,
            # Default: auto-generated diverse framings of the problem
    ) -> None: ...
```

---

## Engine 6: RewardPredictionError

### File: `reward_prediction.py`

### Python Class Name

```python
class RewardPredictionError(ReasoningEngine):
```

### Neuroscience Basis

**Dopaminergic Reward Prediction Error (Schultz, Dayan & Montague 1997; Niv 2009)**

Wolfram Schultz's seminal 1997 paper revealed that midbrain dopamine neurons
(in the ventral tegmental area, VTA, and substantia nigra pars compacta, SNc) do
not simply encode reward -- they encode the *difference* between expected and
received reward, known as the reward prediction error (RPE):

- **Positive RPE** (better than expected): Phasic burst of dopamine firing.
  Signals: "This is unexpectedly good -- strengthen the preceding behavior/thought."
- **Zero RPE** (as expected): Baseline firing. No update needed.
- **Negative RPE** (worse than expected): Pause in dopamine firing (dip below baseline).
  Signals: "This is disappointing -- weaken the preceding behavior/thought."

This is a biological implementation of the temporal difference (TD) learning
algorithm from reinforcement learning (Sutton & Barto 1998). The RPE drives
learning: it is the gradient signal that adjusts the brain's value estimates.

Over learning, the dopaminergic response shifts from the time of reward to the
time of the earliest predictive cue (predictive coding of value). This means the
brain learns to assign credit not just to the final outcome, but backward along
the chain of events that led to it -- the credit assignment problem.

Key nuclei: VTA projects to nucleus accumbens (motivational value) and prefrontal
cortex (working memory and action planning). The RPE signal literally updates the
"expected value" function stored in the striatum (O'Doherty et al. 2004).

### Algorithm

The engine iteratively generates candidate solutions, evaluates their "reward"
(quality), computes the prediction error (how much better or worse than expected),
and uses the RPE to steer subsequent generation. Positive RPE causes the engine
to exploit similar approaches (stay near what worked); negative RPE causes
exploration of different approaches (move away from what failed). The key is that
the engine does not just evaluate solutions -- it learns from the SURPRISE in
the evaluation.

**Phase 1: VALUE PREDICTION**
1. LLM call: "Before attempting to solve this problem, predict how confident you are
   that you can produce a good solution. What is your expected quality level (0.0-1.0)?
   What aspects will be easy vs. hard? Where do you expect to struggle?"
   - Records: `value_prediction` step

**Phase 2: INITIAL ATTEMPT**
2. LLM call: "Now solve the problem. Produce your best attempt."
   - Temperature: 0.5
   - Records: `attempt` step

**Phase 3: REWARD EVALUATION**
3. LLM call: "Evaluate this solution's quality across these dimensions: correctness,
   completeness, elegance, practicality. Rate the overall quality (0.0-1.0). Be
   honest and calibrated -- a score of 0.8 should mean genuinely excellent."
   - Temperature: 0.2
   - Records: `reward_evaluation` step

**Phase 4: PREDICTION ERROR COMPUTATION**
4. Programmatic + LLM call: Compute RPE = actual_reward - predicted_reward.
   Then LLM call: "The prediction error is [RPE]. This means the solution was
   [better/worse] than expected. Analyze: what specifically caused the surprise?
   If positive RPE: what went unexpectedly well? Double down on it. If negative
   RPE: what went wrong that you did not anticipate? What assumption was violated?"
   - Records: `prediction_error` step with RPE value in metadata

**Phase 5: CREDIT ASSIGNMENT (temporal difference)**
5. LLM call: "Trace the chain of reasoning in the solution. Assign credit:
   which specific reasoning steps contributed most to the overall quality (or
   lack thereof)? If RPE was positive, which step was the key insight? If RPE
   was negative, which step was the key mistake? Rate each step's contribution
   (-1.0 to +1.0)."
   - Records: `credit_assignment` step

**Phase 6: POLICY UPDATE AND NEXT ATTEMPT**
6. LLM call: "Based on the prediction error and credit assignment, produce an
   improved solution. If RPE was positive: stay close to the previous approach
   but deepen the successful elements. If RPE was negative: substantially change
   direction, especially at the step that received the most negative credit.
   Also update your value prediction for the next round."
   - Temperature adjusted by RPE: base_temp + abs(RPE) * 0.3 (more surprise = more exploration)
   - Records: `policy_update` step

**Phase 7: REPEAT (phases 3-6 for learning_rounds)**

**Phase 8: FINAL ANSWER**
7. LLM call: "Present the final solution. Reflect on the learning trajectory:
   how did the value predictions change? What was the biggest surprise? What
   was the most important lesson (the highest-magnitude prediction error)?"
   - Records: `final_answer` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: 1 call
- Phases 3-6: 4 calls per learning round
- Phase 8: 1 call
- Total: 3 + 4 * learning_rounds
- With defaults (learning_rounds=3): 15 calls

### When to Use

- Iterative refinement problems where learning from each attempt is valuable
- Problems where knowing what went WRONG is as important as fixing it
- Tasks where calibration of confidence/expectations matters
- Optimization problems with unclear objective landscapes
- Any problem where "was this better or worse than expected?" is informative
- Educational contexts where the reasoning process itself should improve

### Step Types

| Step Type | Description |
|-----------|-------------|
| `value_prediction` | Expected solution quality before attempting |
| `attempt` | A candidate solution |
| `reward_evaluation` | Quality assessment of the attempt |
| `prediction_error` | Surprise signal (actual - expected quality) |
| `credit_assignment` | Which reasoning steps caused the outcome |
| `policy_update` | Improved solution based on RPE and credit |
| `final_answer` | Final solution with learning trajectory reflection |

### Parameters

```python
class RewardPredictionError(ReasoningEngine):
    def __init__(
        self,
        learning_rounds: int = 3,          # Attempt-evaluate-learn cycles
        base_temperature: float = 0.5,     # Temperature before RPE adjustment
        rpe_temperature_sensitivity: float = 0.3, # How much RPE affects temperature
        initial_expected_value: float = 0.6, # Starting value prediction
        exploration_bonus: float = 0.1,    # Added exploration on negative RPE
    ) -> None: ...
```

---

## Engine 7: SomaticMarker

### File: `somatic_marker.py`

### Python Class Name

```python
class SomaticMarker(ReasoningEngine):
```

### Neuroscience Basis

**Somatic Marker Hypothesis (Damasio 1994; Bechara, Damasio & Damasio 2000)**

Antonio Damasio's Somatic Marker Hypothesis proposes that decision-making is
not purely rational -- it is fundamentally guided by emotional "body signals"
(somatic markers) that rapidly bias choices before conscious deliberation occurs.

The neural basis involves:
- **Ventromedial prefrontal cortex (vmPFC)**: Stores associations between situations
  and their emotional outcomes. Patients with vmPFC lesions (like Damasio's famous
  patient Phineas Gage and patient "Elliot") have intact IQ but catastrophically
  impaired decision-making because they cannot access somatic markers.
- **Insula**: Interoceptive cortex that monitors body states (heart rate, gut
  feelings, skin conductance). Translates peripheral physiological signals into
  conscious feelings.
- **Amygdala**: Rapid emotional evaluation of stimuli. Provides the "gut reaction"
  before conscious analysis.
- **Anterior cingulate cortex (ACC)**: Conflict monitoring -- detects when somatic
  markers from different options pull in different directions.

The Iowa Gambling Task (Bechara et al. 1994) demonstrated that healthy subjects
develop anticipatory skin conductance responses (SCRs) to risky decks BEFORE
they can consciously articulate why those decks are bad. The body "knows" before
the mind does.

Somatic markers function as a rapid pre-selection mechanism: by tagging options
with positive or negative emotional valence, they dramatically reduce the search
space for conscious deliberation. Without them, decision-making becomes
paralyzingly slow (as seen in vmPFC patients who take hours to make trivial choices).

### Algorithm

The engine generates candidate options, then evaluates each through TWO parallel
channels: a fast "gut reaction" (somatic marker) channel that provides emotional
valence, and a slow "rational analysis" channel that provides logical evaluation.
When the two channels agree, the decision is straightforward. When they CONFLICT
(the gut says no but the logic says yes, or vice versa), the engine explicitly
investigates the conflict -- because such conflicts often reveal hidden risks or
overlooked opportunities.

**Phase 1: OPTION GENERATION**
1. LLM call: "Generate 4-6 distinct approaches or options for this problem. Each
   should be a genuine alternative, not a variation of the same idea."
   - Temperature: 0.7
   - Records: `option_generation` step

**Phase 2: SOMATIC MARKERS (fast emotional evaluation)**
2. LLM call: "You are the gut reaction system. For each option, give an instant
   emotional response -- do NOT analyze, do NOT reason, just react. What is your
   FEELING about each option? Rate each on two dimensions: valence (-1.0 negative
   to +1.0 positive) and intensity (0.0 = no reaction, 1.0 = strong reaction).
   Trust your instincts. Give one-sentence gut reactions."
   - System prompt: "You are an emotional evaluator. React instinctively. Do not
     reason or analyze -- just feel."
   - Temperature: 0.8
   - Records: `somatic_marker` step with valence/intensity per option

**Phase 3: RATIONAL ANALYSIS (slow deliberate evaluation)**
3. For each option: LLM call: "Rationally analyze this option. Evaluate:
   feasibility, expected outcome, risks, resource requirements. Rate overall
   rational appeal (0.0-1.0). Show your reasoning."
   - Temperature: 0.3
   - Records: `rational_analysis` step per option

**Phase 4: CONFLICT DETECTION (ACC monitoring)**
4. Programmatic + LLM call: Compare somatic markers with rational scores. Identify
   options where the emotional and rational evaluations disagree significantly
   (e.g., positive gut + negative rational = "seductive but risky"; negative gut +
   positive rational = "smart but feels wrong").
   LLM call: "The following options show conflict between gut reaction and rational
   analysis: [list]. For each conflict, investigate: WHY does the gut disagree with
   the head? The gut may be detecting a hidden risk the analysis missed, OR the
   analysis may have overlooked an emotional/motivational factor. Investigate each
   conflict deeply."
   - Records: `conflict_investigation` step

**Phase 5: SOMATIC-RATIONAL INTEGRATION**
5. LLM call: "Integrate the somatic markers, rational analyses, and conflict
   investigations. Rank the options based on BOTH channels. Options that score
   highly on both rational AND emotional dimensions are strong candidates. Options
   with unresolved conflicts should be flagged with caveats. Select the best option
   and explain why, accounting for both rational and emotional evidence."
   - Records: `somatic_rational_integration` step

**Phase 6: DECISION AND ELABORATION**
6. LLM call: "Elaborate the selected option into a complete solution. Describe how
   to implement it, addressing both the rational considerations and the emotional/
   motivational factors that will affect execution. If there were unresolved gut
   warnings, include mitigation strategies."
   - Records: `decision_elaboration` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: 1 call
- Phase 3: N calls (one per option, parallelizable)
- Phase 4: 1 call
- Phase 5: 1 call
- Phase 6: 1 call
- Total: 5 + N
- With defaults (N=5 options): 10 calls

### When to Use

- Decisions where intuition and analysis pull in different directions
- Problems with hidden risks that may not surface in pure rational analysis
- Situations where stakeholder emotions and motivations are important
- Any decision where "it looks good on paper but feels wrong" is a relevant concern
- Product decisions, hiring decisions, strategic pivots
- Ethical dilemmas where emotional dimensions carry legitimate information

### Step Types

| Step Type | Description |
|-----------|-------------|
| `option_generation` | Initial candidate options |
| `somatic_marker` | Fast emotional/gut evaluation |
| `rational_analysis` | Slow deliberate logical evaluation |
| `conflict_investigation` | Deep investigation of gut-vs-head disagreements |
| `somatic_rational_integration` | Combined ranking from both channels |
| `decision_elaboration` | Full solution from the selected option |

### Parameters

```python
class SomaticMarker(ReasoningEngine):
    def __init__(
        self,
        num_options: int = 5,             # Options to generate
        conflict_threshold: float = 0.4,  # Disagreement above this triggers investigation
        gut_temperature: float = 0.8,     # High temperature for instinctive reactions
        rational_temperature: float = 0.3, # Low temperature for deliberate analysis
        integration_temperature: float = 0.4,
        weight_somatic: float = 0.4,      # Weight of somatic markers in final ranking
        weight_rational: float = 0.6,     # Weight of rational analysis in final ranking
    ) -> None: ...
```

---

## Engine 8: WorkingMemoryBuffer

### File: `working_memory.py`

### Python Class Name

```python
class WorkingMemoryBuffer(ReasoningEngine):
```

### Neuroscience Basis

**Baddeley's Working Memory Model and Prefrontal Cortex (Baddeley 2000; D'Esposito & Postle 2015)**

Alan Baddeley's influential model of working memory (WM) proposes multiple
specialized storage buffers coordinated by a central executive:

1. **Phonological loop**: Stores and rehearses verbal/acoustic information. Neural
   basis in left inferior frontal gyrus (Broca's area) and superior temporal gyrus.
   Capacity: ~2 seconds of speech, maintainable by subvocal rehearsal.

2. **Visuospatial sketchpad**: Stores and manipulates visual and spatial information.
   Neural basis in right posterior parietal cortex and premotor areas.

3. **Episodic buffer** (added in 2000): Integrates information from the other
   buffers with long-term memory into coherent "episodes." Neural basis in right
   frontal cortex. This is the binding mechanism that creates unified representations
   from multi-modal information.

4. **Central executive**: The attentional control system that coordinates the
   buffers, selects what to attend to, switches between tasks, and inhibits
   irrelevant information. Neural basis in dorsolateral prefrontal cortex (dlPFC).

Working memory is fundamentally capacity-limited: Cowan (2001) estimated the
capacity at 4 +/- 1 chunks. Miller's (1956) classic "7 +/- 2" referred to chunks
that can be formed by grouping items. The capacity limit forces prioritization --
not everything can be held in mind simultaneously, so the central executive must
continuously decide what to maintain and what to let go.

Curtis & D'Esposito (2003) showed that dlPFC maintains representations via
persistent activity (attractor states), while ventrolateral PFC (vlPFC) selects
and manipulates maintained information. The interaction between maintenance
and manipulation is the core of working memory.

### Algorithm

The engine explicitly manages a capacity-limited working memory buffer. It
decomposes the problem into informational chunks, processes them through
specialized buffers (verbal, structural, episodic), and uses a central executive
to manage what is in the buffer at any time. The critical discipline is that the
buffer has a hard capacity limit -- when new information arrives, old information
must either be consolidated (committed to long-term storage as a summary) or
dropped. This forces aggressive prioritization and distillation.

**Phase 1: CHUNKING**
1. LLM call: "Decompose this problem into distinct informational chunks. Each
   chunk should be a self-contained piece of information or sub-problem. Aim for
   6-10 chunks. For each, provide a brief label and content."
   - Records: `chunking` step

**Phase 2: BUFFER LOADING (initial batch)**
2. Load the first `buffer_capacity` chunks into the working memory buffer.
   For each loaded chunk, LLM call: "Process this chunk through the [verbal /
   structural] buffer. Extract its key content, connections to other loaded
   chunks, and any questions it raises."
   - Records: `buffer_load` step per chunk

**Phase 3: CENTRAL EXECUTIVE PROCESSING**
3. LLM call: "You are the central executive. You currently have [buffer_capacity]
   chunks in working memory: [list]. The remaining unprocessed chunks are: [list].
   Perform the following operations:
   (a) INTEGRATE: What connections or patterns exist across the currently loaded chunks?
   (b) PRIORITIZE: Which currently loaded chunk is LEAST important and can be
       consolidated (compressed into a one-sentence summary and moved to long-term storage)?
   (c) SCHEDULE: Which unloaded chunk should be brought in next?"
   - Temperature: 0.3
   - Records: `executive_decision` step

**Phase 4: CONSOLIDATION AND SWAP**
4. The least-important chunk is compressed into a one-sentence summary (consolidation)
   and removed from the buffer. The next prioritized chunk is loaded.
   LLM call for consolidation: "Compress the following chunk into a single sentence
   that captures its essential contribution to the problem. This sentence will be
   all that remains in long-term storage."
   - Records: `consolidation` step for the removed chunk
   LLM call for new chunk processing (same as Phase 2).
   - Records: `buffer_load` step for the new chunk

**Phase 5: REPEAT (phases 3-4 until all chunks processed)**

**Phase 6: EPISODIC BINDING**
5. LLM call: "You are the episodic buffer. You have the following in working
   memory: [current buffer contents]. And the following consolidated summaries
   in long-term storage: [summaries]. Bind all of this into a single coherent
   episode -- a unified representation of the entire problem and its solution.
   The binding should integrate verbal content, structural relationships, and
   the episodic narrative."
   - Records: `episodic_binding` step

**Phase 7: FINAL OUTPUT**
6. LLM call: "Produce the final answer based on the episodic binding. Structure
   the answer so that the most important information (from the chunks that
   persisted longest in working memory) receives the most emphasis."
   - Records: `final_output` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: C calls (C = buffer_capacity, parallelizable)
- Phases 3-4: 3 calls per swap cycle (executive + consolidation + new load)
- Number of swaps = max(0, total_chunks - buffer_capacity)
- Phase 6: 1 call
- Phase 7: 1 call
- Total: 3 + C + 3 * swaps
- With defaults (total_chunks=8, buffer_capacity=4, 4 swaps): 3 + 4 + 12 = 19 calls

### When to Use

- Problems with too much information to consider simultaneously
- Complex analysis tasks requiring systematic processing of many factors
- Situations where prioritization and triage of information is critical
- Long documents, multi-faceted requirements, or information-dense problems
- Any task where "what to focus on" is itself a key challenge
- Research synthesis with many sources to integrate

### Step Types

| Step Type | Description |
|-----------|-------------|
| `chunking` | Decomposition into informational chunks |
| `buffer_load` | Processing a chunk through a specialized buffer |
| `executive_decision` | Central executive integration and scheduling |
| `consolidation` | Compressing a chunk for long-term storage |
| `episodic_binding` | Integrating all sources into a unified representation |
| `final_output` | Answer structured by information priority |

### Parameters

```python
class WorkingMemoryBuffer(ReasoningEngine):
    def __init__(
        self,
        buffer_capacity: int = 4,          # Max chunks in WM simultaneously (Cowan's K)
        target_chunks: int = 8,            # Total chunks to decompose problem into
        consolidation_temperature: float = 0.2, # Low temperature for compression
        executive_temperature: float = 0.3,
        processing_temperature: float = 0.5,
        prioritization_strategy: str = "relevance",
            # Options: "relevance", "recency", "complexity"
    ) -> None: ...
```

---

## Engine 9: InterhemisphericIntegration

### File: `interhemispheric.py`

### Python Class Name

```python
class InterhemisphericIntegration(ReasoningEngine):
```

### Neuroscience Basis

**Hemispheric Specialization and Callosal Transfer (Gazzaniga 2000; McGilchrist 2009; Corballis 2014)**

The two cerebral hemispheres, while structurally similar, exhibit significant
functional specialization:

**Left hemisphere** (in most right-handed individuals):
- Analytical, sequential processing (Gazzaniga 1998)
- Language production and syntactic processing (Broca's area, Wernicke's area)
- Detail-focused, narrow attention (focused beam; Navon 1977)
- Categorization and labeling
- Logical inference and rule application
- Optimistic interpreter that constructs narratives to explain behavior
  (the "left brain interpreter"; Gazzaniga 2000)

**Right hemisphere**:
- Holistic, parallel, gestalt processing (Robertson & Lamb 1991)
- Prosody, metaphor, pragmatics, humor, and non-literal language (Coulson & Van Petten 2002)
- Broad, vigilant attention (diffuse lantern; Corbetta & Shulman 2002)
- Spatial reasoning and mental rotation
- Novelty detection and processing of unexpected stimuli (Goldberg & Costa 1981)
- Maintaining awareness of context and the "big picture"

The **corpus callosum** (200-300 million axons) connects corresponding regions
of the two hemispheres, enabling information transfer and integration. However,
callosal transfer is not instantaneous -- it introduces a 10-20ms delay, and the
two hemispheres initially process independently before integration occurs.

Split-brain research (Sperry, Gazzaniga) demonstrated that when the corpus callosum
is severed, the two hemispheres can reach different conclusions about the same
stimulus, with the left hemisphere confabulating explanations for the right
hemisphere's actions. This reveals that normal cognition depends on a continuous
dialog between two fundamentally different processing styles.

McGilchrist (2009) argued that the essential difference is not what each hemisphere
does, but HOW it attends to the world: the left hemisphere grasps (narrowly focused,
certain, decontextualized) while the right hemisphere is open to (broadly aware,
uncertain, contextualized). Optimal cognition requires both.

### Algorithm

The engine processes the problem through two fundamentally different cognitive
modes ("hemispheres"), then integrates their outputs through an explicit "callosal
transfer" protocol. The left-mode processes sequentially, analytically, and with
narrow focus. The right-mode processes holistically, contextually, and with broad
awareness. Conflicts between the modes are investigated rather than averaged.

**Phase 1: SPLIT PROCESSING (parallel)**
1a. LEFT MODE: LLM call: "Process this problem analytically and sequentially.
    Break it into parts. Apply rules and logic. Categorize and label. Focus on
    the literal, explicit, and precise. Be thorough but narrow. Produce a
    step-by-step logical analysis."
    - System: "You are a precise, analytical, sequential processor. Focus on details,
      rules, logic, and explicit structure. Do not be vague or metaphorical."
    - Temperature: 0.3
    - Records: `left_hemisphere` step

1b. RIGHT MODE: LLM call: "Process this problem holistically and contextually.
    What is the big picture? What is the overall pattern or gestalt? Consider
    metaphors, analogies, and non-obvious connections. What is the context that
    the literal analysis might miss? What is the emotional/social/relational
    dimension? Trust your intuition about what 'feels' right."
    - System: "You are a holistic, contextual, intuitive processor. Focus on the
      big picture, patterns, metaphors, context, and what is not being said."
    - Temperature: 0.8
    - Records: `right_hemisphere` step

**Phase 2: CALLOSAL TRANSFER (each mode receives the other's output)**
2a. LEFT receives RIGHT: LLM call: "Your analytical/sequential processor has
    produced: [left output]. Your holistic/contextual processor has produced:
    [right output]. From the ANALYTICAL perspective, evaluate the holistic
    analysis: what does it add that the analytical approach missed? What is
    too vague or unsubstantiated?"
    - Records: `callosal_left_receives_right` step

2b. RIGHT receives LEFT: LLM call: "From the HOLISTIC perspective, evaluate the
    analytical analysis: is it missing the forest for the trees? Does the
    step-by-step logic miss an important gestalt? Is anything decontextualized
    in a misleading way?"
    - Records: `callosal_right_receives_left` step

**Phase 3: CONFLICT IDENTIFICATION**
3. LLM call: "Where do the analytical and holistic perspectives agree? Where do
   they conflict? For each conflict, is it because: (a) the analytical view is
   too narrow (missing context), (b) the holistic view is too vague (missing
   specifics), or (c) they are seeing genuinely different aspects of the truth?"
   - Records: `conflict_identification` step

**Phase 4: INTEGRATED SYNTHESIS (commissural binding)**
4. LLM call: "Produce an integrated answer that combines the precision of the
   analytical mode with the contextual richness of the holistic mode. For areas
   of agreement, present the shared conclusion with both analytical evidence and
   contextual framing. For areas of conflict, present both perspectives and
   explain which is more applicable and why. The final answer should be both
   rigorous AND contextualized."
   - Records: `integrated_synthesis` step

**Phase 5: INTERPRETER CHECK (left brain interpreter)**
5. LLM call: "Review the integrated answer. Is it making any of these errors:
   (a) confabulation -- constructing a plausible-sounding narrative that does not
   actually follow from the evidence? (b) false coherence -- smoothing over genuine
   tensions to make the answer feel neater than reality warrants? (c) premature
   closure -- settling on an answer when genuine uncertainty remains? Flag any
   such issues."
   - Records: `interpreter_check` step

**Phase 6: FINAL OUTPUT**
6. LLM call: "Produce the final answer, correcting any confabulation or false
   coherence issues identified. Preserve genuine uncertainty where it exists."
   - Records: `final_output` step

### LLM Call Pattern

- Phase 1: 2 parallel calls
- Phase 2: 2 parallel calls
- Phase 3: 1 call
- Phase 4: 1 call
- Phase 5: 1 call
- Phase 6: 1 call
- Total: 8 calls
- Parallelism: Phases 1 and 2 each have 2 parallel calls

### When to Use

- Problems that benefit from both analytical precision and contextual understanding
- Tasks where "technically correct but missing the point" is a risk
- Situations where intuition and logic need to be reconciled
- Communication tasks (writing, presenting) where both structure and tone matter
- Complex decisions with both quantitative and qualitative dimensions
- Any problem where a purely analytical or purely intuitive approach is insufficient

### Step Types

| Step Type | Description |
|-----------|-------------|
| `left_hemisphere` | Sequential, analytical, detail-focused processing |
| `right_hemisphere` | Holistic, contextual, pattern-focused processing |
| `callosal_left_receives_right` | Analytical evaluation of holistic insights |
| `callosal_right_receives_left` | Holistic evaluation of analytical results |
| `conflict_identification` | Mapping agreements and disagreements |
| `integrated_synthesis` | Combined rigorous + contextualized answer |
| `interpreter_check` | Detecting confabulation and false coherence |
| `final_output` | Corrected final answer |

### Parameters

```python
class InterhemisphericIntegration(ReasoningEngine):
    def __init__(
        self,
        left_temperature: float = 0.3,    # Precise, analytical
        right_temperature: float = 0.8,    # Broad, contextual
        integration_temperature: float = 0.5,
        enable_interpreter_check: bool = True, # Check for confabulation
        left_focus: str = "analytical",    # Override left-mode instruction
        right_focus: str = "holistic",     # Override right-mode instruction
    ) -> None: ...
```

---

## Engine 10: NeuralDarwinism

### File: `neural_darwinism.py`

### Python Class Name

```python
class NeuralDarwinism(ReasoningEngine):
```

### Neuroscience Basis

**Neural Darwinism / Neuronal Group Selection (Edelman 1987; Edelman & Tononi 2000)**

Gerald Edelman's Theory of Neuronal Group Selection (TNGS), also called Neural
Darwinism, proposes that the brain develops through a process analogous to
natural selection operating on populations of neuronal groups (not individual
neurons). Three fundamental mechanisms:

1. **Developmental selection**: During brain development, an enormous initial
   diversity of neural circuits is generated through differential gene expression,
   cell adhesion molecule (CAM) variation, and stochastic wiring. This creates a
   "primary repertoire" of varied neuronal groups -- analogous to the genetic
   diversity in a population of organisms.

2. **Experiential selection**: Through experience, some neuronal groups are
   strengthened (their synapses potentiated) because they are useful for behavior,
   while others are weakened. This is NOT instructive learning (the environment
   does not specify the circuit); instead, the environment SELECTS from
   pre-existing variation -- just as natural selection does not create variation
   but selects from it. This creates a "secondary repertoire" of functionally
   adapted circuits.

3. **Reentrant signaling**: Different neuronal groups communicate through massively
   parallel, reciprocal connections called reentrant circuits. Reentrant signaling
   is not mere feedback -- it is the simultaneous, bidirectional exchange of signals
   across multiple cortical maps. This enables the binding of distributed neuronal
   group activities into coherent cognitive states. Edelman argued that reentrant
   signaling is the neural basis of consciousness itself.

The key insight distinguishing this from standard evolutionary algorithms: the
variation is not random mutation but structured developmental diversity, and the
selection mechanism is not a fitness function but a process of "selective
amplification" through reentrant engagement with the environment and other
neuronal groups. Groups that resonate (reentrant signaling produces coherent
activity) are amplified; groups that do not resonate are suppressed.

Tononi & Edelman (1998) formalized this as "degeneracy" -- the ability of
structurally different neural elements to perform the same function. Degeneracy
provides robustness (many ways to achieve the same result) and is a hallmark of
biological systems.

### Algorithm

The engine creates a "primary repertoire" of structurally diverse reasoning
approaches (not variations of a single approach, but genuinely different
cognitive architectures/strategies for attacking the problem). These are then
exposed to the problem through "experiential selection" -- each approach
processes a challenge scenario and is evaluated not on a fitness function but
on whether it produces coherent results when integrated with other approaches
via "reentrant signaling." Approaches that resonate with multiple others are
amplified; isolated approaches are suppressed.

**Phase 1: DEVELOPMENTAL DIVERSITY (primary repertoire)**
1. LLM call: "Generate 5-7 structurally different reasoning strategies for
   approaching this problem. These should not be variations of one strategy --
   each should use a fundamentally different cognitive architecture: one might use
   decomposition, another analogy, another simulation, another first-principles
   derivation, another case-based reasoning, another constraint satisfaction,
   another narrative construction. For each, describe the strategy and its
   structural logic."
   - Temperature: 0.9 (maximum diversity)
   - Records: `developmental_diversity` step

**Phase 2: INITIAL EXPRESSION (each strategy applied to the problem)**
2. For each strategy: LLM call: "Apply the following reasoning strategy to the
   problem: [strategy description]. Produce a solution using ONLY this approach.
   Do not mix in other approaches -- stay true to the strategy's cognitive
   architecture."
   - Temperature: 0.5
   - Records: `strategy_expression` step per strategy

**Phase 3: REENTRANT SIGNALING (cross-strategy dialog, multiple rounds)**
3. For each pair of strategies (or top-K most promising pairs): LLM call:
   "Strategy A produced: [output A]. Strategy B produced: [output B]. These two
   approaches are now engaged in reentrant signaling. Can they communicate
   productively? Does strategy A's output reinforce, contradict, or complement
   strategy B's? Is there a coherent combined signal? Rate the reentrant resonance
   (0.0 = no coherence, 1.0 = strong mutual reinforcement)."
   - Records: `reentrant_signal` step per pair

**Phase 4: EXPERIENTIAL SELECTION (selective amplification)**
4. Programmatic: Compute each strategy's total reentrant resonance (sum of resonance
   scores from all pairs involving it). Strategies with high total resonance are
   "selected" (amplified); strategies with low resonance are "suppressed."
   LLM call: "The following strategies received the highest reentrant resonance
   (they produced outputs that coherently integrated with the most other strategies):
   [top strategies with resonance scores]. The following strategies were suppressed
   (low resonance -- their outputs were isolated or contradictory): [low strategies].
   Explain why the selected strategies resonate -- what structural feature makes
   them compatible with multiple approaches?"
   - Records: `experiential_selection` step

**Phase 5: DEGENERATE SYNTHESIS**
5. LLM call: "The surviving strategies represent 'degenerate' solutions --
   structurally different approaches that converge on compatible answers. This
   degeneracy is a strength: it means the answer is robust because multiple
   independent reasoning paths support it. Synthesize the final answer by
   weaving together the selected strategies' outputs. Note where different
   strategies provide different angles on the same conclusion (degeneracy) vs.
   where they provide genuinely complementary insights."
   - Records: `degenerate_synthesis` step

**Phase 6: ROBUSTNESS CHECK**
6. LLM call: "The degeneracy-based answer should be robust. Test it: is there any
   critique or edge case that ALL of the selected strategies would fail on? If so,
   is there a suppressed strategy that might have caught this? If a suppressed
   strategy would have helped, reintegrate its relevant insight."
   - Records: `robustness_check` step

### LLM Call Pattern

- Phase 1: 1 call
- Phase 2: S parallel calls (S = number of strategies)
- Phase 3: P calls (P = number of pairs evaluated, typically C(S,2) for small S
  or top-K pairs for large S)
- Phase 4: 1 call
- Phase 5: 1 call
- Phase 6: 1 call
- Total: 4 + S + P
- With defaults (S=5 strategies, P=10 pairs for C(5,2)): 19 calls
- Parallelism: Phase 2 is fully parallel; Phase 3 pairs can be parallelized

### When to Use

- Problems where no single reasoning approach is obviously correct
- Situations requiring robust answers that hold under multiple analytical frameworks
- Tasks where structural diversity of thinking (not just content diversity) is valuable
- Innovation and design where the "meta-question" is "how should we think about this?"
- Problems where convergence from multiple independent methods provides confidence
- Any task where robustness and redundancy matter more than speed

### Step Types

| Step Type | Description |
|-----------|-------------|
| `developmental_diversity` | Generation of structurally diverse reasoning strategies |
| `strategy_expression` | Application of one strategy to the problem |
| `reentrant_signal` | Cross-strategy coherence evaluation |
| `experiential_selection` | Amplification of resonant strategies |
| `degenerate_synthesis` | Answer from multiple convergent reasoning paths |
| `robustness_check` | Testing whether suppressed strategies offer missed insights |

### Parameters

```python
class NeuralDarwinism(ReasoningEngine):
    def __init__(
        self,
        num_strategies: int = 5,          # Size of primary repertoire
        max_reentrant_pairs: int = 10,    # Maximum cross-strategy evaluations
        selection_top_k: int = 3,         # Number of strategies to amplify
        diversity_temperature: float = 0.9,
        expression_temperature: float = 0.5,
        resonance_threshold: float = 0.3,  # Minimum resonance to count as compatible
        enable_robustness_check: bool = True,
    ) -> None: ...
```

---

## Summary Table

| # | Engine | Neuroscience Basis | File | LLM Calls (typical) |
|---|--------|-------------------|------|---------------------|
| 1 | `CorticalColumnHierarchy` | Mountcastle columns, Thousand Brains Theory, hierarchical prediction | `cortical_column.py` | 29 |
| 2 | `PlaceFieldNavigation` | Hippocampal place/grid cells, cognitive maps (O'Keefe & Moser) | `place_field.py` | 14 |
| 3 | `CerebellumErrorCorrection` | Cerebellar forward models, Purkinje/climbing fiber error correction | `cerebellar_correction.py` | 14 |
| 4 | `ThalamicRelay` | Thalamic gating, reticular nucleus, tonic/burst modes | `thalamic_relay.py` | 15 |
| 5 | `SynapticConsolidation` | Complementary Learning Systems, fast/slow memory (McClelland) | `synaptic_consolidation.py` | 12 |
| 6 | `RewardPredictionError` | Dopaminergic RPE, temporal difference learning (Schultz) | `reward_prediction.py` | 15 |
| 7 | `SomaticMarker` | Damasio's somatic marker hypothesis, vmPFC, insula | `somatic_marker.py` | 10 |
| 8 | `WorkingMemoryBuffer` | Baddeley's WM model, capacity limits, central executive | `working_memory.py` | 19 |
| 9 | `InterhemisphericIntegration` | Hemispheric specialization, corpus callosum, left brain interpreter | `interhemispheric.py` | 8 |
| 10 | `NeuralDarwinism` | Edelman's TNGS, neuronal group selection, reentrant signaling | `neural_darwinism.py` | 19 |

---

## Performance Characteristics

| Engine | LLM Calls | Parallelism | Latency | Best For |
|--------|-----------|-------------|---------|----------|
| `CorticalColumnHierarchy` | 29 | Moderate (column processing parallel, passes sequential) | High | Multi-level abstraction problems |
| `PlaceFieldNavigation` | 14 | Low (sequential navigation) | Medium | Tradeoff navigation, optimization |
| `CerebellumErrorCorrection` | 14 | Low (sequential calibration) | Medium | Iterative stress-testing and refinement |
| `ThalamicRelay` | 15 | High (streams and integrations parallel) | Medium | Multi-perspective with filtering |
| `SynapticConsolidation` | 12 | Moderate (episodes parallel) | Medium | Case-based reasoning, generalization |
| `RewardPredictionError` | 15 | Low (sequential learning) | Medium | Iterative improvement with learning |
| `SomaticMarker` | 10 | Moderate (rational analyses parallel) | Low-Medium | Decisions with emotional dimensions |
| `WorkingMemoryBuffer` | 19 | Low (sequential buffer management) | Medium-High | Information-dense problems |
| `InterhemisphericIntegration` | 8 | High (hemispheres parallel) | Low | Analysis + context balance |
| `NeuralDarwinism` | 19 | High (strategies and pairs parallel) | Medium | Robust, multi-method validation |

---

## Decision Guide: When to Use Which Engine

```
Problem Type                              Recommended Engine
-----------                              -----------------
"Spans multiple abstraction levels"   ->  CorticalColumnHierarchy
"Navigate a tradeoff space"           ->  PlaceFieldNavigation
"Need iterative stress-testing"       ->  CerebellumErrorCorrection
"Too many perspectives, need filter"  ->  ThalamicRelay
"Learn from specific cases"           ->  SynapticConsolidation
"Improve through surprise signals"    ->  RewardPredictionError
"Gut vs. head disagreement"           ->  SomaticMarker
"Information overload, need triage"   ->  WorkingMemoryBuffer
"Need both precision and context"     ->  InterhemisphericIntegration
"Want multi-method robustness"        ->  NeuralDarwinism
```

---

## Quality vs. Speed Tradeoff

```
Quality
  ^
  |  *CorticalColumn
  |       *NeuralDarwinism
  |    *WorkingMemory    *CerebellumError
  |        *RPE     *ThalamicRelay
  |   *SynapticConsolidation
  |          *PlaceField
  |     *SomaticMarker   *Interhemispheric
  |
  +-------------------------------------------> Speed
  Fast                                    Slow
```

---

## Implementation Notes

### Registration

All 10 new engines must be added to `openagentflow/reasoning/__init__.py`:

```python
from openagentflow.reasoning.cortical_column import CorticalColumnHierarchy
from openagentflow.reasoning.place_field import PlaceFieldNavigation
from openagentflow.reasoning.cerebellar_correction import CerebellumErrorCorrection
from openagentflow.reasoning.thalamic_relay import ThalamicRelay
from openagentflow.reasoning.synaptic_consolidation import SynapticConsolidation
from openagentflow.reasoning.reward_prediction import RewardPredictionError
from openagentflow.reasoning.somatic_marker import SomaticMarker
from openagentflow.reasoning.working_memory import WorkingMemoryBuffer
from openagentflow.reasoning.interhemispheric import InterhemisphericIntegration
from openagentflow.reasoning.neural_darwinism import NeuralDarwinism

__all__ = [
    # ... existing engines ...
    "CerebellumErrorCorrection",
    "CorticalColumnHierarchy",
    "InterhemisphericIntegration",
    "NeuralDarwinism",
    "PlaceFieldNavigation",
    "RewardPredictionError",
    "SomaticMarker",
    "SynapticConsolidation",
    "ThalamicRelay",
    "WorkingMemoryBuffer",
]
```

### Parallelism Opportunities

Several engines have embarrassingly parallel phases that should use `asyncio.gather()`:

| Engine | Parallel Phases |
|--------|----------------|
| `CorticalColumnHierarchy` | Phase 2 (column processing), Phase 5 (error computation) |
| `ThalamicRelay` | Phase 1 (input streams), Phase 6 (cross-stream integrations) |
| `SynapticConsolidation` | Phase 1 (episode encoding) |
| `SomaticMarker` | Phase 3 (rational analyses per option) |
| `InterhemisphericIntegration` | Phase 1 (both hemispheres), Phase 2 (both callosal transfers) |
| `NeuralDarwinism` | Phase 2 (strategy expressions), Phase 3 (reentrant pairs) |

### Shared Utilities

These engines can reuse existing parsing utilities from the codebase:

- `_parse_json_safe(text, default)` from `MetaCognitiveLoop`
- `_parse_json_list(raw, expected)` from `ResonanceNetwork`
- `_parse_score(text, label)` from `DreamWakeCycle`
- `_parse_numbered_list(text, expected)` from `DreamWakeCycle`

New shared utilities to consider adding to the `ReasoningEngine` base class:

- `_parse_json_array_of_objects(text, expected_keys)` -- for structured outputs with
  multiple fields per item (used by ThalamicRelay, SynapticConsolidation, SomaticMarker)
- `_compute_pairwise_scores(items, evaluator_fn)` -- for engines that evaluate all
  pairs (NeuralDarwinism reentrant signaling, CorticalColumnHierarchy cross-level errors)

### Token Budget Management

Engines with many LLM calls (CorticalColumnHierarchy=29, WorkingMemoryBuffer=19,
NeuralDarwinism=19) should implement graceful degradation:

- Track cumulative tokens via `trace.total_tokens`
- Accept an optional `max_tokens_budget` parameter
- For CorticalColumnHierarchy: reduce number of levels or iterations
- For WorkingMemoryBuffer: increase buffer capacity (fewer swaps) or reduce chunks
- For NeuralDarwinism: reduce strategies or limit reentrant pair evaluations

### References

**Engine 1 -- CorticalColumnHierarchy:**
- Mountcastle VB (1957). Modality and topographic properties of single neurons of cat's somatic sensory cortex. J Neurophysiol.
- Hawkins J, Ahmad S (2016). Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex. Front Neural Circuits.
- Hawkins J (2021). A Thousand Brains: A New Theory of Intelligence. Basic Books.
- Bastos AM et al. (2012). Canonical Microcircuits for Predictive Coding. Neuron.

**Engine 2 -- PlaceFieldNavigation:**
- O'Keefe J, Nadel L (1978). The Hippocampus as a Cognitive Map. Oxford University Press.
- Moser EI, Kropff E, Moser MB (2008). Place Cells, Grid Cells, and the Brain's Spatial Representation System. Annu Rev Neurosci.
- Behrens TEJ et al. (2018). What Is a Cognitive Map? Organizing Knowledge for Flexible Behavior. Neuron.

**Engine 3 -- CerebellumErrorCorrection:**
- Wolpert DM, Miall RC, Kawato M (1998). Internal models in the cerebellum. Trends Cogn Sci.
- Ito M (2008). Control of mental activities by internal models in the cerebellum. Nat Rev Neurosci.
- Marr D (1969). A theory of cerebellar cortex. J Physiol.
- Albus JS (1971). A theory of cerebellar function. Math Biosci.

**Engine 4 -- ThalamicRelay:**
- Sherman SM, Guillery RW (2006). Exploring the Thalamus and Its Role in Cortical Function. MIT Press.
- Halassa MM, Kastner S (2017). Thalamic functions in distributed cognitive control. Nat Neurosci.
- Crick F (1984). Function of the thalamic reticular complex: the searchlight hypothesis. PNAS.
- Saalmann YB et al. (2012). The pulvinar regulates information transmission between cortical areas based on attention demands. Science.

**Engine 5 -- SynapticConsolidation:**
- McClelland JL, McNaughton BL, O'Reilly RC (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychol Rev.
- Kumaran D, Hassabis D, McClelland JL (2016). What Learning Systems Do Intelligent Agents Need? Complementary Learning Systems Theory Updated. Trends Cogn Sci.
- Dudai Y (2004). The neurobiology of consolidations, or, how stable is the engram? Annu Rev Psychol.

**Engine 6 -- RewardPredictionError:**
- Schultz W, Dayan P, Montague PR (1997). A neural substrate of prediction and reward. Science.
- Niv Y (2009). Reinforcement learning in the brain. J Math Psychol.
- O'Doherty JP et al. (2004). Dissociable roles of ventral and dorsal striatum in instrumental conditioning. Science.

**Engine 7 -- SomaticMarker:**
- Damasio AR (1994). Descartes' Error: Emotion, Reason, and the Human Brain. Putnam.
- Bechara A, Damasio H, Damasio AR (2000). Emotion, decision making and the orbitofrontal cortex. Cereb Cortex.
- Bechara A et al. (1994). Insensitivity to future consequences following damage to human prefrontal cortex. Cognition.

**Engine 8 -- WorkingMemoryBuffer:**
- Baddeley A (2000). The episodic buffer: a new component of working memory? Trends Cogn Sci.
- D'Esposito M, Postle BR (2015). The cognitive neuroscience of working memory. Annu Rev Psychol.
- Cowan N (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behav Brain Sci.
- Miller GA (1956). The magical number seven, plus or minus two. Psychol Rev.

**Engine 9 -- InterhemisphericIntegration:**
- Gazzaniga MS (2000). Cerebral specialization and interhemispheric communication: does the corpus callosum enable the human condition? Brain.
- McGilchrist I (2009). The Master and His Emissary: The Divided Brain and the Making of the Western World. Yale University Press.
- Corballis MC (2014). Left Brain, Right Brain: Facts and Fantasies. PLoS Biol.

**Engine 10 -- NeuralDarwinism:**
- Edelman GM (1987). Neural Darwinism: The Theory of Neuronal Group Selection. Basic Books.
- Edelman GM, Tononi G (2000). A Universe of Consciousness: How Matter Becomes Imagination. Basic Books.
- Tononi G, Edelman GM (1998). Consciousness and complexity. Science.
