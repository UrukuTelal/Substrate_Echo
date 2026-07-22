# Substrate_Echo Implementation Plan

## Overview

This plan details the scaffolding and phased implementation of Substrate_Echo — an ontological world-state architecture for embodied AI.

## Dependency Map: Substrate_Echo ↔ BCFVT Math

```
Phase A (NOW — no math dependency)     Phase B (NEEDS BCFVT math)
─────────────────────────────────      ─────────────────────────────────
Dynamics ODE solvers          ──┐      BCFVT-00: Conservation Framework
State Transition Manager      ──┤      BCFVT-02: Field Theory Core
Diffusion tensor              ──┤      BCFVT-05: 16D Pillar Coupling
Enhanced attractor memory     ──┤      BCFVT-01: Dynamic Metric
Cognitive agent refinement    ──┤      BCFVT-03: Vortex Dynamics
Integration bridges (live)    ──┤      BCFVT-04: Topology Transitions
Conservation hooks (stubs)    ──┼────→ BCFVT-06: CUDA Kernels
Metric interface (contract)   ──┘
Topology events (definitions) ──┘
Visualization layer           ──┘
Tests                          ──┘
```

**Key insight:** Substrate_Echo's `OntologicalField.evolve()` is currently a placeholder with basic attractor/repulsor dynamics. BCFVT-02 provides the real field equation (Ginzburg-Landau). Phase A builds the numerical infrastructure; Phase B drops in the real physics.

## Phase S1: Core Scaffolding (COMPLETE ✓)

Create all directory structure, `__init__.py` files, and interface definitions (types, dataclasses, abstract methods). No implementations — just contracts.

### Files Created ✓

| File | Purpose | Status |
|------|---------|--------|
| `substrate_echo/models/world_object.py` | PhysicalState, RelationalState, WorldObject, SpatialGraph | ✓ |
| `substrate_echo/models/experience.py` | Experience (8 types), factory methods | ✓ |
| `substrate_echo/models/memory_trace.py` | MemoryTrace (5 trace types) | ✓ |
| `substrate_echo/models/action.py` | Action, ActionType, MotorCommand | ✓ |
| `substrate_echo/core/spatial_world.py` | SpatialWorldModel with grid indexing, region queries | ✓ |
| `substrate_echo/core/ontological_field.py` | OntologicalField, Attractor, Repulsor | ✓ |
| `substrate_echo/core/attractor_memory.py` | AttractorMemory (encode→recall→consolidate) | ✓ |
| `substrate_echo/core/cognitive_agents.py` | 5 agents + AgentEcology orchestrator | ✓ |
| `substrate_echo/core/embodiment_bridge.py` | EmbodimentBridge (AR/robotics) | ✓ |
| `substrate_echo/integration/psv_bridge.py` | BlochPSV ↔ numpy, VNES-Lab | ✓ |
| `substrate_echo/integration/void_bridge.py` | VacuumDefect ↔ WorldObject | ✓ |
| `substrate_echo/integration/engine_bridge.py` | Engine Entity ↔ Substrate_Echo | ✓ |
| `substrate_echo/integration/council_bridge.py` | Council consensus → action | ✓ |
| `tests/test_*.py` | 28 tests, all passing | ✓ |
| `requirements.txt` + `pyproject.toml` | Project config | ✓ |

## Phase S2: Dynamics Module (NO MATH DEPENDENCY — DO NOW)

Numerical methods that work independently of BCFVT physics. These provide the infrastructure that BCFVT-02 will drop into.

### S2.1: ODE Solver
- Runge-Kutta 4th order for non-stiff terms
- Semi-implicit Crank-Nicolson for stiff terms (Ginzburg-Landau)
- Adaptive time stepping with error control
- File: `dynamics/field_evolution.py` → `FieldEvolver` class

### S2.2: Diffusion Tensor
- 16×16 inter-pillar coupling matrix
- Default: identity (pillars independent)
- Learned: correlations between pillars strengthen coupling
- File: `dynamics/diffusion.py` → `DiffusionTensor` class

### S2.3: Attractor Dynamics
- Formation rate: repeated stimulation → crystallization
- Decay: exponential forgetting for unused attractors
- Strengthening: frequent access increases strength
- Merging: similar attractors combine when overlap > threshold
- File: `dynamics/attractor_dynamics.py` → `AttractorDynamics` class

### S2.3: State Transition Manager
- `StateTransition` — the atomic unit of change (source, target, cause, energy, info)
- `TransitionCause` — FIELD_CHANGE, AGENT_ACTION, TOPOLOGY_EVENT, MEMORY_UPDATE, SENSOR_INPUT, EXTERNAL_FORCING, CONSERVATION_CORRECTION
- `StateTransitionManager` — validates, distributes, records all state changes
- `TransitionConstraint` — conservation laws that transitions must satisfy
- `TransitionCallback` — listeners that react to transitions
- File: `dynamics/state_transitions.py` → `StateTransitionManager` class

## Phase S3: Enhanced Memory (NO MATH DEPENDENCY — DO NOW)

Extend attractor memory with consolidation, identity formation, and temporal dynamics.

### S3.1: Consolidation Pipeline
- Periodic sweep: strengthen frequently-accessed, decay rarely-accessed
- Merging: similar attractors combine (cosine similarity > 0.9)
- Pruning: remove attractors below strength threshold
- Integration with BCFVT conservation: total "memory mass" conserved

### S3.2: Identity Formation
- Persistent attractor cluster = identity pattern
- Identity coherence metric: how aligned are core attractors
- Identity shift detection: significant reorganization events
- Connection to BCFVT topological charge: identity as invariant

## Phase S4: Agent Ecology Refinement (NO MATH DEPENDENCY — DO NOW)

Improve activation logic, consensus mechanisms, and inter-agent communication.

### S4.1: Dynamic Activation
- Threshold adapts based on task complexity
- Agent specialization: learn which pillars each agent handles best
- Resource allocation: active agents consume "energy" from shared pool

### S4.2: Consensus Mechanisms
- Weighted voting: agents vote with confidence weights
- Deliberation: agents can negotiate before final decision
- Dissent handling: minority reports for unusual situations

### S4.3: Inter-Agent Communication
- Message passing between agents
- Information propagation through ecology
- Collective intelligence emergence

## Phase S5: Integration Bridges Live (NO MATH DEPENDENCY — DO NOW)

Wire bridges to actual infrastructure: BlochPSV, void_svt, Engine entities, Council.

### S5.1: PSV Bridge Live
- Read/write actual BlochPSV state from DeveloperConsole
- Map VNES-Lab PillarState to Substrate_Echo format
- Real-time synchronization with entity state

### S5.2: Void Bridge Live
- Read VacuumDefect properties from void_svt.py
- Map defect dynamics to spatial world objects
- Integrate SVT physics with ontological field

### S5.3: Engine Bridge Live
- Read Entity PSV from Van_Nueman_Engine
- Map engine cognition to cognitive agents
- Synchronize with engine tick loop

### S5.4: Council Bridge Live
- Submit decisions to Pillar Council for validation
- Map council votes to cognitive responses
- Use 16-pillar review for action approval

## Phase S6: Conservation Hooks (STUB NOW — WIRE TO BCFVT-00 LATER)

Define conservation check interfaces. BCFVT-00 provides the real invariants.

### S6.1: Invariant Definitions
- Norm conservation: d/dt ∫|ℱ|² dx = 0
- Energy functional: E[ℱ] = ∫(½|∇ℱ|² + V(ℱ)) dx
- Topological charge: Q = ∫ ℱ·(∇×ℱ) dx
- Information: Fisher metric bounds

### S6.2: Check Functions
- `check_norm_drift(field, tolerance=1e-10)` → bool
- `check_energy_monotonic(field, dt)` → bool
- `check_charge_quantization(field)` → bool
- `check_fisher_bound(field, params)` → bool

## Phase S7: Metric Interface (CONTRACT NOW — IMPLEMENT WITH BCFVT-01)

Define the API for dynamic metric evolution. BCFVT-01 provides Ricci flow.

### S7.1: Metric API
- `compute_ricci_tensor(metric, field)` → R_ij
- `compute_stress_energy(field, metric)` → T_ij
- `evolve_metric(metric, R_ij, T_ij, dt)` → new_metric
- `check_cfl_condition(metric, dt)` → bool

### S7.2: Integration Points
- Metric couples to spatial world model distances
- Metric affects attractor basin shapes
- Metric evolution constrained by conservation laws

## Phase S8: Topology Event System (DEFINITIONS NOW — RATES FROM BCFVT-04)

Define topology transition events. BCFVT-04 provides transition rates.

### S8.1: Event Types
- `TopologyEvent`: creation, annihilation, merging, splitting
- `EventRate`: Γ = A·exp(-S_E/ℏ) (Euclidean action)
- `EnergyBarrier`: computation for tunneling events

### S8.2: Event Queue
- Priority queue of pending topology events
- Energy conservation check after each event
- Rollback mechanism if energy increases beyond tolerance

## Phase S9: Visualization Layer (NO MATH DEPENDENCY — DO NOW)

Render field state, world model, and agent activity.

### S9.1: Field Visualization
- 16D state as color-coded bar chart
- Attractor positions in projected 2D/3D
- Diffusion tensor as heatmap

### S9.2: World Model Visualization
- Spatial graph with object nodes and relationship edges
- Object permanence indicators
- Trajectory prediction lines

### S9.3: Agent Activity Visualization
- Agent activation states (active/dormant)
- Consensus decision display
- Inter-agent communication flow

## Phase S10: BCFVT Integration (NEEDS BCFVT MATH)

Wire BCFVT tasks to Substrate_Echo infrastructure.

### S10.1: BCFVT-00 Conservation → Conservation Hooks
- Replace stubs with real invariant checks
- Enforce conservation during field evolution
- Alert on conservation violations

### S10.2: BCFVT-02 Field Theory → OntologicalField.evolve()
- Replace placeholder with Ginzburg-Landau dynamics
- D(ℱ) = D∇²ℱ (diffusion)
- I(ℱ) = -∂V/∂ℱ* (interaction)
- T(ℱ) = -γℱ + η (dissipation + noise)

### S10.3: BCFVT-05 Pillar Coupling → PSV Mapping
- Map BCFVT field components to 16D pillars
- WHT basis modes as pillar decomposition
- Back-reaction: pillar evolution affects field

### S10.4: BCFVT-01 Metric → Metric Interface
- Implement Ricci flow on discrete lattice
- Metric evolution couples to field dynamics
- CFL stability enforcement

### S10.5: BCFVT-03 Vortex Dynamics → Attractor Memories
- Vortex identification via winding number
- Vortex interactions as attractor dynamics
- Kelvin wave excitations on vortex filaments

### S10.6: BCFVT-04 Topology → Topology Events
- Implement transition rates from Euclidean action
- Foam node creation/annihilation
- Anti-runaway: rate limiter, energy conservation checks

### S10.7: BCFVT-06 CUDA → GPU Acceleration
- Field evolution kernel (semi-implicit step)
- Ricci tensor computation kernel
- Vortex identification kernel

## Phase S11: Experiments (NEEDS BOTH PHASES A + B)

### Experiment 1: Spatial Understanding
- Measure: object persistence, environmental mapping, contextual reasoning
- Compare: with/without relational state on objects
- Metric: accuracy of predictions, consistency of world model

### Experiment 2: Memory Continuity
- Measure: long-term consistency, personal context retention, adaptive behavior
- Compare: attractor memory vs. database memory
- Metric: recall accuracy, forgetting curve, identity coherence

### Experiment 3: Agent Coordination
- Measure: task completion, adaptability, error recovery, reasoning efficiency
- Compare: single-agent vs. multi-agent ecology
- Metric: response time, solution quality, resource usage

### Experiment 4: Embodied Transfer
- Measure: same world model across virtual → AR → robotics
- Compare: transfer fidelity, adaptation time
- Metric: performance retention, calibration requirements

### S2.1: WorldObject
- PhysicalState: position, orientation, dimensions, velocity, affordances
- RelationalState: familiarity, importance, history, associations, context_psv
- WorldObject combines both, has update methods

### S2.2: SpatialGraph
- Adjacency representation of object relationships
- Edge types: spatial_proximity, functional_association, causal_link, semantic_relation
- Graph operations: neighbors, shortest_path, cluster_detection

### S2.3: SpatialWorldModel
- Object registry (dict[str, WorldObject])
- Spatial indexing (grid or kd-tree for region queries)
- update_from_perception() — integrate sensor data
- predict_trajectory() — physics + intent prediction
- Integration with void_svt.py VacuumDefect as spatial objects

## Phase S3: Ontological Field Dynamics

### S3.1: Field Evolution
- State: BlochPSV + field_grid (discretized over state space)
- Evolution: ∂ψ/∂t = D·∇²ψ + F(ψ) + η(t)
- D = 16x16 diffusion tensor (inter-pillar coupling strengths)
- F(ψ) = local dynamics (attractor potential, repulsor repulsion)
- η(t) = sensory noise/input from perception

### S3.2: Attractor/Repulsor Dynamics
- Attractor: stable point in state space with basin of attraction
- Repulsor: unstable point that pushes state away
- Formation: repeated stimulation → attractor crystallization
- Decay: unused attractors weaken over time (exponential decay)
- Strengthening: frequent access increases attractor strength

### S3.3: Diffusion Tensor
- Default: identity (pillars independent)
- Learned: correlations between pillars strengthen coupling
- Example: Awareness↑ often co-occurs with Depth↑ → strong off-diagonal coupling

## Phase S4: Attractor Memory

### S4.1: Experience → Attractor Pipeline
1. Experience arrives (sensory + context + action + result)
2. Experience is encoded as a BlochPSV pattern
3. Pattern is injected into ontological field
4. If pattern matches existing attractor → strengthen it
5. If pattern is novel → form new attractor
6. Attractor stores: center, basin_width, strength, access_count, formed_at, events

### S4.2: Recall by Cue
1. Cue arrives as BlochPSV (partial state)
2. Compute distance from cue to all attractors
3. Return k nearest attractors above strength threshold
4. Each attractor unfolds into full memory trace

### S4.3: Consolidation
- Periodic consolidation: strengthen frequently-accessed, decay rarely-accessed
- Merging: similar attractors merge (overlap > threshold)
- Identity: the persistent attractor cluster = the agent's identity pattern

## Phase S5: Cognitive Agents

### S5.1: Agent Base Class
- pillar_affinity: which pillars this agent specializes in
- activation_threshold: PSV similarity threshold to activate
- state: agent's own BlochPSV (specialized sub-state)
- evaluate(context) → AgentResponse

### S5.2: Five Specialized Agents
| Agent | Pillar Affinity | Role |
|-------|----------------|------|
| PerceptionAgent | Awareness(0), Presence(8) | Process sensor data → world model |
| MemoryAgent | Memory(10), Depth(15) | Encode/recall attractor memories |
| PlanningAgent | Willpower(1), Force(2), Influence(3) | Generate action plans |
| CreativityAgent | Distortion(13), Flux(14), Depth(15) | Combine attractors creatively |
| EnvironmentAgent | Relation(7), Resistance(4), Awareness(0) | Monitor world changes |

### S5.3: Agent Ecology
- Root agent orchestrates: receives world state, activates relevant agents, collects responses
- Activation: based on task type and agent pillar affinity
- Consensus: weighted vote (like Pillar Council but cognitive)
- Deactivation: agents go dormant when not needed (resource conservation)

## Phase S6: Embodiment Bridge

### S6.1: Sensor Processing
- Raw sensor data → PerceptionAgent → WorldObject updates
- Filter, normalize, fuse multi-modal sensors
- Temporal integration (smooth over time)

### S6.2: Action Generation
- PlanningAgent decision → MotorCommand
- MotorCommand: joint angles, velocities, gripper states
- Safety checks: collision avoidance, joint limits

### S6.3: AR Integration
- Sync SpatialWorldModel with AR environment mapping
- Render ontological field as visible overlay
- User interaction → experience → memory

### S6.4: Robotics Transfer
- Export world model configuration for robotic platform
- Map motor commands to robot-specific actuators
- Sensor abstraction layer for different robot hardware

## Phase S7: Experiments

### Experiment 1: Spatial Understanding
- Measure: object persistence, environmental mapping, contextual reasoning
- Compare: with/without relational state on objects
- Metric: accuracy of predictions, consistency of world model

### Experiment 2: Memory Continuity
- Measure: long-term consistency, personal context retention, adaptive behavior
- Compare: attractor memory vs. database memory
- Metric: recall accuracy, forgetting curve, identity coherence

### Experiment 3: Agent Coordination
- Measure: task completion, adaptability, error recovery, reasoning efficiency
- Compare: single-agent vs. multi-agent ecology
- Metric: response time, solution quality, resource usage

### Experiment 4: Embodied Transfer
- Measure: same world model across virtual → AR → robotics
- Compare: transfer fidelity, adaptation time
- Metric: performance retention, calibration requirements

## Phase S8: External Agent Integration

Interface for external AI agents (Moltbook, etc.) to feed information into the
cognitive substrate. External agents are modeled as foreign dynamical systems
producing perturbations, not as knowledge sources.

### Key Findings (EXP-EXT-001, EXP-EXT-001B)
- WHT is distance-preserving (orthogonal, H^T H = I): does NOT provide filtering
- Quantization provides invariance (3.6x compression of same-concept distances)
- Truncation preserves contrast well
- WHT+truncation is WORSE than raw truncation (spreads energy uniformly)
- **The actual filter is the evaluation manifold, not the spectral transform**
- Pipeline: feature selection → projection → quantization → evaluation
- Rename: "WHT Semantic Washing Layer" → **Spectral Normalization Layer**
  (decorrelates dimensions; the bottleneck stack does the filtering)

### Architecture

```
External Interaction
        |
        v
  InteractionEncoder (dual-path)
        |
        +--- SemanticFeatures (16D)  ──> Spectral Normalization (WHT)
        |
        +--- RelationalFeatures (16D) ──> Spectral Normalization (WHT)
        |
        v
  Combined 32D → PSVBridge → existing latent space
        |
        v
  ForeignEvaluator (DynamicsMemory + MetaCognition + attractors)
        |
        v
  IntegrationGate (REJECT / OBSERVE / CANDIDATE / ACCEPT)
        |
        v
  IntegratedAgent.think()
```

### S8.0: Representation Invariance (COMPLETE)
- EXP-EXT-001: WHT is distance-preserving (orthogonal, cosine invariant)
- EXP-EXT-001B: Quantization provides 3.6x invariance; WHT+truncation degrades
- Scripts: `scripts/exp_ext_001_representation_invariance.py`,
  `scripts/exp_ext_001b_what_provides_invariance.py`

### S8.1: MemoryCandidate + CandidateQueue
- `MemoryCandidate`: interaction spectrum, evaluation result, status, timestamps
- `CandidateQueue`: pending/accepted/rejected/stale routing with capacity limits
- `IntegrationDecision`: REJECT, OBSERVE, CANDIDATE, ACCEPT
- File: `substrate_echo/external/candidate_queue.py`

### S8.2: ForeignAgent + ReputationVector
- `ForeignAgent(AgentState)`: extends existing agent state with origin, model_family,
  reputation, interaction history
- `ReputationVector`: behavioral metrics (self_consistency, correction_rate,
  contradiction_rate, novelty_contribution, prediction_alignment, interaction_stability)
- File: `substrate_echo/external/foreign_node.py`

### S8.3: InteractionEncoder
- Dual-path: semantic features (16D) + relational features (16D) → 32D combined
- Semantic: novelty, abstraction, concept relationships, contradiction, uncertainty
- Relational: repetition, persuasion pressure, correction behavior, confidence
- Spectral Normalization (WHT) applied for decorrelation before evaluation
- File: `substrate_echo/external/interaction_encoder.py`

### S8.4: ForeignEvaluator
- Alignment: cosine similarity to existing attractors
- Novelty: DynamicsMemory.novelty()
- Risk: prediction_error under perturbation
- Reputation: per-domain behavioral reputation
- File: `substrate_echo/external/foreign_evaluator.py`

### S8.5: IntegrationGate
- Routes candidates through REJECT / OBSERVE / CANDIDATE / ACCEPT
- OBSERVE: valuable but noisy → accumulate more samples
- Integrated into IntegratedAgent.think() cycle
- File: `substrate_echo/external/integration_gate.py`

### S8.6: MoltbookAdapter (after synthetic validation)
- Actual adapter for external Moltbook agent ecosystem
- Only implemented after S8.1-S8.5 validated with synthetic data
