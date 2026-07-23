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

## Phase S9: Substrate Kernel — Cognitive Backend (COMPLETE ✓)

Separate cognition from embodiment. The kernel owns all persistent cognitive
state. Clients publish observations and receive actions.

### Architecture

```
                   Substrate Kernel (cognition)
+---------------------------------------------------+
|  DynamicsMemory   |  BasinTopology               |
|  AttractorNetwork |  AbstractionEngine            |
|  EnergyLandscape  |  CognitiveBudget              |
|  EpisodicMemory   |  WorldModels                  |
|  MetaAttractors   |  Learning                     |
+---------------------------------------------------+
        Cognitive Plane (streaming)          Control Plane (REST)
        observations, goals, actions         state, topology, config
               |                                  |
     +---------+---------+              +--------+--------+
     |         |         |              |                 |
  Desktop    Robot     VR Avatar    Checkpoint        Health
  Client     Client    Client
```

### S9.1: Kernel Core (COMPLETE)
- `kernel/__init__.py`: SubstrateKernel, state schema, cognitive engine
- State types: Observation, Goal, Reward, Action, Prediction, EmbodimentState, CognitiveState
- Clients never manipulate cognition — they publish state, kernel decides
- File: `substrate_echo/kernel/__init__.py`

### S9.2: Two-Plane API (COMPLETE)
- Control Plane (REST): /kernel/state, /kernel/topology, /kernel/abstraction, /kernel/embodiments, /kernel/checkpoint, /health
- Cognitive Plane (WebSocket): streaming observation → action loop
- File: `substrate_echo/kernel/api.py`

### S9.3: Client Library (COMPLETE)
- SubstrateClient: in-process client for testing
- StreamingClient: continuous loop at fixed tick rate for real-time embodiments
- File: `substrate_echo/kernel/client.py`

### S9.4: Basin Topology (COMPLETE)
- Basin depth, volume, entropy, balance metrics
- AttractorState with plasticity: stability, plasticity, novelty, confidence
- Structural event detection: births, deaths, merges, splits
- File: `substrate_echo/dynamics/basin_topology.py`

### S9.5: Abstraction Engine (COMPLETE)
- AttractorCorrelation: time-proximity co-activation tracking
- AbstractionEngine: meta-attractor creation from correlated clusters
- CognitiveBudget: finite energy, competition for resources
- File: `substrate_echo/dynamics/abstraction.py`

### Experiments (COMPLETE)
- EXP-SUB-001: Dynamics + semantics complementary (4.9x improvement)
- EXP-SUB-002: Closed feedback loop self-reinforcing (2→14 attractors, coherence 0.924)
- EXP-SUB-003: Basin topology with plasticity分化 (16 attractors, depth 0.457)
- EXP-SUB-004: Abstraction hierarchy (4 meta-attractors from correlation)
- Demo: two embodiments sharing one cognitive kernel

## Phase S10: Executive Function Layer

Governs goal prioritization, attention allocation, resource arbitration,
and goal lifecycle. Does not replace goals — manages them.

### Principle

> Goals remain a primitive. A goal describes "a desired state."
> Executive Function determines "which desired states matter right now?"

### S10.1: Goal Manager
- Goal lifecycle: Created → Active → Paused → Completed → Archived
- Alternative: Created → Failed → Abandoned
- Goal state machine with transitions
- File: `substrate_echo/kernel/executive.py` → `GoalManager`

### S10.2: Goal Prioritization
- Scoring: urgency × importance × confidence × expected_value / resource_cost
- Dynamic reprioritization based on observations
- Conflict detection: mutually exclusive goals
- Conflict resolution: defer, compromise, or abandon
- File: `substrate_echo/kernel/executive.py` → `PriorityScorer`

### S10.3: Attention Allocation
- Not every event receives equal processing
- Attention weights based on: relevance, novelty, urgency, safety
- Attention landscape: where the kernel "looks" right now
- Influences which attractors get updated, which predictions get checked
- File: `substrate_echo/kernel/executive.py` → `AttentionAllocator`

### S10.4: Goal Creation from Observations
- Observations may generate goals automatically
- Example: battery=10% → Goal(find_charging_source)
- Rules: safety triggers, opportunity detection, maintenance needs
- File: `substrate_echo/kernel/executive.py` → `GoalGenerator`

### State Type: ExecutiveState
```python
@dataclass
class ExecutiveState:
    active_goals: List[GoalState]
    priority_weights: Dict[str, float]
    attention_focus: List[int]    # which attractors have attention
    resource_budget: ResourceBudget
    conflicts: List[GoalConflict]
    uncertainty: float
```

## Phase S11: Resource Manager

With multiple embodiments sharing one substrate, resources become finite.
Embodiments compete for GPU time, inference capacity, memory updates,
attention, and learning capacity.

### S11.1: Resource Accounting
- Compute budget: GPU/CPU allocation across tasks
- Memory budget: attractor count, consolidation capacity
- Learning budget: how many new samples per tick
- Attention budget: how many events get full processing
- File: `substrate_echo/kernel/resources.py` → `ResourceManager`

### S11.2: Resource Priority Stack
```
Tier 0: Safety / Critical Constraints
Tier 1: Maintenance (consolidation, decay)
Tier 2: Active Goals
Tier 3: Learning (new attractors, vector field updates)
Tier 4: Exploration (novelty-seeking)
Tier 5: Idle Abstraction (meta-attractor formation)
```

### S11.3: Embodiment Scheduling
- Multiple clients request resources simultaneously
- Kernel grants, modifies, or denies requests
- Time-sharing: cognitive resource leases
- Priority: safety > maintenance > active goals > learning

### S11.4: Cognitive Resource Leases
```python
@dataclass
class ResourceLease:
    embodiment_id: str
    attention: float       # [0, 1]
    compute: float         # [0, 1]
    learning: float        # [0, 1]
    duration: float        # seconds
    priority: str          # "safety", "maintenance", "goal", "learning"
```

### State Type: ResourceState
```python
@dataclass
class ResourceState:
    compute_available: float
    memory_available: float
    attention_available: float
    active_embodiments: int
    allocation_history: List[ResourceAllocation]
    tier_usage: Dict[int, float]
```

## Phase S12: Council / Metacognition Layer

Scheduled and event-driven review process. Produces reports, not
direct modifications. Executive Function decides whether recommendations
become actions.

### Principle

> The council is not part of the constant cognition loop.
> It operates as an audit process — like a periodic health check.

### S12.1: Scheduled Audits
- Every N cognitive cycles (configurable)
- Time-based: daily, weekly
- Depth: quick scan vs. deep review
- File: `substrate_echo/kernel/council.py` → `ScheduledAuditor`

### S12.2: Event-Based Audits
Triggers:
- Attractor collapse (sudden strength loss)
- Excessive entropy reduction (over-specialization)
- Memory explosion (too many attractors)
- Unstable confidence (oscillating predictions)
- Prediction failure (high prediction error)
- Major architecture change (new meta-attractors)
- Abnormal resource usage

### S12.3: Audit Report
```python
@dataclass
class AuditReport:
    tick: int
    audit_type: str           # "scheduled", "event", "manual"
    observations: List[str]   # what was noticed
    anomalies: List[str]      # what seems wrong
    hypotheses: List[str]     # possible explanations
    recommendations: List[str]  # suggested actions
    confidence: float         # how sure the council is
    severity: str             # "info", "warning", "critical"
```

### S12.4: Drift Detection
- Compare current substrate state to historical baseline
- Detect architectural drift: has the system changed character?
- Detect concept drift: are predictions becoming less accurate?
- Detect attention drift: is the system focusing on the wrong things?

### S12.5: Experiment Suggestions
- Council proposes experiments to test hypotheses
- Example: "Attractor A02 and A05 are highly correlated. Suggest merging."
- Example: "Prediction error increasing. Suggest retraining dynamics model."
- Executive Function decides whether to act on suggestions

### State Type: CouncilState
```python
@dataclass
class CouncilState:
    last_audit_tick: int
    pending_reports: List[AuditReport]
    drift_score: float
    health_score: float
    n_audits: int
    n_recommendations: int
```

## Phase S13: Integration — Goals Influence Landscape

Wire the new layers into the existing attractor dynamics.

### S13.1: Goals → Attention Landscape
- Active goals create attention gradients in the energy landscape
- High-priority goals attract more processing
- Attention influences which attractors get updated
- Unattended attractors decay faster

### S13.2: Resources → Attractor Strength
- Resource allocation affects attractor strengthening
- High-learning-budget → faster attractor formation
- Low-learning-budget → consolidation only
- Resource scarcity → competitive pruning

### S13.3: Prediction Errors → Confidence
- PredictionError drives confidence updates
- Consistent prediction → confidence increases
- Prediction failure → confidence decreases, triggers audit
- High prediction error → novelty signal → exploration

### S13.4: Council → Executive Function
- AuditReport recommendations feed into goal creation
- Drift detection triggers reprioritization
- Experiment suggestions become goals
- Health score influences resource allocation

## Phase S14: Full Kernel Integration Test

### Experiment: Multi-Embodiment Cognitive Substrate
- 3 embodiments: desktop, robot, simulation
- Shared kernel with executive function, resource manager, council
- Measure: goal completion rate, resource utilization, audit frequency
- Measure: abstraction hierarchy depth, attractor count stability
- Measure: cross-embodiment learning transfer

### Success Criteria
- Embodiments share attractor landscape without interference
- Executive Function correctly prioritizes competing goals
- Resource Manager prevents any embodiment from starving others
- Council detects anomalies and produces actionable reports
- Prediction errors drive confidence updates correctly
- Abstraction hierarchy forms naturally from correlation
