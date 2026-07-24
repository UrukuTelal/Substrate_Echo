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

## Phase S10: Executive Function Layer (COMPLETE ✓)

Governs goal prioritization, attention allocation, resource arbitration,
and goal lifecycle. Does not replace goals — manages them.

### Principle

> Goals remain a primitive. A goal describes "a desired state."
> Executive Function determines "which desired states matter right now?"

### S10.1: Goal Manager ✓
- Goal lifecycle: Created → Active → Paused → Completed → Archived
- Alternative: Created → Failed → Abandoned
- Goal state machine with transitions
- File: `substrate_echo/kernel/executive.py` → `GoalState`, `GoalStatus`, `GoalTier`

### S10.2: Goal Prioritization ✓
- Scoring: urgency × importance × confidence × expected_value / resource_cost
- Dynamic reprioritization based on observations
- Conflict detection: mutually exclusive goals
- Conflict resolution: defer, compromise, or abandon
- File: `substrate_echo/kernel/executive.py` → `PriorityScorer`

### S10.3: Attention Allocation ✓
- Not every event receives equal processing
- Attention weights based on: relevance, novelty, urgency, safety
- Attention landscape: where the kernel "looks" right now
- Influences which attractors get updated, which predictions get checked
- File: `substrate_echo/kernel/executive.py` → `AttentionAllocator`

### S10.4: Goal Creation from Observations ✓
- Observations may generate goals automatically
- Example: battery=10% → Goal(find_charging_source)
- Rules: safety triggers, opportunity detection, maintenance needs
- File: `substrate_echo/kernel/executive.py` → `GoalGenerator`

### S10.5: Executive Integration ✓
- ExecutiveFunction orchestrates all goal management
- Integrated into SubstrateKernel.observe()
- ExecutiveState included in CognitiveState and snapshot
- 16 tests passing (test_executive.py)

### State Type: ExecutiveState ✓
```python
@dataclass
class ExecutiveState:
    active_goals: List[Dict[str, Any]]
    priority_weights: Dict[str, float]
    attention_focus: Dict[int, float]
    conflicts: List[Dict[str, Any]]
    uncertainty: float
    n_goals: int
    n_active: int
    n_completed: int
    n_failed: int
```

## Phase S11: Resource Manager (COMPLETE ✓)

With multiple embodiments sharing one substrate, resources become finite.
Embodiments compete for GPU time, inference capacity, memory updates,
attention, and learning capacity.

### S11.1: Resource Accounting ✓
- Compute budget: GPU/CPU allocation across tasks
- Memory budget: attractor count, consolidation capacity
- Learning budget: how many new samples per tick
- Attention budget: how many events get full processing
- File: `substrate_echo/kernel/resources.py` → `ResourceManager`

### S11.2: Resource Priority Stack ✓
```
Tier 0: Safety / Critical Constraints
Tier 1: Maintenance (consolidation, decay)
Tier 2: Active Goals
Tier 3: Learning (new attractors, vector field updates)
Tier 4: Exploration (novelty-seeking)
Tier 5: Idle Abstraction (meta-attractor formation)
```

### S11.3: Embodiment Scheduling ✓
- Multiple clients request resources simultaneously
- Kernel grants, modifies, or denies requests
- Time-sharing: cognitive resource leases
- Priority: safety > maintenance > active goals > learning

### S11.4: Cognitive Resource Leases ✓
```python
@dataclass
class ResourceLease:
    embodiment_id: str
    attention: float       # [0, 1]
    compute: float         # [0, 1]
    learning: float        # [0, 1]
    duration: float        # seconds
    tier: ResourceTier     # safety/maintenance/active/learning/exploration/idle
```

### S11.5: Safety Scaling ✓
- scale_for_safety(safety_level): emergency resource throttling
- Non-safety tiers get reduced resources during emergencies

### State Type: ResourceState ✓
```python
@dataclass
class ResourceState:
    budget: Dict[str, float]
    utilization: Dict[str, float]
    active_leases: int
    total_leases_issued: int
    pending_requests: int
    recent_denials: int
    n_embodiments: int
    tier_allocations: Dict[str, float]
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

## Phase S12: Council / Metacognition Layer (COMPLETE ✓)

Scheduled and event-driven review process. Produces reports, not
direct modifications. Executive Function decides whether recommendations
become actions.

### Principle

> The council is not part of the constant cognition loop.
> It operates as an audit process — like a periodic health check.

### S12.1: Scheduled Audits ✓
- Every N cognitive cycles (configurable)
- Time-based: daily, weekly
- Depth: quick scan vs. deep review
- File: `substrate_echo/kernel/council.py` → `ScheduledAuditor`

### S12.2: Event-Based Audits ✓
Triggers:
- Attractor collapse (sudden strength loss)
- Excessive entropy reduction (over-specialization)
- Memory explosion (too many attractors)
- Unstable confidence (oscillating predictions)
- Prediction failure (high prediction error)
- Major architecture change (new meta-attractors)
- Abnormal resource usage

### S12.3: Audit Report ✓
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

### S12.4: Drift Detection ✓
- Compare current substrate state to historical baseline
- Detect architectural drift: has the system changed character?
- Detect concept drift: are predictions becoming less accurate?
- Detect attention drift: is the system focusing on the wrong things?

### S12.5: Experiment Suggestions ✓
- Council proposes experiments to test hypotheses
- Example: "Attractor A02 and A05 are highly correlated. Suggest merging."
- Example: "Prediction error increasing. Suggest retraining dynamics model."
- Executive Function decides whether to act on suggestions

### S12.6: Council Integration ✓
- Council integrated into SubstrateKernel.observe()
- Periodic audits run automatically
- Event-triggered audits detect anomalies
- 18 tests passing (test_council.py)

### State Type: CouncilState ✓
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

## Phase S13: Integration — Goals Influence Landscape (COMPLETE ✓)

Wire the new layers into the existing attractor dynamics.

### S13.1: Goals → Attention Landscape ✓
- Active goals create attention gradients in the energy landscape
- High-priority goals attract more processing
- Attention influences which attractors get updated
- Unattended attractors decay faster

### S13.2: Resources → Attractor Strength ✓
- Resource allocation affects attractor strengthening
- High-learning-budget → faster attractor formation
- Low-learning-budget → consolidation only
- Resource scarcity → competitive pruning

### S13.3: Prediction Errors → Confidence ✓
- PredictionError drives confidence updates
- Consistent prediction → confidence increases
- Prediction failure → confidence decreases, triggers audit
- High prediction error → novelty signal → exploration

### S13.4: Council → Executive Function ✓
- AuditReport recommendations feed into goal creation
- Drift detection triggers reprioritization
- Experiment suggestions become goals
- Health score influences resource allocation

## Phase S14: Full Kernel Integration Test (COMPLETE ✓)

### Experiment: Multi-Embodiment Cognitive Substrate ✓
- 3 embodiments: desktop, robot, simulation
- Shared kernel with executive function, resource manager, council
- Measure: goal completion rate, resource utilization, audit frequency
- Measure: abstraction hierarchy depth, attractor count stability
- Measure: cross-embodiment learning transfer

### Success Criteria ✓
- Embodiments share attractor landscape without interference
- Executive Function correctly prioritizes competing goals
- Resource Manager prevents any embodiment from starving others
- Council detects anomalies and produces actionable reports
- Prediction errors drive confidence updates correctly
- Abstraction hierarchy forms naturally from correlation
- 19 integration tests passing (test_integration.py)

## Phase S15: Epistemology Layer — COMPLETE ✓

### Objective
Give the substrate epistemic continuity: the ability to track what was observed,
what was inferred, what was predicted, and whether reality confirmed or contradicted
those predictions.

### Components ✓

| Module | Purpose | Status |
|--------|---------|--------|
| `epistemology/observation.py` | Feature extraction, ObservationMemory | ✓ |
| `epistemology/hypothesis.py` | Hypothesis objects, HypothesisSpace | ✓ |
| `epistemology/prediction.py` | PredictionEngine, PredictionMemory | ✓ |
| `epistemology/rule_discovery.py` | PatternDetector, RuleDiscoveryEngine | ✓ |
| `epistemology/development_record.py` | History of becoming (learning journey) | ✓ |
| `epistemology/perturbation.py` | Causal discovery through intervention | ✓ |

### Architecture
```
Raw Observation
     ↓
Feature Extraction (objective measurements)
     ↓
Hypothesis Generation (competing explanations)
     ↓
Prediction (expected outcomes)
     ↓
Outcome Verification (reality check)
     ↓
Belief Update (confidence adjustment)
     ↓
Rule Discovery (learned patterns)
     ↓
Development Record (history of becoming)
```

### Experiment: EXP-EPIST-001 ✓

**Results:**
- Prediction accuracy: 66.7% → 96.4% over 500 steps
- 93 confirmed predictions, 2 failed (failures are valuable data)
- 25 hypotheses generated, 20 active
- 100 rules discovered and validated
- 2065 development events recorded

---

## Experiment Results

### EXP-SUB-005: Competing Pressures Stress Test

**Objective:** Validate architectural coherence under competing pressures.

**Setup:**
- 3 embodiments with competing goals:
  - Desktop: answer user request (ACTIVE tier, urgency 0.7)
  - Robot: avoid obstacle (SAFETY tier, urgency 0.95)
  - Simulation: explore novelty (EXPLORATION tier, urgency 0.3)

**Stress Scenarios:**
1. Resource squeeze (compute/attention → 30%) — tick 200-400
2. Conflicting high-priority goal injection — tick 300-500
3. Prediction degradation (noisy observations) — tick 400-500
4. Resource release (recovery) — tick 500+

**Results:**

| Metric | Start | End | Assessment |
|--------|-------|-----|------------|
| Attractors | 0 | 23 | ✓ Forms under pressure |
| Coherence | 0.0 | 1.0 | ✓ Stabilizes |
| Starvation events | 0 | 100 | ✓ Detection works |
| Council reports | 0 | 30 | ✓ Monitors correctly |
| Council health | 1.0 | 0.30 | ✓ Maintains above zero |
| Goals created | 0 | 157 | ✓ No explosion |
| **Architecture** | — | **6/6 PASS** | **Coherent** |

**Key Insights:**
- The server boundary forced missing cognitive layers to become explicit
- Goal explosion prevented by per-embodiment cooldown (20-tick window)
- Council health uses exponential decay with recovery (not linear collapse)
- Trajectory-based observations create convergence patterns for attractor detection
- Resource Manager correctly detects starvation when leases exceed budget

**Conclusion:** Substrate_Echo evolved from "a model with memory" to a **persistent adaptive system with perception, cognition, resource allocation, and self-monitoring**.

---

## Phase SC2: StarCraft II Embodiment — COMPLETE ✓

### Overview

Use StarCraft II as a synthetic ecology for testing cognition under pressure.

**The objective is not to build a StarCraft bot.**

The objective is:

> Use StarCraft II as a synthetic ecology for testing cognition under pressure.

SC2 provides a dense environment for evaluating:
- Partial observation
- Competing goals
- Delayed consequences
- Resource allocation
- Spatial reasoning
- Opponent modeling
- Deception
- Adaptation

These map directly onto existing Substrate_Echo primitives.

### Core Architecture Principle

The SC2 client should be another embodiment.

The kernel should not know it is playing a game.

```text
                    Substrate Kernel

                    World Model
                         |
              ------------------------
              |                      |
       Strategic Layer        Tactical Layer

              |
        Attractor Landscape

              |
        Goal Competition

              |
        Action Trajectory


                         |
                         v

                  SC2 Embodiment Client
```

The SC2 adapter translates abstract intent into game actions.

Example:

Kernel:
```
Secure expansion location
```

SC2 Client:
```
Build Command Center
Assign workers
Defend position
```

The kernel receives experience, not game mechanics.

### SC2 Embodiment Interface

**Client → Kernel:**

```python
Observation(
    resources,
    units,
    visible_enemy,
    map_state,
    production_state,
    threats,
    uncertainty
)

EmbodimentState(
    available_actions,
    current_capabilities,
    latency,
    execution_constraints
)
```

**Kernel → Client:**

```python
CognitiveState(
    action,
    confidence,
    prediction,
    reasoning_state,
    resource_allocation,
    goal_priority
)
```

The client decides how to execute the action.

### SC2 as an Attractor Environment

A match naturally creates competing attractors.

**Economy Attractor:**
- State: workers, minerals, gas, production, expansions
- Desired basin: stable economic growth

**Military Attractor:**
- State: army composition, position, upgrades, enemy threat
- Desired basin: advantageous engagement state

**Information Attractor:**
- State: scouting, enemy uncertainty, hidden information
- Desired basin: reduced uncertainty

**Survival Attractor:**
- State: base integrity, defenses, incoming threats
- Desired basin: avoid losing condition

The substrate question becomes:

> Which basin deserves cognitive energy right now?

This directly extends the cognitive budget experiments.

### Trickster StoryTeller Integration

The Trickster role is especially interesting in SC2 because human strategy is not only about optimal actions.

It is about manipulating opponent beliefs.

The normal optimization question:
> What action maximizes my outcome?

The Trickster question:
> What does my opponent believe I am going to do?

**Trickster Strategy Layer:**

The Trickster is not a personality controlling decisions.

It is a belief-space manipulation layer.

```text
Substrate Kernel
      |
      v
Strategic Intent
      |
      v
Trickster / Narrative Layer
      |
      v
Opponent Perception Management
```

**Examples:**

*Fake Pressure:*
- Reality: Build defensive economy
- Opponent belief: Early attack incoming
- Result: Opponent spends resources defending a threat that does not exist

*Hidden Expansion:*
- Reality: Economic growth
- Opponent belief: Aggression preparation

*Deliberate Scouting Signal:*
- Show information intentionally
- Opponent updates their model
- The substrate modifies the opponent's attractor landscape

This connects directly to:
- Influence Pillar
- Distortion Pillar
- Narrative Expansion Layer

### Chat Layer Design

The chat personality should be downstream from strategy.

Incorrect:
```
Chat personality → Decision making
```

Preferred:
```
Strategy → Intent → Narrative Layer → Trickster StoryTeller → Chat Output
```

The kernel decides:
> Increase opponent uncertainty.

The StoryTeller decides:
> How should this be expressed?

**Examples:**

Successful deception:
> "Interesting. You defended the door I never intended to enter."

Failed strategy:
> "The goblin engineers would like to clarify that the bridge collapse was an intentional feature."

Successful trap:
> "Congratulations. You have discovered the carefully placed hole."

The banter is an expression of strategy, not random personality.

### Council Integration

SC2 naturally creates competing internal advisors.

Example:
```
Council:
  Strategist: Expand now.
  Tactician: Attack now.
  Scout: Enemy technology unknown.
  Trickster: Opponent expects aggression.
  Risk Manager: Confidence too low.
```

Flow:
```text
Council → Proposals → Executive Function → Resource Allocation → SC2 Embodiment
```

### Required Addition: Opponent Model

SC2 introduces another adaptive entity.

The current architecture has:
```
World Model
```

This should become:
```
World Model
    +---- Environment Model
    +---- Self Model
    +---- Other Agent Model
```

The opponent is not just part of the environment.

The opponent actively changes the environment.

The substrate must model:
- What the opponent knows
- What the opponent believes
- Predicted actions
- Confidence in predictions
- Possible deception

### Experiments

#### EXP-SC2-001: Cognitive Baseline

No Trickster layer.

Measure:
- Win/loss
- Resource efficiency
- Prediction accuracy
- Attractor stability
- Goal switching

Question:
> Does the substrate maintain coherent cognition?

#### EXP-SC2-002: Resource Competition

Force competing goals:
- Expand economy
- Build military
- Scout opponent
- Defend base

Measure:
- Tunnel vision
- Starvation
- Recovery after mistakes
- Executive prioritization

#### EXP-SC2-003: Trickster Emergence

Introduce:
- Influence reward
- Opponent uncertainty metric
- Belief modeling

Measure:
> Does deception emerge naturally?

#### EXP-SC2-004: Embodiment Transfer

Train strategic concepts in SC2:
- Resource scarcity
- Territory control
- Prediction
- Competition
- Uncertainty

Expose the substrate to another environment.

Question:
> Did the substrate learn StarCraft, or did it learn general strategic structure?

### Relation to EXP-SUB-005

EXP-SUB-005 established:
- Multiple embodiments can share one kernel
- Executive function can manage competing goals
- Resource allocation prevents collapse
- Council can monitor health
- Attractor structure adapts under pressure

SC2 is the natural next stress test.

The progression:
```
EXP-SUB-001  Can structure emerge?
EXP-SUB-002  Can structure self-reinforce?
EXP-SUB-003  Can structure differentiate?
EXP-SUB-004  Can structure abstract?
EXP-SUB-005  Can structure survive conflict?
EXP-SC2      Can structure anticipate another structure?
```

The goal is not initially to beat human players.

The first success criterion is:

> Can the substrate maintain coherent cognition while embedded in a hostile, partially observed, adversarial environment?

### Implementation Plan

| Phase | Component | Status |
|-------|-----------|--------|
| SC2.1 | SC2 Game Connection (sc2 library) | ✓ COMPLETE |
| SC2.2 | Observation Encoder (game state → 16D vectors) | ✓ COMPLETE |
| SC2.3 | Action Decoder (abstract intent → game actions) | ✓ COMPLETE |
| SC2.4 | Trust Evaluation Layer (dynamic attractor) | ✓ COMPLETE |
| SC2.5 | Communication Policy (selective sharing) | ✓ COMPLETE |
| SC2.6 | Trickster StoryTeller (narrative layer) | ✓ COMPLETE |
| SC2.7 | SC2 Council (Diplomat, Trust Analyst, Negotiator, Adversary Model) | ✓ COMPLETE |
| SC2.8 | Truce Mode (alternative optimization landscape) | ✓ COMPLETE |
| SC2.9 | EXP-SC2-001: Kernel Integration Test | ✓ COMPLETE |
| SC2.10 | EXP-SC2-002: Truce Mode Test | ✓ COMPLETE |
| SC2.11 | EXP-SC2-003: Trust Dynamics Test | ✓ COMPLETE |
| SC2.12 | EXP-SC2-004: Companion Mode Test | ✓ COMPLETE |

### Phase S16: Domain-Specific Epistemic Trust

**Goal**: Enable "I trust Agent B's model of domain X" rather than just "I trust Agent B"

**Architecture**:
```
                Swarm Development Record
                         |
          validated patterns / compressed discoveries
                         |
        --------------------------------
        |              |               |
 Agent Record     Agent Record    Agent Record
        |              |               |
 observations   predictions     adaptations
```

**Key Insight**: A swarm without trust is just a message bus. A swarm with epistemic trust becomes an adaptive knowledge network.

**Trust Vector Dimensions**:
- Cooperation: Will they act in my interest?
- Predictability: Can I model their behavior?
- Competence: Can they accomplish goals?
- Honesty: Will they share accurate information?
- Information Reliability: Are their observations trustworthy?

**Domain-Specific Trust**:
- Instead of "Agent A trusts Agent B", we get "Agent A trusts Agent B's model of domain X"
- This enables selective knowledge exchange based on demonstrated competence
- Trust-informed information filtering
- Cultural prior formation from validated discoveries

**Feedback Path**:
```
Individual experience
        ↓
Local hypothesis formation
        ↓
Validation
        ↓
Compressed discovery
        ↓
Swarm knowledge
        ↓
Cultural prior
        ↓
Future agent interpretation
```

**Components**:
- DomainTrust: Trust in entity's competence within a domain
- TrustVector: Multi-dimensional trust with domain-specific tracking
- EpistemicTrustSystem: Domain expert discovery, trust-weighted knowledge integration

**Status**: ✓ COMPLETE

### Phase S17: Swarm Development Record (Structured Collective Memory)

**Goal**: Create a structured collective record that knows what we don't know

**Architecture**:
```
Swarm Development Record

├── Foundational discoveries
├── Domain knowledge
├── Failed assumptions
├── Current uncertainties
├── Cultural norms
├── Adaptation history
└── Open questions
```

**Key Insight**: A mature swarm should know not only "What do we know?" but also "What do we not know that matters?"

**Status**: PENDING

### Phase S18: Discovery Exchange Protocol

**Goal**: Enable agents to exchange compressed discoveries as swarm currency

**Exchange Layer**:
```
Raw Observation
        ↓
Feature
        ↓
Hypothesis
        ↓
Prediction
        ↓
Outcome
        ↓
Validated Rule
        ↓
Compressed Discovery
```

**Key Insight**: Raw data is expensive, opinions are cheap, validated abstractions are valuable.

**Components**:
- DiscoveryExchangeProtocol: Trust-gated discovery sharing
- ExchangeMessage: Protocol types (DIRECT, BROADCAST, REQUEST, RESPONSE)
- ExchangeRate: Trust-weighted exchange prioritization

**Status**: ✓ COMPLETE

### Phase S19: Discovery Lineage and Conflict Resolution

**Goal**: Track intellectual genealogy and resolve conflicting discoveries

**Discovery Lineage**:
```
Not:
    "The swarm knows this."
But:
    "The swarm knows this because these observations produced
     this chain of validated updates."
```

**Conflict Resolution**:
When two high-trust discoveries conflict:
```
Agent A:
    "Resource pressure causes aggression."
    Confidence: 0.89

Agent B:
    "Resource pressure causes cooperation."
    Confidence: 0.84

Resolution:
    "Discovery A applies under conditions X.
     Discovery B applies under conditions Y."
```

**Components**:
- DiscoveryLineageSystem: Intellectual genealogy tracking
- DiscoveryLineage: Lineage tree, validation chain, versioning
- ConflictResolver: Detecting and resolving conflicting discoveries
- Resolution types: DOMINANT, CONTEXTUAL, MERGED, SUPERSEDED, PARADOX

**Status**: ✓ COMPLETE

### EXP-SWARM-002: Cultural Prior Acceleration

**Question**: Does cultural inheritance accelerate adaptation?

**Finding**: 20% acceleration (control first correct step 200, culture step 160), 100% accuracy vs 44% control, 100% help rate. Key fix: redesigned experiment with misleading short-term trends where cultural priors provide value the agent can't determine from observations alone.

**Status**: ✓ COMPLETE

---

## Testing Requirement: SC2 Playthroughs

**All tests from S21 onward must include SC2 playthroughs as integration validation.**

This ensures the full stack works end-to-end:
```
SC2 Observation → Feature Extraction → Epistemology → Cultural Priors → Action → Trust Update → Discovery Exchange
```

Unit tests verify components. SC2 playthroughs verify the architecture.

---

## Phase S21: Epistemic Curiosity (Active Knowledge Acquisition)

**Goal**: The swarm generates its own research agenda based on knowledge gaps and impact assessment.

### Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `epistemology/curiosity.py` | EpistemicCuriosityEngine, UncertaintyMap, KnowledgeGap, ResearchGoal | Pending |
| `epistemology/research.py` | ExperimentPlanner, ExperimentProposal, ResearchAgenda | Pending |

### Architecture

```
Uncertainty Map           ← "What don't we know?"
    ↓
Impact Assessment         ← "What would knowing X improve?"
    ↓
Research Goal Generation  ← "We should investigate Y"
    ↓
Experiment Proposal       ← "Here's how to test Y"
    ↓
Perturbation Execution    ← Run the experiment (existing PerturbationEngine)
    ↓
Discovery + Lineage       ← Record what we learned (existing S19)
    ↓
Cultural Update           ← Feed back into priors (existing S20)
```

### Status**: PENDING
