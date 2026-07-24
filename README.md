# Substrate_Echo

A self-organizing cognitive substrate with emergent attractor dynamics, basin topology, abstraction hierarchy, executive function, and a client-server architecture separating cognition from embodiment.

## What It Is

Substrate_Echo is not a chatbot, a RAG pipeline, or a prompt framework. It is a **dynamical cognitive substrate** that:

- Learns vector fields from experience (DynamicsMemory)
- Self-organizes attractors from trajectory convergence
- Builds basin topology (depth, volume, entropy, balance)
- Develops plasticity (stability, novelty, confidence per attractor)
- Creates abstraction hierarchy (meta-attractors from correlated base attractors)
- Manages goals through executive function (prioritization, attention, lifecycle)
- Allocates finite resources across competing embodiments
- Self-monitors through scheduled and event-driven council audits
- Separates cognition (kernel) from embodiment (clients)

## Architecture

```
                         Substrate Kernel

+-----------------------------------------------------+
|                 Embodiment Interface                |
|                                                     |
| Observation | Goal | Reward | EmbodimentState       |
+--------------------------+--------------------------+
                           |
                           v
+-----------------------------------------------------+
|              Cognitive Runtime                     |
|                                                     |
| DynamicsMemory     |  BasinTopology               |
| AttractorMemory    |  AbstractionEngine            |
| Energy Landscape   |  CognitiveBudget              |
| Meta-Attractor Hierarchy  |  World Models          |
| Prediction         |  Learning                     |
+--------------------------+--------------------------+
                           |
                           v
+-----------------------------------------------------+
|              Executive Function                     |
|                                                     |
| Goal Management     |  Goal Prioritization         |
| Attention Allocation |  Conflict Resolution        |
| Resource Arbitration |  Goal Lifecycle             |
+--------------------------+--------------------------+
                           |
                           v
+-----------------------------------------------------+
|              Resource Manager                       |
|                                                     |
| Compute Budget      |  Memory Budget               |
| Learning Budget     |  Attention Budget            |
| Embodiment Scheduling | Resource Priority Stack    |
+-----------------------------------------------------+

                           ^
                           |
                           |

+-----------------------------------------------------+
|              Council / Metacognition                |
|                                                     |
| Scheduled Audits    |  Event-Based Audits          |
| Drift Detection     |  Contradiction Detection     |
| Experiment Suggestions | Architecture Review       |
+-----------------------------------------------------+
```

**Four layers:**
- **Cognitive Runtime**: learns, remembers, predicts, self-organizes
- **Executive Function**: decides what matters right now
- **Resource Manager**: allocates finite cognitive resources
- **Council**: monitors health, detects drift, suggests experiments

**Two planes:**
- **Cognitive Plane** (WebSocket): streaming observations → actions. State-based, not message-based.
- **Control Plane** (REST): admin, persistence, topology queries, health checks.

**Clients never manipulate cognition.** They publish sensor state. The kernel decides how experience changes the cognitive landscape.

## Project Structure

```
substrate_echo/
├── kernel/           # Substrate Kernel: cognitive backend + API
│   ├── __init__.py   # Kernel, state schema, cognitive engine
│   ├── api.py        # Control Plane (REST) + Cognitive Plane (WebSocket)
│   └── client.py     # Client library (in-process + streaming)
├── core/             # Cognitive modules (42 modules)
│   ├── dynamics_memory.py      # Learns V(x) from (state, velocity)
│   ├── attractor_memory.py     # Persistent attractors in ontological field
│   ├── cognitive_loop.py       # Perception-action-learning cycle
│   ├── episodic_memory.py      # Temporal narrative memory
│   ├── world_model.py          # Predictive world model
│   ├── meta_cognition.py       # Self-monitoring and calibration
│   └── ...                     # planner, goal tracker, habit formation, etc.
├── dynamics/         # Field dynamics + topology + abstraction
│   ├── field_evolution.py      # ODE solvers (RK4, Crank-Nicolson)
│   ├── basin_topology.py       # Basin depth, volume, entropy, events
│   ├── abstraction.py          # Meta-attractor creation from correlation
│   ├── attractor_dynamics.py   # Attractor formation, decay, merging
│   ├── pillar_coupling.py      # 16-pillar inter-coupling
│   └── ...
├── epistemology/     # Epistemic continuity layer
│   ├── observation.py          # Feature extraction, ObservationMemory
│   ├── hypothesis.py           # Hypothesis objects, HypothesisSpace
│   ├── prediction.py           # PredictionEngine, PredictionMemory
│   ├── rule_discovery.py       # PatternDetector, RuleDiscoveryEngine
│   ├── development_record.py   # History of becoming
│   └── perturbation.py         # Causal discovery through intervention
├── external/         # Epistemic firewall for external agents
├── social/           # Persona genomes, relationship dynamics
├── models/           # Data structures (Experience, MemoryTrace, etc.)
├── integration/      # Bridges (PSV, Void, Engine, Council)
├── scenarios/        # End-to-end simulations
└── visualization/    # Field and agent rendering

scripts/
├── experiments/      # Reproducible experiments (EXP-SUB-001 through 005)
└── demo_kernel.py    # Kernel demo: two embodiments sharing one mind

tests/                # 772 automated tests
```

## Implementation Plan

| Phase | Layer | Status |
|-------|-------|--------|
| S1-S8 | Core scaffolding, dynamics, memory, agents, bridges, external | COMPLETE |
| S9 | Substrate Kernel: cognitive backend, basin topology, abstraction | COMPLETE |
| S10 | Executive Function: goal management, prioritization, attention | COMPLETE |
| S11 | Resource Manager: compute/memory/attention budgets, scheduling | COMPLETE |
| S12 | Council: scheduled audits, drift detection, experiment suggestions | COMPLETE |
| S13 | Integration: goals→landscape, resources→attractors, council→executive | COMPLETE |
| S14 | Full kernel integration test: multi-embodiment cognitive substrate | COMPLETE |

## Experiments

### Substrate Dynamics

| Experiment | Result | Implication |
|-----------|--------|-------------|
| EXP-SUB-001 | 4.9x improvement (0.075→0.365) | Dynamics and semantics are complementary |
| EXP-SUB-002 | 2→14 attractors, coherence 0.202→0.924 | Closed feedback loop is self-reinforcing |
| EXP-SUB-003 | 16 attractors, depth 0.457, plasticity分化 | Feedback develops measurable internal geometry |
| EXP-SUB-004 | 4 meta-attractors from correlation | Correlated attractors produce abstraction hierarchy |
| EXP-SUB-005 | 6/6 architecture checks pass | Architecture coherent under competing pressures |

### Stress Test: Competing Pressures (EXP-SUB-005)

Three embodiments with competing goals under resource constraints:

| Embodiment | Goal | Tier | Urgency |
|------------|------|------|---------|
| Desktop | Answer user request | ACTIVE | 0.7 |
| Robot | Avoid obstacle | SAFETY | 0.95 |
| Simulation | Explore novelty | EXPLORATION | 0.3 |

**Stress scenarios applied:**
1. Resource squeeze (compute/attention → 30%)
2. Conflicting high-priority goal injection
3. Prediction degradation (noisy observations)
4. Resource release (recovery)

**Results:**

| Metric | Value | Assessment |
|--------|-------|------------|
| Attractors formed | 0 → 23 | ✓ Forms under pressure |
| Coherence | 0 → 1.0 | ✓ Stabilizes |
| Starvation detected | 100 events | ✓ Detection works |
| Council reports | 30 | ✓ Monitors correctly |
| Council health | 0.30 | ✓ Maintains above zero |
| Goals managed | 157 created | ✓ No explosion |
| **Overall** | **6/6 PASS** | **Architecture coherent** |

The server boundary forced the missing cognitive layers to become explicit. Substrate_Echo evolved from "a model with memory" to a **persistent adaptive system with perception, cognition, resource allocation, and self-monitoring**.

### External Agent Integration

| Experiment | Result | Implication |
|-----------|--------|-------------|
| EXP-EXT-001 | WHT ratio = 1.0000 | WHT is distance-preserving, not a filter |
| EXP-EXT-001B | Quantization 3.6x | Scalar binning preserves contrast |
| EXP-EXT-002 | FAR = 0% | Temporal drift correctly detected |
| EXP-EXT-003 | 86% contamination | Heuristic encoder is weakest component |
| EXP-EXT-004 | B(0.716) > A(0.610) | System values usefulness over presentation |
| EXP-EXT-005 | Physics A>C>B | Domain-specific trust works |
| EXP-SOC-001 | Divergence 0.085 stable | Persona anchoring prevents convergence |

### StarCraft II Embodiment (COMPLETE ✓)

SC2 is now a controllable external environment for Substrate_Echo.

**Embodiment Boundary Validated:**
- Observation Encoder: SC2 game state → 16D substrate vectors
- Action Decoder: Abstract intent → SC2 commands
- Trust Layer: Dynamic trust attractor for multi-agent interactions
- Communication Policy: Selective information sharing
- Trickster StoryTeller: Narrative/social intelligence
- SC2 Council: Diplomat, Trust Analyst, Negotiator, Adversary Model
- Truce Mode: Alternative optimization landscape for cooperative play

**Connection Verified:**
```text
500 steps completed successfully against Easy AI
9 Melee maps installed
```

**Experiment Progression:**
```
EXP-SUB-001  Can structure emerge?                    ✓
EXP-SUB-002  Can structure self-reinforce?            ✓
EXP-SUB-003  Can structure differentiate?             ✓
EXP-SUB-004  Can structure abstract?                  ✓
EXP-SUB-005  Can structure survive conflict?          ✓
EXP-SC2-001  Kernel Integration Test                  ✓
EXP-SC2-002  Truce Mode Test                          ✓
EXP-SC2-003  Trust Dynamics Test                       ✓
EXP-SC2-004  Companion Mode Test                       ✓
```

**First Success Criterion:**
> Can the substrate maintain coherent cognition while embedded in a hostile, partially observed, adversarial environment?

**All SC2 experiments complete.** See [PLAN.md](PLAN.md) for full SC2 architecture and experiment details.

## Quick Start

```bash
pip install -e .
pytest tests/ -q  # 772 tests
```

Run the kernel demo:
```bash
python scripts/demo_kernel.py
```

Run an experiment:
```bash
python scripts/experiments/exp_sub_002_feedback_loop.py
python scripts/experiments/exp_sub_004_abstraction.py
python scripts/exp_sub_005_competing_pressures.py
```

Start the kernel server:
```bash
uvicorn substrate_echo.kernel.api:create_app --factory --port 8000
```

## Key Concepts

### Basin Topology
The energy landscape has measurable geometry: basin depth (energy contrast), volume (isolation), entropy (diversity), and balance (dominance). These describe the **shape of cognition**, not just the quantity of memories.

### Plasticity
Each attractor has stability, plasticity, novelty, and confidence. Frequently accessed attractors become more stable and confident but less plastic. Unused attractors remain frozen — candidate hypotheses, not yet memories.

### Abstraction
When base attractors co-activate repeatedly, the system creates meta-attractors representing their shared structure. This builds hierarchy: specific instances → general concepts → abstract categories.

### Executive Function
Goals are primitives. Executive Function determines which goals matter right now — through prioritization, attention allocation, conflict resolution, and lifecycle management.

### Cognitive Plane
Clients publish `Observation`, `Goal`, `Reward`, `EmbodimentState`. The kernel responds with `CognitiveState` containing `Action` and `Prediction`. The client never calls `remember()` or `plan()` — it publishes state, and the kernel decides how experience changes the landscape.

### Council
The council is not part of the constant cognition loop. It performs scheduled and event-driven audits, produces reports with anomalies/hypotheses/recommendations, and Executive Function decides whether to act on them.

## License

MIT
