# Substrate_Echo

A self-organizing cognitive substrate with emergent attractor dynamics, basin topology, abstraction hierarchy, and a client-server architecture separating cognition from embodiment.

## What It Is

Substrate_Echo is not a chatbot, a RAG pipeline, or a prompt framework. It is a **dynamical cognitive substrate** that:

- Learns vector fields from experience (DynamicsMemory)
- Self-organizes attractors from trajectory convergence
- Builds basin topology (depth, volume, entropy, balance)
- Develops plasticity (stability, novelty, confidence per attractor)
- Creates abstraction hierarchy (meta-attractors from correlated base attractors)
- Separates cognition (kernel) from embodiment (clients)

## Architecture

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
├── external/         # Epistemic firewall for external agents
├── social/           # Persona genomes, relationship dynamics
├── models/           # Data structures (Experience, MemoryTrace, etc.)
├── integration/      # Bridges (PSV, Void, Engine, Council)
├── scenarios/        # End-to-end simulations
└── visualization/    # Field and agent rendering

scripts/
├── experiments/      # Reproducible experiments (EXP-SUB-001 through 004)
└── demo_kernel.py    # Kernel demo: two embodiments sharing one mind

tests/                # 702 automated tests
```

## Experiments

### Substrate Dynamics

| Experiment | Result | Implication |
|-----------|--------|-------------|
| EXP-SUB-001 | 4.9x improvement (0.075→0.365) | Dynamics and semantics are complementary |
| EXP-SUB-002 | 2→14 attractors, coherence 0.202→0.924 | Closed feedback loop is self-reinforcing |
| EXP-SUB-003 | 16 attractors, depth 0.457, plasticity分化 | Feedback develops measurable internal geometry |
| EXP-SUB-004 | 4 meta-attractors from correlation | Correlated attractors produce abstraction hierarchy |

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

## Quick Start

```bash
pip install -e .
pytest tests/ -q  # 702 tests
```

Run the kernel demo:
```bash
python scripts/demo_kernel.py
```

Run an experiment:
```bash
python scripts/experiments/exp_sub_002_feedback_loop.py
python scripts/experiments/exp_sub_004_abstraction.py
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

### Cognitive Plane
Clients publish `Observation`, `Goal`, `Reward`, `EmbodimentState`. The kernel responds with `CognitiveState` containing `Action` and `Prediction`. The client never calls `remember()` or `plan()` — it publishes state, and the kernel decides how experience changes the landscape.

## License

MIT
