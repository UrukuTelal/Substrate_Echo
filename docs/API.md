# Substrate Echo — API Documentation

## Overview

The Substrate Kernel exposes two communication planes:

- **Control Plane (REST)** — Admin, persistence, monitoring
- **Cognitive Plane (WebSocket)** — Streaming observation/action loop

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Substrate Kernel                          │
├─────────────────────────────────────────────────────────────┤
│  Executive Function │ Resource Manager │ Council             │
│  (goals, attention) │ (leases, budgets)│ (audits, drift)    │
├─────────────────────────────────────────────────────────────┤
│  Dynamics Memory    │ Basin Topology   │ Abstraction Engine  │
│  (V(x) field)      │ (energy grid)    │ (meta-attractors)   │
├─────────────────────┴──────────────────┴─────────────────────┤
│              Control Plane (REST)                            │
│  /health  /kernel/state  /kernel/checkpoint                  │
├─────────────────────────────────────────────────────────────┤
│              Cognitive Plane (WebSocket)                     │
│  observation → action + prediction (streaming)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Control Plane (REST)

Base URL: `http://localhost:8000`

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "tick": 1234
}
```

### Kernel State

```
GET /kernel/state
```

**Response:**
```json
{
  "tick": 1234,
  "n_attractors": 5,
  "n_meta_attractors": 2,
  "basin_balance": 0.85,
  "mean_depth": 0.42,
  "volume_entropy": 1.23,
  "coherence": 0.78,
  "cognitive_energy": 8.5,
  "n_goals": 3,
  "n_rewards": 12,
  "n_embodiments": 2,
  "active_embodiments": 2,
  "abstraction_events": 5,
  "topology_events": 20,
  "resources": {
    "total_compute": 1.0,
    "used_compute": 0.4,
    "available_compute": 0.6
  }
}
```

### Topology History

```
GET /kernel/topology
```

**Response:**
```json
[
  {"tick": 100, "attractors": 3, "depth": 0.4, "entropy": 1.1, "balance": 0.8},
  {"tick": 200, "attractors": 5, "depth": 0.45, "entropy": 1.2, "balance": 0.85}
]
```

### Abstraction Events

```
GET /kernel/abstraction
```

**Response:**
```json
[
  {"tick": 500, "type": "meta_created", "parent_ids": [0, 1], "child_id": 100}
]
```

### Connected Embodiments

```
GET /kernel/embodiments
```

**Response:**
```json
{
  "desktop": {"type": "generic", "active": true, "modalities": ["visual", "audio"]},
  "robot": {"type": "generic", "active": true, "modalities": ["proprio"]}
}
```

### Checkpoint

```
POST /kernel/checkpoint
```

**Request:**
```json
{
  "path": "checkpoint.json"
}
```

**Response:**
```json
{
  "status": "saved",
  "path": "checkpoint.json"
}
```

---

## Cognitive Plane (WebSocket)

Endpoint: `ws://localhost:8000/cognitive`

### Message Types

#### Observation

Publish sensor state from an embodiment:

```json
{
  "type": "observation",
  "vector": [0.1, 0.2, 0.3, ...],
  "modality": "visual",
  "embodiment_id": "desktop",
  "metadata": {}
}
```

#### Goal

Publish a desired state:

```json
{
  "type": "goal",
  "target": [0.5, 0.5, 0.5, ...],
  "priority": 0.8,
  "description": "Reach target position",
  "embodiment_id": "robot"
}
```

#### Reward

Publish reinforcement signal:

```json
{
  "type": "reward",
  "value": 0.7,
  "target_attractor": 3,
  "embodiment_id": "desktop"
}
```

#### Embodiment State

Register/update embodiment:

```json
{
  "type": "embodiment",
  "embodiment_id": "drone",
  "embodiment_type": "aerial",
  "available_modalities": ["visual", "lidar"],
  "is_active": true
}
```

### Response Format

Every message returns a `CognitiveState`:

```json
{
  "tick": 1234,
  "action": {
    "vector": [0.1, 0.2, ...],
    "confidence": 0.85,
    "source": "attractor_3"
  },
  "prediction": {
    "expected_next": [0.15, 0.25, ...],
    "confidence": 0.72
  },
  "n_attractors": 5,
  "n_meta_attractors": 2,
  "coherence": 0.78,
  "basin_balance": 0.85,
  "mean_depth": 0.42,
  "volume_entropy": 1.23,
  "active_goals": 3,
  "active_embodiments": 2
}
```

---

## Python Client

### In-Process Client

```python
from substrate_echo.kernel import SubstrateKernel, Observation
from substrate_echo.kernel.client import SubstrateClient

kernel = SubstrateKernel()
client = SubstrateClient(kernel)

# Publish observation
state = client.observe(
    vector=[0.1, 0.2, 0.3, ...],
    modality="visual",
    embodiment_id="desktop"
)

print(f"Tick: {state.tick}")
print(f"Action: {state.action}")
print(f"Coherence: {state.coherence}")
```

### Streaming Client

```python
from substrate_echo.kernel.client import StreamingClient

client = StreamingClient("ws://localhost:8000/cognitive")

# Streaming observation loop
for obs in sensor_stream():
    state = client.send_observation(
        vector=obs.vector,
        modality=obs.modality,
        embodiment_id="robot"
    )
    execute(state.action)
```

---

## Executive Function API

```python
from substrate_echo.kernel.executive import GoalState, GoalTier

# Add goal
goal = GoalState(
    target=[0.5] * 16,
    description="Explore environment",
    urgency=0.8,
    importance=0.7,
    tier=GoalTier.EXPLORATION,
)
goal_id = kernel.executive.add_goal(goal)
kernel.executive.activate_goal(goal_id)

# Tick executive
exec_state = kernel.executive.tick()
print(f"Active goals: {exec_state.n_active}")
print(f"Attention focus: {exec_state.attention_focus}")

# Complete/fail goal
kernel.executive.complete_goal(goal_id)
```

---

## Resource Manager API

```python
from substrate_echo.kernel.resources import ResourceRequest, ResourceTier

# Request resources
req = ResourceRequest(
    embodiment_id="desktop",
    compute=0.3,
    attention=0.2,
    learning=0.1,
    tier=ResourceTier.ACTIVE,
)
result = kernel.resources.request(req)

if result.granted:
    print(f"Lease ID: {result.lease.lease_id}")
    print(f"Compute: {result.modified_compute}")

# Release resources
kernel.resources.release(result.lease.lease_id)

# Check state
state = kernel.resources.get_state()
print(f"Utilization: {state.utilization}")
```

---

## Council API

```python
# Manual audit
report = kernel.council.manual_audit(
    substrate_state={"tick": 100, "n_attractors": 5},
    observations=["Manual inspection"]
)
print(f"Recommendations: {report.recommendations}")

# Get recent reports
reports = kernel.council.get_reports(n=5)
for r in reports:
    print(f"{r.audit_type.value}: {r.severity.value}")

# Council state
state = kernel.council.get_state()
print(f"Health: {state.health_score}")
print(f"Drift: {state.drift_score}")
```

---

## Configuration

### KernelConfig

```python
from substrate_echo.kernel import KernelConfig

config = KernelConfig(
    dim=16,                    # State dimension
    model_type="global",       # Dynamics model type
    min_samples_for_fit=50,    # Min samples before fitting
    max_samples=4000,          # Max buffer size
    attractor_radius=0.12,     # Clustering radius
    attractor_min_cluster=6,   # Min cluster size
    convergence_window=40,     # Convergence detection window
    correlation_threshold=0.25,# Abstraction correlation threshold
    meta_sigma=0.5,            # Meta-attractor spread
    coupling_strength=0.3,     # Goal-to-dynamics coupling
    total_energy=10.0,         # Initial cognitive energy
    topology_interval=200,     # Topology snapshot interval
)

kernel = SubstrateKernel(config)
```

---

## Error Handling

### REST Errors

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid request body |
| 404 | Endpoint not found |
| 422 | Validation error |
| 500 | Internal error |

### WebSocket Errors

- Connection refused: Kernel not running
- JSON parse error: Invalid message format
- Missing field: Required field not in message

---

## Examples

### Minimal Working Example

```python
from substrate_echo.kernel import SubstrateKernel, Observation
import numpy as np

kernel = SubstrateKernel()

for i in range(100):
    obs = Observation(
        vector=np.random.uniform(0.2, 0.8, 16).tolist(),
        embodiment_id="demo"
    )
    state = kernel.publish_observation(obs)
    print(f"Tick {state.tick}: coherence={state.coherence:.3f}")
```

### Multi-Embodiment Example

```python
from substrate_echo.kernel import SubstrateKernel, Observation
from substrate_echo.kernel.resources import ResourceRequest, ResourceTier

kernel = SubstrateKernel()

# Desktop
for i in range(50):
    kernel.publish_observation(Observation(
        vector=np.random.uniform(0.3, 0.7, 16).tolist(),
        embodiment_id="desktop"
    ))

# Robot (shares same kernel)
for i in range(50):
    kernel.publish_observation(Observation(
        vector=np.random.uniform(0.3, 0.7, 16).tolist(),
        embodiment_id="robot"
    ))

# Both share attractor landscape
print(f"Attractors: {kernel.get_snapshot()['n_attractors']}")
print(f"Embodiments: {kernel.get_snapshot()['n_embodiments']}")
```
