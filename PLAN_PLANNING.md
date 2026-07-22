# Plan: Model-Based Cognition — From Prediction to Decision

## Architecture Overview

```
Intent (symbolic goals)
    ↓
Planner (produces desired PSV trajectory)
    ↓
Controller (computes feasible ΔPSV)
    ↓
Physics (field evolution)
    ↓
DynamicsMemory (learns V(x, u, c))
    ↓
Evaluator (scores outcomes via pillar utility)
    ↓
Back to Planner
```

### Three-Layer Control

| Layer | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| **Intent** | Agent personality + situation | Goal enum (Explore, Cooperate, Defend, etc.) | What to achieve |
| **Planner** | Intent + current PSV + WorldModel | Desired PSV trajectory | Where to go |
| **Controller** | Desired PSV + current PSV + constraints | ΔPSV vector | How to get there feasibly |

### Why three layers?

- DynamicsMemory learns passive evolution V(x). It never needs to know about actions.
- The controller injects control u, making the learned system V(x, u, c) — state + control + context.
- Intent stays symbolic (no PSV leakage into goal representation).
- The planner can simulate hypothetical futures without committing to physical actions.

---

## Phase 0: Validation Experiment (prove the premise)

Before building the full architecture, run a controlled experiment.

### Experiment: Reactive vs Predictive Agents

Two identical agents in the same environment:

**Agent A — Reactive (AttractorMemory only)**
- Recalls nearest attractors
- Selects action based on recall similarity + pillar activation
- No lookahead

**Agent B — Predictive (DynamicsMemory + Simulation)**
- Learns V(x) from experience
- For each candidate action: simulates outcome, evaluates via pillar utility
- Selects action with highest predicted utility

### Metrics (over 1000+ episodes)

| Metric | What it measures |
|--------|-----------------|
| Task completion rate | Goal achievement |
| Mean prediction error | World model quality |
| Recovery time after perturbation | Resilience |
| Catastrophic failure count | Safety |
| Adaptation speed (new environment) | Generalization |
| Basin transition success rate | Dynamics awareness |

### Implementation

File: `scripts/benchmark_planning.py`

```
- Create two CognitiveLoop instances
- Agent A: AttractorMemory, standard action selection
- Agent B: DynamicsMemory, simulation-based action selection
- Run identical scenarios (perturbations, goal pursuit, environment changes)
- Compare metrics
- Save results to planning_benchmark_results.json
```

---

## Phase 1: Core Components

### 1.1 WorldModel

File: `substrate_echo/core/world_model.py`

Wraps DynamicsMemory + environment context. Answers: "What happens if nothing intervenes?"

```python
class WorldModel:
    def __init__(self, dynamics_memory: DynamicsMemory, dim: int = 16)
    
    # Passive prediction
    def predict(self, state: np.ndarray, steps: int = 10) -> np.ndarray
    def predict_trajectory(self, state: np.ndarray, steps: int = 50) -> list[np.ndarray]
    
    # Structure discovery
    def get_attractors(self) -> list[np.ndarray]
    def get_basin(self, state: np.ndarray) -> int
    def get_stability(self, state: np.ndarray) -> dict
    
    # Transition reasoning
    def transition_probability(self, from_basin: int, steps: int = 100) -> dict[int, float]
    def would_transition(self, state: np.ndarray, target_basin: int) -> float
    
    # Uncertainty
    def prediction_confidence(self, state: np.ndarray) -> float
    def coverage(self, state: np.ndarray) -> float  # how well-known is this region
```

### 1.2 Simulator

File: `substrate_echo/core/simulator.py`

Takes (state, action) → predicted future. Models interventions, not just passive evolution.

```python
@dataclass
class SimConfig:
    prediction_horizon: int = 50     # steps to simulate
    dt: float = 1.0                  # per-step (matches DynamicsMemory velocity scale)
    n_ensemble: int = 1              # ensemble predictions for uncertainty
    noise_scale: float = 0.0         # process noise per step

class Simulator:
    def __init__(self, world_model: WorldModel, config: SimConfig = None)
    
    # Core simulation
    def simulate(self, state: np.ndarray, action: ActionDelta, 
                 steps: int = None) -> SimResult
    
    # Batch simulation (for planner)
    def simulate_batch(self, state: np.ndarray, 
                       actions: list[ActionDelta]) -> list[SimResult]
    
    # With control dynamics
    def simulate_controlled(self, state: np.ndarray, 
                            target_psv: np.ndarray,
                            controller: Controller) -> SimResult

@dataclass
class SimResult:
    final_state: np.ndarray          # PSV after simulation
    trajectory: list[np.ndarray]     # full state trajectory
    basin_transitions: list[int]     # which basins were visited
    stability: str                   # final stability classification
    confidence: float                # prediction confidence

@dataclass
class ActionDelta:
    """A PSV-level action: additive perturbation to pillar state."""
    delta: np.ndarray                # 16D perturbation vector
    source_action: Optional[ActionType] = None  # high-level origin
    description: str = ""
    magnitude: float = 0.0           # |delta| for scaling
```

The simulator applies action as:
```python
x = state + action.delta  # apply control
for _ in range(steps):
    v = world_model.dynamics_memory.predict_velocity(x)
    x = x + dt * v
    x = clip(x, 0, 1)
```

### 1.3 Evaluator

File: `substrate_echo/core/evaluator.py`

Scores a future PSV using pillar-based utility. Different agents weight pillars differently.

```python
@dataclass
class UtilityWeights:
    """Per-agent utility weights over 16 pillars."""
    # Named weights (map to pillar indices)
    integrity: float = 1.0      # [5] internal coherence
    memory: float = 0.5         # [10] retention capacity
    harm: float = -2.0          # [12] damage (negative = avoid)
    distortion: float = -1.0    # [13] deviation (negative = avoid)
    awareness: float = 0.3      # [0] perceptual openness
    willpower: float = 0.4      # [1] goal-directed energy
    cohesion: float = 0.6       # [6] binding force
    relation: float = 0.3       # [7] social connection
    depth: float = 0.2          # [15] processing complexity
    
    # Meta-utility signals
    stability_weight: float = 0.5   # prefer stable attractors
    novelty_weight: float = 0.3     # prefer novel states
    information_weight: float = 0.2  # prefer informative states
    
    # Task-specific
    task_reward: float = 0.0    # external reward signal

@dataclass  
class EvalResult:
    utility: float               # scalar utility
    pillar_scores: np.ndarray   # per-pillar utility contribution
    meta_scores: dict           # stability, novelty, information scores
    breakdown: dict             # human-readable breakdown

class Evaluator:
    def __init__(self, weights: UtilityWeights, world_model: WorldModel)
    
    def evaluate(self, state: np.ndarray) -> EvalResult
    def evaluate_trajectory(self, trajectory: list[np.ndarray]) -> EvalResult
    def evaluate_action(self, sim_result: SimResult) -> EvalResult
    
    # Meta-utility components
    def _compute_stability_score(self, state: np.ndarray) -> float
    def _compute_novelty_score(self, state: np.ndarray, 
                                memory: DynamicsMemory) -> float
    def _compute_information_score(self, state: np.ndarray,
                                    world_model: WorldModel) -> float
```

Utility computation:
```python
# Pillar utility
pillar_utility = sum(w_i * pillar_i for w_i, pillar_i in zip(weights, state))

# Meta utility
stability = world_model.get_stability(state)['max_eigenvalue_real']  # more negative = more stable
novelty = distance_to_nearest_recalled_state  # how different from past
information = -prediction_confidence  # uncertain = informative

# Combined
U = pillar_utility + w_s * stability + w_n * novelty + w_i * information + task_reward
```

### 1.4 Controller

File: `substrate_echo/core/controller.py`

Computes feasible ΔPSV from desired target, subject to constraints.

```python
@dataclass
class ControlConfig:
    gain: float = 0.3             # proportional gain (K in KP control)
    max_delta: float = 0.2        # max |ΔPSV| per step
    energy_budget: float = 0.1    # max energy per action
    conservation_enforce: bool = True
    topology_enforce: bool = True

class Controller:
    def __init__(self, config: ControlConfig = None)
    
    def compute_control(self, current: np.ndarray, 
                        target: np.ndarray) -> ActionDelta
    
    def compute_trajectory_control(self, current: np.ndarray,
                                    target_trajectory: list[np.ndarray]
                                    ) -> list[ActionDelta]
```

Control law:
```python
def compute_control(self, current, target):
    error = target - current
    delta = self.config.gain * error
    
    # Clamp magnitude
    mag = np.linalg.norm(delta)
    if mag > self.config.max_delta:
        delta = delta * (self.config.max_delta / mag)
    
    # Enforce conservation
    if self.config.conservation_enforce:
        delta = self._project_onto_conservation_manifold(delta, current)
    
    # Enforce topology
    if self.config.topology_enforce:
        delta = self._enforce_topology_constraints(delta, current)
    
    return ActionDelta(delta=delta, magnitude=np.linalg.norm(delta))
```

---

## Phase 2: Planner

File: `substrate_echo/core/planner.py`

Model-based planning: simulate → evaluate → select best.

```python
@dataclass
class PlannerConfig:
    n_candidates: int = 20        # candidate actions per planning step
    lookahead_steps: int = 10     # planning horizon
    replan_interval: int = 5      # ticks between replanning
    exploration_rate: float = 0.1 # epsilon-greedy exploration
    beam_width: int = 5           # beam search width

class Planner:
    def __init__(self, simulator: Simulator, evaluator: Evaluator,
                 controller: Controller, config: PlannerConfig = None)
    
    # Core planning
    def plan(self, state: np.ndarray, intent: Intent) -> Plan
    def replan(self, state: np.ndarray, current_plan: Plan) -> Plan
    
    # Candidate generation
    def generate_candidates(self, state: np.ndarray, 
                            intent: Intent) -> list[ActionDelta]
    
    # Basin-aware planning
    def plan_to_basin(self, state: np.ndarray, 
                      target_basin: int) -> Plan
    
    # Safety
    def is_safe(self, state: np.ndarray) -> bool
    def find_safe_action(self, state: np.ndarray) -> ActionDelta

@dataclass
class Plan:
    intent: Intent
    action_sequence: list[ActionDelta]
    predicted_trajectory: list[np.ndarray]
    predicted_utilities: list[float]
    predicted_basins: list[int]
    confidence: float
    total_utility: float

class Intent(Enum):
    EXPLORE = auto()
    COOPERATE = auto()
    DEFEND = auto()
    LEARN = auto()
    INVESTIGATE = auto()
    INCREASE_COHESION = auto()
    REDUCE_DISTORTION = auto()
    MAINTAIN_STABILITY = auto()
    SEEK_NOVELTY = auto()
    AVOID_HARM = auto()
```

Planning algorithm (simplified beam search):
```python
def plan(self, state, intent):
    candidates = self.generate_candidates(state, intent)
    
    best_plans = []
    for action in candidates:
        sim_result = self.simulator.simulate(state, action)
        eval_result = self.evaluator.evaluate_action(sim_result)
        best_plans.append((eval_result.utility, action, sim_result))
    
    best_plans.sort(reverse=True)
    
    # Beam search: expand top beam_width candidates
    final_plans = []
    for _, action, sim_result in best_plans[:self.config.beam_width]:
        # Extend trajectory with further actions
        extended = self._extend_plan(state, action, intent)
        final_plans.append(extended)
    
    return max(final_plans, key=lambda p: p.total_utility)
```

---

## Phase 3: Intent System

File: `substrate_echo/core/intent.py`

Maps agent personality + situation to symbolic goals.

```python
@dataclass
class AgentPersonality:
    """Defines agent's intrinsic drives via utility weight vector."""
    exploration_drive: float = 0.3    # tendency to seek novelty
    safety_drive: float = 0.5         # tendency to avoid harm
    social_drive: float = 0.3         # tendency to cooperate
    achievement_drive: float = 0.4    # tendency to complete tasks
    curiosity_drive: float = 0.2      # tendency to investigate uncertainty

class IntentGenerator:
    def __init__(self, personality: AgentPersonality, world_model: WorldModel)
    
    def generate_intent(self, state: np.ndarray, 
                        situation: Situation) -> Intent
    
    def _assess_situation(self, state: np.ndarray) -> Situation
    
    def _priority_weighted_choice(self, state: np.ndarray,
                                   intents: list[Intent]) -> Intent

class Situation(Enum):
    STABLE = auto()        # in attractor, nothing happening
    UNSTABLE = auto()      # near saddle/repellor
    NOVEL = auto()         # unfamiliar state
    THREATENED = auto()    # harm/distortion high
    SOCIAL = auto()        # other agents nearby
    OPPORTUNITY = auto()   # favorable conditions
```

---

## Phase 4: Integration

### 4.1 Updated Cognitive Loop

File: `substrate_echo/core/cognitive_loop.py` (modify existing)

Replace the current reactive flow with model-based cognition:

```python
# Old flow:
# recall → evolve → perceive → act → learn

# New flow:
# recall → evolve → perceive → intent → plan → control → act → learn
```

The `tick()` method becomes:
```python
def tick(self, field_evolver, memory_system, agent_ecology, world_model=None):
    # 1. Recall memories
    recalled = self.recall_memories(memory_system, self._pillar_state)
    memory_influence = self.compute_memory_influence(recalled, self._field_state)
    
    # 2. Evolve field
    self.evolve_field(field_evolver, memory_influence, steps=self.config.max_steps_per_tick)
    
    # 3. Update pillars
    self._pillar_state = self._project_to_pillars(self._field_state)
    
    # 4. Perceive
    responses = self.perceive(agent_ecology, world_model, memory_system)
    
    # 5. NEW: Generate intent
    intent = self.intent_generator.generate_intent(self._pillar_state, situation)
    
    # 6. NEW: Plan (simulate → evaluate → select)
    plan = self.planner.plan(self._pillar_state, intent)
    
    # 7. NEW: Control (compute feasible ΔPSV)
    action_delta = self.controller.compute_control(
        self._pillar_state, plan.action_sequence[0].delta + self._pillar_state
    )
    
    # 8. Act
    action = self.select_action(responses, agent_ecology, plan)
    
    # 9. Learn
    self.learn(memory_system, action_result=action)
    
    # 10. Consume energy
    self._energy -= self.config.cognitive_energy_cost
    
    return {
        "field_state": self._field_state,
        "pillar_state": self._pillar_state,
        "action": action,
        "intent": intent.name if intent else "none",
        "plan_confidence": plan.confidence if plan else 0.0,
        "stats": self.stats(),
    }
```

### 4.2 Updated End-to-End Scenario

File: `substrate_echo/scenarios/end_to_end.py` (modify existing)

Wire in the full planning stack:
```python
def _setup_components(self):
    # ... existing components ...
    
    # NEW: Planning stack
    self.dynamics_memory = DynamicsMemory(dim=16)
    self.world_model = WorldModel(self.dynamics_memory)
    self.simulator = Simulator(self.world_model)
    self.evaluator = Evaluator(UtilityWeights(), self.world_model)
    self.controller = Controller()
    self.planner = Planner(self.simulator, self.evaluator, self.controller)
    self.intent_generator = IntentGenerator(AgentPersonality(), self.world_model)
```

---

## Phase 5: Intrinsic Drives & Personality

Multiple utility signals create agent personality without changing physics.

### UtilityWeights as personality

Different agents have different weight vectors:

```python
# Cautious agent
cautious = UtilityWeights(
    integrity=2.0, harm=-3.0, distortion=-2.0,
    stability_weight=1.0, novelty_weight=0.0, exploration_drive=0.0
)

# Curious agent
curious = UtilityWeights(
    integrity=0.5, harm=-1.0, distortion=0.5,  # tolerate distortion
    stability_weight=0.0, novelty_weight=1.0, information_weight=1.0
)

# Social agent
social = UtilityWeights(
    relation=2.0, cohesion=1.5, warmth=1.0,
    stability_weight=0.3, novelty_weight=0.2
)

# Achievement agent
achiever = UtilityWeights(
    willpower=2.0, force=1.5, memory=1.0,
    task_reward=3.0, stability_weight=0.5
)
```

### Multiple intrinsic drives

The evaluator combines multiple signals:

```
U = w_p * pillar_utility     # pillar-based
  + w_s * stability           # prefer stable states
  + w_n * novelty             # prefer novel states  
  + w_i * information         # prefer informative states (curiosity)
  + w_r * task_reward         # external reward
  + w_c * coherence           # prefer coherent states
```

Each agent has different w_* weights. No new code — only different weight vectors.

---

## Implementation Order

| Step | Component | Files | Depends on |
|------|-----------|-------|------------|
| 0 | Validation experiment | `scripts/benchmark_planning.py` | DynamicsMemory (exists) |
| 1.1 | WorldModel | `core/world_model.py` | DynamicsMemory |
| 1.2 | Simulator + ActionDelta | `core/simulator.py` | WorldModel |
| 1.3 | Evaluator + UtilityWeights | `core/evaluator.py` | WorldModel |
| 1.4 | Controller | `core/controller.py` | — |
| 2 | Planner + Intent | `core/planner.py`, `core/intent.py` | Simulator, Evaluator, Controller |
| 3 | Updated CognitiveLoop | `core/cognitive_loop.py` | Planner, Intent |
| 4 | Updated EndToEnd | `scenarios/end_to_end.py` | All above |
| 5 | Tests | `tests/test_planning.py` | All above |
| 6 | Benchmark | `scripts/benchmark_planning.py` | All above |

---

## Future Directions (after this plan)

1. **V(x, u, c)** — Extend DynamicsMemory to learn control-dependent and context-dependent dynamics
2. **Reinforcement learning** — Replace hand-tuned UtilityWeights with learned value functions
3. **Multi-agent planning** — Agents share world models, negotiate intents
4. **Geodesic planning** — Use learned metric tensor g_ij(x) for optimal paths through state space
5. **Counterfactual reasoning** — "What would have happened if I had done X?"
6. **Temporal abstraction** — Options/subgoals for multi-step plans
