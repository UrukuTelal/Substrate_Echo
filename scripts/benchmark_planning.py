"""Benchmark: Reactive vs Predictive Agents.

Agent A (Reactive): AttractorMemory, picks action from agent ecology consensus
Agent B (Predictive): DynamicsMemory + WorldModel + Planner, simulates futures

Measures:
1. Task completion: reach a target attractor from a random start
2. Recovery: bounce back after a perturbation mid-task
3. Adaptation: goal changes mid-task
4. Planning overhead: time cost of simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.core.world_model import WorldModel
from substrate_echo.core.simulator import Simulator, ActionDelta
from substrate_echo.core.evaluator import Evaluator, UtilityWeights
from substrate_echo.core.controller import Controller
from substrate_echo.core.planner import Planner
from substrate_echo.core.intent import Intent, IntentProposal
from substrate_echo.models.experience import Experience, ExperienceType


# ── Shared Environment ──────────────────────────────────────────

class SimpleEnvironment:
    """A multi-basin dynamics environment with attractors and a danger zone."""
    
    def __init__(self, dim=16, n_basins=4, seed=42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
        # Fixed attractor locations with distinct positions in each dimension
        self.attractors = np.array([
            [0.2] * dim,
            [0.8] * dim,
            [0.2 if d % 2 == 0 else 0.8 for d in range(dim)],
            [0.8 if d % 3 == 0 else 0.2 for d in range(dim)],
        ][:n_basins])
        
        # Danger zone center: midpoint between first two attractors
        # States near here get a penalty (high harm)
        self.danger_center = 0.5 * np.ones(dim)
        self.danger_radius = 0.3
        
        # Linear dynamics: V(x) = A(x - nearest_attractor)
        self.A = -0.3 * np.eye(dim)
        self.dt = 1.0  # match simulator dt
    
    def dynamics(self, state):
        """True dynamics: flow toward nearest attractor."""
        dists = np.linalg.norm(self.attractors - state, axis=1)
        nearest = self.attractors[np.argmin(dists)]
        return self.A @ (state - nearest)
    
    def step(self, state, dt=None):
        """One dynamics step."""
        if dt is None:
            dt = self.dt
        v = self.dynamics(state)
        next_state = state + dt * v
        return np.clip(next_state, 0.0, 1.0)
    
    def get_basin(self, state):
        """Return basin ID (index of nearest attractor)."""
        dists = np.linalg.norm(self.attractors - state, axis=1)
        return int(np.argmin(dists))
    
    def is_at_target(self, state, target_basin, tolerance=0.25):
        """Check if state is within tolerance of target attractor."""
        target = self.attractors[target_basin]
        return np.linalg.norm(state - target) < tolerance
    
    def in_danger(self, state):
        """Check if state is in the danger zone."""
        return np.linalg.norm(state - self.danger_center) < self.danger_radius
    
    def danger_penalty(self, state):
        """Penalty for being in danger zone. Returns 0 if safe."""
        dist = np.linalg.norm(state - self.danger_center)
        if dist < self.danger_radius:
            return (self.danger_radius - dist) / self.danger_radius
        return 0.0


def generate_training_data(env, n_trajectories=20, steps_per=40, seed=42):
    """Generate diverse trajectories through the environment.
    
    IMPORTANT: training step dt must match simulator dt (1.0)
    so that learned velocities are in the correct scale.
    """
    rng = np.random.RandomState(seed)
    experiences = []
    
    for traj in range(n_trajectories):
        # Start from random state
        state = rng.uniform(0.1, 0.9, size=env.dim)
        for step in range(steps_per):
            experiences.append(state.copy())
            v = env.dynamics(state)
            state = np.clip(state + env.dt * v + rng.randn(env.dim) * 0.01, 0.0, 1.0)
    
    return experiences


def train_memory(memory, experiences):
    """Train a memory system on experiences."""
    for i, state in enumerate(experiences):
        exp = Experience(
            experience_id=f"train_{i:04d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        memory.encode(exp)


# ── Agent A: Reactive ──────────────────────────────────────────

class ReactiveAgent:
    """AttractorMemory only. Picks actions based on recall + simple heuristic.
    
    Myopic: no look-ahead, no simulation. Just "recall nearest, move toward it."
    """
    
    def __init__(self, memory, env):
        self.memory = memory
        self.env = env
    
    def act(self, state, target_basin):
        """Simple reactive: move toward nearest recalled attractor of target type."""
        recalled = self.memory.recall(state, k=3)
        
        if not recalled:
            # No memories: random perturbation
            return ActionDelta.random(len(state), magnitude=0.1)
        
        # Try to find a recalled attractor near the target
        target = self.env.attractors[target_basin]
        
        best = None
        best_dist = float('inf')
        for trace in recalled:
            center = np.array(trace.attractor_center)
            dist = np.linalg.norm(center - target)
            if dist < best_dist:
                best_dist = dist
                best = center
        
        if best is not None and best_dist < 0.5:
            # Move toward recalled target attractor
            delta = best - state
            mag = np.linalg.norm(delta)
            if mag > 0.01:
                delta = delta * min(0.15, mag) / mag
                return ActionDelta(delta=delta, description="recall_directed")
        
        # Fallback: random
        return ActionDelta.random(len(state), magnitude=0.1)
    
    def observe(self, state):
        """Learn from observation."""
        exp = Experience(
            experience_id=f"obs_{id(state)}_{time.time_ns()}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        self.memory.encode(exp)


# ── Agent B: Predictive ────────────────────────────────────────

class PredictiveAgent:
    """DynamicsMemory + WorldModel + Planner. Simulates before acting."""
    
    def __init__(self, memory, env):
        self.memory = memory
        self.env = env
        self.world_model = WorldModel(memory)
        self.simulator = Simulator(self.world_model)
        self.evaluator = Evaluator(UtilityWeights(), self.world_model)
        self.controller = Controller()
        self.planner = Planner(self.simulator, self.evaluator, self.controller)
    
    def act(self, state, target_basin):
        """Plan an action that moves toward target basin."""
        target_state = self.env.attractors[target_basin]
        
        intent = IntentProposal(
            intent=Intent.EXPLORE,
            priority=0.8,
            confidence=0.7,
            reasoning=f"reach basin {target_basin}",
            target_state=target_state,
        )
        
        plan = self.planner.plan(state, intent)
        
        if plan.actions:
            return plan.actions[0]
        
        # Fallback
        return ActionDelta.toward_target(target_state, state, max_magnitude=0.15)
    
    def observe(self, state):
        """Learn from observation."""
        exp = Experience(
            experience_id=f"obs_{id(state)}_{time.time_ns()}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        self.memory.encode(exp)


# ── Experiments ──────────────────────────────────────────────────

def run_episode(agent, env, start_state, target_basin, max_steps=60,
                perturbation_step=None, perturbation_delta=None,
                goal_change_step=None, new_target_basin=None):
    """Run a single episode and return metrics.
    
    Returns:
        dict with steps_to_target, final_distance, reached, danger_encounters, etc.
    """
    state = start_state.copy()
    step_times = []
    distances = []
    danger_encounters = 0
    cumulative_danger = 0.0
    
    for step in range(max_steps):
        t0 = time.time()
        
        # Act
        action = agent.act(state, target_basin)
        step_times.append(time.time() - t0)
        
        # Apply action
        state = np.clip(state + action.delta, 0.0, 1.0)
        
        # Environment dynamics
        state = env.step(state)
        
        # Learn
        agent.observe(state)
        
        # Record distance
        dist = np.linalg.norm(state - env.attractors[target_basin])
        distances.append(dist)
        
        # Track danger
        penalty = env.danger_penalty(state)
        cumulative_danger += penalty
        if penalty > 0.1:
            danger_encounters += 1
        
        # Mid-task perturbation
        if perturbation_step is not None and step == perturbation_step and perturbation_delta is not None:
            state = np.clip(state + perturbation_delta, 0.0, 1.0)
        
        # Goal change
        if goal_change_step is not None and step == goal_change_step and new_target_basin is not None:
            target_basin = new_target_basin
        
        # Check success
        if env.is_at_target(state, target_basin):
            return {
                "steps_to_target": step + 1,
                "final_distance": float(dist),
                "reached": True,
                "danger_encounters": danger_encounters,
                "cumulative_danger": cumulative_danger,
                "distances": distances,
                "step_times": step_times,
                "perturbation_step": perturbation_step,
                "goal_change_step": goal_change_step,
                "new_target_basin": new_target_basin,
            }
    
    return {
        "steps_to_target": max_steps,
        "final_distance": float(distances[-1]) if distances else float('inf'),
        "reached": False,
        "danger_encounters": danger_encounters,
        "cumulative_danger": cumulative_danger,
        "distances": distances,
        "step_times": step_times,
        "perturbation_step": perturbation_step,
        "goal_change_step": goal_change_step,
        "new_target_basin": new_target_basin,
    }


def experiment_task_completion(n_episodes=8, dim=16, n_basins=3):
    """Exp 1: Can the agent reach a target attractor from a random start?"""
    env = SimpleEnvironment(dim=dim, n_basins=n_basins)
    training = generate_training_data(env, n_trajectories=20, steps_per=40)
    
    rng = np.random.RandomState(99)
    
    # Agent A: reactive
    mem_a = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_a, training)
    agent_a = ReactiveAgent(mem_a, env)
    
    # Agent B: predictive (global linear)
    mem_b = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_b, training)
    agent_b = PredictiveAgent(mem_b, env)
    
    # Agent C: predictive (local linear) — reduced planner + memory budget
    # k-NN is slow in 16D; this tests whether local model can still help despite limitations
    mem_c = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(
        model_type="local", min_samples_for_fit=20, k_neighbors=3, bandwidth=1.0))
    train_memory(mem_c, training)
    agent_c = PredictiveAgent(mem_c, env)
    agent_c.planner.config.n_candidates = 3
    agent_c.planner.config.lookahead_steps = 2
    agent_c.planner.config.beam_width = 1
    
    results_a = []
    results_b = []
    results_c = []
    
    for ep in range(n_episodes):
        start = rng.uniform(0.1, 0.9, size=dim)
        target = ep % n_basins
        
        r_a = run_episode(agent_a, env, start, target)
        r_b = run_episode(agent_b, env, start, target)
        r_c = run_episode(agent_c, env, start, target)
        
        results_a.append(r_a)
        results_b.append(r_b)
        results_c.append(r_c)
    
    return {
        "reactive": {
            "reached": sum(r["reached"] for r in results_a),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_a]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_a]),
            "avg_planning_ms": np.mean([np.mean(r["step_times"]) * 1000 for r in results_a]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_a]),
        },
        "predictive_global": {
            "reached": sum(r["reached"] for r in results_b),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_b]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_b]),
            "avg_planning_ms": np.mean([np.mean(r["step_times"]) * 1000 for r in results_b]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_b]),
        },
        "predictive_local": {
            "reached": sum(r["reached"] for r in results_c),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_c]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_c]),
            "avg_planning_ms": np.mean([np.mean(r["step_times"]) * 1000 for r in results_c]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_c]),
        },
    }


def experiment_recovery(n_episodes=6, dim=16, n_basins=3):
    """Exp 2: Agent reaches target, gets perturbed, must recover."""
    env = SimpleEnvironment(dim=dim, n_basins=n_basins)
    training = generate_training_data(env, n_trajectories=20, steps_per=40)
    
    rng = np.random.RandomState(77)
    
    mem_a = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_a, training)
    agent_a = ReactiveAgent(mem_a, env)
    
    mem_b = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_b, training)
    agent_b = PredictiveAgent(mem_b, env)
    
    results_a = []
    results_b = []
    
    for ep in range(n_episodes):
        start = rng.uniform(0.1, 0.9, size=dim)
        target = ep % n_basins
        perturbation = rng.randn(dim) * 0.3
        
        r_a = run_episode(agent_a, env, start, target,
                          perturbation_step=10, perturbation_delta=perturbation)
        r_b = run_episode(agent_b, env, start, target,
                          perturbation_step=10, perturbation_delta=perturbation)
        
        results_a.append(r_a)
        results_b.append(r_b)
    
    return {
        "reactive": {
            "reached": sum(r["reached"] for r in results_a),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_a]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_a]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_a]),
        },
        "predictive": {
            "reached": sum(r["reached"] for r in results_b),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_b]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_b]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_b]),
        },
    }


def experiment_adaptation(n_episodes=6, dim=16, n_basins=3):
    """Exp 3: Goal changes mid-task. Agent must adapt."""
    env = SimpleEnvironment(dim=dim, n_basins=n_basins)
    training = generate_training_data(env, n_trajectories=20, steps_per=40)
    
    rng = np.random.RandomState(55)
    
    mem_a = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_a, training)
    agent_a = ReactiveAgent(mem_a, env)
    
    mem_b = DynamicsMemory(dim=dim, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem_b, training)
    agent_b = PredictiveAgent(mem_b, env)
    
    results_a = []
    results_b = []
    
    for ep in range(n_episodes):
        start = rng.uniform(0.1, 0.9, size=dim)
        target1 = ep % n_basins
        target2 = (ep + 1) % n_basins  # change goal
        
        r_a = run_episode(agent_a, env, start, target1,
                          goal_change_step=15, new_target_basin=target2)
        r_b = run_episode(agent_b, env, start, target1,
                          goal_change_step=15, new_target_basin=target2)
        
        results_a.append(r_a)
        results_b.append(r_b)
    
    return {
        "reactive": {
            "reached": sum(r["reached"] for r in results_a),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_a]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_a]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_a]),
        },
        "predictive": {
            "reached": sum(r["reached"] for r in results_b),
            "total": n_episodes,
            "avg_steps": np.mean([r["steps_to_target"] for r in results_b]),
            "avg_final_dist": np.mean([r["final_distance"] for r in results_b]),
            "avg_danger": np.mean([r["danger_encounters"] for r in results_b]),
        },
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("BENCHMARK: Reactive (AttractorMemory) vs Predictive (Planner)")
    print("=" * 65)
    
    # Exp 1: Task completion
    print("\n--- Experiment 1: Task Completion ---")
    r1 = experiment_task_completion(n_episodes=8)
    print(f"  Reactive:       {r1['reactive']['reached']}/{r1['reactive']['total']} reached, "
          f"avg {r1['reactive']['avg_steps']:.1f} steps, "
          f"avg dist {r1['reactive']['avg_final_dist']:.3f}, "
          f"act {r1['reactive']['avg_planning_ms']:.1f}ms")
    print(f"  Predictive(g):  {r1['predictive_global']['reached']}/{r1['predictive_global']['total']} reached, "
          f"avg {r1['predictive_global']['avg_steps']:.1f} steps, "
          f"avg dist {r1['predictive_global']['avg_final_dist']:.3f}, "
          f"act {r1['predictive_global']['avg_planning_ms']:.1f}ms")
    print(f"  Predictive(l):  {r1['predictive_local']['reached']}/{r1['predictive_local']['total']} reached, "
          f"avg {r1['predictive_local']['avg_steps']:.1f} steps, "
          f"avg dist {r1['predictive_local']['avg_final_dist']:.3f}, "
          f"act {r1['predictive_local']['avg_planning_ms']:.1f}ms")
    
    # Exp 2: Recovery
    print("\n--- Experiment 2: Recovery from Perturbation ---")
    r2 = experiment_recovery(n_episodes=6)
    print(f"  Reactive:   {r2['reactive']['reached']}/{r2['reactive']['total']} recovered, "
          f"avg {r2['reactive']['avg_steps']:.1f} steps, "
          f"avg dist {r2['reactive']['avg_final_dist']:.3f}, "
          f"danger {r2['reactive']['avg_danger']:.1f}")
    print(f"  Predictive: {r2['predictive']['reached']}/{r2['predictive']['total']} recovered, "
          f"avg {r2['predictive']['avg_steps']:.1f} steps, "
          f"avg dist {r2['predictive']['avg_final_dist']:.3f}, "
          f"danger {r2['predictive']['avg_danger']:.1f}")
    
    # Exp 3: Adaptation
    print("\n--- Experiment 3: Goal Adaptation ---")
    r3 = experiment_adaptation(n_episodes=6)
    print(f"  Reactive:   {r3['reactive']['reached']}/{r3['reactive']['total']} adapted, "
          f"avg {r3['reactive']['avg_steps']:.1f} steps, "
          f"avg dist {r3['reactive']['avg_final_dist']:.3f}, "
          f"danger {r3['reactive']['avg_danger']:.1f}")
    print(f"  Predictive: {r3['predictive']['reached']}/{r3['predictive']['total']} adapted, "
          f"avg {r3['predictive']['avg_steps']:.1f} steps, "
          f"avg dist {r3['predictive']['avg_final_dist']:.3f}, "
          f"danger {r3['predictive']['avg_danger']:.1f}")
    
    # Summary
    print("\n--- Summary ---")
    a_total = r1["reactive"]["reached"] + r2["reactive"]["reached"] + r3["reactive"]["reached"]
    bg_total = r1["predictive_global"]["reached"] + r2["predictive"]["reached"] + r3["predictive"]["reached"]
    bl_total = r1["predictive_local"]["reached"]
    a_all = r1["reactive"]["total"] + r2["reactive"]["total"] + r3["reactive"]["total"]
    bg_all = r1["predictive_global"]["total"] + r2["predictive"]["total"] + r3["predictive"]["total"]
    bl_all = r1["predictive_local"]["total"]
    print(f"  Reactive overall:       {a_total}/{a_all} ({a_total/a_all:.0%})")
    print(f"  Predictive(global):     {bg_total}/{bg_all} ({bg_total/bg_all:.0%})")
    print(f"  Predictive(local):      {bl_total}/{bl_all} ({bl_total/bl_all:.0%})")
    
    avg_planning_g = r1["predictive_global"]["avg_planning_ms"]
    avg_planning_l = r1["predictive_local"]["avg_planning_ms"]
    avg_reactive_ms = r1["reactive"]["avg_planning_ms"]
    print(f"  Planning overhead: global {avg_planning_g:.1f}ms, local {avg_planning_l:.1f}ms "
          f"(reactive {avg_reactive_ms:.1f}ms)")
    
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
