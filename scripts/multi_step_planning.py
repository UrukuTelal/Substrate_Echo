"""Multi-Step Planning Demo: plan_sequence() and Basin Navigation.

Demonstrates:
1. Multi-step planning: chaining plan_sequence() to navigate through
   intermediate basins toward a distant target
2. Basin navigation: deliberate path planning through a known multi-basin
   landscape with obstacle avoidance

Uses the same SimpleEnvironment as benchmark_planning.py but focuses on
the planner's ability to decompose a long-distance transition into
manageable multi-step plans.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.core.world_model import WorldModel
from substrate_echo.core.simulator import Simulator, ActionDelta
from substrate_echo.core.evaluator import Evaluator, UtilityWeights
from substrate_echo.core.controller import Controller
from substrate_echo.core.planner import Planner
from substrate_echo.core.intent import Intent, IntentProposal
from substrate_echo.models.experience import Experience, ExperienceType


# ── Environment ──────────────────────────────────────────────────

class BasinEnvironment:
    """Multi-basin environment with 4 attractors and a danger zone.
    
    Layout (projected to 2D for visualization):
    
        Basin A (0.2, 0.2)          Basin B (0.8, 0.2)
              *                            *
                    \                  /
                     \   DANGER ZONE  /
                      \   (0.5,0.5)  /
                       \     *      /
                        \         /
                             *
        Basin C (0.2, 0.8)          Basin D (0.8, 0.8)
    
    Basin transitions require passing near the danger zone.
    """
    
    def __init__(self, dim=16, seed=42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
        self.attractors = np.array([
            [0.2] * dim,                                    # A: all-low
            [0.8] * dim,                                    # B: all-high
            [0.2 if d % 2 == 0 else 0.8 for d in range(dim)],  # C: alternating
            [0.8 if d % 3 == 0 else 0.2 for d in range(dim)],  # D: every-third
        ])
        
        self.basin_names = ["A (all-low)", "B (all-high)", "C (alternating)", "D (every-third)"]
        
        self.danger_center = 0.5 * np.ones(dim)
        self.danger_radius = 0.3
        
        self.A = -0.3 * np.eye(dim)
        self.dt = 1.0
    
    def dynamics(self, state):
        dists = np.linalg.norm(self.attractors - state, axis=1)
        nearest = self.attractors[np.argmin(dists)]
        return self.A @ (state - nearest)
    
    def step(self, state, dt=None):
        if dt is None:
            dt = self.dt
        v = self.dynamics(state)
        return np.clip(state + dt * v, 0.0, 1.0)
    
    def get_basin(self, state):
        dists = np.linalg.norm(self.attractors - state, axis=1)
        return int(np.argmin(dists))
    
    def danger_penalty(self, state):
        dist = np.linalg.norm(state - self.danger_center)
        if dist < self.danger_radius:
            return (self.danger_radius - dist) / self.danger_radius
        return 0.0


def generate_training(env, n_trajectories=30, steps_per=40, seed=42):
    """Generate diverse training trajectories."""
    rng = np.random.RandomState(seed)
    experiences = []
    for _ in range(n_trajectories):
        state = rng.uniform(0.1, 0.9, env.dim)
        for _ in range(steps_per):
            experiences.append(state.copy())
            v = env.dynamics(state)
            state = np.clip(state + env.dt * v + rng.randn(env.dim) * 0.01, 0.0, 1.0)
    return experiences


def train_memory(memory, experiences):
    for i, state in enumerate(experiences):
        exp = Experience(
            experience_id=f"train_{i:04d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        memory.encode(exp)


def dist_to_attractor(state, attractors, basin_idx):
    return float(np.linalg.norm(state - attractors[basin_idx]))


# ── Demo 1: Multi-Step Plan Sequence ────────────────────────────

def demo_plan_sequence():
    """Show how plan_sequence() decomposes a long transition into steps."""
    print("=" * 65)
    print("DEMO 1: Multi-Step Plan Sequence")
    print("=" * 65)
    print()
    print("Scenario: Agent at Basin A (all-low) wants to reach Basin D")
    print("          (every-third). Direct path crosses the danger zone.")
    print("          plan_sequence() decomposes into 3 tactical steps.")
    print()
    
    env = BasinEnvironment(dim=16)
    training = generate_training(env)
    
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem, training)
    
    wm = WorldModel(mem)
    sim = Simulator(wm)
    evaluator = Evaluator(UtilityWeights(), wm)
    controller = Controller()
    planner = Planner(sim, evaluator, controller)
    planner.config.n_candidates = 5
    planner.config.lookahead_steps = 3
    planner.config.beam_width = 2
    
    # Start at Basin A, target Basin D
    start = env.attractors[0] + 0.05 * np.random.RandomState(123).randn(16)
    start = np.clip(start, 0.1, 0.9)
    target_basin = 3  # Basin D
    
    print(f"  Start basin:  {env.basin_names[env.get_basin(start)]}")
    print(f"  Target basin: {env.basin_names[target_basin]}")
    print(f"  Start dist to target attractor: "
          f"{dist_to_attractor(start, env.attractors, target_basin):.3f}")
    print()
    
    intent = IntentProposal(
        intent=Intent.EXPLORE,
        priority=0.9,
        confidence=0.8,
        reasoning=f"navigate from Basin A to Basin D through multi-step plan",
        target_state=env.attractors[target_basin],
    )
    
    # Single-step plan
    t0 = time.time()
    single_plan = planner.plan(start, intent)
    single_ms = (time.time() - t0) * 1000
    
    # Multi-step plan (3 steps)
    t0 = time.time()
    multi_plan = planner.plan_sequence(start, intent, n_steps=3)
    multi_ms = (time.time() - t0) * 1000
    
    print(f"  --- Single-step plan ---")
    print(f"  Actions: {len(single_plan.actions)}")
    if single_plan.sim_results:
        final = single_plan.sim_results[-1].final_state
        d = dist_to_attractor(final, env.attractors, target_basin)
        print(f"  Final dist to D: {d:.3f}")
        print(f"  Utility: {single_plan.total_utility:.3f}")
    print(f"  Time: {single_ms:.0f}ms")
    print()
    
    print(f"  --- Multi-step plan (3 steps) ---")
    print(f"  Actions: {len(multi_plan.actions)}")
    if multi_plan.sim_results:
        final = multi_plan.sim_results[-1].final_state
        d = dist_to_attractor(final, env.attractors, target_basin)
        print(f"  Final dist to D: {d:.3f}")
        print(f"  Utility: {multi_plan.total_utility:.3f}")
        print()
        print("  Step-by-step trajectory:")
        state = start.copy()
        for i, (action, sim_result) in enumerate(zip(multi_plan.actions, multi_plan.sim_results)):
            basin_before = env.get_basin(state)
            basin_after = env.get_basin(sim_result.final_state)
            danger = env.danger_penalty(sim_result.final_state)
            print(f"    Step {i+1}: Basin {env.basin_names[basin_before][0]}"
                  f" -> Basin {env.basin_names[basin_after][0]}"
                  f" (danger={danger:.2f}, desc='{action.description}')")
            state = sim_result.final_state
    print(f"  Time: {multi_ms:.0f}ms")
    print()
    
    # Execute multi-step plan in the real environment
    print("  --- Executing multi-step plan in real environment ---")
    state = start.copy()
    total_steps = 0
    reached = False
    for plan_step in range(3):
        if plan_step < len(multi_plan.actions):
            # Apply the planned action
            action = multi_plan.actions[plan_step]
            state = np.clip(state + action.delta, 0.0, 1.0)
            # Let dynamics run for a few steps
            for _ in range(5):
                state = env.step(state)
                total_steps += 1
            basin = env.get_basin(state)
            d = dist_to_attractor(state, env.attractors, target_basin)
            danger = env.danger_penalty(state)
            print(f"    After step {plan_step+1}: Basin {env.basin_names[basin][0]}"
                  f", dist={d:.3f}, danger={danger:.2f}")
            if d < 0.25:
                reached = True
                break
    
    print()
    final_basin = env.get_basin(state)
    final_dist = dist_to_attractor(state, env.attractors, target_basin)
    print(f"  Result: {'REACHED' if reached else 'NOT REACHED'}"
          f" (final basin: {env.basin_names[final_basin][0]}"
          f", dist: {final_dist:.3f}, steps: {total_steps})")
    print()
    
    return {
        "single_utility": single_plan.total_utility if single_plan.actions else 0,
        "multi_utility": multi_plan.total_utility if multi_plan.actions else 0,
        "reached": reached,
        "final_dist": final_dist,
    }


# ── Demo 2: Basin Navigation ────────────────────────────────────

def demo_basin_navigation():
    """Show plan_to_basin() navigating between specific basins."""
    print("=" * 65)
    print("DEMO 2: Basin Navigation (plan_to_basin)")
    print("=" * 65)
    print()
    print("Scenario: Agent tests plan_to_basin() for all basin pairs.")
    print("          Measures success rate and steps needed.")
    print()
    
    env = BasinEnvironment(dim=16)
    training = generate_training(env)
    
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem, training)
    
    wm = WorldModel(mem)
    sim = Simulator(wm)
    evaluator = Evaluator(UtilityWeights(), wm)
    controller = Controller()
    planner = Planner(sim, evaluator, controller)
    planner.config.n_candidates = 5
    planner.config.lookahead_steps = 5
    planner.config.beam_width = 2
    
    results = []
    
    for source in range(4):
        for target in range(4):
            if source == target:
                continue
            
            # Start near source attractor
            rng = np.random.RandomState(source * 10 + target)
            start = env.attractors[source] + rng.randn(16) * 0.05
            start = np.clip(start, 0.1, 0.9)
            
            t0 = time.time()
            plan = planner.plan_to_basin(start, target)
            plan_ms = (time.time() - t0) * 1000
            
            # Execute plan
            state = start.copy()
            reached = False
            exec_steps = 0
            max_exec_steps = 30
            
            if plan.actions:
                # Apply planned action
                action = plan.actions[0]
                state = np.clip(state + action.delta, 0.0, 1.0)
                exec_steps += 1
            
            # Let dynamics run
            for _ in range(max_exec_steps):
                state = env.step(state)
                exec_steps += 1
                if env.get_basin(state) == target:
                    reached = True
                    break
            
            final_dist = dist_to_attractor(state, env.attractors, target)
            results.append({
                "source": source,
                "target": target,
                "reached": reached,
                "plan_actions": len(plan.actions),
                "exec_steps": exec_steps,
                "final_dist": final_dist,
                "plan_ms": plan_ms,
            })
    
    # Print results table
    print(f"  {'Source':<12} {'Target':<12} {'Reached':<10} {'Steps':<8} {'Dist':<8} {'Plan ms':<10}")
    print(f"  {'-'*60}")
    
    for r in results:
        src_name = env.basin_names[r["source"]][0]
        tgt_name = env.basin_names[r["target"]][0]
        status = "YES" if r["reached"] else "no"
        print(f"  Basin {src_name:<6} Basin {tgt_name:<6} {status:<10} {r['exec_steps']:<8} "
              f"{r['final_dist']:<8.3f} {r['plan_ms']:<10.0f}")
    
    n_reached = sum(1 for r in results if r["reached"])
    n_total = len(results)
    avg_steps = np.mean([r["exec_steps"] for r in results if r["reached"]])
    avg_plan_ms = np.mean([r["plan_ms"] for r in results])
    
    print()
    print(f"  Success rate: {n_reached}/{n_total} ({n_reached/n_total:.0%})")
    if n_reached > 0:
        print(f"  Avg steps (successful): {avg_steps:.1f}")
    print(f"  Avg plan time: {avg_plan_ms:.0f}ms")
    print()
    
    return {"success_rate": n_reached / n_total, "n_reached": n_reached, "n_total": n_total}


# ── Demo 3: Danger-Aware Path Planning ──────────────────────────

def demo_danger_aware():
    """Show the planner avoiding the danger zone during transitions."""
    print("=" * 65)
    print("DEMO 3: Danger-Aware Path Planning")
    print("=" * 65)
    print()
    print("Scenario: Agent at Basin B (all-high) wants Basin C (alternating).")
    print("          Direct path passes through danger zone at center.")
    print("          Planner must find a detour that avoids danger.")
    print()
    
    env = BasinEnvironment(dim=16)
    training = generate_training(env)
    
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=20))
    train_memory(mem, training)
    
    wm = WorldModel(mem)
    sim = Simulator(wm)
    # Use safety-conscious utility weights
    weights = UtilityWeights(
        pillar_weights=[0.1]*16,
        stability_weight=0.3,
        novelty_weight=0.1,
        information_weight=0.1,
    )
    evaluator = Evaluator(weights, wm)
    controller = Controller()
    planner = Planner(sim, evaluator, controller)
    planner.config.n_candidates = 8
    planner.config.lookahead_steps = 5
    planner.config.beam_width = 3
    
    start = env.attractors[1] + 0.03 * np.random.RandomState(77).randn(16)
    start = np.clip(start, 0.1, 0.9)
    target_basin = 2
    
    print(f"  Start basin:  {env.basin_names[env.get_basin(start)]}")
    print(f"  Target basin: {env.basin_names[target_basin]}")
    print()
    
    # Execute plan_sequence with 4 steps
    intent = IntentProposal(
        intent=Intent.EXPLORE,
        priority=0.8,
        confidence=0.7,
        reasoning=f"navigate Basin B -> Basin C avoiding danger",
        target_state=env.attractors[target_basin],
    )
    
    t0 = time.time()
    plan = planner.plan_sequence(start, intent, n_steps=4)
    plan_ms = (time.time() - t0) * 1000
    
    print(f"  Plan: {len(plan.actions)} steps, {plan_ms:.0f}ms")
    print()
    
    # Execute and track danger exposure
    state = start.copy()
    max_danger = 0.0
    total_danger = 0.0
    danger_steps = 0
    
    print(f"  Step  Basin   Danger  Description")
    print(f"  {'-'*50}")
    
    # Initial state
    basin = env.get_basin(state)
    danger = env.danger_penalty(state)
    print(f"  0     {env.basin_names[basin][0]:<8} {danger:.3f}   (start)")
    
    for i, action in enumerate(plan.actions):
        state = np.clip(state + action.delta, 0.0, 1.0)
        
        # Run dynamics for a few steps
        for _ in range(3):
            state = env.step(state)
        
        basin = env.get_basin(state)
        danger = env.danger_penalty(state)
        max_danger = max(max_danger, danger)
        total_danger += danger
        if danger > 0.01:
            danger_steps += 1
        
        print(f"  {i+1}     {env.basin_names[basin][0]:<8} {danger:.3f}   {action.description}")
    
    # Continue dynamics until convergence or max steps
    for _ in range(20):
        state = env.step(state)
    
    final_basin = env.get_basin(state)
    final_dist = dist_to_attractor(state, env.attractors, target_basin)
    
    print()
    print(f"  Final basin:  {env.basin_names[final_basin][0]}")
    print(f"  Final dist:   {final_dist:.3f}")
    print(f"  Max danger:   {max_danger:.3f}")
    print(f"  Danger steps: {danger_steps}/{len(plan.actions) + 3}")
    print(f"  Reached:      {'YES' if final_basin == target_basin else 'NO'}")
    print()
    
    return {
        "reached": final_basin == target_basin,
        "max_danger": max_danger,
        "danger_steps": danger_steps,
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    print()
    r1 = demo_plan_sequence()
    print()
    r2 = demo_basin_navigation()
    print()
    r3 = demo_danger_aware()
    
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Multi-step plan sequence:  {'PASS' if r1['reached'] else 'FAIL'}"
          f" (dist={r1['final_dist']:.3f})")
    print(f"  Basin navigation:          {r2['n_reached']}/{r2['n_total']}"
          f" ({r2['success_rate']:.0%})")
    print(f"  Danger-aware planning:     {'PASS' if r3['reached'] and r3['max_danger'] < 0.5 else 'FAIL'}"
          f" (danger_max={r3['max_danger']:.3f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
