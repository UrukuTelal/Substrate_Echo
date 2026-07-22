"""P9.1 — Continuous Autonomous Operation

Long-duration integrated run: agents living in a world.
Measures cognitive, behavioral, and social health.

Usage:
    python scripts/p9_ecology_run.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from substrate_echo.core.world import World, WorldConfig
from substrate_echo.core.integrated_agent import IntegratedAgent, IntegratedAgentConfig


def run_ecology(n_agents: int = 5, n_ticks: int = 1000,
                verbose: bool = True) -> dict:
    """Run an ecology experiment.

    Returns metrics dict with cognitive, behavioral, social health.
    """
    # Setup world
    world_config = WorldConfig(
        grid_size=10.0,
        n_resources=30,
        resource_regen_rate=0.05,
        energy_decay_rate=0.001,
        observation_range=4.0,
        storm_interval=500,
        event_probability=0.02,
    )
    world = World(world_config)

    # Create agents
    agent_config = IntegratedAgentConfig(
        curiosity_drive=0.3,
        social_drive=0.2,
        survival_drive=0.5,
        plan_interval=10,
        calibration_interval=50,
    )

    agents = []
    for i in range(n_agents):
        agent = IntegratedAgent(agent_id=i, config=agent_config)
        pos = np.random.uniform(1, 9, 2)  # avoid edges
        world.add_agent(i, position=pos)
        agents.append(agent)

    # Tracking
    energy_history = np.zeros((n_ticks, n_agents))
    alive_history = np.zeros((n_ticks, n_agents), dtype=bool)
    confidence_history = np.zeros((n_ticks, n_agents))
    actions_per_tick = np.zeros(n_ticks)
    social_per_tick = np.zeros(n_ticks)

    start_time = time.time()

    for tick in range(n_ticks):
        # World tick
        world.tick()

        # Agent ticks
        for i, agent in enumerate(agents):
            if agent.id not in world.agent_ids:
                continue

            # Observe and act
            obs = world.observe(agent.id)
            action = agent.think(obs)
            result = world.apply_action(agent.id, action)

            # Track
            energy_history[tick, i] = agent._energy
            alive_history[tick, i] = agent._energy > 0
            meta = agent.meta_cognition.get_meta_state()
            confidence_history[tick, i] = meta.calibrated_confidence
            actions_per_tick[tick] += 1
            if obs.get("nearby_agents"):
                social_per_tick[tick] += 1

        # Progress
        if verbose and (tick + 1) % 1000 == 0:
            alive_count = sum(1 for a in agents if a._energy > 0)
            avg_energy = np.mean([a._energy for a in agents if a._energy > 0])
            elapsed = time.time() - start_time
            print(f"  Tick {tick+1}/{n_ticks} | "
                  f"Alive: {alive_count}/{n_agents} | "
                  f"Avg energy: {avg_energy:.3f} | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time

    # Compute health metrics
    health = _compute_health_metrics(
        agents, energy_history, alive_history,
        confidence_history, actions_per_tick,
        social_per_tick, n_ticks, n_agents, elapsed, world)

    return health


def _compute_health_metrics(agents, energy_history, alive_history,
                            confidence_history, actions_per_tick,
                            social_per_tick, n_ticks, n_agents,
                            elapsed, world) -> dict:
    """Compute comprehensive health metrics."""

    # Cognitive health
    avg_confidence = np.mean(confidence_history)
    confidence_trend = np.polyfit(range(n_ticks), np.mean(confidence_history, axis=1), 1)[0]
    calibration_error = 0.0
    for agent in agents:
        cal = agent.meta_cognition.get_meta_state().calibration_error
        calibration_error += cal
    calibration_error /= max(1, len(agents))

    # Behavioral health
    survival_rate = np.mean(alive_history[-100:])
    avg_energy_final = np.mean([a._energy for a in agents if a._energy > 0]) if any(a._energy > 0 for a in agents) else 0

    total_habits = sum(len(a.habit_formation.get_established_habits()) for a in agents)
    total_episodes = sum(a.metrics.episodes_stored for a in agents)
    total_plans = sum(a.metrics.plans_created for a in agents)

    # Social health
    total_social = int(np.sum(social_per_tick))
    avg_social_per_tick = np.mean(social_per_tick)

    # Resource dynamics
    world_stats = world.stats()

    return {
        "cognitive_health": {
            "avg_confidence": float(avg_confidence),
            "confidence_trend": float(confidence_trend),
            "calibration_error": float(calibration_error),
        },
        "behavioral_health": {
            "survival_rate": float(survival_rate),
            "avg_energy_final": float(avg_energy_final),
            "total_habits": total_habits,
            "total_episodes": total_episodes,
            "total_plans": total_plans,
        },
        "social_health": {
            "total_social_interactions": total_social,
            "avg_social_per_tick": float(avg_social_per_tick),
        },
        "world": world_stats,
        "performance": {
            "n_ticks": n_ticks,
            "n_agents": n_agents,
            "elapsed_seconds": elapsed,
            "ticks_per_second": n_ticks / elapsed if elapsed > 0 else 0,
        },
    }


def main():
    """Run P9.1 ecology experiments."""
    print("=" * 60)
    print("P9.1 — Continuous Autonomous Operation")
    print("=" * 60)

    # Test 1: Small ecology
    print("\n--- Test 1: Small ecology (5 agents, 1000 ticks) ---")
    health1 = run_ecology(n_agents=5, n_ticks=1000, verbose=True)
    _print_health(health1)

    # Test 2: Medium ecology
    print("\n--- Test 2: Medium ecology (10 agents, 5000 ticks) ---")
    health2 = run_ecology(n_agents=10, n_ticks=5000, verbose=True)
    _print_health(health2)

    # Test 3: Large ecology
    print("\n--- Test 3: Large ecology (20 agents, 10000 ticks) ---")
    health3 = run_ecology(n_agents=20, n_ticks=10000, verbose=True)
    _print_health(health3)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


def _print_health(health: dict) -> None:
    """Pretty-print health metrics."""
    ch = health["cognitive_health"]
    bh = health["behavioral_health"]
    sh = health["social_health"]
    perf = health["performance"]

    print(f"  Cognitive: confidence={ch['avg_confidence']:.3f} "
          f"trend={ch['confidence_trend']:+.6f} "
          f"cal_error={ch['calibration_error']:.4f}")
    print(f"  Behavioral: survival={bh['survival_rate']:.1%} "
          f"energy={bh['avg_energy_final']:.3f} "
          f"habits={bh['total_habits']} "
          f"episodes={bh['total_episodes']} "
          f"plans={bh['total_plans']}")
    print(f"  Social: interactions={sh['total_social_interactions']} "
          f"avg/tick={sh['avg_social_per_tick']:.2f}")
    print(f"  Performance: {perf['ticks_per_second']:.1f} ticks/sec")


if __name__ == "__main__":
    main()
