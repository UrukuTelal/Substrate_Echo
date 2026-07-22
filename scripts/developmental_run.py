"""Long Developmental Run — 10K ticks of curiosity-driven exploration.

Measures:
1. Prediction error over time (should decrease)
2. Novelty over time (should decrease)
3. Exploration coverage (should increase)
4. Information gain as planning signal
5. Learning curves with social sharing
"""

import sys
import time
import numpy as np

sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.dynamics_memory import DynamicsMemory
from substrate_echo.core.world_model import WorldModel
from substrate_echo.core.evaluator import Evaluator, UtilityWeights
from substrate_echo.core.intent_generator import IntentGenerator, AgentPersonality
from substrate_echo.core.intent import Intent


def run_developmental_experiment(n_ticks=10000, n_agents=3, dim=16):
    """Run a developmental experiment with curiosity-driven exploration."""
    print(f"\n=== Developmental Run: {n_ticks} ticks, {n_agents} agents ===")
    rng = np.random.RandomState(42)

    # True dynamics: 3 basins at different locations
    basin_centers = [
        np.full(dim, 0.2),
        np.full(dim, 0.5),
        np.full(dim, 0.8),
    ]

    def true_dynamics(x):
        dists = [np.linalg.norm(x - c) for c in basin_centers]
        nearest = np.argmin(dists)
        return -0.5 * (x - basin_centers[nearest]) + rng.randn(dim) * 0.005

    # Create agents
    agents = []
    for i in range(n_agents):
        dm = DynamicsMemory(dim=dim)
        wm = WorldModel(dm)
        personality = AgentPersonality.explorer() if i == 0 else AgentPersonality()
        intent_gen = IntentGenerator(personality=personality, world_model=wm)
        evaluator = Evaluator(world_model=wm)
        agents.append({
            "wm": wm,
            "intent_gen": intent_gen,
            "evaluator": evaluator,
            "position": rng.uniform(0.0, 1.0, dim),
            "tick": 0,
            "error_history": [],
            "novelty_history": [],
            "info_gain_history": [],
        })

    # Track metrics over time
    metrics = {
        "ticks": [],
        "avg_error": [],
        "avg_novelty": [],
        "avg_info_gain": [],
        "n_training_samples": [],
    }

    t_start = time.time()

    for tick in range(n_ticks):
        for agent in agents:
            # 1. Generate intent based on current state
            intent = agent["intent_gen"].generate_intent(agent["position"])

            # 2. Choose action based on intent
            if intent.intent == Intent.INFORMATION_GAIN:
                # Explore: move toward highest novelty
                candidates = [agent["position"] + rng.randn(dim) * 0.1
                              for _ in range(5)]
                candidates = [np.clip(c, 0.0, 1.0) for c in candidates]
                if agent["wm"].memory._states and len(agent["wm"].memory._states) > 10:
                    novelties = [agent["wm"].memory.novelty(c) for c in candidates]
                    agent["position"] = candidates[np.argmax(novelties)]
                else:
                    agent["position"] = candidates[0]
            elif intent.intent == Intent.EXPLORE:
                # Random exploration
                agent["position"] = np.clip(
                    agent["position"] + rng.randn(dim) * 0.15, 0.0, 1.0)
            else:
                # Stay near current position (exploit)
                agent["position"] = np.clip(
                    agent["position"] + rng.randn(dim) * 0.05, 0.0, 1.0)

            # 3. Observe true dynamics
            v = true_dynamics(agent["position"])

            # 4. Learn
            agent["wm"].memory._states.append(agent["position"].copy())
            agent["wm"].memory._velocities.append(v.copy())
            agent["tick"] += 1

            # Fit every 500 ticks
            if agent["tick"] % 500 == 0 and agent["tick"] > 0:
                agent["wm"].memory._fit_dynamics()

            # 5. Measure metrics (every 10 ticks to save time)
            if tick % 10 == 0 and agent["wm"].memory._fitted and agent["wm"].memory._states:
                error = agent["wm"].memory.prediction_error(agent["position"], v)
                novelty = agent["wm"].memory.novelty(agent["position"])
                info_gain = agent["wm"].memory.information_gain(agent["position"])

                agent["error_history"].append(error)
                agent["novelty_history"].append(novelty)
                agent["info_gain_history"].append(info_gain)

        # Social sharing every 500 ticks
        if tick > 0 and tick % 500 == 0:
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        agents[i]["wm"].share_observations(agents[j]["wm"])

        # Record aggregate metrics every 50 ticks
        if tick % 50 == 0:
            all_errors = [a["error_history"][-1] for a in agents
                          if a["error_history"]]
            all_novelties = [a["novelty_history"][-1] for a in agents
                             if a["novelty_history"]]
            all_info = [a["info_gain_history"][-1] for a in agents
                        if a["info_gain_history"]]
            n_samples = sum(len(a["wm"].memory._states) for a in agents)

            metrics["ticks"].append(tick)
            metrics["avg_error"].append(np.mean(all_errors) if all_errors else 0)
            metrics["avg_novelty"].append(np.mean(all_novelties) if all_novelties else 0)
            metrics["avg_info_gain"].append(np.mean(all_info) if all_info else 0)
            metrics["n_training_samples"].append(n_samples)

    t_elapsed = time.time() - t_start

    # Print results
    print(f"  Elapsed time: {t_elapsed:.1f}s")
    print(f"  Throughput: {n_ticks / t_elapsed:.0f} ticks/sec")
    print(f"  Final training samples: {metrics['n_training_samples'][-1]}")

    # Print learning curve
    print(f"\n  Learning curve (every {n_ticks//10} ticks):")
    print(f"  {'Tick':>8} {'Error':>10} {'Novelty':>10} {'InfoGain':>10} {'Samples':>8}")
    print(f"  {'--------':>8} {'----------':>10} {'----------':>10} {'----------':>10} {'--------':>8}")
    step = max(1, len(metrics["ticks"]) // 10)
    for i in range(0, len(metrics["ticks"]), step):
        t = metrics["ticks"][i]
        e = metrics["avg_error"][i]
        n = metrics["avg_novelty"][i]
        ig = metrics["avg_info_gain"][i]
        s = metrics["n_training_samples"][i]
        print(f"  {t:>8} {e:>10.6f} {n:>10.6f} {ig:>10.6f} {s:>8}")

    # Check if learning occurred
    early_error = np.mean(metrics["avg_error"][:5])
    late_error = np.mean(metrics["avg_error"][-5:])
    error_reduction = (early_error - late_error) / max(early_error, 1e-10)

    early_novelty = np.mean(metrics["avg_novelty"][:5])
    late_novelty = np.mean(metrics["avg_novelty"][-5:])
    novelty_reduction = (early_novelty - late_novelty) / max(early_novelty, 1e-10)

    print(f"\n  Error reduction: {error_reduction:.1%} "
          f"({early_error:.6f} -> {late_error:.6f})")
    print(f"  Novelty reduction: {novelty_reduction:.1%} "
          f"({early_novelty:.6f} -> {late_novelty:.6f})")
    print(f"  Learning occurred: {'YES' if error_reduction > 0.1 else 'NO'}")
    print(f"  Curiosity self-extinguished: {'YES' if novelty_reduction > 0.1 else 'NO'}")

    # Measure final coverage
    print(f"\n  Final agent states:")
    for i, agent in enumerate(agents):
        n = len(agent["wm"].memory._states)
        fitted = agent["wm"].memory._fitted
        err = agent["error_history"][-1] if agent["error_history"] else 0
        nov = agent["novelty_history"][-1] if agent["novelty_history"] else 0
        print(f"    Agent {i}: {n} samples, fitted={fitted}, "
              f"error={err:.6f}, novelty={nov:.6f}")

    return {
        "ticks": n_ticks,
        "elapsed": t_elapsed,
        "throughput": n_ticks / t_elapsed,
        "error_reduction": error_reduction,
        "novelty_reduction": novelty_reduction,
        "final_samples": metrics["n_training_samples"][-1],
        "metrics": metrics,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Developmental Run")
    print("=" * 60)

    run_developmental_experiment(n_ticks=2000, n_agents=3)

    print("\n" + "=" * 60)
    print("Developmental run complete.")
    print("=" * 60)
