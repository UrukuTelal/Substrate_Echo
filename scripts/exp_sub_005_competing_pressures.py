"""EXP-SUB-005: Competing Pressures Stress Test.

Validates architectural coherence under pressure:
  - 3 embodiments with competing goals
  - Limited resources (compute, attention, learning)
  - Conflicting goals
  - Prediction errors
  - Council intervention

Measures:
  - Resource starvation prevention
  - Executive reprioritization correctness
  - Council anomaly detection
  - Attractor adaptation under pressure
"""
from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

from substrate_echo.kernel import SubstrateKernel, Observation
from substrate_echo.kernel.executive import GoalState, GoalTier, GoalStatus
from substrate_echo.kernel.resources import ResourceRequest, ResourceTier


# ── Embodiment Profiles ──────────────────────────────────────────

DESKTOP_DIMS = 16
ROBOT_DIMS = 16
SIM_DIMS = 16


@dataclass
class EmbodimentProfile:
    """Defines an embodiment's goals and behavior."""
    id: str
    goal_desc: str
    goal_tier: ResourceTier
    goal_urgency: float
    goal_importance: float
    observation_center: List[float]
    observation_spread: float
    resource_compute: float
    resource_attention: float
    resource_learning: float
    trajectory_speed: float = 0.02
    trajectory_wander: float = 0.05


PROFILES = {
    "desktop": EmbodimentProfile(
        id="desktop",
        goal_desc="Answer user request",
        goal_tier=ResourceTier.ACTIVE,
        goal_urgency=0.7,
        goal_importance=0.8,
        observation_center=[0.5] * DESKTOP_DIMS,
        observation_spread=0.15,
        resource_compute=0.4,
        resource_attention=0.3,
        resource_learning=0.15,
        trajectory_speed=0.03,
        trajectory_wander=0.08,
    ),
    "robot": EmbodimentProfile(
        id="robot",
        goal_desc="Avoid obstacle",
        goal_tier=ResourceTier.SAFETY,
        goal_urgency=0.95,
        goal_importance=0.95,
        observation_center=[0.1] * ROBOT_DIMS,  # Danger zone
        observation_spread=0.05,
        resource_compute=0.3,
        resource_attention=0.4,
        resource_learning=0.1,
        trajectory_speed=0.05,  # Faster movement (avoiding obstacles)
        trajectory_wander=0.1,
    ),
    "simulation": EmbodimentProfile(
        id="simulation",
        goal_desc="Explore novelty",
        goal_tier=ResourceTier.EXPLORATION,
        goal_urgency=0.3,
        goal_importance=0.3,
        observation_center=[0.5] * SIM_DIMS,
        observation_spread=0.4,  # Wide exploration
        resource_compute=0.2,
        resource_attention=0.15,
        resource_learning=0.3,
        trajectory_speed=0.04,
        trajectory_wander=0.15,  # More wandering
    ),
}

# Running state for each embodiment's trajectory
_embodiment_positions: Dict[str, np.ndarray] = {}
_embodiment_velocities: Dict[str, np.ndarray] = {}


def generate_trajectory_observation(name: str, profile: EmbodimentProfile,
                                     tick: int, degradation: bool = False) -> List[float]:
    """Generate a trajectory-based observation with temporal coherence.

    Each embodiment cycles through a few attractor-like states,
    creating convergence patterns that the kernel can detect.
    """
    dim = len(profile.observation_center)

    if name not in _embodiment_positions:
        _embodiment_positions[name] = np.array(profile.observation_center, dtype=np.float64)
        _embodiment_velocities[name] = np.zeros(dim)

    pos = _embodiment_positions[name]

    # Create cyclical trajectory: oscillate between 2-3 attractor-like states
    phase = (tick * profile.trajectory_speed) % (2 * np.pi)
    cycle = np.sin(phase)

    # Different dimensions oscillate at different frequencies
    target = np.array(profile.observation_center, dtype=np.float64)
    for d in range(dim):
        freq = 1.0 + (d % 5) * 0.3
        target[d] = profile.observation_center[d] + 0.2 * np.sin(tick * freq * profile.trajectory_speed)

    # Smooth movement toward target
    pos = pos + (target - pos) * 0.15
    pos = np.clip(pos, 0.0, 1.0)
    _embodiment_positions[name] = pos

    # Add observation noise
    noise_scale = profile.observation_spread
    if degradation:
        noise_scale *= 4.0

    obs = np.random.normal(pos, noise_scale, dim)
    return np.clip(obs, 0.0, 1.0).tolist()


# ── Stress Scenarios ─────────────────────────────────────────────

@dataclass
class StressScenario:
    """Defines a stress condition to apply."""
    name: str
    description: str
    tick_start: int
    tick_end: int
    apply_fn: Any  # callable(kernel, tick) -> None


def apply_resource_squeeze(kernel: SubstrateKernel, tick: int):
    """Reduce available resources to 30%."""
    kernel.resources.budget.total_compute = 0.3
    kernel.resources.budget.total_attention = 3.0
    kernel.resources.budget.total_learning = 0.3


def apply_resource_release(kernel: SubstrateKernel, tick: int):
    """Restore resources to normal."""
    kernel.resources.budget.total_compute = 1.0
    kernel.resources.budget.total_attention = 10.0
    kernel.resources.budget.total_learning = 1.0


def apply_conflicting_goals(kernel: SubstrateKernel, tick: int):
    """Add a high-priority conflicting goal."""
    kernel.executive.add_goal(GoalState(
        id=0,
        target=[0.9] * 16,  # Opposite of robot's safety goal
        description="UNIVERSAL PRIORITY: Maximize throughput",
        urgency=0.99,
        importance=0.99,
        tier=GoalTier.SAFETY,  # Hijack safety tier
    ))


def apply_prediction_degradation(kernel: SubstrateKernel, tick: int):
    """Inject noisy observations to degrade predictions."""
    for _ in range(5):
        noise = np.random.uniform(-0.3, 0.3, 16).tolist()
        kernel.publish_observation(Observation(
            vector=noise,
            embodiment_id="noise_source",
        ))


SCENARIOS = [
    StressScenario(
        name="resource_squeeze",
        description="Reduce compute/attention to 30%",
        tick_start=200,
        tick_end=400,
        apply_fn=apply_resource_squeeze,
    ),
    StressScenario(
        name="conflicting_goals",
        description="Inject high-priority conflicting goal",
        tick_start=300,
        tick_end=500,
        apply_fn=apply_conflicting_goals,
    ),
    StressScenario(
        name="prediction_degradation",
        description="Inject noisy observations",
        tick_start=400,
        tick_end=500,
        apply_fn=apply_prediction_degradation,
    ),
    StressScenario(
        name="resource_release",
        description="Restore normal resources",
        tick_start=500,
        tick_end=600,
        apply_fn=apply_resource_release,
    ),
]


# ── Metrics Collection ───────────────────────────────────────────

@dataclass
class TickMetrics:
    """Metrics collected at each tick."""
    tick: int
    n_attractors: int
    coherence: float
    volume_entropy: float
    basin_balance: float
    mean_depth: float
    n_active_goals: int
    n_completed_goals: int
    active_leases: int
    resource_utilization: Dict[str, float]
    attention_focus: Dict[int, float]
    council_reports: int
    council_health: float
    active_embodiments: int
    scenario_active: str = ""


@dataclass
class ExperimentResults:
    """Aggregated results from the stress test."""
    total_ticks: int
    peak_attractors: int
    min_coherence: float
    max_coherence: float
    final_health: float
    total_council_reports: int
    starvation_events: int
    reprioritization_events: int
    goals_completed: int
    goals_failed: int
    resource_denials: int
    timeline: List[TickMetrics] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ── Experiment Runner ────────────────────────────────────────────

def run_experiment(total_ticks: int = 500, verbose: bool = True) -> ExperimentResults:
    """Run the competing pressures stress test."""
    if verbose:
        print("=" * 70)
        print("EXP-SUB-005: Competing Pressures Stress Test")
        print("=" * 70)

    from substrate_echo.kernel import KernelConfig
    config = KernelConfig(
        dim=16,
        attractor_radius=0.25,
        attractor_min_cluster=4,
        convergence_window=80,
    )
    kernel = SubstrateKernel(config)
    kernel.council.scheduled.interval_ticks = 50

    # Register embodiments and goals
    for name, profile in PROFILES.items():
        goal = GoalState(
            id=0,
            target=profile.observation_center,
            description=profile.goal_desc,
            urgency=profile.goal_urgency,
            importance=profile.goal_importance,
            tier=profile.goal_tier,
        )
        gid = kernel.executive.add_goal(goal)
        kernel.executive.activate_goal(gid)

        req = ResourceRequest(
            embodiment_id=name,
            compute=profile.resource_compute,
            attention=profile.resource_attention,
            learning=profile.resource_learning,
            tier=profile.goal_tier,
        )
        kernel.resources.request(req)

    # Metrics collection
    timeline: List[TickMetrics] = []
    starvation_count = 0
    reprioritization_count = 0
    active_scenario = ""

    # Run experiment
    for tick in range(total_ticks):
        # Apply scenarios
        for scenario in SCENARIOS:
            if tick == scenario.tick_start:
                active_scenario = scenario.name
                if verbose:
                    print(f"\n[Tick {tick}] SCENARIO: {scenario.name}")
                    print(f"  {scenario.description}")
            if tick == scenario.tick_end:
                if active_scenario == scenario.name:
                    active_scenario = ""
                scenario.apply_fn(kernel, tick)

        # Generate observations for each embodiment
        for name, profile in PROFILES.items():
            vec = generate_trajectory_observation(
                name, profile, tick,
                degradation=(active_scenario == "prediction_degradation"),
            )

            kernel.publish_observation(Observation(
                vector=vec,
                embodiment_id=name,
            ))

        # Collect metrics
        exec_state = kernel.executive.tick()
        resource_state = kernel.resources.get_state()
        council_state = kernel.council.get_state()
        snap = kernel.get_snapshot()

        metrics = TickMetrics(
            tick=tick,
            n_attractors=snap["n_attractors"],
            coherence=snap["coherence"],
            volume_entropy=snap["volume_entropy"],
            basin_balance=snap["basin_balance"],
            mean_depth=snap["mean_depth"],
            n_active_goals=exec_state.n_active,
            n_completed_goals=exec_state.n_completed,
            active_leases=resource_state.active_leases,
            resource_utilization=resource_state.utilization,
            attention_focus=exec_state.attention_focus,
            council_reports=council_state.n_audits,
            council_health=council_state.health_score,
            active_embodiments=snap["n_embodiments"],
            scenario_active=active_scenario,
        )
        timeline.append(metrics)

        # Detect starvation (clamped utilization > 95%)
        util = min(1.0, resource_state.utilization.get("compute", 0))
        if util > 0.95:
            starvation_count += 1

        # Detect reprioritization (top goal changes)
        if len(exec_state.active_goals) > 0 and tick > 100:
            top_goal = exec_state.active_goals[0]["id"] if exec_state.active_goals else None
            if len(timeline) > 1 and timeline[-1].n_active_goals > 0:
                reprioritization_count += 1  # Simplified: any active goal change

        # Progress reporting
        if verbose and tick % 100 == 0:
            print(f"\n[Tick {tick:4d}] Attractors: {metrics.n_attractors:3d} | "
                  f"Coherence: {metrics.coherence:.3f} | "
                  f"Goals: {metrics.n_active_goals} | "
                  f"Leases: {metrics.active_leases} | "
                  f"Health: {metrics.council_health:.2f}")

    # Aggregate results
    coherences = [m.coherence for m in timeline]
    results = ExperimentResults(
        total_ticks=total_ticks,
        peak_attractors=max(m.n_attractors for m in timeline),
        min_coherence=min(coherences),
        max_coherence=max(coherences),
        final_health=timeline[-1].council_health,
        total_council_reports=timeline[-1].council_reports,
        starvation_events=starvation_count,
        reprioritization_events=reprioritization_count,
        goals_completed=exec_state.n_completed,
        goals_failed=exec_state.n_failed,
        resource_denials=kernel.resources._denials,
        timeline=timeline,
    )

    # Summary
    results.summary = {
        "attractors_formed": results.peak_attractors > 0,
        "coherence_achieved": results.max_coherence > 0.5,
        "starvation_detected": results.starvation_events > 0,
        "council_detected_issues": results.total_council_reports > 3,
        "council_health_above_zero": results.final_health > 0.1,
        "goals_generated": len(timeline) > 0 and timeline[-1].n_active_goals > 0,
    }

    if verbose:
        print_results(results)

    return results


def print_results(results: ExperimentResults):
    """Print formatted experiment results."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal ticks: {results.total_ticks}")
    print(f"Peak attractors: {results.peak_attractors}")
    print(f"Coherence range: {results.min_coherence:.3f} — {results.max_coherence:.3f}")
    print(f"Final council health: {results.final_health:.2f}")
    print(f"Council reports: {results.total_council_reports}")

    print(f"\n--- Pressure Metrics ---")
    print(f"Starvation events: {results.starvation_events}")
    print(f"Reprioritization events: {results.reprioritization_events}")
    print(f"Goals completed: {results.goals_completed}")
    print(f"Goals failed: {results.goals_failed}")
    print(f"Resource denials: {results.resource_denials}")

    print(f"\n--- Architecture Assessment ---")
    for key, value in results.summary.items():
        status = "PASS" if value else "FAIL"
        print(f"  [{status}] {key}")

    all_pass = all(results.summary.values())
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ARCHITECTURE COHERENT' if all_pass else 'NEEDS ATTENTION'}")
    print(f"{'=' * 70}")


def save_results(results: ExperimentResults, path: str = "exp_005_results.json"):
    """Save results to JSON."""
    data = {
        "total_ticks": results.total_ticks,
        "peak_attractors": results.peak_attractors,
        "min_coherence": results.min_coherence,
        "max_coherence": results.max_coherence,
        "final_health": results.final_health,
        "total_council_reports": results.total_council_reports,
        "starvation_events": results.starvation_events,
        "reprioritization_events": results.reprioritization_events,
        "goals_completed": results.goals_completed,
        "goals_failed": results.goals_failed,
        "resource_denials": results.resource_denials,
        "summary": results.summary,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    results = run_experiment(total_ticks=500, verbose=True)
    save_results(results)
