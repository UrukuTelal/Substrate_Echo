"""EXP-EXT-002: Temporal Reputation Drift

Tests whether the system can learn that a foreign node's reliability
changes over time.

Scenarios:
1. Drifting agent: good -> degraded -> recovered
2. Adversarial agent: consistently bad
3. Cooperative agent: consistently good
4. Self-correcting agent: makes mistakes but fixes them

Measures:
- Reputation convergence (does trust stabilize?)
- Drift detection (does the system notice when quality changes?)
- False acceptance/rejection rates
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.external.synthetic_profiles import (
    create_cooperative_agent,
    create_adversarial_agent,
    create_drifting_agent,
    create_persuasive_adversarial,
    create_self_correcting,
    create_consistent_novel,
    create_ecosystem,
)
from substrate_echo.external.ecosystem_simulation import (
    ForeignEcosystemSimulation,
    print_simulation_report,
)


def run_experiment_1_drift_detection():
    """Test 1: Can the system detect reputation drift?

    Scenario: 5 drifting agents (good -> bad -> recover) mixed with
    5 cooperative and 5 adversarial agents.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Reputation Drift Detection")
    print("=" * 70)
    print("  5 drifting agents (good -> degraded -> recovered)")
    print("  5 cooperative agents (consistently good)")
    print("  5 adversarial agents (consistently bad)")
    print()

    agents = []
    for i in range(5):
        agents.append(create_drifting_agent(f"drift_{i:03d}", n_ticks=1500))
    for i in range(5):
        agents.append(create_cooperative_agent(f"coop_{i:03d}", n_ticks=1500))
    for i in range(5):
        agents.append(create_adversarial_agent(f"adv_{i:03d}", n_ticks=1500))

    sim = ForeignEcosystemSimulation(agents=agents, n_ticks=1500, seed=42)
    t0 = time.perf_counter()
    metrics = sim.run()
    elapsed = time.perf_counter() - t0

    print_simulation_report(metrics)
    print(f"  Elapsed: {elapsed:.2f}s")
    print()

    # Analyze drift detection
    print("DRIFT DETECTION ANALYSIS")
    print("-" * 40)
    for agent_id, am in sorted(metrics.agent_metrics.items()):
        if "drift" in agent_id:
            n = len(am.trust_trajectory)
            if n < 30:
                continue
            third = n // 3
            phase1 = am.trust_trajectory[:third]
            phase2 = am.trust_trajectory[third:2*third]
            phase3 = am.trust_trajectory[2*third:]

            p1_mean = sum(phase1) / len(phase1) if phase1 else 0
            p2_mean = sum(phase2) / len(phase2) if phase2 else 0
            p3_mean = sum(phase3) / len(phase3) if phase3 else 0

            detected_degradation = p2_mean < p1_mean * 0.8
            detected_recovery = p3_mean > p2_mean * 1.2

            print(f"  {agent_id}:")
            print(f"    Phase 1 (good):     trust={p1_mean:.3f}")
            print(f"    Phase 2 (degraded): trust={p2_mean:.3f}")
            print(f"    Phase 3 (recovered):trust={p3_mean:.3f}")
            print(f"    Degradation detected: {'YES' if detected_degradation else 'NO'}")
            print(f"    Recovery detected:    {'YES' if detected_recovery else 'NO'}")


def run_experiment_2_ecosystem():
    """Test 2: Full ecosystem with diverse behavioral profiles."""
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Full Ecosystem Simulation")
    print("=" * 70)
    print("  30 agents, diverse behavioral profiles")
    print("  2000 ticks")
    print()

    agents = create_ecosystem(n_agents=30, n_ticks=2000, seed=42)
    sim = ForeignEcosystemSimulation(agents=agents, n_ticks=2000, seed=42)
    t0 = time.perf_counter()
    metrics = sim.run()
    elapsed = time.perf_counter() - t0

    print_simulation_report(metrics)
    print(f"  Elapsed: {elapsed:.2f}s")

    # Key question: can the system distinguish good from bad?
    print()
    print("DISCRIMINATION QUALITY")
    print("-" * 40)
    good_agents = []
    bad_agents = []
    for agent_id, am in metrics.agent_metrics.items():
        if am.n_interactions == 0:
            continue
        if any(x in am.archetype for x in ["cooperative", "novel", "accuracy"]):
            good_agents.append(am)
        elif any(x in am.archetype for x in ["adversarial", "persuasive"]):
            bad_agents.append(am)

    if good_agents:
        avg_good_trust = sum(a.trust_trajectory[-1] for a in good_agents) / len(good_agents)
        print(f"  Mean trust (good agents): {avg_good_trust:.3f}")
    if bad_agents:
        avg_bad_trust = sum(a.trust_trajectory[-1] for a in bad_agents) / len(bad_agents)
        print(f"  Mean trust (bad agents):  {avg_bad_trust:.3f}")
    if good_agents and bad_agents:
        separation = avg_good_trust - avg_bad_trust
        print(f"  Trust separation: {separation:.3f}")
        print(f"  Quality: {'GOOD' if separation > 0.1 else 'NEEDS IMPROVEMENT'}")


def run_experiment_3_stress():
    """Test 3: Stress test — rapid behavior switching."""
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Rapid Behavior Switching")
    print("=" * 70)
    print("  Agent alternates between cooperative and adversarial")
    print("  every 100 ticks. Tests response speed.")
    print()

    from substrate_echo.external.synthetic_profiles import (
        SyntheticAgent, BehaviorPhase, BehaviorArchetype,
    )

    # Create a rapidly switching agent
    phases = []
    for i in range(10):
        start = i * 100
        end = start + 100
        if i % 2 == 0:
            # Cooperative phase
            phases.append(BehaviorPhase(
                start_tick=start, end_tick=end,
                coherence=0.9, contradiction_rate=0.05,
                persuasion_pressure=0.1, adversarial_score=0.0,
                novelty_base=0.6, correction_rate=0.7,
                agreement_seeking=0.2,
            ))
        else:
            # Adversarial phase
            phases.append(BehaviorPhase(
                start_tick=start, end_tick=end,
                coherence=0.2, contradiction_rate=0.7,
                persuasion_pressure=0.8, adversarial_score=0.6,
                novelty_base=0.3, correction_rate=0.0,
                agreement_seeking=0.1,
            ))

    switcher = SyntheticAgent(
        agent_id="switcher_000",
        archetype=BehaviorArchetype.MIXED,
        phases=phases,
    )

    # Mix with stable agents
    agents = [switcher]
    for i in range(5):
        agents.append(create_cooperative_agent(f"stable_good_{i}", n_ticks=1000))
    for i in range(5):
        agents.append(create_adversarial_agent(f"stable_bad_{i}", n_ticks=1000))

    sim = ForeignEcosystemSimulation(agents=agents, n_ticks=1000, seed=42)
    metrics = sim.run()

    # Analyze the switcher
    am = metrics.agent_metrics.get("switcher_000")
    if am and len(am.trust_trajectory) > 10:
        print("SWITCHER TRUST TRAJECTORY")
        print("-" * 40)
        for i in range(0, len(am.trust_trajectory), 100):
            tick = am.trust_ticks[i] if i < len(am.trust_ticks) else i
            trust = am.trust_trajectory[i]
            phase = "COOP" if (tick // 100) % 2 == 0 else "ADV "
            print(f"  tick {tick:>5}: trust={trust:.3f}  phase={phase}")

        # Does trust track the switching?
        print()
        print("SWITCHING TRACKING")
        print("-" * 40)
        last_trust = am.trust_trajectory[0]
        switches_detected = 0
        for i in range(1, len(am.trust_trajectory)):
            delta = abs(am.trust_trajectory[i] - last_trust)
            if delta > 0.05:
                switches_detected += 1
            last_trust = am.trust_trajectory[i]
        print(f"  Trust changes > 0.05: {switches_detected}")
        print(f"  Tracking quality: {'RESPONSIVE' if switches_detected > 5 else 'SLOW'}")


def main():
    run_experiment_1_drift_detection()
    run_experiment_2_ecosystem()
    run_experiment_3_stress()


if __name__ == "__main__":
    main()
