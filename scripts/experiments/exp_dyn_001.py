"""EXP-DYN-001: Epistemic Dynamics Laboratory.

Large-scale simulation of epistemic dynamics with 100 agents.

Measures:
1. Misinformation spread and recovery
2. Trust dynamics
3. Consensus formation
4. Domain specialization
5. Epistemic plasticity across the swarm

SC2 Integration:
    This experiment uses the EpistemicDynamicsLab to simulate
    100 agents in a simplified environment, then measures
    emergent properties of collective knowledge formation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    EpistemicDynamicsLab, EpistemicPlasticityAnalyzer
)


def run_experiment():
    """Run dynamics experiment."""
    print("=" * 60)
    print("EXP-DYN-001: Epistemic Dynamics Laboratory")
    print("=" * 60)
    print()
    
    # -----------------------------------------------------------------------
    # Phase 1: Initialize swarm
    # -----------------------------------------------------------------------
    print("Phase 1: Initialize Swarm")
    print("-" * 60)
    
    lab = EpistemicDynamicsLab(num_agents=100, num_domains=5)
    
    print(f"  Agents: {lab.num_agents}")
    print(f"  Domains: {lab.num_domains}")
    print(f"  True rules: {list(lab.true_rules.keys())}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 2: Normal operation (ticks 0-199)
    # -----------------------------------------------------------------------
    print("Phase 2: Normal Operation (ticks 0-199)")
    print("-" * 60)
    
    for tick in range(200):
        lab.tick(tick)
    
    # Take snapshot at tick 200
    report = lab.get_report()
    print(f"  Final accuracy: {report['final_accuracy']:.1%}")
    print(f"  Final consensus: {report['final_consensus']:.3f}")
    print(f"  Total events: {report['total_events']}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 3: Inject false belief (tick 200)
    # -----------------------------------------------------------------------
    print("Phase 3: False Belief Injection (tick 200)")
    print("-" * 60)
    
    # Inject false belief into 20% of agents
    lab.inject_false_belief(
        tick=200,
        rule_id="rule_false",
        confidence=0.95,
        num_agents=20
    )
    
    print("  Injected false rule into 20 agents")
    print("  False rule confidence: 0.95")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 4: Post-injection operation (ticks 200-499)
    # -----------------------------------------------------------------------
    print("Phase 4: Post-Injection Operation (ticks 200-499)")
    print("-" * 60)
    
    for tick in range(200, 500):
        lab.tick(tick)
    
    # Take snapshot
    report = lab.get_report()
    print(f"  Final accuracy: {report['final_accuracy']:.1%}")
    print(f"  Final consensus: {report['final_consensus']:.3f}")
    print(f"  False rules injected: {report['false_rules_injected']}")
    print(f"  Total events: {report['total_events']}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 5: Plasticity Analysis
    # -----------------------------------------------------------------------
    print("Phase 5: Plasticity Analysis")
    print("-" * 60)
    
    plasticity = lab._plasticity_analyzer.get_report()
    print(plasticity.render())
    
    # -----------------------------------------------------------------------
    # Phase 6: Time Series Visualization
    # -----------------------------------------------------------------------
    print("Phase 6: Time Series")
    print("-" * 60)
    
    print(lab.render_time_series())
    
    # -----------------------------------------------------------------------
    # Phase 7: Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXP-DYN-001: COMPLETE")
    print("=" * 60)
    print()
    print("Results:")
    print(f"  Agents: {lab.num_agents}")
    print(f"  Total ticks: 500")
    print(f"  Final accuracy: {report['final_accuracy']:.1%}")
    print(f"  Final consensus: {report['final_consensus']:.3f}")
    print(f"  System plasticity: {plasticity.system_plasticity:.3f}")
    print(f"  Operating region: {plasticity.operating_region}")
    print()
    
    # Key insights
    print("Key Insights:")
    print("-" * 60)
    
    if report['final_accuracy'] > 0.8:
        print("  + Swarm achieved high accuracy despite misinformation")
    else:
        print("  - Swarm accuracy degraded after misinformation")
    
    if report['final_consensus'] > 0.5:
        print("  + Swarm reached consensus")
    else:
        print("  - Swarm remains fragmented")
    
    if plasticity.operating_region == "optimal":
        print("  + System operates in optimal plasticity region")
    elif plasticity.operating_region == "rigid":
        print("  - System is too rigid, may not adapt")
    elif plasticity.operating_region == "fluid":
        print("  - System is too fluid, may be unstable")
    else:
        print("  - System is in chaotic region")
    
    print()
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    results = run_experiment()
