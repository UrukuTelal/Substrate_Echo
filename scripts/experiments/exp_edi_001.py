"""EXP-EDI-001: Epistemic Dependency Index Measurement.

Creates multiple reasoning scenarios with different cognitive
profiles, then measures how each architecture balances
observation, priors, trust, prediction, and memory.

SC2 Integration:
    Simulates 5 different reasoning scenarios:
    1. Scout-heavy (observation dominates)
    2. Prior-heavy (cultural priors dominate)
    3. Trust crisis (trust updates dominate)
    4. Balanced (all sources contribute)
    5. Memory-driven (discoveries dominate)

    For each scenario, runs counterfactual analysis to compute
    the Epistemic Dependency Index (EDI).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    CausalReplay, CounterfactualEngine, DependencyAnalyzer,
    NodeType
)


def build_scenario_scout_heavy(replay: CausalReplay) -> str:
    """Scenario 1: Many observations, few priors."""
    obs1 = replay.record_observation(100, {"scout": True, "type": "marine"})
    obs2 = replay.record_observation(101, {"scout_count": 3})
    obs3 = replay.record_observation(102, {"position": "north"})
    obs4 = replay.record_observation(103, {"timing": "early"})
    
    feat = replay.record_feature(104, {"aggression": 0.8}, caused_by=obs4.node_id)
    hyp = replay.record_hypothesis(105, {"attack": 0.7}, caused_by=feat.node_id)
    
    # Only one prior, weak
    prior = replay.record_cultural_prior(106, {"prior": "early_scout", "boost": 0.1},
                                         caused_by=hyp.node_id, weight=0.15)
    
    pred = replay.record_prediction(107, {"attack_in_30s": 0.7}, caused_by=prior.node_id)
    decision = replay.record_decision(108, {"action": "defend"}, caused_by=pred.node_id)
    
    return decision.node_id


def build_scenario_prior_heavy(replay: CausalReplay) -> str:
    """Scenario 2: Strong priors, weak observations."""
    obs1 = replay.record_observation(200, {"enemy_building": True})
    feat = replay.record_feature(201, {"building_type": "barracks"}, caused_by=obs1.node_id)
    
    # Strong priors dominate
    prior1 = replay.record_cultural_prior(202, {"prior": "barracks_means_rush", "boost": 0.4},
                                           caused_by=feat.node_id, weight=0.5)
    prior2 = replay.record_cultural_prior(203, {"prior": "rush_pattern_recognized", "boost": 0.3},
                                           caused_by=prior1.node_id, weight=0.4)
    
    hyp = replay.record_hypothesis(204, {"rush": 0.8}, caused_by=prior2.node_id)
    pred = replay.record_prediction(205, {"rush_in_20s": 0.8}, caused_by=hyp.node_id)
    decision = replay.record_decision(206, {"action": "bunker_up"}, caused_by=pred.node_id)
    
    return decision.node_id


def build_scenario_trust_crisis(replay: CausalReplay) -> str:
    """Scenario 3: Trust updates dominate reasoning."""
    obs1 = replay.record_observation(300, {"scout": True})
    feat = replay.record_feature(301, {"threat": 0.6}, caused_by=obs1.node_id)
    hyp = replay.record_hypothesis(302, {"expansion": 0.5}, caused_by=feat.node_id)
    
    # Trust chain: prior helps, then trust drops, then new trust forms
    prior = replay.record_cultural_prior(303, {"prior": "trust_this_prior", "boost": 0.3},
                                          caused_by=hyp.node_id, weight=0.3)
    trust1 = replay.record_trust(304, {"agent": "prior", "before": 0.7, "after": 0.3,
                                        "reason": "prior was wrong last time"},
                                  caused_by=prior.node_id, weight=0.6)
    trust2 = replay.record_trust(305, {"agent": "new_source", "before": 0.2, "after": 0.8,
                                        "reason": "new source proven reliable"},
                                  caused_by=trust1.node_id, weight=0.5)
    
    pred = replay.record_prediction(306, {"expansion": 0.5}, caused_by=trust2.node_id)
    decision = replay.record_decision(307, {"action": "expand_cautiously"}, caused_by=pred.node_id)
    
    return decision.node_id


def build_scenario_balanced(replay: CausalReplay) -> str:
    """Scenario 4: All sources contribute roughly equally."""
    obs1 = replay.record_observation(400, {"scout": True})
    obs2 = replay.record_observation(401, {"no_proxy": True})
    feat = replay.record_feature(402, {"mixed_signals": 0.5}, caused_by=obs2.node_id)
    
    hyp1 = replay.record_hypothesis(403, {"expansion": 0.4}, caused_by=feat.node_id)
    hyp2 = replay.record_hypothesis(404, {"rush": 0.3}, caused_by=feat.node_id)
    
    prior = replay.record_cultural_prior(405, {"prior": "balanced_prior", "boost": 0.2},
                                          caused_by=hyp1.node_id, weight=0.25)
    
    council = replay.record_council(406, {"vote": "expansion", "confidence": 0.55},
                                     caused_by=prior.node_id, weight=0.3)
    
    trust = replay.record_trust(407, {"agent": "council", "level": 0.6},
                                 caused_by=council.node_id, weight=0.2)
    
    pred = replay.record_prediction(408, {"outcome": "expansion"}, caused_by=trust.node_id)
    decision = replay.record_decision(409, {"action": "expand"}, caused_by=pred.node_id)
    
    return decision.node_id


def build_scenario_memory_driven(replay: CausalReplay) -> str:
    """Scenario 5: Past discoveries dominate reasoning."""
    obs1 = replay.record_observation(500, {"enemy": True})
    feat = replay.record_feature(501, {"threat": 0.5}, caused_by=obs1.node_id)
    
    # Strong discovery from past
    disc1 = replay.record_discovery(502, {"rule": "always_expand", "confidence": 0.9},
                                     caused_by=feat.node_id, weight=0.5)
    disc2 = replay.record_discovery(503, {"rule": "expansion_wins", "confidence": 0.85},
                                     caused_by=disc1.node_id, weight=0.45)
    
    # Prior derived from discovery
    prior = replay.record_cultural_prior(504, {"prior": "discovery_backed", "boost": 0.4},
                                          caused_by=disc2.node_id, weight=0.35)
    
    pred = replay.record_prediction(505, {"expansion_success": 0.9}, caused_by=prior.node_id)
    decision = replay.record_decision(506, {"action": "expand"}, caused_by=pred.node_id)
    
    return decision.node_id


def run_experiment():
    """Run EDI measurement across scenarios."""
    print("=" * 60)
    print("EXP-EDI-001: Epistemic Dependency Index")
    print("=" * 60)
    print()
    
    # Build all scenarios
    replay = CausalReplay()
    
    scenarios = {
        "Scout Heavy": build_scenario_scout_heavy(replay),
        "Prior Heavy": build_scenario_prior_heavy(replay),
        "Trust Crisis": build_scenario_trust_crisis(replay),
        "Balanced": build_scenario_balanced(replay),
        "Memory Driven": build_scenario_memory_driven(replay),
    }
    
    # Analyze each scenario
    engine = CounterfactualEngine(replay)
    analyzer = DependencyAnalyzer(engine)
    
    all_reports = {}
    
    for name, decision_id in scenarios.items():
        print(f"Analyzing: {name}")
        print("-" * 60)
        
        report = analyzer.analyze_decision(decision_id)
        all_reports[name] = report
        
        print(report.render())
        print()
    
    # Summary comparison
    print("=" * 60)
    print("Cross-Scenario Comparison")
    print("=" * 60)
    print()
    
    # Header
    header = f"{'Scenario':20s}"
    for node_type in NodeType:
        header += f" {node_type.value[:8]:>9s}"
    header += f" {'HHI':>7s} {'SrcHHI':>7s}"
    print(header)
    print("-" * 60)
    
    for name, report in all_reports.items():
        row = f"{name:20s}"
        for node_type in NodeType:
            profile = report.profiles.get(node_type)
            if profile:
                row += f" {profile.influence_ratio:>8.1%}"
            else:
                row += f" {'---':>9s}"
        row += f" {report.herfindahl_index:>6.3f} {report.source_herfindahl_index:>6.3f}"
        print(row)
    
    print()
    
    # Interpretations
    print("Interpretations")
    print("-" * 60)
    
    for name, report in all_reports.items():
        dominant = report.dominant_type.value if report.dominant_type else "none"
        balance = "balanced" if report.herfindahl_index < 0.25 else "concentrated"
        print(f"  {name:20s}: dominant={dominant:15s} balance={balance} (HHI={report.herfindahl_index:.3f})")
    
    print()
    
    # Key insights
    print("Key Insights")
    print("-" * 60)
    
    # Find most imbalanced scenario
    most_imbalanced = max(all_reports.values(), key=lambda r: r.herfindahl_index)
    most_balanced = min(all_reports.values(), key=lambda r: r.herfindahl_index)
    
    most_imbalanced_name = [n for n, r in all_reports.items() if r is most_imbalanced][0]
    most_balanced_name = [n for n, r in all_reports.items() if r is most_balanced][0]
    
    print(f"  Most imbalanced: {most_imbalanced_name} (HHI={most_imbalanced.herfindahl_index:.3f})")
    print(f"  Most balanced:   {most_balanced_name} (HHI={most_balanced.herfindahl_index:.3f})")
    print()
    
    # Architecture recommendations
    print("Architecture Recommendations")
    print("-" * 60)
    
    for name, report in all_reports.items():
        if report.herfindahl_index > 0.4:
            dominant = report.dominant_type.value if report.dominant_type else "?"
            print(f"  {name}: CONCENTRATED on {dominant} — add more cognitive sources")
        elif report.herfindahl_index < 0.15:
            print(f"  {name}: BALANCED — architecture is healthy")
        else:
            print(f"  {name}: MODERATE — some imbalance but acceptable")
    
    print()
    print("=" * 60)
    print("EXP-EDI-001: COMPLETE")
    print("=" * 60)
    
    return all_reports


if __name__ == "__main__":
    results = run_experiment()
