"""EXP-INST-001: Research Instruments Validation.

Validates the three research instruments:
1. CausalReplay — expand decisions into reasoning trees
2. CounterfactualEngine — remove observations, compare reasoning
3. KnowledgeProvenanceGraph — track discovery lineage

SC2 Integration:
    Simulates a full reasoning cycle with observations, hypotheses,
    predictions, council deliberations, and decisions. Then uses
    each instrument to inspect the reasoning.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    CausalReplay, CounterfactualEngine, KnowledgeProvenanceGraph,
    Modification, ModificationType,
    NodeType
)


def run_experiment():
    """Run the research instruments validation."""
    print("=" * 60)
    print("EXP-INST-001: Research Instruments Validation")
    print("=" * 60)
    print()
    
    # -----------------------------------------------------------------------
    # Phase 1: Build a reasoning chain with CausalReplay
    # -----------------------------------------------------------------------
    print("Phase 1: Building Reasoning Chain")
    print("-" * 60)
    
    replay = CausalReplay()
    
    # Simulate a full reasoning cycle
    # Tick 100: Observation
    obs1 = replay.record_observation(100, {
        "type": "enemy_unit",
        "unit_type": "scout",
        "count": 1,
        "position": "own_base",
    })
    
    # Tick 101: Feature extraction
    feat1 = replay.record_feature(101, {
        "feature": "early_scout",
        "timing": "before_4_min",
        "aggression_signal": 0.3,
    }, caused_by=obs1.node_id)
    
    # Tick 102: Hypotheses generated
    hyp1 = replay.record_hypothesis(102, {
        "hypothesis": "expansion",
        "confidence": 0.45,
    }, caused_by=feat1.node_id, weight=0.3)
    
    hyp2 = replay.record_hypothesis(102, {
        "hypothesis": "proxy_attack",
        "confidence": 0.35,
    }, caused_by=feat1.node_id, weight=0.3)
    
    hyp3 = replay.record_hypothesis(102, {
        "hypothesis": "rush",
        "confidence": 0.20,
    }, caused_by=feat1.node_id, weight=0.3)
    
    # Tick 103: Cultural prior applied
    prior1 = replay.record_cultural_prior(103, {
        "prior": "early_scout_likely_expansion",
        "confidence_boost": 0.15,
        "applied_to": "expansion",
    }, caused_by=hyp1.node_id, weight=0.4)
    
    # Tick 104: Council deliberation
    council1 = replay.record_council(104, {
        "council": "reality",
        "vote": "expansion",
        "confidence": 0.62,
        "reasoning": "Scout timing matches expansion, not rush",
    }, caused_by=prior1.node_id, weight=0.35)
    
    # Tick 105: Trust update
    trust1 = replay.record_trust(105, {
        "agent": "cultural_prior",
        "trust_before": 0.6,
        "trust_after": 0.65,
        "reason": "prior helped identify expansion",
    }, caused_by=council1.node_id, weight=0.2)
    
    # Tick 106: Prediction
    pred1 = replay.record_prediction(106, {
        "prediction": "expansion_within_40s",
        "confidence": 0.62,
        "evidence": "scout + prior + council",
    }, caused_by=council1.node_id, weight=0.25)
    
    # Tick 107: Decision
    decision1 = replay.record_decision(107, {
        "action": "expand_economy",
        "reasoning": "Expanding based on scout analysis",
        "confidence": 0.62,
    }, caused_by=pred1.node_id, weight=1.0)
    
    # Tick 110: Action executed
    action1 = replay.record_action(110, {
        "action_type": "build_command_center",
        "result": "success",
    }, caused_by=decision1.node_id, weight=0.8)
    
    print(f"Recorded {len(replay._events)} events in reasoning chain")
    print()
    
    # Expand the decision into a tree
    tree = replay.expand_decision(decision1.node_id)
    print("Reasoning Tree for Decision 'expand_economy':")
    print()
    print(tree.render())
    print()
    
    # Get most influential nodes
    influential = tree.get_most_influential(3)
    print("Most Influential Nodes:")
    for node in influential:
        data_str = str(node.data)[:40]
        print(f"  [{node.node_type.value}] {data_str} (weight={node.weight:.2f})")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 2: Counterfactual Analysis
    # -----------------------------------------------------------------------
    print("Phase 2: Counterfactual Analysis")
    print("-" * 60)
    
    engine = CounterfactualEngine(replay)
    
    # Scenario: What if we never saw the enemy scout?
    mod_remove_obs = Modification(
        mod_type=ModificationType.REMOVE_OBSERVATION,
        target_id=obs1.node_id,
        description="What if we never saw the enemy scout?",
    )
    
    report1 = engine.compare(tree, mod_remove_obs)
    print("Counterfactual: Remove Scout Observation")
    print()
    print(report1.render())
    print()
    
    # Scenario: What if cultural prior was disabled?
    mod_disable_prior = Modification(
        mod_type=ModificationType.DISABLE_CULTURAL_PRIOR,
        target_id="early_scout_likely_expansion",
        description="What if the cultural prior was disabled?",
    )
    
    report2 = engine.compare(tree, mod_disable_prior)
    print("Counterfactual: Disable Cultural Prior")
    print()
    print(report2.render())
    print()
    
    # Scenario: What if trust in cultural prior was low?
    mod_low_trust = Modification(
        mod_type=ModificationType.CHANGE_TRUST,
        target_id="cultural_prior",
        new_value=0.2,
        description="What if trust in cultural prior was low?",
    )
    
    report3 = engine.compare(tree, mod_low_trust)
    print("Counterfactual: Low Trust in Cultural Prior")
    print()
    print(report3.render())
    print()
    
    # -----------------------------------------------------------------------
    # Phase 3: Knowledge Provenance Graph
    # -----------------------------------------------------------------------
    print("Phase 3: Knowledge Provenance Graph")
    print("-" * 60)
    
    graph = KnowledgeProvenanceGraph()
    
    # Record observations
    obs_a = graph.record_observation(100, {"enemy_scout": True})
    obs_b = graph.record_observation(110, {"no_proxy_buildings": True})
    obs_c = graph.record_observation(120, {"expansion_timing": "4_min"})
    
    # Record hypotheses
    hyp_a = graph.record_hypothesis(105, {
        "hypothesis": "enemy_expanding",
        "confidence": 0.7,
    }, parents=[obs_a.node_id, obs_b.node_id])
    
    hyp_b = graph.record_hypothesis(115, {
        "hypothesis": "enemy_rushing",
        "confidence": 0.3,
    }, parents=[obs_a.node_id])
    
    # Record predictions
    pred_a = graph.record_prediction(110, {
        "prediction": "expansion_in_40s",
        "confidence": 0.7,
    }, parents=[hyp_a.node_id])
    
    # Record discoveries
    disc_a = graph.record_discovery(130, {
        "rule": "early_scout_means_expansion",
        "confidence": 0.75,
    }, parents=[hyp_a.node_id, pred_a.node_id])
    
    disc_b = graph.record_discovery(140, {
        "rule": "expansion_timing_predictable",
        "confidence": 0.8,
    }, parents=[disc_a.node_id, obs_c.node_id])
    
    # Record a prior
    prior_a = graph.record_prior(150, {
        "prior": "expansion_bias",
        "confidence": 0.7,
    }, parents=[disc_a.node_id])
    
    # Record decisions
    dec_a = graph.record_decision(160, {
        "action": "expand_economy",
        "confidence": 0.8,
    }, parents=[prior_a.node_id, disc_b.node_id])
    
    # Record validation
    graph.record_validation(170, pred_a.node_id, {
        "outcome": "enemy_expanded",
        "correct": True,
    })
    
    # Record an overturning
    hyp_c = graph.record_hypothesis(125, {
        "hypothesis": "enemy_proxying",
        "confidence": 0.15,
    }, parents=[obs_a.node_id])
    
    graph.record_overturning(135, hyp_c.node_id, hyp_a.node_id)
    
    print("Provenance Graph:")
    print()
    print(graph.render())
    
    # Get lineage of a discovery
    print("Lineage of 'early_scout_means_expansion':")
    print("-" * 60)
    lineage = graph.get_lineage(disc_a.node_id)
    for node in lineage:
        data_str = str(node.data)[:50]
        print(f"  [{node.node_type}] tick={node.tick} {data_str}")
    print()
    
    # Get influence of an observation
    print("Influence of first observation:")
    print("-" * 60)
    influence = graph.get_influence(obs_a.node_id)
    print(f"  Total descendants: {influence['total_descendants']}")
    print(f"  Decisions influenced: {influence['decisions_influenced']}")
    print(f"  Discoveries led to: {influence['discoveries_led_to']}")
    print(f"  By type: {influence['by_type']}")
    print()
    
    # Get discovery trees
    print("Discovery Trees:")
    print("-" * 60)
    trees = graph.get_discovery_trees()
    for tree_data in trees:
        disc = tree_data["discovery"]
        print(f"  Discovery: {disc['data'].get('rule', 'unknown')}")
        print(f"    Lineage depth: {len(tree_data['lineage'])}")
        print(f"    Influence score: {tree_data['influence']['influence_score']:.1f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 4: Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXP-INST-001: COMPLETE")
    print("=" * 60)
    print()
    print("CausalReplay:")
    print(f"  - Expanded {len(replay._events)} events into reasoning tree")
    print(f"  - Tree depth: {tree.depth}")
    print(f"  - Most influential: {influential[0].node_type.value}")
    print()
    print("CounterfactualEngine:")
    print(f"  - Ran 3 counterfactual analyses")
    print(f"  - Scenario 1 (remove obs): decision changed = {report1.decision_changed}")
    print(f"  - Scenario 2 (disable prior): decision changed = {report2.decision_changed}")
    print(f"  - Scenario 3 (low trust): decision changed = {report3.decision_changed}")
    print()
    print("KnowledgeProvenanceGraph:")
    print(f"  - Tracked {len(graph._nodes)} nodes, {len(graph._edges)} edges")
    print(f"  - {len(graph.get_discovery_trees())} discovery trees")
    print(f"  - {len(graph.get_roots())} root nodes")
    print()
    
    # Return results
    return {
        "causal_replay": {
            "events": len(replay._events),
            "tree_depth": tree.depth,
            "most_influential_type": influential[0].node_type.value,
        },
        "counterfactual": {
            "scenarios_run": 3,
            "scenario_1_changed": report1.decision_changed,
            "scenario_2_changed": report2.decision_changed,
            "scenario_3_changed": report3.decision_changed,
        },
        "provenance": {
            "nodes": len(graph._nodes),
            "edges": len(graph._edges),
            "discoveries": len(graph.get_discovery_trees()),
            "roots": len(graph.get_roots()),
        },
    }


if __name__ == "__main__":
    results = run_experiment()
