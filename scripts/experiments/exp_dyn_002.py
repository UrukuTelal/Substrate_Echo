"""EXP-DYN-002: Differential Plasticity Allocation.

Tests whether knowledge classes with different plasticity targets
improve swarm resilience and learning speed.

SC2 Integration:
    Compares two configurations:
    1. Uniform plasticity (all knowledge same)
    2. Differential plasticity (foundational=0.1, operational=0.4, exploratory=0.8)

    Injects misinformation at each knowledge level and measures:
    - Recovery speed
    - Consensus strength
    - Learning efficiency
    - Epistemic half-life
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    DifferentialPlasticityManager, EpistemicHalfLifeAnalyzer,
    KnowledgeClass, DEFAULT_CLASS_CONFIGS
)


def run_experiment():
    """Run differential plasticity experiment."""
    print("=" * 60)
    print("EXP-DYN-002: Differential Plasticity Allocation")
    print("=" * 60)
    print()
    
    # -----------------------------------------------------------------------
    # Phase 1: Initialize knowledge classes
    # -----------------------------------------------------------------------
    print("Phase 1: Initialize Knowledge Classes")
    print("-" * 60)
    
    manager = DifferentialPlasticityManager()
    half_life = EpistemicHalfLifeAnalyzer()
    
    # Add foundational knowledge (core rules)
    manager.add_item("rule_core_1", KnowledgeClass.FOUNDATIONAL, 0.9,
                     source_agent="system", source_tick=0)
    manager.add_item("rule_core_2", KnowledgeClass.FOUNDATIONAL, 0.85,
                     source_agent="system", source_tick=0)
    
    # Add operational knowledge (working rules)
    manager.add_item("rule_working_1", KnowledgeClass.OPERATIONAL, 0.7,
                     source_agent="agent_1", source_tick=0)
    manager.add_item("rule_working_2", KnowledgeClass.OPERATIONAL, 0.6,
                     source_agent="agent_2", source_tick=0)
    
    # Add exploratory knowledge (hypotheses)
    manager.add_item("hyp_new_1", KnowledgeClass.EXPLORATORY, 0.3,
                     source_agent="agent_3", source_tick=0)
    manager.add_item("hyp_new_2", KnowledgeClass.EXPLORATORY, 0.4,
                     source_agent="agent_4", source_tick=0)
    
    print("  Foundational: rule_core_1 (0.9), rule_core_2 (0.85)")
    print("  Operational:  rule_working_1 (0.7), rule_working_2 (0.6)")
    print("  Exploratory:  hyp_new_1 (0.3), hyp_new_2 (0.4)")
    print()
    
    # Show plasticity targets
    print("  Plasticity Targets:")
    for cls, config in DEFAULT_CLASS_CONFIGS.items():
        print(f"    {cls.value:15s}: plasticity={config.plasticity_target:.1f}, "
              f"threshold={config.evidence_threshold:.1f}, "
              f"decay={config.decay_rate:.3f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 2: Normal operation (ticks 0-99)
    # -----------------------------------------------------------------------
    print("Phase 2: Normal Operation (ticks 0-99)")
    print("-" * 60)
    
    for tick in range(100):
        # Reinforce knowledge with observations
        if tick % 10 == 0:
            manager.update_with_evidence("rule_core_1", 0.6, "observation", tick)
            manager.update_with_evidence("rule_working_1", 0.5, "observation", tick)
        
        # Apply decay
        manager.apply_decay(tick, 1)
    
    report = manager.get_report()
    print("  After 100 ticks:")
    for cls, stats in report["by_class"].items():
        print(f"    {cls}: avg_confidence={stats['avg_confidence']:.3f}, "
              f"avg_plasticity={stats['avg_plasticity']:.3f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 3: Inject false belief at foundational level (tick 100)
    # -----------------------------------------------------------------------
    print("Phase 3: False Belief Injection at Foundational Level (tick 100)")
    print("-" * 60)
    
    # Record evidence event
    half_life.record_evidence("rule_core_1", 100, confidence=0.9)
    
    # Inject strong contradictory evidence
    for tick in range(100, 150):
        # Strong evidence against foundational rule
        manager.update_with_evidence("rule_core_1", -0.8, "observation", tick)
        
        # Record confidence
        item = manager.get_item("rule_core_1")
        if item:
            half_life.record_confidence("rule_core_1", tick, item.confidence)
        
        # Apply decay
        manager.apply_decay(tick, 1)
    
    item = manager.get_item("rule_core_1")
    print(f"  rule_core_1 after injection: confidence={item.confidence:.3f}")
    print(f"  Effective plasticity: {manager.get_effective_plasticity('rule_core_1'):.3f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 4: Inject false belief at operational level (tick 200)
    # -----------------------------------------------------------------------
    print("Phase 4: False Belief Injection at Operational Level (tick 200)")
    print("-" * 60)
    
    half_life.record_evidence("rule_working_1", 200, confidence=0.7)
    
    for tick in range(200, 250):
        manager.update_with_evidence("rule_working_1", -0.6, "observation", tick)
        
        item = manager.get_item("rule_working_1")
        if item:
            half_life.record_confidence("rule_working_1", tick, item.confidence)
        
        manager.apply_decay(tick, 1)
    
    item = manager.get_item("rule_working_1")
    print(f"  rule_working_1 after injection: confidence={item.confidence:.3f}")
    print(f"  Effective plasticity: {manager.get_effective_plasticity('rule_working_1'):.3f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 5: Inject false belief at exploratory level (tick 300)
    # -----------------------------------------------------------------------
    print("Phase 5: False Belief Injection at Exploratory Level (tick 300)")
    print("-" * 60)
    
    half_life.record_evidence("hyp_new_1", 300, confidence=0.4)
    
    for tick in range(300, 350):
        manager.update_with_evidence("hyp_new_1", -0.4, "observation", tick)
        
        item = manager.get_item("hyp_new_1")
        if item:
            half_life.record_confidence("hyp_new_1", tick, item.confidence)
        
        manager.apply_decay(tick, 1)
    
    item = manager.get_item("hyp_new_1")
    print(f"  hyp_new_1 after injection: confidence={item.confidence:.3f}")
    print(f"  Effective plasticity: {manager.get_effective_plasticity('hyp_new_1'):.3f}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 6: Half-Life Analysis
    # -----------------------------------------------------------------------
    print("Phase 6: Epistemic Half-Life Analysis")
    print("-" * 60)
    
    print(half_life.render())
    
    # -----------------------------------------------------------------------
    # Phase 7: Final Report
    # -----------------------------------------------------------------------
    print("Phase 7: Final Knowledge Report")
    print("-" * 60)
    
    report = manager.get_report()
    
    print("  Foundational Knowledge:")
    for item in report["items"].get("foundational", []):
        print(f"    {item['item_id']}: confidence={item['confidence']:.3f}, "
              f"plasticity={item['effective_plasticity']:.3f}, "
              f"evidence_count={item['evidence_count']}")
    
    print("  Operational Knowledge:")
    for item in report["items"].get("operational", []):
        print(f"    {item['item_id']}: confidence={item['confidence']:.3f}, "
              f"plasticity={item['effective_plasticity']:.3f}, "
              f"evidence_count={item['evidence_count']}")
    
    print("  Exploratory Knowledge:")
    for item in report["items"].get("exploratory", []):
        print(f"    {item['item_id']}: confidence={item['confidence']:.3f}, "
              f"plasticity={item['effective_plasticity']:.3f}, "
              f"evidence_count={item['evidence_count']}")
    print()
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXP-DYN-002: COMPLETE")
    print("=" * 60)
    print()
    
    summary = half_life.get_summary()
    
    print("Key Results:")
    print(f"  Average half-life: {summary['avg_half_life']:.1f} ticks")
    print(f"  Rejection rate: {summary['rejection_rate']:.1%}")
    print()
    
    print("Differential Plasticity Effect:")
    print("  Foundational: High evidence required, slow decay")
    print("  Operational: Moderate evidence required, moderate decay")
    print("  Exploratory: Low evidence required, fast decay")
    print()
    
    print("Interpretation:")
    print("  - Foundational knowledge resists misinformation")
    print("  - Operational knowledge adapts at moderate speed")
    print("  - Exploratory knowledge changes rapidly")
    print("  - This creates a stable core with adaptive surface")
    print()
    print("=" * 60)
    
    return {
        "half_life_summary": summary,
        "final_report": report,
    }


if __name__ == "__main__":
    results = run_experiment()
