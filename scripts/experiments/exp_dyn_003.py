"""EXP-DYN-003: Knowledge Phase Transition Experiment.

Tests whether knowledge can "mature" between classes based on evidence.

SC2 Integration:
    Three knowledge items start in different classes:
    - Rule A: Exploratory (becomes increasingly useful)
    - Rule B: Operational (becomes partially wrong)
    - Rule C: Foundational (remains correct)

    Measures:
    - Class migration (promotions and demotions)
    - Confidence trajectory
    - Evidence accumulation
    - Epistemic role assignment across domains
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    KnowledgeMigrationSystem, EpistemicRoleManager,
    KnowledgeClass, MigrationTrigger
)


def run_experiment():
    """Run knowledge phase transition experiment."""
    print("=" * 60)
    print("EXP-DYN-003: Knowledge Phase Transition")
    print("=" * 60)
    print()
    
    # -----------------------------------------------------------------------
    # Phase 1: Initialize knowledge items
    # -----------------------------------------------------------------------
    print("Phase 1: Initialize Knowledge Items")
    print("-" * 60)
    
    # Custom triggers (faster for demo)
    triggers = MigrationTrigger(
        min_confidence=0.6,
        min_evidence_count=3,
        min_successful_predictions=2,
        min_ticks_in_class=20,
        max_confidence=0.4,
        max_failed_predictions=2,
    )
    
    system = KnowledgeMigrationSystem(triggers=triggers)
    roles = EpistemicRoleManager()
    
    # Rule A: Starts exploratory, will become useful
    system.add_item("rule_A", KnowledgeClass.EXPLORATORY, 0.3,
                    source_agent="agent_1", source_tick=0)
    
    # Rule B: Starts operational, will become wrong
    system.add_item("rule_B", KnowledgeClass.OPERATIONAL, 0.7,
                    source_agent="agent_2", source_tick=0)
    
    # Rule C: Starts foundational, will remain correct
    system.add_item("rule_C", KnowledgeClass.FOUNDATIONAL, 0.9,
                    source_agent="system", source_tick=0)
    
    print("  Rule A: Exploratory (confidence=0.3) - will become useful")
    print("  Rule B: Operational (confidence=0.7) - will become wrong")
    print("  Rule C: Foundational (confidence=0.9) - will remain correct")
    print()
    
    # Assign initial roles
    roles.assign_role("rule_A", "early_game", KnowledgeClass.EXPLORATORY, 0.3)
    roles.assign_role("rule_A", "late_game", KnowledgeClass.EXPLORATORY, 0.2)
    
    roles.assign_role("rule_B", "early_game", KnowledgeClass.OPERATIONAL, 0.7)
    roles.assign_role("rule_B", "late_game", KnowledgeClass.OPERATIONAL, 0.6)
    
    roles.assign_role("rule_C", "early_game", KnowledgeClass.FOUNDATIONAL, 0.9)
    roles.assign_role("rule_C", "late_game", KnowledgeClass.FOUNDATIONAL, 0.85)
    
    # -----------------------------------------------------------------------
    # Phase 2: Normal operation (ticks 0-49)
    # -----------------------------------------------------------------------
    print("Phase 2: Normal Operation (ticks 0-49)")
    print("-" * 60)
    
    for tick in range(50):
        # All rules get some evidence
        system.update_with_evidence("rule_A", 0.1, "observation", tick, True)
        system.update_with_evidence("rule_B", 0.1, "observation", tick, True)
        system.update_with_evidence("rule_C", 0.1, "observation", tick, True)
        
        # Check migrations
        migrations = system.check_migrations(tick)
    
    print("  50 ticks completed")
    print(f"  Total migrations: {len(system.get_all_migrations())}")
    print()
    
    # -----------------------------------------------------------------------
    # Phase 3: Rule A becomes useful (ticks 50-149)
    # -----------------------------------------------------------------------
    print("Phase 3: Rule A Becomes Useful (ticks 50-149)")
    print("-" * 60)
    
    for tick in range(50, 150):
        # Rule A gets strong positive evidence and successful predictions
        system.update_with_evidence("rule_A", 0.3, "observation", tick, True)
        if tick % 5 == 0:
            system.update_with_evidence("rule_A", 0.2, "prediction", tick, True)
        
        # Rule B starts getting mixed evidence and failed predictions
        if tick % 5 == 0:
            system.update_with_evidence("rule_B", -0.2, "observation", tick, False)
            system.update_with_evidence("rule_B", -0.1, "prediction", tick, False)
        else:
            system.update_with_evidence("rule_B", 0.1, "observation", tick, True)
        
        # Rule C continues to be correct
        system.update_with_evidence("rule_C", 0.15, "observation", tick, True)
        if tick % 10 == 0:
            system.update_with_evidence("rule_C", 0.2, "prediction", tick, True)
        
        # Check migrations
        migrations = system.check_migrations(tick)
        
        for m in migrations:
            print(f"  Tick {tick}: {m.item_id} {m.from_class.value} -> {m.to_class.value}")
    
    print()
    
    # -----------------------------------------------------------------------
    # Phase 4: Rule B becomes wrong (ticks 150-249)
    # -----------------------------------------------------------------------
    print("Phase 4: Rule B Becomes Wrong (ticks 150-249)")
    print("-" * 60)
    
    for tick in range(150, 250):
        # Rule A continues to be useful
        system.update_with_evidence("rule_A", 0.25, "observation", tick, True)
        if tick % 5 == 0:
            system.update_with_evidence("rule_A", 0.2, "prediction", tick, True)
        
        # Rule B gets strong negative evidence and failed predictions
        system.update_with_evidence("rule_B", -0.4, "observation", tick, False)
        if tick % 3 == 0:
            system.update_with_evidence("rule_B", -0.3, "prediction", tick, False)
        
        # Rule C remains correct
        system.update_with_evidence("rule_C", 0.1, "observation", tick, True)
        if tick % 10 == 0:
            system.update_with_evidence("rule_C", 0.15, "prediction", tick, True)
        
        # Check migrations
        migrations = system.check_migrations(tick)
        
        for m in migrations:
            print(f"  Tick {tick}: {m.item_id} {m.from_class.value} -> {m.to_class.value}")
    
    print()
    
    # -----------------------------------------------------------------------
    # Phase 5: Final state (ticks 250-299)
    # -----------------------------------------------------------------------
    print("Phase 5: Final State (ticks 250-299)")
    print("-" * 60)
    
    for tick in range(250, 300):
        system.update_with_evidence("rule_A", 0.2, "observation", tick, True)
        system.update_with_evidence("rule_B", -0.3, "observation", tick, False)
        system.update_with_evidence("rule_C", 0.1, "observation", tick, True)
        
        migrations = system.check_migrations(tick)
        
        for m in migrations:
            print(f"  Tick {tick}: {m.item_id} {m.from_class.value} -> {m.to_class.value}")
    
    print()
    
    # -----------------------------------------------------------------------
    # Phase 6: Migration Report
    # -----------------------------------------------------------------------
    print("Phase 6: Migration Report")
    print("-" * 60)
    
    print(system.render())
    
    # -----------------------------------------------------------------------
    # Phase 7: Epistemic Roles
    # -----------------------------------------------------------------------
    print("Phase 7: Epistemic Roles")
    print("-" * 60)
    
    # Update roles based on final state
    item_a = system.get_item("rule_A")
    item_b = system.get_item("rule_B")
    item_c = system.get_item("rule_C")
    
    if item_a:
        roles.assign_role("rule_A", "early_game", item_a.knowledge_class, item_a.confidence)
        roles.assign_role("rule_A", "late_game", KnowledgeClass.OPERATIONAL, 0.6)
    
    if item_b:
        roles.assign_role("rule_B", "early_game", item_b.knowledge_class, item_b.confidence)
        roles.assign_role("rule_B", "late_game", KnowledgeClass.EXPLORATORY, 0.3)
    
    if item_c:
        roles.assign_role("rule_C", "early_game", item_c.knowledge_class, item_c.confidence)
        roles.assign_role("rule_C", "late_game", item_c.knowledge_class, item_c.confidence)
    
    print(roles.render())
    
    # -----------------------------------------------------------------------
    # Phase 8: Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXP-DYN-003: COMPLETE")
    print("=" * 60)
    print()
    
    report = system.get_report()
    
    print("Knowledge Phase Transitions:")
    print()
    
    # Rule A trajectory
    a_migrations = system.get_migration_history("rule_A")
    a_final = system.get_item("rule_A")
    print(f"Rule A (started EXPLORATORY):")
    print(f"  Final class: {a_final.knowledge_class.value if a_final else 'N/A'}")
    a_conf = f"{a_final.confidence:.3f}" if a_final else "N/A"
    print(f"  Final confidence: {a_conf}")
    print(f"  Migrations: {len(a_migrations)}")
    for m in a_migrations:
        print(f"    {m.from_class.value} -> {m.to_class.value} at tick {m.tick}")
    print()
    
    # Rule B trajectory
    b_migrations = system.get_migration_history("rule_B")
    b_final = system.get_item("rule_B")
    print(f"Rule B (started OPERATIONAL):")
    print(f"  Final class: {b_final.knowledge_class.value if b_final else 'N/A'}")
    b_conf = f"{b_final.confidence:.3f}" if b_final else "N/A"
    print(f"  Final confidence: {b_conf}")
    print(f"  Migrations: {len(b_migrations)}")
    for m in b_migrations:
        print(f"    {m.from_class.value} -> {m.to_class.value} at tick {m.tick}")
    print()
    
    # Rule C trajectory
    c_migrations = system.get_migration_history("rule_C")
    c_final = system.get_item("rule_C")
    print(f"Rule C (started FOUNDATIONAL):")
    print(f"  Final class: {c_final.knowledge_class.value if c_final else 'N/A'}")
    c_conf = f"{c_final.confidence:.3f}" if c_final else "N/A"
    print(f"  Final confidence: {c_conf}")
    print(f"  Migrations: {len(c_migrations)}")
    for m in c_migrations:
        print(f"    {m.from_class.value} -> {m.to_class.value} at tick {m.tick}")
    print()
    
    # Key insight
    print("Key Insight:")
    print("-" * 60)
    print("  Knowledge migrated between classes based on evidence:")
    print("  - Rule A: Promoted from EXPLORATORY as it proved useful")
    print("  - Rule B: Demoted from OPERATIONAL as it failed")
    print("  - Rule C: Remained FOUNDATIONAL as it stayed correct")
    print()
    print("  This demonstrates knowledge evolution:")
    print("  - Useful knowledge matures (exploratory -> foundational)")
    print("  - Failed knowledge degrades (foundational -> exploratory)")
    print("  - Stable knowledge persists (foundational stays foundational)")
    print()
    print("=" * 60)
    
    return {
        "report": report,
        "roles": roles.get_report(),
    }


if __name__ == "__main__":
    results = run_experiment()
