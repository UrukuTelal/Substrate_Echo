"""EXP-DYN-004: Knowledge Competition Experiment.

Tests whether competing explanations survive, merge, split, or specialize
when exposed to mixed environments.

SC2 Integration:
    Three competing explanations for the same phenomenon:
    - Scout predicts expansion (scouts should gather intelligence)
    - Defense predicts safety (defenses should protect bases)
    - Random predicts unpredictability (mixed approach)

    Environments:
    - Early game: Scout-heavy (scout strategy wins)
    - Mid game: Balanced (both strategies partially work)
    - Late game: Defense-heavy (defense strategy wins)

    Measures:
    - Competition outcomes (survive, merge, retire)
    - Confidence trajectories under environmental pressure
    - Domain specialization (early vs late game)
    - Epistemic age of survivors vs retired
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.instrumentation import (
    EpistemicGovernance, GovernanceRule, GovernanceRuleType,
    EpistemicAgeAnalyzer, KnowledgeMigrationSystem,
    KnowledgeClass, KnowledgeItem, MigrationTrigger
)


def run_experiment():
    """Run knowledge competition experiment."""
    print("=" * 60)
    print("EXP-DYN-004: Knowledge Competition")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Initialize competing explanations
    # -----------------------------------------------------------------------
    print("Phase 1: Initialize Competing Explanations")
    print("-" * 60)

    # Governance rules
    governance = EpistemicGovernance()

    governance.add_rule(GovernanceRule(
        rule_id="promotion_rule",
        rule_type=GovernanceRuleType.PROMOTION,
        conditions={"min_confidence": 0.7, "min_evidence": 5},
        actions={"promote": True},
        priority=10,
    ))

    governance.add_rule(GovernanceRule(
        rule_id="retirement_rule",
        rule_type=GovernanceRuleType.RETIREMENT,
        conditions={"min_confidence": 0.2},
        actions={"retire": True},
        priority=5,
    ))

    governance.add_rule(GovernanceRule(
        rule_id="merger_rule",
        rule_type=GovernanceRuleType.MERGER,
        conditions={"max_confidence_diff": 0.2},
        actions={"merge": True},
        priority=3,
    ))

    # Epistemic age tracker
    age_analyzer = EpistemicAgeAnalyzer()

    # Knowledge items (competing explanations)
    items = {
        "scout_strategy": KnowledgeItem(
            item_id="scout_strategy",
            knowledge_class=KnowledgeClass.EXPLORATORY,
            confidence=0.5,
            evidence_count=0,
            successful_predictions=0,
            failed_predictions=0,
            source_agent="agent_1",
            source_tick=0,
        ),
        "defense_strategy": KnowledgeItem(
            item_id="defense_strategy",
            knowledge_class=KnowledgeClass.EXPLORATORY,
            confidence=0.5,
            evidence_count=0,
            successful_predictions=0,
            failed_predictions=0,
            source_agent="agent_2",
            source_tick=0,
        ),
        "random_strategy": KnowledgeItem(
            item_id="random_strategy",
            knowledge_class=KnowledgeClass.EXPLORATORY,
            confidence=0.3,
            evidence_count=0,
            successful_predictions=0,
            failed_predictions=0,
            source_agent="agent_3",
            source_tick=0,
        ),
    }

    # Track epistemic age
    for item_id in items:
        age_analyzer.track_item(item_id, birth_tick=0,
                                initial_class=KnowledgeClass.EXPLORATORY)

    print("  Scout Strategy: confidence=0.5 (gathers intelligence)")
    print("  Defense Strategy: confidence=0.5 (protects bases)")
    print("  Random Strategy: confidence=0.3 (unpredictable)")
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Early Game (ticks 0-49) - Scout-heavy environment
    # -----------------------------------------------------------------------
    print("Phase 2: Early Game (ticks 0-49) - Scout-heavy")
    print("-" * 60)

    for tick in range(50):
        # Scout strategy works well in early game
        if tick % 3 == 0:  # Every 3 ticks
            items["scout_strategy"].successful_predictions += 1
            items["scout_strategy"].evidence_count += 1
            items["scout_strategy"].confidence = min(
                1.0, items["scout_strategy"].confidence + 0.02
            )
            age_analyzer.record_prediction("scout_strategy", successful=True)
            age_analyzer.record_confirmation("scout_strategy")

        # Defense strategy partially works
        if tick % 5 == 0:  # Every 5 ticks
            items["defense_strategy"].successful_predictions += 1
            items["defense_strategy"].evidence_count += 1
            items["defense_strategy"].confidence = min(
                1.0, items["defense_strategy"].confidence + 0.005
            )
            age_analyzer.record_prediction("defense_strategy", successful=True)

        # Random strategy rarely works
        if tick % 10 == 0:  # Every 10 ticks
            items["random_strategy"].successful_predictions += 1
            items["random_strategy"].evidence_count += 1
            items["random_strategy"].confidence = min(
                1.0, items["random_strategy"].confidence + 0.003
            )
            age_analyzer.record_prediction("random_strategy", successful=True)

        # Update ticks
        age_analyzer.update_ticks("scout_strategy", tick)
        age_analyzer.update_ticks("defense_strategy", tick)
        age_analyzer.update_ticks("random_strategy", tick)

    # Competition at tick 50
    competition = governance.compete(
        list(items.values()), tick=50,
        context={"phase": "early_game"}
    )

    print(f"  Competition result: {competition.result.value}")
    print(f"  Survivors: {competition.survivors}")
    print(f"  Retired: {competition.retired}")
    print()

    # -----------------------------------------------------------------------
    # Phase 3: Mid Game (ticks 50-99) - Balanced environment
    # -----------------------------------------------------------------------
    print("Phase 3: Mid Game (ticks 50-99) - Balanced")
    print("-" * 60)

    for tick in range(50, 100):
        # Both scout and defense work
        if tick % 4 == 0:  # Every 4 ticks
            items["scout_strategy"].successful_predictions += 1
            items["scout_strategy"].evidence_count += 1
            items["scout_strategy"].confidence = min(
                1.0, items["scout_strategy"].confidence + 0.01
            )
            age_analyzer.record_prediction("scout_strategy", successful=True)

            items["defense_strategy"].successful_predictions += 1
            items["defense_strategy"].evidence_count += 1
            items["defense_strategy"].confidence = min(
                1.0, items["defense_strategy"].confidence + 0.01
            )
            age_analyzer.record_prediction("defense_strategy", successful=True)

        # Random strategy still rarely works
        if tick % 12 == 0:
            items["random_strategy"].successful_predictions += 1
            items["random_strategy"].evidence_count += 1
            items["random_strategy"].confidence = min(
                1.0, items["random_strategy"].confidence + 0.002
            )
            age_analyzer.record_prediction("random_strategy", successful=True)

        # Update ticks
        age_analyzer.update_ticks("scout_strategy", tick)
        age_analyzer.update_ticks("defense_strategy", tick)
        age_analyzer.update_ticks("random_strategy", tick)

    # Competition at tick 100
    competition = governance.compete(
        list(items.values()), tick=100,
        context={"phase": "mid_game"}
    )

    print(f"  Competition result: {competition.result.value}")
    print(f"  Survivors: {competition.survivors}")
    print(f"  Retired: {competition.retired}")
    print()

    # -----------------------------------------------------------------------
    # Phase 4: Late Game (ticks 100-149) - Defense-heavy environment
    # -----------------------------------------------------------------------
    print("Phase 4: Late Game (ticks 100-149) - Defense-heavy")
    print("-" * 60)

    for tick in range(100, 150):
        # Defense strategy now works best
        if tick % 3 == 0:  # Every 3 ticks
            items["defense_strategy"].successful_predictions += 1
            items["defense_strategy"].evidence_count += 1
            items["defense_strategy"].confidence = min(
                1.0, items["defense_strategy"].confidence + 0.02
            )
            age_analyzer.record_prediction("defense_strategy", successful=True)
            age_analyzer.record_confirmation("defense_strategy")

        # Scout strategy partially works
        if tick % 6 == 0:
            items["scout_strategy"].successful_predictions += 1
            items["scout_strategy"].evidence_count += 1
            items["scout_strategy"].confidence = min(
                1.0, items["scout_strategy"].confidence + 0.005
            )
            age_analyzer.record_prediction("scout_strategy", successful=True)

        # Random strategy rarely works
        if tick % 15 == 0:
            items["random_strategy"].successful_predictions += 1
            items["random_strategy"].evidence_count += 1
            items["random_strategy"].confidence = min(
                1.0, items["random_strategy"].confidence + 0.001
            )
            age_analyzer.record_prediction("random_strategy", successful=True)

        # Update ticks
        age_analyzer.update_ticks("scout_strategy", tick)
        age_analyzer.update_ticks("defense_strategy", tick)
        age_analyzer.update_ticks("random_strategy", tick)

    # Competition at tick 150
    competition = governance.compete(
        list(items.values()), tick=150,
        context={"phase": "late_game"}
    )

    print(f"  Competition result: {competition.result.value}")
    print(f"  Survivors: {competition.survivors}")
    print(f"  Retired: {competition.retired}")
    print()

    # -----------------------------------------------------------------------
    # Phase 5: Final Analysis
    # -----------------------------------------------------------------------
    print("Phase 5: Final Analysis")
    print("-" * 60)

    # Get governance report
    gov_report = governance.get_report()
    print(f"  Governance rules: {gov_report['total_rules']}")
    print(f"  Competition events: {gov_report['competition_events']}")
    print(f"  Retired items: {gov_report['retired_items']}")
    print()

    # Get epistemic age profiles
    print("  Epistemic Age Profiles:")
    for item_id in ["scout_strategy", "defense_strategy", "random_strategy"]:
        profile = age_analyzer.get_age_profile(item_id)
        if profile:
            print(f"    {item_id}: age={profile.epistemic_age:.3f}, "
                  f"predictions={profile.successful_predictions}, "
                  f"confirmations={profile.independent_confirmations}")

    print()

    # Competition history
    print("  Competition History:")
    for event in governance.get_competition_history():
        print(f"    Tick {event.tick}: {event.result.value}")
        print(f"      Survivors: {event.survivors}")
        if event.retired:
            print(f"      Retired: {event.retired}")

    print()
    print("=" * 60)
    print("EXP-DYN-004 Complete")
    print("=" * 60)

    return {
        "governance": gov_report,
        "age_profiles": {
            item_id: age_analyzer.get_age_profile(item_id).to_dict()
            for item_id in ["scout_strategy", "defense_strategy", "random_strategy"]
            if age_analyzer.get_age_profile(item_id)
        },
        "competition_events": [
            e.to_dict() for e in governance.get_competition_history()
        ],
    }


if __name__ == "__main__":
    run_experiment()
