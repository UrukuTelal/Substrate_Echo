"""EXP-EPIST-002: Blind Agent vs Belief Agent vs Full Epistemic Agent.

Compares three agent configurations to test whether epistemic
architecture improves world prediction and action quality.

Configuration 1 — Blind Agent:
    No entity model, no predictions, no governance.
    Pure affordance-based action selection.

Configuration 2 — Belief Agent:
    Entity model with evidence-backed relationships.
    Predictions inform action but no governance gate.

Configuration 3 — Full Epistemic Agent:
    Entity model + predictions + governance gate.
    Actions scored by confidence × prediction × affordance.

Measures:
    - Action diversity (entropy)
    - Prediction accuracy
    - Confidence calibration
    - Governance intervention rate
    - Survival time (proxy for decision quality)

SC2 Integration:
    All three agents run the same bot code with different
    epistemology layers enabled/disabled.
"""
import sys
import os
import time
import random
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from substrate_echo.epistemology.affordance_tracer import (
    AffordanceTracer, AffordanceCandidate, SC2ActionType,
    CostLevel, WorldState
)
from substrate_echo.epistemology.action_bridge import EpistemicActionBridge
from substrate_echo.epistemology.governance_gate import GovernanceGate, GovernanceDecision
from substrate_echo.epistemology.entity_model import (
    EntityModel, EvidenceType, RelationshipType
)
from substrate_echo.epistemology.chain_recorder import EpistemicChainRecorder


@dataclass
class AgentConfig:
    """Agent configuration for comparison."""
    name: str
    use_entity_model: bool = False
    use_predictions: bool = False
    use_governance: bool = False
    use_action_bridge: bool = False


@dataclass
class SimulatedEntity:
    """Simulated enemy entity for testing."""
    entity_id: str
    behavior: str  # "aggressive", "passive", "scout", "expand"
    strength: float = 0.5
    position: float = 0.5
    alive: bool = True

    def get_action(self, tick: int) -> str:
        """Deterministic behavior pattern."""
        if not self.alive:
            return "dead"
        if self.behavior == "aggressive":
            return "attack" if tick % 10 < 6 else "defend"
        elif self.behavior == "passive":
            return "expand" if tick % 15 < 4 else "defend"
        elif self.behavior == "scout":
            return "scout" if tick % 8 < 5 else "expand"
        else:
            return "expand" if tick % 12 < 5 else "attack"


@dataclass
class SimulatedWorld:
    """Simulated game world for testing."""
    tick: int = 0
    minerals: float = 100.0
    vespene: float = 0.0
    workers: int = 12
    army: int = 0
    bases: int = 1
    supply_used: int = 12
    supply_cap: int = 15
    uncertainty: float = 0.7
    enemy_units_seen: int = 0
    last_scout_tick: int = 0

    def tick_forward(self, action: str):
        """Simulate one tick of game state."""
        self.tick += 1

        # Economy grows
        self.minerals += self.workers * 0.8

        # Workers gather
        if action == "build_economy" and self.minerals >= 50:
            self.workers += 1
            self.minerals -= 50
            self.supply_used += 1

        # Army grows
        if action == "build_army" and self.minerals >= 50 and self.supply_used < self.supply_cap:
            self.army += 1
            self.minerals -= 50
            self.supply_used += 1

        # Expand
        if action == "expand" and self.minerals >= 400:
            self.bases += 1
            self.minerals -= 400
            self.workers += 1  # SCV for new base

        # Scout reduces uncertainty
        if action == "scout":
            self.uncertainty = max(0.1, self.uncertainty - 0.15)
            self.last_scout_tick = self.tick
            self.enemy_units_seen += random.randint(0, 3)

        # Build supply
        if self.supply_used >= self.supply_cap - 2 and self.minerals >= 100:
            self.supply_cap += 8
            self.minerals -= 100

        # Attack reduces army
        if action == "attack" and self.army > 0:
            losses = random.randint(0, max(1, self.army // 3))
            self.army = max(0, self.army - losses)

        # Cap values
        self.minerals = min(5000, self.minerals)
        self.uncertainty = max(0.05, min(1.0, self.uncertainty))

    def to_world_state(self) -> WorldState:
        return WorldState(
            minerals=self.minerals,
            vespene=self.vespene,
            supply_used=self.supply_used,
            supply_cap=self.supply_cap,
            workers=self.workers,
            bases=self.bases,
            army_count=self.army,
            army_value=self.army * 100,
            enemy_units_seen=self.enemy_units_seen,
            last_scout_tick=self.last_scout_tick,
            uncertainty=self.uncertainty,
            current_tick=self.tick,
            game_phase="early" if self.tick < 100 else "mid",
        )

    def get_observation(self) -> Dict[str, Any]:
        return {
            "minerals": self.minerals,
            "workers": self.workers,
            "army": self.army,
            "bases": self.bases,
            "supply_used": self.supply_used,
            "supply_cap": self.supply_cap,
            "uncertainty": self.uncertainty,
        }


def run_agent(config: AgentConfig, ticks: int = 200) -> Dict[str, Any]:
    """Run a single agent configuration."""
    world = SimulatedWorld()
    tracer = AffordanceTracer()
    chain = EpistemicChainRecorder()

    # Components based on config
    entity_model = EntityModel() if config.use_entity_model else None
    bridge = EpistemicActionBridge() if config.use_action_bridge else None
    governance = GovernanceGate() if config.use_governance else None

    # Create simulated enemy
    enemy_entity = None
    simulated_enemy = None
    if entity_model:
        enemy_entity = entity_model.create_entity("enemy_1", embodiment="sim")
        simulated_enemy = SimulatedEntity("enemy_1", behavior="aggressive")
        enemy_entity.add_evidence(
            EvidenceType.OBSERVED_BEHAVIOR,
            "Initial contact",
            tick=0,
        )

    actions_taken = []
    prediction_outcomes = []
    governance_interventions = []

    for tick in range(ticks):
        # ── OBSERVE ──
        obs = world.get_observation()
        ws = world.to_world_state()

        # ── ENTITY MODEL UPDATE ──
        entity_confidence = 0.5
        if entity_model and enemy_entity and simulated_enemy:
            # Simulate enemy behavior
            enemy_action = simulated_enemy.get_action(tick)
            if enemy_action == "attack":
                enemy_entity.add_evidence(
                    EvidenceType.ATTACK,
                    "Enemy attacked",
                    supports=RelationshipType.ADVERSARIAL,
                    tick=tick,
                )
            elif enemy_action == "expand":
                enemy_entity.add_evidence(
                    EvidenceType.EXPANSION,
                    "Enemy expanded",
                    supports=RelationshipType.COMPETITIVE,
                    tick=tick,
                )

            entity_model.update_all(tick)
            dominant, conf = enemy_entity.get_dominant_relationship()
            entity_confidence = conf

        # ── AFFORDANCE GENERATION ──
        candidates = tracer.generate(ws, entity_model)

        # ── ACTION SELECTION ──
        selected = candidates[0] if candidates else None
        action_type = selected.action_type.value if selected else "hold"

        # ── ACTION BRIDGE (if enabled) ──
        if bridge and selected:
            scores = bridge.score_candidates(
                candidates,
                entity_confidence=entity_confidence,
            )
            if scores:
                best = scores[0]
                action_type = best.action_type

        # ── GOVERNANCE CHECK (if enabled) ──
        if governance and selected:
            cost_level = selected.cost_level.value if hasattr(selected.cost_level, 'value') else 0
            threat = enemy_entity.get_threat_level() if enemy_entity else 0.0

            verdict = governance.check(
                action_type=action_type,
                confidence=entity_confidence,
                cost_level=cost_level,
                uncertainty=ws.uncertainty,
                army_exposure=selected.army_exposure,
                threat_level=threat,
            )

            governance_interventions.append({
                "tick": tick,
                "original": action_type,
                "decision": verdict.decision.value,
                "adjusted": verdict.adjusted_action,
            })

            if verdict.decision == GovernanceDecision.DENY:
                action_type = "hold"
            elif verdict.decision == GovernanceDecision.MODIFY and verdict.adjusted_action:
                action_type = verdict.adjusted_action

        # ── RECORD ──
        chain.record_action(tick, action_type, [], "simulation")
        actions_taken.append(action_type)

        # ── EXECUTE ──
        world.tick_forward(action_type)

        # ── OUTCOME ──
        chain.record_outcome(tick, world.get_observation())

    # ── ANALYSIS ──
    action_counts = {}
    for a in actions_taken:
        action_counts[a] = action_counts.get(a, 0) + 1

    # Action entropy
    total = len(actions_taken)
    entropy = 0.0
    for count in action_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    # Prediction accuracy from bridge
    pred_accuracy = bridge.prediction_accuracy if bridge else 0.5

    # Governance stats
    gov_denials = sum(1 for g in governance_interventions
                      if g["decision"] == "deny")
    gov_mods = sum(1 for g in governance_interventions
                   if g["decision"] == "modify")

    return {
        "config": config.name,
        "ticks": ticks,
        "final_state": {
            "minerals": world.minerals,
            "workers": world.workers,
            "army": world.army,
            "bases": world.bases,
        },
        "action_distribution": action_counts,
        "action_entropy": entropy,
        "prediction_accuracy": pred_accuracy,
        "governance_denials": gov_denials,
        "governance_modifications": gov_mods,
        "governance_interventions": governance_interventions,
    }


def run_experiment():
    """Run comparison experiment."""
    print("=" * 60)
    print("EXP-EPIST-002: Blind vs Belief vs Full Epistemic Agent")
    print("=" * 60)
    print()

    configs = [
        AgentConfig(
            name="Blind Agent",
            use_entity_model=False,
            use_predictions=False,
            use_governance=False,
            use_action_bridge=False,
        ),
        AgentConfig(
            name="Belief Agent",
            use_entity_model=True,
            use_predictions=False,
            use_governance=False,
            use_action_bridge=False,
        ),
        AgentConfig(
            name="Full Epistemic Agent",
            use_entity_model=True,
            use_predictions=True,
            use_governance=True,
            use_action_bridge=True,
        ),
    ]

    results = []
    for config in configs:
        print(f"Running: {config.name}...")
        result = run_agent(config, ticks=200)
        results.append(result)
        print(f"  Done. Actions: {result['action_distribution']}")
        print()

    # ── COMPARISON ──
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    # Header
    print(f"  {'Metric':30s} {'Blind':>12s} {'Belief':>12s} {'Full Epistemic':>15s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*15}")

    # Action diversity (entropy)
    entropies = [r["action_entropy"] for r in results]
    print(f"  {'Action Entropy':30s} {entropies[0]:12.3f} {entropies[1]:12.3f} {entropies[2]:15.3f}")

    # Final army
    armies = [r["final_state"]["army"] for r in results]
    print(f"  {'Final Army':30s} {armies[0]:12d} {armies[1]:12d} {armies[2]:15d}")

    # Final bases
    bases = [r["final_state"]["bases"] for r in results]
    print(f"  {'Final Bases':30s} {bases[0]:12d} {bases[1]:12d} {bases[2]:15d}")

    # Final workers
    workers = [r["final_state"]["workers"] for r in results]
    print(f"  {'Final Workers':30s} {workers[0]:12d} {workers[1]:12d} {workers[2]:15d}")

    # Governance interventions
    denials = [r["governance_denials"] for r in results]
    mods = [r["governance_modifications"] for r in results]
    print(f"  {'Governance Denials':30s} {denials[0]:12d} {denials[1]:12d} {denials[2]:15d}")
    print(f"  {'Governance Modifications':30s} {mods[0]:12d} {mods[1]:12d} {mods[2]:15d}")

    # Action distribution
    print()
    print("  Action Distribution:")
    all_actions = set()
    for r in results:
        all_actions.update(r["action_distribution"].keys())

    for action in sorted(all_actions):
        counts = [r["action_distribution"].get(action, 0) for r in results]
        pcts = [c / 200 * 100 for c in counts]
        print(f"    {action:15s}: {pcts[0]:5.1f}% {pcts[1]:5.1f}% {pcts[2]:5.1f}%")

    print()
    print("=" * 60)
    print("EXP-EPIST-002 Complete")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_experiment()
