"""Affordance Tracer — Generates possible actions with consequences.

Sits between perception and action. Given entity model + world state,
generates candidate actions with estimated success probabilities and
expected consequences.

Architecture:
    Entity Model + World State
              |
              v
    Affordance Generator
              |
              v
    Candidate Actions (each with)
      - success_probability
      - expected_reward
      - resource_cost
      - risk
      - information_gain
              |
              v
    Action Evaluation (scored)
              |
              v
    Ranked Candidates → Action Bridge → Governance → Execution
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class SC2ActionType(Enum):
    """Available SC2 actions as affordance categories."""
    EXPAND = "expand"
    BUILD_ARMY = "build_army"
    BUILD_ECONOMY = "build_economy"
    DEFEND = "defend"
    ATTACK = "attack"
    SCOUT = "scout"
    TECH_UP = "tech_up"
    RETREAT = "retreat"
    HOLD = "hold"


class CostLevel(Enum):
    """Resource cost categories."""
    FREE = 0        # no cost (hold, observe)
    LOW = 1         # < 200 minerals
    MEDIUM = 2      # 200-500 minerals
    HIGH = 3        # 500+ minerals or army loss
    CRITICAL = 4    # game-ending risk


@dataclass
class AffordanceCandidate:
    """A possible action with estimated consequences."""
    action_type: SC2ActionType
    description: str = ""

    # Probability estimates
    success_probability: float = 0.5  # [0, 1]
    expected_reward: float = 0.0      # net resource gain expected
    resource_cost: float = 0.0        # minerals/gas required
    cost_level: CostLevel = CostLevel.LOW

    # Risk
    risk: float = 0.0                 # [0, 1] chance of significant loss
    army_exposure: float = 0.0        # fraction of army at risk

    # Information
    information_gain: float = 0.0     # [0, 1] how much we learn
    reduces_uncertainty: bool = False

    # Source
    confidence: float = 0.5           # how confident are we in this candidate
    evidence_count: int = 0           # how many observations support this

    # Context
    requires_unit: Optional[str] = None  # e.g., "SCV", "Marine"
    requires_structure: Optional[str] = None
    prerequisite_actions: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Epistemic action score.
        
        Score = expected_reward * confidence * success_probability
              + information_gain * 0.3
              - risk * cost_level * 0.2
        """
        reward_score = self.expected_reward * self.confidence * self.success_probability
        info_score = self.information_gain * 0.3
        risk_penalty = self.risk * self.cost_level.value * 0.2
        return reward_score + info_score - risk_penalty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_type.value,
            "description": self.description,
            "success_prob": round(self.success_probability, 3),
            "expected_reward": round(self.expected_reward, 3),
            "cost": round(self.resource_cost, 1),
            "cost_level": self.cost_level.name,
            "risk": round(self.risk, 3),
            "info_gain": round(self.information_gain, 3),
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 3),
        }


@dataclass
class WorldState:
    """Current world state for affordance generation."""
    # Economy
    minerals: float = 0.0
    vespene: float = 0.0
    supply_used: float = 0.0
    supply_cap: float = 0.0
    workers: int = 0
    bases: int = 0
    production_buildings: int = 0

    # Military
    army_count: int = 0
    army_value: float = 0.0

    # Information
    enemy_units_seen: int = 0
    enemy_bases_known: int = 0
    last_scout_tick: int = 0
    uncertainty: float = 0.5  # [0, 1]

    # Timing
    current_tick: int = 0
    game_phase: str = "early"  # early, mid, late

    @classmethod
    def from_botai(cls, bot, tick: int = 0) -> 'WorldState':
        """Create from BotAI instance."""
        from sc2.constants import UnitTypeId
        workers = len(bot.units.of_type(UnitTypeId.SCV))
        army = bot.units.exclude_type(UnitTypeId.SCV)
        bases = len(bot.units.of_type(UnitTypeId.COMMANDCENTER))
        production = (
            len(bot.units.of_type(UnitTypeId.BARRACKS))
            + len(bot.units.of_type(UnitTypeId.FACTORY))
            + len(bot.units.of_type(UnitTypeId.STARPORT))
        )

        # Determine game phase
        if tick < 300:
            phase = "early"
        elif tick < 800:
            phase = "mid"
        else:
            phase = "late"

        return cls(
            minerals=bot.minerals,
            vespene=bot.vespene,
            supply_used=bot.supply_used,
            supply_cap=bot.supply_cap,
            workers=workers,
            bases=bases,
            production_buildings=production,
            army_count=len(army),
            army_value=sum(u.health + u.shield for u in army),
            current_tick=tick,
            game_phase=phase,
        )


class AffordanceTracer:
    """Generates possible actions with estimated consequences.
    
    Given world state and entity model, produces ranked candidate actions
    with success probabilities, expected rewards, and risk estimates.
    
    Usage:
        tracer = AffordanceTracer()
        
        world = WorldState.from_botai(bot, tick)
        candidates = tracer.generate(world, entity_model)
        
        for c in candidates:
            print(f"{c.action_type.value}: score={c.score:.3f}")
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def generate(self, world: WorldState,
                 entities: Any = None) -> List[AffordanceCandidate]:
        """Generate all possible affordances for current state."""
        candidates = []

        # Economy actions
        candidates.extend(self._economy_affordances(world))

        # Military actions
        candidates.extend(self._military_affordances(world, entities))

        # Information actions
        candidates.extend(self._information_affordances(world))

        # Always available
        candidates.append(AffordanceCandidate(
            action_type=SC2ActionType.HOLD,
            description="Hold position, maintain current state",
            success_probability=1.0,
            expected_reward=0.0,
            resource_cost=0.0,
            cost_level=CostLevel.FREE,
            risk=0.0,
            information_gain=0.0,
            confidence=1.0,
        ))

        # Score and sort
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Record
        self._history.append({
            "tick": world.current_tick,
            "candidates": [c.to_dict() for c in candidates[:5]],
        })

        return candidates

    def _economy_affordances(self, w: WorldState) -> List[AffordanceCandidate]:
        """Generate economy-related affordances."""
        candidates = []

        # Expand
        if w.bases < 4 and w.minerals >= 400 and w.workers >= 14:
            success = min(0.9, 0.5 + w.workers * 0.02)
            reward = 200.0 if w.bases < 2 else 100.0
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.EXPAND,
                description=f"Build base #{w.bases + 1}",
                success_probability=success,
                expected_reward=reward,
                resource_cost=400.0,
                cost_level=CostLevel.HIGH,
                risk=0.15,
                confidence=0.8,
                requires_unit="SCV",
            ))

        # Build economy (SCVs)
        if w.workers < 60 and w.minerals >= 50 and w.supply_used < w.supply_cap:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.BUILD_ECONOMY,
                description=f"Train worker ({w.workers}/60)",
                success_probability=0.95,
                expected_reward=50.0,
                resource_cost=50.0,
                cost_level=CostLevel.LOW,
                risk=0.0,
                confidence=0.9,
                requires_structure="CommandCenter",
            ))

        # Build supply
        if w.supply_used >= w.supply_cap - 3 and w.minerals >= 100:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.BUILD_ECONOMY,
                description=f"Build supply depot ({w.supply_used}/{w.supply_cap})",
                success_probability=0.95,
                expected_reward=0.0,
                resource_cost=100.0,
                cost_level=CostLevel.LOW,
                risk=0.0,
                confidence=0.95,
                reduces_uncertainty=False,
                requires_unit="SCV",
            ))

        # Tech up
        if w.production_buildings >= 2 and w.minerals >= 150 and w.vespene >= 100:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.TECH_UP,
                description="Advance tech tree",
                success_probability=0.85,
                expected_reward=80.0,
                resource_cost=250.0,
                cost_level=CostLevel.MEDIUM,
                risk=0.05,
                confidence=0.7,
            ))

        return candidates

    def _military_affordances(self, w: WorldState,
                              entities: Any = None) -> List[AffordanceCandidate]:
        """Generate military-related affordances."""
        candidates = []

        # Build army
        if w.minerals >= 50 and w.supply_used < w.supply_cap:
            army_rate = 1.0 if w.production_buildings >= 2 else 0.5
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.BUILD_ARMY,
                description=f"Train military units (army={w.army_count})",
                success_probability=0.9,
                expected_reward=60.0 * army_rate,
                resource_cost=50.0,
                cost_level=CostLevel.LOW,
                risk=0.0,
                confidence=0.85,
            ))

        # Attack
        if w.army_count >= 10:
            # Estimate attack success based on army size and intelligence
            success = min(0.8, 0.3 + w.army_count * 0.01)
            risk = max(0.1, 0.5 - w.army_count * 0.01)
            info_gain = 0.3 if w.enemy_bases_known == 0 else 0.1

            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.ATTACK,
                description=f"Attack with {w.army_count} units",
                success_probability=success,
                expected_reward=150.0,
                resource_cost=0.0,
                cost_level=CostLevel.FREE,
                risk=risk,
                army_exposure=min(1.0, w.army_count / max(1, w.army_count + 10)),
                information_gain=info_gain,
                confidence=0.6 if w.uncertainty > 0.5 else 0.8,
            ))

        # Defend
        if w.army_count >= 3:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.DEFEND,
                description="Move army to defensive position",
                success_probability=0.85,
                expected_reward=30.0,
                resource_cost=0.0,
                cost_level=CostLevel.FREE,
                risk=0.05,
                confidence=0.9,
            ))

        # Retreat (if army is small and we've seen enemies)
        if w.army_count <= 5 and w.enemy_units_seen > 5:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.RETREAT,
                description="Retreat to defensive position",
                success_probability=0.9,
                expected_reward=10.0,
                resource_cost=0.0,
                cost_level=CostLevel.FREE,
                risk=0.0,
                confidence=0.8,
            ))

        return candidates

    def _information_affordances(self, w: WorldState) -> List[AffordanceCandidate]:
        """Generate information-gathering affordances."""
        candidates = []

        # Scout
        scout_value = 0.5 if w.uncertainty > 0.5 else 0.2
        ticks_since_scout = w.current_tick - w.last_scout_tick
        urgency = min(1.0, ticks_since_scout / 200)

        candidates.append(AffordanceCandidate(
            action_type=SC2ActionType.SCOUT,
            description="Send worker to scout enemy",
            success_probability=0.7,
            expected_reward=0.0,
            resource_cost=0.0,
            cost_level=CostLevel.FREE,
            risk=0.1,  # scout might die
            information_gain=scout_value * urgency,
            reduces_uncertainty=w.uncertainty > 0.4,
            confidence=0.7,
            requires_unit="SCV",
        ))

        return candidates

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def render(self, candidates: List[AffordanceCandidate]) -> str:
        """Render candidate actions."""
        lines = []
        lines.append("Affordance Candidates:")
        lines.append("-" * 60)
        for i, c in enumerate(candidates):
            marker = ">>>" if i == 0 else "   "
            lines.append(
                f"  {marker} {c.action_type.value:12s} "
                f"score={c.score:.3f} "
                f"prob={c.success_probability:.2f} "
                f"risk={c.risk:.2f} "
                f"cost={c.cost_level.name:8s} "
                f"info={c.information_gain:.2f}"
            )
            lines.append(f"       {c.description}")
        return "\n".join(lines)
