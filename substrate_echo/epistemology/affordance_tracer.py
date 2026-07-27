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
import numpy as np

from substrate_echo.embodiments.sc2.unit_classifier import (
    UnitClassifier, Role, Movement, AttackCapability,
)


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

    # ── Drive Affinities ──────────────────────────────────────────
    # How much this action serves each need type.
    # Key = NeedType.value (string), Value = affinity [0, 1]
    # These are SC2-domain knowledge — this is where StarCraft lives.
    need_affinities: Dict[str, float] = field(default_factory=dict)

    def drive_utility(self, deficits: Dict[str, float]) -> float:
        """Dot product of deficit vector × affinity vector.

        deficits: mapping from need_type_str → deficit [0, 1]
        Returns a scalar utility in [0, 1].
        """
        if not self.need_affinities or not deficits:
            return 0.0
        total = 0.0
        for need_str, affinity in self.need_affinities.items():
            deficit = deficits.get(need_str, 0.0)
            total += deficit * affinity
        return total

    @property
    def score(self) -> float:
        """Epistemic action score (without drives).
        
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
            "need_affinities": {
                k: round(v, 3) for k, v in self.need_affinities.items()
            },
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

    # Spatial resources — distinct mineral patches and gas geysers
    mineral_fields: List[Dict] = field(default_factory=list)  # [{pos, amount, distance_to_base}]
    gas_geysers: List[Dict] = field(default_factory=list)    # [{pos, amount, has_refinery, distance_to_base}]

    # Military
    army_count: int = 0
    army_value: float = 0.0
    anti_air_count: int = 0          # units that can target air
    dual_attack_count: int = 0       # units that hit both air+ground
    enemy_air_count: int = 0         # visible enemy air units

    # Information
    enemy_units_seen: int = 0
    enemy_bases_known: int = 0
    last_scout_tick: int = 0
    uncertainty: float = 0.5  # [0, 1]

    # Timing
    current_tick: int = 0
    game_phase: str = "early"  # early, mid, late

    # Terrain — from SC2 pathing_grid, terrain_height, visibility
    map_revealed: float = 0.0        # fraction of walkable map visible to us
    terrain_complexity: float = 0.0   # fraction of pathing grid blocked
    cliff_density: float = 0.0        # fraction of cells with height discontinuity
    cliff_traversable_count: int = 0  # our units that can cross cliffs
    visibility_advantage: float = 0.5 # our visible tiles / (our + enemy visible)

    @classmethod
    def from_botai(cls, bot, tick: int = 0) -> 'WorldState':
        """Create from BotAI instance — fully race-agnostic."""
        workers = len(bot.workers)
        army = len(bot.units)
        bases = len(bot.townhalls)
        # Production = structures that aren't townhalls
        townhall_ids = {s.type_id for s in bot.townhalls}
        production = sum(1 for s in bot.units.structure if s.type_id not in townhall_ids)

        # Spatial resources — distinct mineral patches and gas geysers
        mineral_fields = []
        for mf in bot.state.mineral_field:
            mineral_fields.append({
                "pos": (mf.position.x, mf.position.y),
                "amount": mf.mineral_contents if hasattr(mf, 'mineral_contents') else 0,
                "distance_to_base": min(
                    (mf.position.distance_to(th.position) for th in bot.townhalls),
                    default=999
                ) if bot.townhalls else 999,
            })
        gas_geysers = []
        for g in bot.geysers:
            gas_geysers.append({
                "pos": (g.position.x, g.position.y),
                "amount": g.vespene_contents if hasattr(g, 'vespene_contents') else 0,
                "distance_to_base": min(
                    (g.position.distance_to(th.position) for th in bot.townhalls),
                    default=999
                ) if bot.townhalls else 999,
            })

        # Game phase from tick
        if tick < 300:
            phase = "early"
        elif tick < 800:
            phase = "mid"
        else:
            phase = "late"

        # ── Army composition (race-agnostic) ──
        uc = UnitClassifier()
        worker_ids = {u.tag for u in bot.workers}
        supply_ids = {u.tag for u in bot.units if any(
            kw in u.name.upper() for kw in ("OVERLORD", "OVERSEER", "OBSERVER"))}
        spawned_names = {"LOCUST", "BROODLING", "INTERCEPTOR", "AUTOTURRET"}

        combat_units = []
        for u in bot.units:
            if u.tag in worker_ids or u.tag in supply_ids:
                continue
            if u.is_structure or not u.can_attack:
                continue
            if u.name.upper() in spawned_names:
                continue
            info = uc.classify(u)
            if info and (Role.ARMY in info.roles or Role.SCOUT in info.roles):
                combat_units.append(u)

        anti_air = 0
        dual = 0
        for u in combat_units:
            info = uc.classify(u)
            if info:
                caps = info.attack_caps
                can_hit_air = AttackCapability.GVA in caps or AttackCapability.AVA in caps
                can_hit_ground = AttackCapability.GVG in caps or AttackCapability.AVG in caps
                if can_hit_air:
                    anti_air += 1
                if can_hit_air and can_hit_ground:
                    dual += 1

        # Enemy air units visible
        enemy_air = 0
        for eu in bot.known_enemy_units:
            if not eu.is_structure and eu.type_id.name.upper() not in spawned_names:
                einfo = uc.classify(eu)
                if einfo and einfo.movement == Movement.AIR:
                    enemy_air += 1

        # ── Cliff-traversable ground units ──
        cliff_traversable = 0
        for u in combat_units:
            info = uc.classify(u)
            if info and uc.has_cliff_traversal(info):
                cliff_traversable += 1

        # ── Terrain metrics from SC2 API ──
        map_revealed = 0.0
        terrain_complexity = 0.0
        cliff_density = 0.0
        visibility_advantage = 0.5

        try:
            pathing = bot.state.pathing_grid
            height = bot.state.terrain_height
            visibility = bot.state.visibility

            if pathing is not None:
                pathing_arr = np.array(pathing, dtype=np.float32)
                total_cells = pathing_arr.size
                if total_cells > 0:
                    terrain_complexity = float(
                        np.sum(pathing_arr != 0)) / total_cells

                # Map revealed
                if visibility is not None:
                    vis_arr = np.array(visibility, dtype=np.float32)
                    walkable = pathing_arr == 0
                    walkable_count = max(1, int(np.sum(walkable)))
                    visible_walkable = int(np.sum((vis_arr > 0) & walkable))
                    map_revealed = min(1.0, visible_walkable / walkable_count)

                    # Visibility advantage
                    enemy_vis_cells = 0
                    for eu in bot.known_enemy_units:
                        try:
                            ex = int(eu.position.x)
                            ey = int(eu.position.y)
                            r = 10
                            x0, x1 = max(0, ex-r), min(vis_arr.shape[0], ex+r)
                            y0, y1 = max(0, ey-r), min(vis_arr.shape[1], ey+r)
                            enemy_vis_cells += int(np.sum(
                                walkable[x0:x1, y0:y1]))
                        except (IndexError, AttributeError):
                            pass
                    total_vis = visible_walkable + max(1, enemy_vis_cells)
                    visibility_advantage = visible_walkable / total_vis

            # Cliff density
            if height is not None:
                height_arr = np.array(height, dtype=np.float32)
                dh_row = np.abs(np.diff(height_arr, axis=0))
                dh_col = np.abs(np.diff(height_arr, axis=1))
                cliff_threshold = 1.5
                cliff_cells = (
                    int(np.sum(dh_row > cliff_threshold))
                    + int(np.sum(dh_col > cliff_threshold))
                )
                max_pairs = (
                    (height_arr.shape[0] - 1) * height_arr.shape[1]
                    + height_arr.shape[0] * (height_arr.shape[1] - 1))
                if max_pairs > 0:
                    cliff_density = min(1.0, cliff_cells / max_pairs)

        except (AttributeError, TypeError, ValueError):
            pass

        return cls(
            minerals=bot.minerals,
            vespene=bot.vespene,
            supply_used=bot.supply_used,
            supply_cap=bot.supply_cap,
            workers=workers,
            bases=bases,
            production_buildings=production,
            army_count=army,
            army_value=sum(u.health + u.shield for u in bot.units),
            anti_air_count=anti_air,
            dual_attack_count=dual,
            enemy_air_count=enemy_air,
            enemy_units_seen=len(bot.known_enemy_units),
            enemy_bases_known=len(bot.known_enemy_structures),
            current_tick=tick,
            game_phase=phase,
            mineral_fields=mineral_fields,
            gas_geysers=gas_geysers,
            map_revealed=map_revealed,
            terrain_complexity=terrain_complexity,
            cliff_density=cliff_density,
            cliff_traversable_count=cliff_traversable,
            visibility_advantage=visibility_advantage,
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
            # need_affinities provided by DriveAffinityLearner
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
        if w.bases < 4 and w.minerals >= 300 and w.workers >= 8:
            success = min(0.9, 0.5 + w.workers * 0.02)
            reward = 200.0 if w.bases < 2 else 100.0
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.EXPAND,
                description=f"Build base #{w.bases + 1}",
                success_probability=success,
                expected_reward=reward,
                resource_cost=300.0,
                cost_level=CostLevel.MEDIUM,
                risk=0.15,
                confidence=0.8,
                requires_unit="SCV",
                # need_affinities provided by AffordanceModel (learned)
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
                # need_affinities provided by AffordanceModel (learned)
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
                # need_affinities provided by AffordanceModel (learned)
            ))

        # Tech up — build structures or research upgrades
        if w.minerals >= 100 and w.production_buildings >= 1:
            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.TECH_UP,
                description="Build tech structure or research upgrade",
                success_probability=0.85,
                expected_reward=80.0,
                resource_cost=150.0,
                cost_level=CostLevel.MEDIUM,
                risk=0.05,
                confidence=0.7,
                # need_affinities provided by AffordanceModel (learned)
            ))

        return candidates

    def _military_affordances(self, w: WorldState,
                              entities: Any = None) -> List[AffordanceCandidate]:
        """Generate military-related affordances.

        Uses army composition data (anti-air, dual-attack, enemy air)
        to adjust attack success probability and risk.
        """
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
                # need_affinities provided by AffordanceModel (learned)
            ))

        # Attack — adjusted by army composition and terrain
        if w.army_count >= 5:
            success = min(0.8, 0.3 + w.army_count * 0.01)
            risk = max(0.1, 0.5 - w.army_count * 0.01)
            info_gain = 0.3 if w.enemy_bases_known == 0 else 0.1

            # Composition adjustment: if enemy has air, we need anti-air
            if w.enemy_air_count > 0:
                if w.anti_air_count == 0:
                    # No anti-air vs enemy air — risky attack
                    success *= 0.5
                    risk = min(0.9, risk + 0.3)
                    info_gain += 0.2  # high info: we're vulnerable
                elif w.dual_attack_count >= w.enemy_air_count:
                    # Good anti-air coverage — confident attack
                    success = min(0.9, success + 0.1)
                    risk = max(0.05, risk - 0.1)
                else:
                    # Partial anti-air — moderate adjustment
                    success = min(0.85, success + 0.05)

            # No enemy air and we have dual-attack units — bonus
            if w.enemy_air_count == 0 and w.dual_attack_count > 0:
                success = min(0.9, success + 0.05)

            # Terrain adjustment: high cliff density penalizes ground-only
            # attacks, but cliff-traversable units bypass this penalty
            if w.cliff_density > 0.1:
                total_army = max(1, w.army_count)
                cliff_ratio = w.cliff_traversable_count / total_army
                if cliff_ratio < 0.2:
                    # Mostly non-traversable army on cliff-heavy map
                    success = max(0.1, success * (1.0 - w.cliff_density * 0.3))
                    risk = min(0.9, risk + w.cliff_density * 0.1)
                elif cliff_ratio > 0.5:
                    # Good cliff traversal — terrain advantage
                    success = min(0.95, success + w.cliff_density * 0.1)

            # Visibility advantage: better vision = better attacks
            if w.visibility_advantage > 0.6:
                success = min(0.95, success + 0.05)
            elif w.visibility_advantage < 0.3:
                success = max(0.1, success * 0.8)
                risk = min(0.9, risk + 0.1)

            desc = f"Attack with {w.army_count} units"
            if w.enemy_air_count > 0:
                desc += f" (enemy air: {w.enemy_air_count}, our AA: {w.anti_air_count})"
            if w.cliff_density > 0.1:
                desc += f" (cliffs: {w.cliff_density:.0%}, traverse: {w.cliff_traversable_count})"

            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.ATTACK,
                description=desc,
                success_probability=success,
                expected_reward=150.0,
                resource_cost=0.0,
                cost_level=CostLevel.FREE,
                risk=risk,
                army_exposure=min(1.0, w.army_count / max(1, w.army_count + 10)),
                information_gain=info_gain,
                confidence=0.6 if w.uncertainty > 0.5 else 0.8,
                # need_affinities provided by AffordanceModel (learned)
            ))

        # Defend
        if w.army_count >= 3:
            defend_success = 0.85
            # If enemy air is near and we lack anti-air, defend is more urgent
            if w.enemy_air_count > 0 and w.anti_air_count == 0:
                defend_success = 0.7  # harder to defend without AA

            candidates.append(AffordanceCandidate(
                action_type=SC2ActionType.DEFEND,
                description="Move army to defensive position",
                success_probability=defend_success,
                expected_reward=30.0,
                resource_cost=0.0,
                cost_level=CostLevel.FREE,
                risk=0.05,
                confidence=0.9,
                # need_affinities provided by AffordanceModel (learned)
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
                # need_affinities provided by AffordanceModel (learned)
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
            # need_affinities provided by AffordanceModel (learned)
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
