"""Tactical Brain — Observe, hypothesize, experiment, learn.

The missing link between "I see enemy air" and "I should build hydras."

Architecture:
    BattleState          — full snapshot of health, attacks, saturation, enemy comp
    EnemyComposition     — what the enemy has (type counts, total value, upgrades)
    CounterHypothesis    — "unit X counters enemy composition Y" (confidence from outcomes)
    Experiment           — a test: "I will build Z units and see if it beats Y"
    TacticalBrain        — orchestrates observe → hypothesize → experiment → evaluate

Key principles:
    - NO hardcoded counter knowledge. Everything is learned from battle outcomes.
    - Per-base worker saturation from SC2 API (ideal_harvesters), not assumed averages.
    - Mid-game adaptation AND cross-game learning.

Usage:
    brain = TacticalBrain()

    # Each tick:
    state = brain.capture_state(bot, tick)
    brain.analyze_battles(bot, tick)

    # When choosing what to build:
    suggestion = brain.suggest_unit_composition(my_composition, enemy_composition)
    # suggestion = {"HYDRALISK": 6, "ROACH": 4} or None if no hypothesis yet

    # After battle:
    brain.record_battle_outcome(my_units, enemy_units, won, tick)

    # Persistence:
    brain.save("data/tactical_brain.json")
    brain.load("data/tactical_brain.json")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import time
from pathlib import Path


# ── Battle State ─────────────────────────────────────────────────

@dataclass
class UnitSnapshot:
    """State of a single unit at a point in time."""
    tag: int
    name: str
    health: float
    health_max: float
    shield: float
    shield_max: float
    is_attacking: bool
    engaged_target_tag: int = 0
    position_x: float = 0.0
    position_y: float = 0.0
    is_flying: bool = False
    can_attack_air: bool = False
    can_attack_ground: bool = False
    attack_upgrade: int = 0
    armor_upgrade: int = 0
    movement_speed: float = 0.0

    @property
    def is_alive(self) -> bool:
        return self.health > 0 or self.shield > 0

    @property
    def effective_hp(self) -> float:
        return self.health + self.shield

    @property
    def hp_ratio(self) -> float:
        total = self.health_max + self.shield_max
        return self.effective_hp / total if total > 0 else 0.0


@dataclass
class BaseSaturation:
    """Worker saturation at a single base."""
    townhall_tag: int
    townhall_name: str
    assigned_harvesters: int  # current workers
    ideal_harvesters: int     # SC2 API optimal (e.g. 16 for minerals)
    surplus_harvesters: int   # excess workers
    mineral_patches: int = 0
    has_vespene: bool = False
    position_x: float = 0.0
    position_y: float = 0.0

    @property
    def saturation_ratio(self) -> float:
        """How saturated this base is [0, ~1.5+]. 1.0 = perfectly saturated."""
        if self.ideal_harvesters == 0:
            return 0.0
        return self.assigned_harvesters / self.ideal_harvesters

    @property
    def is_undersaturated(self) -> bool:
        return self.assigned_harvesters < self.ideal_harvesters

    @property
    def is_oversaturated(self) -> bool:
        return self.surplus_harvesters > 0


@dataclass
class EnemyComposition:
    """Snapshot of visible enemy forces."""
    tick: int = 0
    unit_counts: Dict[str, int] = field(default_factory=dict)  # name -> count
    total_value: float = 0.0  # sum of health+shield of all units
    air_count: int = 0
    ground_count: int = 0
    anti_air_count: int = 0  # units that can attack air
    has_detection: bool = False
    has_casters: bool = False
    average_upgrade_level: float = 0.0
    unit_tags: Set[int] = field(default_factory=set)  # track which units we've seen

    @property
    def air_ratio(self) -> float:
        total = self.air_count + self.ground_count
        return self.air_count / total if total > 0 else 0.0

    @property
    def ground_ratio(self) -> float:
        total = self.air_count + self.ground_count
        return self.ground_count / total if total > 0 else 0.0


@dataclass
class BattleState:
    """Complete game state snapshot."""
    tick: int = 0

    # Own forces
    own_units: List[UnitSnapshot] = field(default_factory=list)
    own_army_count: int = 0
    own_army_value: float = 0.0
    own_air_count: int = 0
    own_ground_count: int = 0
    own_anti_air_count: int = 0

    # Own economy
    minerals: float = 0.0
    vespene: float = 0.0
    supply_used: int = 0
    supply_cap: int = 0
    worker_count: int = 0
    base_count: int = 0
    base_saturations: List[BaseSaturation] = field(default_factory=list)
    total_ideal_harvesters: int = 0
    total_assigned_harvesters: int = 0

    # Enemy
    enemy: EnemyComposition = field(default_factory=EnemyComposition)

    # Events
    units_under_attack: List[int] = field(default_factory=list)  # tags of own units being attacked
    units_attacking: List[int] = field(default_factory=list)     # tags of own units that are attacking


# ── Counter Hypothesis ───────────────────────────────────────────

class HypothesisStatus(Enum):
    ACTIVE = "active"           # being tested
    VALIDATED = "validated"     # evidence supports it
    REJECTED = "rejected"       # evidence contradicts it
    STALE = "stale"             # too old, needs retesting


@dataclass
class CounterHypothesis:
    """'Unit type X counters enemy composition Y'.

    confidence starts at 0.5, moves toward 1.0 with battle wins,
    toward 0.0 with battle losses.
    """
    id: str = ""
    counter_unit: str = ""                          # what to build (e.g. "HYDRALISK")
    targets_enemy: Dict[str, int] = field(default_factory=dict)  # enemy comp it targets
    confidence: float = 0.5
    wins: int = 0
    losses: int = 0
    total_tests: int = 0
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    created_tick: int = 0
    last_tested_tick: int = 0
    last_outcome: str = ""  # "win", "loss", "inconclusive"

    @property
    def win_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.wins / self.total_tests

    def update(self, won: bool, tick: int):
        """Update hypothesis from battle outcome."""
        self.total_tests += 1
        self.last_tested_tick = tick
        if won:
            self.wins += 1
            self.last_outcome = "win"
            # Confidence moves toward 1.0
            self.confidence = min(0.95, self.confidence + 0.1 * (1.0 - self.confidence))
        else:
            self.losses += 1
            self.last_outcome = "loss"
            # Confidence moves toward 0.0
            self.confidence = max(0.05, self.confidence - 0.1 * self.confidence)

        # Auto-validate or reject after enough tests
        if self.total_tests >= 5 and self.win_rate >= 0.7:
            self.status = HypothesisStatus.VALIDATED
        elif self.total_tests >= 5 and self.win_rate <= 0.3:
            self.status = HypothesisStatus.REJECTED
        else:
            self.status = HypothesisStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "counter_unit": self.counter_unit,
            "targets_enemy": self.targets_enemy,
            "confidence": round(self.confidence, 4),
            "wins": self.wins,
            "losses": self.losses,
            "total_tests": self.total_tests,
            "win_rate": round(self.win_rate, 4),
            "status": self.status.value,
            "created_tick": self.created_tick,
            "last_tested_tick": self.last_tested_tick,
            "last_outcome": self.last_outcome,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CounterHypothesis:
        h = cls(
            id=d["id"],
            counter_unit=d["counter_unit"],
            targets_enemy=d.get("targets_enemy", {}),
            confidence=d.get("confidence", 0.5),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            total_tests=d.get("total_tests", 0),
            status=HypothesisStatus(d.get("status", "active")),
            created_tick=d.get("created_tick", 0),
            last_tested_tick=d.get("last_tested_tick", 0),
            last_outcome=d.get("last_outcome", ""),
        )
        return h


# ── Experiment ───────────────────────────────────────────────────

class ExperimentPhase(Enum):
    QUEUED = "queued"           # waiting to start
    BUILDING = "building"       # producing the counter units
    OBSERVING = "observing"     # waiting for battle outcome
    COMPLETE = "complete"       # outcome recorded


@dataclass
class Experiment:
    """A test: 'build N of unit X vs enemy composition Y and see what happens'.

    Tracks the full lifecycle from hypothesis → production → battle → outcome.
    """
    id: str = ""
    hypothesis_id: str = ""
    unit_type: str = ""
    target_count: int = 0       # how many we're trying to build
    built_count: int = 0        # how many we've built so far
    enemy_snapshot: Dict[str, int] = field(default_factory=dict)  # enemy comp when experiment started
    phase: ExperimentPhase = ExperimentPhase.QUEUED
    created_tick: int = 0
    started_tick: int = 0
    completed_tick: int = 0
    outcome: str = ""           # "win", "loss", "inconclusive"
    result_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "unit_type": self.unit_type,
            "target_count": self.target_count,
            "built_count": self.built_count,
            "enemy_snapshot": self.enemy_snapshot,
            "phase": self.phase.value,
            "created_tick": self.created_tick,
            "started_tick": self.started_tick,
            "completed_tick": self.completed_tick,
            "outcome": self.outcome,
            "result_notes": self.result_notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Experiment:
        return cls(
            id=d["id"],
            hypothesis_id=d.get("hypothesis_id", ""),
            unit_type=d.get("unit_type", ""),
            target_count=d.get("target_count", 0),
            built_count=d.get("built_count", 0),
            enemy_snapshot=d.get("enemy_snapshot", {}),
            phase=ExperimentPhase(d.get("phase", "queued")),
            created_tick=d.get("created_tick", 0),
            started_tick=d.get("started_tick", 0),
            completed_tick=d.get("completed_tick", 0),
            outcome=d.get("outcome", ""),
            result_notes=d.get("result_notes", ""),
        )


# ── Tactical Brain ───────────────────────────────────────────────

TACTICAL_BRAIN_PATH = str(Path(__file__).parent.parent.parent / "data" / "tactical_brain.json")

# Enemy composition signature: sorted tuple of (unit_name, count) — used as dict key
def _comp_signature(comp: Dict[str, int]) -> str:
    """Create a stable string key from an enemy composition."""
    items = sorted((k, v) for k, v in comp.items() if v > 0)
    return "|".join(f"{k}:{v}" for k, v in items)


class TacticalBrain:
    """Observe → Hypothesize → Experiment → Learn.

    Pure learning: no hardcoded counter knowledge.
    The brain observes battles, tracks what unit compositions beat what,
    and generates hypotheses from accumulated evidence.

    Mid-game: suggests counter units based on active hypotheses.
    Cross-game: persists hypotheses and experiments to JSON.
    """

    def __init__(self):
        self._hypotheses: Dict[str, CounterHypothesis] = {}
        self._experiments: Dict[str, Experiment] = {}
        self._battle_log: List[Dict[str, Any]] = []
        self._state_history: List[BattleState] = []
        self._next_hyp_id = 1
        self._next_exp_id = 1

        # Config
        self._max_history = 200
        self._max_battle_log = 500
        self._min_army_for_hypothesis = 5
        self._experiment_target_count = 6  # build 6 of the hypothesized counter unit

    # ── State Capture ────────────────────────────────────────────

    def capture_state(self, bot: Any, tick: int) -> BattleState:
        """Capture full game state from BotAI — per-unit health, saturation, enemy."""
        state = BattleState(tick=tick)
        state.minerals = bot.minerals
        state.vespene = bot.vespene
        state.supply_used = bot.supply_used
        state.supply_cap = bot.supply_cap
        state.worker_count = len(bot.workers)
        state.base_count = len(bot.townhalls)

        # ── Own units: full health/status capture ──
        own_snapshots = []
        air_count = 0
        ground_count = 0
        anti_air = 0
        army_value = 0.0

        for u in bot.units:
            if u.is_structure:
                continue

            snap = UnitSnapshot(
                tag=u.tag,
                name=u.name,
                health=u.health,
                health_max=u.health_max,
                shield=getattr(u, 'shield', 0.0),
                shield_max=getattr(u, 'shield_max', 0.0),
                is_attacking=u.is_attacking,
                engaged_target_tag=getattr(u, 'engaged_target_tag', 0) or 0,
                position_x=u.position.x,
                position_y=u.position.y,
                is_flying=u.is_flying,
                can_attack_air=getattr(u, 'can_attack_air', False),
                can_attack_ground=getattr(u, 'can_attack_ground', False),
                attack_upgrade=getattr(u, 'attack_upgrade_level', 0),
                armor_upgrade=getattr(u, 'armor_upgrade_level', 0),
                movement_speed=getattr(u, 'movement_speed', 0.0),
            )
            own_snapshots.append(snap)

            # Count combat units (exclude workers, supply)
            if u.can_attack and not u.is_flying:
                ground_count += 1
                army_value += u.health + u.shield
                if getattr(u, 'can_attack_air', False):
                    anti_air += 1
            elif u.can_attack and u.is_flying:
                air_count += 1
                army_value += u.health + u.shield
                if getattr(u, 'can_attack_air', False):
                    anti_air += 1
            elif not u.can_attack and u.name.upper() not in ("OVERLORD", "OVERSEER", "OBSERVER"):
                # Workers still count for army_value if they can attack (some can)
                pass

            # Track units under attack
            if u.is_attacking:
                state.units_attacking.append(u.tag)
            # SC2 doesn't directly expose "is_under_attack" but we can infer
            # from engaged_target_tag (enemy targeting us)
            if hasattr(u, 'is_revealed') and u.is_revealed and u.is_enemy:
                pass  # enemy unit

        state.own_units = own_snapshots
        state.own_army_count = air_count + ground_count
        state.own_army_value = army_value
        state.own_air_count = air_count
        state.own_ground_count = ground_count
        state.own_anti_air_count = anti_air

        # ── Base saturation: per-base from SC2 API ──
        total_ideal = 0
        total_assigned = 0
        for th in bot.townhalls:
            ideal = getattr(th, 'ideal_harvesters', 16)
            assigned = getattr(th, 'assigned_harvesters', 0)
            surplus = getattr(th, 'surplus_harvesters', 0)
            has_vespene = False
            # Check if there's an extractor/g refinery near this base
            for s in bot.units.structure:
                if s.name.upper() in ("EXTRACTOR", "REFINERY", "ASSIMILATOR"):
                    if s.position.distance_to(th.position) < 10:
                        has_vespene = True
                        break

            sat = BaseSaturation(
                townhall_tag=th.tag,
                townhall_name=th.name,
                assigned_harvesters=assigned,
                ideal_harvesters=ideal,
                surplus_harvesters=surplus,
                has_vespene=has_vespene,
                position_x=th.position.x,
                position_y=th.position.y,
            )
            state.base_saturations.append(sat)
            total_ideal += ideal
            total_assigned += assigned

        state.total_ideal_harvesters = total_ideal
        state.total_assigned_harvesters = total_assigned

        # ── Enemy composition: from known_enemy_units ──
        enemy = EnemyComposition(tick=tick)
        enemy_air = 0
        enemy_ground = 0
        enemy_anti_air = 0
        enemy_value = 0.0
        upgrade_sum = 0
        upgrade_count = 0

        for eu in bot.known_enemy_units:
            if eu.is_structure:
                continue
            name = eu.name
            enemy.unit_counts[name] = enemy.unit_counts.get(name, 0) + 1
            enemy.unit_tags.add(eu.tag)
            enemy_value += eu.health + getattr(eu, 'shield', 0.0)

            if eu.is_flying:
                enemy_air += 1
            else:
                enemy_ground += 1

            if getattr(eu, 'can_attack_air', False):
                enemy_anti_air += 1

            atk = getattr(eu, 'attack_upgrade_level', 0)
            arm = getattr(eu, 'armor_upgrade_level', 0)
            upgrade_sum += atk + arm
            upgrade_count += 1

            # Check for detectors
            if getattr(eu, 'is_detector', False):
                enemy.has_detection = True

            # Check for casters (energy > 0)
            energy = getattr(eu, 'energy', 0)
            if energy and energy > 0:
                enemy.has_casters = True

        enemy.air_count = enemy_air
        enemy.ground_count = enemy_ground
        enemy.anti_air_count = enemy_anti_air
        enemy.total_value = enemy_value
        enemy.average_upgrade_level = upgrade_sum / max(1, upgrade_count)
        state.enemy = enemy

        # Store in history
        self._state_history.append(state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        return state

    # ── Battle Analysis ──────────────────────────────────────────

    def analyze_battles(self, bot: Any, tick: int) -> Optional[Dict[str, Any]]:
        """Detect recent battle outcomes by tracking unit deaths.

        Returns a battle outcome dict if a battle was detected, else None.
        """
        if len(self._state_history) < 2:
            return None

        prev = self._state_history[-2]
        curr = self._state_history[-1]

        # Detect our unit deaths (tags that were in prev but not in curr)
        prev_tags = {s.tag for s in prev.own_units if s.is_alive}
        curr_tags = {s.tag for s in curr.own_units if s.is_alive}
        dead_tags = prev_tags - curr_tags

        if not dead_tags:
            return None

        # Get the names of units we lost
        lost_units = {}
        for snap in prev.own_units:
            if snap.tag in dead_tags:
                lost_units[snap.name] = lost_units.get(snap.name, 0) + 1

        # Get current enemy composition
        enemy_comp = dict(curr.enemy.unit_counts)

        # Detect enemy unit deaths
        prev_enemy_tags = prev.enemy.unit_tags
        curr_enemy_tags = curr.enemy.unit_tags
        dead_enemy = prev_enemy_tags - curr_enemy_tags

        lost_enemy = {}
        if dead_enemy:
            # We need to know what enemy units died — check enemy snapshot
            for name, count in prev.enemy.unit_counts.items():
                # Approximate: if we killed some, reduce proportionally
                curr_count = curr.enemy.unit_counts.get(name, 0)
                killed = max(0, count - curr_count)
                if killed > 0:
                    lost_enemy[name] = killed

        # Determine battle outcome
        own_value_lost = sum(
            s.health_max + s.shield_max
            for s in prev.own_units if s.tag in dead_tags
        )
        enemy_value_lost = sum(
            curr.enemy.total_value for _ in range(sum(lost_enemy.values()))
        ) if lost_enemy else 0

        # Simple heuristic: we won if we killed more value than we lost
        won = enemy_value_lost > own_value_lost * 0.5 if own_value_lost > 0 else False
        outcome = "win" if won else "loss"

        battle = {
            "tick": tick,
            "our_composition": {s.name for s in prev.own_units if s.is_alive and s.can_attack},
            "our_units_lost": lost_units,
            "enemy_composition": enemy_comp,
            "enemy_units_lost": lost_enemy,
            "own_value_lost": own_value_lost,
            "enemy_value_lost": enemy_value_lost,
            "outcome": outcome,
        }

        self._battle_log.append(battle)
        if len(self._battle_log) > self._max_battle_log:
            self._battle_log.pop(0)

        return battle

    def record_battle_outcome(self, my_comp: Dict[str, int],
                               enemy_comp: Dict[str, int],
                               won: bool, tick: int):
        """Manually record a battle outcome (for cross-game learning)."""
        battle = {
            "tick": tick,
            "our_composition": my_comp,
            "enemy_composition": enemy_comp,
            "outcome": "win" if won else "loss",
            "cross_game": True,
        }
        self._battle_log.append(battle)
        if len(self._battle_log) > self._max_battle_log:
            self._battle_log.pop(0)

        # Update hypotheses from this outcome
        self._update_hypotheses_from_battle(my_comp, enemy_comp, won, tick)

    def _update_hypotheses_from_battle(self, my_comp: Dict[str, int],
                                        enemy_comp: Dict[str, int],
                                        won: bool, tick: int):
        """Update or create hypotheses based on battle outcome."""
        enemy_sig = _comp_signature(enemy_comp)

        # Find or create hypothesis for each unit type we used vs this enemy comp
        for unit_type, count in my_comp.items():
            if count <= 0:
                continue

            # Find existing hypothesis for this unit vs this enemy comp
            found = None
            for hyp in self._hypotheses.values():
                if (hyp.counter_unit == unit_type
                        and _comp_signature(hyp.targets_enemy) == enemy_sig):
                    found = hyp
                    break

            if found:
                found.update(won, tick)
            else:
                # Create new hypothesis
                hid = f"h{self._next_hyp_id}"
                self._next_hyp_id += 1
                hyp = CounterHypothesis(
                    id=hid,
                    counter_unit=unit_type,
                    targets_enemy=dict(enemy_comp),
                    confidence=0.6 if won else 0.4,
                    wins=1 if won else 0,
                    losses=0 if won else 1,
                    total_tests=1,
                    created_tick=tick,
                    last_tested_tick=tick,
                    last_outcome="win" if won else "loss",
                )
                hyp.update(won, tick)  # this will set status correctly
                self._hypotheses[hid] = hyp

    # ── Hypothesis Generation ────────────────────────────────────

    def generate_hypothesis(self, enemy_comp: Dict[str, int],
                             tick: int) -> Optional[CounterHypothesis]:
        """Generate a new counter-hypothesis for an enemy composition.

        Pure learning: uses battle log to find what worked vs similar compositions.
        No hardcoded knowledge.
        """
        if not enemy_comp:
            return None

        enemy_sig = _comp_signature(enemy_comp)

        # Check if we already have a hypothesis for this comp
        for hyp in self._hypotheses.values():
            if (_comp_signature(hyp.targets_enemy) == enemy_sig
                    and hyp.status != HypothesisStatus.REJECTED):
                return hyp  # already tracking this

        # Look through battle log for what worked vs similar enemy comps
        # "Similar" = shares at least 50% of unit types
        similar_battles = []
        enemy_types = set(enemy_comp.keys())

        for battle in self._battle_log:
            battle_enemy = battle.get("enemy_composition", {})
            battle_enemy_types = set(battle_enemy.keys())
            if not battle_enemy_types:
                continue
            overlap = len(enemy_types & battle_enemy_types) / max(len(enemy_types), len(battle_enemy_types))
            if overlap >= 0.5:
                similar_battles.append(battle)

        if not similar_battles:
            return None

        # Find what our most successful composition was vs similar enemies
        unit_wins: Dict[str, int] = {}
        unit_totals: Dict[str, int] = {}

        for battle in similar_battles:
            our_comp = battle.get("our_composition", {})
            won = battle.get("outcome") == "win"

            # our_comp might be a set (from analyze_battles) or dict (from manual records)
            if isinstance(our_comp, set):
                for unit_name in our_comp:
                    unit_totals[unit_name] = unit_totals.get(unit_name, 0) + 1
                    if won:
                        unit_wins[unit_name] = unit_wins.get(unit_name, 0) + 1
            elif isinstance(our_comp, dict):
                for unit_name, count in our_comp.items():
                    unit_totals[unit_name] = unit_totals.get(unit_name, 0) + count
                    if won:
                        unit_wins[unit_name] = unit_wins.get(unit_name, 0) + count

        if not unit_totals:
            return None

        # Pick the unit with best win rate (min 2 battles)
        best_unit = None
        best_winrate = 0.0
        for unit, wins in unit_wins.items():
            total = unit_totals[unit]
            if total >= 2:
                wr = wins / total
                if wr > best_winrate:
                    best_winrate = wr
                    best_unit = unit

        if not best_unit or best_winrate < 0.5:
            return None

        # Create hypothesis
        hid = f"h{self._next_hyp_id}"
        self._next_hyp_id += 1
        hyp = CounterHypothesis(
            id=hid,
            counter_unit=best_unit,
            targets_enemy=dict(enemy_comp),
            confidence=best_winrate,
            created_tick=tick,
            last_tested_tick=tick,
        )
        self._hypotheses[hid] = hyp
        return hyp

    def suggest_counter_unit(self, enemy_comp: Dict[str, int],
                              tick: int) -> Optional[str]:
        """Suggest what single unit type to build to counter enemy composition.

        Returns unit name or None.
        """
        if not enemy_comp:
            return None

        # First: check existing hypotheses
        best_hyp = None
        best_conf = 0.0
        enemy_sig = _comp_signature(enemy_comp)

        for hyp in self._hypotheses.values():
            if hyp.status == HypothesisStatus.REJECTED:
                continue
            if _comp_signature(hyp.targets_enemy) == enemy_sig:
                if hyp.confidence > best_conf:
                    best_conf = hyp.confidence
                    best_hyp = hyp

        if best_hyp and best_conf >= 0.5:
            return best_hyp.counter_unit

        # No existing hypothesis: generate one from battle history
        new_hyp = self.generate_hypothesis(enemy_comp, tick)
        if new_hyp and new_hyp.confidence >= 0.5:
            return new_hyp.counter_unit

        return None

    def suggest_unit_composition(self, enemy_comp: Dict[str, int],
                                  tick: int,
                                  current_army: Dict[str, int] = None) -> Optional[Dict[str, int]]:
        """Suggest a full unit composition to counter enemy.

        Returns {unit_type: count_to_build} or None.
        """
        counter = self.suggest_counter_unit(enemy_comp, tick)
        if not counter:
            return None

        # How many do we already have?
        current_count = (current_army or {}).get(counter, 0)
        target = self._experiment_target_count
        needed = max(0, target - current_count)

        if needed == 0:
            return None

        return {counter: needed}

    # ── Experiment Management ────────────────────────────────────

    def start_experiment(self, unit_type: str, enemy_comp: Dict[str, int],
                          hypothesis_id: str, tick: int) -> Experiment:
        """Start a new experiment: build N of unit_type vs enemy comp."""
        eid = f"e{self._next_exp_id}"
        self._next_exp_id += 1
        exp = Experiment(
            id=eid,
            hypothesis_id=hypothesis_id,
            unit_type=unit_type,
            target_count=self._experiment_target_count,
            enemy_snapshot=dict(enemy_comp),
            phase=ExperimentPhase.BUILDING,
            created_tick=tick,
            started_tick=tick,
        )
        self._experiments[eid] = exp
        return exp

    def get_active_experiment(self) -> Optional[Experiment]:
        """Get the currently active experiment, if any."""
        for exp in self._experiments.values():
            if exp.phase in (ExperimentPhase.QUEUED, ExperimentPhase.BUILDING,
                             ExperimentPhase.OBSERVING):
                return exp
        return None

    def update_experiment_progress(self, built_count: int, tick: int):
        """Update how many counter units have been built."""
        exp = self.get_active_experiment()
        if not exp:
            return
        exp.built_count = built_count
        if exp.built_count >= exp.target_count and exp.phase == ExperimentPhase.BUILDING:
            exp.phase = ExperimentPhase.OBSERVING

    def complete_experiment(self, won: bool, tick: int, notes: str = ""):
        """Record the outcome of an active experiment."""
        exp = self.get_active_experiment()
        if not exp:
            return
        exp.phase = ExperimentPhase.COMPLETE
        exp.completed_tick = tick
        exp.outcome = "win" if won else "loss"
        exp.result_notes = notes

        # Update the linked hypothesis
        if exp.hypothesis_id in self._hypotheses:
            self._hypotheses[exp.hypothesis_id].update(won, tick)

    # ── Per-Base Saturation Helpers ──────────────────────────────

    def get_saturation_status(self, state: BattleState) -> Dict[str, Any]:
        """Get per-base saturation info for drive updates."""
        bases = []
        total_ideal = 0
        total_assigned = 0

        for sat in state.base_saturations:
            bases.append({
                "tag": sat.townhall_tag,
                "name": sat.townhall_name,
                "assigned": sat.assigned_harvesters,
                "ideal": sat.ideal_harvesters,
                "surplus": sat.surplus_harvesters,
                "ratio": round(sat.saturation_ratio, 2),
                "undersaturated": sat.is_undersaturated,
                "oversaturated": sat.is_oversaturated,
                "has_vespene": sat.has_vespene,
            })
            total_ideal += sat.ideal_harvesters
            total_assigned += sat.assigned_harvesters

        return {
            "bases": bases,
            "total_ideal": total_ideal,
            "total_assigned": total_assigned,
            "overall_ratio": round(total_assigned / max(1, total_ideal), 2),
            "workers_needed": max(0, total_ideal - total_assigned),
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str = None):
        """Persist tactical brain state to JSON."""
        path = path or TACTICAL_BRAIN_PATH
        data = {
            "hypotheses": {
                hid: h.to_dict() for hid, h in self._hypotheses.items()
            },
            "experiments": {
                eid: e.to_dict() for eid, e in self._experiments.items()
            },
            "battle_log": self._battle_log[-100:],  # keep last 100
            "next_hyp_id": self._next_hyp_id,
            "next_exp_id": self._next_exp_id,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = None) -> bool:
        """Load tactical brain state from JSON."""
        path = path or TACTICAL_BRAIN_PATH
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._hypotheses = {
                hid: CounterHypothesis.from_dict(hd)
                for hid, hd in data.get("hypotheses", {}).items()
            }
            self._experiments = {
                eid: Experiment.from_dict(ed)
                for eid, ed in data.get("experiments", {}).items()
            }
            self._battle_log = data.get("battle_log", [])
            self._next_hyp_id = data.get("next_hyp_id", len(self._hypotheses) + 1)
            self._next_exp_id = data.get("next_exp_id", len(self._experiments) + 1)
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    # ── Render / Debug ───────────────────────────────────────────

    def render(self, state: Optional[BattleState] = None) -> str:
        """Render tactical brain state for debugging."""
        lines = ["=" * 60, "TACTICAL BRAIN", "=" * 60]

        if state:
            lines.append(f"\n  Army: {state.own_army_count} units "
                         f"({state.own_ground_count} ground, {state.own_air_count} air, "
                         f"{state.own_anti_air_count} anti-air)")
            lines.append(f"  Enemy: {state.enemy.air_count} air, {state.enemy.ground_count} ground")
            if state.enemy.unit_counts:
                lines.append(f"  Enemy comp: {dict(state.enemy.unit_counts)}")

            sat = self.get_saturation_status(state)
            lines.append(f"\n  Worker saturation: {sat['total_assigned']}/{sat['total_ideal']} "
                         f"({sat['overall_ratio']:.0%})")
            for b in sat["bases"]:
                marker = "!" if b["undersaturated"] else ("+" if b["oversaturated"] else " ")
                lines.append(f"    {marker} {b['name']}: {b['assigned']}/{b['ideal']} "
                             f"{'(vespene)' if b['has_vespene'] else ''}")

        # Hypotheses
        active = [h for h in self._hypotheses.values()
                  if h.status == HypothesisStatus.ACTIVE]
        validated = [h for h in self._hypotheses.values()
                     if h.status == HypothesisStatus.VALIDATED]
        rejected = [h for h in self._hypotheses.values()
                    if h.status == HypothesisStatus.REJECTED]

        lines.append(f"\n  Hypotheses: {len(active)} active, "
                     f"{len(validated)} validated, {len(rejected)} rejected")
        for h in sorted(active, key=lambda x: x.confidence, reverse=True)[:5]:
            lines.append(f"    {h.counter_unit} vs {h.targets_enemy}: "
                         f"conf={h.confidence:.2f} "
                         f"W/L={h.wins}/{h.losses} "
                         f"({h.win_rate:.0%})")

        # Active experiment
        exp = self.get_active_experiment()
        if exp:
            lines.append(f"\n  Active experiment: {exp.unit_type} "
                         f"({exp.built_count}/{exp.target_count} built) "
                         f"phase={exp.phase.value}")

        # Battle log
        lines.append(f"\n  Battle log: {len(self._battle_log)} records")

        lines.append("=" * 60)
        return "\n".join(lines)
