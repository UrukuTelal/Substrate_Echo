"""SC2 Action Decoder — Translates abstract intent to game actions.

The kernel produces abstract intents like:
  - "Secure expansion location"
  - "Build economic infrastructure"
  - "Defend current position"

The decoder translates these into executable SC2 actions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class ActionType(Enum):
    """Types of SC2 actions."""
    EXPAND = "expand"
    BUILD_ARMY = "build_army"
    DEFEND = "defend"
    ATTACK = "attack"
    SCOUT = "scout"
    HARASS = "harass"
    TECH_UP = "tech_up"
    EXPLORE = "explore"
    RETREAT = "retreat"
    HOLD = "hold"


@dataclass
class AbstractAction:
    """Abstract intent from the kernel."""
    action_type: ActionType
    target: Optional[Tuple[float, float]] = None  # Map position
    description: str = ""
    confidence: float = 0.5
    urgency: float = 0.5
    resource_cost: float = 0.3


@dataclass
class ConcreteAction:
    """Executable SC2 action."""
    action_type: ActionType
    unit_tags: List[int] = field(default_factory=list)
    target_position: Optional[Tuple[float, float]] = None
    target_unit_tag: Optional[int] = None
    ability_id: Optional[int] = None
    queue_command: bool = False
    description: str = ""


class SC2ActionDecoder:
    """Decodes abstract intent into concrete SC2 actions.

    Maps kernel abstract actions to game mechanics.
    """

    # Ability IDs (SC2 API)
    ABILITY_BUILD_CC = 86
    ABILITY_BUILD_BARRACKS = 21
    ABILITY_BUILD_SUPPLY = 1008
    ABILITY_ATTACK = 12
    ABILITY_MOVE = 16
    ABILITY_STOP = 4
    ABILITY_PATROL = 11

    def __init__(self):
        self._action_history: List[AbstractAction] = []
        self._unit_cache: Dict[int, Any] = {}

    def decode(self, abstract: AbstractAction,
               game_state: Any = None) -> List[ConcreteAction]:
        """Decode abstract intent into concrete actions."""
        self._action_history.append(abstract)
        if len(self._action_history) > 50:
            self._action_history.pop(0)

        if game_state is not None:
            self._update_unit_cache(game_state)

        decoder_map = {
            ActionType.EXPAND: self._decode_expand,
            ActionType.BUILD_ARMY: self._decode_build_army,
            ActionType.DEFEND: self._decode_defend,
            ActionType.ATTACK: self._decode_attack,
            ActionType.SCOUT: self._decode_scout,
            ActionType.HARASS: self._decode_harass,
            ActionType.TECH_UP: self._decode_tech_up,
            ActionType.EXPLORE: self._decode_explore,
            ActionType.RETREAT: self._decode_retreat,
            ActionType.HOLD: self._decode_hold,
        }

        decoder = decoder_map.get(abstract.action_type, self._decode_hold)
        return decoder(abstract)

    def _decode_expand(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Build a new command center."""
        workers = [tag for tag, unit in self._unit_cache.items()
                   if unit.unit_type == 45 and unit.is_idle][:4]

        actions = []
        if abstract.target:
            actions.append(ConcreteAction(
                action_type=ActionType.EXPAND,
                unit_tag=workers[0] if workers else None,
                target_position=abstract.target,
                ability_id=self.ABILITY_BUILD_CC,
                description=f"Expand to {abstract.target}",
            ))
        return actions

    def _decode_build_army(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Build military units."""
        actions = []
        for tag, unit in self._unit_cache.items():
            if unit.unit_type in [21, 22, 23] and unit.is_idle:  # Production buildings
                actions.append(ConcreteAction(
                    action_type=ActionType.BUILD_ARMY,
                    unit_tag=tag,
                    ability_id=self.ABILITY_BUILD_BARRACKS,
                    description="Queue military unit",
                ))
                if len(actions) >= 3:
                    break
        return actions

    def _decode_defend(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Move army to defensive position."""
        army = [tag for tag, unit in self._unit_cache.items()
                if unit.unit_type not in [45, 86, 130, 131]][:10]

        if army and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.DEFEND,
                unit_tags=army,
                target_position=abstract.target,
                ability_id=self.ABILITY_MOVE,
                description=f"Defend position {abstract.target}",
            )]
        return []

    def _decode_attack(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Move army to attack position."""
        army = [tag for tag, unit in self._unit_cache.items()
                if unit.unit_type not in [45, 86, 130, 131]][:15]

        if army and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.ATTACK,
                unit_tags=army,
                target_position=abstract.target,
                ability_id=self.ABILITY_ATTACK,
                description=f"Attack position {abstract.target}",
            )]
        return []

    def _decode_scout(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Send a unit to scout."""
        workers = [tag for tag, unit in self._unit_cache.items()
                   if unit.unit_type == 45 and unit.is_idle][:1]

        if workers and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.SCOUT,
                unit_tag=workers[0],
                target_position=abstract.target,
                ability_id=self.ABILITY_MOVE,
                description=f"Scout position {abstract.target}",
            )]
        return []

    def _decode_harass(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Send a small group to harass."""
        army = [tag for tag, unit in self._unit_cache.items()
                if unit.unit_type not in [45, 86, 130, 131]][:5]

        if army and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.HARASS,
                unit_tags=army,
                target_position=abstract.target,
                ability_id=self.ABILITY_ATTACK,
                description=f"Harass position {abstract.target}",
            )]
        return []

    def _decode_tech_up(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Research upgrades."""
        actions = []
        for tag, unit in self._unit_cache.items():
            if unit.unit_type in [21, 22, 23, 39, 62]:  # Production/upgrade buildings
                actions.append(ConcreteAction(
                    action_type=ActionType.TECH_UP,
                    unit_tag=tag,
                    description="Research upgrade",
                ))
                if len(actions) >= 2:
                    break
        return actions

    def _decode_explore(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Explore unknown areas."""
        workers = [tag for tag, unit in self._unit_cache.items()
                   if unit.unit_type == 45 and unit.is_idle][:1]

        if workers and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.EXPLORE,
                unit_tag=workers[0],
                target_position=abstract.target,
                ability_id=self.ABILITY_MOVE,
                description=f"Explore position {abstract.target}",
            )]
        return []

    def _decode_retreat(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Retreat army to safety."""
        army = [tag for tag, unit in self._unit_cache.items()
                if unit.unit_type not in [45, 86, 130, 131]][:10]

        if army and abstract.target:
            return [ConcreteAction(
                action_type=ActionType.RETREAT,
                unit_tags=army,
                target_position=abstract.target,
                ability_id=self.ABILITY_MOVE,
                description=f"Retreat to {abstract.target}",
            )]
        return []

    def _decode_hold(self, abstract: AbstractAction) -> List[ConcreteAction]:
        """Hold current position."""
        army = [tag for tag, unit in self._unit_cache.items()
                if unit.unit_type not in [45, 86, 130, 131]][:10]

        if army:
            return [ConcreteAction(
                action_type=ActionType.HOLD,
                unit_tags=army,
                ability_id=self.ABILITY_STOP,
                description="Hold position",
            )]
        return []

    def _update_unit_cache(self, game_state: Any):
        """Update unit cache from game state."""
        try:
            self._unit_cache = {u.tag: u for u in game_state.observation.units}
        except (AttributeError, TypeError):
            pass
