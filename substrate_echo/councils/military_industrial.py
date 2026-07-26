"""Military Industrial Council - Unit composition and production planning."""

from __future__ import annotations
from typing import Dict, List, Any, Optional

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class MilitaryIndustrialCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("military_industrial", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._composition_targets: Dict[str, float] = {}
        self._production_queue: List[Dict] = []

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.UNIT_PRODUCED, self._on_unit_produced)
        self.subscribe(EventType.UNIT_LOST, self._on_unit_lost)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.MILITARY_INDUSTRIAL]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        army_count = game_state.get("army_count", 0)
        minerals = game_state.get("minerals", 0)
        gas = game_state.get("vespene", 0)
        supply_used = game_state.get("supply_used", 0)
        supply_cap = game_state.get("supply_cap", 0)
        enemy_army = game_state.get("enemy_army_count", 0)

        if army_count < 10 and minerals >= 100:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="build_army",
                content={"unit_type": "any", "utility": 0.8},
                priority=0.6,
                confidence=0.8,
            ))

        if enemy_army > army_count * 1.5 and minerals >= 150:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="build_army",
                content={"unit_type": "any", "utility": 0.9, "reason": "outnumbered"},
                priority=0.8,
                confidence=0.7,
            ))

        self.write_belief(BlackboardChannel.MILITARY_INDUSTRIAL,
                          "army_count", army_count, ttl=10.0)
        self.write_belief(BlackboardChannel.MILITARY_INDUSTRIAL,
                          "composition", self._composition_targets, ttl=30.0)
        return proposals

    def _on_unit_produced(self, event) -> None:
        unit_type = event.data.get("unit_type")
        if unit_type:
            self._composition_targets[unit_type] = self._composition_targets.get(unit_type, 0) + 1

    def _on_unit_lost(self, event) -> None:
        unit_type = event.data.get("unit_type")
        if unit_type and unit_type in self._composition_targets:
            self._composition_targets[unit_type] = max(0, self._composition_targets[unit_type] - 1)
