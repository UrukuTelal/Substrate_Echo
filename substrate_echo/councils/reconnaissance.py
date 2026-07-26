"""Reconnaissance Council - Scouting and map awareness."""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Set

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class ReconnaissanceCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("reconnaissance", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._scouted_positions: Set[tuple] = set()
        self._last_scout_tick = 0
        self._scout_interval = 500

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.ENEMY_UNIT_SCOUTED, self._on_enemy_scouted)
        self.subscribe(EventType.ENEMY_STRUCTURE_SCOUTED, self._on_structure_scouted)
        self.subscribe(EventType.ENEMY_EXPANSION_DETECTED, self._on_expansion_detected)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.INTELLIGENCE]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        enemy_army = game_state.get("enemy_army_count", 0)
        scout_count = game_state.get("scout_count", 0)
        game_time = game_state.get("game_time", 0)

        if tick - self._last_scout_tick > self._scout_interval and scout_count == 0:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="send_scout",
                content={"utility": 0.6, "reason": "no_active_scouts"},
                priority=0.4,
                confidence=0.8,
            ))
            self._last_scout_tick = tick

        if enemy_army > 0 and game_time > 180:
            self.write_belief(BlackboardChannel.INTELLIGENCE,
                              "enemy_active", True, confidence=0.9, ttl=30.0)

        self.write_belief(BlackboardChannel.INTELLIGENCE,
                          "scouted_count", len(self._scouted_positions), ttl=30.0)
        return proposals

    def _on_enemy_scouted(self, event) -> None:
        pos = event.data.get("position")
        if pos:
            self._scouted_positions.add(tuple(pos) if isinstance(pos, list) else pos)
        self.write_belief(BlackboardChannel.INTELLIGENCE,
                          "last_enemy_sighting", event.data, confidence=0.9, ttl=60.0)

    def _on_structure_scouted(self, event) -> None:
        self.write_belief(BlackboardChannel.INTELLIGENCE,
                          "enemy_structure", event.data, confidence=0.8, ttl=120.0)

    def _on_expansion_detected(self, event) -> None:
        self.write_belief(BlackboardChannel.INTELLIGENCE,
                          "enemy_expansion", event.data, confidence=0.7, ttl=180.0)
