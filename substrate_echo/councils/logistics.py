"""Logistics Council - Supply management and worker transfer."""

from __future__ import annotations
from typing import Dict, List, Any, Optional

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class LogisticsCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("logistics", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._supply_buffer = 4
        self._last_supply_check = 0

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.SUPPLY_CHANGED, self._on_supply_change)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.LOGISTICS]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        supply_used = game_state.get("supply_used", 0)
        supply_cap = game_state.get("supply_cap", 0)
        minerals = game_state.get("minerals", 0)
        bases = game_state.get("base_count", 1)

        supply_remaining = supply_cap - supply_used
        if supply_remaining < self._supply_buffer and minerals >= 100:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="build_supply",
                content={"utility": 0.9, "supply_remaining": supply_remaining},
                priority=0.9,
                confidence=0.95,
            ))

        self.write_belief(BlackboardChannel.LOGISTICS, "supply_status",
                          {"used": supply_used, "cap": supply_cap, "remaining": supply_remaining},
                          ttl=5.0)
        return proposals

    def _on_supply_change(self, event) -> None:
        pass
