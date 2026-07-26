"""Technology Council - Upgrade research and tech tree progression."""

from __future__ import annotations
from typing import Dict, List, Any, Optional

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class TechnologyCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("technology", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._researched_upgrades: Dict[str, bool] = {}
        self._available_upgrades: List[str] = []

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.UPGRADE_STARTED, self._on_upgrade_started)
        self.subscribe(EventType.UPGRADE_COMPLETED, self._on_upgrade_completed)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.TECHNOLOGY]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        minerals = game_state.get("minerals", 0)
        gas = game_state.get("vespene", 0)
        has_pool = game_state.get("has_spawning_pool", False)
        has_evo = game_state.get("has_evolution_chamber", False)

        if has_pool and not self._researched_upgrades.get("zergling_speed", False):
            if minerals >= 100 and gas >= 100:
                proposals.append(CouncilProposal(
                    council_name=self.name,
                    proposal_type="research_upgrade",
                    content={"upgrade": "zergling_speed", "utility": 0.7},
                    priority=0.5,
                    confidence=0.9,
                ))

        if has_evo and not self._researched_upgrades.get("melee_attack", False):
            if minerals >= 100 and gas >= 100:
                proposals.append(CouncilProposal(
                    council_name=self.name,
                    proposal_type="research_upgrade",
                    content={"upgrade": "melee_attack", "utility": 0.6},
                    priority=0.4,
                    confidence=0.8,
                ))

        self.write_belief(BlackboardChannel.TECHNOLOGY,
                          "researched", self._researched_upgrades, ttl=60.0)
        return proposals

    def _on_upgrade_started(self, event) -> None:
        upgrade = event.data.get("upgrade_name")
        if upgrade:
            self._available_upgrades.append(upgrade)

    def _on_upgrade_completed(self, event) -> None:
        upgrade = event.data.get("upgrade_name")
        if upgrade:
            self._researched_upgrades[upgrade] = True
