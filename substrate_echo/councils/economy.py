"""Economy Council - Manages workers, bases, and resource income."""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class EconomyCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("economy", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._workers_per_mineral = 2
        self._workers_per_gas = 3
        self._base_worker_targets: Dict[int, int] = {}

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.MINERALS_CHANGED, self._on_resource_change)
        self.subscribe(EventType.WORKER_COUNT_CHANGED, self._on_worker_change)
        self.subscribe(EventType.BASE_COUNT_CHANGED, self._on_base_change)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.ECONOMY]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        minerals = game_state.get("minerals", 0)
        gas = game_state.get("vespene", 0)
        workers = game_state.get("worker_count", 0)
        bases = game_state.get("base_count", 1)
        supply_used = game_state.get("supply_used", 0)
        supply_cap = game_state.get("supply_cap", 0)

        workers_needed = self._calculate_worker_needs(bases)

        if workers < workers_needed and minerals >= 50:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="produce_worker",
                content={"unit_type": "worker", "count": 1, "utility": 0.8},
                priority=0.7,
                confidence=0.9,
            ))

        if self._should_expand(bases, workers, minerals):
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="expand",
                content={"utility": 0.6, "bases": bases + 1},
                priority=0.5,
                confidence=0.7,
            ))

        gas_geysers = game_state.get("gas_geysers", 0)
        extractors = game_state.get("extractors", 0)
        if extractors < gas_geysers and minerals >= 75 and workers >= 14:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="build_extractor",
                content={"utility": 0.5},
                priority=0.4,
                confidence=0.8,
            ))

        self.write_belief(BlackboardChannel.ECONOMY, "worker_target", workers_needed, ttl=10.0)
        self.write_belief(BlackboardChannel.ECONOMY, "income_rate", self._estimate_income(game_state), ttl=10.0)
        return proposals

    def _calculate_worker_needs(self, bases: int) -> int:
        return bases * 22

    def _should_expand(self, bases: int, workers: int, minerals: float) -> bool:
        if minerals < 400:
            return False
        if workers >= bases * 20 and bases < 5:
            return True
        return False

    def _estimate_income(self, game_state: Dict[str, Any]) -> float:
        workers = game_state.get("worker_count", 0)
        bases = game_state.get("base_count", 1)
        mining_efficiency = min(1.0, workers / (bases * 16))
        return mining_efficiency * 400 * bases

    def _on_resource_change(self, event) -> None:
        pass

    def _on_worker_change(self, event) -> None:
        pass

    def _on_base_change(self, event) -> None:
        pass
