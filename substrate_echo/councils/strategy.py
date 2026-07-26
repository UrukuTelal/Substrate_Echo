"""Strategy Council - High-level strategic planning."""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from enum import Enum

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class StrategicIntent(Enum):
    EXPAND = "expand"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    TECH = "tech"
    HARASS = "harass"
    CONTAIN = "contain"
    RECOVER = "recover"
    TRADE = "trade"
    END_GAME = "end_game"


class StrategyCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("strategy", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._current_intent = StrategicIntent.EXPAND
        self._intent_history: List[Dict] = []

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.STRATEGY_CHANGED, self._on_strategy_changed)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.STRATEGY]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        new_intent = self._evaluate_intent(game_state, tick)
        if new_intent != self._current_intent:
            old = self._current_intent
            self._current_intent = new_intent
            self._intent_history.append({
                "from": old.value, "to": new_intent.value, "tick": tick,
            })
            self.event_bus.publish_simple(
                EventType.STRATEGY_CHANGED, self.name,
                {"old": old.value, "new": new_intent.value}, tick
            )

        proposals.append(CouncilProposal(
            council_name=self.name,
            proposal_type="strategic_intent",
            content={"intent": self._current_intent.value, "utility": 0.5},
            priority=0.3,
            confidence=0.6,
        ))

        self.write_belief(BlackboardChannel.STRATEGY, "current_intent",
                          self._current_intent.value, ttl=30.0)
        return proposals

    def _evaluate_intent(self, game_state: Dict[str, Any], tick: int) -> StrategicIntent:
        army = game_state.get("army_count", 0)
        enemy_army = game_state.get("enemy_army_count", 0)
        bases = game_state.get("base_count", 1)
        minerals = game_state.get("minerals", 0)
        game_time = game_state.get("game_time", 0)

        if army < 5 and enemy_army > 20:
            return StrategicIntent.RECOVER

        if army > enemy_army * 1.5 and army > 30:
            return StrategicIntent.AGGRESSIVE

        if bases < 3 and minerals > 400:
            return StrategicIntent.EXPAND

        if game_time > 600 and army > 50:
            return StrategicIntent.END_GAME

        if army < 15:
            return StrategicIntent.DEFENSIVE

        return StrategicIntent.EXPAND

    def _on_strategy_changed(self, event) -> None:
        pass
