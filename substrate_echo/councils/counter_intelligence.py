"""Counter-Intelligence Council - Deception detection and hidden threats."""

from __future__ import annotations
from typing import Dict, List, Any, Optional

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType
from substrate_echo.blackboard.evidence import OpponentBelief


class CounterIntelligenceCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("counter_intelligence", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._deception_signals: List[Dict] = []
        self._hidden_threats: List[Dict] = []

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.ENEMY_UNIT_SCOUTED, self._on_intel)
        self.subscribe(EventType.ENEMY_STRUCTURE_SCOUTED, self._on_intel)
        self.subscribe(EventType.ENEMY_TECH_DETECTED, self._on_tech)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.COUNTER_INTEL]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        enemy_structures = game_state.get("enemy_structures", [])
        enemy_units = game_state.get("enemy_units_visible", [])
        game_time = game_state.get("game_time", 0)

        if game_time > 300 and len(enemy_structures) < 3:
            self._detect_hidden_tech(game_state, tick)

        if self._hidden_threats:
            proposals.append(CouncilProposal(
                council_name=self.name,
                proposal_type="counter_threat",
                content={"threats": self._hidden_threats, "utility": 0.7},
                priority=0.6,
                confidence=0.5,
            ))

        self._publish_beliefs(tick)
        return proposals

    def _detect_hidden_tech(self, game_state: Dict[str, Any], tick: int) -> None:
        enemy_bases = game_state.get("enemy_base_count", 1)
        visible_structures = len(game_state.get("enemy_structures", []))
        if enemy_bases > visible_structures:
            threat = {
                "type": "hidden_base",
                "confidence": 0.4,
                "tick": tick,
                "reason": f"Expected ~{enemy_bases * 8} structures, only {visible_structures} visible",
            }
            self._hidden_threats.append(threat)

    def _publish_beliefs(self, tick: int) -> None:
        if self._hidden_threats:
            belief = OpponentBelief(
                claim="Opponent has hidden infrastructure",
                confidence=0.5,
                tick=tick,
                opponent_id=1,
                belief_type="hidden_threat",
            )
            self.blackboard.add_opponent_belief(1, belief)

    def _on_intel(self, event) -> None:
        self.write_belief(BlackboardChannel.COUNTER_INTEL,
                          f"intel_{event.event_type.value}", event.data, confidence=0.7, ttl=60.0)

    def _on_tech(self, event) -> None:
        self.write_belief(BlackboardChannel.COUNTER_INTEL,
                          "enemy_tech", event.data, confidence=0.8, ttl=120.0)
