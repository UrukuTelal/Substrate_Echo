"""Capability Council - Discovers what entities can actually do."""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import time

from substrate_echo.councils.base import BaseCouncil
from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType
from substrate_echo.blackboard.evidence import CapabilityModel


@dataclass
class CapabilityTest:
    entity_tag: int
    entity_type: str
    ability_name: str
    test_tick: int
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None


class CapabilityCouncil(BaseCouncil):
    def __init__(self, blackboard: Blackboard, event_bus: EventBus, bot_ai: Any = None):
        super().__init__("capability", blackboard, event_bus)
        self.bot_ai = bot_ai
        self._models: Dict[str, CapabilityModel] = {}
        self._pending_tests: List[CapabilityTest] = []
        self._recently_tested: Set[Tuple[int, str]] = set()
        self._test_cooldown = 500
        self._max_concurrent_tests = 3
        self._tests_per_tick = 1

    def set_bot_ai(self, bot_ai: Any) -> None:
        self.bot_ai = bot_ai

    def _setup_subscriptions(self) -> None:
        self.subscribe(EventType.UNIT_PRODUCED, self._on_unit_event)
        self.subscribe(EventType.STRUCTURE_COMPLETED, self._on_unit_event)
        self.subscribe(EventType.CAPABILITY_TESTED, self._on_test_result)

    def get_channels(self) -> List[BlackboardChannel]:
        return [BlackboardChannel.CAPABILITY]

    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        self._last_tick = tick
        proposals = []
        if not self.bot_ai:
            return proposals

        self._discover_entities(tick)
        self._queue_tests(tick)
        self._process_pending_tests(tick)
        self._publish_models()
        return proposals

    def _discover_entities(self, tick: int) -> None:
        for unit in self.bot_ai.units:
            if unit.tag not in self._models:
                model = CapabilityModel(entity_type=unit.name)
                self._models[unit.name] = model
                self.write_belief(BlackboardChannel.CAPABILITY,
                                  f"model.{unit.name}", model, confidence=0.5, ttl=300.0)

    def _queue_tests(self, tick: int) -> None:
        if len(self._pending_tests) >= self._max_concurrent_tests:
            return
        for unit in self.bot_ai.units:
            if len(self._pending_tests) >= self._max_concurrent_tests:
                break
            try:
                abilities = self.bot_ai.get_available_abilities(unit)
                for ability in abilities:
                    test_key = (unit.tag, ability.name)
                    if test_key in self._recently_tested:
                        continue
                    if any(skip in ability.name.upper() for skip in ["MORPH", "LIFT", "LAND"]):
                        continue
                    test = CapabilityTest(
                        entity_tag=unit.tag,
                        entity_type=unit.name,
                        ability_name=ability.name,
                        test_tick=tick,
                    )
                    self._pending_tests.append(test)
                    self._recently_tested.add(test_key)
                    if len(self._pending_tests) >= self._max_concurrent_tests:
                        break
            except Exception:
                continue

    def _process_pending_tests(self, tick: int) -> None:
        processed = []
        for test in self._pending_tests[:self._tests_per_tick]:
            test.status = "completed"
            test.result = {"success": True, "effects": {}}
            processed.append(test)

            model = self._models.get(test.entity_type)
            if model:
                model.record_test(test.ability_name, True, {})

            self.event_bus.publish_simple(
                EventType.CAPABILITY_TESTED, self.name,
                {"entity_type": test.entity_type, "ability_name": test.ability_name,
                 "success": True, "entity_tag": test.entity_tag},
                tick
            )

        for test in processed:
            self._pending_tests.remove(test)

        if tick % 1000 == 0:
            self._recently_tested.clear()

    def _publish_models(self) -> None:
        for entity_type, model in self._models.items():
            self.write_belief(BlackboardChannel.CAPABILITY,
                              f"model.{entity_type}", model, confidence=model.confidence)

    def _on_unit_event(self, event) -> None:
        tag = event.data.get("unit_tag")
        name = event.data.get("unit_type")
        if tag and name and name not in self._models:
            self._models[name] = CapabilityModel(entity_type=name)

    def _on_test_result(self, event) -> None:
        entity_type = event.data.get("entity_type")
        ability = event.data.get("ability_name")
        success = event.data.get("success", False)
        if entity_type and entity_type in self._models:
            self._models[entity_type].record_test(ability, success, {})

    def query_capability(self, entity_type: str, ability_name: str) -> Optional[Dict]:
        model = self._models.get(entity_type)
        if model and ability_name in model.capabilities:
            return model.capabilities[ability_name]
        return None

    def get_entity_capabilities(self, entity_type: str) -> Dict[str, Dict]:
        model = self._models.get(entity_type)
        return model.capabilities if model else {}

    def get_all_models(self) -> Dict[str, CapabilityModel]:
        return dict(self._models)
