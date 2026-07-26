"""Base council class and council registry."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import time

from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.events import EventBus, EventType


class CouncilState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"


@dataclass
class CouncilMetrics:
    proposals_made: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    avg_proposal_utility: float = 0.0
    avg_proposal_confidence: float = 0.0
    last_update_tick: int = 0
    total_compute_time: float = 0.0

    def acceptance_rate(self) -> float:
        total = self.proposals_accepted + self.proposals_rejected
        return self.proposals_accepted / total if total > 0 else 0.0


class BaseCouncil(ABC):
    def __init__(self, name: str, blackboard: Blackboard, event_bus: EventBus):
        self.name = name
        self.blackboard = blackboard
        self.event_bus = event_bus
        self.state = CouncilState.INITIALIZING
        self.metrics = CouncilMetrics()
        self._last_tick = 0
        self._tick_interval = 1
        self._setup_subscriptions()
        self.state = CouncilState.ACTIVE

    @abstractmethod
    def _setup_subscriptions(self) -> None:
        pass

    @abstractmethod
    def update(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        pass

    @abstractmethod
    def get_channels(self) -> List[BlackboardChannel]:
        pass

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        self.event_bus.subscribe(event_type, callback)

    def should_update(self, tick: int) -> bool:
        return (tick - self._last_tick) >= self._tick_interval

    def write_belief(self, channel: BlackboardChannel, key: str, value: Any,
                     confidence: float = 1.0, ttl: Optional[float] = None) -> None:
        self.blackboard.write(channel, key, value, source=self.name, confidence=confidence, ttl=ttl)

    def read_belief(self, channel: BlackboardChannel, key: str, default: Any = None) -> Any:
        return self.blackboard.read(channel, key, default)

    def submit_proposal(self, proposal: CouncilProposal) -> None:
        self.blackboard.submit_proposal(proposal)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "proposals_made": self.metrics.proposals_made,
            "proposals_accepted": self.metrics.proposals_accepted,
            "proposals_rejected": self.metrics.proposals_rejected,
            "acceptance_rate": self.metrics.acceptance_rate(),
            "avg_utility": self.metrics.avg_proposal_utility,
            "avg_confidence": self.metrics.avg_proposal_confidence,
        }

    def suspend(self) -> None:
        self.state = CouncilState.SUSPENDED

    def resume(self) -> None:
        self.state = CouncilState.ACTIVE


class CouncilRegistry:
    def __init__(self, blackboard: Blackboard, event_bus: EventBus):
        self.blackboard = blackboard
        self.event_bus = event_bus
        self._councils: Dict[str, BaseCouncil] = {}
        self._update_order: List[str] = []

    def register(self, council: BaseCouncil) -> None:
        self._councils[council.name] = council
        self._rebuild_order()

    def _rebuild_order(self) -> None:
        priority = ["capability", "economy", "logistics", "reconnaissance",
                     "counter_intelligence", "military_industrial", "technology", "strategy"]
        ordered = [n for n in priority if n in self._councils]
        ordered += [n for n in self._councils if n not in ordered]
        self._update_order = ordered

    def get_council(self, name: str) -> Optional[BaseCouncil]:
        return self._councils.get(name)

    def get_all_councils(self) -> Dict[str, BaseCouncil]:
        return dict(self._councils)

    def update_all(self, tick: int, game_state: Dict[str, Any]) -> List[CouncilProposal]:
        all_proposals = []
        for name in self._update_order:
            council = self._councils.get(name)
            if not council or council.state != CouncilState.ACTIVE:
                continue
            if not council.should_update(tick):
                continue
            start = time.time()
            try:
                proposals = council.update(tick, game_state)
                council.metrics.total_compute_time += time.time() - start
                for p in proposals:
                    all_proposals.append(p)
            except Exception as e:
                print(f"Council {name} error: {e}")
                council.state = CouncilState.ERROR
        return all_proposals

    def get_all_metrics(self) -> Dict[str, Any]:
        return {name: c.get_metrics() for name, c in self._councils.items()}
