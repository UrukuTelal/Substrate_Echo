"""Event bus for asynchronous council communication."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import time
import threading


class EventType(Enum):
    TICK = "tick"
    GAME_START = "game_start"
    GAME_END = "game_end"
    
    MINERALS_CHANGED = "minerals_changed"
    VESPENE_CHANGED = "vespene_changed"
    SUPPLY_CHANGED = "supply_changed"
    WORKER_COUNT_CHANGED = "worker_count_changed"
    BASE_COUNT_CHANGED = "base_count_changed"
    INCOME_CHANGED = "income_changed"
    
    ENEMY_UNIT_SCOUTED = "enemy_unit_scouted"
    ENEMY_STRUCTURE_SCOUTED = "enemy_structure_scouted"
    ENEMY_EXPANSION_DETECTED = "enemy_expansion_detected"
    ENEMY_TECH_DETECTED = "enemy_tech_detected"
    ENEMY_ARMY_COMPOSITION_CHANGED = "enemy_army_composition_changed"
    
    CAPABILITY_DISCOVERED = "capability_discovered"
    CAPABILITY_TESTED = "capability_tested"
    CAPABILITY_FAILED = "capability_failed"
    
    UNIT_PRODUCED = "unit_produced"
    UNIT_LOST = "unit_lost"
    STRUCTURE_STARTED = "structure_started"
    STRUCTURE_COMPLETED = "structure_completed"
    STRUCTURE_LOST = "structure_lost"
    UPGRADE_STARTED = "upgrade_started"
    UPGRADE_COMPLETED = "upgrade_completed"
    
    COMBAT_ENGAGED = "combat_engaged"
    COMBAT_ENDED = "combat_ended"
    UNIT_UNDER_ATTACK = "unit_under_attack"
    
    PROPOSAL_SUBMITTED = "proposal_submitted"
    PROPOSAL_ACCEPTED = "proposal_accepted"
    PROPOSAL_REJECTED = "proposal_rejected"
    STRATEGY_CHANGED = "strategy_changed"
    
    BELIEF_UPDATED = "belief_updated"
    BELIEF_CONTRADICTED = "belief_contradicted"


@dataclass
class Event:
    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tick: int = 0
    event_id: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.event_type.value}_{self.timestamp:.6f}"


class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable[[Event], None]] = []
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = 5000
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._wildcard_subscribers.append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def unsubscribe_all(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if callback in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(callback)
            for subs in self._subscribers.values():
                if callback in subs:
                    subs.remove(callback)
    
    def publish(self, event: Event) -> None:
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception:
                    pass
            
            for callback in self._wildcard_subscribers:
                try:
                    callback(event)
                except Exception:
                    pass
    
    def publish_simple(self, event_type: EventType, source: str, data: Dict[str, Any] = None, tick: int = 0) -> None:
        event = Event(event_type=event_type, source=source, data=data or {}, tick=tick)
        self.publish(event)
    
    def get_recent_events(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        with self._lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events[-limit:]
    
    def get_history_since(self, timestamp: float) -> List[Event]:
        with self._lock:
            return [e for e in self._event_history if e.timestamp > timestamp]
    
    def clear_history(self) -> None:
        with self._lock:
            self._event_history.clear()


_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(bus: EventBus) -> None:
    global _global_event_bus
    _global_event_bus = bus
