"""Causal Graph — chains events into consequences.

The causal graph is the dynamic layer that records what happened
and what it caused. It enables the Frame System to ask "what led
to this?" and the Narrative Layer to compress sequences into stories.

Events are timestamped observations of state changes:
  - Unit created, destroyed, took damage, changed location
  - Economy changed (minerals gathered, supply blocked)
  - Strategic events (army engaged, expansion started)

Consequences are the predicted effects of events:
  - Damage → eventual destruction
  - Supply block → delayed production
  - Army movement → territory control change

Architecture:
    Event (observed state change)
          |
          v
    Consequence (predicted effect)
          |
          v
    Event Chain (sequence of causally related events)
          |
          v
    Narrative Layer (compression into stories)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import time


class EventType(Enum):
    """Categories of observable events."""
    UNIT_CREATED = "unit_created"
    UNIT_DESTROYED = "unit_destroyed"
    UNIT_DAMAGED = "unit_damaged"
    UNIT_MOVED = "unit_moved"
    STRUCTURE_CREATED = "structure_created"
    STRUCTURE_DESTROYED = "structure_destroyed"
    RESOURCE_CHANGED = "resource_changed"
    SUPPLY_CHANGED = "supply_changed"
    ARMY_ENGAGED = "army_engaged"
    EXPANSION_STARTED = "expansion_started"
    TECH_RESEARCHED = "tech_researched"
    VISION_CHANGED = "vision_changed"
    TACTICAL_MOVE = "tactical_move"


class ConsequenceType(Enum):
    """Types of predicted consequences."""
    ECONOMIC_IMPACT = "economic_impact"
    MILITARY_IMPACT = "military_impact"
    TERRITORY_CHANGE = "territory_change"
    INFORMATION_GAIN = "information_gain"
    PRODUCTION_DELAY = "production_delay"
    COMPOSITION_SHIFT = "composition_shift"


@dataclass
class CausalEvent:
    """A timestamped observation of a state change."""
    event_id: str
    event_type: EventType
    tick: int
    entity_id: str
    entity_name: str
    description: str
    confidence: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "interpreter"  # what produced this event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "tick": self.tick,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "description": self.description,
            "confidence": self.confidence,
            "data": self.data,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalEvent":
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            tick=d["tick"],
            entity_id=d["entity_id"],
            entity_name=d["entity_name"],
            description=d["description"],
            confidence=d.get("confidence", 1.0),
            data=d.get("data", {}),
            source=d.get("source", "interpreter"),
        )


@dataclass
class Consequence:
    """A predicted effect of one or more events."""
    consequence_id: str
    consequence_type: ConsequenceType
    caused_by: List[str]       # event_ids that caused this
    tick: int
    description: str
    severity: float = 0.5      # [0, 1] how significant
    confidence: float = 0.5    # how sure we are
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consequence_id": self.consequence_id,
            "consequence_type": self.consequence_type.value,
            "caused_by": self.caused_by,
            "tick": self.tick,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Consequence":
        return cls(
            consequence_id=d["consequence_id"],
            consequence_type=ConsequenceType(d["consequence_type"]),
            caused_by=d["caused_by"],
            tick=d["tick"],
            description=d["description"],
            severity=d.get("severity", 0.5),
            confidence=d.get("confidence", 0.5),
            data=d.get("data", {}),
        )


@dataclass
class EventChain:
    """A causally related sequence of events.

    Chains are the raw material for narratives — they group related
    events into coherent threads.
    """
    chain_id: str
    events: List[str]          # event_ids in temporal order
    consequences: List[str]    # consequence_ids
    description: str = ""
    start_tick: int = 0
    end_tick: int = 0
    is_open: bool = True       # still accumulating events?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "events": self.events,
            "consequences": self.consequences,
            "description": self.description,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "is_open": self.is_open,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventChain":
        return cls(
            chain_id=d["chain_id"],
            events=d["events"],
            consequences=d.get("consequences", []),
            description=d.get("description", ""),
            start_tick=d.get("start_tick", 0),
            end_tick=d.get("end_tick", 0),
            is_open=d.get("is_open", True),
        )


class CausalGraph:
    """Records events and their consequences.

    The causal graph is the "what happened and why" layer. It enables:
      - Backward reasoning: "what caused this state?"
      - Forward prediction: "what will this cause?"
      - Pattern detection: "this sequence has happened before"
      - Narrative generation: compress event chains into stories

    Design:
      - Events are immutable once recorded (append-only)
      - Consequences are computed from events (can be refined)
      - Event chains group causally related events
      - Old chains are compressed into summaries
    """

    def __init__(self, max_events: int = 5000, max_chains: int = 100):
        self._events: Dict[str, CausalEvent] = {}
        self._consequences: Dict[str, Consequence] = {}
        self._chains: Dict[str, EventChain] = {}
        self._events_by_tick: Dict[int, List[str]] = {}  # tick → event_ids
        self._events_by_entity: Dict[str, List[str]] = {}  # entity → event_ids
        self._max_events = max_events
        self._max_chains = max_chains
        self._next_id: int = 0
        self._tick: int = 0

    def _gen_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    # ── Event Recording ──────────────────────────────────────────

    def record_event(self, event_type: EventType, tick: int,
                     entity_id: str, entity_name: str,
                     description: str, confidence: float = 1.0,
                     data: Optional[Dict[str, Any]] = None) -> CausalEvent:
        """Record a new event in the causal graph."""
        event = CausalEvent(
            event_id=self._gen_id("evt"),
            event_type=event_type,
            tick=tick,
            entity_id=entity_id,
            entity_name=entity_name,
            description=description,
            confidence=confidence,
            data=data or {},
        )
        self._events[event.event_id] = event

        # Index by tick
        if tick not in self._events_by_tick:
            self._events_by_tick[tick] = []
        self._events_by_tick[tick].append(event.event_id)

        # Index by entity
        if entity_id not in self._events_by_entity:
            self._events_by_entity[entity_id] = []
        self._events_by_entity[entity_id].append(event.event_id)

        # Auto-generate consequences for high-impact events
        self._auto_consequences(event)

        return event

    def record_unit_destroyed(self, entity_id: str, entity_name: str,
                              tick: int, killed_by: str = "",
                              confidence: float = 1.0) -> CausalEvent:
        """Convenience: record a unit destruction."""
        return self.record_event(
            EventType.UNIT_DESTROYED, tick, entity_id, entity_name,
            f"{entity_name} destroyed" + (f" by {killed_by}" if killed_by else ""),
            confidence, {"killed_by": killed_by},
        )

    def record_engagement(self, attacker_id: str, defender_id: str,
                          tick: int, damage: float = 0.0) -> CausalEvent:
        """Convenience: record an army engagement."""
        return self.record_event(
            EventType.ARMY_ENGAGED, tick, attacker_id, "engagement",
            f"Engagement: {attacker_id} → {defender_id}",
            0.8, {"defender": defender_id, "damage": damage},
        )

    # ── Consequence Computation ──────────────────────────────────

    def _auto_consequences(self, event: CausalEvent) -> None:
        """Generate consequences for high-impact events."""
        if event.event_type == EventType.UNIT_DESTROYED:
            self._consequences[self._gen_id("con")] = Consequence(
                consequence_id=self._gen_id("con"),
                consequence_type=ConsequenceType.MILITARY_IMPACT,
                caused_by=[event.event_id],
                tick=event.tick,
                description=f"Military loss: {event.entity_name}",
                severity=0.7,
                confidence=event.confidence,
            )
        elif event.event_type == EventType.STRUCTURE_CREATED:
            self._consequences[self._gen_id("con")] = Consequence(
                consequence_id=self._gen_id("con"),
                consequence_type=ConsequenceType.ECONOMIC_IMPACT,
                caused_by=[event.event_id],
                tick=event.tick,
                description=f"New structure: {event.entity_name}",
                severity=0.4,
                confidence=event.confidence,
            )
        elif event.event_type == EventType.ARMY_ENGAGED:
            self._consequences[self._gen_id("con")] = Consequence(
                consequence_id=self._gen_id("con"),
                consequence_type=ConsequenceType.MILITARY_IMPACT,
                caused_by=[event.event_id],
                tick=event.tick,
                description=f"Army engagement involving {event.entity_name}",
                severity=0.6,
                confidence=event.confidence,
            )

    # ── Event Chain Management ───────────────────────────────────

    def get_events_at_tick(self, tick: int) -> List[CausalEvent]:
        """All events that occurred at a specific tick."""
        ids = self._events_by_tick.get(tick, [])
        return [self._events[eid] for eid in ids if eid in self._events]

    def get_events_for_entity(self, entity_id: str,
                              limit: int = 50) -> List[CausalEvent]:
        """Recent events involving a specific entity."""
        ids = self._events_by_entity.get(entity_id, [])[-limit:]
        return [self._events[eid] for eid in ids if eid in self._events]

    def get_events_in_range(self, start_tick: int,
                            end_tick: int) -> List[CausalEvent]:
        """All events in a time range."""
        events = []
        for tick in range(start_tick, end_tick + 1):
            events.extend(self.get_events_at_tick(tick))
        return events

    def get_causes(self, event_id: str) -> List[CausalEvent]:
        """Find events that could have caused this event (temporal backwards)."""
        event = self._events.get(event_id)
        if not event:
            return []
        # Look at events in the 100 ticks before this one
        candidates = []
        for tick in range(max(0, event.tick - 100), event.tick):
            for e in self.get_events_at_tick(tick):
                # Consider it a potential cause if related entity or type
                if (e.entity_id == event.entity_id or
                    e.event_type in self._related_types(event.event_type)):
                    candidates.append(e)
        return candidates

    def get_effects(self, event_id: str) -> List[Consequence]:
        """Find consequences caused by this event."""
        return [c for c in self._consequences.values()
                if event_id in c.caused_by]

    def _related_types(self, event_type: EventType) -> Set[EventType]:
        """Types that could causally relate to this type."""
        relations = {
            EventType.UNIT_CREATED: {EventType.STRUCTURE_CREATED},
            EventType.UNIT_DESTROYED: {EventType.ARMY_ENGAGED, EventType.UNIT_DAMAGED},
            EventType.UNIT_DAMAGED: {EventType.ARMY_ENGAGED},
            EventType.ARMY_ENGAGED: {EventType.UNIT_MOVED, EventType.TACTICAL_MOVE},
            EventType.STRUCTURE_CREATED: {EventType.RESOURCE_CHANGED},
            EventType.EXPANSION_STARTED: {EventType.STRUCTURE_CREATED},
            EventType.VISION_CHANGED: {EventType.UNIT_MOVED},
        }
        return relations.get(event_type, set())

    # ── Compression ──────────────────────────────────────────────

    def compress_old_events(self, before_tick: int,
                            max_per_entity: int = 20) -> int:
        """Compress old events into summaries.

        Keeps the most significant events per entity and removes the rest.
        Returns number of events removed.
        """
        removed = 0
        for entity_id, event_ids in list(self._events_by_entity.items()):
            old_ids = [eid for eid in event_ids
                       if self._events.get(eid, CausalEvent("", EventType.UNIT_CREATED, 0, "", "")).tick < before_tick]
            if len(old_ids) > max_per_entity:
                # Keep the most significant (destroyed > engaged > damaged > moved)
                priority = {
                    EventType.UNIT_DESTROYED: 0,
                    EventType.ARMY_ENGAGED: 1,
                    EventType.UNIT_DAMAGED: 2,
                    EventType.STRUCTURE_CREATED: 3,
                    EventType.UNIT_CREATED: 4,
                    EventType.UNIT_MOVED: 5,
                }
                old_ids.sort(key=lambda eid: priority.get(
                    self._events[eid].event_type, 99))
                to_remove = old_ids[max_per_entity:]
                for eid in to_remove:
                    if eid in self._events:
                        del self._events[eid]
                        removed += 1
                self._events_by_entity[entity_id] = [
                    eid for eid in event_ids if eid in self._events]

        return removed

    # ── Summary ──────────────────────────────────────────────────

    def summary_text(self, window: int = 100) -> str:
        """Summary of recent causal activity."""
        lines = ["=== CAUSAL GRAPH ==="]
        lines.append(f"  Events: {len(self._events)}")
        lines.append(f"  Consequences: {len(self._consequences)}")

        # Event type breakdown
        type_counts: Dict[str, int] = {}
        for e in self._events.values():
            type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1
        if type_counts:
            lines.append("  Events by type:")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {t}: {c}")

        # Recent significant events
        if self._events:
            recent = sorted(self._events.values(), key=lambda e: e.tick)[-5:]
            lines.append("  Recent events:")
            for e in recent:
                lines.append(f"    [{e.tick}] {e.event_type.value}: {e.description}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": {k: v.to_dict() for k, v in self._events.items()},
            "consequences": {k: v.to_dict() for k, v in self._consequences.items()},
            "chains": {k: v.to_dict() for k, v in self._chains.items()},
            "next_id": self._next_id,
        }

    def save(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self._events = {
                k: CausalEvent.from_dict(v)
                for k, v in data.get("events", {}).items()
            }
            self._consequences = {
                k: Consequence.from_dict(v)
                for k, v in data.get("consequences", {}).items()
            }
            self._chains = {
                k: EventChain.from_dict(v)
                for k, v in data.get("chains", {}).items()
            }
            self._next_id = data.get("next_id", 0)
            # Rebuild indexes
            self._events_by_tick.clear()
            self._events_by_entity.clear()
            for eid, event in self._events.items():
                self._events_by_tick.setdefault(event.tick, []).append(eid)
                self._events_by_entity.setdefault(event.entity_id, []).append(eid)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
