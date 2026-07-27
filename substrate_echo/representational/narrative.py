"""Narrative Layer — compresses causality into temporal explanations.

Narratives answer the question "what happened?" in human-readable form.
They compress event chains into stories with causes, effects, and
significance. This is where the TacticalBrain's hypothesis system
connects to the representational layer.

Architecture:
    Causal Graph (event chains)
          |
          v
    Narrative Layer
      - Compresses event chains into stories
      - Identifies significant moments
      - Tracks narrative arcs (tension, climax, resolution)
      - Provides temporal context for Frame System
          |
          v
    Narrative (compressed temporal explanation)
          |
          v
    Governance / TacticalBrain / Human-readable output
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class NarrativeType(Enum):
    """Types of narratives the system generates."""
    ENGAGEMENT = "engagement"       # battle story
    ECONOMIC = "economic"           # economy story
    STRATEGIC = "strategic"         # high-level strategy story
    TERRAIN = "terrain"             # terrain-driven events
    COMPOSITION = "composition"     # army composition shifts
    SCOUTING = "scouting"           # information gathering story


class NarrativeArc(Enum):
    """The dramatic arc of a narrative."""
    SETUP = "setup"              # establishing context
    RISING_ACTION = "rising"     # tension building
    CLIMAX = "climax"            # peak moment
    FALLING_ACTION = "falling"   # aftermath
    RESOLUTION = "resolution"    # outcome determined


@dataclass
class NarrativeEvent:
    """A significant moment in a narrative — not every event, just the ones
    that matter for the story."""
    event_id: str
    tick: int
    description: str
    significance: float = 0.5    # [0, 1] how important to this narrative
    arc: NarrativeArc = NarrativeArc.SETUP
    entities: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tick": self.tick,
            "description": self.description,
            "significance": round(self.significance, 3),
            "arc": self.arc.value,
            "entities": self.entities,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NarrativeEvent":
        return cls(
            event_id=d["event_id"],
            tick=d["tick"],
            description=d["description"],
            significance=d.get("significance", 0.5),
            arc=NarrativeArc(d.get("arc", "setup")),
            entities=d.get("entities", []),
            data=d.get("data", {}),
        )


@dataclass
class Narrative:
    """A compressed temporal explanation.

    A narrative groups related events into a coherent story with:
      - A type (battle, economy, strategy, etc.)
      - Significant moments with their dramatic arc
      - An outcome
      - A one-line summary

    Narratives are what humans would write as battle reports.
    They compress thousands of individual events into readable stories.
    """
    narrative_id: str
    narrative_type: NarrativeType
    title: str
    summary: str = ""
    events: List[NarrativeEvent] = field(default_factory=list)
    start_tick: int = 0
    end_tick: int = 0
    outcome: str = ""            # "victory", "defeat", "stalemate", "ongoing"
    significance: float = 0.5    # overall significance of this narrative
    entities_involved: List[str] = field(default_factory=list)

    def duration(self) -> int:
        return self.end_tick - self.start_tick

    def peak_moment(self) -> Optional[NarrativeEvent]:
        """The most significant event in this narrative."""
        if not self.events:
            return None
        return max(self.events, key=lambda e: e.significance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative_id": self.narrative_id,
            "narrative_type": self.narrative_type.value,
            "title": self.title,
            "summary": self.summary,
            "events": [e.to_dict() for e in self.events],
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "outcome": self.outcome,
            "significance": round(self.significance, 3),
            "entities_involved": self.entities_involved,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Narrative":
        return cls(
            narrative_id=d["narrative_id"],
            narrative_type=NarrativeType(d["narrative_type"]),
            title=d["title"],
            summary=d.get("summary", ""),
            events=[NarrativeEvent.from_dict(e) for e in d.get("events", [])],
            start_tick=d.get("start_tick", 0),
            end_tick=d.get("end_tick", 0),
            outcome=d.get("outcome", ""),
            significance=d.get("significance", 0.5),
            entities_involved=d.get("entities_involved", []),
        )


class NarrativeLayer:
    """Compresses causal event chains into temporal narratives.

    The narrative layer is the "what happened and why" storyteller.
    It takes raw events from the CausalGraph and groups them into
    coherent narratives with dramatic arcs.

    Usage:
        narrative_layer = NarrativeLayer(causal_graph)
        narrative_layer.process_tick(tick, events)
        active = narrative_layer.get_active_narratives()
        all_narratives = narrative_layer.get_completed_narratives()
    """

    def __init__(self, causal_graph=None, max_active: int = 10,
                 max_completed: int = 50):
        self.causal = causal_graph
        self._active: Dict[str, Narrative] = {}     # ongoing narratives
        self._completed: List[Narrative] = []        # finished narratives
        self._max_active = max_active
        self._max_completed = max_completed
        self._next_id: int = 0
        self._tick: int = 0

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"narr_{self._next_id}"

    # ── Tick Processing ──────────────────────────────────────────

    def process_tick(self, tick: int,
                     events: Optional[List[Any]] = None) -> None:
        """Process new events and update active narratives.

        Args:
            tick: current game tick
            events: list of CausalEvent to incorporate
        """
        self._tick = tick

        if events is None and self.causal:
            events = self.causal.get_events_at_tick(tick)

        if not events:
            return

        for event in events:
            self._incorporate_event(event)

        # Check if any active narratives should be closed
        self._check_completion()

    def _incorporate_event(self, event: Any) -> None:
        """Add an event to the appropriate active narrative, or start a new one."""
        from substrate_echo.representational.causal_graph import EventType

        # Determine which narrative this event belongs to
        narrative_type = self._classify_event(event)

        # Try to find an existing active narrative of this type
        target = None
        for nid, narrative in self._active.items():
            if (narrative.narrative_type == narrative_type and
                    tick_distance(event.tick, narrative.end_tick) < 200):
                target = narrative
                break

        if target is None:
            # Start a new narrative
            target = Narrative(
                narrative_id=self._gen_id(),
                narrative_type=narrative_type,
                title=self._generate_title(event),
                start_tick=event.tick,
            )
            self._active[target.narrative_id] = target

        # Add the event
        significance = self._assess_significance(event)
        arc = self._determine_arc(target, event, significance)

        narrative_event = NarrativeEvent(
            event_id=event.event_id if hasattr(event, 'event_id') else str(self._next_id),
            tick=event.tick,
            description=event.description if hasattr(event, 'description') else str(event),
            significance=significance,
            arc=arc,
            entities=[event.entity_id] if hasattr(event, 'entity_id') else [],
        )

        target.events.append(narrative_event)
        target.end_tick = event.tick
        if hasattr(event, 'entity_id') and event.entity_id:
            if event.entity_id not in target.entities_involved:
                target.entities_involved.append(event.entity_id)

    def _classify_event(self, event: Any) -> NarrativeType:
        """Map an event to a narrative type."""
        from substrate_echo.representational.causal_graph import EventType
        if not hasattr(event, 'event_type'):
            return NarrativeType.STRATEGIC

        mapping = {
            EventType.UNIT_DESTROYED: NarrativeType.ENGAGEMENT,
            EventType.UNIT_DAMAGED: NarrativeType.ENGAGEMENT,
            EventType.ARMY_ENGAGED: NarrativeType.ENGAGEMENT,
            EventType.STRUCTURE_CREATED: NarrativeType.ECONOMIC,
            EventType.STRUCTURE_DESTROYED: NarrativeType.ENGAGEMENT,
            EventType.RESOURCE_CHANGED: NarrativeType.ECONOMIC,
            EventType.SUPPLY_CHANGED: NarrativeType.ECONOMIC,
            EventType.EXPANSION_STARTED: NarrativeType.ECONOMIC,
            EventType.UNIT_MOVED: NarrativeType.STRATEGIC,
            EventType.TACTICAL_MOVE: NarrativeType.STRATEGIC,
            EventType.VISION_CHANGED: NarrativeType.SCOUTING,
            EventType.TECH_RESEARCHED: NarrativeType.COMPOSITION,
        }
        return mapping.get(event.event_type, NarrativeType.STRATEGIC)

    def _assess_significance(self, event: Any) -> float:
        """How significant is this event?"""
        from substrate_echo.representational.causal_graph import EventType
        base = {
            EventType.UNIT_DESTROYED: 0.8,
            EventType.ARMY_ENGAGED: 0.7,
            EventType.STRUCTURE_DESTROYED: 0.9,
            EventType.STRUCTURE_CREATED: 0.4,
            EventType.UNIT_CREATED: 0.2,
            EventType.UNIT_MOVED: 0.1,
            EventType.EXPANSION_STARTED: 0.6,
            EventType.TECH_RESEARCHED: 0.5,
        }
        return base.get(getattr(event, 'event_type', None), 0.3)

    def _determine_arc(self, narrative: Narrative, event: Any,
                       significance: float) -> NarrativeArc:
        """Determine where in the dramatic arc this event falls."""
        event_count = len(narrative.events)

        if event_count == 0:
            return NarrativeArc.SETUP
        elif significance > 0.8:
            return NarrativeArc.CLIMAX
        elif event_count > 5 and significance > 0.5:
            return NarrativeArc.FALLING_ACTION
        elif event_count > 3:
            return NarrativeArc.RISING_ACTION
        else:
            return NarrativeArc.SETUP

    def _generate_title(self, event: Any) -> str:
        """Generate a narrative title from the first event."""
        if hasattr(event, 'entity_name') and hasattr(event, 'event_type'):
            return f"{event.event_type.value}: {event.entity_name}"
        return f"Narrative at tick {getattr(event, 'tick', '?')}"

    def _check_completion(self) -> None:
        """Check if any active narratives should be completed."""
        to_close = []
        for nid, narrative in self._active.items():
            # Close if no events for 500 ticks
            if self._tick - narrative.end_tick > 500:
                narrative.outcome = "stalemate"
                to_close.append(nid)
            # Close if we have enough events for a complete arc
            elif len(narrative.events) > 20:
                has_climax = any(e.arc == NarrativeArc.CLIMAX
                               for e in narrative.events)
                if has_climax:
                    narrative.outcome = self._determine_outcome(narrative)
                    to_close.append(nid)

        for nid in to_close:
            narrative = self._active.pop(nid)
            narrative.summary = self._generate_summary(narrative)
            self._completed.append(narrative)

        # Trim completed narratives
        if len(self._completed) > self._max_completed:
            self._completed = self._completed[-self._max_completed:]

    def _determine_outcome(self, narrative: Narrative) -> str:
        """Determine the outcome of a narrative from its events."""
        # Simple heuristic based on significant events
        peaks = [e for e in narrative.events if e.arc == NarrativeArc.CLIMAX]
        if not peaks:
            return "stalemate"

        # If most significant entities are enemy → we're winning
        own_entities = [eid for eid in narrative.entities_involved
                        if eid.startswith("own")]
        enemy_entities = [eid for eid in narrative.entities_involved
                         if eid.startswith("enemy")]

        if len(own_entities) > len(enemy_entities) * 2:
            return "victory"
        elif len(enemy_entities) > len(own_entities) * 2:
            return "defeat"
        return "stalemate"

    def _generate_summary(self, narrative: Narrative) -> str:
        """Generate a one-line summary of a narrative."""
        peak = narrative.peak_moment()
        duration = narrative.duration()
        n_events = len(narrative.events)

        parts = [f"{narrative.title}"]
        parts.append(f"({n_events} events over {duration} ticks)")
        if peak:
            parts.append(f"peak: {peak.description}")
        if narrative.outcome:
            parts.append(f"→ {narrative.outcome}")
        return " | ".join(parts)

    # ── Query ────────────────────────────────────────────────────

    def get_active_narratives(self) -> List[Narrative]:
        return list(self._active.values())

    def get_completed_narratives(self, limit: int = 20) -> List[Narrative]:
        return self._completed[-limit:]

    def get_all_narratives(self) -> List[Narrative]:
        return list(self._active.values()) + self._completed

    def get_narratives_for_entity(self, entity_id: str) -> List[Narrative]:
        return [n for n in self.get_all_narratives()
                if entity_id in n.entities_involved]

    # ── Summary ──────────────────────────────────────────────────

    def summary_text(self) -> str:
        lines = ["=== NARRATIVE LAYER ==="]
        lines.append(f"  Active: {len(self._active)}")
        lines.append(f"  Completed: {len(self._completed)}")

        for narrative in sorted(self._active.values(),
                               key=lambda n: -n.significance)[:3]:
            lines.append(f"  [{narrative.narrative_type.value}] "
                        f"{narrative.title} "
                        f"({len(narrative.events)} events, "
                        f"sig={narrative.significance:.2f})")

        if self._completed:
            recent = self._completed[-3:]
            lines.append("  Recent completed:")
            for n in recent:
                lines.append(f"    {n.summary}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": {k: v.to_dict() for k, v in self._active.items()},
            "completed": [n.to_dict() for n in self._completed],
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
            self._active = {
                k: Narrative.from_dict(v)
                for k, v in data.get("active", {}).items()
            }
            self._completed = [
                Narrative.from_dict(n) for n in data.get("completed", [])
            ]
            self._next_id = data.get("next_id", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass


def tick_distance(t1: int, t2: int) -> int:
    """Absolute distance between two ticks."""
    return abs(t1 - t2)
