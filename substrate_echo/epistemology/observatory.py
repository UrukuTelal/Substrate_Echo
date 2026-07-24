"""Epistemic Observatory — Cognitive instrumentation.

Live observability into the swarm's reasoning processes.

Architecture:
    Cognitive Telemetry
          |
    Event Recording
          |
    Timeline Construction
          |
    Live HUD Display
          |
    Time Travel Debugging
          |
    Counterfactual Replay

Every cognitive subsystem exposes telemetry.
All events are recorded with timestamps.
The HUD displays reasoning alongside behavior.
Time travel enables debugging reasoning chains.
Counterfactuals enable what-if analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import numpy as np
import time
import uuid
import json


class EventType(Enum):
    """Types of cognitive events."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    PREDICTION_OUTCOME = "prediction_outcome"
    TRUST_UPDATE = "trust_update"
    CULTURAL_PRIOR = "cultural_prior"
    DISCOVERY = "discovery"
    RULE_LEARNED = "rule_learned"
    CURIOSITY = "curiosity"
    EXPERIMENT = "experiment"
    META_COGNITION = "meta_cognition"
    ACTION = "action"
    GOAL = "goal"


class ModuleType(Enum):
    """Cognitive modules that expose telemetry."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    TRUST = "trust"
    CULTURAL_PRIOR = "cultural_prior"
    DISCOVERY = "discovery"
    CURIOSITY = "curiosity"
    EXPERIMENT = "experiment"
    META_COGNITION = "meta_cognition"
    EXECUTIVE = "executive"


@dataclass
class CognitiveEvent:
    """A single cognitive event with timestamp."""
    event_id: str
    tick: int
    module: ModuleType
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = 0.0
    
    # Causal chain
    caused_by: Optional[str] = None  # event_id that caused this
    causes: List[str] = field(default_factory=list)  # events this causes
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tick": self.tick,
            "module": self.module.value,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "caused_by": self.caused_by,
        }


@dataclass
class ModuleTelemetry:
    """Telemetry from a cognitive module."""
    module: ModuleType
    
    # Counters
    events_generated: int = 0
    
    # Module-specific metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Recent events (ring buffer)
    recent_events: List[CognitiveEvent] = field(default_factory=list)
    max_recent: int = 50
    
    def record_event(self, event: CognitiveEvent):
        """Record a cognitive event."""
        self.events_generated += 1
        self.recent_events.append(event)
        
        # Keep only recent events
        if len(self.recent_events) > self.max_recent:
            self.recent_events = self.recent_events[-self.max_recent:]
    
    def update_metric(self, key: str, value: float):
        """Update a metric value."""
        self.metrics[key] = value
    
    def get_metric(self, key: str, default: float = 0.0) -> float:
        """Get a metric value."""
        return self.metrics.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module.value,
            "events_generated": self.events_generated,
            "metrics": self.metrics,
            "recent_events": len(self.recent_events),
        }


@dataclass
class CognitiveSnapshot:
    """Snapshot of all cognitive state at a specific tick."""
    tick: int
    timestamp: float
    
    # Active state
    current_goal: Optional[str] = None
    active_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trust state
    trust_levels: Dict[str, float] = field(default_factory=dict)
    
    # Knowledge state
    cultural_priors_applied: List[str] = field(default_factory=list)
    discoveries_recent: List[str] = field(default_factory=list)
    
    # Curiosity state
    research_goals: List[str] = field(default_factory=list)
    uncertainty_level: float = 0.5
    
    # Meta-cognitive state
    reasoning_strategy: str = "default"
    calibration: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "current_goal": self.current_goal,
            "active_hypotheses": len(self.active_hypotheses),
            "predictions": len(self.predictions),
            "trust_levels": self.trust_levels,
            "cultural_priors_applied": len(self.cultural_priors_applied),
            "discoveries_recent": len(self.discoveries_recent),
            "research_goals": len(self.research_goals),
            "uncertainty_level": round(self.uncertainty_level, 3),
            "reasoning_strategy": self.reasoning_strategy,
            "calibration": round(self.calibration, 3),
        }


class EventTimeline:
    """Timeline of all cognitive events."""
    
    def __init__(self):
        self._events: List[CognitiveEvent] = []
        self._events_by_tick: Dict[int, List[CognitiveEvent]] = {}
        self._events_by_module: Dict[ModuleType, List[CognitiveEvent]] = {}
        self._events_by_type: Dict[EventType, List[CognitiveEvent]] = {}
        
        # Snapshots for time travel
        self._snapshots: Dict[int, CognitiveSnapshot] = {}
    
    def record(self, event: CognitiveEvent):
        """Record a cognitive event."""
        self._events.append(event)
        
        # Index by tick
        if event.tick not in self._events_by_tick:
            self._events_by_tick[event.tick] = []
        self._events_by_tick[event.tick].append(event)
        
        # Index by module
        if event.module not in self._events_by_module:
            self._events_by_module[event.module] = []
        self._events_by_module[event.module].append(event)
        
        # Index by type
        if event.event_type not in self._events_by_type:
            self._events_by_type[event.event_type] = []
        self._events_by_type[event.event_type].append(event)
    
    def record_snapshot(self, snapshot: CognitiveSnapshot):
        """Record a cognitive snapshot for time travel."""
        self._snapshots[snapshot.tick] = snapshot
    
    def get_events_at_tick(self, tick: int) -> List[CognitiveEvent]:
        """Get all events at a specific tick."""
        return self._events_by_tick.get(tick, [])
    
    def get_events_in_range(self, start_tick: int, 
                            end_tick: int) -> List[CognitiveEvent]:
        """Get all events in a tick range."""
        events = []
        for tick in range(start_tick, end_tick + 1):
            events.extend(self._events_by_tick.get(tick, []))
        return events
    
    def get_events_by_module(self, module: ModuleType) -> List[CognitiveEvent]:
        """Get all events from a specific module."""
        return self._events_by_module.get(module, [])
    
    def get_events_by_type(self, event_type: EventType) -> List[CognitiveEvent]:
        """Get all events of a specific type."""
        return self._events_by_type.get(event_type, [])
    
    def get_snapshot(self, tick: int) -> Optional[CognitiveSnapshot]:
        """Get snapshot at a specific tick."""
        return self._snapshots.get(tick)
    
    def get_nearest_snapshot(self, tick: int) -> Optional[CognitiveSnapshot]:
        """Get nearest snapshot to a tick."""
        if not self._snapshots:
            return None
        
        # Find nearest tick
        ticks = sorted(self._snapshots.keys())
        nearest = min(ticks, key=lambda t: abs(t - tick))
        return self._snapshots[nearest]
    
    def get_causal_chain(self, event_id: str, 
                         depth: int = 10) -> List[CognitiveEvent]:
        """Get the causal chain leading to an event."""
        chain = []
        visited = set()
        
        def _trace(eid: str, current_depth: int):
            if current_depth <= 0 or eid in visited:
                return
            visited.add(eid)
            
            # Find event
            for event in self._events:
                if event.event_id == eid:
                    chain.append(event)
                    # Trace causes
                    for cause_id in event.caused_by if event.caused_by else []:
                        _trace(cause_id, current_depth - 1)
                    break
        
        _trace(event_id, depth)
        return list(reversed(chain))
    
    def get_timeline_summary(self) -> Dict[str, Any]:
        """Get summary of the timeline."""
        return {
            "total_events": len(self._events),
            "ticks_covered": len(self._events_by_tick),
            "snapshots": len(self._snapshots),
            "by_module": {
                m.value: len(events)
                for m, events in self._events_by_module.items()
            },
            "by_type": {
                t.value: len(events)
                for t, events in self._events_by_type.items()
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return self.get_timeline_summary()


class CognitiveTelemetry:
    """Telemetry collection for all cognitive modules.
    
    Every module registers here and exposes its metrics.
    The observatory reads from here to display the HUD.
    """
    
    def __init__(self):
        self._modules: Dict[ModuleType, ModuleTelemetry] = {}
        self._timeline = EventTimeline()
        
        # Initialize all modules
        for module_type in ModuleType:
            self._modules[module_type] = ModuleTelemetry(module=module_type)
    
    def record_event(self, tick: int, module: ModuleType,
                     event_type: EventType, data: Dict[str, Any],
                     caused_by: Optional[str] = None) -> CognitiveEvent:
        """Record a cognitive event."""
        event = CognitiveEvent(
            event_id=str(uuid.uuid4()),
            tick=tick,
            module=module,
            event_type=event_type,
            data=data,
            caused_by=caused_by,
        )
        
        # Record in timeline
        self._timeline.record(event)
        
        # Record in module telemetry
        self._modules[module].record_event(event)
        
        return event
    
    def update_metric(self, module: ModuleType, key: str, value: float):
        """Update a module metric."""
        if module in self._modules:
            self._modules[module].update_metric(key, value)
    
    def get_module_telemetry(self, module: ModuleType) -> ModuleTelemetry:
        """Get telemetry for a specific module."""
        return self._modules.get(module)
    
    def get_all_telemetry(self) -> Dict[ModuleType, ModuleTelemetry]:
        """Get telemetry for all modules."""
        return self._modules
    
    def get_timeline(self) -> EventTimeline:
        """Get the event timeline."""
        return self._timeline
    
    def get_snapshot(self, tick: int) -> CognitiveSnapshot:
        """Build a snapshot of cognitive state at a tick."""
        events = self._timeline.get_events_at_tick(tick)
        
        # Build snapshot from recent events
        snapshot = CognitiveSnapshot(
            tick=tick,
            timestamp=time.time(),
        )
        
        # Extract state from events
        for event in events:
            if event.event_type == EventType.GOAL:
                snapshot.current_goal = event.data.get("goal", None)
            elif event.event_type == EventType.HYPOTHESIS:
                snapshot.active_hypotheses.append(event.data)
            elif event.event_type == EventType.PREDICTION:
                snapshot.predictions.append(event.data)
            elif event.event_type == EventType.TRUST_UPDATE:
                agent = event.data.get("agent", "unknown")
                level = event.data.get("level", 0.5)
                snapshot.trust_levels[agent] = level
            elif event.event_type == EventType.CULTURAL_PRIOR:
                prior_id = event.data.get("prior_id", "unknown")
                snapshot.cultural_priors_applied.append(prior_id)
            elif event.event_type == EventType.DISCOVERY:
                desc = event.data.get("description", "unknown")
                snapshot.discoveries_recent.append(desc)
            elif event.event_type == EventType.CURIOSITY:
                goals = event.data.get("research_goals", [])
                snapshot.research_goals.extend(goals)
                snapshot.uncertainty_level = event.data.get("uncertainty", 0.5)
        
        # Record snapshot
        self._timeline.record_snapshot(snapshot)
        
        return snapshot
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get summary of all telemetry."""
        return {
            "modules": {
                m.value: t.to_dict()
                for m, t in self._modules.items()
            },
            "timeline": self._timeline.get_timeline_summary(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return self.get_telemetry_summary()


class EpistemicObservatory:
    """The main observatory interface.
    
    Coordinates telemetry collection, event recording,
    and HUD display.
    
    Usage:
        observatory = EpistemicObservatory()
        
        # Record events
        observatory.record_observation(tick, {"enemy_scout": True})
        observatory.record_hypothesis(tick, {"scouting": 0.7, "attack": 0.3})
        observatory.record_prediction(tick, {"leaves_in": 12, "confidence": 0.8})
        
        # Get current state
        snapshot = observatory.get_current_state()
        
        # Display HUD
        hud_text = observatory.render_hud()
    """
    
    def __init__(self):
        self.telemetry = CognitiveTelemetry()
        self._current_tick: int = 0
        self._snapshots: List[CognitiveSnapshot] = []
    
    def tick(self, tick: int):
        """Update observatory to a new tick."""
        self._current_tick = tick
        
        # Take snapshot
        snapshot = self.telemetry.get_snapshot(tick)
        self._snapshots.append(snapshot)
    
    def record_observation(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record an observation event."""
        return self.telemetry.record_event(
            tick, ModuleType.OBSERVATION, EventType.OBSERVATION, data
        )
    
    def record_hypothesis(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a hypothesis event."""
        return self.telemetry.record_event(
            tick, ModuleType.HYPOTHESIS, EventType.HYPOTHESIS, data
        )
    
    def record_prediction(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a prediction event."""
        return self.telemetry.record_event(
            tick, ModuleType.PREDICTION, EventType.PREDICTION, data
        )
    
    def record_prediction_outcome(self, tick: int, data: Dict[str, Any],
                                   caused_by: Optional[str] = None) -> CognitiveEvent:
        """Record a prediction outcome event."""
        return self.telemetry.record_event(
            tick, ModuleType.PREDICTION, EventType.PREDICTION_OUTCOME, data,
            caused_by=caused_by
        )
    
    def record_trust_update(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a trust update event."""
        return self.telemetry.record_event(
            tick, ModuleType.TRUST, EventType.TRUST_UPDATE, data
        )
    
    def record_cultural_prior(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a cultural prior event."""
        return self.telemetry.record_event(
            tick, ModuleType.CULTURAL_PRIOR, EventType.CULTURAL_PRIOR, data
        )
    
    def record_discovery(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a discovery event."""
        return self.telemetry.record_event(
            tick, ModuleType.DISCOVERY, EventType.DISCOVERY, data
        )
    
    def record_curiosity(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a curiosity event."""
        return self.telemetry.record_event(
            tick, ModuleType.CURIOSITY, EventType.CURIOSITY, data
        )
    
    def record_experiment(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record an experiment event."""
        return self.telemetry.record_event(
            tick, ModuleType.EXPERIMENT, EventType.EXPERIMENT, data
        )
    
    def record_action(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record an action event."""
        return self.telemetry.record_event(
            tick, ModuleType.EXECUTIVE, EventType.ACTION, data
        )
    
    def record_goal(self, tick: int, data: Dict[str, Any]) -> CognitiveEvent:
        """Record a goal event."""
        return self.telemetry.record_event(
            tick, ModuleType.EXECUTIVE, EventType.GOAL, data
        )
    
    def get_current_state(self) -> CognitiveSnapshot:
        """Get current cognitive state."""
        return self.telemetry.get_snapshot(self._current_tick)
    
    def get_state_at_tick(self, tick: int) -> Optional[CognitiveSnapshot]:
        """Get cognitive state at a specific tick."""
        return self.telemetry.get_timeline().get_snapshot(tick)
    
    def render_hud(self) -> str:
        """Render the live HUD as text."""
        snapshot = self.get_current_state()
        
        lines = []
        lines.append("=" * 50)
        lines.append(f"Epistemic Observatory -- Tick {snapshot.tick}")
        lines.append("=" * 50)
        
        # Current Goal
        lines.append("")
        lines.append("Current Goal")
        lines.append("-" * 50)
        lines.append(f"  {snapshot.current_goal or 'None'}")
        
        # Observations
        lines.append("")
        lines.append("Recent Observations")
        lines.append("-" * 50)
        obs_events = self.telemetry.get_timeline().get_events_at_tick(snapshot.tick)
        obs_events = [e for e in obs_events if e.event_type == EventType.OBSERVATION]
        for event in obs_events[-3:]:
            for key, value in event.data.items():
                lines.append(f"  [+] {key}: {value}")
        
        # Hypotheses
        lines.append("")
        lines.append("Active Hypotheses")
        lines.append("-" * 50)
        for hyp in snapshot.active_hypotheses[-3:]:
            desc = hyp.get("description", "unknown")
            prob = hyp.get("probability", 0)
            bar_len = int(prob * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"  {desc}")
            lines.append(f"    {bar} {prob:.1%}")
        
        # Predictions
        lines.append("")
        lines.append("Predictions")
        lines.append("-" * 50)
        for pred in snapshot.predictions[-3:]:
            desc = pred.get("description", "unknown")
            conf = pred.get("confidence", 0)
            lines.append(f"  {desc} (confidence: {conf:.1%})")
        
        # Trust
        lines.append("")
        lines.append("Trust Levels")
        lines.append("-" * 50)
        for agent, level in list(snapshot.trust_levels.items())[:5]:
            bar_len = int(level * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"  {agent}: {bar} {level:.2f}")
        
        # Cultural Priors
        lines.append("")
        lines.append("Cultural Priors Applied")
        lines.append("-" * 50)
        for prior in snapshot.cultural_priors_applied[-3:]:
            lines.append(f"  [+] {prior[:20]}...")
        
        # Discoveries
        lines.append("")
        lines.append("Recent Discoveries")
        lines.append("-" * 50)
        for disc in snapshot.discoveries_recent[-3:]:
            lines.append(f"  + {disc[:40]}")
        
        # Curiosity
        lines.append("")
        lines.append("Curiosity")
        lines.append("-" * 50)
        lines.append(f"  Uncertainty: {snapshot.uncertainty_level:.1%}")
        for goal in snapshot.research_goals[-2:]:
            lines.append(f"  -> {goal[:40]}")
        
        # Meta-cognition
        lines.append("")
        lines.append("Meta-Cognition")
        lines.append("-" * 50)
        lines.append(f"  Strategy: {snapshot.reasoning_strategy}")
        lines.append(f"  Calibration: {snapshot.calibration:.1%}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def render_event_feed(self, num_events: int = 10) -> str:
        """Render recent events as a feed."""
        timeline = self.telemetry.get_timeline()
        all_events = timeline._events[-num_events:]
        
        lines = []
        lines.append("=" * 50)
        lines.append("Event Feed")
        lines.append("=" * 50)
        
        for event in all_events:
            lines.append("")
            lines.append(f"[Tick {event.tick}] {event.event_type.value}")
            lines.append(f"  Module: {event.module.value}")
            for key, value in event.data.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_tick": self._current_tick,
            "telemetry": self.telemetry.to_dict(),
            "snapshots": len(self._snapshots),
        }
