"""Object Identity Tracking — P6.11

Persistent identity across observation frames.

Raw perception gives us: "there's a human at (1.0, 0.0)"
each frame, but doesn't tell us if it's the SAME human
from the previous frame.

This module solves the identity association problem:
1. Match new observations to existing tracked entities
2. Assign stable IDs that persist across frames
3. Handle entries (new entities appear) and exits (entities leave)
4. Track dwell time, visit count, and behavioral history

The algorithm uses nearest-neighbor matching with a maximum
association distance. When no match is found, a new identity
is created. When an entity is not seen for N frames, it is
marked as exited.

Usage:
    tracker = IdentityTracker(max_distance=2.0)
    
    # Frame 1
    tracker.update([{"type": "human", "position": [1.0, 0.0]}])
    # → assigns ID "human_0"
    
    # Frame 2
    tracker.update([{"type": "human", "position": [1.1, 0.1]}])
    # → matches to "human_0" (close enough)
    
    # Frame 3: entity not seen
    tracker.update([])
    # → "human_0" marked as exited after timeout
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class TrackedEntity:
    """A persistent entity tracked across frames."""
    entity_id: str
    entity_type: str               # "human", "animal", "plant", etc.
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Tracking metadata
    first_seen: float = 0.0        # timestamp of first observation
    last_seen: float = 0.0         # timestamp of most recent observation
    frames_tracked: int = 0        # total frames this entity was seen
    frames_missing: int = 0        # consecutive frames not seen
    
    # Behavioral history
    position_history: list[np.ndarray] = field(default_factory=list)
    max_history: int = 30
    
    # Properties from last observation
    properties: dict = field(default_factory=dict)
    
    @property
    def dwell_time(self) -> float:
        """How long this entity has been tracked (seconds)."""
        return self.last_seen - self.first_seen
    
    @property
    def is_active(self) -> bool:
        """Was this entity seen in the most recent frame?"""
        return self.frames_missing == 0
    
    @property
    def speed(self) -> float:
        """Current speed."""
        return float(np.linalg.norm(self.velocity))
    
    @property
    def mean_position(self) -> np.ndarray:
        """Average position over tracking history."""
        if not self.position_history:
            return self.position.copy()
        return np.mean(self.position_history, axis=0)
    
    @property
    def trajectory_length(self) -> float:
        """Total distance traveled."""
        if len(self.position_history) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.position_history)):
            total += np.linalg.norm(
                self.position_history[i] - self.position_history[i-1])
        return total
    
    def update_position(self, new_pos: np.ndarray, timestamp: float,
                        dt: float = 0.5) -> None:
        """Update position and compute velocity."""
        old_pos = self.position.copy()
        self.position = new_pos.copy()
        
        # Velocity (EMA-smoothed)
        raw_vel = (new_pos - old_pos) / max(dt, 1e-6)
        alpha = 0.3
        self.velocity = alpha * raw_vel + (1 - alpha) * self.velocity
        
        # History
        self.position_history.append(new_pos.copy())
        if len(self.position_history) > self.max_history:
            self.position_history = self.position_history[-self.max_history:]
        
        self.last_seen = timestamp
        self.frames_tracked += 1
        self.frames_missing = 0
    
    def mark_missing(self) -> None:
        """Mark this entity as missing for one frame."""
        self.frames_missing += 1
    
    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "position": self.position.tolist(),
            "speed": self.speed,
            "dwell_time": self.dwell_time,
            "frames_tracked": self.frames_tracked,
            "frames_missing": self.frames_missing,
            "is_active": self.is_active,
            "trajectory_length": self.trajectory_length,
        }


@dataclass
class IdentityTrackerConfig:
    """Configuration for identity tracking."""
    max_association_distance: float = 2.0   # max distance to match entities
    exit_timeout: int = 5                   # frames before marking as exited
    min_dwell_frames: int = 3               # min frames to consider "real"
    type_weight: float = 0.3               # weight for type matching
    position_weight: float = 0.7           # weight for position matching


class IdentityTracker:
    """Tracks persistent entity identities across observation frames.
    
    The tracker maintains a registry of known entities and matches
    new observations to existing entities using nearest-neighbor
    association with configurable distance threshold.
    
    This is the bridge between "what exists right now" and
    "who is this entity that I've been observing."
    
    Usage:
        tracker = IdentityTracker()
        
        # Each perception frame:
        observations = [
            {"type": "human", "position": [1.0, 0.0], "properties": {...}},
        ]
        result = tracker.update(observations)
        
        for entity in result["active"]:
            print(f"{entity.entity_id}: {entity.entity_type} at {entity.position}")
    """
    
    def __init__(self, config: Optional[IdentityTrackerConfig] = None):
        self.config = config or IdentityTrackerConfig()
        self._entities: dict[str, TrackedEntity] = {}
        self._next_id: dict[str, int] = {}  # type → next ID number
        self._frame_count = 0
        self._timestamp = 0.0
    
    @property
    def active_entities(self) -> list[TrackedEntity]:
        return [e for e in self._entities.values() if e.is_active]
    
    @property
    def all_entities(self) -> dict[str, TrackedEntity]:
        return dict(self._entities)
    
    def update(self, observations: list[dict],
               timestamp: Optional[float] = None) -> dict:
        """Process a new frame of observations.
        
        Args:
            observations: list of dicts with keys:
                - type: str (entity type)
                - position: list[float] (3D position)
                - properties: dict (optional additional properties)
            timestamp: current time (auto-generated if None)
        
        Returns:
            dict with:
                - active: list of currently visible TrackedEntity
                - new: list of newly created entities
                - exited: list of entities that just exited
                - matched: dict mapping entity_id → observation index
        """
        self._frame_count += 1
        self._timestamp = timestamp or time.time()
        
        # Mark all current entities as potentially missing
        for entity in self._entities.values():
            entity.mark_missing()
        
        # Match observations to existing entities
        matched = {}
        unmatched_obs = []
        
        for i, obs in enumerate(observations):
            obs_pos = np.asarray(obs.get("position", [0, 0, 0]), dtype=np.float64)
            obs_type = obs.get("type", "unknown")
            
            # Find best matching entity
            best_id = None
            best_score = float('inf')
            
            for eid, entity in self._entities.items():
                if entity.entity_type != obs_type:
                    continue
                
                dist = np.linalg.norm(entity.position - obs_pos)
                if dist <= self.config.max_association_distance:
                    score = dist * self.config.position_weight
                    if score < best_score:
                        best_score = score
                        best_id = eid
            
            if best_id is not None:
                # Match to existing entity
                self._entities[best_id].update_position(
                    obs_pos, self._timestamp)
                if "properties" in obs:
                    self._entities[best_id].properties = obs["properties"]
                matched[best_id] = i
            else:
                unmatched_obs.append((i, obs))
        
        # Create new entities for unmatched observations
        new_entities = []
        for i, obs in unmatched_obs:
            new_id = self._next_id.get(obs.get("type", "unknown"), 0)
            self._next_id[obs.get("type", "unknown")] = new_id + 1
            
            entity_id = f"{obs.get('type', 'unknown')}_{new_id}"
            entity = TrackedEntity(
                entity_id=entity_id,
                entity_type=obs.get("type", "unknown"),
                position=np.asarray(obs.get("position", [0, 0, 0]), dtype=np.float64),
                first_seen=self._timestamp,
                last_seen=self._timestamp,
                frames_tracked=1,
                properties=obs.get("properties", {}),
            )
            entity.position_history.append(entity.position.copy())
            self._entities[entity_id] = entity
            new_entities.append(entity)
            matched[entity_id] = i
        
        # Identify exited entities
        exited = []
        for eid, entity in list(self._entities.items()):
            if (entity.frames_missing > self.config.exit_timeout and
                    entity.entity_id not in matched):
                exited.append(entity)
                del self._entities[eid]
        
        # Identify currently active
        active = [e for e in self._entities.values() if e.is_active]
        
        return {
            "active": active,
            "new": new_entities,
            "exited": exited,
            "matched": matched,
            "frame": self._frame_count,
        }
    
    def get_entity(self, entity_id: str) -> Optional[TrackedEntity]:
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> list[TrackedEntity]:
        return [e for e in self._entities.values()
                if e.entity_type == entity_type]
    
    def get_nearest(self, position: np.ndarray,
                    entity_type: Optional[str] = None) -> Optional[TrackedEntity]:
        """Get the nearest active entity to a position."""
        candidates = self.active_entities
        if entity_type:
            candidates = [e for e in candidates if e.entity_type == entity_type]
        if not candidates:
            return None
        return min(candidates, key=lambda e: np.linalg.norm(e.position - position))
    
    def get_visit_count(self, position: np.ndarray,
                        radius: float = 1.0) -> int:
        """Count how many distinct entities have been seen near a position."""
        count = 0
        for entity in self._entities.values():
            if entity.entity_type == "unknown":
                continue
            # Check if any position in history was within radius
            for hist_pos in entity.position_history:
                if np.linalg.norm(hist_pos - position) <= radius:
                    count += 1
                    break
        return count
    
    def reset(self) -> None:
        """Clear all tracked entities."""
        self._entities.clear()
        self._next_id.clear()
        self._frame_count = 0
