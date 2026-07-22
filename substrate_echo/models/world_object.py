"""World Object — dual physical + relational state representation.

The key insight: objects are not just physical entities with position and velocity.
They carry relational state — familiarity, importance, history, associations —
that shapes how the agent interacts with them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class Affordance(Enum):
    """What can be done with an object."""
    GRASPABLE = auto()
    READABLE = auto()
    CONTAINER = auto()
    SURFACE = auto()
    MOVABLE = auto()
    ILLUMINATES = auto()
    COMMUNICATES = auto()
    OBSERVABLE = auto()


class RelationshipType(Enum):
    """Types of relationships between objects."""
    SPATIAL_PROXIMITY = auto()    # physically near
    FUNCTIONAL_ASSOCIATION = auto()  # used together
    CAUSAL_LINK = auto()          # one affects the other
    SEMANTIC_RELATION = auto()    # conceptually related
    TEMPORAL_ASSOCIATION = auto()  # co-occur in time


@dataclass
class PhysicalState:
    """Physical properties of a world object."""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: tuple[float, float, float] = (0.1, 0.1, 0.1)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    temperature: float = 20.0  # Celsius
    affordances: list[Affordance] = field(default_factory=list)
    is_visible: bool = True
    is_solid: bool = True

    def distance_to(self, other: PhysicalState) -> float:
        """Euclidean distance to another object."""
        return ((self.position[0] - other.position[0]) ** 2 +
                (self.position[1] - other.position[1]) ** 2 +
                (self.position[2] - other.position[2]) ** 2) ** 0.5

    def predicted_position(self, dt: float) -> tuple[float, float, float]:
        """Predict position after dt seconds (linear extrapolation)."""
        return (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            self.position[2] + self.velocity[2] * dt,
        )


@dataclass
class RelationalState:
    """Relational properties — how the agent relates to this object.
    
    This is what makes Substrate_Echo different from a simple spatial model.
    Objects carry meaning beyond their physical properties.
    """
    familiarity: float = 0.0      # 0-1: how well agent knows this object
    importance: float = 0.5       # 0-1: current relevance to agent goals
    emotional_valence: float = 0.0  # -1 to 1: negative=avoidant, positive=approach
    interaction_count: int = 0    # how many times agent has interacted
    last_interaction: float = 0.0  # timestamp of last interaction
    associations: list[str] = field(default_factory=list)  # IDs of related objects
    context_description: str = ""  # human-readable context

    def decay_importance(self, dt: float, decay_rate: float = 0.01) -> None:
        """Importance decays over time without reinforcement."""
        self.importance *= (1.0 - decay_rate * dt)
        self.importance = max(0.01, self.importance)

    def reinforce(self, interaction_strength: float = 0.1) -> None:
        """Interaction reinforces familiarity and importance."""
        self.familiarity = min(1.0, self.familiarity + interaction_strength)
        self.importance = min(1.0, self.importance + interaction_strength * 0.5)
        self.interaction_count += 1
        self.last_interaction = time.time()


@dataclass
class WorldObject:
    """An object in the spatial world model.
    
    Combines physical state (where it is, what it looks like)
    with relational state (what it means to the agent).
    """
    object_id: str
    name: str
    object_type: str  # e.g., "cup", "chair", "person"
    physical: PhysicalState = field(default_factory=PhysicalState)
    relational: RelationalState = field(default_factory=RelationalState)
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def update_from_perception(self, physical_update: PhysicalState) -> None:
        """Update physical state from new perception data."""
        self.physical = physical_update
        self.relational.reinforce(0.01)  # seeing it counts as interaction

    def interact(self, interaction_type: str, strength: float = 0.1) -> None:
        """Record an interaction with this object."""
        self.relational.reinforce(strength)
        self.metadata.setdefault("interactions", []).append({
            "type": interaction_type,
            "time": time.time(),
        })

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "type": self.object_type,
            "position": list(self.physical.position),
            "familiarity": round(self.relational.familiarity, 3),
            "importance": round(self.relational.importance, 3),
            "interactions": self.relational.interaction_count,
        }


@dataclass
class Relationship:
    """A relationship between two world objects."""
    source_id: str
    target_id: str
    rel_type: RelationshipType
    strength: float = 0.5   # 0-1
    description: str = ""
    formed_at: float = field(default_factory=time.time)

    def decay(self, dt: float, decay_rate: float = 0.005) -> None:
        """Relationship weakens without reinforcement."""
        self.strength *= (1.0 - decay_rate * dt)
        self.strength = max(0.0, self.strength)


class SpatialGraph:
    """Graph of relationships between world objects.
    
    Adjacency list representation with typed, weighted edges.
    """
    def __init__(self):
        self._adjacency: dict[str, list[Relationship]] = {}

    def add_relationship(self, rel: Relationship) -> None:
        self._adjacency.setdefault(rel.source_id, []).append(rel)

    def get_neighbors(self, object_id: str) -> list[Relationship]:
        return self._adjacency.get(object_id, [])

    def get_relationships(self, object_id: str, rel_type: Optional[RelationshipType] = None) -> list[Relationship]:
        rels = self._adjacency.get(object_id, [])
        if rel_type is not None:
            rels = [r for r in rels if r.rel_type == rel_type]
        return rels

    def decay_all(self, dt: float) -> None:
        """Decay all relationships over time."""
        for rels in self._adjacency.values():
            for r in rels:
                r.decay(dt)

    def prune(self, min_strength: float = 0.01) -> int:
        """Remove dead relationships. Returns count removed."""
        count = 0
        for object_id in list(self._adjacency.keys()):
            before = len(self._adjacency[object_id])
            self._adjacency[object_id] = [
                r for r in self._adjacency[object_id]
                if r.strength >= min_strength
            ]
            count += before - len(self._adjacency[object_id])
            if not self._adjacency[object_id]:
                del self._adjacency[object_id]
        return count
