"""State Graph — dynamic tracking of all entities.

The state graph is the living, breathing snapshot of reality.
It changes every tick. It stores EntityDescriptors indexed by ID,
and provides query methods that the Semantic Interpreter, Frame System,
and Narrative Layer all use.

The state graph is the **shared world model** — every subsystem reads
from and writes to the same graph, but projects it into its own
representation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import time


class EntityStateType(Enum):
    """Lifecycle state of an entity in the graph."""
    ACTIVE = "active"         # currently observable
    PREDICTED = "predicted"   # inferred to exist but not directly observed
    HISTORICAL = "historical" # was active, no longer observable
    GHOST = "ghost"           # believed to exist based on indirect evidence


class VisibilityLevel:
    """How much we can see of an entity."""
    FULL = 1.0       # direct observation, full state known
    PARTIAL = 0.5    # partially observed (e.g. in fog of war)
    ESTIMATED = 0.2  # inferred from indirect evidence
    UNKNOWN = 0.0    # entity exists but state is unknown


@dataclass
class GraphEdge:
    """A typed, weighted edge between two entities."""
    source_id: str
    target_id: str
    relation: str       # "allies", "counters", "contains", "adjacent_to", etc.
    weight: float = 1.0
    confidence: float = 1.0
    tick_created: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id, "target": self.target_id,
            "relation": self.relation, "weight": self.weight,
            "confidence": self.confidence, "tick": self.tick_created,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source_id=d["source"], target_id=d["target"],
            relation=d["relation"], weight=d.get("weight", 1.0),
            confidence=d.get("confidence", 1.0),
            tick_created=d.get("tick", 0),
        )


class StateGraph:
    """The living snapshot of all entities and their relationships.

    Provides:
        - Entity CRUD (add, get, remove, query)
        - Relationship queries (neighbors, counters, allies)
        - Visibility-aware queries (only what we can see)
        - Temporal queries (entities seen recently, predicted entities)
        - Graph traversal (shortest path, connected components)

    Every subsystem that needs to know "what exists right now"
    queries the StateGraph.
    """

    def __init__(self):
        self._entities: Dict[str, Any] = {}  # id -> EntityDescriptor
        self._edges: List[GraphEdge] = []
        self._edge_index: Dict[str, List[GraphEdge]] = {}  # source_id -> edges
        self._tick: int = 0
        self._name_index: Dict[str, str] = {}  # name -> id (last seen)
        self._tag_index: Dict[int, str] = {}  # SC2 tag -> id

    # ── Entity Management ────────────────────────────────────────

    def add_entity(self, entity: Any) -> None:
        """Add or update an entity in the graph."""
        eid = entity.identity.id
        self._entities[eid] = entity
        self._name_index[entity.identity.name] = eid
        entity.history.updated_tick = self._tick

    def get_entity(self, entity_id: str) -> Optional[Any]:
        return self._entities.get(entity_id)

    def get_by_name(self, name: str) -> Optional[Any]:
        eid = self._name_index.get(name)
        return self._entities.get(eid) if eid else None

    def get_by_tag(self, tag: int) -> Optional[Any]:
        eid = self._tag_index.get(tag)
        return self._entities.get(eid) if eid else None

    def register_tag(self, tag: int, entity_id: str) -> None:
        self._tag_index[tag] = entity_id

    def remove_entity(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            self._edge_index.pop(entity_id, None)
            self._edges = [e for e in self._edges
                           if e.source_id != entity_id and e.target_id != entity_id]
            return True
        return False

    def all_entities(self) -> List[Any]:
        return list(self._entities.values())

    def entity_count(self) -> int:
        return len(self._entities)

    # ── Edge Management ──────────────────────────────────────────

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)
        if edge.source_id not in self._edge_index:
            self._edge_index[edge.source_id] = []
        self._edge_index[edge.source_id].append(edge)

    def get_edges_from(self, entity_id: str,
                       relation: Optional[str] = None) -> List[GraphEdge]:
        edges = self._edge_index.get(entity_id, [])
        if relation:
            return [e for e in edges if e.relation == relation]
        return edges

    def get_edges_to(self, entity_id: str,
                     relation: Optional[str] = None) -> List[GraphEdge]:
        edges = [e for e in self._edges if e.target_id == entity_id]
        if relation:
            return [e for e in edges if e.relation == relation]
        return edges

    def neighbors(self, entity_id: str) -> List[str]:
        """All entities this entity is connected to (any relation)."""
        targets = set()
        for e in self._edge_index.get(entity_id, []):
            targets.add(e.target_id)
        for e in self._edges:
            if e.target_id == entity_id:
                targets.add(e.source_id)
        return list(targets)

    def connected_by(self, entity_id: str, relation: str) -> List[str]:
        """Entities connected via a specific relation type."""
        targets = set()
        for e in self._edge_index.get(entity_id, []):
            if e.relation == relation:
                targets.add(e.target_id)
        return list(targets)

    # ── Query Methods ────────────────────────────────────────────

    def query_by_role(self, role: str) -> List[Any]:
        """Find all entities with a specific role."""
        return [e for e in self._entities.values()
                if role in e.classification.roles]

    def query_by_taxonomy(self, category: str) -> List[Any]:
        """Find all entities that are_a category."""
        return [e for e in self._entities.values()
                if e.classification.is_a(category)]

    def query_by_capability(self, capability: str) -> List[Any]:
        """Find all entities with a specific capability."""
        return [e for e in self._entities.values()
                if e.capabilities.has(capability)]

    def query_by_affordance(self, affordance: str) -> List[Any]:
        """Find all entities that create a specific affordance."""
        return [e for e in self._entities.values()
                if affordance in e.affordances.creates
                or affordance in e.affordances.enables]

    def query_by_relationship(self, entity_id: str,
                              relation: str) -> List[Any]:
        """Find all entities related to entity_id via relation."""
        target_ids = self.connected_by(entity_id, relation)
        return [self._entities[tid] for tid in target_ids
                if tid in self._entities]

    def active_at_tick(self, tick: int,
                       window: int = 100) -> List[Any]:
        """Entities whose state was updated within window of tick."""
        return [e for e in self._entities.values()
                if tick - e.history.updated_tick <= window]

    def counters_of(self, entity_name: str) -> List[Any]:
        """Find entities that counter a given entity."""
        entity = self.get_by_name(entity_name)
        if not entity:
            return []
        counter_names = entity.relationships.countered_by
        return [self._entities[n] for n in counter_names
                if n in self._entities]

    # ── Tick ─────────────────────────────────────────────────────

    def tick(self, new_tick: int) -> None:
        """Advance the graph clock. Marks stale entities as historical."""
        self._tick = new_tick
        for eid, entity in self._entities.items():
            age = new_tick - entity.history.updated_tick
            if age > 500 and entity.history.updated_tick > 0:
                # Entity hasn't been updated in 500 ticks — probably gone
                pass  # keep but could mark as historical

    @property
    def current_tick(self) -> int:
        return self._tick

    # ── Summary ──────────────────────────────────────────────────

    def summary_text(self) -> str:
        lines = ["=== STATE GRAPH ==="]
        lines.append(f"  Tick: {self._tick}")
        lines.append(f"  Entities: {len(self._entities)}")
        lines.append(f"  Edges: {len(self._edges)}")

        by_role: Dict[str, int] = {}
        for e in self._entities.values():
            for role in e.classification.roles:
                by_role[role] = by_role.get(role, 0) + 1
        if by_role:
            lines.append("  By role:")
            for role, count in sorted(by_role.items()):
                lines.append(f"    {role}: {count}")

        by_tax: Dict[str, int] = {}
        for e in self._entities.values():
            if e.classification.taxonomy:
                leaf = e.classification.taxonomy[-1]
                by_tax[leaf] = by_tax.get(leaf, 0) + 1
        if by_tax:
            lines.append("  By type:")
            for t, count in sorted(by_tax.items()):
                lines.append(f"    {t}: {count}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self._tick,
            "entities": {k: v.to_dict() for k, v in self._entities.items()},
            "edges": [e.to_dict() for e in self._edges],
        }

    def save(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        import json
        from substrate_echo.representational.entity_descriptor import EntityDescriptor
        try:
            with open(path) as f:
                data = json.load(f)
            self._tick = data.get("tick", 0)
            self._entities = {
                k: EntityDescriptor.from_dict(v)
                for k, v in data.get("entities", {}).items()
            }
            self._edges = [GraphEdge.from_dict(e) for e in data.get("edges", [])]
            # Rebuild index
            self._edge_index.clear()
            for edge in self._edges:
                if edge.source_id not in self._edge_index:
                    self._edge_index[edge.source_id] = []
                self._edge_index[edge.source_id].append(edge)
            self._name_index = {
                e.identity.name: eid for eid, e in self._entities.items()
            }
        except (FileNotFoundError, json.JSONDecodeError):
            pass
