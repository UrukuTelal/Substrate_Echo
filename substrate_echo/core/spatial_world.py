"""Spatial World Model — the agent's internal representation of physical space.

Combines spatial indexing for efficient queries with a relational graph
that captures how objects relate to each other and to the agent.
"""

from __future__ import annotations
from typing import Optional
import sys
import os

# Add DeveloperConsole to path for BlochPSV access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "DeveloperConsole", "backend"))

from ..models.world_object import WorldObject, Relationship, SpatialGraph, RelationshipType


class SpatialWorldModel:
    """The agent's internal model of physical space.
    
    Features:
    - Object registry with spatial indexing
    - Relationship graph between objects
    - Prediction of object trajectories
    - Region queries for spatial awareness
    """
    
    def __init__(self, grid_resolution: float = 0.1):
        self.objects: dict[str, WorldObject] = {}
        self.graph = SpatialGraph()
        self.grid_resolution = grid_resolution  # meters per cell
        self._spatial_index: dict[tuple[int, int, int], list[str]] = {}  # grid cell → object IDs
    
    def add_object(self, obj: WorldObject) -> None:
        """Register a new object in the world model."""
        self.objects[obj.object_id] = obj
        self._index_object(obj)
    
    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the world model."""
        if object_id in self.objects:
            self._unindex_object(self.objects[object_id])
            del self.objects[object_id]
            return True
        return False
    
    def get_object(self, object_id: str) -> Optional[WorldObject]:
        return self.objects.get(object_id)
    
    def update_from_perception(self, object_id: str, physical_update) -> None:
        """Update an object's physical state from new sensor data."""
        if object_id in self.objects:
            self._unindex_object(self.objects[object_id])
            self.objects[object_id].update_from_perception(physical_update)
            self._index_object(self.objects[object_id])
    
    def query_region(self, center: tuple[float, float, float],
                     radius: float) -> list[WorldObject]:
        """Get all objects within a spherical region."""
        results = []
        cx, cy, cz = center
        r2 = radius * radius
        for obj in self.objects.values():
            px, py, pz = obj.physical.position
            d2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if d2 <= r2:
                results.append(obj)
        return results
    
    def get_nearest(self, position: tuple[float, float, float],
                    k: int = 1) -> list[WorldObject]:
        """Get k nearest objects to a position."""
        ranked = sorted(
            self.objects.values(),
            key=lambda o: sum((a - b) ** 2 for a, b in zip(o.physical.position, position))
        )
        return ranked[:k]
    
    def get_relationships(self, object_id: str,
                          rel_type: Optional[RelationshipType] = None) -> list[Relationship]:
        """Get all relationships involving an object."""
        return self.graph.get_relationships(object_id, rel_type)
    
    def predict_positions(self, dt: float) -> dict[str, tuple[float, float, float]]:
        """Predict where all objects will be after dt seconds."""
        return {
            oid: obj.physical.predicted_position(dt)
            for oid, obj in self.objects.items()
        }
    
    def decay_importance(self, dt: float) -> None:
        """All objects' importance decays over time."""
        for obj in self.objects.values():
            obj.relational.decay_importance(dt)
        self.graph.decay_all(dt)
    
    def prune(self) -> dict[str, int]:
        """Remove dead objects and relationships. Returns counts."""
        # Prune weak relationships
        rels_pruned = self.graph.prune()
        # Prune inactive objects
        objs_pruned = 0
        for oid in list(self.objects.keys()):
            if not self.objects[oid].is_active:
                self._unindex_object(self.objects[oid])
                del self.objects[oid]
                objs_pruned += 1
        return {"objects_pruned": objs_pruned, "relationships_pruned": rels_pruned}
    
    def _grid_cell(self, position: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(p / self.grid_resolution) for p in position)
    
    def _index_object(self, obj: WorldObject) -> None:
        cell = self._grid_cell(obj.physical.position)
        self._spatial_index.setdefault(cell, []).append(obj.object_id)
    
    def _unindex_object(self, obj: WorldObject) -> None:
        cell = self._grid_cell(obj.physical.position)
        if cell in self._spatial_index:
            self._spatial_index[cell] = [
                oid for oid in self._spatial_index[cell]
                if oid != obj.object_id
            ]
    
    def to_dict(self) -> dict:
        return {
            "object_count": len(self.objects),
            "objects": [o.to_dict() for o in self.objects.values()],
        }
