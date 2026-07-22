"""Procedural Spatial Memory — P6.5

Attach affordances and action sequences to spatial locations.

"What can happen here?"

As the agent observes entities interacting with the world, each
location accumulates a history of affordances — what actions were
possible, who performed them, and what the outcomes were.

This supports three capabilities:
1. Action suggestion: "At this location, agents typically GATHER"
2. Outcome prediction: "GATHER here has 80% success rate"
3. Social anticipation: "Humans at this location usually COMMUNICATE"

The memory uses a grid-based spatial index. Each cell stores an
AffordanceSummary per (entity_type, action_type) pair. Summaries
track frequency, success rate, and recency-weighted statistics.

Usage:
    memory = SpatialMemory(cell_size=2.0)
    
    # Record an observation
    memory.record(
        position=(1.0, 0.0, 0.0),
        entity_type="human",
        action_type="GATHER",
        success=True,
    )
    
    # Query what can happen here
    affordances = memory.query(position=(1.0, 0.0, 0.0))
    # → [AffordanceSummary(entity_type="human", action_type="GATHER", frequency=0.8)]
    
    # Query what a specific entity type can do
    affordances = memory.query(
        position=(1.0, 0.0, 0.0),
        entity_type="human",
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AffordanceSummary:
    """Statistics for a (entity_type, action_type) at a location."""
    entity_type: str
    action_type: str
    
    count: int = 0              # total observations
    success_count: int = 0      # successful outcomes
    failure_count: int = 0      # failed outcomes
    
    # Recency-weighted statistics (EMA)
    frequency_ema: float = 0.0  # how often this affordance is observed
    success_rate_ema: float = 0.5
    
    # Temporal
    first_observed: float = 0.0
    last_observed: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
    
    @property
    def total(self) -> int:
        return self.success_count + self.failure_count
    
    def record(self, success: bool, timestamp: float = 0.0,
               ema_alpha: float = 0.3) -> None:
        """Record a new observation."""
        self.count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # EMA update
        observed = 1.0
        self.frequency_ema = ema_alpha * observed + (1 - ema_alpha) * self.frequency_ema
        actual = 1.0 if success else 0.0
        self.success_rate_ema = ema_alpha * actual + (1 - ema_alpha) * self.success_rate_ema
        
        if self.first_observed == 0.0:
            self.first_observed = timestamp
        self.last_observed = timestamp
    
    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "action_type": self.action_type,
            "count": self.count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "frequency_ema": self.frequency_ema,
        }


@dataclass
class SpatialCell:
    """A grid cell containing affordance summaries."""
    cell_key: tuple[int, int, int]
    affordances: dict[str, AffordanceSummary] = field(default_factory=dict)
    
    def get_key(self, entity_type: str, action_type: str) -> str:
        return f"{entity_type}::{action_type}"
    
    def record(self, entity_type: str, action_type: str,
               success: bool, timestamp: float = 0.0) -> AffordanceSummary:
        """Record an affordance observation in this cell."""
        key = self.get_key(entity_type, action_type)
        if key not in self.affordances:
            self.affordances[key] = AffordanceSummary(
                entity_type=entity_type,
                action_type=action_type,
            )
        self.affordances[key].record(success, timestamp)
        return self.affordances[key]
    
    def query(self, entity_type: Optional[str] = None,
              action_type: Optional[str] = None) -> list[AffordanceSummary]:
        """Query affordances in this cell with optional filters."""
        results = list(self.affordances.values())
        if entity_type:
            results = [a for a in results if a.entity_type == entity_type]
        if action_type:
            results = [a for a in results if a.action_type == action_type]
        return sorted(results, key=lambda a: a.count, reverse=True)
    
    def decay(self, decay_rate: float = 0.01) -> None:
        """Decay all affordance frequencies."""
        for affordance in self.affordances.values():
            affordance.frequency_ema *= (1.0 - decay_rate)
    
    def prune(self, min_count: int = 1) -> int:
        """Remove affordances with count below threshold."""
        to_remove = [
            key for key, a in self.affordances.items()
            if a.count < min_count
        ]
        for key in to_remove:
            del self.affordances[key]
        return len(to_remove)


class SpatialMemory:
    """Memory of affordances attached to spatial locations.
    
    The world is divided into a grid. Each cell accumulates a
    history of what actions were observed there, broken down by
    entity type and action type. Queries return ranked affordance
    summaries that inform planning and anticipation.
    
    This is the bridge between "where am I?" and "what can I do here?"
    
    Usage:
        memory = SpatialMemory(cell_size=2.0)
        
        # During perception, record what happens at each location
        for entity in perceived_entities:
            memory.record(
                position=entity.position,
                entity_type=entity.type,
                action_type=inferred_action,
                success=True,
            )
        
        # During planning, query what's possible
        options = memory.query(position=agent.position, entity_type="human")
        for affordance in options:
            print(f"{affordance.action_type}: {affordance.success_rate:.0%} success")
    """
    
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self._cells: dict[tuple[int, int, int], SpatialCell] = {}
        self._total_records = 0
    
    @property
    def total_records(self) -> int:
        return self._total_records
    
    @property
    def cell_count(self) -> int:
        return len(self._cells)
    
    def _cell_key(self, position) -> tuple[int, int, int]:
        """Convert world position to grid cell key."""
        pos = np.asarray(position, dtype=np.float64)
        return tuple(int(np.floor(p / self.cell_size)) for p in pos)
    
    def _get_or_create_cell(self, cell_key: tuple[int, int, int]) -> SpatialCell:
        if cell_key not in self._cells:
            self._cells[cell_key] = SpatialCell(cell_key=cell_key)
        return self._cells[cell_key]
    
    def record(self, position, entity_type: str, action_type: str,
               success: bool = True, timestamp: float = 0.0) -> None:
        """Record an affordance observation at a position.
        
        Args:
            position: (x, y, z) or (x, y) world coordinates
            entity_type: type of entity performing the action
            action_type: type of action observed
            success: whether the action succeeded
            timestamp: observation time (for recency tracking)
        """
        cell_key = self._cell_key(position)
        cell = self._get_or_create_cell(cell_key)
        cell.record(entity_type, action_type, success, timestamp)
        self._total_records += 1
    
    def query(self, position, entity_type: Optional[str] = None,
              action_type: Optional[str] = None,
              include_neighbors: bool = False) -> list[AffordanceSummary]:
        """Query affordances at a position.
        
        Args:
            position: (x, y, z) or (x, y) world coordinates
            entity_type: filter by entity type (None = all)
            action_type: filter by action type (None = all)
            include_neighbors: if True, also query adjacent cells
        
        Returns:
            List of AffordanceSummary, sorted by frequency (most common first)
        """
        cell_key = self._cell_key(position)
        
        if include_neighbors:
            summaries = {}
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nk = (cell_key[0] + dx, cell_key[1] + dy, cell_key[2] + dz)
                        if nk in self._cells:
                            for a in self._cells[nk].query(entity_type, action_type):
                                key = (a.entity_type, a.action_type)
                                if key not in summaries:
                                    summaries[key] = AffordanceSummary(
                                        entity_type=a.entity_type,
                                        action_type=a.action_type,
                                    )
                                # Merge counts
                                summaries[key].count += a.count
                                summaries[key].success_count += a.success_count
                                summaries[key].failure_count += a.failure_count
                                summaries[key].frequency_ema = max(
                                    summaries[key].frequency_ema, a.frequency_ema)
            return sorted(summaries.values(), key=lambda a: a.count, reverse=True)
        
        if cell_key not in self._cells:
            return []
        return self._cells[cell_key].query(entity_type, action_type)
    
    def suggest_actions(self, position, entity_type: str,
                        top_k: int = 3) -> list[AffordanceSummary]:
        """Suggest most likely actions for an entity at a position.
        
        Returns the top-k most frequent affordances for the entity type.
        """
        affordances = self.query(position, entity_type=entity_type)
        return affordances[:top_k]
    
    def predict_outcome(self, position, entity_type: str,
                        action_type: str) -> Optional[float]:
        """Predict success probability for an action at a position.
        
        Returns success rate (0-1) or None if no data.
        """
        affordances = self.query(
            position, entity_type=entity_type, action_type=action_type)
        if not affordances:
            return None
        return affordances[0].success_rate
    
    def get_density(self, position, radius: float = 1.0) -> int:
        """Count total observations near a position."""
        cell_key = self._cell_key(position)
        count = 0
        r_cells = int(np.ceil(radius / self.cell_size))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    nk = (cell_key[0] + dx, cell_key[1] + dy, cell_key[2] + dz)
                    if nk in self._cells:
                        count += sum(a.count for a in self._cells[nk].affordances.values())
        return count
    
    def decay(self, decay_rate: float = 0.01) -> None:
        """Decay all affordance frequencies across all cells."""
        for cell in self._cells.values():
            cell.decay(decay_rate)
    
    def prune(self, min_count: int = 1) -> dict[str, int]:
        """Remove low-count affordances and empty cells."""
        cells_pruned = 0
        affordances_pruned = 0
        for cell_key in list(self._cells.keys()):
            n = self._cells[cell_key].prune(min_count)
            affordances_pruned += n
            if not self._cells[cell_key].affordances:
                del self._cells[cell_key]
                cells_pruned += 1
        return {
            "cells_pruned": cells_pruned,
            "affordances_pruned": affordances_pruned,
        }
    
    def to_dict(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "cell_count": self.cell_count,
            "total_records": self.total_records,
        }
