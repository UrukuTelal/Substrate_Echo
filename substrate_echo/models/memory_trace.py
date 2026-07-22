"""Memory Trace — a snapshot of an attractor memory.

When an attractor is recalled, it unfolds into a MemoryTrace that contains
the rich experiential data that formed the attractor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class TraceType(Enum):
    """Types of memory traces."""
    EPISODIC = auto()      # specific event记忆
    SEMANTIC = auto()      # general knowledge
    PROCEDURAL = auto()    # how to do something
    EMOTIONAL = auto()     # emotional association
    SPATIAL = auto()       # spatial/layout memory


@dataclass
class MemoryTrace:
    """A recalled memory trace from an attractor.
    
    This is what you get when you query the attractor memory.
    It contains the experiential data that was compressed into the attractor.
    """
    trace_id: str
    trace_type: TraceType
    formed_at: float = field(default_factory=time.time)
    last_recalled: float = field(default_factory=time.time)
    recall_count: int = 0
    
    # The attractor center (16D state that defines this memory)
    attractor_center: list[float] = field(default_factory=lambda: [0.0] * 16)
    
    # Experiential content
    description: str = ""
    object_ids: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # chronological events
    
    # Emotional/emphasis content
    emotional_valence: float = 0.0  # -1 to 1
    importance: float = 0.5
    confidence: float = 0.8  # how reliable is this memory
    
    # Associations
    associated_traces: list[str] = field(default_factory=list)  # other trace IDs
    
    # Decay parameters
    strength: float = 1.0
    decay_rate: float = 0.001  # per second

    def recall(self) -> None:
        """Access this memory (reinforces it)."""
        self.last_recalled = time.time()
        self.recall_count += 1
        self.strength = min(1.0, self.strength + 0.05)  # reinforcement

    def decay(self, dt: float) -> None:
        """Memory weakens over time without access."""
        time_since_recall = time.time() - self.last_recalled
        self.strength *= (1.0 - self.decay_rate * dt)
        self.strength = max(0.0, self.strength)

    def is_viable(self, min_strength: float = 0.05) -> bool:
        """Is this memory strong enough to be recalled?"""
        return self.strength >= min_strength

    def to_dict(self) -> dict:
        return {
            "id": self.trace_id,
            "type": self.trace_type.name,
            "description": self.description,
            "strength": round(self.strength, 3),
            "recall_count": self.recall_count,
            "importance": round(self.importance, 3),
            "age_hours": round((time.time() - self.formed_at) / 3600, 1),
        }
