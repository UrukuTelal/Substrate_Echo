"""Proposal data structures for council submissions."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid


class ProposalType(Enum):
    ECONOMIC_ACTION = "economic_action"
    TECH_ACTION = "tech_action"
    MILITARY_ACTION = "military_action"
    RECON_ACTION = "recon_action"
    DEFENSIVE_ACTION = "defensive_action"
    INFRASTRUCTURE_ACTION = "infrastructure_action"
    STRATEGIC_INTENT = "strategic_intent"
    CAPABILITY_QUERY = "capability_query"


class Priority(Enum):
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


@dataclass
class Proposal:
    proposal_type: ProposalType
    source_council: str
    description: str
    utility: float
    confidence: float
    priority: Priority = Priority.NORMAL
    
    minerals_cost: int = 0
    vespene_cost: int = 0
    supply_cost: int = 0
    time_cost: float = 0.0
    
    preconditions: Dict[str, Any] = field(default_factory=dict)
    expected_effects: Dict[str, float] = field(default_factory=dict)
    
    target: Optional[str] = None
    target_position: Optional[tuple] = None
    
    ability_name: Optional[str] = None
    unit_tags: List[int] = field(default_factory=list)
    
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    tick: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "Proposal") -> bool:
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.utility > other.utility
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "type": self.proposal_type.value,
            "source": self.source_council,
            "description": self.description,
            "utility": self.utility,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "costs": {"minerals": self.minerals_cost, "vespene": self.vespene_cost, "supply": self.supply_cost},
            "time_cost": self.time_cost,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "target": self.target,
            "ability": self.ability_name,
            "tick": self.tick,
        }


@dataclass
class AllocationResult:
    accepted: List[Proposal]
    rejected: List[Proposal]
    deferred: List[Proposal]
    reasoning: Dict[str, str]
