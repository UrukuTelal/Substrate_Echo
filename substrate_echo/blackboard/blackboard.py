"""Blackboard — Shared communication layer for all councils."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import defaultdict
import time
import threading

from substrate_echo.blackboard.evidence import Belief, Evidence, EvidenceType, CapabilityModel, OpponentBelief


class BlackboardChannel(Enum):
    """Channels for different types of information."""
    WORLD_STATE = "world_state"
    ECONOMY = "economy"
    INTELLIGENCE = "intelligence"
    COUNTER_INTEL = "counter_intel"
    MILITARY_INDUSTRIAL = "military_industrial"
    TECHNOLOGY = "technology"
    STRATEGY = "strategy"
    BUILD_PRIORITY = "build_priority"
    CAPABILITY = "capability"
    LOGISTICS = "logistics"
    GOVERNANCE = "governance"


@dataclass
class BlackboardEntry:
    """A single entry on the blackboard."""
    channel: BlackboardChannel
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    confidence: float = 1.0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


@dataclass
class CouncilProposal:
    """A proposal from a council for consideration."""
    council_name: str
    proposal_type: str
    content: Dict[str, Any]
    priority: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    def __lt__(self, other: "CouncilProposal"):
        return self.priority > other.priority

class Blackboard:
    """
    Thread-safe shared blackboard for inter-council communication.
    """
    
    def __init__(self):
        self._entries: Dict[str, BlackboardEntry] = {}
        self._channels: Dict[BlackboardChannel, Dict[str, BlackboardEntry]] = defaultdict(dict)
        self._beliefs: Dict[str, Belief] = {}
        self._capability_models: Dict[str, CapabilityModel] = {}
        self._opponent_beliefs: Dict[int, Dict[str, OpponentBelief]] = defaultdict(dict)
        self._proposals: List[CouncilProposal] = []
        self._accepted_proposals: List[CouncilProposal] = []
        self._rejected_proposals: List[CouncilProposal] = []
        
        self._subscribers: Dict[BlackboardChannel, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        
        self._history: List[Dict] = []
        self._max_history = 10000
    
    def write(self, channel: BlackboardChannel, key: str, value: Any, 
              source: str = "", confidence: float = 1.0, ttl: Optional[float] = None) -> None:
        """Write an entry to the blackboard."""
        with self._lock:
            entry = BlackboardEntry(
                channel=channel,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                ttl=ttl
            )
            entry_id = f"{channel.value}.{key}"
            self._entries[entry_id] = entry
            self._channels[channel][key] = entry
            self._notify_subscribers(channel, key, value)
            self._record_history("write", channel, key, value, source)
    
    def read(self, channel: BlackboardChannel, key: str, default: Any = None) -> Any:
        """Read an entry from the blackboard."""
        with self._lock:
            entry = self._channels[channel].get(key)
            if entry is None or entry.is_expired():
                return default
            return entry.value
    
    def read_entry(self, channel: BlackboardChannel, key: str) -> Optional[BlackboardEntry]:
        """Read full entry with metadata."""
        with self._lock:
            entry = self._channels[channel].get(key)
            if entry and entry.is_expired():
                return None
            return entry
    
    def read_channel(self, channel: BlackboardChannel) -> Dict[str, Any]:
        """Read all non-expired entries from a channel."""
        with self._lock:
            self._cleanup_expired(channel)
            return {k: v.value for k, v in self._channels[channel].items()}
    
    def delete(self, channel: BlackboardChannel, key: str) -> bool:
        """Delete an entry."""
        with self._lock:
            entry_id = f"{channel.value}.{key}"
            if entry_id in self._entries:
                del self._entries[entry_id]
                if key in self._channels[channel]:
                    del self._channels[channel][key]
                return True
            return False
    
    def _cleanup_expired(self, channel: BlackboardChannel) -> None:
        expired = [k for k, v in self._channels[channel].items() if v.is_expired()]
        for k in expired:
            entry_id = f"{channel.value}.{k}"
            if entry_id in self._entries:
                del self._entries[entry_id]
            del self._channels[channel][k]

    # ===== Belief Management =====
    
    def add_belief(self, belief: Belief) -> None:
        """Add or update a belief."""
        with self._lock:
            key = f"{belief.evidence_type.value}.{belief.subject}"
            self._beliefs[key] = belief
            self.write(BlackboardChannel.INTELLIGENCE, f"belief.{key}", belief, source="belief_system")
    
    def get_belief(self, evidence_type: EvidenceType, subject: str) -> Optional[Belief]:
        """Get a belief by type and subject."""
        with self._lock:
            key = f"{evidence_type.value}.{subject}"
            return self._beliefs.get(key)
    
    def get_beliefs(self, evidence_type: Optional[EvidenceType] = None) -> List[Belief]:
        """Get all beliefs, optionally filtered by type."""
        with self._lock:
            if evidence_type:
                return [b for b in self._beliefs.values() if b.evidence_type == evidence_type]
            return list(self._beliefs.values())
    
    # ===== Capability Model Management =====
    
    def add_capability_model(self, model: CapabilityModel) -> None:
        """Add or update a capability model."""
        with self._lock:
            self._capability_models[model.entity_type] = model
            self.write(BlackboardChannel.CAPABILITY, f"model.{model.entity_type}", model, source="capability_council")
    
    def get_capability_model(self, entity_type: str) -> Optional[CapabilityModel]:
        """Get capability model for entity type."""
        with self._lock:
            return self._capability_models.get(entity_type)
    
    def get_all_capability_models(self) -> Dict[str, CapabilityModel]:
        """Get all capability models."""
        with self._lock:
            return dict(self._capability_models)
    
    # ===== Opponent Belief Management =====
    
    def add_opponent_belief(self, opponent_id: int, belief: OpponentBelief) -> None:
        """Add or update a belief about an opponent."""
        with self._lock:
            key = f"{belief.belief_type}.{belief.subject}"
            self._opponent_beliefs[opponent_id][key] = belief
            self.write(BlackboardChannel.COUNTER_INTEL, 
                       f"opponent.{opponent_id}.{key}", belief, source="counter_intel_council")
    
    def get_opponent_beliefs(self, opponent_id: int, belief_type: Optional[str] = None) -> List[OpponentBelief]:
        """Get beliefs about an opponent."""
        with self._lock:
            beliefs = self._opponent_beliefs.get(opponent_id, {})
            if belief_type:
                return [b for b in beliefs.values() if b.belief_type == belief_type]
            return list(beliefs.values())

    # ===== Proposal Management =====
    
    def submit_proposal(self, proposal: CouncilProposal) -> None:
        """Submit a proposal for consideration."""
        with self._lock:
            self._proposals.append(proposal)
            self._proposals.sort()
            self.write(BlackboardChannel.GOVERNANCE, f"proposal.{len(self._proposals)}", proposal, 
                       source=proposal.council_name)
    
    def get_proposals(self, min_priority: float = 0.0) -> List[CouncilProposal]:
        """Get all pending proposals above minimum priority."""
        with self._lock:
            return [p for p in self._proposals if p.priority >= min_priority]
    
    def get_proposals_by_type(self, proposal_type: str) -> List[CouncilProposal]:
        """Get proposals of a specific type."""
        with self._lock:
            return [p for p in self._proposals if p.proposal_type == proposal_type]
    
    def accept_proposal(self, proposal: CouncilProposal) -> None:
        """Mark a proposal as accepted."""
        with self._lock:
            if proposal in self._proposals:
                self._proposals.remove(proposal)
            self._accepted_proposals.append(proposal)
            self.write(BlackboardChannel.GOVERNANCE, f"accepted.{proposal.council_name}", proposal, 
                       source="governance")
    
    def reject_proposal(self, proposal: CouncilProposal, reason: str = "") -> None:
        """Mark a proposal as rejected."""
        with self._lock:
            if proposal in self._proposals:
                self._proposals.remove(proposal)
            proposal.content["rejection_reason"] = reason
            self._rejected_proposals.append(proposal)
    
    def clear_proposals(self) -> None:
        """Clear all pending proposals."""
        with self._lock:
            self._proposals.clear()
    
    # ===== Subscriptions =====
    
    def subscribe(self, channel: BlackboardChannel, callback: Callable) -> None:
        """Subscribe to channel updates."""
        with self._lock:
            self._subscribers[channel].append(callback)
    
    def unsubscribe(self, channel: BlackboardChannel, callback: Callable) -> None:
        """Unsubscribe from channel updates."""
        with self._lock:
            if callback in self._subscribers[channel]:
                self._subscribers[channel].remove(callback)
    
    def _notify_subscribers(self, channel: BlackboardChannel, key: str, value: Any) -> None:
        """Notify all subscribers of a channel update."""
        for callback in self._subscribers[channel]:
            try:
                callback(channel, key, value)
            except Exception:
                pass
    
    # ===== History / Debugging =====
    
    def _record_history(self, action: str, channel: BlackboardChannel, key: str, value: Any, source: str) -> None:
        """Record action to history."""
        self._history.append({
            "action": action,
            "channel": channel.value,
            "key": key,
            "value_type": type(value).__name__,
            "source": source,
            "timestamp": time.time()
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent history."""
        with self._lock:
            return self._history[-limit:]
    
    # ===== Utility =====
    
    def snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of the blackboard state."""
        with self._lock:
            return {
                "entries": {k: {"channel": v.channel.value, "key": v.key, "value": v.value, 
                               "source": v.source, "confidence": v.confidence, "timestamp": v.timestamp}
                           for k, v in self._entries.items() if not v.is_expired()},
                "beliefs": {k: v.__dict__ for k, v in self._beliefs.items()},
                "capability_models": {k: v.__dict__ for k, v in self._capability_models.items()},
                "opponent_beliefs": {oid: {k: v.__dict__ for k, v in b.items()} 
                                    for oid, b in self._opponent_beliefs.items()},
                "pending_proposals": len(self._proposals),
                "accepted_proposals": len(self._accepted_proposals),
                "rejected_proposals": len(self._rejected_proposals),
            }
    
    def clear_expired(self) -> int:
        """Clear all expired entries. Returns count cleared."""
        with self._lock:
            count = 0
            for channel in BlackboardChannel:
                before = len(self._channels[channel])
                self._cleanup_expired(channel)
                count += before - len(self._channels[channel])
            return count
