"""Discovery Exchange Protocol — Swarm Currency.

Raw data is expensive.
Opinions are cheap.
Validated abstractions are valuable.

The exchange layer transforms individual learning into
collective intelligence:

    Raw Observation
           ↓
       Feature
           ↓
      Hypothesis
           ↓
      Prediction
           ↓
       Outcome
           ↓
    Validated Rule
           ↓
   Compressed Discovery
           ↓
      Swarm Culture

A discovery object is the currency of exchange:

    Discovery:
        Domain: social interaction
        Pattern: entities that observe without aggression are not hostile
        Confidence: 0.87
        Validated: 243 times
        Sources: SC2-001, Simulation-014, Robot-003
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time
import uuid


class ExchangeProtocol(Enum):
    """Types of discovery exchanges."""
    DIRECT = "direct"           # Agent to agent
    BROADCAST = "broadcast"     # Agent to all
    REQUEST = "request"         # Request specific knowledge
    RESPONSE = "response"       # Response to request


@dataclass
class ExchangeMessage:
    """A message in the discovery exchange."""
    message_id: str
    protocol: ExchangeProtocol
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    
    # Content
    discovery: Optional[Any] = None  # CompressedDiscovery
    request_domain: Optional[str] = None  # For REQUEST protocol
    request_type: Optional[str] = None
    
    # Metadata
    timestamp: float = 0.0
    priority: float = 0.5  # [0, 1] importance of this exchange
    ttl: float = 100.0  # Time to live before expiry
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def is_expired(self, current_time: float) -> bool:
        """Check if message has expired."""
        return (current_time - self.timestamp) > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id,
            "protocol": self.protocol.value,
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "priority": round(self.priority, 3),
            "age": round(time.time() - self.timestamp, 3),
        }


@dataclass
class ExchangeRate:
    """Trust-weighted exchange rate for discoveries."""
    domain: str
    sender_trust: float
    receiver_trust: float
    
    # Exchange quality metrics
    acceptance_rate: float = 0.5  # How often exchanges are accepted
    value_rating: float = 0.5     # How valuable the exchanges are
    
    def calculate_priority(self, discovery_confidence: float) -> float:
        """Calculate priority for this exchange."""
        # Trust-weighted confidence
        trust_factor = (self.sender_trust + self.receiver_trust) / 2
        
        # Value factor
        value_factor = self.value_rating
        
        # Combined priority
        priority = discovery_confidence * trust_factor * value_factor
        
        return min(1.0, priority)


class DiscoveryExchangeProtocol:
    """Protocol for exchanging discoveries between agents.
    
    The exchange layer transforms individual learning into
    collective intelligence through trust-gated discovery sharing.
    
    Architecture:
        Agent A
           |
           | compress discovery
           ↓
        Exchange Message
           |
           | trust-gated routing
           ↓
        Agent B
           |
           | ingest & integrate
           ↓
        Swarm Knowledge
    """
    
    def __init__(self):
        # Message queues
        self._outbox: Dict[str, List[ExchangeMessage]] = {}  # agent_id -> messages
        self._inbox: Dict[str, List[ExchangeMessage]] = {}
        
        # Exchange history
        self._exchange_history: List[ExchangeMessage] = []
        
        # Trust system reference (injected)
        self._trust_system = None
        
        # Swarm record reference (injected)
        self._swarm_record = None
        
        # Exchange rates per domain
        self._exchange_rates: Dict[Tuple[str, str, str], ExchangeRate] = {}
    
    def set_trust_system(self, trust_system):
        """Set the trust system for trust-gated exchanges."""
        self._trust_system = trust_system
    
    def set_swarm_record(self, swarm_record):
        """Set the swarm development record."""
        self._swarm_record = swarm_record
    
    def send_discovery(self, sender_id: str, discovery: Any,
                       receiver_id: Optional[str] = None,
                       priority: float = 0.5) -> ExchangeMessage:
        """Send a discovery to one or all agents.
        
        Args:
            sender_id: Agent sending the discovery
            discovery: The compressed discovery to share
            receiver_id: Specific agent (None for broadcast)
            priority: Exchange priority [0, 1]
        
        Returns:
            The exchange message created
        """
        message = ExchangeMessage(
            message_id=str(uuid.uuid4()),
            protocol=ExchangeProtocol.BROADCAST if receiver_id is None
                     else ExchangeProtocol.DIRECT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            discovery=discovery,
            priority=priority,
        )
        
        # Add to sender's outbox
        if sender_id not in self._outbox:
            self._outbox[sender_id] = []
        self._outbox[sender_id].append(message)
        
        return message
    
    def request_knowledge(self, requester_id: str, domain: str,
                          request_type: str = "pattern") -> ExchangeMessage:
        """Request knowledge about a domain.
        
        Args:
            requester_id: Agent requesting knowledge
            domain: Domain to request knowledge about
            request_type: Type of knowledge requested
        
        Returns:
            The request message
        """
        message = ExchangeMessage(
            message_id=str(uuid.uuid4()),
            protocol=ExchangeProtocol.REQUEST,
            sender_id=requester_id,
            receiver_id=None,  # Broadcast to all
            request_domain=domain,
            request_type=request_type,
        )
        
        # Add to requester's outbox
        if requester_id not in self._outbox:
            self._outbox[requester_id] = []
        self._outbox[requester_id].append(message)
        
        return message
    
    def process_outbox(self, agent_id: str) -> List[ExchangeMessage]:
        """Process outbox for an agent, routing messages.
        
        Returns:
            List of messages that were routed
        """
        if agent_id not in self._outbox:
            return []
        
        outbox = self._outbox[agent_id]
        routed = []
        
        for message in outbox:
            if message.is_expired(time.time()):
                continue  # Skip expired messages
            
            if message.protocol == ExchangeProtocol.BROADCAST:
                # Route to all other agents
                for target_id in list(self._inbox.keys()) + [agent_id]:
                    if target_id != agent_id:
                        self._route_message(message, target_id)
                routed.append(message)
            
            elif message.protocol == ExchangeProtocol.DIRECT:
                # Route to specific agent
                if message.receiver_id:
                    self._route_message(message, message.receiver_id)
                    routed.append(message)
            
            elif message.protocol == ExchangeProtocol.REQUEST:
                # Route request to all agents
                for target_id in list(self._inbox.keys()):
                    if target_id != agent_id:
                        self._route_message(message, target_id)
                routed.append(message)
        
        # Clear processed outbox
        self._outbox[agent_id] = []
        
        return routed
    
    def _route_message(self, message: ExchangeMessage, target_id: str):
        """Route a message to a target agent's inbox."""
        if target_id not in self._inbox:
            self._inbox[target_id] = []
        self._inbox[target_id].append(message)
    
    def receive_messages(self, agent_id: str) -> List[ExchangeMessage]:
        """Receive messages from inbox.
        
        Returns:
            List of messages received
        """
        if agent_id not in self._inbox:
            return []
        
        messages = self._inbox[agent_id]
        self._inbox[agent_id] = []
        
        # Filter out expired
        current_time = time.time()
        valid_messages = [m for m in messages if not m.is_expired(current_time)]
        
        # Sort by priority
        valid_messages.sort(key=lambda m: -m.priority)
        
        return valid_messages
    
    def should_accept_discovery(self, receiver_id: str, message: ExchangeMessage) -> bool:
        """Determine if a discovery should be accepted.
        
        Uses trust system for trust-gated acceptance.
        """
        if message.discovery is None:
            return False
        
        # Check trust if available
        if self._trust_system:
            trust = self._trust_system.get_trust(message.sender_id)
            if trust:
                # Trust-gated acceptance
                if trust.cooperation < 0.3:
                    return False  # Don't trust the sender
                
                # Domain-specific trust
                domain = getattr(message.discovery, 'pattern', {}).get('domain', 'general')
                domain_trust = self._trust_system.get_domain_trust(
                    message.sender_id, domain
                )
                if domain_trust < 0.4:
                    return False  # Don't trust their domain expertise
        
        # Check discovery confidence
        if hasattr(message.discovery, 'confidence'):
            if message.discovery.confidence < 0.6:
                return False  # Not confident enough
        
        return True
    
    def record_exchange(self, message: ExchangeMessage, accepted: bool):
        """Record an exchange event."""
        self._exchange_history.append(message)
        
        # Update exchange rates
        if message.discovery and hasattr(message.discovery, 'pattern'):
            domain = message.discovery.pattern.get('domain', 'general')
            key = (domain, message.sender_id, message.receiver_id or 'broadcast')
            
            if key not in self._exchange_rates:
                self._exchange_rates[key] = ExchangeRate(
                    domain=domain,
                    sender_trust=0.5,
                    receiver_trust=0.5,
                )
            
            rate = self._exchange_rates[key]
            # Update acceptance rate
            alpha = 0.1
            rate.acceptance_rate = (
                alpha * (1.0 if accepted else 0.0) +
                (1 - alpha) * rate.acceptance_rate
            )
    
    def get_exchange_statistics(self) -> Dict[str, Any]:
        """Get statistics about exchanges."""
        total = len(self._exchange_history)
        
        if total == 0:
            return {"total_exchanges": 0}
        
        by_protocol = {}
        for msg in self._exchange_history:
            proto = msg.protocol.value
            by_protocol[proto] = by_protocol.get(proto, 0) + 1
        
        return {
            "total_exchanges": total,
            "by_protocol": by_protocol,
            "active_outboxes": sum(len(q) for q in self._outbox.values()),
            "active_inboxes": sum(len(q) for q in self._inbox.values()),
            "exchange_rates": len(self._exchange_rates),
        }
    
    def get_domain_expertise_map(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get map of domain -> list of (agent_id, trust) sorted by trust."""
        if not self._trust_system:
            return {}
        
        # Collect domain expertise from all entities
        expertise_map: Dict[str, List[Tuple[str, float]]] = {}
        
        for entity_id, trust_vector in self._trust_system._entities.items():
            for domain, domain_trust in trust_vector.domain_trust.items():
                if domain not in expertise_map:
                    expertise_map[domain] = []
                expertise_map[domain].append((entity_id, domain_trust.confidence))
        
        # Sort by trust
        for domain in expertise_map:
            expertise_map[domain].sort(key=lambda x: -x[1])
        
        return expertise_map
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "outbox_sizes": {k: len(v) for k, v in self._outbox.items()},
            "inbox_sizes": {k: len(v) for k, v in self._inbox.items()},
            "exchange_history": len(self._exchange_history),
            "exchange_rates": len(self._exchange_rates),
        }