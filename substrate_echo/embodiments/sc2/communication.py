"""Communication Policy Layer — Selective information sharing.

The substrate decides what to reveal, what to hide, and what
information has strategic value. Like biology, communication
is selective.

Architecture:
    Internal State → Trust Layer → Communication Policy → External Message
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class InfoCategory(Enum):
    """Types of information."""
    POSITION = "position"
    ECONOMY = "economy"
    ARMY = "army"
    INTENT = "intent"
    CAPABILITY = "capability"
    WEAKNESS = "weakness"
    ALLIANCE = "alliance"
    DECEPTION = "deception"
    HUMOR = "humor"
    UNKNOWN = "unknown"


@dataclass
class CommunicationPolicy:
    """Policy for what information to share."""
    category: InfoCategory
    min_trust: float  # Minimum trust to share
    value_weight: float  # Strategic value multiplier
    can_deceive: bool = False  # Can we send false version?
    deception_cost: float = 0.0  # Trust cost if caught


@dataclass
class Message:
    """An outgoing or incoming message."""
    sender: str
    receiver: str
    category: InfoCategory
    content: Any
    is_deception: bool = False
    trust_impact: float = 0.0


class CommunicationPolicyLayer:
    """Controls what information is shared with other agents.
    
    The substrate decides:
    - What to reveal
    - What to hide
    - What information has strategic value
    """
    
    # Default policies for each category
    DEFAULT_POLICIES = {
        InfoCategory.POSITION: CommunicationPolicy(
            category=InfoCategory.POSITION,
            min_trust=0.3,
            value_weight=1.0,
            can_deceive=True,
            deception_cost=0.05,
        ),
        InfoCategory.ECONOMY: CommunicationPolicy(
            category=InfoCategory.ECONOMY,
            min_trust=0.5,
            value_weight=1.5,
            can_deceive=True,
            deception_cost=0.08,
        ),
        InfoCategory.ARMY: CommunicationPolicy(
            category=InfoCategory.ARMY,
            min_trust=0.7,
            value_weight=2.0,
            can_deceive=True,
            deception_cost=0.12,
        ),
        InfoCategory.INTENT: CommunicationPolicy(
            category=InfoCategory.INTENT,
            min_trust=0.8,
            value_weight=2.5,
            can_deceive=True,
            deception_cost=0.15,
        ),
        InfoCategory.CAPABILITY: CommunicationPolicy(
            category=InfoCategory.CAPABILITY,
            min_trust=0.6,
            value_weight=1.8,
            can_deceive=True,
            deception_cost=0.10,
        ),
        InfoCategory.WEAKNESS: CommunicationPolicy(
            category=InfoCategory.WEAKNESS,
            min_trust=0.9,
            value_weight=3.0,
            can_deceive=False,
            deception_cost=0.0,
        ),
        InfoCategory.ALLIANCE: CommunicationPolicy(
            category=InfoCategory.ALLIANCE,
            min_trust=0.85,
            value_weight=2.8,
            can_deceive=False,
            deception_cost=0.0,
        ),
        InfoCategory.DECEPTION: CommunicationPolicy(
            category=InfoCategory.DECEPTION,
            min_trust=0.0,  # Can always deceive
            value_weight=1.0,
            can_deceive=True,
            deception_cost=0.20,
        ),
        InfoCategory.HUMOR: CommunicationPolicy(
            category=InfoCategory.HUMOR,
            min_trust=0.3,
            value_weight=0.5,
            can_deceive=False,
            deception_cost=0.0,
        ),
    }
    
    def __init__(self, trust_layer=None):
        """Initialize with trust evaluation layer."""
        self._trust_layer = trust_layer
        self._policies = dict(self.DEFAULT_POLICIES)
        self._outgoing: List[Message] = []
        self._incoming: List[Message] = []
        self._deception_history: List[Message] = []
    
    def should_send(self, receiver_id: str, category: InfoCategory,
                    info_value: float = 0.5) -> bool:
        """Determine if information should be sent to receiver."""
        policy = self._policies.get(category, self.DEFAULT_POLICIES[InfoCategory.UNKNOWN])
        
        # Get trust level
        trust = 0.5
        if self._trust_layer:
            trust = self._trust_layer.get_trust(receiver_id)
        
        # Check minimum trust
        if trust < policy.min_trust:
            return False
        
        # Check strategic value
        adjusted_threshold = policy.min_trust - (info_value * 0.1)
        return trust >= adjusted_threshold
    
    def prepare_message(self, sender: str, receiver: str,
                       category: InfoCategory, content: Any,
                       truth: Any = None) -> Message:
        """Prepare a message, possibly with deception.
        
        If truth is provided and deception is allowed,
        may send a false version based on trust level.
        """
        policy = self._policies.get(category, self.DEFAULT_POLICIES[InfoCategory.UNKNOWN])
        
        # Decide whether to deceive
        is_deception = False
        trust = 0.5
        if self._trust_layer:
            trust = self._trust_layer.get_trust(receiver)
        
        if policy.can_deceive and truth is not None:
            # Higher trust = less likely to deceive (relationship preservation)
            deceive_chance = max(0.0, 1.0 - trust) * 0.3
            if np.random.random() < deceive_chance:
                is_deception = True
                content = self._create_deception(content, truth, category)
        
        msg = Message(
            sender=sender,
            receiver=receiver,
            category=category,
            content=content,
            is_deception=is_deception,
            trust_impact=-policy.deception_cost if is_deception else 0.0,
        )
        
        self._outgoing.append(msg)
        if is_deception:
            self._deception_history.append(msg)
        
        return msg
    
    def _create_deception(self, cover: Any, truth: Any,
                         category: InfoCategory) -> Any:
        """Create a deceptive version of information."""
        if category == InfoCategory.POSITION:
            # Offset position slightly
            if isinstance(cover, (list, tuple)) and len(cover) >= 2:
                return (cover[0] + np.random.uniform(-5, 5),
                       cover[1] + np.random.uniform(-5, 5))
        elif category == InfoCategory.ECONOMY:
            # Under-report resources
            if isinstance(cover, dict):
                return {k: v * 0.8 for k, v in cover.items()}
        elif category == InfoCategory.ARMY:
            # Under-report army size
            if isinstance(cover, dict):
                return {k: int(v * 0.7) for k, v in cover.items()}
        
        return cover
    
    def receive_message(self, message: Message) -> Dict:
        """Process incoming message and assess credibility."""
        self._incoming.append(message)
        
        # Get trust level
        trust = 0.5
        if self._trust_layer:
            trust = self._trust_layer.get_trust(message.sender)
        
        # Assess likelihood of deception
        deception_risk = max(0.0, 1.0 - trust) * 0.5
        
        return {
            "message": message,
            "trust_level": trust,
            "deception_risk": deception_risk,
            "should_act": trust >= 0.5,
            "confidence": trust * (1.0 - deception_risk),
        }
    
    def set_policy(self, category: InfoCategory, policy: CommunicationPolicy):
        """Update policy for a category."""
        self._policies[category] = policy
    
    def get_status(self) -> Dict:
        """Get communication status."""
        return {
            "outgoing": len(self._outgoing),
            "incoming": len(self._incoming),
            "deceptions": len(self._deception_history),
            "policies": len(self._policies),
        }
