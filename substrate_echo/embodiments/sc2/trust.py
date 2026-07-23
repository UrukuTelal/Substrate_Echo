"""Trust Evaluation Layer — Dynamic trust attractor for agent relationships.

Trust is not a binary flag but a dynamic attractor that evolves
through interaction history. Positive evidence strengthens trust,
negative evidence weakens it.

Architecture:
    Observation → Trust Update → Trust State → Communication Policy
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque


class TrustLevel(Enum):
    """Trust classification."""
    HOSTILE = 0      # Trust < 0.2
    DISTRUSTFUL = 1  # Trust 0.2-0.4
    CAUTIOUS = 2     # Trust 0.4-0.6
    TRUSTWORTHY = 3  # Trust 0.6-0.8
    ALLIED = 4       # Trust > 0.8


@dataclass
class InteractionRecord:
    """Single interaction that affects trust."""
    timestamp: int
    interaction_type: str  # "agreement", "deception", "attack", "share", "predict"
    outcome: str          # "positive", "negative", "neutral"
    trust_delta: float    # Change in trust
    evidence: str = ""    # Description


@dataclass
class AgentTrust:
    """Trust state for a specific agent."""
    agent_id: str
    trust_score: float = 0.5  # [0, 1]
    confidence: float = 0.1   # How certain we are
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: int = 0
    
    # Decay parameters
    decay_rate: float = 0.01  # Trust decays toward 0.5 without evidence
    recovery_rate: float = 0.005
    
    def update(self, evidence: str, delta: float, timestamp: int):
        """Update trust based on evidence."""
        # Apply delta
        self.trust_score = np.clip(self.trust_score + delta, 0.0, 1.0)
        
        # Update confidence (more interactions = more certain)
        self.confidence = min(1.0, self.confidence + 0.05)
        
        # Record
        self.history.append(InteractionRecord(
            timestamp=timestamp,
            interaction_type=evidence,
            outcome="positive" if delta > 0 else "negative" if delta < 0 else "neutral",
            trust_delta=delta,
            evidence=evidence,
        ))
        self.last_update = timestamp
    
    def decay(self, current_time: int):
        """Decay trust toward neutral (0.5) over time."""
        time_since = current_time - self.last_update
        if time_since > 0:
            # Decay toward 0.5
            decay = (self.trust_score - 0.5) * self.decay_rate * min(time_since, 100)
            self.trust_score -= decay
    
    def get_level(self) -> TrustLevel:
        """Get trust classification."""
        if self.trust_score < 0.2:
            return TrustLevel.HOSTILE
        elif self.trust_score < 0.4:
            return TrustLevel.DISTRUSTFUL
        elif self.trust_score < 0.6:
            return TrustLevel.CAUTIOUS
        elif self.trust_score < 0.8:
            return TrustLevel.TRUSTWORTHY
        else:
            return TrustLevel.ALLIED
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.trust_score,
            self.confidence,
            self.get_level().value / 4.0,  # Normalized level
            len(self.history) / 100.0,     # Interaction density
        ], dtype=np.float64)


class TrustEvaluationLayer:
    """Dynamic trust evaluation for multi-agent interactions.
    
    Maintains trust states for all known agents and provides
    trust-informed decisions for communication and cooperation.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentTrust] = {}
        self._global_trust: float = 0.5
        self._step: int = 0
    
    def register_agent(self, agent_id: str, initial_trust: float = 0.5):
        """Register a new agent with initial trust."""
        self._agents[agent_id] = AgentTrust(
            agent_id=agent_id,
            trust_score=initial_trust,
        )
    
    def observe(self, agent_id: str, observation: Dict) -> float:
        """Process observation and update trust.
        
        Returns the updated trust score.
        """
        if agent_id not in self._agents:
            self.register_agent(agent_id)
        
        agent = self._agents[agent_id]
        delta = self._compute_trust_delta(observation)
        agent.update(observation.get("type", "unknown"), delta, self._step)
        
        return agent.trust_score
    
    def _compute_trust_delta(self, observation: Dict) -> float:
        """Compute trust change from observation."""
        obs_type = observation.get("type", "")
        outcome = observation.get("outcome", "neutral")
        
        # Positive evidence
        if obs_type == "agreement_honored":
            return 0.05
        elif obs_type == "accurate_prediction":
            return 0.03
        elif obs_type == "predictable_behavior":
            return 0.02
        elif obs_type == "information_shared":
            return 0.04
        
        # Negative evidence
        elif obs_type == "agreement_broken":
            return -0.08
        elif obs_type == "deception_detected":
            return -0.10
        elif obs_type == "unprovoked_attack":
            return -0.12
        elif obs_type == "unpredictable_behavior":
            return -0.03
        
        return 0.0
    
    def get_trust(self, agent_id: str) -> float:
        """Get current trust score for agent."""
        if agent_id not in self._agents:
            return 0.5  # Unknown agents have neutral trust
        return self._agents[agent_id].trust_score
    
    def get_level(self, agent_id: str) -> TrustLevel:
        """Get trust level for agent."""
        if agent_id not in self._agents:
            return TrustLevel.CAUTIOUS
        return self._agents[agent_id].get_level()
    
    def should_share_info(self, agent_id: str, info_value: float) -> bool:
        """Determine if information should be shared with agent.
        
        info_value: 0.0 (worthless) to 1.0 (critical)
        """
        trust = self.get_trust(agent_id)
        threshold = 0.3 + (info_value * 0.4)  # Higher value = higher trust needed
        return trust >= threshold
    
    def should_cooperate(self, agent_id: str) -> bool:
        """Determine if cooperation should be attempted."""
        return self.get_trust(agent_id) >= 0.6
    
    def tick(self, current_time: int):
        """Update all trust states."""
        self._step = current_time
        for agent in self._agents.values():
            agent.decay(current_time)
    
    def get_all_trusts(self) -> Dict[str, float]:
        """Get all agent trust scores."""
        return {aid: a.trust_score for aid, a in self._agents.items()}
    
    def get_summary(self) -> Dict:
        """Get trust system summary."""
        if not self._agents:
            return {"agents": 0, "avg_trust": 0.5}
        
        trusts = [a.trust_score for a in self._agents.values()]
        return {
            "agents": len(self._agents),
            "avg_trust": float(np.mean(trusts)),
            "min_trust": float(np.min(trusts)),
            "max_trust": float(np.max(trusts)),
            "distrustful": sum(1 for t in trusts if t < 0.4),
            "trustworthy": sum(1 for t in trusts if t > 0.6),
        }
