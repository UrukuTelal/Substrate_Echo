"""SC2 Council Expansion — New roles for multi-agent interaction.

Adds specialized roles for:
- Diplomacy decisions
- Trust analysis
- Negotiation
- Opponent modeling
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class SC2CouncilRole(Enum):
    """SC2-specific council roles."""
    DIPLOMAT = "diplomat"
    TRUST_ANALYST = "trust_analyst"
    NEGOTIATOR = "negotiator"
    ADVERSARY_MODEL = "adversary_model"


@dataclass
class CouncilProposal:
    """A proposal for council consideration."""
    proposer: str
    role: SC2CouncilRole
    action: str
    reasoning: str
    confidence: float
    expected_outcome: str
    risks: List[str] = field(default_factory=list)


class Diplomat:
    """Should cooperation be attempted?"""
    
    def evaluate(self, proposal: CouncilProposal,
                trust_level: float, game_context: Dict) -> Dict:
        """Evaluate diplomatic proposal."""
        # Higher trust = more likely to approve cooperation
        approval_threshold = 0.5 - (trust_level * 0.3)
        
        should_approve = proposal.confidence > approval_threshold
        
        return {
            "approved": should_approve,
            "reasoning": f"Trust {trust_level:.2f}, confidence {proposal.confidence:.2f}",
            "adjustments": [],
        }


class TrustAnalyst:
    """How reliable is this entity?"""
    
    def evaluate(self, agent_id: str, trust_history: List[float],
                recent_behavior: List[str]) -> Dict:
        """Analyze trust for an agent."""
        if not trust_history:
            return {
                "reliability": 0.5,
                "trend": "unknown",
                "recommendation": "observe",
            }
        
        # Calculate trend
        recent = trust_history[-10:] if len(trust_history) >= 10 else trust_history
        trend = "stable"
        if len(recent) >= 2:
            if recent[-1] > recent[0] + 0.1:
                trend = "improving"
            elif recent[-1] < recent[0] - 0.1:
                trend = "declining"
        
        # Analyze behavior patterns
        deception_count = sum(1 for b in recent_behavior if "deception" in b)
        cooperation_count = sum(1 for b in recent_behavior if "cooperation" in b)
        
        reliability = trust_history[-1] if trust_history else 0.5
        reliability *= (1.0 - deception_count * 0.1)
        reliability *= (1.0 + cooperation_count * 0.05)
        
        return {
            "reliability": max(0.0, min(1.0, reliability)),
            "trend": trend,
            "recommendation": "engage" if reliability > 0.6 else "observe" if reliability > 0.3 else "avoid",
        }


class Negotiator:
    """What information should be exchanged?"""
    
    def evaluate(self, proposal: CouncilProposal,
                trust_level: float, info_value: float) -> Dict:
        """Evaluate negotiation proposal."""
        # Calculate exchange value
        exchange_value = info_value * trust_level
        
        # Determine terms
        if trust_level > 0.8:
            terms = "full_disclosure"
        elif trust_level > 0.6:
            terms = "partial_exchange"
        elif trust_level > 0.4:
            terms = "limited_info"
        else:
            terms = "minimal_contact"
        
        return {
            "proceed": exchange_value > 0.3,
            "terms": terms,
            "exchange_value": exchange_value,
            "counter_proposal": None,
        }


class AdversaryModel:
    """What does the other system want?"""
    
    def __init__(self):
        self._agent_goals: Dict[str, List[str]] = {}
        self._agent_patterns: Dict[str, Dict] = {}
    
    def observe(self, agent_id: str, behavior: Dict):
        """Observe agent behavior to model their goals."""
        if agent_id not in self._agent_goals:
            self._agent_goals[agent_id] = []
            self._agent_patterns[agent_id] = {
                "aggression": 0.5,
                "economy_focus": 0.5,
                "expansion_rate": 0.5,
            }
        
        # Update patterns based on behavior
        if "attack_count" in behavior:
            self._agent_patterns[agent_id]["aggression"] = min(1.0,
                self._agent_patterns[agent_id]["aggression"] + behavior["attack_count"] * 0.1)
        
        if "worker_count" in behavior:
            self._agent_patterns[agent_id]["economy_focus"] = min(1.0,
                behavior["worker_count"] / 50.0)
        
        if "base_count" in behavior:
            self._agent_patterns[agent_id]["expansion_rate"] = min(1.0,
                behavior["base_count"] / 5.0)
    
    def predict_goal(self, agent_id: str) -> str:
        """Predict agent's primary goal."""
        if agent_id not in self._agent_patterns:
            return "unknown"
        
        patterns = self._agent_patterns[agent_id]
        
        # Simple goal inference
        if patterns["aggression"] > 0.7:
            return "aggressive_elimination"
        elif patterns["economy_focus"] > 0.7:
            return "economic_victory"
        elif patterns["expansion_rate"] > 0.7:
            return "map_control"
        else:
            return "balanced_play"
    
    def get_assessment(self, agent_id: str) -> Dict:
        """Get full assessment of agent."""
        if agent_id not in self._agent_patterns:
            return {"known": False}
        
        return {
            "known": True,
            "predicted_goal": self.predict_goal(agent_id),
            "patterns": self._agent_patterns[agent_id],
            "goal_history": self._agent_goals.get(agent_id, []),
        }


class SC2Council:
    """Expanded council for SC2 multi-agent interactions.
    
    Combines existing council with new SC2-specific roles.
    """
    
    def __init__(self):
        self.diplomat = Diplomat()
        self.trust_analyst = TrustAnalyst()
        self.negotiator = Negotiator()
        self.adversary_model = AdversaryModel()
        
        self._proposals: List[CouncilProposal] = []
        self._decisions: List[Dict] = []
    
    def evaluate_diplomacy(self, proposal: CouncilProposal,
                          trust_level: float, game_context: Dict) -> Dict:
        """Evaluate a diplomatic proposal."""
        self._proposals.append(proposal)
        
        decision = self.diplomat.evaluate(proposal, trust_level, game_context)
        self._decisions.append({
            "type": "diplomacy",
            "proposal": proposal,
            "decision": decision,
        })
        
        return decision
    
    def assess_trust(self, agent_id: str, trust_history: List[float],
                    behavior: List[str]) -> Dict:
        """Assess trust for an agent."""
        return self.trust_analyst.evaluate(agent_id, trust_history, behavior)
    
    def negotiate(self, proposal: CouncilProposal,
                 trust_level: float, info_value: float) -> Dict:
        """Negotiate information exchange."""
        return self.negotiator.evaluate(proposal, trust_level, info_value)
    
    def model_adversary(self, agent_id: str, behavior: Dict) -> Dict:
        """Model adversary goals and patterns."""
        self.adversary_model.observe(agent_id, behavior)
        return self.adversary_model.get_assessment(agent_id)
    
    def get_status(self) -> Dict:
        """Get council status."""
        return {
            "proposals": len(self._proposals),
            "decisions": len(self._decisions),
            "roles": [r.value for r in SC2CouncilRole],
        }
