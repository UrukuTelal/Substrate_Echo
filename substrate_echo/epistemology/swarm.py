"""Swarm Epistemology — Distributed learning across agents.

The swarm doesn't vote on opinions. It compares:
- which model predicted better
- which evidence each model used
- which assumptions failed
- which interpretations generalize

Architecture:
    Agent Development Records (individual)
            |
            | compressed discoveries
            v
    Swarm Development Record (collective)
            |
            | identity patterns
            v
    Agent Development Records (influenced by swarm)

Key concept: Compressed discoveries are the currency of exchange.
Not raw observations, not opinions, but validated lessons learned.

Swarm Development Record Structure:
    ├── Foundational discoveries
    ├── Domain knowledge
    ├── Failed assumptions
    ├── Current uncertainties
    ├── Cultural norms
    ├── Adaptation history
    └── Open questions

A mature swarm knows not only "What do we know?" but also
"What do we not know that matters?"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np
import time


class DiscoveryType(Enum):
    """Types of discoveries that can be shared."""
    PATTERN = "pattern"           # Recurring observation pattern
    RULE = "rule"                 # Validated behavioral rule
    PREDICTION_STRATEGY = "prediction_strategy"  # What works for prediction
    ANTI_PATTERN = "anti_pattern"  # What doesn't work
    CALIBRATION = "calibration"   # Confidence adjustment insight
    FOUNDATIONAL = "foundational"  # Core discovery that shapes understanding


class OpenQuestionType(Enum):
    """Types of open questions the swarm tracks."""
    PREDICTION = "prediction"     # What will happen next?
    CAUSAL = "causal"             # Why did this happen?
    STRATEGIC = "strategic"       # What should we do?
    UNKNOWN = "unknown"           # What don't we know?


@dataclass
class OpenQuestion:
    """A question the swarm knows it should answer but can't yet."""
    question_id: str
    question_type: OpenQuestionType
    description: str
    
    # Why this question matters
    importance: float = 0.5       # [0, 1] how much this matters
    urgency: float = 0.5         # [0, 1] how soon we need an answer
    
    # Attempts to answer
    attempts: int = 0
    best_confidence: float = 0.0
    best_answer: Optional[str] = None
    
    # Metadata
    created_at: float = 0.0
    last_attempted: float = 0.0
    sources: List[str] = field(default_factory=list)  # Which agents raised this
    
    def record_attempt(self, confidence: float, answer: Optional[str],
                       source: str, timestamp: float):
        """Record an attempt to answer this question."""
        self.attempts += 1
        self.last_attempted = timestamp
        self.sources.append(source)
        
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_answer = answer
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.question_id,
            "type": self.question_type.value,
            "description": self.description,
            "importance": round(self.importance, 3),
            "urgency": round(self.urgency, 3),
            "attempts": self.attempts,
            "best_confidence": round(self.best_confidence, 3),
        }


@dataclass
class CompressedDiscovery:
    """A validated lesson learned, compressed for sharing.
    
    This is the currency of swarm epistemology.
    """
    discovery_id: str
    discovery_type: DiscoveryType
    description: str
    
    # The lesson
    pattern: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    confidence: float = 0.5
    evidence_count: int = 0
    validation_source: str = ""  # Which agent validated this
    
    # Metadata
    discovered_at: float = 0.0
    discovered_by: str = ""
    generalized: bool = False  # Has this been tested across domains?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.discovery_id,
            "type": self.discovery_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence_count,
            "source": self.discovered_by,
            "generalized": self.generalized,
        }


@dataclass
class AgentEpistemicState:
    """Epistemic state of an individual agent in the swarm."""
    agent_id: str
    
    # Individual development record
    observations: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[CompressedDiscovery] = field(default_factory=list)
    
    # Trust relationships with other agents
    trust_in_others: Dict[str, float] = field(default_factory=dict)
    
    # Learning metrics
    prediction_accuracy: float = 0.5
    discovery_count: int = 0
    contributions_to_swarm: int = 0
    
    def compress_discovery(self, discovery: CompressedDiscovery) -> CompressedDiscovery:
        """Compress a discovery for sharing with the swarm.
        
        This is where raw learning becomes transferable knowledge.
        """
        # Validate before sharing
        if discovery.confidence < 0.6:
            return None  # Not confident enough to share
        
        # Compress pattern
        compressed = CompressedDiscovery(
            discovery_id=discovery.discovery_id,
            discovery_type=discovery.discovery_type,
            description=discovery.description,
            pattern=self._extract_core_pattern(discovery.pattern),
            conditions=discovery.conditions,
            confidence=discovery.confidence,
            evidence_count=discovery.evidence_count,
            discovered_at=time.time(),
            discovered_by=self.agent_id,
        )
        
        return compressed
    
    def _extract_core_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Extract core pattern from detailed pattern."""
        # Simplify pattern to essential features
        core = {}
        for key, value in pattern.items():
            if isinstance(value, (int, float)):
                core[key] = round(value, 3)
            elif isinstance(value, bool):
                core[key] = value
        return core
    
    def ingest_discovery(self, discovery: CompressedDiscovery,
                         source_trust: float) -> bool:
        """Ingest a discovery from another agent.
        
        Returns True if discovery was accepted.
        """
        # Trust-gated ingestion
        if source_trust < 0.3:
            return False  # Don't trust the source
        
        # Confidence-gated ingestion
        effective_confidence = discovery.confidence * source_trust
        if effective_confidence < 0.4:
            return False  # Not confident enough
        
        # Accept discovery
        self.discoveries.append(discovery)
        self.discovery_count += 1
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "observations": len(self.observations),
            "hypotheses": len(self.hypotheses),
            "predictions": len(self.predictions),
            "discoveries": len(self.discoveries),
            "prediction_accuracy": self.prediction_accuracy,
            "contributions": self.contributions_to_swarm,
        }


class SwarmDevelopmentRecord:
    """Collective development record of the swarm.
    
    Synthesizes patterns from individual agent records
    to create shared "culture" - the knowledge that shapes
    future agent development.
    
    Structure:
        ├── Foundational discoveries
        ├── Domain knowledge
        ├── Failed assumptions
        ├── Current uncertainties
        ├── Cultural norms
        ├── Adaptation history
        └── Open questions
    """
    
    def __init__(self):
        self._discoveries: List[CompressedDiscovery] = []
        self._agent_states: Dict[str, AgentEpistemicState] = {}
        self._consensus_patterns: Dict[str, Any] = {}
        self._swarm_accuracy: float = 0.5
        
        # Structured memory sections
        self._foundational_discoveries: List[CompressedDiscovery] = []
        self._domain_knowledge: Dict[str, List[CompressedDiscovery]] = {}
        self._failed_assumptions: List[Dict[str, Any]] = []
        self._current_uncertainties: List[str] = []
        self._cultural_norms: Dict[str, Any] = {}
        self._adaptation_history: List[Dict[str, Any]] = []
        self._open_questions: List[OpenQuestion] = []
    
    def register_agent(self, agent_id: str):
        """Register an agent with the swarm."""
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = AgentEpistemicState(agent_id=agent_id)
    
    def submit_discovery(self, agent_id: str,
                         discovery: CompressedDiscovery) -> bool:
        """Submit a discovery from an agent to the swarm.
        
        Returns True if discovery was accepted.
        """
        if agent_id not in self._agent_states:
            self.register_agent(agent_id)
        
        agent = self._agent_states[agent_id]
        
        # Compress for sharing
        compressed = agent.compress_discovery(discovery)
        if compressed is None:
            return False
        
        # Check for redundancy
        if self._is_redundant(compressed):
            return False
        
        # Accept discovery
        self._discoveries.append(compressed)
        agent.contributions_to_swarm += 1
        
        # Categorize discovery
        self._categorize_discovery(compressed)
        
        # Update consensus patterns
        self._update_consensus(compressed)
        
        return True
    
    def _categorize_discovery(self, discovery: CompressedDiscovery):
        """Categorize discovery into appropriate memory section."""
        # Foundational discoveries (high confidence, high evidence)
        if discovery.confidence >= 0.9 and discovery.evidence_count >= 10:
            self._foundational_discoveries.append(discovery)
        
        # Domain knowledge
        domain = discovery.pattern.get("domain", "general")
        if domain not in self._domain_knowledge:
            self._domain_knowledge[domain] = []
        self._domain_knowledge[domain].append(discovery)
    
    def record_failed_assumption(self, assumption: str, evidence: str,
                                 agent_id: str):
        """Record a failed assumption."""
        self._failed_assumptions.append({
            "assumption": assumption,
            "evidence": evidence,
            "reported_by": agent_id,
            "timestamp": time.time(),
        })
    
    def add_open_question(self, question: OpenQuestion):
        """Add an open question to the swarm's memory."""
        # Check if similar question exists
        for existing in self._open_questions:
            if (existing.question_type == question.question_type and
                self._questions_similar(existing, question)):
                # Update existing question
                existing.importance = max(existing.importance, question.importance)
                existing.urgency = max(existing.urgency, question.urgency)
                return
        
        self._open_questions.append(question)
    
    def _questions_similar(self, q1: OpenQuestion, q2: OpenQuestion) -> bool:
        """Check if two questions are similar."""
        # Simple similarity based on description overlap
        words1 = set(q1.description.lower().split())
        words2 = set(q2.description.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5
    
    def update_cultural_norm(self, key: str, value: Any, confidence: float):
        """Update a cultural norm."""
        self._cultural_norms[key] = {
            "value": value,
            "confidence": confidence,
            "updated_at": time.time(),
        }
    
    def record_adaptation(self, description: str, agent_id: str,
                          success: bool):
        """Record an adaptation event."""
        self._adaptation_history.append({
            "description": description,
            "agent_id": agent_id,
            "success": success,
            "timestamp": time.time(),
        })
    
    def _is_redundant(self, discovery: CompressedDiscovery) -> bool:
        """Check if discovery is redundant with existing ones."""
        for existing in self._discoveries:
            if (existing.discovery_type == discovery.discovery_type and
                self._patterns_similar(existing.pattern, discovery.pattern)):
                return True
        return False
    
    def _patterns_similar(self, p1: Dict[str, Any],
                          p2: Dict[str, Any]) -> bool:
        """Check if two patterns are similar."""
        if not p1 or not p2:
            return False
        
        common_keys = set(p1.keys()) & set(p2.keys())
        if not common_keys:
            return False
        
        similarities = []
        for key in common_keys:
            v1, v2 = p1[key], p2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(v1 - v2) / max(abs(v1), 1e-6) < 0.2:
                    similarities.append(True)
                else:
                    similarities.append(False)
            elif v1 == v2:
                similarities.append(True)
            else:
                similarities.append(False)
        
        return sum(similarities) / len(similarities) > 0.7
    
    def _update_consensus(self, discovery: CompressedDiscovery):
        """Update consensus patterns based on new discovery."""
        key = discovery.discovery_type.value
        
        if key not in self._consensus_patterns:
            self._consensus_patterns[key] = {
                "count": 0,
                "avg_confidence": 0.0,
                "patterns": [],
            }
        
        consensus = self._consensus_patterns[key]
        consensus["count"] += 1
        consensus["avg_confidence"] = (
            consensus["avg_confidence"] * (consensus["count"] - 1) +
            discovery.confidence
        ) / consensus["count"]
        consensus["patterns"].append(discovery.pattern)
        
        # Keep only recent patterns
        if len(consensus["patterns"]) > 100:
            consensus["patterns"] = consensus["patterns"][-100:]
    
    def get_discoveries(self, discovery_type: Optional[DiscoveryType] = None,
                        min_confidence: float = 0.0) -> List[CompressedDiscovery]:
        """Get discoveries, optionally filtered."""
        discoveries = self._discoveries
        
        if discovery_type:
            discoveries = [d for d in discoveries if d.discovery_type == discovery_type]
        
        if min_confidence > 0:
            discoveries = [d for d in discoveries if d.confidence >= min_confidence]
        
        return discoveries
    
    def get_fundamental_discoveries(self) -> List[CompressedDiscovery]:
        """Get foundational discoveries that shape swarm understanding."""
        return self._foundational_discoveries
    
    def get_domain_knowledge(self, domain: str) -> List[CompressedDiscovery]:
        """Get discoveries for a specific domain."""
        return self._domain_knowledge.get(domain, [])
    
    def get_open_questions(self, question_type: Optional[OpenQuestionType] = None,
                           min_importance: float = 0.0) -> List[OpenQuestion]:
        """Get open questions, optionally filtered."""
        questions = self._open_questions
        
        if question_type:
            questions = [q for q in questions if q.question_type == question_type]
        
        if min_importance > 0:
            questions = [q for q in questions if q.importance >= min_importance]
        
        return sorted(questions, key=lambda q: (-q.importance, -q.urgency))
    
    def get_uncertainties(self) -> List[str]:
        """Get current uncertainties the swarm tracks."""
        return self._current_uncertainties
    
    def add_uncertainty(self, uncertainty: str):
        """Add a new uncertainty."""
        if uncertainty not in self._current_uncertainties:
            self._current_uncertainties.append(uncertainty)
    
    def get_cultural_norms(self) -> Dict[str, Any]:
        """Get cultural norms that shape agent behavior."""
        return self._cultural_norms
    
    def get_failed_assumptions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed assumptions."""
        return self._failed_assumptions[-limit:]
    
    def get_adaptation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent adaptations."""
        return self._adaptation_history[-limit:]
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of what the swarm knows and doesn't know."""
        return {
            "what_we_know": {
                "foundational_discoveries": len(self._foundational_discoveries),
                "domain_knowledge": {
                    domain: len(discoveries)
                    for domain, discoveries in self._domain_knowledge.items()
                },
                "cultural_norms": len(self._cultural_norms),
            },
            "what_we_dont_know": {
                "open_questions": len(self._open_questions),
                "uncertainties": len(self._current_uncertainties),
                "failed_assumptions": len(self._failed_assumptions),
            },
            "learning_trajectory": {
                "total_discoveries": len(self._discoveries),
                "adaptations": len(self._adaptation_history),
            },
        }
    
    def get_agent_contributions(self) -> Dict[str, int]:
        """Get contributions by each agent."""
        contributions = {}
        for agent_id, state in self._agent_states.items():
            contributions[agent_id] = state.contributions_to_swarm
        return contributions
    
    def get_consensus(self) -> Dict[str, Any]:
        """Get current consensus patterns."""
        return self._consensus_patterns
    
    def distribute_to_agents(self, agents: Dict[str, AgentEpistemicState]):
        """Distribute relevant discoveries to agents."""
        for discovery in self._discoveries:
            for agent_id, agent in agents.items():
                # Don't send back to originator
                if discovery.discovered_by == agent_id:
                    continue
                
                # Get trust from agent's perspective
                trust = agent.trust_in_others.get(discovery.discovered_by, 0.5)
                
                # Distribute
                agent.ingest_discovery(discovery, trust)
    
    def get_swarm_learning_trajectory(self) -> List[Dict[str, Any]]:
        """Get trajectory of swarm learning."""
        trajectory = []
        
        for discovery in self._discoveries:
            trajectory.append({
                "timestamp": discovery.discovered_at,
                "type": discovery.discovery_type.value,
                "confidence": discovery.confidence,
                "source": discovery.discovered_by,
            })
        
        return sorted(trajectory, key=lambda x: x["timestamp"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_discoveries": len(self._discoveries),
            "agents": len(self._agent_states),
            "consensus_patterns": len(self._consensus_patterns),
            "discoveries_by_type": {
                dt.value: len([d for d in self._discoveries if d.discovery_type == dt])
                for dt in DiscoveryType
            },
            "knowledge_summary": self.get_knowledge_summary(),
        }