"""Discovery Lineage and Conflict Resolution.

Discovery Lineage:
    Not:
        "The swarm knows this."
    But:
        "The swarm knows this because these observations produced
         this chain of validated updates."

    Intellectual genealogy enables:
    - Traceable knowledge provenance
    - Evidence quality assessment
    - Confidence calibration based on source reliability
    - Understanding how knowledge evolved

Conflict Resolution:
    When two high-trust discoveries conflict:
    
    Agent A:
        "Resource pressure causes aggression."
        Confidence: 0.89
    
    Agent B:
        "Resource pressure causes cooperation."
        Confidence: 0.84
    
    The answer may not be "pick one."
    The answer may become:
    
    "Discovery A applies under conditions X.
     Discovery B applies under conditions Y."
    
    This leads to contextual knowledge graphs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
import time
import uuid


class ConflictResolution(Enum):
    """How conflicts between discoveries are resolved."""
    DOMINANT = "dominant"           # Higher confidence wins
    CONTEXTUAL = "contextual"      # Both apply in different contexts
    MERGED = "merged"              # Discoveries are combined
    SUPERSeded = "superseded"      # New discovery replaces old
    PARADOX = "paradox"            # Conflict remains unresolved


class LineageNodeType(Enum):
    """Types of nodes in the discovery lineage tree."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    DISCOVERY = "discovery"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"


@dataclass
class LineageNode:
    """A node in the discovery lineage tree."""
    node_id: str
    node_type: LineageNodeType
    description: str
    
    # Links
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    
    # Content
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    agent_id: str = ""
    timestamp: float = 0.0
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "description": self.description,
            "parents": len(self.parent_ids),
            "children": len(self.child_ids),
            "agent": self.agent_id,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DiscoveryLineage:
    """Intellectual genealogy of a discovery.
    
    Not:
        "The swarm knows this."
    But:
        "The swarm knows this because these observations produced
         this chain of validated updates."
    """
    discovery_id: str
    
    # Origin
    origin_agent: str = ""
    origin_timestamp: float = 0.0
    
    # Lineage tree
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    root_node_id: Optional[str] = None
    
    # Validation chain
    validating_agents: Set[str] = field(default_factory=set)
    validation_count: int = 0
    
    # Evolution
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current state
    current_confidence: float = 0.5
    current_version: int = 1
    
    def add_node(self, node: LineageNode, parent_id: Optional[str] = None):
        """Add a node to the lineage tree."""
        self.nodes[node.node_id] = node
        
        if parent_id and parent_id in self.nodes:
            node.parent_ids.append(parent_id)
            self.nodes[parent_id].child_ids.append(node.node_id)
        
        if self.root_node_id is None:
            self.root_node_id = node.node_id
    
    def record_validation(self, agent_id: str, confidence: float,
                          timestamp: float):
        """Record a validation event."""
        self.validating_agents.add(agent_id)
        self.validation_count += 1
        
        # Update confidence (running average with recency bias)
        alpha = 0.1
        self.current_confidence = (
            alpha * confidence + (1 - alpha) * self.current_confidence
        )
        
        # Add validation node
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            node_type=LineageNodeType.VALIDATION,
            description=f"Validated by {agent_id}",
            agent_id=agent_id,
            timestamp=timestamp,
            confidence=confidence,
        )
        self.add_node(node)
    
    def record_modification(self, agent_id: str, description: str,
                            changes: Dict[str, Any], timestamp: float):
        """Record a modification to the discovery."""
        self.modifications.append({
            "agent_id": agent_id,
            "description": description,
            "changes": changes,
            "timestamp": timestamp,
            "version": self.current_version + 1,
        })
        self.current_version += 1
        
        # Add modification node
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            node_type=LineageNodeType.DISCOVERY,
            description=description,
            agent_id=agent_id,
            timestamp=timestamp,
            data=changes,
        )
        self.add_node(node)
    
    def get_ancestry(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get ancestry chain for a node."""
        ancestry = []
        current_id = node_id
        
        for _ in range(max_depth):
            if current_id not in self.nodes:
                break
            
            node = self.nodes[current_id]
            if not node.parent_ids:
                break
            
            current_id = node.parent_ids[0]
            ancestry.append(current_id)
        
        return ancestry
    
    def get_descendants(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get descendant chain for a node."""
        descendants = []
        current_ids = [node_id]
        
        for _ in range(max_depth):
            next_ids = []
            for cid in current_ids:
                if cid in self.nodes:
                    next_ids.extend(self.nodes[cid].child_ids)
            if not next_ids:
                break
            descendants.extend(next_ids)
            current_ids = next_ids
        
        return descendants
    
    def get_lineage_path(self) -> List[Dict[str, Any]]:
        """Get the main lineage path from root to current."""
        if not self.root_node_id:
            return []
        
        path = []
        current_id = self.root_node_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            path.append(node.to_dict())
            
            if node.child_ids:
                current_id = node.child_ids[0]  # Follow first child
            else:
                break
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "origin_agent": self.origin_agent,
            "origin_timestamp": self.origin_timestamp,
            "nodes": len(self.nodes),
            "validating_agents": len(self.validating_agents),
            "validation_count": self.validation_count,
            "modifications": len(self.modifications),
            "current_confidence": round(self.current_confidence, 3),
            "current_version": self.current_version,
        }


@dataclass
class ConflictPair:
    """A pair of conflicting discoveries."""
    conflict_id: str
    discovery_a_id: str
    discovery_b_id: str
    
    # Conflict details
    conflict_domain: str = ""
    conflict_description: str = ""
    
    # Confidence comparison
    confidence_a: float = 0.5
    confidence_b: float = 0.5
    
    # Evidence comparison
    evidence_count_a: int = 0
    evidence_count_b: int = 0
    
    # Agent comparison
    agent_trust_a: float = 0.5
    agent_trust_b: float = 0.5
    
    # Resolution
    resolution: ConflictResolution = ConflictResolution.PARADOX
    resolution_description: str = ""
    resolved: bool = False
    
    # Context conditions (for contextual resolution)
    conditions_a: Dict[str, Any] = field(default_factory=dict)
    conditions_b: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.conflict_id,
            "discovery_a": self.discovery_a_id,
            "discovery_b": self.discovery_b_id,
            "domain": self.conflict_domain,
            "resolution": self.resolution.value,
            "resolved": self.resolved,
        }


class ConflictResolver:
    """Resolves conflicts between discoveries.
    
    When two high-trust discoveries conflict, the answer may not be
    "pick one." The answer may become:
    
    "Discovery A applies under conditions X.
     Discovery B applies under conditions Y."
    
    This leads to contextual knowledge graphs.
    """
    
    def __init__(self):
        self._conflicts: Dict[str, ConflictPair] = {}
        self._resolutions: List[Dict[str, Any]] = []
    
    def detect_conflict(self, discovery_a: Any, discovery_b: Any,
                        trust_a: float = 0.5, trust_b: float = 0.5) -> Optional[ConflictPair]:
        """Detect if two discoveries conflict.
        
        Returns ConflictPair if conflict detected, None otherwise.
        """
        # Check if discoveries are about the same domain
        domain_a = discovery_a.pattern.get("domain", "general")
        domain_b = discovery_b.pattern.get("domain", "general")
        
        if domain_a != domain_b:
            return None  # Different domains, no conflict
        
        # Check if patterns contradict
        if not self._patterns_contradict(discovery_a.pattern, discovery_b.pattern):
            return None  # No contradiction
        
        # Create conflict pair
        conflict = ConflictPair(
            conflict_id=str(uuid.uuid4()),
            discovery_a_id=discovery_a.discovery_id,
            discovery_b_id=discovery_b.discovery_id,
            conflict_domain=domain_a,
            conflict_description=f"Conflicting patterns in {domain_a}",
            confidence_a=discovery_a.confidence,
            confidence_b=discovery_b.confidence,
            evidence_count_a=discovery_a.evidence_count,
            evidence_count_b=discovery_b.evidence_count,
            agent_trust_a=trust_a,
            agent_trust_b=trust_b,
            conditions_a=discovery_a.conditions,
            conditions_b=discovery_b.conditions,
        )
        
        self._conflicts[conflict.conflict_id] = conflict
        
        return conflict
    
    def _patterns_contradict(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        """Check if two patterns contradict each other."""
        # Simple contradiction detection
        # Look for opposite values on the same keys
        common_keys = set(p1.keys()) & set(p2.keys())
        
        contradictions = 0
        for key in common_keys:
            v1, v2 = p1[key], p2[key]
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Check for opposite signs or very different values
                if v1 * v2 < 0:  # Opposite signs
                    contradictions += 1
                elif abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6) > 0.5:  # Very different
                    contradictions += 1
            elif isinstance(v1, bool) and isinstance(v2, bool):
                if v1 != v2:
                    contradictions += 1
        
        # Need at least 2 contradictions to consider it a real conflict
        return contradictions >= 2
    
    def resolve_conflict(self, conflict_id: str,
                         resolution: ConflictResolution,
                         description: str = "") -> bool:
        """Resolve a conflict."""
        if conflict_id not in self._conflicts:
            return False
        
        conflict = self._conflicts[conflict_id]
        conflict.resolution = resolution
        conflict.resolution_description = description
        conflict.resolved = True
        
        self._resolutions.append({
            "conflict_id": conflict_id,
            "resolution": resolution.value,
            "description": description,
            "timestamp": time.time(),
        })
        
        return True
    
    def get_unresolved_conflicts(self) -> List[ConflictPair]:
        """Get all unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.resolved]
    
    def get_conflicts_by_domain(self, domain: str) -> List[ConflictPair]:
        """Get conflicts for a specific domain."""
        return [c for c in self._conflicts.values() if c.conflict_domain == domain]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_conflicts": len(self._conflicts),
            "resolved": sum(1 for c in self._conflicts.values() if c.resolved),
            "unresolved": sum(1 for c in self._conflicts.values() if not c.resolved),
            "by_resolution": {
                res.value: sum(1 for c in self._conflicts.values() 
                              if c.resolution == res)
                for res in ConflictResolution
            },
        }


class DiscoveryLineageSystem:
    """System for tracking discovery lineage and resolving conflicts.
    
    Architecture:
        Discovery A
              |
              | lineage
              v
        Observation → Hypothesis → Prediction → Validation
              |
              | conflict with
              v
        Discovery B
              |
              | resolution
              v
        Contextual Knowledge Graph
    """
    
    def __init__(self):
        self._lineages: Dict[str, DiscoveryLineage] = {}
        self._conflict_resolver = ConflictResolver()
        self._discovery_index: Dict[str, str] = {}  # discovery_id -> lineage_id
    
    def register_discovery(self, discovery_id: str, agent_id: str,
                           description: str, confidence: float = 0.5) -> DiscoveryLineage:
        """Register a new discovery and start its lineage."""
        lineage = DiscoveryLineage(
            discovery_id=discovery_id,
            origin_agent=agent_id,
            origin_timestamp=time.time(),
            current_confidence=confidence,
        )
        
        # Add origin node
        origin_node = LineageNode(
            node_id=str(uuid.uuid4()),
            node_type=LineageNodeType.DISCOVERY,
            description=description,
            agent_id=agent_id,
            confidence=confidence,
        )
        lineage.add_node(origin_node)
        
        self._lineages[discovery_id] = lineage
        self._discovery_index[discovery_id] = discovery_id
        
        return lineage
    
    def record_observation(self, discovery_id: str, agent_id: str,
                           observation: Dict[str, Any]) -> bool:
        """Record an observation that contributed to a discovery."""
        if discovery_id not in self._lineages:
            return False
        
        lineage = self._lineages[discovery_id]
        
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            node_type=LineageNodeType.OBSERVATION,
            description=f"Observation by {agent_id}",
            agent_id=agent_id,
            data=observation,
        )
        
        # Link to root
        if lineage.root_node_id:
            lineage.add_node(node, parent_id=lineage.root_node_id)
        else:
            lineage.add_node(node)
        
        return True
    
    def record_hypothesis(self, discovery_id: str, agent_id: str,
                          hypothesis: str, confidence: float = 0.5) -> bool:
        """Record a hypothesis that led to a discovery."""
        if discovery_id not in self._lineages:
            return False
        
        lineage = self._lineages[discovery_id]
        
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            node_type=LineageNodeType.HYPOTHESIS,
            description=hypothesis,
            agent_id=agent_id,
            confidence=confidence,
        )
        
        # Find latest node to link
        latest_node = self._get_latest_node(lineage)
        if latest_node:
            lineage.add_node(node, parent_id=latest_node.node_id)
        else:
            lineage.add_node(node)
        
        return True
    
    def record_validation(self, discovery_id: str, agent_id: str,
                          confidence: float) -> bool:
        """Record a validation event."""
        if discovery_id not in self._lineages:
            return False
        
        lineage = self._lineages[discovery_id]
        lineage.record_validation(agent_id, confidence, time.time())
        
        return True
    
    def check_for_conflicts(self, discovery_a_id: str,
                            discovery_b_id: str,
                            trust_a: float = 0.5,
                            trust_b: float = 0.5) -> Optional[ConflictPair]:
        """Check if two discoveries conflict."""
        # This would need access to the actual discovery objects
        # For now, return a placeholder
        return None
    
    def _get_latest_node(self, lineage: DiscoveryLineage) -> Optional[LineageNode]:
        """Get the most recent node in a lineage."""
        if not lineage.nodes:
            return None
        
        return max(lineage.nodes.values(), key=lambda n: n.timestamp)
    
    def get_lineage(self, discovery_id: str) -> Optional[DiscoveryLineage]:
        """Get lineage for a discovery."""
        return self._lineages.get(discovery_id)
    
    def get_lineage_path(self, discovery_id: str) -> List[Dict[str, Any]]:
        """Get the main lineage path for a discovery."""
        lineage = self._lineages.get(discovery_id)
        if not lineage:
            return []
        
        return lineage.get_lineage_path()
    
    def get_conflict_resolver(self) -> ConflictResolver:
        """Get the conflict resolver."""
        return self._conflict_resolver
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lineages": len(self._lineages),
            "conflicts": self._conflict_resolver.to_dict(),
        }