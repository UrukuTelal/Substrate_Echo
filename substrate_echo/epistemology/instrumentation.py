"""Research Instruments — Causal Replay, Counterfactual Engine, Provenance Graph.

Three capabilities that turn the observatory into a research tool.

CausalReplay
    Make event chains executable. Expand any decision into a reasoning tree.
    Answer: "Why did hypothesis B beat A?" "Which observation contributed most?"

CounterfactualEngine
    Remove one observation, re-run, compare. Disable cultural prior, re-run, compare.
    Causal analysis, not just visualization.

KnowledgeProvenanceGraph
    How ideas evolved over time. Intellectual family tree.
    "What observations eventually led to this strategy?"

Architecture:
    CausalReplay
          |
    Reasoning Tree (expandable nodes)
          |
    CounterfactualEngine
          |
    Comparison Report
          |
    KnowledgeProvenanceGraph
          |
    Discovery Lineage Tree
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
import copy
import time
import uuid


class NodeType(Enum):
    """Types of nodes in a reasoning tree."""
    OBSERVATION = "observation"
    FEATURE = "feature"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    COUNCIL = "council"
    TRUST = "trust"
    CULTURAL_PRIOR = "cultural_prior"
    DECISION = "decision"
    OUTCOME = "outcome"
    BELIEF_UPDATE = "belief_update"
    DISCOVERY = "discovery"
    ACTION = "action"


@dataclass
class ReasoningNode:
    """A single node in a reasoning tree."""
    node_id: str
    node_type: NodeType
    tick: int
    data: Dict[str, Any]
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Causal weight (how much this node influenced the outcome)
    weight: float = 0.0
    
    # Metadata
    timestamp: float = 0.0
    module: str = ""
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "tick": self.tick,
            "data": self.data,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "weight": round(self.weight, 3),
            "module": self.module,
        }


@dataclass
class ReasoningTree:
    """A tree of reasoning leading to a decision."""
    root_id: str
    nodes: Dict[str, ReasoningNode]
    decision_tick: int
    decision_data: Dict[str, Any]
    
    # Summary
    total_weight: float = 0.0
    depth: int = 0
    
    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[ReasoningNode]:
        """Get children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_path_to_root(self, node_id: str) -> List[ReasoningNode]:
        """Get path from a node to the root."""
        path = []
        current = node_id
        while current:
            node = self.nodes.get(current)
            if not node:
                break
            path.append(node)
            current = node.parent_id
        return list(reversed(path))
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ReasoningNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_most_influential(self, top_k: int = 5) -> List[ReasoningNode]:
        """Get the most influential nodes."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: abs(n.weight),
            reverse=True
        )
        return sorted_nodes[:top_k]
    
    def render(self, indent: int = 0) -> str:
        """Render the tree as indented text."""
        lines = []
        prefix = "  " * indent
        
        def _render_node(node_id: str, depth: int):
            node = self.nodes.get(node_id)
            if not node:
                return
            
            p = "  " * depth
            weight_bar = "#" * int(abs(node.weight) * 20) if node.weight else ""
            
            # Format data
            data_str = ""
            if node.data:
                key_items = list(node.data.items())[:3]
                data_str = ", ".join(f"{k}={v}" for k, v in key_items)
                if len(node.data) > 3:
                    data_str += f" (+{len(node.data) - 3} more)"
            
            lines.append(
                f"{p}[{node.node_type.value}] {data_str} "
                f"{weight_bar} (w={node.weight:.2f})"
            )
            
            for child_id in node.children_ids:
                _render_node(child_id, depth + 1)
        
        _render_node(self.root_id, 0)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "decision_tick": self.decision_tick,
            "decision_data": self.decision_data,
            "total_nodes": len(self.nodes),
            "depth": self.depth,
            "total_weight": round(self.total_weight, 3),
        }


class CausalReplay:
    """Make event chains executable.
    
    Given a decision event, expand its full reasoning tree
    showing every observation, hypothesis, prediction, council
    deliberation, and trust update that led to it.
    
    Usage:
        replay = CausalReplay()
        
        # Record events during cognition
        obs_event = replay.record_observation(tick, {"enemy_scout": True})
        hyp_event = replay.record_hypothesis(tick, {"scouting": 0.7}, caused_by=obs_event.event_id)
        pred_event = replay.record_prediction(tick, {"leaves_in": 12}, caused_by=hyp_event.event_id)
        decision = replay.record_decision(tick, {"action": "expand"}, caused_by=pred_event.event_id)
        
        # Expand decision into reasoning tree
        tree = replay.expand_decision(decision.event_id)
        
        # Render
        print(tree.render())
    """
    
    def __init__(self):
        self._events: Dict[str, Dict[str, Any]] = {}
        self._trees: Dict[str, ReasoningTree] = {}
    
    def _record_event(self, tick: int, node_type: NodeType,
                      data: Dict[str, Any],
                      caused_by: Optional[str] = None,
                      weight: float = 0.0) -> ReasoningNode:
        """Record an event as a reasoning node."""
        node_id = str(uuid.uuid4())
        
        node = ReasoningNode(
            node_id=node_id,
            node_type=node_type,
            tick=tick,
            data=data,
            parent_id=caused_by,
            weight=weight,
        )
        
        self._events[node_id] = {
            "tick": tick,
            "node_type": node_type,
            "data": data,
            "caused_by": caused_by,
            "weight": weight,
        }
        
        # Update parent's children
        if caused_by and caused_by in self._events:
            # We'll rebuild trees on expand
            pass
        
        return node
    
    def record_observation(self, tick: int, data: Dict[str, Any],
                           caused_by: Optional[str] = None,
                           weight: float = 0.1) -> ReasoningNode:
        """Record an observation."""
        return self._record_event(tick, NodeType.OBSERVATION, data, caused_by, weight)
    
    def record_feature(self, tick: int, data: Dict[str, Any],
                       caused_by: Optional[str] = None,
                       weight: float = 0.15) -> ReasoningNode:
        """Record a feature extraction."""
        return self._record_event(tick, NodeType.FEATURE, data, caused_by, weight)
    
    def record_hypothesis(self, tick: int, data: Dict[str, Any],
                          caused_by: Optional[str] = None,
                          weight: float = 0.3) -> ReasoningNode:
        """Record a hypothesis generation."""
        return self._record_event(tick, NodeType.HYPOTHESIS, data, caused_by, weight)
    
    def record_prediction(self, tick: int, data: Dict[str, Any],
                          caused_by: Optional[str] = None,
                          weight: float = 0.25) -> ReasoningNode:
        """Record a prediction."""
        return self._record_event(tick, NodeType.PREDICTION, data, caused_by, weight)
    
    def record_council(self, tick: int, data: Dict[str, Any],
                       caused_by: Optional[str] = None,
                       weight: float = 0.35) -> ReasoningNode:
        """Record a council deliberation."""
        return self._record_event(tick, NodeType.COUNCIL, data, caused_by, weight)
    
    def record_trust(self, tick: int, data: Dict[str, Any],
                     caused_by: Optional[str] = None,
                     weight: float = 0.2) -> ReasoningNode:
        """Record a trust update."""
        return self._record_event(tick, NodeType.TRUST, data, caused_by, weight)
    
    def record_cultural_prior(self, tick: int, data: Dict[str, Any],
                              caused_by: Optional[str] = None,
                              weight: float = 0.4) -> ReasoningNode:
        """Record a cultural prior application."""
        return self._record_event(tick, NodeType.CULTURAL_PRIOR, data, caused_by, weight)
    
    def record_decision(self, tick: int, data: Dict[str, Any],
                        caused_by: Optional[str] = None,
                        weight: float = 1.0) -> ReasoningNode:
        """Record a decision (the root of a reasoning tree)."""
        return self._record_event(tick, NodeType.DECISION, data, caused_by, weight)
    
    def record_outcome(self, tick: int, data: Dict[str, Any],
                       caused_by: Optional[str] = None,
                       weight: float = 0.5) -> ReasoningNode:
        """Record an outcome."""
        return self._record_event(tick, NodeType.OUTCOME, data, caused_by, weight)
    
    def record_belief_update(self, tick: int, data: Dict[str, Any],
                             caused_by: Optional[str] = None,
                             weight: float = 0.3) -> ReasoningNode:
        """Record a belief update."""
        return self._record_event(tick, NodeType.BELIEF_UPDATE, data, caused_by, weight)
    
    def record_action(self, tick: int, data: Dict[str, Any],
                      caused_by: Optional[str] = None,
                      weight: float = 0.8) -> ReasoningNode:
        """Record an action."""
        return self._record_event(tick, NodeType.ACTION, data, caused_by, weight)
    
    def record_discovery(self, tick: int, data: Dict[str, Any],
                         caused_by: Optional[str] = None,
                         weight: float = 0.45) -> ReasoningNode:
        """Record a discovery."""
        return self._record_event(tick, NodeType.DISCOVERY, data, caused_by, weight)
    
    def expand_decision(self, decision_id: str) -> Optional[ReasoningTree]:
        """Expand a decision into a full reasoning tree."""
        if decision_id not in self._events:
            return None
        
        decision_data = self._events[decision_id]
        decision_tick = decision_data["tick"]
        
        # Build tree by tracing causes
        nodes: Dict[str, ReasoningNode] = {}
        visited: Set[str] = set()
        max_depth = 0
        
        def _build_tree(node_id: str, depth: int = 0):
            nonlocal max_depth
            if node_id in visited or depth > 20:
                return
            
            visited.add(node_id)
            event = self._events.get(node_id)
            if not event:
                return
            
            parent_id = event.get("caused_by")
            
            node = ReasoningNode(
                node_id=node_id,
                node_type=event["node_type"],
                tick=event["tick"],
                data=event["data"],
                parent_id=parent_id,
                weight=event["weight"],
            )
            
            nodes[node_id] = node
            
            # Link to parent
            if parent_id and parent_id in nodes:
                nodes[parent_id].children_ids.append(node_id)
            
            max_depth = max(max_depth, depth)
            
            # Trace to parent (what caused this event)
            if parent_id:
                _build_tree(parent_id, depth + 1)
            
            # Also trace to children (what this event caused)
            for eid, edata in self._events.items():
                if edata.get("caused_by") == node_id and eid not in visited:
                    _build_tree(eid, depth + 1)
        
        _build_tree(decision_id)
        
        tree = ReasoningTree(
            root_id=decision_id,
            nodes=nodes,
            decision_tick=decision_tick,
            decision_data=decision_data["data"],
            depth=max_depth,
            total_weight=sum(n.weight for n in nodes.values()),
        )
        
        self._trees[decision_id] = tree
        return tree
    
    def get_all_trees(self) -> List[ReasoningTree]:
        """Get all expanded trees."""
        return list(self._trees.values())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "total_trees": len(self._trees),
        }


# ---------------------------------------------------------------------------
# Counterfactual Engine
# ---------------------------------------------------------------------------

class ModificationType(Enum):
    """Types of modifications for counterfactual analysis."""
    REMOVE_OBSERVATION = "remove_observation"
    DISABLE_CULTURAL_PRIOR = "disable_cultural_prior"
    CHANGE_TRUST = "change_trust"
    REMOVE_DISCOVERY = "remove_discovery"
    CHANGE_HYPOTHESIS = "change_hypothesis"
    REMOVE_PREDICTION = "remove_prediction"


@dataclass
class Modification:
    """A modification to apply for counterfactual analysis."""
    mod_type: ModificationType
    target_id: str  # event_id, prior_id, agent_id, etc.
    new_value: Any = None  # for CHANGE_TRUST, CHANGE_HYPOTHESIS
    description: str = ""


@dataclass
class ComparisonReport:
    """Report comparing original vs counterfactual reasoning."""
    original_tree: ReasoningTree
    counterfactual_tree: Optional[ReasoningTree]
    modification: Modification
    
    # Differences
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_changed: List[str] = field(default_factory=list)
    
    # Decision comparison
    original_decision: Dict[str, Any] = field(default_factory=dict)
    counterfactual_decision: Dict[str, Any] = field(default_factory=dict)
    decision_changed: bool = False
    
    # Weight comparison
    original_total_weight: float = 0.0
    counterfactual_total_weight: float = 0.0
    
    def render(self) -> str:
        """Render comparison as text."""
        lines = []
        lines.append("=" * 60)
        lines.append("Counterfactual Analysis")
        lines.append("=" * 60)
        lines.append(f"Modification: {self.modification.mod_type.value}")
        lines.append(f"Target: {self.modification.target_id[:16]}...")
        lines.append(f"Description: {self.modification.description}")
        lines.append("")
        
        # Decision comparison
        lines.append("Decision Comparison")
        lines.append("-" * 60)
        lines.append(f"  Original:      {self.original_decision}")
        if self.counterfactual_decision:
            lines.append(f"  Counterfactual: {self.counterfactual_decision}")
            lines.append(f"  Changed: {self.decision_changed}")
        else:
            lines.append("  Counterfactual: (no tree generated)")
        
        lines.append("")
        
        # Tree comparison
        lines.append("Tree Comparison")
        lines.append("-" * 60)
        lines.append(f"  Original nodes: {len(self.original_tree.nodes)}")
        if self.counterfactual_tree:
            lines.append(f"  Counterfactual nodes: {len(self.counterfactual_tree.nodes)}")
            lines.append(f"  Nodes added: {len(self.nodes_added)}")
            lines.append(f"  Nodes removed: {len(self.nodes_removed)}")
            lines.append(f"  Nodes changed: {len(self.nodes_changed)}")
        
        lines.append("")
        
        # Weight comparison
        lines.append("Weight Comparison")
        lines.append("-" * 60)
        lines.append(f"  Original total: {self.original_total_weight:.3f}")
        if self.counterfactual_tree:
            lines.append(f"  Counterfactual total: {self.counterfactual_total_weight:.3f}")
            delta = self.counterfactual_total_weight - self.original_total_weight
            lines.append(f"  Delta: {delta:+.3f}")
        
        lines.append("")
        
        # Original tree
        lines.append("Original Reasoning Tree")
        lines.append("-" * 60)
        lines.append(self.original_tree.render())
        
        # Counterfactual tree
        if self.counterfactual_tree:
            lines.append("")
            lines.append("Counterfactual Reasoning Tree")
            lines.append("-" * 60)
            lines.append(self.counterfactual_tree.render())
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "modification": self.modification.mod_type.value,
            "target_id": self.modification.target_id,
            "decision_changed": self.decision_changed,
            "original_nodes": len(self.original_tree.nodes),
            "counterfactual_nodes": (
                len(self.counterfactual_tree.nodes)
                if self.counterfactual_tree else 0
            ),
            "nodes_added": len(self.nodes_added),
            "nodes_removed": len(self.nodes_removed),
            "nodes_changed": len(self.nodes_changed),
        }


class CounterfactualEngine:
    """Replay reasoning with modifications.
    
    Takes an original reasoning tree, applies a modification,
    and rebuilds the tree to show how the decision changes.
    
    Usage:
        engine = CounterfactualEngine(replay)
        
        # Get original tree
        tree = replay.expand_decision(decision_id)
        
        # Create modification
        mod = Modification(
            mod_type=ModificationType.REMOVE_OBSERVATION,
            target_id=observation_event_id,
            description="What if we never saw the enemy scout?"
        )
        
        # Compare
        report = engine.compare(tree, mod)
        print(report.render())
    """
    
    def __init__(self, replay: CausalReplay):
        self._replay = replay
    
    def compare(self, original_tree: ReasoningTree,
                modification: Modification) -> ComparisonReport:
        """Compare original vs counterfactual reasoning."""
        
        # Build counterfactual tree
        counterfactual_tree = self._build_counterfactual(
            original_tree, modification
        )
        
        # Compare
        report = ComparisonReport(
            original_tree=original_tree,
            counterfactual_tree=counterfactual_tree,
            modification=modification,
            original_decision=original_tree.decision_data,
            original_total_weight=original_tree.total_weight,
        )
        
        if counterfactual_tree:
            report.counterfactual_decision = counterfactual_tree.decision_data
            report.decision_changed = (
                original_tree.decision_data != counterfactual_tree.decision_data
            )
            report.counterfactual_total_weight = counterfactual_tree.total_weight
            
            # Find differences
            original_ids = set(original_tree.nodes.keys())
            counter_ids = set(counterfactual_tree.nodes.keys())
            
            report.nodes_removed = list(original_ids - counter_ids)
            report.nodes_added = list(counter_ids - original_ids)
            
            # Check for changed nodes
            for nid in original_ids & counter_ids:
                orig = original_tree.nodes[nid]
                counter = counterfactual_tree.nodes[nid]
                if orig.data != counter.data or orig.weight != counter.weight:
                    report.nodes_changed.append(nid)
        
        return report
    
    def _build_counterfactual(self, original_tree: ReasoningTree,
                              modification: Modification) -> Optional[ReasoningTree]:
        """Build a counterfactual tree with modification applied."""
        
        # Deep copy the tree
        new_nodes: Dict[str, ReasoningNode] = {}
        for nid, node in original_tree.nodes.items():
            new_node = ReasoningNode(
                node_id=node.node_id,
                node_type=node.node_type,
                tick=node.tick,
                data=copy.deepcopy(node.data),
                parent_id=node.parent_id,
                children_ids=list(node.children_ids),
                weight=node.weight,
                module=node.module,
            )
            new_nodes[nid] = new_node
        
        # Apply modification
        if modification.mod_type == ModificationType.REMOVE_OBSERVATION:
            # Remove the observation node and all its descendants
            target_id = modification.target_id
            if target_id in new_nodes:
                self._remove_node_and_descendants(target_id, new_nodes)
        
        elif modification.mod_type == ModificationType.DISABLE_CULTURAL_PRIOR:
            # Remove cultural prior nodes
            for nid, node in list(new_nodes.items()):
                if (node.node_type == NodeType.CULTURAL_PRIOR and
                    modification.target_id in str(node.data)):
                    self._remove_node_and_descendants(nid, new_nodes)
        
        elif modification.mod_type == ModificationType.CHANGE_TRUST:
            # Modify trust values
            new_value = modification.new_value or 0.5
            for node in new_nodes.values():
                if node.node_type == NodeType.TRUST:
                    if modification.target_id in str(node.data):
                        # Reduce weight (less influence)
                        node.weight *= 0.3
                        node.data["modified"] = True
                        node.data["original_weight"] = node.weight
        
        elif modification.mod_type == ModificationType.REMOVE_DISCOVERY:
            # Remove discovery nodes
            for nid, node in list(new_nodes.items()):
                if (node.node_type == NodeType.DISCOVERY and
                    modification.target_id in str(node.data)):
                    self._remove_node_and_descendants(nid, new_nodes)
        
        elif modification.mod_type == ModificationType.CHANGE_HYPOTHESIS:
            # Modify hypothesis confidence
            for node in new_nodes.values():
                if node.node_type == NodeType.HYPOTHESIS:
                    if modification.target_id in str(node.data):
                        # Flip probabilities
                        for key in node.data:
                            if isinstance(node.data[key], (int, float)):
                                node.data[key] = 1.0 - node.data[key]
                        node.data["modified"] = True
        
        # Rebuild tree
        if not new_nodes:
            return None
        
        # Find root (node with no parent that's still in the tree)
        root_id = None
        for nid, node in new_nodes.items():
            if node.parent_id is None or node.parent_id not in new_nodes:
                root_id = nid
                break
        
        if not root_id:
            return None
        
        # Calculate depth and weight
        max_depth = 0
        total_weight = 0.0
        
        def _calc_depth(node_id: str, depth: int = 0):
            nonlocal max_depth, total_weight
            node = new_nodes.get(node_id)
            if not node:
                return
            max_depth = max(max_depth, depth)
            total_weight += node.weight
            for child_id in node.children_ids:
                if child_id in new_nodes:
                    _calc_depth(child_id, depth + 1)
        
        _calc_depth(root_id)
        
        return ReasoningTree(
            root_id=root_id,
            nodes=new_nodes,
            decision_tick=original_tree.decision_tick,
            decision_data=new_nodes[root_id].data,
            depth=max_depth,
            total_weight=total_weight,
        )
    
    def _remove_node_and_descendants(self, node_id: str,
                                     nodes: Dict[str, ReasoningNode]):
        """Remove a node and all its descendants."""
        node = nodes.get(node_id)
        if not node:
            return
        
        # Remove children first
        for child_id in list(node.children_ids):
            self._remove_node_and_descendants(child_id, nodes)
        
        # Remove from parent's children list
        if node.parent_id and node.parent_id in nodes:
            parent = nodes[node.parent_id]
            parent.children_ids = [
                cid for cid in parent.children_ids if cid != node_id
            ]
        
        # Remove node
        del nodes[node_id]


# ---------------------------------------------------------------------------
# Knowledge Provenance Graph
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceNode:
    """A node in the knowledge provenance graph."""
    node_id: str
    node_type: str  # "observation", "hypothesis", "prediction", "discovery", "prior", "decision"
    tick: int
    data: Dict[str, Any]
    timestamp: float = 0.0
    
    # Edges
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    
    # Metrics
    influence_score: float = 0.0  # how much this node influenced later decisions
    validation_count: int = 0  # how many times this was validated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "tick": self.tick,
            "data": self.data,
            "parents": self.parents,
            "children": self.children,
            "influence_score": round(self.influence_score, 3),
            "validation_count": self.validation_count,
        }


@dataclass
class ProvenanceEdge:
    """An edge in the knowledge provenance graph."""
    edge_id: str
    source_id: str
    target_id: str
    relationship: str  # "led_to", "validated_by", "influenced", "overturned_by"
    tick: int
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "tick": self.tick,
            "weight": round(self.weight, 3),
        }


class KnowledgeProvenanceGraph:
    """Track how discoveries evolved over time.
    
    An intellectual family tree showing:
    - Which observations led to which discoveries
    - Which predictions were validated
    - How cultural priors formed
    - What decisions were influenced by what knowledge
    
    Usage:
        graph = KnowledgeProvenanceGraph()
        
        # Record provenance
        obs = graph.record_observation(tick, {"enemy_scout": True})
        hyp = graph.record_hypothesis(tick, {"scouting": 0.7}, parents=[obs])
        disc = graph.record_discovery(tick, {"rule": "scout_early"}, parents=[hyp])
        
        # Get lineage
        lineage = graph.get_lineage(disc.node_id)
        
        # Get influence
        influence = graph.get_influence(obs.node_id)
    """
    
    def __init__(self):
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._edges: List[ProvenanceEdge] = []
        self._edges_by_source: Dict[str, List[ProvenanceEdge]] = {}
        self._edges_by_target: Dict[str, List[ProvenanceEdge]] = {}
    
    def _create_node(self, node_type: str, tick: int,
                     data: Dict[str, Any]) -> ProvenanceNode:
        """Create a new provenance node."""
        node_id = str(uuid.uuid4())
        
        node = ProvenanceNode(
            node_id=node_id,
            node_type=node_type,
            tick=tick,
            data=data,
            timestamp=time.time(),
        )
        
        self._nodes[node_id] = node
        return node
    
    def _create_edge(self, source_id: str, target_id: str,
                     relationship: str, tick: int,
                     weight: float = 1.0) -> ProvenanceEdge:
        """Create a new provenance edge."""
        edge_id = str(uuid.uuid4())
        
        edge = ProvenanceEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            tick=tick,
            weight=weight,
        )
        
        self._edges.append(edge)
        
        # Index
        if source_id not in self._edges_by_source:
            self._edges_by_source[source_id] = []
        self._edges_by_source[source_id].append(edge)
        
        if target_id not in self._edges_by_target:
            self._edges_by_target[target_id] = []
        self._edges_by_target[target_id].append(edge)
        
        # Update nodes
        if source_id in self._nodes:
            self._nodes[source_id].children.append(target_id)
        if target_id in self._nodes:
            self._nodes[target_id].parents.append(source_id)
        
        return edge
    
    def record_observation(self, tick: int, data: Dict[str, Any],
                           parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record an observation in the provenance graph."""
        node = self._create_node("observation", tick, data)
        
        # Connect to parents
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "led_to", tick)
        
        return node
    
    def record_hypothesis(self, tick: int, data: Dict[str, Any],
                          parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record a hypothesis."""
        node = self._create_node("hypothesis", tick, data)
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "led_to", tick)
        return node
    
    def record_prediction(self, tick: int, data: Dict[str, Any],
                          parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record a prediction."""
        node = self._create_node("prediction", tick, data)
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "led_to", tick)
        return node
    
    def record_discovery(self, tick: int, data: Dict[str, Any],
                         parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record a discovery."""
        node = self._create_node("discovery", tick, data)
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "led_to", tick)
        return node
    
    def record_prior(self, tick: int, data: Dict[str, Any],
                     parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record a cultural prior."""
        node = self._create_node("prior", tick, data)
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "led_to", tick)
        return node
    
    def record_decision(self, tick: int, data: Dict[str, Any],
                        parents: Optional[List[str]] = None) -> ProvenanceNode:
        """Record a decision."""
        node = self._create_node("decision", tick, data)
        for parent_id in (parents or []):
            self._create_edge(parent_id, node.node_id, "influenced", tick)
        return node
    
    def record_validation(self, tick: int, prediction_id: str,
                          outcome_data: Dict[str, Any]) -> ProvenanceNode:
        """Record that a prediction was validated."""
        outcome = self._create_node("outcome", tick, outcome_data)
        self._create_edge(prediction_id, outcome.node_id, "validated_by", tick)
        
        # Update validation count
        if prediction_id in self._nodes:
            self._nodes[prediction_id].validation_count += 1
        
        return outcome
    
    def record_overturning(self, tick: int, old_id: str,
                           new_id: str) -> ProvenanceEdge:
        """Record that one idea overturned another."""
        return self._create_edge(old_id, new_id, "overturned_by", tick)
    
    def get_lineage(self, node_id: str, depth: int = 20) -> List[ProvenanceNode]:
        """Get the full lineage of a node (all ancestors)."""
        lineage = []
        visited = set()
        
        def _trace(nid: str, current_depth: int):
            if current_depth <= 0 or nid in visited:
                return
            visited.add(nid)
            
            node = self._nodes.get(nid)
            if not node:
                return
            
            lineage.append(node)
            
            # Trace parents
            for edge in self._edges_by_target.get(nid, []):
                _trace(edge.source_id, current_depth - 1)
        
        _trace(node_id, depth)
        return list(reversed(lineage))
    
    def get_descendants(self, node_id: str, depth: int = 20) -> List[ProvenanceNode]:
        """Get all descendants of a node."""
        descendants = []
        visited = set()
        
        def _trace(nid: str, current_depth: int):
            if current_depth <= 0 or nid in visited:
                return
            visited.add(nid)
            
            node = self._nodes.get(nid)
            if not node:
                return
            
            descendants.append(node)
            
            for edge in self._edges_by_source.get(nid, []):
                _trace(edge.target_id, current_depth - 1)
        
        _trace(node_id, depth)
        return descendants
    
    def get_influence(self, node_id: str) -> Dict[str, Any]:
        """Calculate the influence of a node."""
        descendants = self.get_descendants(node_id)
        
        # Count by type
        by_type = {}
        for desc in descendants:
            by_type[desc.node_type] = by_type.get(desc.node_type, 0) + 1
        
        # Count decisions influenced
        decisions = [d for d in descendants if d.node_type == "decision"]
        
        # Count discoveries led to
        discoveries = [d for d in descendants if d.node_type == "discovery"]
        
        return {
            "node_id": node_id,
            "total_descendants": len(descendants),
            "by_type": by_type,
            "decisions_influenced": len(decisions),
            "discoveries_led_to": len(discoveries),
            "influence_score": len(descendants) * 0.1,
        }
    
    def get_roots(self) -> List[ProvenanceNode]:
        """Get all root nodes (no parents)."""
        return [
            node for node in self._nodes.values()
            if not node.parents
        ]
    
    def get_discovery_trees(self) -> List[Dict[str, Any]]:
        """Get the intellectual family tree for each discovery."""
        discoveries = [
            node for node in self._nodes.values()
            if node.node_type == "discovery"
        ]
        
        trees = []
        for disc in discoveries:
            lineage = self.get_lineage(disc.node_id)
            influence = self.get_influence(disc.node_id)
            
            trees.append({
                "discovery": disc.to_dict(),
                "lineage": [n.to_dict() for n in lineage],
                "influence": influence,
            })
        
        return trees
    
    def render(self) -> str:
        """Render the provenance graph as text."""
        lines = []
        lines.append("=" * 60)
        lines.append("Knowledge Provenance Graph")
        lines.append("=" * 60)
        lines.append(f"Nodes: {len(self._nodes)}")
        lines.append(f"Edges: {len(self._edges)}")
        lines.append("")
        
        # Get roots
        roots = self.get_roots()
        lines.append(f"Root nodes: {len(roots)}")
        lines.append("")
        
        # Render each root's tree
        for root in roots[:5]:  # Show first 5 trees
            lines.append(f"--- Tree from {root.node_type} (tick {root.tick}) ---")
            
            def _render_node(nid: str, indent: int = 0):
                node = self._nodes.get(nid)
                if not node:
                    return
                
                prefix = "  " * indent
                data_str = str(node.data)[:50]
                lines.append(
                    f"{prefix}[{node.node_type}] {data_str} "
                    f"(influence={node.influence_score:.1f})"
                )
                
                for edge in self._edges_by_source.get(nid, []):
                    _render_node(edge.target_id, indent + 1)
            
            _render_node(root.node_id)
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "roots": len(self.get_roots()),
            "discoveries": len([
                n for n in self._nodes.values() if n.node_type == "discovery"
            ]),
        }


# ---------------------------------------------------------------------------
# Dependency Analyzer & Epistemic Dependency Index
# ---------------------------------------------------------------------------

@dataclass
class NodeInfluenceProfile:
    """Influence profile of a single node type."""
    node_type: NodeType
    count: int = 0
    total_weight_original: float = 0.0
    total_weight_counterfactual: float = 0.0
    weight_delta: float = 0.0
    decisions_changed: int = 0
    decisions_tested: int = 0
    
    @property
    def influence_ratio(self) -> float:
        """How much removing this node type changes the reasoning."""
        if self.decisions_tested == 0:
            return 0.0
        if self.total_weight_original == 0:
            return 0.0
        return abs(self.weight_delta) / self.total_weight_original
    
    @property
    def decision_change_rate(self) -> float:
        """Fraction of decisions that changed when this was removed."""
        if self.decisions_tested == 0:
            return 0.0
        return self.decisions_changed / self.decisions_tested
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "count": self.count,
            "influence_ratio": round(self.influence_ratio, 3),
            "decision_change_rate": round(self.decision_change_rate, 3),
            "weight_delta": round(self.weight_delta, 3),
            "decisions_changed": self.decisions_changed,
            "decisions_tested": self.decisions_tested,
        }


@dataclass
class DependencyReport:
    """Report of reasoning dependency analysis."""
    total_decisions_tested: int = 0
    total_nodes_analyzed: int = 0
    
    # Per-type influence
    profiles: Dict[NodeType, NodeInfluenceProfile] = field(default_factory=dict)
    
    # Balance metrics
    herfindahl_index: float = 0.0  # 0=perfect balance, 1=total concentration
    source_herfindahl_index: float = 0.0  # only cognitive sources, not structural
    dominant_type: Optional[NodeType] = None
    dominant_source: Optional[NodeType] = None
    
    # Cognitive source types (what the user cares about)
    SOURCE_TYPES = {
        NodeType.OBSERVATION, NodeType.CULTURAL_PRIOR, NodeType.TRUST,
        NodeType.PREDICTION, NodeType.DISCOVERY, NodeType.HYPOTHESIS,
    }
    
    def compute_balance(self):
        """Compute balance metrics from profiles."""
        if not self.profiles:
            return
        
        # All-node HHI
        total_influence = sum(p.influence_ratio for p in self.profiles.values())
        if total_influence == 0:
            return
        
        shares = {
            nt: p.influence_ratio / total_influence
            for nt, p in self.profiles.items()
        }
        
        self.herfindahl_index = sum(s ** 2 for s in shares.values())
        
        if shares:
            self.dominant_type = max(shares, key=shares.get)
        
        # Source-only HHI (cognitive sources only)
        source_profiles = {
            nt: p for nt, p in self.profiles.items()
            if nt in self.SOURCE_TYPES and p.influence_ratio > 0
        }
        
        if source_profiles:
            source_total = sum(p.influence_ratio for p in source_profiles.values())
            if source_total > 0:
                source_shares = {
                    nt: p.influence_ratio / source_total
                    for nt, p in source_profiles.items()
                }
                self.source_herfindahl_index = sum(s ** 2 for s in source_shares.values())
                self.dominant_source = max(source_shares, key=source_shares.get)
    
    def render(self) -> str:
        """Render dependency report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Dependency Index (EDI)")
        lines.append("=" * 60)
        lines.append(f"Decisions tested: {self.total_decisions_tested}")
        lines.append(f"Nodes analyzed: {self.total_nodes_analyzed}")
        lines.append("")
        
        # Influence profile
        lines.append("Influence Profile")
        lines.append("-" * 60)
        
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.influence_ratio,
            reverse=True
        )
        
        for profile in sorted_profiles:
            bar_len = int(profile.influence_ratio * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            lines.append(
                f"  {profile.node_type.value:20s} "
                f"{bar} "
                f"{profile.influence_ratio:.1%} "
                f"(changed {profile.decisions_changed}/{profile.decisions_tested})"
            )
        
        lines.append("")
        
        # Balance
        lines.append("Balance Metrics")
        lines.append("-" * 60)
        lines.append(f"  Herfindahl Index: {self.herfindahl_index:.3f}")
        lines.append(f"    (0.0=perfect balance, 1.0=total concentration)")
        lines.append(f"  Source-only HHI:  {self.source_herfindahl_index:.3f}")
        lines.append(f"    (cognitive sources only: observation, prior, trust, prediction, discovery, hypothesis)")
        
        if self.dominant_type:
            lines.append(f"  Dominant type: {self.dominant_type.value}")
        if self.dominant_source:
            lines.append(f"  Dominant source: {self.dominant_source.value}")
        
        # Interpretation
        lines.append("")
        lines.append("Interpretation")
        lines.append("-" * 60)
        if self.herfindahl_index > 0.4:
            lines.append("  HIGH CONCENTRATION — one source dominates reasoning")
        elif self.herfindahl_index > 0.25:
            lines.append("  MODERATE CONCENTRATION — some imbalance")
        else:
            lines.append("  BALANCED — multiple sources contribute")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_decisions_tested": self.total_decisions_tested,
            "total_nodes_analyzed": self.total_nodes_analyzed,
            "profiles": {
                nt.value: p.to_dict() for nt, p in self.profiles.items()
            },
            "herfindahl_index": round(self.herfindahl_index, 4),
            "source_herfindahl_index": round(self.source_herfindahl_index, 4),
            "dominant_type": self.dominant_type.value if self.dominant_type else None,
            "dominant_source": self.dominant_source.value if self.dominant_source else None,
        }


class DependencyAnalyzer:
    """Analyze reasoning dependencies via counterfactual testing.
    
    For each node type, removes all nodes of that type and measures
    how much the reasoning changes. This produces an influence profile
    showing which cognitive sources dominate decisions.
    
    Usage:
        analyzer = DependencyAnalyzer(counterfactual_engine)
        
        # Analyze a single decision
        report = analyzer.analyze_decision(decision_id)
        
        # Analyze across many decisions
        report = analyzer.analyze_all(decision_ids)
        
        print(report.render())
    """
    
    def __init__(self, engine: CounterfactualEngine):
        self._engine = engine
    
    def analyze_decision(self, decision_id: str) -> DependencyReport:
        """Analyze dependencies for a single decision."""
        tree = self._engine._replay.expand_decision(decision_id)
        if not tree:
            return DependencyReport()
        
        report = DependencyReport(
            total_decisions_tested=1,
            total_nodes_analyzed=len(tree.nodes),
        )
        
        # Initialize profiles
        for node_type in NodeType:
            report.profiles[node_type] = NodeInfluenceProfile(node_type=node_type)
        
        # Test removing each node type
        for node_type in NodeType:
            nodes_of_type = [
                n for n in tree.nodes.values()
                if n.node_type == node_type
            ]
            
            profile = report.profiles[node_type]
            profile.count = len(nodes_of_type)
            profile.decisions_tested = 1
            
            if not nodes_of_type:
                continue
            
            # Remove each node of this type and measure
            total_delta = 0.0
            changed = 0
            
            for node in nodes_of_type:
                mod = Modification(
                    mod_type=ModificationType.REMOVE_OBSERVATION,
                    target_id=node.node_id,
                    description=f"Remove {node_type.value}",
                )
                
                result = self._engine.compare(tree, mod)
                
                if result.decision_changed:
                    changed += 1
                
                total_delta += abs(result.counterfactual_total_weight - result.original_total_weight)
            
            profile.weight_delta = total_delta / len(nodes_of_type)
            profile.total_weight_original = tree.total_weight
            profile.decisions_changed = changed
        
        report.compute_balance()
        return report
    
    def analyze_all(self, decision_ids: List[str]) -> DependencyReport:
        """Analyze dependencies across multiple decisions."""
        combined = DependencyReport()
        
        # Initialize profiles
        for node_type in NodeType:
            combined.profiles[node_type] = NodeInfluenceProfile(node_type=node_type)
        
        for decision_id in decision_ids:
            single = self.analyze_decision(decision_id)
            
            combined.total_decisions_tested += single.total_decisions_tested
            combined.total_nodes_analyzed += single.total_nodes_analyzed
            
            for node_type, profile in single.profiles.items():
                combined.profiles[node_type].count += profile.count
                combined.profiles[node_type].weight_delta += profile.weight_delta
                combined.profiles[node_type].decisions_changed += profile.decisions_changed
                combined.profiles[node_type].decisions_tested += profile.decisions_tested
        
        combined.compute_balance()
        return combined


# ---------------------------------------------------------------------------
# Resilience Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvidenceDiversity:
    """How many independent evidence streams contribute to decisions."""
    decision_id: str
    tick: int
    
    # Unique source types that contributed
    source_types_used: Set[NodeType] = field(default_factory=set)
    
    # Count of independent evidence streams
    stream_count: int = 0
    
    # Diversity score (0=单一来源, 1=many independent sources)
    diversity_score: float = 0.0
    
    # Shannon entropy of source distribution
    shannon_entropy: float = 0.0
    
    def compute(self, tree: ReasoningTree):
        """Compute diversity from a reasoning tree."""
        if not tree.nodes:
            return
        
        source_types = {
            NodeType.OBSERVATION, NodeType.CULTURAL_PRIOR, NodeType.TRUST,
            NodeType.PREDICTION, NodeType.DISCOVERY, NodeType.HYPOTHESIS,
        }
        
        self.source_types_used = {
            n.node_type for n in tree.nodes.values()
            if n.node_type in source_types
        }
        
        self.stream_count = len(self.source_types_used)
        
        # Normalize by max possible sources
        self.diversity_score = self.stream_count / len(source_types) if source_types else 0
        
        # Shannon entropy
        import math
        type_counts = {}
        for n in tree.nodes.values():
            if n.node_type in source_types:
                type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        
        total = sum(type_counts.values())
        if total > 0:
            probs = [c / total for c in type_counts.values()]
            self.shannon_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "tick": self.tick,
            "source_types_used": [s.value for s in self.source_types_used],
            "stream_count": self.stream_count,
            "diversity_score": round(self.diversity_score, 3),
            "shannon_entropy": round(self.shannon_entropy, 3),
        }


@dataclass
class CorrectionLatency:
    """How many cycles elapse between error detection and correction."""
    # Event sequence
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Latency measurements
    error_detection_tick: Optional[int] = None
    trust_adjustment_tick: Optional[int] = None
    behavioral_change_tick: Optional[int] = None
    
    # Computed latencies
    detection_to_trust: int = 0  # ticks from error to trust adjustment
    trust_to_behavior: int = 0   # ticks from trust adjustment to behavior change
    total_correction_latency: int = 0  # total ticks from error to correction
    
    def record_error(self, tick: int, data: Dict[str, Any]):
        """Record an error detection event."""
        self.error_detection_tick = tick
        self.events.append({"type": "error", "tick": tick, "data": data})
    
    def record_trust_adjustment(self, tick: int, data: Dict[str, Any]):
        """Record a trust adjustment."""
        self.trust_adjustment_tick = tick
        self.events.append({"type": "trust_adjustment", "tick": tick, "data": data})
        
        if self.error_detection_tick is not None:
            self.detection_to_trust = tick - self.error_detection_tick
    
    def record_behavioral_change(self, tick: int, data: Dict[str, Any]):
        """Record a behavioral change."""
        self.behavioral_change_tick = tick
        self.events.append({"type": "behavioral_change", "tick": tick, "data": data})
        
        if self.trust_adjustment_tick is not None:
            self.trust_to_behavior = tick - self.trust_adjustment_tick
        
        if self.error_detection_tick is not None:
            self.total_correction_latency = tick - self.error_detection_tick
    
    @property
    def is_complete(self) -> bool:
        """Whether all three events have been recorded."""
        return all(t is not None for t in [
            self.error_detection_tick,
            self.trust_adjustment_tick,
            self.behavioral_change_tick,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_detection_tick": self.error_detection_tick,
            "trust_adjustment_tick": self.trust_adjustment_tick,
            "behavioral_change_tick": self.behavioral_change_tick,
            "detection_to_trust": self.detection_to_trust,
            "trust_to_behavior": self.trust_to_behavior,
            "total_correction_latency": self.total_correction_latency,
            "is_complete": self.is_complete,
        }


@dataclass
class EpistemicInertia:
    """How resistant is a belief to change?"""
    belief_id: str
    initial_confidence: float = 0.0
    
    # History of confidence changes
    confidence_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Contradictory evidence encountered
    contradictions: int = 0
    
    # Times confidence decreased
    decreases: int = 0
    
    # Times confidence increased
    increases: int = 0
    
    # Average magnitude of change
    avg_change_magnitude: float = 0.0
    
    # Inertia score (0=very fluid, 1=very rigid)
    inertia_score: float = 0.0
    
    def record_confidence(self, tick: int, confidence: float,
                          evidence_type: str = "unknown"):
        """Record a confidence observation."""
        change = 0.0
        if self.confidence_history:
            prev = self.confidence_history[-1]["confidence"]
            change = confidence - prev
            
            if change < -0.01:
                self.decreases += 1
                self.contradictions += 1
            elif change > 0.01:
                self.increases += 1
        
        self.confidence_history.append({
            "tick": tick,
            "confidence": confidence,
            "change": change,
            "evidence_type": evidence_type,
        })
        
        # Update average magnitude
        if self.confidence_history:
            changes = [abs(e["change"]) for e in self.confidence_history]
            self.avg_change_magnitude = sum(changes) / len(changes)
        
        # Inertia = 1 - average change magnitude
        self.inertia_score = max(0, min(1, 1 - self.avg_change_magnitude * 10))
    
    @property
    def current_confidence(self) -> float:
        """Current confidence value."""
        if self.confidence_history:
            return self.confidence_history[-1]["confidence"]
        return self.initial_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "initial_confidence": self.initial_confidence,
            "current_confidence": self.current_confidence,
            "contradictions": self.contradictions,
            "decreases": self.decreases,
            "increases": self.increases,
            "avg_change_magnitude": round(self.avg_change_magnitude, 4),
            "inertia_score": round(self.inertia_score, 3),
            "history_length": len(self.confidence_history),
        }


@dataclass
class RecoveryTime:
    """Measure recovery after disruption."""
    disruption_tick: int = 0
    
    # Recovery phases
    recognition_tick: Optional[int] = None  # when system notices something wrong
    adjustment_tick: Optional[int] = None   # when trust/weights adjust
    stability_tick: Optional[int] = None    # when metrics return to baseline
    
    # Recovery metrics
    recognition_latency: int = 0
    adjustment_latency: int = 0
    full_recovery_latency: int = 0
    
    # Whether recovery succeeded
    recovered: bool = False
    
    # Confidence trajectory after disruption
    confidence_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_disruption(self, tick: int, description: str = ""):
        """Record the disruption event."""
        self.disruption_tick = tick
        self.confidence_trajectory.append({
            "tick": tick,
            "phase": "disruption",
            "description": description,
        })
    
    def record_recognition(self, tick: int, description: str = ""):
        """Record when system recognizes the problem."""
        self.recognition_tick = tick
        self.recognition_latency = tick - self.disruption_tick
        self.confidence_trajectory.append({
            "tick": tick,
            "phase": "recognition",
            "description": description,
        })
    
    def record_adjustment(self, tick: int, description: str = ""):
        """Record when system begins adjusting."""
        self.adjustment_tick = tick
        self.adjustment_latency = tick - self.disruption_tick
        self.confidence_trajectory.append({
            "tick": tick,
            "phase": "adjustment",
            "description": description,
        })
    
    def record_stability(self, tick: int, description: str = ""):
        """Record when system returns to stable operation."""
        self.stability_tick = tick
        self.full_recovery_latency = tick - self.disruption_tick
        self.recovered = True
        self.confidence_trajectory.append({
            "tick": tick,
            "phase": "stability",
            "description": description,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "disruption_tick": self.disruption_tick,
            "recognition_tick": self.recognition_tick,
            "adjustment_tick": self.adjustment_tick,
            "stability_tick": self.stability_tick,
            "recognition_latency": self.recognition_latency,
            "adjustment_latency": self.adjustment_latency,
            "full_recovery_latency": self.full_recovery_latency,
            "recovered": self.recovered,
            "trajectory_length": len(self.confidence_trajectory),
        }


class ResilienceAnalyzer:
    """Analyze resilience of the epistemic architecture.
    
    Combines EvidenceDiversity, CorrectionLatency, EpistemicInertia,
    and RecoveryTime to measure how well the system handles
    misinformation, incorrect priors, and environmental changes.
    
    Usage:
        analyzer = ResilienceAnalyzer(replay)
        
        # Analyze a single decision
        report = analyzer.analyze_decision(decision_id)
        
        # Analyze recovery from a false discovery
        recovery = analyzer.measure_recovery(discovery_id)
    """
    
    def __init__(self, replay: CausalReplay):
        self._replay = replay
    
    def analyze_diversity(self, decision_id: str) -> EvidenceDiversity:
        """Measure evidence diversity for a decision."""
        tree = self._replay.expand_decision(decision_id)
        if not tree:
            return EvidenceDiversity(decision_id=decision_id, tick=0)
        
        div = EvidenceDiversity(
            decision_id=decision_id,
            tick=tree.decision_tick,
        )
        div.compute(tree)
        return div
    
    def analyze_inertia(self, node_id: str, belief_id: str) -> EpistemicInertia:
        """Measure epistemic inertia for a belief."""
        inertia = EpistemicInertia(belief_id=belief_id)
        
        # Find all nodes related to this belief
        for eid, edata in self._replay._events.items():
            if node_id in str(edata.get("data", {})):
                tick = edata["tick"]
                data = edata.get("data", {})
                
                # Extract confidence
                confidence = data.get("confidence", data.get("level", 0.5))
                evidence_type = edata.get("node_type", "unknown")
                
                inertia.record_confidence(tick, confidence, str(evidence_type))
        
        return inertia
    
    def analyze_correction(self, error_tick: int,
                           events: List[Dict[str, Any]]) -> CorrectionLatency:
        """Measure correction latency from a sequence of events."""
        latency = CorrectionLatency()
        
        for event in events:
            tick = event.get("tick", 0)
            event_type = event.get("type", "")
            
            if event_type == "error":
                latency.record_error(tick, event.get("data", {}))
            elif event_type == "trust_adjustment":
                latency.record_trust_adjustment(tick, event.get("data", {}))
            elif event_type == "behavioral_change":
                latency.record_behavioral_change(tick, event.get("data", {}))
        
        return latency
    
    def analyze_recovery(self, disruption_tick: int,
                         recovery_events: List[Dict[str, Any]]) -> RecoveryTime:
        """Measure recovery time from disruption events."""
        recovery = RecoveryTime()
        recovery.record_disruption(disruption_tick, "False discovery injected")
        
        for event in recovery_events:
            tick = event.get("tick", 0)
            phase = event.get("phase", "")
            description = event.get("description", "")
            
            if phase == "recognition":
                recovery.record_recognition(tick, description)
            elif phase == "adjustment":
                recovery.record_adjustment(tick, description)
            elif phase == "stability":
                recovery.record_stability(tick, description)
        
        return recovery


# ---------------------------------------------------------------------------
# Epistemic Plasticity
# ---------------------------------------------------------------------------

@dataclass
class PlasticityRecord:
    """A single belief change event."""
    tick: int
    belief_id: str
    confidence_before: float
    confidence_after: float
    evidence_type: str
    evidence_strength: float = 0.0
    
    @property
    def change(self) -> float:
        return self.confidence_after - self.confidence_before
    
    @property
    def abs_change(self) -> float:
        return abs(self.change)
    
    @property
    def is_increase(self) -> bool:
        return self.change > 0.01
    
    @property
    def is_decrease(self) -> bool:
        return self.change < -0.01
    
    @property
    def is_stable(self) -> bool:
        return abs(self.change) <= 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "belief_id": self.belief_id,
            "confidence_before": round(self.confidence_before, 3),
            "confidence_after": round(self.confidence_after, 3),
            "change": round(self.change, 3),
            "evidence_type": self.evidence_type,
            "evidence_strength": round(self.evidence_strength, 3),
        }


@dataclass
class PlasticityProfile:
    """Complete plasticity profile for a belief."""
    belief_id: str
    
    # All change events
    records: List[PlasticityRecord] = field(default_factory=list)
    
    # Aggregate metrics
    total_changes: int = 0
    increases: int = 0
    decreases: int = 0
    stable: int = 0
    
    # Magnitude metrics
    avg_abs_change: float = 0.0
    max_increase: float = 0.0
    max_decrease: float = 0.0
    
    # Plasticity score (0=rigid, 1=fluid)
    plasticity_score: float = 0.0
    
    # Response ratio (fraction of evidence that causes change)
    response_ratio: float = 0.0
    
    # Asymmetric response (does evidence type matter?)
    observation_response: float = 0.0
    prior_response: float = 0.0
    trust_response: float = 0.0
    
    def compute(self):
        """Compute aggregate metrics from records."""
        if not self.records:
            return
        
        self.total_changes = len(self.records)
        self.increases = sum(1 for r in self.records if r.is_increase)
        self.decreases = sum(1 for r in self.records if r.is_decrease)
        self.stable = sum(1 for r in self.records if r.is_stable)
        
        if self.total_changes > 0:
            self.avg_abs_change = sum(r.abs_change for r in self.records) / self.total_changes
            self.response_ratio = (self.total_changes - self.stable) / self.total_changes
        
        if self.records:
            self.max_increase = max(r.change for r in self.records)
            self.max_decrease = min(r.change for r in self.records)
        
        # Plasticity score: combines response ratio and average change magnitude
        # Low response ratio + low change = rigid
        # High response ratio + high change = fluid
        self.plasticity_score = min(1.0, self.response_ratio * self.avg_abs_change * 10)
        
        # Asymmetric response by evidence type
        obs_records = [r for r in self.records if r.evidence_type == "observation"]
        prior_records = [r for r in self.records if r.evidence_type == "cultural_prior"]
        trust_records = [r for r in self.records if r.evidence_type == "trust"]
        
        if obs_records:
            self.observation_response = sum(r.abs_change for r in obs_records) / len(obs_records)
        if prior_records:
            self.prior_response = sum(r.abs_change for r in prior_records) / len(prior_records)
        if trust_records:
            self.trust_response = sum(r.abs_change for r in trust_records) / len(trust_records)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "total_changes": self.total_changes,
            "increases": self.increases,
            "decreases": self.decreases,
            "stable": self.stable,
            "avg_abs_change": round(self.avg_abs_change, 4),
            "max_increase": round(self.max_increase, 4),
            "max_decrease": round(self.max_decrease, 4),
            "plasticity_score": round(self.plasticity_score, 3),
            "response_ratio": round(self.response_ratio, 3),
            "observation_response": round(self.observation_response, 4),
            "prior_response": round(self.prior_response, 4),
            "trust_response": round(self.trust_response, 4),
        }


@dataclass
class PlasticityReport:
    """Aggregate plasticity across all beliefs."""
    profiles: Dict[str, PlasticityProfile] = field(default_factory=dict)
    
    # System-wide metrics
    system_plasticity: float = 0.0  # average across all beliefs
    system_response_ratio: float = 0.0
    total_beliefs: int = 0
    
    # Operating region
    operating_region: str = "unknown"  # rigid, optimal, fluid, chaotic
    
    def compute(self):
        """Compute system-wide metrics."""
        if not self.profiles:
            return
        
        self.total_beliefs = len(self.profiles)
        
        plasticities = [p.plasticity_score for p in self.profiles.values()]
        response_ratios = [p.response_ratio for p in self.profiles.values()]
        
        self.system_plasticity = sum(plasticities) / len(plasticities)
        self.system_response_ratio = sum(response_ratios) / len(response_ratios)
        
        # Classify operating region
        if self.system_plasticity < 0.1:
            self.operating_region = "rigid"
        elif self.system_plasticity < 0.3:
            self.operating_region = "optimal"
        elif self.system_plasticity < 0.6:
            self.operating_region = "fluid"
        else:
            self.operating_region = "chaotic"
    
    def render(self) -> str:
        """Render plasticity report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Plasticity Report")
        lines.append("=" * 60)
        lines.append(f"Total beliefs tracked: {self.total_beliefs}")
        lines.append(f"System plasticity: {self.system_plasticity:.3f}")
        lines.append(f"System response ratio: {self.system_response_ratio:.1%}")
        lines.append(f"Operating region: {self.operating_region}")
        lines.append("")
        
        # Per-belief profiles
        lines.append("Belief Profiles")
        lines.append("-" * 60)
        
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.plasticity_score,
            reverse=True
        )
        
        for profile in sorted_profiles:
            bar_len = int(profile.plasticity_score * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            lines.append(
                f"  {profile.belief_id:20s} "
                f"{bar} "
                f"P={profile.plasticity_score:.2f} "
                f"R={profile.response_ratio:.0%}"
            )
        
        lines.append("")
        
        # Interpretation
        lines.append("Interpretation")
        lines.append("-" * 60)
        
        if self.operating_region == "rigid":
            lines.append("  RIGID: System resists change. May become dogmatic.")
            lines.append("  Consider: increase learning rate or trust sensitivity.")
        elif self.operating_region == "optimal":
            lines.append("  OPTIMAL: System balances stability and adaptation.")
            lines.append("  Beliefs change when evidence warrants, but not too easily.")
        elif self.operating_region == "fluid":
            lines.append("  FLUID: System changes easily. May be unstable.")
            lines.append("  Consider: increase inertia or decrease trust sensitivity.")
        elif self.operating_region == "chaotic":
            lines.append("  CHAOTIC: System changes too rapidly. No long-term memory.")
            lines.append("  Consider: add dampening or minimum evidence thresholds.")
        
        lines.append("")
        
        # Asymmetric response
        lines.append("Asymmetric Response")
        lines.append("-" * 60)
        
        avg_obs = sum(p.observation_response for p in self.profiles.values()) / max(1, self.total_beliefs)
        avg_prior = sum(p.prior_response for p in self.profiles.values()) / max(1, self.total_beliefs)
        avg_trust = sum(p.trust_response for p in self.profiles.values()) / max(1, self.total_beliefs)
        
        lines.append(f"  Observation response: {avg_obs:.4f}")
        lines.append(f"  Prior response:       {avg_prior:.4f}")
        lines.append(f"  Trust response:       {avg_trust:.4f}")
        lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_plasticity": round(self.system_plasticity, 3),
            "system_response_ratio": round(self.system_response_ratio, 3),
            "operating_region": self.operating_region,
            "total_beliefs": self.total_beliefs,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
        }


class EpistemicPlasticityAnalyzer:
    """Measure how willing the system is to change its beliefs.
    
    Tracks confidence changes over time and computes:
    - Plasticity score per belief
    - System-wide plasticity
    - Operating region (rigid/optimal/fluid/chaotic)
    - Asymmetric response to different evidence types
    
    Usage:
        analyzer = EpistemicPlasticityAnalyzer(replay)
        
        # Track belief changes
        analyzer.record_change(tick, "rule_1", 0.8, 0.6, "observation", 0.9)
        
        # Get report
        report = analyzer.get_report()
        print(report.render())
    """
    
    def __init__(self, replay: CausalReplay):
        self._replay = replay
        self._profiles: Dict[str, PlasticityProfile] = {}
        self._all_records: List[PlasticityRecord] = []
    
    def record_change(self, tick: int, belief_id: str,
                      confidence_before: float, confidence_after: float,
                      evidence_type: str = "unknown",
                      evidence_strength: float = 0.0):
        """Record a belief change event."""
        record = PlasticityRecord(
            tick=tick,
            belief_id=belief_id,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            evidence_type=evidence_type,
            evidence_strength=evidence_strength,
        )
        
        self._all_records.append(record)
        
        if belief_id not in self._profiles:
            self._profiles[belief_id] = PlasticityProfile(belief_id=belief_id)
        
        self._profiles[belief_id].records.append(record)
    
    def get_report(self) -> PlasticityReport:
        """Get complete plasticity report."""
        report = PlasticityReport(profiles=dict(self._profiles))
        
        for profile in report.profiles.values():
            profile.compute()
        
        report.compute()
        return report
    
    def get_time_series(self, belief_id: str) -> List[Dict[str, Any]]:
        """Get confidence time series for a belief."""
        if belief_id not in self._profiles:
            return []
        
        return [
            {
                "tick": r.tick,
                "confidence_before": r.confidence_before,
                "confidence_after": r.confidence_after,
                "change": r.change,
                "evidence_type": r.evidence_type,
            }
            for r in self._profiles[belief_id].records
        ]


# ---------------------------------------------------------------------------
# Epistemic Dynamics Laboratory
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """State of a single agent in the dynamics simulation."""
    agent_id: str
    
    # Beliefs
    beliefs: Dict[str, float] = field(default_factory=dict)  # belief_id -> confidence
    
    # Trust in other agents
    trust: Dict[str, float] = field(default_factory=dict)  # agent_id -> trust
    
    # Domain expertise
    domains: Dict[str, float] = field(default_factory=dict)  # domain -> expertise
    
    # Metrics
    total_observations: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "beliefs": {k: round(v, 3) for k, v in self.beliefs.items()},
            "trust_count": len(self.trust),
            "domain_count": len(self.domains),
            "total_observations": self.total_observations,
            "accuracy": round(self.accuracy, 3),
        }


@dataclass
class DynamicsSnapshot:
    """Snapshot of the entire swarm at a point in time."""
    tick: int
    
    # Agent states
    agents: List[AgentState] = field(default_factory=list)
    
    # Aggregate metrics
    avg_plasticity: float = 0.0
    avg_accuracy: float = 0.0
    belief_entropy: float = 0.0  # how diverse are beliefs
    trust_entropy: float = 0.0  # how evenly distributed is trust
    consensus_level: float = 0.0  # how much agreement exists
    
    # Misinformation metrics
    false_belief_prevalence: float = 0.0
    correction_rate: float = 0.0
    
    def compute(self):
        """Compute aggregate metrics from agent states."""
        if not self.agents:
            return
        
        self.avg_accuracy = sum(a.accuracy for a in self.agents) / len(self.agents)
        
        # Belief entropy
        all_beliefs = {}
        for agent in self.agents:
            for belief_id, confidence in agent.beliefs.items():
                if belief_id not in all_beliefs:
                    all_beliefs[belief_id] = []
                all_beliefs[belief_id].append(confidence)
        
        if all_beliefs:
            import math
            entropies = []
            for belief_id, confidences in all_beliefs.items():
                mean_conf = sum(confidences) / len(confidences)
                variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
                # Higher variance = higher entropy (more disagreement)
                entropies.append(variance)
            self.belief_entropy = sum(entropies) / len(entropies)
        
        # Consensus level (1 - belief_entropy, normalized)
        self.consensus_level = max(0, min(1, 1 - self.belief_entropy * 10))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "agent_count": len(self.agents),
            "avg_plasticity": round(self.avg_plasticity, 3),
            "avg_accuracy": round(self.avg_accuracy, 3),
            "belief_entropy": round(self.belief_entropy, 4),
            "consensus_level": round(self.consensus_level, 3),
            "false_belief_prevalence": round(self.false_belief_prevalence, 3),
        }


class EpistemicDynamicsLab:
    """Large-scale simulation of epistemic dynamics.
    
    Runs 50-1000 agents in a simplified environment to study:
    - Misinformation spread
    - Trust dynamics
    - Consensus formation
    - Domain specialization
    - Recovery from false beliefs
    
    Usage:
        lab = EpistemicDynamicsLab(num_agents=100)
        
        # Run simulation
        for tick in range(1000):
            lab.tick(tick)
        
        # Get results
        report = lab.get_report()
    """
    
    def __init__(self, num_agents: int = 100, num_domains: int = 5):
        self.num_agents = num_agents
        self.num_domains = num_domains
        
        # Initialize agents
        self.agents: List[AgentState] = [
            AgentState(agent_id=f"agent_{i}")
            for i in range(num_agents)
        ]
        
        # Initialize trust (sparse network)
        import random
        for agent in self.agents:
            # Each agent trusts 5-15 others
            num_trusted = random.randint(5, min(15, num_agents - 1))
            others = [a for a in self.agents if a.agent_id != agent.agent_id]
            trusted = random.sample(others, num_trusted)
            for t in trusted:
                agent.trust[t.agent_id] = random.uniform(0.5, 0.9)
        
        # True rules (what should be believed)
        self.true_rules: Dict[str, bool] = {
            f"rule_{d}": True for d in range(num_domains)
        }
        
        # False rules (misinformation)
        self.false_rules: Dict[str, bool] = {}
        
        # Time series
        self._snapshots: List[DynamicsSnapshot] = []
        self._plasticity_analyzer = EpistemicPlasticityAnalyzer(CausalReplay())
        
        # Event log
        self._events: List[Dict[str, Any]] = []
    
    def tick(self, tick: int):
        """Simulate one tick of dynamics."""
        import random
        
        for agent in self.agents:
            # 1. Observe (some agents see truth, some see lies)
            if random.random() < 0.7:  # 70% see truth
                self._agent_observe_truth(agent, tick)
            else:
                self._agent_observe_noise(agent, tick)
            
            # 2. Update beliefs based on observations
            self._agent_update_beliefs(agent, tick)
            
            # 3. Exchange with trusted agents
            if random.random() < 0.3:  # 30% chance to exchange
                self._agent_exchange(agent, tick)
            
            # 4. Make predictions
            self._agent_predict(agent, tick)
        
        # Take snapshot
        snapshot = self._take_snapshot(tick)
        self._snapshots.append(snapshot)
    
    def inject_false_belief(self, tick: int, rule_id: str,
                            confidence: float = 0.9,
                            num_agents: int = 10):
        """Inject a false belief into some agents."""
        import random
        
        self.false_rules[rule_id] = False
        
        targets = random.sample(self.agents, min(num_agents, len(self.agents)))
        for agent in targets:
            old_conf = agent.beliefs.get(rule_id, 0.5)
            agent.beliefs[rule_id] = confidence
            
            self._plasticity_analyzer.record_change(
                tick, rule_id, old_conf, confidence,
                "injection", confidence
            )
            
            self._events.append({
                "tick": tick,
                "type": "false_injection",
                "agent": agent.agent_id,
                "rule": rule_id,
                "confidence": confidence,
            })
    
    def _agent_observe_truth(self, agent: AgentState, tick: int):
        """Agent observes the truth."""
        import random
        
        agent.total_observations += 1
        
        for rule_id, is_true in self.true_rules.items():
            if rule_id not in agent.beliefs:
                agent.beliefs[rule_id] = 0.5
            
            # Observation pushes toward truth
            current = agent.beliefs[rule_id]
            if is_true:
                new_conf = current + random.uniform(0.05, 0.15)
            else:
                new_conf = current - random.uniform(0.05, 0.15)
            
            new_conf = max(0.01, min(0.99, new_conf))
            
            if abs(new_conf - current) > 0.01:
                self._plasticity_analyzer.record_change(
                    tick, rule_id, current, new_conf,
                    "observation", 0.8
                )
            
            agent.beliefs[rule_id] = new_conf
    
    def _agent_observe_noise(self, agent: AgentState, tick: int):
        """Agent observes noisy/misleading data."""
        import random
        
        agent.total_observations += 1
        
        # Random noise
        for rule_id in list(agent.beliefs.keys()):
            current = agent.beliefs[rule_id]
            noise = random.uniform(-0.1, 0.1)
            new_conf = max(0.01, min(0.99, current + noise))
            
            if abs(new_conf - current) > 0.01:
                self._plasticity_analyzer.record_change(
                    tick, rule_id, current, new_conf,
                    "noise", 0.3
                )
            
            agent.beliefs[rule_id] = new_conf
    
    def _agent_update_beliefs(self, agent: AgentState, tick: int):
        """Agent updates beliefs based on internal consistency."""
        import random
        
        for rule_id in list(agent.beliefs.keys()):
            current = agent.beliefs[rule_id]
            
            # If confidence is very high, small adjustments
            # If confidence is moderate, larger adjustments possible
            adjustment = random.uniform(-0.02, 0.02) * (1 - abs(current - 0.5) * 2)
            
            new_conf = max(0.01, min(0.99, current + adjustment))
            
            if abs(new_conf - current) > 0.005:
                self._plasticity_analyzer.record_change(
                    tick, rule_id, current, new_conf,
                    "internal", 0.2
                )
            
            agent.beliefs[rule_id] = new_conf
    
    def _agent_exchange(self, agent: AgentState, tick: int):
        """Agent exchanges beliefs with trusted peers."""
        import random
        
        if not agent.trust:
            return
        
        # Pick a random trusted agent
        peer_id = random.choice(list(agent.trust.keys()))
        peer = next((a for a in self.agents if a.agent_id == peer_id), None)
        
        if not peer:
            return
        
        trust_level = agent.trust[peer_id]
        
        # Blend beliefs based on trust
        for rule_id in set(list(agent.beliefs.keys()) + list(peer.beliefs.keys())):
            if rule_id not in agent.beliefs:
                agent.beliefs[rule_id] = peer.beliefs.get(rule_id, 0.5)
                continue
            if rule_id not in peer.beliefs:
                continue
            
            my_conf = agent.beliefs[rule_id]
            peer_conf = peer.beliefs[rule_id]
            
            # Weighted average
            new_conf = my_conf * (1 - trust_level * 0.3) + peer_conf * trust_level * 0.3
            new_conf = max(0.01, min(0.99, new_conf))
            
            if abs(new_conf - my_conf) > 0.005:
                self._plasticity_analyzer.record_change(
                    tick, rule_id, my_conf, new_conf,
                    "trust", trust_level
                )
            
            agent.beliefs[rule_id] = new_conf
    
    def _agent_predict(self, agent: AgentState, tick: int):
        """Agent makes a prediction based on beliefs."""
        agent.total_predictions += 1
        
        # Simple: if belief matches truth, correct prediction
        correct = True
        for rule_id, is_true in self.true_rules.items():
            conf = agent.beliefs.get(rule_id, 0.5)
            predicted_true = conf > 0.5
            if predicted_true != is_true:
                correct = False
                break
        
        if correct:
            agent.correct_predictions += 1
    
    def _take_snapshot(self, tick: int) -> DynamicsSnapshot:
        """Take a snapshot of the swarm state."""
        snapshot = DynamicsSnapshot(tick=tick, agents=list(self.agents))
        snapshot.compute()
        return snapshot
    
    def get_report(self) -> Dict[str, Any]:
        """Get complete dynamics report."""
        if not self._snapshots:
            return {"error": "No snapshots"}
        
        # Time series
        accuracy_ts = [s.avg_accuracy for s in self._snapshots]
        entropy_ts = [s.belief_entropy for s in self._snapshots]
        consensus_ts = [s.consensus_level for s in self._snapshots]
        
        # Plasticity report
        plasticity = self._plasticity_analyzer.get_report()
        
        return {
            "num_agents": self.num_agents,
            "num_domains": self.num_domains,
            "total_ticks": len(self._snapshots),
            "accuracy_trajectory": accuracy_ts,
            "entropy_trajectory": entropy_ts,
            "consensus_trajectory": consensus_ts,
            "final_accuracy": accuracy_ts[-1] if accuracy_ts else 0,
            "final_consensus": consensus_ts[-1] if consensus_ts else 0,
            "plasticity": plasticity.to_dict(),
            "false_rules_injected": len(self.false_rules),
            "total_events": len(self._events),
        }
    
    def render_time_series(self) -> str:
        """Render time series as ASCII plot."""
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Dynamics Time Series")
        lines.append("=" * 60)
        lines.append("")
        
        # Accuracy
        lines.append("Accuracy Over Time")
        lines.append("-" * 60)
        self._render_ascii_plot(
            lines,
            [s.avg_accuracy for s in self._snapshots],
            "Accuracy", 0.0, 1.0
        )
        lines.append("")
        
        # Belief Entropy
        lines.append("Belief Entropy Over Time")
        lines.append("-" * 60)
        self._render_ascii_plot(
            lines,
            [s.belief_entropy for s in self._snapshots],
            "Entropy", 0.0, 0.5
        )
        lines.append("")
        
        # Consensus
        lines.append("Consensus Over Time")
        lines.append("-" * 60)
        self._render_ascii_plot(
            lines,
            [s.consensus_level for s in self._snapshots],
            "Consensus", 0.0, 1.0
        )
        lines.append("")
        
        return "\n".join(lines)
    
    def _render_ascii_plot(self, lines: List[str], values: List[float],
                           label: str, min_val: float, max_val: float,
                           height: int = 15, width: int = 50):
        """Render an ASCII plot."""
        if not values:
            return
        
        # Downsample to width
        step = max(1, len(values) // width)
        sampled = values[::step][:width]
        
        # Normalize to 0-1
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in sampled]
        else:
            normalized = [0.5] * len(sampled)
        
        # Render
        for row in range(height, -1, -1):
            threshold = row / height
            line = f"  {max_val * threshold:5.2f} |"
            for val in normalized:
                if val >= threshold:
                    line += "#"
                else:
                    line += " "
            lines.append(line)
        
        lines.append(f"        +{'-' * len(sampled)}")
        lines.append(f"         Tick 0{' ' * (len(sampled) - 6)}Tick {len(values)}")


# ---------------------------------------------------------------------------
# Knowledge Classes & Differential Plasticity
# ---------------------------------------------------------------------------

class KnowledgeClass(Enum):
    """Classes of knowledge with different plasticity targets."""
    FOUNDATIONAL = "foundational"  # core rules, high evidence required
    OPERATIONAL = "operational"    # working knowledge, moderate evidence
    EXPLORATORY = "exploratory"    # hypotheses, high plasticity


@dataclass
class KnowledgeClassConfig:
    """Configuration for a knowledge class."""
    knowledge_class: KnowledgeClass
    plasticity_target: float  # target plasticity (0=rigid, 1=fluid)
    evidence_threshold: float  # min evidence to change belief
    decay_rate: float  # how fast belief decays without reinforcement
    
    # Derived
    stability_score: float = 0.0  # 1 - plasticity_target
    
    def __post_init__(self):
        self.stability_score = 1.0 - self.plasticity_target


# Default configurations
DEFAULT_CLASS_CONFIGS = {
    KnowledgeClass.FOUNDATIONAL: KnowledgeClassConfig(
        knowledge_class=KnowledgeClass.FOUNDATIONAL,
        plasticity_target=0.1,
        evidence_threshold=0.7,
        decay_rate=0.001,
    ),
    KnowledgeClass.OPERATIONAL: KnowledgeClassConfig(
        knowledge_class=KnowledgeClass.OPERATIONAL,
        plasticity_target=0.4,
        evidence_threshold=0.4,
        decay_rate=0.01,
    ),
    KnowledgeClass.EXPLORATORY: KnowledgeClassConfig(
        knowledge_class=KnowledgeClass.EXPLORATORY,
        plasticity_target=0.8,
        evidence_threshold=0.2,
        decay_rate=0.05,
    ),
}


@dataclass
class KnowledgeItem:
    """A single piece of knowledge with class assignment."""
    item_id: str
    knowledge_class: KnowledgeClass
    confidence: float = 0.5
    
    # Source
    source_agent: str = ""
    source_tick: int = 0
    
    # History
    evidence_count: int = 0
    last_reinforced: int = 0
    
    # Predictions
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # Effective plasticity (adjusted by class)
    effective_plasticity: float = 0.0
    
    def __post_init__(self):
        config = DEFAULT_CLASS_CONFIGS.get(
            self.knowledge_class,
            DEFAULT_CLASS_CONFIGS[KnowledgeClass.OPERATIONAL]
        )
        self.effective_plasticity = config.plasticity_target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "knowledge_class": self.knowledge_class.value,
            "confidence": round(self.confidence, 3),
            "effective_plasticity": round(self.effective_plasticity, 3),
            "evidence_count": self.evidence_count,
            "source_agent": self.source_agent,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
        }


class DifferentialPlasticityManager:
    """Manage knowledge with different plasticity by class.
    
    Foundational knowledge requires strong evidence to change.
    Operational knowledge has moderate plasticity.
    Exploratory knowledge changes easily.
    
    Usage:
        manager = DifferentialPlasticityManager()
        
        # Add knowledge
        manager.add_item("rule_core", KnowledgeClass.FOUNDATIONAL, 0.9)
        manager.add_item("rule_working", KnowledgeClass.OPERATIONAL, 0.6)
        manager.add_item("hypothesis_new", KnowledgeClass.EXPLORATORY, 0.3)
        
        # Update with evidence
        manager.update_with_evidence("rule_core", 0.8, "observation")
        
        # Get effective plasticity
        plasticity = manager.get_effective_plasticity("rule_core")
    """
    
    def __init__(self, class_configs: Optional[Dict[KnowledgeClass, KnowledgeClassConfig]] = None):
        self._configs = class_configs or dict(DEFAULT_CLASS_CONFIGS)
        self._items: Dict[str, KnowledgeItem] = {}
        self._history: List[Dict[str, Any]] = []
    
    def add_item(self, item_id: str, knowledge_class: KnowledgeClass,
                 confidence: float = 0.5, source_agent: str = "",
                 source_tick: int = 0) -> KnowledgeItem:
        """Add a knowledge item."""
        item = KnowledgeItem(
            item_id=item_id,
            knowledge_class=knowledge_class,
            confidence=confidence,
            source_agent=source_agent,
            source_tick=source_tick,
        )
        
        self._items[item_id] = item
        return item
    
    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item."""
        return self._items.get(item_id)
    
    def get_effective_plasticity(self, item_id: str) -> float:
        """Get effective plasticity for a knowledge item."""
        item = self._items.get(item_id)
        if not item:
            return 0.5  # default
        
        config = self._configs.get(item.knowledge_class)
        if not config:
            return 0.5
        
        # Base plasticity from class
        base = config.plasticity_target
        
        # Adjust by evidence count (more evidence = less plastic)
        evidence_factor = max(0.5, 1.0 - item.evidence_count * 0.05)
        
        # Adjust by confidence (higher confidence = less plastic)
        confidence_factor = 1.0 - item.confidence * 0.3
        
        return base * evidence_factor * confidence_factor
    
    def update_with_evidence(self, item_id: str, evidence_strength: float,
                             evidence_type: str = "observation",
                             current_tick: int = 0) -> Optional[float]:
        """Update knowledge item with evidence. Returns new confidence."""
        item = self._items.get(item_id)
        if not item:
            return None
        
        config = self._configs.get(item.knowledge_class)
        if not config:
            return None
        
        # Check evidence threshold
        if evidence_strength < config.evidence_threshold:
            # Evidence too weak for this class
            self._history.append({
                "tick": current_tick,
                "item_id": item_id,
                "action": "rejected",
                "reason": f"evidence {evidence_strength:.2f} < threshold {config.evidence_threshold:.2f}",
            })
            return item.confidence
        
        # Calculate change
        effective_plasticity = self.get_effective_plasticity(item_id)
        change = evidence_strength * effective_plasticity
        
        # Apply change (positive evidence increases, negative decreases)
        old_confidence = item.confidence
        item.confidence = max(0.01, min(0.99, item.confidence + change))
        item.evidence_count += 1
        item.last_reinforced = current_tick
        
        self._history.append({
            "tick": current_tick,
            "item_id": item_id,
            "action": "updated",
            "old_confidence": old_confidence,
            "new_confidence": item.confidence,
            "change": item.confidence - old_confidence,
            "evidence_type": evidence_type,
            "effective_plasticity": effective_plasticity,
        })
        
        return item.confidence
    
    def apply_decay(self, current_tick: int, ticks_elapsed: int = 1):
        """Apply natural decay to all knowledge items."""
        for item in self._items.values():
            config = self._configs.get(item.knowledge_class)
            if not config:
                continue
            
            # Decay proportional to time and class decay rate
            decay = config.decay_rate * ticks_elapsed
            item.confidence = max(0.01, item.confidence - decay)
    
    def get_report(self) -> Dict[str, Any]:
        """Get report of all knowledge items."""
        by_class = {}
        for item in self._items.values():
            cls = item.knowledge_class.value
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(item.to_dict())
        
        return {
            "total_items": len(self._items),
            "by_class": {
                cls: {
                    "count": len(items),
                    "avg_confidence": sum(i["confidence"] for i in items) / len(items) if items else 0,
                    "avg_plasticity": sum(i["effective_plasticity"] for i in items) / len(items) if items else 0,
                }
                for cls, items in by_class.items()
            },
            "items": by_class,
        }


# ---------------------------------------------------------------------------
# Epistemic Half-Life
# ---------------------------------------------------------------------------

@dataclass
class HalfLifeRecord:
    """Record of belief decay after contradictory evidence."""
    item_id: str
    evidence_tick: int
    confidence_at_evidence: float
    
    # Decay trajectory
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    # Half-life measurement
    half_life_ticks: Optional[int] = None  # ticks to reach 50% of original
    quarter_life_ticks: Optional[int] = None  # ticks to reach 25%
    fully_rejected: bool = False
    rejection_tick: Optional[int] = None
    
    @property
    def confidence_at_half(self) -> float:
        return self.confidence_at_evidence * 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "evidence_tick": self.evidence_tick,
            "confidence_at_evidence": round(self.confidence_at_evidence, 3),
            "half_life_ticks": self.half_life_ticks,
            "quarter_life_ticks": self.quarter_life_ticks,
            "fully_rejected": self.fully_rejected,
            "rejection_tick": self.rejection_tick,
            "trajectory_length": len(self.trajectory),
        }


class EpistemicHalfLifeAnalyzer:
    """Measure how long beliefs persist after contradictory evidence.
    
    Definition:
        How long does it take for a belief to lose 50% of its influence
        after contradictory evidence appears?
    
    Usage:
        analyzer = EpistemicHalfLifeAnalyzer()
        
        # Record evidence event
        analyzer.record_evidence(item_id, tick, confidence=0.9)
        
        # Record confidence updates
        analyzer.record_confidence(item_id, tick, confidence=0.7)
        analyzer.record_confidence(item_id, tick, confidence=0.5)
        
        # Get half-life
        half_life = analyzer.get_half_life(item_id)
    """
    
    def __init__(self):
        self._records: Dict[str, HalfLifeRecord] = {}
        self._confidences: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_evidence(self, item_id: str, tick: int, confidence: float):
        """Record contradictory evidence event."""
        self._records[item_id] = HalfLifeRecord(
            item_id=item_id,
            evidence_tick=tick,
            confidence_at_evidence=confidence,
        )
        
        if item_id not in self._confidences:
            self._confidences[item_id] = []
        
        self._confidences[item_id].append({
            "tick": tick,
            "confidence": confidence,
            "phase": "evidence",
        })
    
    def record_confidence(self, item_id: str, tick: int, confidence: float):
        """Record a confidence update after evidence."""
        if item_id not in self._records:
            return
        
        record = self._records[item_id]
        
        # Add to trajectory
        record.trajectory.append({
            "tick": tick,
            "confidence": confidence,
        })
        
        # Also track in confidences
        if item_id not in self._confidences:
            self._confidences[item_id] = []
        
        self._confidences[item_id].append({
            "tick": tick,
            "confidence": confidence,
            "phase": "decay",
        })
        
        # Check for half-life
        if record.half_life_ticks is None:
            if confidence <= record.confidence_at_half:
                record.half_life_ticks = tick - record.evidence_tick
        
        # Check for quarter-life
        if record.quarter_life_ticks is None:
            if confidence <= record.confidence_at_evidence * 0.25:
                record.quarter_life_ticks = tick - record.evidence_tick
        
        # Check for full rejection
        if not record.fully_rejected:
            if confidence <= 0.1:
                record.fully_rejected = True
                record.rejection_tick = tick
    
    def get_half_life(self, item_id: str) -> Optional[int]:
        """Get half-life for a belief."""
        record = self._records.get(item_id)
        if not record:
            return None
        return record.half_life_ticks
    
    def get_record(self, item_id: str) -> Optional[HalfLifeRecord]:
        """Get full record for a belief."""
        return self._records.get(item_id)
    
    def get_all_records(self) -> Dict[str, HalfLifeRecord]:
        """Get all records."""
        return dict(self._records)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all half-life measurements."""
        if not self._records:
            return {"total": 0}
        
        half_lives = [
            r.half_life_ticks for r in self._records.values()
            if r.half_life_ticks is not None
        ]
        
        rejected = [
            r for r in self._records.values()
            if r.fully_rejected
        ]
        
        return {
            "total": len(self._records),
            "measured": len(half_lives),
            "avg_half_life": sum(half_lives) / len(half_lives) if half_lives else 0,
            "min_half_life": min(half_lives) if half_lives else 0,
            "max_half_life": max(half_lives) if half_lives else 0,
            "fully_rejected": len(rejected),
            "rejection_rate": len(rejected) / len(self._records) if self._records else 0,
        }
    
    def render(self) -> str:
        """Render half-life report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Half-Life Report")
        lines.append("=" * 60)
        
        summary = self.get_summary()
        lines.append(f"Total beliefs tracked: {summary['total']}")
        lines.append(f"Half-life measured: {summary['measured']}")
        lines.append(f"Average half-life: {summary['avg_half_life']:.1f} ticks")
        lines.append(f"Fully rejected: {summary['fully_rejected']}")
        lines.append(f"Rejection rate: {summary['rejection_rate']:.1%}")
        lines.append("")
        
        # Per-belief details
        lines.append("Belief Details")
        lines.append("-" * 60)
        
        for item_id, record in sorted(self._records.items()):
            half_life = record.half_life_ticks
            rejected = "REJECTED" if record.fully_rejected else "active"
            
            lines.append(
                f"  {item_id:20s} "
                f"half-life={half_life if half_life else 'N/A':>6} "
                f"status={rejected}"
            )
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Knowledge Migration & Phase Transitions
# ---------------------------------------------------------------------------

class MigrationDirection(Enum):
    """Direction of knowledge class migration."""
    PROMOTE = "promote"  # exploratory -> operational -> foundational
    DEMOTE = "demote"    # foundational -> operational -> exploratory


@dataclass
class MigrationEvent:
    """Record of a knowledge class migration."""
    item_id: str
    from_class: KnowledgeClass
    to_class: KnowledgeClass
    tick: int
    direction: MigrationDirection
    
    # Triggers
    confidence_at_migration: float = 0.0
    evidence_count: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # Reason
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "from_class": self.from_class.value,
            "to_class": self.to_class.value,
            "tick": self.tick,
            "direction": self.direction.value,
            "confidence": round(self.confidence_at_migration, 3),
            "evidence_count": self.evidence_count,
            "reason": self.reason,
        }


@dataclass
class MigrationTrigger:
    """Conditions that trigger a class migration."""
    # Promotion triggers (exploratory -> operational -> foundational)
    min_confidence: float = 0.7
    min_evidence_count: int = 5
    min_successful_predictions: int = 3
    min_ticks_in_class: int = 50
    
    # Demotion triggers (foundational -> operational -> exploratory)
    max_confidence: float = 0.3
    max_failed_predictions: int = 3
    min_anomalies: int = 2
    
    # Domain-specific adjustments
    domain_difficulty_multiplier: float = 1.0


class KnowledgeMigrationSystem:
    """Manage knowledge migration between classes.
    
    Knowledge can move between classes based on evidence:
    - Promote: exploratory -> operational -> foundational
    - Demote: foundational -> operational -> exploratory
    
    Usage:
        system = KnowledgeMigrationSystem()
        
        # Add knowledge
        system.add_item("rule_new", KnowledgeClass.EXPLORATORY, 0.3)
        
        # Update with evidence
        system.update_with_evidence("rule_new", 0.8, "observation", tick=100)
        
        # Check for migrations
        migrations = system.check_migrations(tick=150)
        
        # Get migration history
        history = system.get_migration_history("rule_new")
    """
    
    def __init__(self, triggers: Optional[MigrationTrigger] = None):
        self._triggers = triggers or MigrationTrigger()
        self._items: Dict[str, KnowledgeItem] = {}
        self._migrations: List[MigrationEvent] = []
        self._ticks_in_class: Dict[str, int] = {}  # item_id -> ticks since last migration
    
    def add_item(self, item_id: str, knowledge_class: KnowledgeClass,
                 confidence: float = 0.5, source_agent: str = "",
                 source_tick: int = 0) -> KnowledgeItem:
        """Add a knowledge item."""
        item = KnowledgeItem(
            item_id=item_id,
            knowledge_class=knowledge_class,
            confidence=confidence,
            source_agent=source_agent,
            source_tick=source_tick,
        )
        
        self._items[item_id] = item
        self._ticks_in_class[item_id] = 0
        return item
    
    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item."""
        return self._items.get(item_id)
    
    def update_with_evidence(self, item_id: str, evidence_strength: float,
                             evidence_type: str = "observation",
                             current_tick: int = 0,
                             is_successful: bool = True) -> Optional[float]:
        """Update knowledge item with evidence."""
        item = self._items.get(item_id)
        if not item:
            return None
        
        # Update confidence
        old_confidence = item.confidence
        item.confidence = max(0.01, min(0.99, item.confidence + evidence_strength))
        item.evidence_count += 1
        item.last_reinforced = current_tick
        
        # Track predictions
        if evidence_type == "prediction":
            if is_successful:
                item.successful_predictions += 1
            else:
                item.failed_predictions += 1
        
        return item.confidence
    
    def check_migrations(self, current_tick: int) -> List[MigrationEvent]:
        """Check all items for migration eligibility."""
        migrations = []
        
        for item_id, item in self._items.items():
            # Update ticks in class
            self._ticks_in_class[item_id] = self._ticks_in_class.get(item_id, 0) + 1
            
            # Check promotion
            promotion = self._check_promotion(item, current_tick)
            if promotion:
                migrations.append(promotion)
                continue  # Only one migration per tick
            
            # Check demotion
            demotion = self._check_demotion(item, current_tick)
            if demotion:
                migrations.append(demotion)
        
        return migrations
    
    def _check_promotion(self, item: KnowledgeItem,
                         current_tick: int) -> Optional[MigrationEvent]:
        """Check if item should be promoted."""
        triggers = self._triggers
        
        # Determine target class
        if item.knowledge_class == KnowledgeClass.EXPLORATORY:
            target = KnowledgeClass.OPERATIONAL
        elif item.knowledge_class == KnowledgeClass.OPERATIONAL:
            target = KnowledgeClass.FOUNDATIONAL
        else:
            return None  # Already foundational
        
        # Check conditions
        if item.confidence < triggers.min_confidence:
            return None
        if item.evidence_count < triggers.min_evidence_count:
            return None
        if item.successful_predictions < triggers.min_successful_predictions:
            return None
        if self._ticks_in_class.get(item.item_id, 0) < triggers.min_ticks_in_class:
            return None
        
        # Promote
        old_class = item.knowledge_class
        item.knowledge_class = target
        
        # Update plasticity
        config = DEFAULT_CLASS_CONFIGS.get(target)
        if config:
            item.effective_plasticity = config.plasticity_target
        
        # Record migration
        event = MigrationEvent(
            item_id=item.item_id,
            from_class=old_class,
            to_class=target,
            tick=current_tick,
            direction=MigrationDirection.PROMOTE,
            confidence_at_migration=item.confidence,
            evidence_count=item.evidence_count,
            successful_predictions=item.successful_predictions,
            reason=f"Promoted: confidence={item.confidence:.2f}, "
                   f"evidence={item.evidence_count}, "
                   f"successes={item.successful_predictions}",
        )
        
        self._migrations.append(event)
        self._ticks_in_class[item.item_id] = 0
        
        return event
    
    def _check_demotion(self, item: KnowledgeItem,
                        current_tick: int) -> Optional[MigrationEvent]:
        """Check if item should be demoted."""
        triggers = self._triggers
        
        # Determine target class
        if item.knowledge_class == KnowledgeClass.FOUNDATIONAL:
            target = KnowledgeClass.OPERATIONAL
        elif item.knowledge_class == KnowledgeClass.OPERATIONAL:
            target = KnowledgeClass.EXPLORATORY
        else:
            return None  # Already exploratory
        
        # Check conditions
        if item.confidence > triggers.max_confidence:
            return None
        if item.failed_predictions < triggers.max_failed_predictions:
            return None
        
        # Demote
        old_class = item.knowledge_class
        item.knowledge_class = target
        
        # Update plasticity
        config = DEFAULT_CLASS_CONFIGS.get(target)
        if config:
            item.effective_plasticity = config.plasticity_target
        
        # Record migration
        event = MigrationEvent(
            item_id=item.item_id,
            from_class=old_class,
            to_class=target,
            tick=current_tick,
            direction=MigrationDirection.DEMOTE,
            confidence_at_migration=item.confidence,
            evidence_count=item.evidence_count,
            failed_predictions=item.failed_predictions,
            reason=f"Demoted: confidence={item.confidence:.2f}, "
                   f"failures={item.failed_predictions}",
        )
        
        self._migrations.append(event)
        self._ticks_in_class[item.item_id] = 0
        
        return event
    
    def get_migration_history(self, item_id: str) -> List[MigrationEvent]:
        """Get migration history for an item."""
        return [m for m in self._migrations if m.item_id == item_id]
    
    def get_all_migrations(self) -> List[MigrationEvent]:
        """Get all migration events."""
        return list(self._migrations)
    
    def get_report(self) -> Dict[str, Any]:
        """Get migration report."""
        by_class = {}
        for item in self._items.values():
            cls = item.knowledge_class.value
            if cls not in by_class:
                by_class[cls] = 0
            by_class[cls] += 1
        
        promotions = [m for m in self._migrations if m.direction == MigrationDirection.PROMOTE]
        demotions = [m for m in self._migrations if m.direction == MigrationDirection.DEMOTE]
        
        return {
            "total_items": len(self._items),
            "by_class": by_class,
            "total_migrations": len(self._migrations),
            "promotions": len(promotions),
            "demotions": len(demotions),
            "migration_history": [m.to_dict() for m in self._migrations],
        }
    
    def render(self) -> str:
        """Render migration report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Knowledge Migration Report")
        lines.append("=" * 60)
        
        report = self.get_report()
        
        lines.append(f"Total items: {report['total_items']}")
        lines.append(f"By class: {report['by_class']}")
        lines.append(f"Total migrations: {report['total_migrations']}")
        lines.append(f"  Promotions: {report['promotions']}")
        lines.append(f"  Demotions: {report['demotions']}")
        lines.append("")
        
        # Migration history
        lines.append("Migration History")
        lines.append("-" * 60)
        
        for event in self._migrations[-10:]:  # Last 10
            direction = "UP" if event.direction == MigrationDirection.PROMOTE else "DOWN"
            lines.append(
                f"  Tick {event.tick:4d}: {event.item_id:20s} "
                f"{event.from_class.value:12s} -> {event.to_class.value:12s} "
                f"[{direction}]"
            )
            lines.append(f"           {event.reason}")
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Epistemic Roles (Domain-Dependent)
# ---------------------------------------------------------------------------

@dataclass
class EpistemicRole:
    """Domain-dependent role for a knowledge item."""
    item_id: str
    domain: str
    knowledge_class: KnowledgeClass
    confidence: float = 0.5
    
    # Domain-specific metrics
    domain_evidence_count: int = 0
    domain_success_rate: float = 0.0
    
    # Role-specific
    is_core_to_domain: bool = False
    cross_domain_relevance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain,
            "knowledge_class": self.knowledge_class.value,
            "confidence": round(self.confidence, 3),
            "domain_evidence_count": self.domain_evidence_count,
            "domain_success_rate": round(self.domain_success_rate, 3),
            "is_core_to_domain": self.is_core_to_domain,
            "cross_domain_relevance": round(self.cross_domain_relevance, 3),
        }


class EpistemicRoleManager:
    """Manage domain-dependent knowledge roles.
    
    A knowledge item can have different roles in different domains:
    - Foundational in one domain
    - Operational in another
    - Exploratory in a third
    
    Usage:
        manager = EpistemicRoleManager()
        
        # Assign roles
        manager.assign_role("rule_scout", "early_game", KnowledgeClass.OPERATIONAL)
        manager.assign_role("rule_scout", "late_game", KnowledgeClass.EXPLORATORY)
        
        # Get role in domain
        role = manager.get_role("rule_scout", "early_game")
        
        # Get all roles for an item
        roles = manager.get_all_roles("rule_scout")
    """
    
    def __init__(self):
        self._roles: Dict[str, Dict[str, EpistemicRole]] = {}  # item_id -> domain -> role
    
    def assign_role(self, item_id: str, domain: str,
                    knowledge_class: KnowledgeClass,
                    confidence: float = 0.5) -> EpistemicRole:
        """Assign a role to a knowledge item in a domain."""
        if item_id not in self._roles:
            self._roles[item_id] = {}
        
        role = EpistemicRole(
            item_id=item_id,
            domain=domain,
            knowledge_class=knowledge_class,
            confidence=confidence,
        )
        
        self._roles[item_id][domain] = role
        return role
    
    def get_role(self, item_id: str, domain: str) -> Optional[EpistemicRole]:
        """Get role for an item in a domain."""
        return self._roles.get(item_id, {}).get(domain)
    
    def get_all_roles(self, item_id: str) -> List[EpistemicRole]:
        """Get all roles for an item across domains."""
        return list(self._roles.get(item_id, {}).values())
    
    def get_domain_roles(self, domain: str) -> List[EpistemicRole]:
        """Get all roles in a domain."""
        roles = []
        for item_roles in self._roles.values():
            if domain in item_roles:
                roles.append(item_roles[domain])
        return roles
    
    def get_dominant_class(self, item_id: str) -> Optional[KnowledgeClass]:
        """Get the dominant class across all domains."""
        roles = self.get_all_roles(item_id)
        if not roles:
            return None
        
        # Count by class
        class_counts = {}
        for role in roles:
            cls = role.knowledge_class
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Return most common
        return max(class_counts, key=class_counts.get)
    
    def get_report(self) -> Dict[str, Any]:
        """Get role report."""
        by_domain = {}
        for item_id, domain_roles in self._roles.items():
            for domain, role in domain_roles.items():
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(role.to_dict())
        
        return {
            "total_items": len(self._roles),
            "domains": list(by_domain.keys()),
            "by_domain": by_domain,
        }
    
    def render(self) -> str:
        """Render role report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Role Report")
        lines.append("=" * 60)
        
        report = self.get_report()
        
        lines.append(f"Total items: {report['total_items']}")
        lines.append(f"Domains: {report['domains']}")
        lines.append("")
        
        # By domain
        for domain, roles in report["by_domain"].items():
            lines.append(f"Domain: {domain}")
            lines.append("-" * 60)
            
            for role in sorted(roles, key=lambda r: r["knowledge_class"]):
                lines.append(
                    f"  {role['item_id']:20s} "
                    f"{role['knowledge_class']:12s} "
                    f"conf={role['confidence']:.2f}"
                )
            
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Epistemic Governance
# ---------------------------------------------------------------------------

class GovernanceRuleType(Enum):
    """Types of governance rules."""
    PROMOTION = "promotion"
    DEMOTION = "demotion"
    COMPETITION = "competition"
    MERGER = "merger"
    RETIREMENT = "retirement"


@dataclass
class GovernanceRule:
    """A rule governing knowledge evolution."""
    rule_id: str
    rule_type: GovernanceRuleType
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "conditions": self.conditions,
            "actions": self.actions,
            "priority": self.priority,
            "enabled": self.enabled,
        }


class KnowledgeCompetitionResult(Enum):
    """Result of knowledge competition."""
    SURVIVE = "survive"
    MERGE = "merge"
    SPLIT = "split"
    RETIRE = "retire"
    SPECIALIZE = "specialize"


@dataclass
class CompetitionEvent:
    """Record of a knowledge competition."""
    tick: int
    competing_items: List[str]
    result: KnowledgeCompetitionResult
    survivors: List[str] = field(default_factory=list)
    retired: List[str] = field(default_factory=list)
    merged_into: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "competing_items": self.competing_items,
            "result": self.result.value,
            "survivors": self.survivors,
            "retired": self.retired,
            "merged_into": self.merged_into,
            "reason": self.reason,
        }


class EpistemicGovernance:
    """Rules governing knowledge evolution.

    Defines how knowledge gets promoted, competes, merges, or is retired.
    """

    def __init__(self):
        self._rules: List[GovernanceRule] = []
        self._competition_events: List[CompetitionEvent] = []
        self._retired_items: List[str] = []

    def add_rule(self, rule: GovernanceRule):
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def get_rules(self, rule_type: Optional[GovernanceRuleType] = None) -> List[GovernanceRule]:
        if rule_type:
            return [r for r in self._rules if r.rule_type == rule_type and r.enabled]
        return [r for r in self._rules if r.enabled]

    def check_governance(self, item: KnowledgeItem,
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Check governance rules for an item."""
        decisions = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            conditions_met = True
            for key, value in rule.conditions.items():
                if key == "min_confidence" and item.confidence < value:
                    conditions_met = False
                    break
                if key == "min_evidence" and item.evidence_count < value:
                    conditions_met = False
                    break
                if key == "min_predictions" and item.successful_predictions < value:
                    conditions_met = False
                    break
                if key == "max_failures" and item.failed_predictions > value:
                    conditions_met = False
                    break
            if conditions_met:
                decisions.append({
                    "rule": rule.rule_id,
                    "type": rule.rule_type.value,
                    "actions": rule.actions,
                })
        return {"item_id": item.item_id, "decisions": decisions}

    def compete(self, items: List[KnowledgeItem], tick: int,
                context: Dict[str, Any]) -> CompetitionEvent:
        """Run competition between knowledge items."""
        if len(items) < 2:
            return CompetitionEvent(
                tick=tick,
                competing_items=[i.item_id for i in items],
                result=KnowledgeCompetitionResult.SURVIVE,
                survivors=[i.item_id for i in items],
            )

        sorted_items = sorted(items, key=lambda i: i.confidence, reverse=True)

        merger_rule = next(
            (r for r in self._rules if r.rule_type == GovernanceRuleType.MERGER), None
        )
        if merger_rule and len(sorted_items) >= 2:
            top2 = sorted_items[:2]
            if abs(top2[0].confidence - top2[1].confidence) < 0.2:
                merged_id = f"merged_{top2[0].item_id}_{top2[1].item_id}"
                event = CompetitionEvent(
                    tick=tick,
                    competing_items=[i.item_id for i in items],
                    result=KnowledgeCompetitionResult.MERGE,
                    survivors=[merged_id],
                    retired=[top2[1].item_id],
                    merged_into=merged_id,
                    reason=f"Merged {top2[0].item_id} and {top2[1].item_id}",
                )
                self._competition_events.append(event)
                return event

        retirement_rule = next(
            (r for r in self._rules if r.rule_type == GovernanceRuleType.RETIREMENT), None
        )
        if retirement_rule:
            threshold = retirement_rule.conditions.get("min_confidence", 0.2)
            survivors = [i for i in sorted_items if i.confidence >= threshold]
            retired = [i for i in sorted_items if i.confidence < threshold]
            if retired:
                event = CompetitionEvent(
                    tick=tick,
                    competing_items=[i.item_id for i in items],
                    result=KnowledgeCompetitionResult.RETIRE,
                    survivors=[i.item_id for i in survivors],
                    retired=[i.item_id for i in retired],
                    reason=f"Retired {len(retired)} items below confidence threshold",
                )
                self._competition_events.append(event)
                self._retired_items.extend([i.item_id for i in retired])
                return event

        event = CompetitionEvent(
            tick=tick,
            competing_items=[i.item_id for i in items],
            result=KnowledgeCompetitionResult.SURVIVE,
            survivors=[i.item_id for i in items],
        )
        self._competition_events.append(event)
        return event

    def get_competition_history(self) -> List[CompetitionEvent]:
        return list(self._competition_events)

    def get_retired_items(self) -> List[str]:
        return list(self._retired_items)

    def get_report(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for rule in self._rules:
            t = rule.rule_type.value
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_rules": len(self._rules),
            "by_type": by_type,
            "competition_events": len(self._competition_events),
            "retired_items": len(self._retired_items),
        }

    def render(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Governance Report")
        lines.append("=" * 60)
        report = self.get_report()
        lines.append(f"Total rules: {report['total_rules']}")
        lines.append(f"By type: {report['by_type']}")
        lines.append(f"Competition events: {report['competition_events']}")
        lines.append(f"Retired items: {report['retired_items']}")
        lines.append("")
        lines.append("Competition History")
        lines.append("-" * 60)
        for event in self._competition_events[-5:]:
            lines.append(
                f"  Tick {event.tick}: {event.result.value} "
                f"({len(event.competing_items)} items)"
            )
            lines.append(f"    Survivors: {event.survivors}")
            if event.retired:
                lines.append(f"    Retired: {event.retired}")
            if event.merged_into:
                lines.append(f"    Merged into: {event.merged_into}")
            lines.append(f"    Reason: {event.reason}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Epistemic Age
# ---------------------------------------------------------------------------

@dataclass
class EpistemicAgeProfile:
    """Epistemic age profile for a knowledge item."""
    item_id: str
    birth_tick: int = 0
    total_ticks: int = 0
    ticks_survived: int = 0
    successful_predictions: int = 0
    independent_confirmations: int = 0
    contradictions_survived: int = 0
    domains_active: int = 0
    current_class: KnowledgeClass = KnowledgeClass.EXPLORATORY
    epistemic_age: float = 0.0

    def compute_age(self):
        prediction_score = self.successful_predictions * 0.3
        confirmation_score = self.independent_confirmations * 0.2
        contradiction_score = self.contradictions_survived * 0.4
        domain_score = self.domains_active * 0.1
        class_multiplier = {
            KnowledgeClass.EXPLORATORY: 1.0,
            KnowledgeClass.OPERATIONAL: 1.5,
            KnowledgeClass.FOUNDATIONAL: 2.0,
        }.get(self.current_class, 1.0)
        survival_ratio = self.ticks_survived / max(1, self.total_ticks)
        self.epistemic_age = (
            (prediction_score + confirmation_score + contradiction_score + domain_score)
            * class_multiplier * survival_ratio
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "birth_tick": self.birth_tick,
            "total_ticks": self.total_ticks,
            "ticks_survived": self.ticks_survived,
            "successful_predictions": self.successful_predictions,
            "independent_confirmations": self.independent_confirmations,
            "contradictions_survived": self.contradictions_survived,
            "domains_active": self.domains_active,
            "current_class": self.current_class.value,
            "epistemic_age": round(self.epistemic_age, 3),
        }


class EpistemicAgeAnalyzer:
    """Measure how long beliefs have survived and under what conditions.

    Epistemic Age combines predictions, confirmations, contradictions survived,
    and domain coverage.
    """

    def __init__(self):
        self._profiles: Dict[str, EpistemicAgeProfile] = {}

    def track_item(self, item_id: str, birth_tick: int = 0,
                   initial_class: KnowledgeClass = KnowledgeClass.EXPLORATORY):
        self._profiles[item_id] = EpistemicAgeProfile(
            item_id=item_id, birth_tick=birth_tick, current_class=initial_class,
        )

    def update_ticks(self, item_id: str, current_tick: int):
        if item_id in self._profiles:
            p = self._profiles[item_id]
            p.total_ticks = current_tick - p.birth_tick
            p.ticks_survived += 1

    def record_prediction(self, item_id: str, successful: bool = True):
        if item_id in self._profiles and successful:
            self._profiles[item_id].successful_predictions += 1

    def record_confirmation(self, item_id: str):
        if item_id in self._profiles:
            self._profiles[item_id].independent_confirmations += 1

    def record_contradiction(self, item_id: str, survived: bool = True):
        if item_id in self._profiles and survived:
            self._profiles[item_id].contradictions_survived += 1

    def update_class(self, item_id: str, new_class: KnowledgeClass):
        if item_id in self._profiles:
            self._profiles[item_id].current_class = new_class

    def update_domains(self, item_id: str, num_domains: int):
        if item_id in self._profiles:
            self._profiles[item_id].domains_active = num_domains

    def get_age_profile(self, item_id: str) -> Optional[EpistemicAgeProfile]:
        profile = self._profiles.get(item_id)
        if profile:
            profile.compute_age()
        return profile

    def get_all_profiles(self) -> List[EpistemicAgeProfile]:
        for profile in self._profiles.values():
            profile.compute_age()
        return list(self._profiles.values())

    def get_report(self) -> Dict[str, Any]:
        profiles = self.get_all_profiles()
        if not profiles:
            return {"total": 0}
        ages = [p.epistemic_age for p in profiles]
        return {
            "total": len(profiles),
            "avg_age": sum(ages) / len(ages),
            "max_age": max(ages),
            "min_age": min(ages),
            "profiles": {p.item_id: p.to_dict() for p in profiles},
        }

    def render(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Age Report")
        lines.append("=" * 60)
        report = self.get_report()
        lines.append(f"Total items: {report['total']}")
        lines.append(f"Average age: {report['avg_age']:.3f}")
        lines.append(f"Max age: {report['max_age']:.3f}")
        lines.append(f"Min age: {report['min_age']:.3f}")
        lines.append("")
        lines.append("Item Profiles")
        lines.append("-" * 60)
        sorted_profiles = sorted(
            self._profiles.values(), key=lambda p: p.epistemic_age, reverse=True
        )
        for profile in sorted_profiles:
            bar_len = int(min(1.0, profile.epistemic_age / 10) * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            lines.append(
                f"  {profile.item_id:20s} "
                f"{bar} "
                f"age={profile.epistemic_age:.2f} "
                f"class={profile.current_class.value}"
            )
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
