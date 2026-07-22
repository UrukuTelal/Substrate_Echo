"""Hierarchical Planning — P7.3

Goal → Subgoal → Action decomposition with reusable strategies.

Core idea: the planner doesn't just choose actions — it decomposes
high-level goals into subgoals, subgoals into sub-subgoals, until
reaching concrete actions. Each level can be evaluated independently,
and successful decompositions become reusable strategies.

This enables:
- Long-horizon planning without exponential search
- Reusable strategies across different goals
- Abstraction: "find food" doesn't care about specific food type
- Transfer: "approach object" strategy works for food, tools, allies
- Counterfactual evaluation at each level

Architecture:

    Goal: "Get food"
    ├── Subgoal: "Find food source"
    │   ├── Action: SEARCH(region)
    │   └── Action: IDENTIFY(edible)
    ├── Subgoal: "Acquire food"
    │   ├── Action: APPROACH(food)
    │   └── Action: GRASP(food)
    └── Subgoal: "Consume"
        └── Action: EAT(food)

Usage:
    hp = HierarchicalPlanner(counterfactual=cr)

    # Plan a goal
    plan = hp.plan("get_food", state=current_state)

    # Get next concrete actions
    actions = hp.get_next_actions(plan)

    # Evaluate plan quality
    score = hp.evaluate_plan(plan)

    # Learn from successful plans
    hp.record_outcome(plan, success=True)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict
from enum import Enum


class GoalStatus(Enum):
    """Status of a goal node in the hierarchy."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class GoalType(Enum):
    """Type of goal node."""
    ROOT = "root"          # top-level goal
    SUBGOAL = "subgoal"    # intermediate decomposition
    ACTION = "action"      # concrete executable action


@dataclass
class GoalNode:
    """A node in the goal hierarchy."""
    node_id: int
    goal_type: GoalType
    name: str
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING

    # Hierarchy
    parent_id: Optional[int] = None
    children: list[int] = field(default_factory=list)

    # Execution
    action_type: Optional[str] = None   # for ACTION nodes
    action_params: dict = field(default_factory=dict)
    estimated_cost: float = 1.0
    estimated_utility: float = 0.5

    # Evaluation
    actual_cost: float = 0.0
    actual_utility: float = 0.0
    success: Optional[bool] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        return 0  # computed by planner

    @property
    def total_estimated_cost(self) -> float:
        return self.estimated_cost

    @property
    def is_actionable(self) -> bool:
        return self.goal_type == GoalType.ACTION


@dataclass
class GoalPlan:
    """A complete goal hierarchy."""
    plan_id: int
    root_id: int
    nodes: dict[int, GoalNode] = field(default_factory=dict)
    name: str = ""
    tick_created: int = 0

    @property
    def all_nodes(self) -> list[GoalNode]:
        return list(self.nodes.values())

    @property
    def leaves(self) -> list[GoalNode]:
        return [n for n in self.nodes.values() if n.is_leaf]

    @property
    def actions(self) -> list[GoalNode]:
        return [n for n in self.nodes.values() if n.is_actionable]

    @property
    def depth(self) -> int:
        if not self.nodes:
            return 0
        max_d = 0
        for node in self.nodes.values():
            d = 0
            current = node
            while current.parent_id is not None and current.parent_id in self.nodes:
                current = self.nodes[current.parent_id]
                d += 1
            max_d = max(max_d, d)
        return max_d

    @property
    def total_estimated_cost(self) -> float:
        return sum(n.estimated_cost for n in self.leaves)

    @property
    def completion_fraction(self) -> float:
        completed = sum(1 for n in self.nodes.values()
                       if n.status == GoalStatus.COMPLETED)
        return completed / max(1, len(self.nodes))


@dataclass
class DecompositionRule:
    """A learned rule for decomposing a goal into subgoals."""
    goal_name: str
    subgoal_names: list[str]
    subgoal_actions: list[str]  # action type for each subgoal
    success_rate: float = 0.5
    uses: int = 0
    avg_utility: float = 0.0


@dataclass
class HierarchicalPlannerConfig:
    """Configuration for hierarchical planning."""
    max_depth: int = 4
    max_branching: int = 4
    max_plans: int = 50
    planning_budget: int = 20  # max nodes to evaluate
    reuse_threshold: float = 0.6  # min success rate to reuse decomposition
    cost_discount: float = 0.8  # cost increases less at deeper levels


class HierarchicalPlanner:
    """Decomposes goals into subgoals and actions.

    Plans are hierarchies of GoalNodes. The planner decomposes
    high-level goals into subgoals, then subgoals into concrete
    actions. Successful decompositions are remembered and reused.

    Usage:
        hp = HierarchicalPlanner(counterfactual=cr)

        # Create a plan
        plan = hp.plan("get_food", state=current_state)

        # Get executable actions
        next_actions = hp.get_next_actions(plan)

        # Mark actions complete
        hp.mark_complete(plan, node_id, success=True, utility=0.8)

        # Record for learning
        hp.record_outcome(plan, success=True)
    """

    def __init__(self, counterfactual=None,
                 config: Optional[HierarchicalPlannerConfig] = None):
        self.config = config or HierarchicalPlannerConfig()
        self._cr = counterfactual

        self._plans: dict[int, GoalPlan] = {}
        self._next_plan_id = 0
        self._next_node_id = 0

        # Learned decomposition rules
        self._rules: dict[str, list[DecompositionRule]] = defaultdict(list)

    def plan(self, goal_name: str, state: Optional[np.ndarray] = None,
             tick: int = 0, max_depth: Optional[int] = None) -> GoalPlan:
        """Create a hierarchical plan for a goal.

        Args:
            goal_name: what to achieve
            state: current state
            tick: current tick
            max_depth: override max decomposition depth

        Returns:
            GoalPlan with full hierarchy
        """
        if max_depth is None:
            max_depth = self.config.max_depth

        plan_id = self._next_plan_id
        self._next_plan_id += 1

        plan = GoalPlan(
            plan_id=plan_id,
            root_id=0,
            name=goal_name,
            tick_created=tick,
        )

        # Create root node
        root = self._make_node(GoalType.ROOT, goal_name,
                               description=f"Achieve: {goal_name}")
        plan.nodes[root.node_id] = root
        plan.root_id = root.node_id

        # Decompose recursively
        self._decompose(root, plan, state, depth=0, max_depth=max_depth)

        self._plans[plan_id] = plan
        return plan

    def get_next_actions(self, plan: GoalPlan) -> list[GoalNode]:
        """Get the next actionable nodes (depth-first, pending leaves)."""
        actions = []
        for node in plan.nodes.values():
            if (node.is_actionable and
                node.status == GoalStatus.PENDING):
                # Check if all ancestors are active/completed
                if self._ancestors_ready(plan, node):
                    actions.append(node)
        return actions

    def mark_complete(self, plan: GoalPlan, node_id: int,
                      success: bool = True,
                      utility: float = 0.5,
                      cost: float = 1.0) -> None:
        """Mark a node as complete and propagate status."""
        if node_id not in plan.nodes:
            return

        node = plan.nodes[node_id]
        node.status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
        node.success = success
        node.actual_utility = utility
        node.actual_cost = cost

        # Check if parent should be marked complete
        if node.parent_id is not None and node.parent_id in plan.nodes:
            parent = plan.nodes[node.parent_id]
            children = [plan.nodes[cid] for cid in parent.children
                       if cid in plan.nodes]
            if all(c.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)
                   for c in children):
                all_success = all(c.status == GoalStatus.COMPLETED
                                 for c in children)
                parent.status = GoalStatus.COMPLETED if all_success else GoalStatus.FAILED
                parent.actual_utility = np.mean([c.actual_utility for c in children])
                parent.actual_cost = sum(c.actual_cost for c in children)

    def evaluate_plan(self, plan: GoalPlan) -> float:
        """Evaluate plan quality using estimated utilities."""
        if not plan.leaves:
            return 0.0

        utilities = [n.estimated_utility for n in plan.leaves]
        costs = [n.estimated_cost for n in plan.leaves]

        avg_utility = np.mean(utilities)
        total_cost = sum(costs)

        # Utility minus cost, normalized
        score = avg_utility - 0.1 * total_cost
        return float(score)

    def record_outcome(self, plan: GoalPlan, success: bool) -> None:
        """Record plan outcome for learning decomposition rules."""
        root = plan.nodes.get(plan.root_id)
        if root is None:
            return

        # Extract decomposition pattern
        children = [plan.nodes[cid] for cid in root.children
                   if cid in plan.nodes]

        if not children:
            return

        subgoal_names = [c.name for c in children]
        subgoal_actions = [c.action_type or "unknown" for c in children]

        # Check if rule exists
        existing = self._find_rule(root.name, subgoal_names)
        if existing:
            existing.uses += 1
            if success:
                existing.success_rate = (
                    (existing.success_rate * (existing.uses - 1) + 1.0)
                    / existing.uses)
            else:
                existing.success_rate = (
                    existing.success_rate * (existing.uses - 1) / existing.uses)
        else:
            rule = DecompositionRule(
                goal_name=root.name,
                subgoal_names=subgoal_names,
                subgoal_actions=subgoal_actions,
                success_rate=1.0 if success else 0.0,
                uses=1,
            )
            self._rules[root.name].append(rule)

    def get_reusable_strategy(self, goal_name: str) -> Optional[DecompositionRule]:
        """Get a proven decomposition strategy for a goal."""
        rules = self._rules.get(goal_name, [])
        if not rules:
            return None

        # Return best rule with sufficient success rate
        valid = [r for r in rules
                 if r.success_rate >= self.config.reuse_threshold and r.uses >= 2]
        if not valid:
            return None

        return max(valid, key=lambda r: r.success_rate)

    def stats(self) -> dict:
        """Summary statistics."""
        total_rules = sum(len(rules) for rules in self._rules.values())
        proven = sum(1 for rules in self._rules.values()
                    for r in rules if r.success_rate >= 0.7)
        return {
            "n_plans": len(self._plans),
            "n_rules": total_rules,
            "n_proven_rules": proven,
        }

    # ── Private methods ──────────────────────────────────────

    def _decompose(self, node: GoalNode, plan: GoalPlan,
                   state: Optional[np.ndarray],
                   depth: int, max_depth: int) -> None:
        """Recursively decompose a goal into subgoals/actions."""
        if depth >= max_depth:
            # At max depth, make this a concrete action
            node.goal_type = GoalType.ACTION
            node.action_type = self._infer_action(node.name)
            return

        # Check for reusable strategy
        strategy = self.get_reusable_strategy(node.name)
        if strategy:
            self._apply_strategy(node, plan, strategy, state, depth, max_depth)
            return

        # Default decomposition based on goal name
        subgoals = self._decompose_goal(node.name, state)

        if not subgoals:
            # Leaf: make it an action
            node.goal_type = GoalType.ACTION
            node.action_type = self._infer_action(node.name)
            return

        for sg_name, sg_action in subgoals[:self.config.max_branching]:
            child = self._make_node(GoalType.SUBGOAL, sg_name,
                                    parent_id=node.node_id)
            child.action_type = sg_action
            plan.nodes[child.node_id] = child
            node.children.append(child.node_id)

            # Recurse
            self._decompose(child, plan, state, depth + 1, max_depth)

    def _apply_strategy(self, node: GoalNode, plan: GoalPlan,
                        strategy: DecompositionRule,
                        state: Optional[np.ndarray],
                        depth: int, max_depth: int) -> None:
        """Apply a learned decomposition strategy."""
        for sg_name, sg_action in zip(strategy.subgoal_names,
                                       strategy.subgoal_actions):
            child = self._make_node(GoalType.SUBGOAL, sg_name,
                                    parent_id=node.node_id)
            child.action_type = sg_action
            plan.nodes[child.node_id] = child
            node.children.append(child.node_id)

    def _decompose_goal(self, goal_name: str,
                        state: Optional[np.ndarray]) -> list[tuple[str, str]]:
        """Decompose a goal name into (subgoal_name, action_type) pairs."""
        # Common decomposition patterns
        patterns = {
            "get_food": [
                ("find_food_source", "search"),
                ("acquire_food", "grasp"),
                ("consume_food", "eat"),
            ],
            "find_food": [
                ("search_region", "search"),
                ("identify_edible", "identify"),
            ],
            "approach_safely": [
                ("assess_situation", "observe"),
                ("approach", "approach"),
            ],
            "communicate": [
                ("get_attention", "gesture"),
                ("send_message", "speak"),
                ("await_response", "observe"),
            ],
            "explore_area": [
                ("scan_environment", "observe"),
                ("investigate_interesting", "investigate"),
                ("record_findings", "record"),
            ],
            "defend_ally": [
                ("assess_threat", "observe"),
                ("position_defensively", "move"),
                ("protect", "defend"),
            ],
        }

        # Check exact match
        if goal_name in patterns:
            return patterns[goal_name]

        # Check partial match
        for key, decomposition in patterns.items():
            if key in goal_name.lower():
                return decomposition

        # Generic decomposition
        return [
            (f"prepare_{goal_name}", "observe"),
            (f"execute_{goal_name}", self._infer_action(goal_name)),
            (f"verify_{goal_name}", "observe"),
        ]

    def _infer_action(self, goal_name: str) -> str:
        """Infer action type from goal name."""
        name_lower = goal_name.lower()
        if "find" in name_lower or "search" in name_lower:
            return "search"
        elif "approach" in name_lower or "move" in name_lower:
            return "approach"
        elif "grasp" in name_lower or "take" in name_lower:
            return "grasp"
        elif "observe" in name_lower or "look" in name_lower:
            return "observe"
        elif "communicate" in name_lower or "speak" in name_lower:
            return "communicate"
        elif "defend" in name_lower or "protect" in name_lower:
            return "defend"
        elif "eat" in name_lower or "consume" in name_lower:
            return "consume"
        else:
            return "investigate"

    def _make_node(self, goal_type: GoalType, name: str,
                   parent_id: Optional[int] = None,
                   description: str = "") -> GoalNode:
        """Create a new GoalNode with unique ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return GoalNode(
            node_id=node_id,
            goal_type=goal_type,
            name=name,
            description=description,
            parent_id=parent_id,
        )

    def _ancestors_ready(self, plan: GoalPlan, node: GoalNode) -> bool:
        """Check if all ancestors of a node are not failed/blocked.

        PENDING and ACTIVE ancestors are fine — the plan hasn't
        started executing yet, or the parent is in progress.
        """
        current = node
        while current.parent_id is not None:
            if current.parent_id not in plan.nodes:
                return False
            parent = plan.nodes[current.parent_id]
            if parent.status in (GoalStatus.FAILED, GoalStatus.BLOCKED):
                return False
            current = parent
        return True

    def _find_rule(self, goal_name: str,
                   subgoal_names: list[str]) -> Optional[DecompositionRule]:
        """Find matching decomposition rule."""
        rules = self._rules.get(goal_name, [])
        for rule in rules:
            if rule.subgoal_names == subgoal_names:
                return rule
        return None
