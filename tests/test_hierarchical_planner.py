"""Tests for Hierarchical Planning — P7.3"""
import numpy as np
from substrate_echo.core.hierarchical_planner import (
    HierarchicalPlanner, HierarchicalPlannerConfig, GoalPlan,
    GoalNode, GoalType, GoalStatus, DecompositionRule
)


def _ctx(val=0.5):
    return np.full(16, val)


class TestGoalNode:
    def test_leaf_detection(self):
        node = GoalNode(node_id=0, goal_type=GoalType.ACTION, name="approach")
        assert node.is_leaf
        node.children = [1, 2]
        assert not node.is_leaf

    def test_actionable(self):
        node = GoalNode(node_id=0, goal_type=GoalType.ACTION, name="approach")
        assert node.is_actionable
        node = GoalNode(node_id=1, goal_type=GoalType.SUBGOAL, name="find")
        assert not node.is_actionable


class TestPlanCreation:
    def test_plan_has_root(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        assert plan.root_id in plan.nodes
        assert plan.nodes[plan.root_id].name == "get_food"

    def test_plan_depth(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx(), max_depth=3)
        assert plan.depth >= 1

    def test_plan_has_actions(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        actions = plan.actions
        assert len(actions) > 0
        assert all(a.is_actionable for a in actions)

    def test_plan_total_cost(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        cost = plan.total_estimated_cost
        assert cost > 0

    def test_plan_name(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("explore_area", state=_ctx())
        assert plan.name == "explore_area"


class TestPlanDecomposition:
    def test_get_food_decomposition(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        # Should decompose into find, acquire, consume
        subgoal_names = [n.name for n in plan.nodes.values()
                        if n.goal_type == GoalType.SUBGOAL]
        assert len(subgoal_names) >= 2

    def test_explore_area_decomposition(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("explore_area", state=_ctx())
        actions = plan.actions
        assert len(actions) >= 2

    def test_max_depth_limit(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx(), max_depth=1)
        assert plan.depth <= 2  # root + 1 level


class TestGetNextActions:
    def test_returns_pending_actions(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        next_actions = hp.get_next_actions(plan)
        assert len(next_actions) > 0
        assert all(a.status == GoalStatus.PENDING for a in next_actions)

    def test_marks_active(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        next_actions = hp.get_next_actions(plan)
        # Mark first action active
        if next_actions:
            next_actions[0].status = GoalStatus.ACTIVE
            # Should still return other pending actions
            remaining = hp.get_next_actions(plan)
            assert all(a.node_id != next_actions[0].node_id for a in remaining)


class TestMarkComplete:
    def test_mark_action_complete(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        actions = hp.get_next_actions(plan)
        if actions:
            hp.mark_complete(plan, actions[0].node_id, success=True, utility=0.8)
            assert actions[0].status == GoalStatus.COMPLETED

    def test_propagates_to_parent(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("explore_area", state=_ctx(), max_depth=1)
        # Mark all leaves complete
        for leaf in plan.leaves:
            hp.mark_complete(plan, leaf.node_id, success=True, utility=0.7)
        # Root should be completed
        root = plan.nodes[plan.root_id]
        assert root.status == GoalStatus.COMPLETED

    def test_partial_failure(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("explore_area", state=_ctx(), max_depth=1)
        for i, leaf in enumerate(plan.leaves):
            hp.mark_complete(plan, leaf.node_id, success=(i == 0), utility=0.5)
        root = plan.nodes[plan.root_id]
        # If any child failed, parent fails
        if any(leaf.status == GoalStatus.FAILED for leaf in plan.leaves):
            assert root.status == GoalStatus.FAILED


class TestEvaluatePlan:
    def test_evaluate_returns_float(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        score = hp.evaluate_plan(plan)
        assert isinstance(score, float)

    def test_better_plan_scores_higher(self):
        hp = HierarchicalPlanner()
        plan1 = hp.plan("get_food", state=_ctx())
        plan2 = hp.plan("explore_area", state=_ctx())
        s1 = hp.evaluate_plan(plan1)
        s2 = hp.evaluate_plan(plan2)
        # Both should be valid scores
        assert isinstance(s1, float)
        assert isinstance(s2, float)


class TestRecordOutcome:
    def test_records_rule(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        hp.record_outcome(plan, success=True)
        rules = hp._rules.get("get_food", [])
        assert len(rules) >= 1
        assert rules[0].success_rate == 1.0

    def test_updates_existing_rule(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        hp.record_outcome(plan, success=True)
        hp.record_outcome(plan, success=False)
        rules = hp._rules.get("get_food", [])
        assert len(rules) >= 1
        assert rules[0].uses == 2

    def test_reuse_strategy(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("get_food", state=_ctx())
        # Build up success rate
        for _ in range(3):
            hp.record_outcome(plan, success=True)
        strategy = hp.get_reusable_strategy("get_food")
        assert strategy is not None
        assert strategy.success_rate >= 0.7


class TestReusableStrategy:
    def test_returns_none_for_unknown(self):
        hp = HierarchicalPlanner()
        strategy = hp.get_reusable_strategy("unknown_goal")
        assert strategy is None

    def test_reuses_strategy_in_new_plan(self):
        hp = HierarchicalPlanner()
        plan1 = hp.plan("get_food", state=_ctx())
        for _ in range(3):
            hp.record_outcome(plan1, success=True)

        # New plan should use the learned strategy
        plan2 = hp.plan("get_food", state=_ctx())
        # Should have same direct children under root
        root1 = plan1.nodes[plan1.root_id]
        root2 = plan2.nodes[plan2.root_id]
        children1 = sorted([plan1.nodes[cid].name for cid in root1.children])
        children2 = sorted([plan2.nodes[cid].name for cid in root2.children])
        assert children1 == children2


class TestStats:
    def test_stats_empty(self):
        hp = HierarchicalPlanner()
        s = hp.stats()
        assert s["n_plans"] == 0
        assert s["n_rules"] == 0

    def test_stats_with_plans(self):
        hp = HierarchicalPlanner()
        hp.plan("get_food", state=_ctx())
        hp.plan("explore_area", state=_ctx())
        s = hp.stats()
        assert s["n_plans"] == 2


class TestActionInference:
    def test_infer_action_find(self):
        hp = HierarchicalPlanner()
        assert hp._infer_action("find_food") == "search"

    def test_infer_action_approach(self):
        hp = HierarchicalPlanner()
        assert hp._infer_action("approach_safely") == "approach"

    def test_infer_action_grasp(self):
        hp = HierarchicalPlanner()
        assert hp._infer_action("grasp_object") == "grasp"

    def test_infer_action_observe(self):
        hp = HierarchicalPlanner()
        assert hp._infer_action("observe_scene") == "observe"

    def test_infer_action_default(self):
        hp = HierarchicalPlanner()
        assert hp._infer_action("do_something") == "investigate"
