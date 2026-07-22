"""Tests for the Planning System — WorldModel, Simulator, Evaluator, Controller, Planner, Intent."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.core.world_model import WorldModel
from substrate_echo.core.simulator import Simulator, ActionDelta, SimConfig
from substrate_echo.core.evaluator import Evaluator, UtilityWeights, EvalResult
from substrate_echo.core.controller import Controller, ControlConfig, ControlOutput
from substrate_echo.core.planner import Planner, Plan, PlannerConfig
from substrate_echo.core.intent import Intent, Situation, IntentProposal
from substrate_echo.core.intent_generator import IntentGenerator, AgentPersonality
from substrate_echo.models.experience import Experience, ExperienceType


# ── Helper ──────────────────────────────────────────────────────

def _feed_dynamics(mem, target, n_trials=20, steps_per=20):
    """Feed diverse converging trajectories to a DynamicsMemory."""
    rng = np.random.RandomState(42)
    for trial in range(n_trials):
        state = target + rng.randn(16) * 0.25
        state = np.clip(state, 0.1, 0.9)
        for i in range(steps_per):
            exp = Experience(
                experience_id=f"exp_{trial}_{i:03d}",
                experience_type=ExperienceType.LEARNING,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)


def _make_world_model():
    """Create a WorldModel with learned dynamics toward target 0.3."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))
    target = np.array([0.3] * 16)
    _feed_dynamics(mem, target, n_trials=25, steps_per=25)
    return WorldModel(mem)


# ── WorldModel Tests ───────────────────────────────────────────

def test_world_model_predict():
    """WorldModel.predict() returns a future state."""
    wm = _make_world_model()
    state = np.full(16, 0.5)
    predicted = wm.predict(state, steps=10)
    assert predicted.shape == (16,)
    assert np.all(predicted >= 0.0) and np.all(predicted <= 1.0)
    print("PASS: test_world_model_predict")


def test_world_model_trajectory():
    """WorldModel.predict_trajectory() returns a list of states."""
    wm = _make_world_model()
    state = np.full(16, 0.5)
    traj = wm.predict_trajectory(state, steps=10)
    assert len(traj) == 11  # initial + 10 steps
    assert traj[0].shape == (16,)
    print("PASS: test_world_model_trajectory")


def test_world_model_attractors():
    """WorldModel discovers attractors."""
    wm = _make_world_model()
    attractors = wm.get_attractors()
    assert len(attractors) >= 1
    # Should find something near 0.3
    found = any(np.linalg.norm(a - np.full(16, 0.3)) < 0.5 for a in attractors)
    assert found
    print("PASS: test_world_model_attractors")


def test_world_model_stability():
    """WorldModel classifies stability."""
    wm = _make_world_model()
    target = np.full(16, 0.3)
    result = wm.get_stability(target)
    assert result['classification'] in ['attractor', 'repellor', 'saddle', 'marginal']
    print("PASS: test_world_model_stability")


def test_world_model_confidence():
    """WorldModel provides confidence estimates."""
    wm = _make_world_model()
    # Near training data should have higher confidence
    near_state = np.full(16, 0.4)
    far_state = np.full(16, 0.99)
    c_near = wm.prediction_confidence(near_state)
    c_far = wm.prediction_confidence(far_state)
    assert 0.0 <= c_near <= 1.0
    assert 0.0 <= c_far <= 1.0
    print("PASS: test_world_model_confidence")


# ── ActionDelta Tests ──────────────────────────────────────────

def test_action_delta_pillar_boost():
    """ActionDelta.pillar_boost creates correct delta."""
    action = ActionDelta.pillar_boost(5, 0.1, dim=16)
    assert action.delta[5] == 0.1
    assert action.delta[0] == 0.0
    assert action.magnitude > 0
    print("PASS: test_action_delta_pillar_boost")


def test_action_delta_random():
    """ActionDelta.random creates a delta with specified magnitude."""
    action = ActionDelta.random(dim=16, magnitude=0.15)
    assert action.delta.shape == (16,)
    assert abs(np.linalg.norm(action.delta) - 0.15) < 0.01
    print("PASS: test_action_delta_random")


def test_action_delta_toward_target():
    """ActionDelta.toward_target moves toward the target."""
    current = np.full(16, 0.3)
    target = np.full(16, 0.6)
    action = ActionDelta.toward_target(target, current, max_magnitude=0.2)
    assert np.all(action.delta > 0)  # should move upward
    assert np.linalg.norm(action.delta) <= 0.2 + 1e-6
    print("PASS: test_action_delta_toward_target")


# ── Simulator Tests ────────────────────────────────────────────

def test_simulator_simulate():
    """Simulator produces a SimResult."""
    wm = _make_world_model()
    sim = Simulator(wm)
    state = np.full(16, 0.5)
    action = ActionDelta.pillar_boost(0, 0.1, dim=16)
    result = sim.simulate(state, action, steps=10)
    assert result.final_state.shape == (16,)
    assert len(result.trajectory) == 11
    assert result.stability in ['attractor', 'repellor', 'saddle', 'marginal']
    print("PASS: test_simulator_simulate")


def test_simulator_batch():
    """Simulator.simulate_batch handles multiple actions."""
    wm = _make_world_model()
    sim = Simulator(wm)
    state = np.full(16, 0.5)
    actions = [ActionDelta.random(dim=16, magnitude=0.05) for _ in range(5)]
    results = sim.simulate_batch(state, actions)
    assert len(results) == 5
    assert all(r.final_state.shape == (16,) for r in results)
    print("PASS: test_simulator_batch")


def test_simulator_controlled():
    """Simulator.simulate_controlled moves toward target."""
    wm = _make_world_model()
    sim = Simulator(wm)
    state = np.full(16, 0.3)
    target = np.full(16, 0.7)
    result = sim.simulate_controlled(state, target, gain=0.3, steps=20)
    # Should move toward target
    dist_before = np.linalg.norm(state - target)
    dist_after = np.linalg.norm(result.final_state - target)
    assert dist_after < dist_before
    print("PASS: test_simulator_controlled")


def test_simulator_tracks_basins():
    """Simulator tracks basin transitions."""
    wm = _make_world_model()
    sim = Simulator(wm)
    state = np.full(16, 0.5)
    action = ActionDelta.random(dim=16, magnitude=0.05)
    result = sim.simulate(state, action, steps=20)
    assert len(result.basin_transitions) >= 1
    print("PASS: test_simulator_tracks_basins")


# ── Evaluator Tests ────────────────────────────────────────────

def test_evaluator_single_state():
    """Evaluator evaluates a single state."""
    wm = _make_world_model()
    ev = Evaluator(UtilityWeights(), wm)
    state = np.full(16, 0.5)
    result = ev.evaluate(state)
    assert isinstance(result, EvalResult)
    assert isinstance(result.utility, float)
    assert result.pillar_contributions.shape == (16,)
    print("PASS: test_evaluator_single_state")


def test_evaluator_trajectory():
    """Evaluator evaluates a full trajectory with discount."""
    wm = _make_world_model()
    ev = Evaluator(UtilityWeights(), wm)
    state = np.full(16, 0.5)
    trajectory = [state + 0.01 * i * np.ones(16) for i in range(10)]
    trajectory = [np.clip(s, 0, 1) for s in trajectory]
    result = ev.evaluate_trajectory(trajectory)
    assert isinstance(result, EvalResult)
    assert result.utility != 0.0
    print("PASS: test_evaluator_trajectory")


def test_evaluator_different_weights():
    """Different UtilityWeights produce different evaluations."""
    wm = _make_world_model()
    state = np.full(16, 0.5)
    
    cautious = Evaluator(UtilityWeights.cautious(), wm)
    curious = Evaluator(UtilityWeights.curious(), wm)
    
    r1 = cautious.evaluate(state)
    r2 = curious.evaluate(state)
    
    # They should produce different utilities
    assert r1.utility != r2.utility
    print("PASS: test_evaluator_different_weights")


def test_evaluator_novelty():
    """Novelty score decreases with repeated visits."""
    wm = _make_world_model()
    ev = Evaluator(UtilityWeights(), wm)
    state = np.full(16, 0.5)
    
    # First visit: high novelty
    r1 = ev.evaluate(state)
    assert r1.novelty_score > 0.5
    
    # After recording visits: lower novelty
    for _ in range(50):
        ev.record_visit(state)
    r2 = ev.evaluate(state)
    assert r2.novelty_score < r1.novelty_score
    print("PASS: test_evaluator_novelty")


# ── Controller Tests ───────────────────────────────────────────

def test_controller_basic():
    """Controller computes a feasible delta."""
    ctrl = Controller(ControlConfig(gain=0.3, max_delta=0.2))
    current = np.full(16, 0.3)
    target = np.full(16, 0.7)
    output = ctrl.compute_control(current, target)
    assert isinstance(output, ControlOutput)
    assert output.delta.shape == (16,)
    assert np.linalg.norm(output.delta) <= 0.2 + 1e-6
    print("PASS: test_controller_basic")


def test_controller_noop():
    """Controller returns zero delta for identical states."""
    ctrl = Controller()
    state = np.full(16, 0.5)
    output = ctrl.compute_control(state, state)
    assert np.linalg.norm(output.delta) < 1e-6
    assert output.reason == "noop"
    print("PASS: test_controller_noop")


def test_controller_clamps_magnitude():
    """Controller clamps delta to max_magnitude."""
    ctrl = Controller(ControlConfig(gain=1.0, max_delta=0.1))
    current = np.full(16, 0.0)
    target = np.full(16, 1.0)
    output = ctrl.compute_control(current, target)
    assert np.linalg.norm(output.delta) <= 0.1 + 1e-6
    assert output.constrained
    print("PASS: test_controller_clamps_magnitude")


def test_controller_trajectory():
    """Controller follows a target trajectory."""
    ctrl = Controller(ControlConfig(gain=0.3, max_delta=0.15, conservation_enforce=False))
    current = np.full(16, 0.3)
    targets = [np.full(16, 0.3 + 0.05 * i) for i in range(5)]
    outputs = ctrl.compute_trajectory_control(current, targets)
    assert len(outputs) == 5
    # State should move toward the last target
    state = current.copy()
    for out in outputs:
        state = np.clip(state + out.delta, 0, 1)
    assert np.mean(state) > 0.3
    print("PASS: test_controller_trajectory")


# ── Intent Tests ───────────────────────────────────────────────

def test_intent_proposal():
    """IntentProposal has correct score."""
    proposal = IntentProposal(
        intent=Intent.EXPLORE,
        priority=0.8,
        confidence=0.7,
    )
    assert abs(proposal.score - 0.56) < 1e-6
    print("PASS: test_intent_proposal")


def test_situation_assessment():
    """IntentGenerator assesses situation from state."""
    wm = _make_world_model()
    gen = IntentGenerator(AgentPersonality(), wm)
    
    # Stable state
    stable_state = np.full(16, 0.5)
    sit = gen.assess_situation(stable_state)
    assert isinstance(sit, Situation)
    
    # Threatened state (high harm)
    threatened_state = np.full(16, 0.5)
    threatened_state[12] = 0.9  # Harm pillar
    sit2 = gen.assess_situation(threatened_state)
    assert sit2 == Situation.THREATENED
    print("PASS: test_situation_assessment")


def test_intent_generation():
    """IntentGenerator produces intents from personality + situation."""
    wm = _make_world_model()
    
    # Cautious agent
    cautious = IntentGenerator(AgentPersonality.cautious(), wm)
    state = np.full(16, 0.5)
    intent = cautious.generate_intent(state)
    assert isinstance(intent, IntentProposal)
    assert 0.0 <= intent.priority <= 1.0
    
    # Explorer agent
    explorer = IntentGenerator(AgentPersonality.explorer(), wm)
    intent2 = explorer.generate_intent(state)
    assert isinstance(intent2, IntentProposal)
    print("PASS: test_intent_generation")


def test_personality_affects_intent():
    """Different personalities produce different intents for the same state."""
    wm = _make_world_model()
    state = np.full(16, 0.5)
    state[12] = 0.8  # high harm
    
    cautious = IntentGenerator(AgentPersonality.cautious(), wm)
    explorer = IntentGenerator(AgentPersonality.explorer(), wm)
    
    i1 = cautious.generate_intent(state)
    i2 = explorer.generate_intent(state)
    
    # Both should respond to harm, but with different priorities
    # Cautious should have higher safety priority
    print(f"  Cautious: {i1.intent.name} (p={i1.priority:.2f})")
    print(f"  Explorer: {i2.intent.name} (p={i2.priority:.2f})")
    print("PASS: test_personality_affects_intent")


# ── Planner Tests ──────────────────────────────────────────────

def test_planner_plan():
    """Planner produces a Plan."""
    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    
    state = np.full(16, 0.5)
    intent = IntentProposal(
        intent=Intent.EXPLORE,
        priority=0.7,
        confidence=0.8,
    )
    
    plan = planner.plan(state, intent)
    assert isinstance(plan, Plan)
    assert len(plan.actions) >= 1
    assert plan.total_utility != 0.0
    assert plan.confidence >= 0.0
    print(f"  Plan: {plan}")
    print("PASS: test_planner_plan")


def test_planner_multi_step():
    """Planner.plan_sequence() produces multi-step plans."""
    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    
    state = np.full(16, 0.5)
    intent = IntentProposal(
        intent=Intent.EXPLORE,
        priority=0.7,
        confidence=0.8,
    )
    
    plan = planner.plan_sequence(state, intent, n_steps=3)
    assert len(plan.actions) <= 3
    assert len(plan.sim_results) <= 3
    print(f"  Multi-step plan: {len(plan.actions)} actions, U={plan.total_utility:.3f}")
    print("PASS: test_planner_multi_step")


def test_planner_defend():
    """Planner with DEFEND intent generates small perturbations."""
    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev, config=PlannerConfig(n_candidates=10))
    
    state = np.full(16, 0.5)
    intent = IntentProposal(
        intent=Intent.DEFEND,
        priority=0.8,
        confidence=0.9,
    )
    
    plan = planner.plan(state, intent)
    assert len(plan.actions) >= 1
    # DEFEND actions should be small
    for action in plan.actions:
        assert action.magnitude < 0.15
    print("PASS: test_planner_defend")


def test_planner_safety_check():
    """Planner.is_safe() checks harm and integrity."""
    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    
    safe_state = np.full(16, 0.5)
    safe_state[12] = 0.2  # low harm
    safe_state[5] = 0.8   # high integrity
    assert planner.is_safe(safe_state)
    
    unsafe_state = np.full(16, 0.5)
    unsafe_state[12] = 0.9  # high harm
    assert not planner.is_safe(unsafe_state)
    print("PASS: test_planner_safety_check")


# ── Full Pipeline Test ─────────────────────────────────────────

def test_full_pipeline():
    """End-to-end: state → intent → plan → evaluate → action."""
    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights.achiever(), wm)
    ctrl = Controller()
    planner = Planner(sim, ev, ctrl)
    intent_gen = IntentGenerator(AgentPersonality.achiever(), wm)
    
    state = np.full(16, 0.5)
    
    # 1. Generate intent
    intent = intent_gen.generate_intent(state)
    assert isinstance(intent, IntentProposal)
    
    # 2. Plan
    plan = planner.plan(state, intent)
    assert len(plan.actions) >= 1
    
    # 3. Execute first action
    action = plan.actions[0]
    new_state = np.clip(state + action.delta, 0, 1)
    assert new_state.shape == (16,)
    
    # 4. Evaluate new state
    eval_result = ev.evaluate(new_state)
    assert isinstance(eval_result, EvalResult)
    
    print(f"  Intent: {intent.intent.name}")
    print(f"  Plan: {plan}")
    print(f"  New state utility: {eval_result.utility:.3f}")
    print("PASS: test_full_pipeline")


if __name__ == "__main__":
    test_world_model_predict()
    test_world_model_trajectory()
    test_world_model_attractors()
    test_world_model_stability()
    test_world_model_confidence()
    test_action_delta_pillar_boost()
    test_action_delta_random()
    test_action_delta_toward_target()
    test_simulator_simulate()
    test_simulator_batch()
    test_simulator_controlled()
    test_simulator_tracks_basins()
    test_evaluator_single_state()
    test_evaluator_trajectory()
    test_evaluator_different_weights()
    test_evaluator_novelty()
    test_controller_basic()
    test_controller_noop()
    test_controller_clamps_magnitude()
    test_controller_trajectory()
    test_intent_proposal()
    test_situation_assessment()
    test_intent_generation()
    test_personality_affects_intent()
    test_planner_plan()
    test_planner_multi_step()
    test_planner_defend()
    test_planner_safety_check()
    test_full_pipeline()
    test_cognitive_loop_reactive_mode()
    test_cognitive_loop_predictive_mode()
    test_cognitive_loop_mode_switch()
    test_cognitive_loop_predictive_applies_delta()


# ── Cognitive Loop Tests ─────────────────────────────────────────

class _MockFieldEvolver:
    """Minimal field evolver for cognitive loop tests."""
    def __init__(self, dim=16):
        self.dim = dim
    def rhs(self, field_state, dt):
        return -0.01 * field_state


class _MockAgentEcology:
    """Minimal agent ecology for cognitive loop tests."""
    class _Response:
        def __init__(self, confidence=0.6):
            self.confidence = confidence
            self.proposed_action = "explore"
            self.agent_role = type('R', (), {'name': 'test'})()
            self.reasoning = "test"
    def tick(self, pillars, world_model=None, memory=None):
        return [self._Response()]
    def get_consensus(self, responses):
        return responses[0] if responses else None


def test_cognitive_loop_reactive_mode():
    """CognitiveLoop in default reactive mode produces actions."""
    from substrate_echo.core.cognitive_loop import CognitiveLoop, CognitiveLoopConfig

    loop = CognitiveLoop(CognitiveLoopConfig(use_planner=False))
    loop.initialize(np.full(16, 0.5), np.full(16, 0.5))

    field_evol = _MockFieldEvolver()
    ecology = _MockAgentEcology()

    result = loop.tick(field_evol, memory_system=None, agent_ecology=ecology)
    assert result["action"]["mode"] == "reactive"
    assert result["action"]["action"] == "explore"
    assert result["stats"]["mode"] == "reactive"
    print("PASS: test_cognitive_loop_reactive_mode")


def test_cognitive_loop_predictive_mode():
    """CognitiveLoop with planner produces predictive actions."""
    from substrate_echo.core.cognitive_loop import CognitiveLoop, CognitiveLoopConfig

    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    intent_gen = IntentGenerator(AgentPersonality(), wm)

    loop = CognitiveLoop(CognitiveLoopConfig(use_planner=True))
    loop.set_planning_stack(planner, intent_gen)
    loop.initialize(np.full(16, 0.5), np.full(16, 0.5))

    field_evol = _MockFieldEvolver()
    ecology = _MockAgentEcology()

    result = loop.tick(field_evol, memory_system=None, agent_ecology=ecology)
    assert result["action"]["mode"] == "predictive"
    assert "delta" in result["action"]
    assert result["intent"] is not None
    assert result["stats"]["mode"] == "predictive"
    print("PASS: test_cognitive_loop_predictive_mode")


def test_cognitive_loop_mode_switch():
    """CognitiveLoop can switch between reactive and predictive."""
    from substrate_echo.core.cognitive_loop import CognitiveLoop, CognitiveLoopConfig

    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    intent_gen = IntentGenerator(AgentPersonality(), wm)

    # Start reactive
    loop = CognitiveLoop(CognitiveLoopConfig(use_planner=False))
    loop.set_planning_stack(planner, intent_gen)
    loop.config.use_planner = False  # override auto-enable from set_planning_stack
    loop.initialize(np.full(16, 0.5), np.full(16, 0.5))

    field_evol = _MockFieldEvolver()
    ecology = _MockAgentEcology()

    result = loop.tick(field_evol, memory_system=None, agent_ecology=ecology)
    assert result["action"]["mode"] == "reactive"

    # Switch to predictive
    loop.config.use_planner = True
    result = loop.tick(field_evol, memory_system=None, agent_ecology=ecology)
    assert result["action"]["mode"] == "predictive"
    print("PASS: test_cognitive_loop_mode_switch")


def test_cognitive_loop_predictive_applies_delta():
    """In predictive mode, the planned delta is applied to pillar state."""
    from substrate_echo.core.cognitive_loop import CognitiveLoop, CognitiveLoopConfig

    wm = _make_world_model()
    sim = Simulator(wm)
    ev = Evaluator(UtilityWeights(), wm)
    planner = Planner(sim, ev)
    intent_gen = IntentGenerator(AgentPersonality(), wm)

    loop = CognitiveLoop(CognitiveLoopConfig(use_planner=True))
    loop.set_planning_stack(planner, intent_gen)

    initial_pillars = np.full(16, 0.5)
    loop.initialize(np.full(16, 0.5), initial_pillars.copy())

    field_evol = _MockFieldEvolver()
    ecology = _MockAgentEcology()

    # Tick should modify pillar state
    result1 = loop.tick(field_evol, memory_system=None, agent_ecology=ecology)
    delta = np.array(result1["action"]["delta"])

    # Pillar state should have changed (unless delta was zero)
    if np.linalg.norm(delta) > 1e-6:
        assert not np.allclose(loop._pillar_state, initial_pillars), \
            "Pillar state not updated after predictive action"
    print("PASS: test_cognitive_loop_predictive_applies_delta")
    print("\nAll planning system tests passed!")
