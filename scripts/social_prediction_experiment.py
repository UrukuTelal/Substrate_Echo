"""Closed-Loop Social Prediction Experiment.

Tests whether HSV + affordance translation improves social cognition
compared to a baseline agent without those modules.

Two agents observe the same human behaviors. One has:
  - IntentTranslator + HSV + GoalManager (Full agent)
The other has:
  - Basic perception only (Baseline agent)

Metrics:
  1. Intent prediction accuracy
  2. Response appropriateness (PSV change matches actual intent)
  3. Recovery after misunderstanding
  4. Adaptation over repeated encounters
  5. Uncertainty reduction through interaction (key metric)

Scenarios:
  1. Friendly approach (human walks toward AI, smiling)
  2. Ambiguous approach (human walks toward AI, neutral expression)
  3. Threatening approach (human walks toward AI quickly, tense)
  4. Social bid (human makes eye contact, gestures)
  5. Fatigued human (slow movement, slouched, blinking)
"""

import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from substrate_echo.core.human_state import (
    HumanStateEstimator, HumanObservation, HSVState
)
from substrate_echo.core.affordance import (
    IntentTranslator, EntityType, PropertyMap, Affordance, PSVDelta
)
from substrate_echo.core.goal_tracker import GoalManager, GoalPhase


# -- Scenario Definitions -----------------------------------------

@dataclass
class ScenarioStep:
    """One tick of a social scenario."""
    position: np.ndarray
    distance: float
    # Behavioral signals
    speech_level: float = 0.0
    gesture_speed: float = 0.0
    motion_speed: float = 0.0
    body_tension: float = 0.0
    facial_openness: float = 0.5
    vocal_tone: float = 0.5
    posture_openness: float = 0.5
    gaze_stability: float = 0.5
    facing_toward: float = 0.5
    approach_behavior: float = 0.0
    eye_contact_frequency: float = 0.0
    blink_rate: float = 0.3
    posture_slump: float = 0.0
    response_latency: float = 0.3
    trajectory_directness: float = 0.5
    gesture_repetition: float = 0.0
    gaze_fixation: float = 0.0
    orientation_consistency: float = 0.5
    fidget_level: float = 0.2
    emotional_variability: float = 0.2
    behavioral_consistency: float = 0.5
    recovery_speed: float = 0.5
    # Ground truth
    true_intent: str = "unknown"
    true_valence: float = 0.5  # actual emotional valence


def make_friendly_approach() -> list[ScenarioStep]:
    """Human approaches with friendly demeanor."""
    steps = []
    for i in range(8):
        t = i / 7.0
        steps.append(ScenarioStep(
            position=np.array([2.0 - t * 1.5, 0.0, 0.0]),
            distance=max(0.3, 2.0 - t * 1.7),
            facial_openness=0.7 + t * 0.15,
            posture_openness=0.7 + t * 0.1,
            facing_toward=0.8 + t * 0.1,
            approach_behavior=0.6 + t * 0.2,
            eye_contact_frequency=0.5 + t * 0.3,
            gesture_speed=0.2 + t * 0.1,
            motion_speed=0.3 + t * 0.2,
            trajectory_directness=0.7 + t * 0.1,
            behavioral_consistency=0.7,
            true_intent="friendly_approach",
            true_valence=0.8,
        ))
    return steps


def make_ambiguous_approach() -> list[ScenarioStep]:
    """Human approaches with neutral/ambiguous signals."""
    steps = []
    for i in range(8):
        t = i / 7.0
        steps.append(ScenarioStep(
            position=np.array([2.0 - t * 1.5, 0.0, 0.0]),
            distance=max(0.3, 2.0 - t * 1.7),
            facial_openness=0.5,  # neutral
            posture_openness=0.5,
            facing_toward=0.6,
            approach_behavior=0.5 + t * 0.3,
            eye_contact_frequency=0.3,
            gesture_speed=0.3,
            motion_speed=0.4 + t * 0.2,
            trajectory_directness=0.6,
            behavioral_consistency=0.5,
            emotional_variability=0.4,
            true_intent="ambiguous",
            true_valence=0.5,
        ))
    return steps


def make_threatening_approach() -> list[ScenarioStep]:
    """Human approaches with aggressive signals."""
    steps = []
    for i in range(8):
        t = i / 7.0
        steps.append(ScenarioStep(
            position=np.array([2.0 - t * 1.8, 0.0, 0.0]),
            distance=max(0.2, 2.0 - t * 1.8),
            speech_level=0.5 + t * 0.3,
            gesture_speed=0.6 + t * 0.2,
            motion_speed=0.6 + t * 0.3,
            body_tension=0.7 + t * 0.2,
            facial_openness=0.2,
            vocal_tone=0.2,
            posture_openness=0.2,
            gaze_stability=0.8,
            facing_toward=0.9,
            approach_behavior=0.8 + t * 0.15,
            trajectory_directness=0.8,
            behavioral_consistency=0.8,
            emotional_variability=0.1,
            true_intent="threatening",
            true_valence=0.15,
        ))
    return steps


def make_social_bid() -> list[ScenarioStep]:
    """Human tries to get AI's attention."""
    steps = []
    for i in range(8):
        t = i / 7.0
        steps.append(ScenarioStep(
            position=np.array([1.0, 0.5 - t * 0.3, 0.0]),
            distance=1.2 - t * 0.3,
            gesture_speed=0.5 + t * 0.3,
            facing_toward=0.8,
            eye_contact_frequency=0.7 + t * 0.2,
            gesture_repetition=min(3, int(t * 4)),
            gaze_fixation=0.7,
            motion_speed=0.2,
            trajectory_directness=0.6,
            behavioral_consistency=0.6,
            true_intent="social_engagement",
            true_valence=0.7,
        ))
    return steps


def make_fatigued_human() -> list[ScenarioStep]:
    """Human is tired and slow."""
    steps = []
    for i in range(8):
        t = i / 7.0
        steps.append(ScenarioStep(
            position=np.array([1.5 - t * 0.3, 0.0, 0.0]),
            distance=max(0.8, 1.5 - t * 0.5),
            blink_rate=0.7 + t * 0.15,
            posture_slump=0.6 + t * 0.15,
            response_latency=0.6 + t * 0.2,
            motion_speed=0.15 - t * 0.05,
            gesture_speed=0.1,
            facial_openness=0.4,
            posture_openness=0.4,
            gaze_stability=0.4 - t * 0.1,
            fidget_level=0.3,
            behavioral_consistency=0.5,
            true_intent="fatigued",
            true_valence=0.4,
        ))
    return steps


SCENARIOS = {
    "friendly_approach": make_friendly_approach,
    "ambiguous_approach": make_ambiguous_approach,
    "threatening_approach": make_threatening_approach,
    "social_bid": make_social_bid,
    "fatigued_human": make_fatigued_human,
}


# -- Agent Types --------------------------------------------------

@dataclass
class AgentResult:
    """Result of one tick for one agent."""
    intent_predictions: list[str] = field(default_factory=list)
    top_intent: str = "unknown"
    top_probability: float = 0.0
    social_intent: float = 0.0
    psv_deltas: list[float] = field(default_factory=list)
    uncertainty: float = 0.25
    hsv_state: Optional[HSVState] = None
    goal_phase: str = "IDLE"


class FullAgent:
    """Agent with HSV + IntentTranslator + GoalManager."""
    
    def __init__(self):
        self.hsv = HumanStateEstimator()
        self.translator = IntentTranslator(dim=16)
        self.goal_mgr = GoalManager()
        self._tick = 0
    
    def observe(self, step: ScenarioStep) -> AgentResult:
        self._tick += 1
        
        # Update HSV
        obs = HumanObservation(
            speech_level=step.speech_level,
            gesture_speed=step.gesture_speed,
            motion_speed=step.motion_speed,
            body_tension=step.body_tension,
            facial_openness=step.facial_openness,
            vocal_tone=step.vocal_tone,
            posture_openness=step.posture_openness,
            gaze_stability=step.gaze_stability,
            facing_toward=step.facing_toward,
            approach_behavior=step.approach_behavior,
            eye_contact_frequency=step.eye_contact_frequency,
            blink_rate=step.blink_rate,
            posture_slump=step.posture_slump,
            response_latency=step.response_latency,
            trajectory_directness=step.trajectory_directness,
            gesture_repetition=step.gesture_repetition,
            gaze_fixation=step.gaze_fixation,
            orientation_consistency=step.orientation_consistency,
            fidget_level=step.fidget_level,
            emotional_variability=step.emotional_variability,
            behavioral_consistency=step.behavioral_consistency,
            recovery_speed=step.recovery_speed,
        )
        self.hsv.observe(obs)
        
        # Intent inference from HSV
        intents = self.hsv.infer_intents()
        top = intents[0] if intents else None
        
        # Translate with HSV context
        aff = self.translator.translate(
            entity_id=1,
            entity_type=EntityType.HUMAN,
            position=step.position,
            properties=PropertyMap(distance=step.distance),
            hsv_state=self.hsv.estimate,
        )
        
        # Goal tracking
        self.goal_mgr.update(
            entity_id=1, position=step.position,
            timestamp=float(self._tick) * 0.5,
            social_intent=aff.social_intent,
        )
        goal = self.goal_mgr.get_state(1)
        
        return AgentResult(
            intent_predictions=[i.label for i in intents[:3]],
            top_intent=top.label if top else "unknown",
            top_probability=top.probability if top else 0.0,
            social_intent=aff.social_intent,
            psv_deltas=[d.delta_theta for d in aff.pillar_deltas],
            uncertainty=self.hsv.estimate.uncertainty,
            hsv_state=self.hsv.estimate,
            goal_phase=goal.phase.name if goal else "IDLE",
        )


class BaselineAgent:
    """Agent with only basic perception (no HSV, no affordance translation)."""
    
    def __init__(self):
        self._tick = 0
        self._social_history: list[float] = []
    
    def observe(self, step: ScenarioStep) -> AgentResult:
        self._tick += 1
        
        # Simple threshold-based intent detection
        social = 0.0
        if step.facing_toward > 0.6 and step.distance < 2.0:
            social += 0.3
        if step.eye_contact_frequency > 0.5:
            social += 0.2
        if step.approach_behavior > 0.5:
            social += 0.2
        
        self._social_history.append(social)
        
        # Simple intent classification
        intent = "unknown"
        if step.motion_speed > 0.5 and step.body_tension > 0.6:
            intent = "threatening"
        elif step.facial_openness > 0.6 and step.approach_behavior > 0.5:
            intent = "friendly_approach"
        elif step.gesture_speed > 0.4 and step.gesture_repetition > 1:
            intent = "social_engagement"
        elif step.blink_rate > 0.6 and step.posture_slump > 0.5:
            intent = "fatigued"
        elif step.approach_behavior > 0.3:
            intent = "ambiguous"
        
        return AgentResult(
            intent_predictions=[intent],
            top_intent=intent,
            top_probability=0.5,
            social_intent=social,
            psv_deltas=[],
            uncertainty=0.5,  # always uncertain, no learning
            goal_phase="UNKNOWN",
        )


# -- Evaluation Metrics -------------------------------------------

def intent_accuracy(results: list[AgentResult],
                    scenario_name: str) -> float:
    """How often did the agent predict the correct intent class?"""
    intent_map = {
        "friendly_approach": ["friendly_approach", "social_engagement"],
        "ambiguous_approach": ["ambiguous", "friendly_approach", "social_engagement"],
        "threatening_approach": ["threatening"],
        "social_bid": ["social_engagement", "friendly_approach"],
        "fatigued_human": ["fatigued"],
    }
    acceptable = intent_map.get(scenario_name, [])
    correct = sum(1 for r in results if r.top_intent in acceptable)
    return correct / max(1, len(results))


def uncertainty_trajectory(results: list[AgentResult]) -> list[float]:
    """Track how uncertainty changes over time."""
    return [r.uncertainty for r in results]


def social_intent_calibration(results: list[AgentResult],
                               scenario_name: str) -> float:
    """How well calibrated is social_intent to the actual scenario?
    
    Friendly/social scenarios should have high social_intent.
    Threatening/fatigued should have lower.
    """
    expected = {
        "friendly_approach": 0.7,
        "ambiguous_approach": 0.5,
        "threatening_approach": 0.3,
        "social_bid": 0.8,
        "fatigued_human": 0.3,
    }
    target = expected.get(scenario_name, 0.5)
    vals = [r.social_intent for r in results]
    mean_social = np.mean(vals) if vals else 0.5
    return 1.0 - abs(mean_social - target)


def hsv_convergence(hsv: HumanStateEstimator,
                    scenario_name: str) -> dict:
    """How well did HSV converge on the true state?"""
    hsv_final = hsv.estimate
    truth = {
        "friendly_approach": {"valence": 0.8, "social_openness": 0.8},
        "ambiguous_approach": {"valence": 0.5, "social_openness": 0.5},
        "threatening_approach": {"valence": 0.15, "arousal": 0.8},
        "social_bid": {"social_openness": 0.8, "attention": 0.7},
        "fatigued_human": {"fatigue": 0.8, "arousal": 0.2},
    }
    t = truth.get(scenario_name, {})
    errors = {}
    for dim, target_val in t.items():
        actual = getattr(hsv_final, dim).mean
        errors[dim] = abs(actual - target_val)
    
    return {
        "mean_error": float(np.mean(list(errors.values()))) if errors else 0.0,
        "errors": errors,
        "final_hsv": repr(hsv_final),
    }


# -- Run Experiment -----------------------------------------------

def run_experiment():
    """Run all scenarios with both agent types, compare results."""
    print("=" * 70)
    print("CLOSED-LOOP SOCIAL PREDICTION EXPERIMENT")
    print("=" * 70)
    
    all_results = {}
    
    for scenario_name, scenario_fn in SCENARIOS.items():
        print(f"\n{'-' * 60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'-' * 60}")
        
        steps = scenario_fn()
        
        # Run with Full agent
        full = FullAgent()
        full_results = [full.observe(s) for s in steps]
        
        # Run with Baseline agent
        baseline = BaselineAgent()
        baseline_results = [baseline.observe(s) for s in steps]
        
        # Evaluate
        full_acc = intent_accuracy(full_results, scenario_name)
        base_acc = intent_accuracy(baseline_results, scenario_name)
        
        full_cal = social_intent_calibration(full_results, scenario_name)
        base_cal = social_intent_calibration(baseline_results, scenario_name)
        
        full_unc = uncertainty_trajectory(full_results)
        base_unc = uncertainty_trajectory(baseline_results)
        
        hsv_conv = hsv_convergence(full.hsv, scenario_name)
        
        print(f"\n  Intent Accuracy:")
        print(f"    Full agent:    {full_acc:.1%}")
        print(f"    Baseline:      {base_acc:.1%}")
        
        print(f"\n  Social Calibration:")
        print(f"    Full agent:    {full_cal:.3f}")
        print(f"    Baseline:      {base_cal:.3f}")
        
        print(f"\n  Uncertainty trajectory (Full):")
        print(f"    Start: {full_unc[0]:.3f} -> End: {full_unc[-1]:.3f}")
        unc_decreased = full_unc[-1] < full_unc[0]
        print(f"    {'v Decreased' if unc_decreased else '-> Stable/Increased'}")
        
        print(f"\n  HSV Convergence:")
        print(f"    Mean error: {hsv_conv['mean_error']:.3f}")
        print(f"    {hsv_conv['final_hsv']}")
        
        print(f"\n  Top intent predictions (Full):")
        for i, r in enumerate(full_results[:3]):
            print(f"    t={i}: {r.top_intent} ({r.top_probability:.2f})")
        print(f"    ...")
        for i, r in enumerate(full_results[-2:]):
            print(f"    t={len(full_results)-2+i}: {r.top_intent} ({r.top_probability:.2f})")
        
        all_results[scenario_name] = {
            "full_accuracy": full_acc,
            "baseline_accuracy": base_acc,
            "full_calibration": full_cal,
            "baseline_calibration": base_cal,
            "uncertainty_decreased": unc_decreased,
            "hsv_mean_error": hsv_conv["mean_error"],
            "full_top_intents": [r.top_intent for r in full_results],
            "baseline_top_intents": [r.top_intent for r in baseline_results],
        }
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    full_accs = [r["full_accuracy"] for r in all_results.values()]
    base_accs = [r["baseline_accuracy"] for r in all_results.values()]
    full_cals = [r["full_calibration"] for r in all_results.values()]
    base_cals = [r["baseline_calibration"] for r in all_results.values()]
    unc_decreases = sum(1 for r in all_results.values() if r["uncertainty_decreased"])
    hsv_errors = [r["hsv_mean_error"] for r in all_results.values()]
    
    print(f"\n  Mean Intent Accuracy:")
    print(f"    Full agent:    {np.mean(full_accs):.1%}")
    print(f"    Baseline:      {np.mean(base_accs):.1%}")
    print(f"    Improvement:   {np.mean(full_accs) - np.mean(base_accs):+.1%}")
    
    print(f"\n  Mean Social Calibration:")
    print(f"    Full agent:    {np.mean(full_cals):.3f}")
    print(f"    Baseline:      {np.mean(base_cals):.3f}")
    
    print(f"\n  Uncertainty Reduction:")
    print(f"    Scenarios with decreasing uncertainty: {unc_decreases}/{len(all_results)}")
    
    print(f"\n  HSV Convergence (mean error across scenarios):")
    print(f"    {np.mean(hsv_errors):.3f}")
    
    print(f"\n  Key Insight:")
    if np.mean(full_accs) > np.mean(base_accs):
        print(f"    HSV + affordance translation improves intent prediction")
    if unc_decreases >= len(all_results) // 2:
        print(f"    Uncertainty decreases through interaction (social learning)")
    if np.mean(full_cals) > np.mean(base_cals):
        print(f"    Full agent has better calibrated social responses")
    
    return all_results


if __name__ == "__main__":
    results = run_experiment()
