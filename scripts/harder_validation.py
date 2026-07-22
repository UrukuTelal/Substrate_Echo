"""Harder Validation Experiments — stress-test the P6 stack.

1. Dynamic group scenarios (merge, split, cross, trade members)
2. Calibration benchmark (intent confidence vs correctness)
3. Curiosity signal measurement (prediction uncertainty as exploration driver)
"""

import sys
import time
import numpy as np

sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.human_state import HSVState
from substrate_echo.core.bsv import BiologicalStateVector
from substrate_echo.core.esv import EnvironmentalStateVector
from substrate_echo.core.probabilistic_psv import ProbabilisticPSV
from substrate_echo.core.affordance import EntityType, IntentTranslator
from substrate_echo.core.goal_tracker import GoalManager, GoalPhase
from substrate_echo.core.communicative_intent import (
    CommunicativeIntentDetector, BehavioralSignals,
)
from substrate_echo.core.response_gate import ResponseGate
from substrate_echo.core.identity_tracker import IdentityTracker
from substrate_echo.core.agent_perception import AgentPerception
from substrate_echo.core.multi_agent_goals import MultiAgentGoalInference
from substrate_echo.core.spatial_memory import SpatialMemory


# ── Hard Group Scenarios ─────────────────────────────────────────

def scenario_merging_groups(width=20.0, seed=42):
    """Two groups of 3 approach each other and merge into one group of 6."""
    rng = np.random.RandomState(seed)
    entities = []
    # Group A: starts at left, moving right
    for i in range(3):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([3.0 + rng.randn() * 0.5,
                                   8.0 + rng.randn() * 0.5, 0.0]),
            "velocity": np.array([0.3, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.8,
        })
    # Group B: starts at right, moving left
    for i in range(3, 6):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([17.0 + rng.randn() * 0.5,
                                   8.0 + rng.randn() * 0.5, 0.0]),
            "velocity": np.array([-0.3, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.8,
        })
    return entities, rng


def scenario_splitting_group(width=20.0, seed=42):
    """A group of 6 splits into two groups of 3 going opposite directions."""
    rng = np.random.RandomState(seed)
    entities = []
    center = np.array([10.0, 10.0, 0.0])
    for i in range(6):
        direction = 1.0 if i < 3 else -1.0
        entities.append({
            "id": i, "type": "human",
            "position": center + rng.randn(3) * 0.5,
            "velocity": np.array([direction * 0.3, rng.randn() * 0.1, 0.0]),
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.8,
        })
    return entities, rng


def scenario_crossing_groups(width=20.0, seed=42):
    """Two groups cross paths through the same region."""
    rng = np.random.RandomState(seed)
    entities = []
    # Group A: moves left to right
    for i in range(4):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([2.0 + rng.randn() * 0.3,
                                   8.0 + rng.randn() * 0.3, 0.0]),
            "velocity": np.array([0.4, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
            "social": 0.1,
        })
    # Group B: moves right to left, different Y
    for i in range(4, 8):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([18.0 + rng.randn() * 0.3,
                                   12.0 + rng.randn() * 0.3, 0.0]),
            "velocity": np.array([-0.4, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
            "social": 0.1,
        })
    return entities, rng


def scenario_member_trade(width=20.0, seed=42):
    """Two groups, one member from each group switches groups."""
    rng = np.random.RandomState(seed)
    entities = []
    # Group A
    for i in range(3):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([5.0 + rng.randn() * 0.5,
                                   10.0 + rng.randn() * 0.5, 0.0]),
            "velocity": np.array([0.1, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.7,
        })
    # Group B
    for i in range(3, 6):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([15.0 + rng.randn() * 0.5,
                                   10.0 + rng.randn() * 0.5, 0.0]),
            "velocity": np.array([-0.1, 0.0, 0.0]),
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.7,
        })
    return entities, rng


def scenario_appear_disappear(width=20.0, seed=42):
    """Entities appear and disappear from the scene."""
    rng = np.random.RandomState(seed)
    entities = []
    for i in range(4):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([5.0 + i * 3, 10.0, 0.0]),
            "velocity": rng.randn(3) * 0.2,
            "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
            "social": 0.5,
        })
    return entities, rng


def step_scenario(entities, rng, tick, width=20.0):
    """Step entities with scenario-specific dynamics."""
    for ent in entities:
        drift = (width / 2 - ent["position"]) * 0.005
        noise = rng.randn(3) * 0.02
        ent["velocity"] = 0.95 * ent["velocity"] + drift + noise
        ent["position"] = np.clip(ent["position"] + ent["velocity"], 0, width)


# ── Agent (minimal) ──────────────────────────────────────────────

class TestAgent:
    def __init__(self):
        self.identity_tracker = IdentityTracker()
        self.agent_perception = AgentPerception()
        self.goal_manager = GoalManager()
        self.multi_agent = MultiAgentGoalInference()
        self.spatial_memory = SpatialMemory(cell_size=5.0)

    def tick(self, entities, timestamp):
        my_pos = np.zeros(3)
        my_psv = np.zeros(16)
        raw = [{"id": e["id"], "position": e["position"].tolist(),
                "velocity": e["velocity"].tolist(),
                "pillars": e["pillars"].tolist(),
                "shadow_state": (e["pillars"] * 0.8).tolist(),
                "active": True} for e in entities]
        percepts = self.agent_perception.process(my_pos, my_psv, raw)

        id_obs = [{"type": e["type"], "position": e["position"].tolist()}
                  for e in entities]
        self.identity_tracker.update(id_obs, timestamp)

        for e in entities:
            self.goal_manager.update(
                entity_id=e["id"], position=e["position"],
                timestamp=timestamp, social_intent=e["social"])

        interactions = self.multi_agent.infer(self.goal_manager)
        return interactions, percepts


# ── Experiments ──────────────────────────────────────────────────

def experiment_harder_groups():
    """Test group detection under challenging conditions."""
    print("\n=== Hard Group Dynamics ===")
    results = {}

    # Merging groups: two groups moving toward each other
    entities, rng = scenario_merging_groups()
    agent = TestAgent()
    convergence_detected = 0
    same_group_detected = 0
    total_ticks = 0
    for t in range(2000):
        step_scenario(entities, rng, t)
        # Push groups toward center
        for i in range(3):
            entities[i]["velocity"][0] += 0.01
        for i in range(3, 6):
            entities[i]["velocity"][0] -= 0.01
        interactions, _ = agent.tick(entities, float(t))
        types = [i.interaction_type.name for i in interactions]
        total_ticks += 1
        if "CONVERGING" in types:
            convergence_detected += 1
        if "CO_PRESENT" in types:
            same_group_detected += 1
    results["merging"] = {
        "convergence_rate": convergence_detected / total_ticks,
        "same_group_rate": same_group_detected / total_ticks,
    }
    print(f"  Merging groups:")
    print(f"    Convergence detected: {convergence_detected}/{total_ticks} "
          f"({convergence_detected/total_ticks:.1%})")
    print(f"    Same-group detected: {same_group_detected}/{total_ticks} "
          f"({same_group_detected/total_ticks:.1%})")

    # Crossing groups: two groups passing through same region
    entities, rng = scenario_crossing_groups()
    agent = TestAgent()
    near_cross = 0
    total_near = 0
    for t in range(2000):
        step_scenario(entities, rng, t)
        interactions, _ = agent.tick(entities, float(t))
        pair_dists = []
        for i in range(4):
            for j in range(4, 8):
                d = np.linalg.norm(
                    entities[i]["position"] - entities[j]["position"])
                pair_dists.append(d)
        min_dist = min(pair_dists) if pair_dists else 999
        if min_dist < 5.0:
            total_near += 1
            types = [i.interaction_type.name for i in interactions]
            if any(t != "NONE" for t in types):
                near_cross += 1
    results["crossing"] = {
        "near_cross_rate": near_cross / max(1, total_near),
        "total_near_ticks": total_near,
    }
    print(f"  Crossing groups:")
    print(f"    Detection when near: {near_cross}/{total_near} "
          f"({near_cross/max(1,total_near):.1%})")

    # Appear/disappear: entity vanishes and reappears
    entities, rng = scenario_appear_disappear()
    agent = TestAgent()
    tracked_ids = set()
    for t in range(2000):
        step_scenario(entities, rng, t)
        if 500 <= t < 1000:
            active_ents = [e for e in entities if e["id"] != 2]
        elif t >= 1000:
            active_ents = entities
        else:
            active_ents = entities
        interactions, _ = agent.tick(active_ents, float(t))
        tracked = {e.entity_id for e in agent.identity_tracker.active_entities}
        tracked_ids.update(tracked)
    results["disappear_reappear"] = {
        "unique_ids_tracked": len(tracked_ids),
    }
    print(f"  Appear/disappear:")
    print(f"    Unique IDs tracked: {len(tracked_ids)}")

    return results


def experiment_calibration():
    """Measure intent detection calibration: confidence vs correctness."""
    print("\n=== Calibration Benchmark ===")

    detector = CommunicativeIntentDetector()

    # Create test cases with known ground truth
    test_cases = [
        # (signals, ground_truth_intent, description)
        (BehavioralSignals(
            speech_level=0.9, facing_toward_me=True,
            distance=2.0, approach_speed=0.5,
            signal_duration=0.8), "social", "clear greeting"),
        (BehavioralSignals(
            speech_level=0.1, facing_toward_me=False,
            distance=8.0, approach_speed=-0.3,
            signal_duration=0.1), "none", "far away, not engaged"),
        (BehavioralSignals(
            speech_level=0.7, facing_toward_me=True,
            distance=1.0, approach_speed=0.8,
            signal_duration=0.5), "social", "close, fast approach"),
        (BehavioralSignals(
            speech_level=0.3, facing_toward_me=True,
            distance=3.0, approach_speed=0.0,
            signal_duration=0.3), "uncertain", "moderate signals"),
    ]

    correct = 0
    total = 0
    confidences = []

    for signals, ground_truth, desc in test_cases:
        result = detector.analyze(
            signals,
            entity_position=np.array([signals.distance, 0, 0]),
            entity_velocity=np.array([-signals.approach_speed, 0, 0]),
        )

        if result is not None:
            detected_intent = "social" if result.confidence > 0.5 else "none"
            is_correct = (detected_intent == ground_truth
                         or ground_truth == "uncertain")
            if is_correct:
                correct += 1
            total += 1
            confidences.append(result.confidence)
            label = "OK" if is_correct else "MISS"
            print(f"  [{label}] {desc}: conf={result.confidence:.3f}, "
                  f"intent={result.intent.name}")

    accuracy = correct / max(1, total)
    avg_confidence = np.mean(confidences) if confidences else 0
    print(f"\n  Accuracy: {accuracy:.1%}")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"  Calibration gap: {abs(accuracy - avg_confidence):.3f}")

    return {"accuracy": accuracy, "avg_confidence": avg_confidence}


def experiment_curiosity_signal():
    """Measure prediction error as a curiosity/exploration signal."""
    print("\n=== Curiosity Signal (Prediction Error) ===")

    agent = TestAgent()
    rng = np.random.RandomState(123)

    # First, train a DynamicsMemory on known dynamics
    from substrate_echo.core.dynamics_memory import DynamicsMemory
    dm = DynamicsMemory(dim=16)

    # Train on linear dynamics: v = -0.5*(x - 0.5) (converges to 0.5)
    for _ in range(200):
        x = rng.uniform(0.1, 0.9, 16)
        v = -0.5 * (x - 0.5)  # linear toward center
        dm._states.append(x)
        dm._velocities.append(v)
    dm._fit_dynamics()

    # Phase 1: known region — entities move according to learned dynamics
    entities = []
    for i in range(3):
        entities.append({
            "id": i, "type": "human",
            "position": np.array([10.0 + 3 * np.cos(0), 10.0 + 3 * np.sin(0), 0.0]),
            "velocity": np.array([0.0, 0.3, 0.0]),
            "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
            "social": 0.3,
        })

    known_errors = []
    prev_pillars = [e["pillars"].copy() / 100.0 for e in entities]
    for t in range(200):
        angle = t * 0.02
        for i, ent in enumerate(entities):
            ent["position"] = np.array([
                10.0 + 3 * np.cos(angle + i * 2.1),
                10.0 + 3 * np.sin(angle + i * 2.1),
                0.0])
            ent["velocity"] = np.array([
                -0.3 * np.sin(angle + i * 2.1),
                0.3 * np.cos(angle + i * 2.1),
                0.0])
            # Pillar velocity: smooth drift toward center
            ent["pillars"] = np.clip(
                ent["pillars"] + rng.randn(16) * 0.5, 0, 100)

        agent.tick(entities, float(t))

        # Measure prediction error in PILLAR space
        for i, ent in enumerate(entities):
            psv = ent["pillars"] / 100.0
            actual_pillar_vel = psv - prev_pillars[i]
            error = dm.prediction_error(psv, actual_pillar_vel)
            known_errors.append(error)
            prev_pillars[i] = psv.copy()

    # Phase 2: novel region — entities move randomly
    novel_errors = []
    for t in range(200):
        for i, ent in enumerate(entities):
            ent["position"] = np.array([
                2.0 + rng.randn() * 0.5,
                2.0 + rng.randn() * 0.5,
                0.0])
            ent["velocity"] = rng.randn(3) * 0.3
            # Pillar velocity: random large jumps
            ent["pillars"] = np.clip(
                ent["pillars"] + rng.randn(16) * 10, 0, 100)

        agent.tick(entities, float(200 + t))

        for i, ent in enumerate(entities):
            psv = ent["pillars"] / 100.0
            actual_pillar_vel = psv - prev_pillars[i]
            error = dm.prediction_error(psv, actual_pillar_vel)
            novel_errors.append(error)
            prev_pillars[i] = psv.copy()

    # Phase 3: region uncertainty — measure variance in different regions
    center_uncertainty = dm.region_uncertainty(np.full(16, 0.5), n_samples=20)
    edge_uncertainty = dm.region_uncertainty(np.full(16, 0.1), n_samples=20)

    known_avg = np.mean(known_errors) if known_errors else 0
    novel_avg = np.mean(novel_errors) if novel_errors else 0
    delta = novel_avg - known_avg

    print(f"  Known region prediction error: {known_avg:.6f}")
    print(f"  Novel region prediction error: {novel_avg:.6f}")
    print(f"  Error increase (curiosity signal): {delta:.6f}")
    print(f"  Center uncertainty: {center_uncertainty:.6f}")
    print(f"  Edge uncertainty: {edge_uncertainty:.6f}")
    print(f"  Signal strength: {'STRONG' if delta > 0.001 else 'WEAK'}")

    return {
        "known_error": known_avg,
        "novel_error": novel_avg,
        "curiosity_signal": delta,
        "center_uncertainty": center_uncertainty,
        "edge_uncertainty": edge_uncertainty,
    }


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Harder Validation Experiments")
    print("=" * 60)

    experiment_harder_groups()
    experiment_calibration()
    experiment_curiosity_signal()

    print("\n" + "=" * 60)
    print("All harder experiments complete.")
    print("=" * 60)
