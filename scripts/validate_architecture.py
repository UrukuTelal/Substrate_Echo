"""Architecture Validation — Full P6 Stack Integration Tests.

Tests the complete cognitive stack under realistic scenarios:
1. Long-duration developmental run with diverse scenarios
2. Dynamics memory learning curve
3. Social inference accuracy (entities that actually communicate vs don't)
4. Planning success rate
5. Response gating effectiveness
"""

import sys
import time
import numpy as np
from collections import Counter

sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.human_state import GaussianDim, HSVState
from substrate_echo.core.bsv import BiologicalStateVector
from substrate_echo.core.esv import EnvironmentalStateVector
from substrate_echo.core.probabilistic_psv import ProbabilisticPSV
from substrate_echo.core.affordance import EntityType, IntentTranslator
from substrate_echo.core.goal_tracker import GoalManager, GoalPhase
from substrate_echo.core.communicative_intent import (
    CommunicativeIntentDetector, BehavioralSignals,
)
from substrate_echo.core.response_gate import ResponseGate, ResponseGateConfig
from substrate_echo.core.identity_tracker import IdentityTracker
from substrate_echo.core.agent_perception import AgentPerception
from substrate_echo.core.multi_agent_goals import MultiAgentGoalInference
from substrate_echo.core.spatial_memory import SpatialMemory
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.models.experience import Experience, ExperienceType


# ── Scenarios ────────────────────────────────────────────────────

def scenario_random_walk(n_entities=5, width=20.0, seed=42):
    """Entities drift randomly. No social signals."""
    rng = np.random.RandomState(seed)
    entities = []
    for i in range(n_entities):
        pos = rng.uniform(0, width, 3)
        vel = rng.randn(3) * 0.3
        pillars = np.clip(rng.uniform(20, 80, 16), 0, 100)
        entities.append({
            "id": i, "type": "human",
            "position": pos, "velocity": vel,
            "pillars": pillars, "social": 0.1,
        })
    return entities, rng


def scenario_converging_pair(n_entities=5, width=20.0, seed=42):
    """Two entities approach each other (social), rest random-walk."""
    rng = np.random.RandomState(seed)
    entities = []
    # Social pair
    entities.append({
        "id": 0, "type": "human",
        "position": np.array([2.0, 10.0, 0.0]),
        "velocity": np.array([0.5, 0.0, 0.0]),
        "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
        "social": 0.8,
    })
    entities.append({
        "id": 1, "type": "human",
        "position": np.array([18.0, 10.0, 0.0]),
        "velocity": np.array([-0.5, 0.0, 0.0]),
        "pillars": np.clip(rng.uniform(40, 80, 16), 0, 100),
        "social": 0.7,
    })
    for i in range(2, n_entities):
        entities.append({
            "id": i, "type": "human",
            "position": rng.uniform(0, width, 3),
            "velocity": rng.randn(3) * 0.2,
            "pillars": np.clip(rng.uniform(20, 60, 16), 0, 100),
            "social": 0.1,
        })
    return entities, rng


def scenario_grouping(n_entities=8, width=20.0, seed=42):
    """3 entities cluster together, rest scatter."""
    rng = np.random.RandomState(seed)
    entities = []
    center = np.array([10.0, 10.0, 0.0])
    for i in range(3):
        pos = center + rng.randn(3) * 1.0
        entities.append({
            "id": i, "type": "human",
            "position": pos,
            "velocity": rng.randn(3) * 0.1,
            "pillars": np.clip(rng.uniform(50, 90, 16), 0, 100),
            "social": 0.9,
        })
    for i in range(3, n_entities):
        entities.append({
            "id": i, "type": "animal",
            "position": rng.uniform(0, width, 3),
            "velocity": rng.randn(3) * 0.4,
            "pillars": np.clip(rng.uniform(10, 40, 16), 0, 100),
            "social": 0.05,
        })
    return entities, rng


def step_entities(entities, rng, width=20.0):
    """Advance all entities one tick."""
    for ent in entities:
        drift = (width / 2 - ent["position"]) * 0.005
        noise = rng.randn(3) * 0.03
        ent["velocity"] = 0.95 * ent["velocity"] + drift + noise
        ent["position"] = np.clip(ent["position"] + ent["velocity"], 0, width)
        if rng.random() < 0.05:
            idx = rng.randint(0, 16)
            ent["pillars"][idx] = np.clip(
                ent["pillars"][idx] + rng.randn() * 1.5, 0, 100)
    return entities


# ── Agent ────────────────────────────────────────────────────────

class ValidationAgent:
    def __init__(self, use_dynamics=True):
        self.identity_tracker = IdentityTracker()
        self.agent_perception = AgentPerception()
        self.hsv = HSVState()
        self.bsv = BiologicalStateVector()
        self.esv = EnvironmentalStateVector()
        self.psv = ProbabilisticPSV()
        self.intent_translator = IntentTranslator()
        self.goal_manager = GoalManager()
        self.spatial_memory = SpatialMemory(cell_size=5.0)
        self.multi_agent = MultiAgentGoalInference()
        self.comm_detector = CommunicativeIntentDetector()
        self.response_gate = ResponseGate()
        self.dynamics = DynamicsMemory() if use_dynamics else None
        
        self.stats = {
            "ticks": 0, "intents_detected": 0,
            "responses_allowed": 0, "responses_blocked": 0,
            "affordances_recorded": 0, "predictions_made": 0,
            "prediction_errors": [],
        }
    
    def tick(self, entities, timestamp):
        self.stats["ticks"] += 1
        my_pos = np.zeros(3)
        my_psv = self.psv.to_deterministic()
        
        # Layer 1: perception
        raw = [{"id": e["id"], "position": e["position"].tolist(),
                "velocity": e["velocity"].tolist(),
                "pillars": e["pillars"].tolist(),
                "shadow_state": (e["pillars"] * 0.8).tolist(),
                "active": True} for e in entities]
        percepts = self.agent_perception.process(my_pos, my_psv, raw)
        
        # Identity tracking
        id_obs = [{"type": e["type"], "position": e["position"].tolist()}
                  for e in entities]
        self.identity_tracker.update(id_obs, timestamp)
        
        # Layer 2: state estimation
        for p in percepts:
            if p.social_signal > 0.5:
                self.hsv.arousal.update(0.6, 0.2)
                self.hsv.social_openness.update(0.7, 0.2)
        
        # Layer 3: goals + affordances + spatial memory
        for e in entities:
            self.goal_manager.update(
                entity_id=e["id"], position=e["position"],
                timestamp=timestamp, social_intent=e["social"])
        
        for p in percepts:
            ent_map = {e["id"]: e for e in entities}
            ent = ent_map.get(p.agent_id)
            etype = ent["type"].upper() if ent else "HUMAN"
            if etype not in [e.name for e in EntityType]:
                etype = "HUMAN"
            aff = self.intent_translator.translate(
                entity_id=p.agent_id,
                entity_type=EntityType[etype],
                position=p.position)
            if aff is not None:
                self.spatial_memory.record(
                    position=p.position, entity_type=etype,
                    action_type="OBSERVED", success=True,
                    timestamp=timestamp)
                self.stats["affordances_recorded"] += 1
        
        interactions = self.multi_agent.infer(self.goal_manager)
        
        # Layer 4: communicative intent
        for p in percepts:
            if p.social_signal > 0.4:
                sig = BehavioralSignals(
                    speech_level=p.social_signal * 0.5,
                    facing_toward_me=p.relative_velocity < 0,
                    distance=p.distance,
                    approach_speed=max(0, -p.relative_velocity),
                    signal_duration=min(1.0, p.frames_tracked * 0.1))
                result = self.comm_detector.analyze(
                    sig, entity_position=p.position,
                    entity_velocity=p.velocity)
                if result is not None:
                    self.stats["intents_detected"] += 1
                    gate = self.response_gate.evaluate(
                        intent_confidence=result.confidence,
                        intent_type=result.intent.name,
                        social_openness=self.hsv.social_openness.mean,
                        observers=len(percepts) - 1,
                        dwell_frames=p.frames_tracked)
                    if gate.allowed:
                        self.stats["responses_allowed"] += 1
                    else:
                        self.stats["responses_blocked"] += 1
        
        # Dynamics memory: record and predict
        if self.dynamics is not None and len(entities) > 0:
            state_vec = my_psv.copy()
            exp = Experience(
                experience_id=f"tick_{self.stats['ticks']}",
                experience_type=ExperienceType.PERCEPTION,
                psv_snapshot=state_vec.tolist(),
                importance=0.5,
            )
            self.dynamics.encode(exp)
            
            if self.dynamics._prev_psv is not None:
                pred = self.dynamics.predict_velocity(state_vec)
                if pred is not None:
                    self.stats["predictions_made"] += 1
                    self.stats["prediction_errors"].append(
                        float(np.linalg.norm(pred)))
        
        return self.stats.copy()


# ── Experiments ──────────────────────────────────────────────────

def experiment_long_duration():
    """5000 ticks, random walk, measure stability."""
    print("\n=== Experiment 1: Long-Duration Developmental Run ===")
    entities, rng = scenario_random_walk(n_entities=5)
    agent = ValidationAgent(use_dynamics=True)
    
    t0 = time.time()
    for t in range(5000):
        step_entities(entities, rng)
        agent.tick(entities, float(t))
    elapsed = time.time() - t0
    
    s = agent.stats
    print(f"  Duration: {elapsed:.1f}s ({5000/elapsed:.0f} ticks/s)")
    print(f"  Entities tracked: {len(agent.identity_tracker.active_entities)}")
    print(f"  Spatial cells filled: {agent.spatial_memory.cell_count}")
    print(f"  Spatial records: {agent.spatial_memory.total_records}")
    print(f"  Intents detected: {s['intents_detected']}")
    print(f"  Responses allowed/blocked: {s['responses_allowed']}/{s['responses_blocked']}")
    print(f"  Predictions made: {s['predictions_made']}")
    print(f"  Avg prediction magnitude: {np.mean(s['prediction_errors']):.4f}" if s['prediction_errors'] else "  No predictions")
    return s


def experiment_social_inference():
    """Converging pair scenario — should detect social interaction."""
    print("\n=== Experiment 2: Social Inference (Converging Pair) ===")
    entities, rng = scenario_converging_pair(n_entities=5)
    agent = ValidationAgent(use_dynamics=False)
    
    social_ticks = 0
    non_social_ticks = 0
    
    for t in range(2000):
        step_entities(entities, rng)
        # Push social pair toward each other
        entities[0]["position"][0] += 0.02
        entities[1]["position"][0] -= 0.02
        
        agent.tick(entities, float(t))
        
        # Check if pair is detected as interacting
        interactions = agent.multi_agent.infer(agent.goal_manager)
        pair_found = any(
            i.entity_a == 0 and i.entity_b == 1
            for i in interactions)
        if pair_found:
            social_ticks += 1
        else:
            non_social_ticks += 1
    
    print(f"  Total ticks: 2000")
    print(f"  Pair interaction detected: {social_ticks}/{social_ticks + non_social_ticks} ticks")
    print(f"  Detection rate: {social_ticks / max(1, social_ticks + non_social_ticks):.1%}")
    return {"detection_rate": social_ticks / max(1, social_ticks + non_social_ticks)}


def experiment_grouping():
    """Grouping scenario — 3 entities cluster, should detect co-presence."""
    print("\n=== Experiment 3: Group Detection (Clustering) ===")
    entities, rng = scenario_grouping(n_entities=8)
    agent = ValidationAgent(use_dynamics=False)
    
    copresent_ticks = 0
    
    for t in range(1500):
        step_entities(entities, rng)
        # Keep group close
        center = np.mean([e["position"] for e in entities[:3]], axis=0)
        for i in range(3):
            to_center = center - entities[i]["position"]
            entities[i]["velocity"] += to_center * 0.02
        
        agent.tick(entities, float(t))
        interactions = agent.multi_agent.infer(agent.goal_manager)
        copresent = sum(1 for i in interactions
                        if i.interaction_type.name in ("CO_PRESENT", "JOINT_GOAL"))
        if copresent > 0:
            copresent_ticks += 1
    
    print(f"  Group co-presence detected: {copresent_ticks}/1500 ticks")
    print(f"  Detection rate: {copresent_ticks / 1500:.1%}")
    return {"group_detection_rate": copresent_ticks / 1500}


def experiment_dynamics_learning():
    """Record same trajectory repeatedly, check if prediction improves."""
    print("\n=== Experiment 4: Dynamics Memory Learning ===")
    dm = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))
    rng = np.random.RandomState(99)
    
    # Record 20 repetitions of a simple trajectory
    for rep in range(20):
        state = np.zeros(16)
        for step in range(50):
            # Simple linear dynamics: each pillar drifts toward target
            target = np.ones(16) * 0.7
            velocity = (target - state) * 0.1 + rng.randn(16) * 0.02
            exp = Experience(
                experience_id=f"rep_{rep}_step_{step}",
                experience_type=ExperienceType.PERCEPTION,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            dm.encode(exp)
            state = state + velocity
    
    # Now predict
    test_state = np.zeros(16)
    pred = dm.predict_velocity(test_state)
    
    if pred is not None:
        # True velocity should be toward 0.7
        true_vel = (np.ones(16) * 0.7 - test_state) * 0.1
        error = np.linalg.norm(pred - true_vel)
        print(f"  Recorded {len(dm.traces)} traces")
        print(f"  Prediction error: {error:.4f}")
        print(f"  True velocity magnitude: {np.linalg.norm(true_vel):.4f}")
        accuracy = 1.0 - error / max(np.linalg.norm(true_vel), 1e-6)
        print(f"  Prediction accuracy: {accuracy:.1%}")
        return {"accuracy": accuracy, "count": len(dm.traces)}
    else:
        print("  No prediction (insufficient data)")
        return {"accuracy": 0, "count": len(dm.traces)}


def experiment_ablation():
    """Compare full system vs no-dynamics vs no-response-gating."""
    print("\n=== Experiment 5: Ablation Study ===")
    results = {}
    
    for label, use_dynamics, gate_threshold in [
        ("full_system", True, 0.5),
        ("no_dynamics", False, 0.5),
        ("no_gating", True, 1.0),  # threshold=1.0 blocks everything
    ]:
        entities, rng = scenario_random_walk(n_entities=5, seed=42)
        agent = ValidationAgent(use_dynamics=use_dynamics)
        if gate_threshold != 0.5:
            agent.response_gate = ResponseGate(
                ResponseGateConfig(confidence_threshold=gate_threshold))
        
        for t in range(2000):
            step_entities(entities, rng)
            agent.tick(entities, float(t))
        
        s = agent.stats
        results[label] = {
            "intents": s["intents_detected"],
            "allowed": s["responses_allowed"],
            "blocked": s["responses_blocked"],
            "predictions": s["predictions_made"],
            "spatial_cells": agent.spatial_memory.cell_count,
        }
    
    for label, r in results.items():
        print(f"  [{label}] intents={r['intents']}, "
              f"allowed={r['allowed']}, blocked={r['blocked']}, "
              f"predictions={r['predictions']}, cells={r['spatial_cells']}")
    return results


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UACF Architecture Validation — Full Stack Integration")
    print("=" * 60)
    
    experiment_long_duration()
    experiment_social_inference()
    experiment_grouping()
    experiment_dynamics_learning()
    experiment_ablation()
    
    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)
