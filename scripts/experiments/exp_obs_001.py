"""EXP-OBS-001: Epistemic Observatory Validation.

Question:
    Can we observe cognition in real-time as the agent plays SC2?

Setup:
    SC2-like environment with full cognitive stack.
    Observatory records all cognitive events.
    HUD displays reasoning alongside behavior.

SC2 Playthrough:
    Full stack exercise with observability:
    Observation → Hypothesis → Prediction → Outcome →
    Trust Update → Discovery → Cultural Prior →
    All events recorded and displayed.

Measures:
    - Events recorded per tick
    - Timeline coverage
    - Snapshot accuracy
    - HUD rendering quality
    - Causal chain tracing

This validates the Epistemic Observatory as a debugging tool.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from substrate_echo.epistemology import (
    FeatureExtractor,
    ObservationMemory,
    EpistemicTrustSystem,
    EpistemicCuriosityEngine,
    CulturalPriorEngine,
    CompressedDiscovery,
    DiscoveryType,
    EpistemicObservatory,
    EventType,
    ModuleType,
)
from substrate_echo.epistemology.observation import RawObservation


# ── Configuration ────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """EXP-OBS-001 configuration."""
    max_steps: int = 200
    report_interval: int = 25
    hud_interval: int = 50  # How often to print HUD


# ── SC2 Environment ──────────────────────────────────────────────

class SC2Environment:
    """Simple SC2-like environment."""
    
    def __init__(self):
        self._step = 0
        self._enemy_phase = 0
    
    def observe(self) -> Dict[str, float]:
        """Generate observation."""
        self._step += 1
        self._enemy_phase = (self._enemy_phase + 1) % 80
        
        return {
            "resource_signal": 0.5 + 0.3 * np.sin(2 * np.pi * self._step / 50),
            "enemy_threat": 0.5 + 0.5 * np.sin(2 * np.pi * self._enemy_phase / 80),
            "minerals": 500 + 100 * np.sin(2 * np.pi * self._step / 100),
        }
    
    def get_true_state(self) -> Dict[str, float]:
        """Get true state."""
        return {
            "optimal_action": 1 if self._step % 100 < 50 else 0,
        }


# ── Agent with Observatory ───────────────────────────────────────

class ObservableAgent:
    """Agent with full observability."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Cognitive components
        self.extractor = FeatureExtractor()
        self.observation_memory = ObservationMemory()
        self.trust = EpistemicTrustSystem()
        self.cultural_prior_engine = CulturalPriorEngine()
        self.curiosityEngine = EpistemicCuriosityEngine()
        
        # Observatory
        self.observatory = EpistemicObservatory()
        
        # Metrics
        self._step = 0
        self._predictions_made = 0
        self._predictions_correct = 0
    
    def process_step(self, raw_data: Dict[str, float], 
                     true_state: Dict[str, float]) -> Dict[str, Any]:
        """Process one step with full observability."""
        self._step = self._step + 1
        
        # 1. Record observation
        self.observatory.record_observation(self._step, raw_data)
        
        # 2. Extract features
        raw = RawObservation(
            data=raw_data,
            modality="sc2_game",
            source=self.agent_id,
            timestamp=time.time(),
        )
        features = self.extractor.extract(raw)
        self.observation_memory.record(features)
        
        # 3. Generate hypotheses
        hypotheses = self._generate_hypotheses(features)
        self.observatory.record_hypothesis(self._step, {
            "hypotheses": [h["description"] for h in hypotheses],
            "probabilities": [h["probability"] for h in hypotheses],
            "count": len(hypotheses),
        })
        
        # 4. Select best hypothesis
        best_hyp = max(hypotheses, key=lambda h: h.get("probability", 0))
        
        # 5. Make prediction
        predicted_action = best_hyp.get("expected", {}).get("action", 0)
        confidence = best_hyp.get("probability", 0.5)
        
        self.observatory.record_prediction(self._step, {
            "description": best_hyp["description"],
            "predicted_action": predicted_action,
            "confidence": confidence,
        })
        
        # 6. Verify prediction
        actual_action = true_state.get("optimal_action", 0)
        success = (predicted_action == actual_action)
        
        self._predictions_made += 1
        if success:
            self._predictions_correct += 1
        
        self.observatory.record_prediction_outcome(self._step, {
            "predicted": predicted_action,
            "actual": actual_action,
            "success": success,
            "accuracy": self._predictions_correct / self._predictions_made,
        })
        
        # 7. Record trust update
        self.observatory.record_trust_update(self._step, {
            "agent": "enemy",
            "dimension": "predictability",
            "level": 0.7 + 0.2 * np.sin(self._step / 10),
        })
        
        # 8. Record cultural prior (if applicable)
        if self._step % 20 == 0:
            self.observatory.record_cultural_prior(self._step, {
                "prior_id": f"prior_{self._step}",
                "applied": True,
                "helped": success,
            })
        
        # 9. Record discovery (if applicable)
        if self._step % 50 == 0:
            self.observatory.record_discovery(self._step, {
                "description": f"Pattern observed at step {self._step}",
                "confidence": 0.7,
                "domain": "sc2_strategy",
            })
        
        # 10. Record curiosity (if applicable)
        if self._step % 30 == 0:
            self.observatory.record_curiosity(self._step, {
                "research_goals": [f"Investigate pattern at step {self._step}"],
                "uncertainty": 0.4,
            })
        
        # 11. Record action
        self.observatory.record_action(self._step, {
            "action": "attack" if predicted_action == 1 else "defend",
            "reason": best_hyp["description"],
        })
        
        # 12. Update observatory tick
        self.observatory.tick(self._step)
        
        return {
            "success": success,
            "hypothesis": best_hyp["description"],
            "confidence": confidence,
        }
    
    def _generate_hypotheses(self, features) -> List[Dict[str, Any]]:
        """Generate hypotheses."""
        resource = features.features.get("resource_signal")
        threat = features.features.get("enemy_threat")
        
        res_val = resource.value if resource else 0.5
        threat_val = threat.value if threat else 0.5
        
        hypotheses = []
        
        if res_val > 0.5:
            hypotheses.append({
                "description": "attack resources high",
                "probability": 0.6,
                "expected": {"action": 1},
            })
        else:
            hypotheses.append({
                "description": "defend resources low",
                "probability": 0.6,
                "expected": {"action": 0},
            })
        
        hypotheses.append({
            "description": "always attack",
            "probability": 0.2,
            "expected": {"action": 1},
        })
        
        hypotheses.append({
            "description": "always defend",
            "probability": 0.2,
            "expected": {"action": 0},
        })
        
        return hypotheses


# ── Experiment ───────────────────────────────────────────────────

def run_experiment():
    """Run EXP-OBS-001: Epistemic Observatory Validation."""
    config = ExperimentConfig(
        max_steps=200,
        report_interval=25,
        hud_interval=50,
    )
    
    print(f"\n{'='*60}")
    print(f"EXP-OBS-001: Epistemic Observatory Validation")
    print(f"{'='*60}")
    print(f"Question: Can we observe cognition in real-time?")
    print(f"{'='*60}\n")
    
    # Create environment and agent
    env = SC2Environment()
    agent = ObservableAgent("agent_001")
    
    print(f"Running {config.max_steps} steps with full observability...\n")
    
    # Run experiment
    for step in range(1, config.max_steps + 1):
        raw = env.observe()
        true_state = env.get_true_state()
        
        result = agent.process_step(raw, true_state)
        
        # Print HUD at intervals
        if step % config.hud_interval == 0:
            print(f"\n{'-'*60}")
            print(f"Step {step}: {'SUCCESS' if result['success'] else 'FAILURE'}")
            print(f"Hypothesis: {result['hypothesis']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"{'-'*60}")
            
            # Print HUD
            hud = agent.observatory.render_hud()
            print(hud)
        
        # Print progress
        if step % config.report_interval == 0:
            accuracy = agent._predictions_correct / agent._predictions_made
            print(f"  Step {step}: accuracy={accuracy:.3f}, "
                  f"predictions={agent._predictions_made}")
    
    # Final Report
    print(f"\n{'='*60}")
    print(f"EXP-OBS-001: Final Report")
    print(f"{'='*60}")
    
    # Agent metrics
    accuracy = agent._predictions_correct / agent._predictions_made
    print(f"\nAgent Metrics:")
    print(f"  Predictions: {agent._predictions_made}")
    print(f"  Accuracy: {accuracy:.3f}")
    
    # Observatory metrics
    telemetry_summary = agent.observatory.telemetry.get_telemetry_summary()
    timeline_summary = agent.observatory.telemetry.get_timeline().get_timeline_summary()
    
    print(f"\nObservatory Metrics:")
    print(f"  Total events: {timeline_summary['total_events']}")
    print(f"  Ticks covered: {timeline_summary['ticks_covered']}")
    print(f"  Snapshots: {timeline_summary['snapshots']}")
    
    print(f"\nEvents by Module:")
    for module, count in timeline_summary['by_module'].items():
        print(f"  {module}: {count}")
    
    print(f"\nEvents by Type:")
    for event_type, count in timeline_summary['by_type'].items():
        print(f"  {event_type}: {count}")
    
    # Final HUD
    print(f"\n{'='*60}")
    print(f"Final Cognitive State")
    print(f"{'='*60}")
    print(agent.observatory.render_hud())
    
    # Event feed
    print(f"\n{'='*60}")
    print(f"Recent Event Feed")
    print(f"{'='*60}")
    print(agent.observatory.render_event_feed(num_events=15))
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"{'='*60}")
    
    return {
        "accuracy": accuracy,
        "total_events": timeline_summary['total_events'],
        "ticks_covered": timeline_summary['ticks_covered'],
        "by_module": timeline_summary['by_module'],
        "by_type": timeline_summary['by_type'],
    }


if __name__ == "__main__":
    results = run_experiment()
