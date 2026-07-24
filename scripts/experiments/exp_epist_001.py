"""EXP-EPIST-001: Epistemology Layer Validation.

Tests the full epistemology pipeline:
Observation → Hypothesis → Prediction → Outcome → Belief Update

Measures:
  - Prediction accuracy over time
  - Confidence calibration
  - Rule discovery
  - Recovery after failed predictions

Architecture:
  Simulated Environment
       |
  Feature Extraction
       |
  Hypothesis Generation
       |
  Prediction
       |
  Outcome Verification
       |
  Belief Update
       |
  Rule Discovery
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
    RawObservation,
    FeatureExtractor,
    FeatureSet,
    FeatureType,
    ObservationMemory,
    HypothesisSpace,
    Hypothesis,
    Evidence,
    PredictionRecord,
    PredictionEngine,
    PredictionMemory,
    PredictionStatus,
    RuleDiscoveryEngine,
    DevelopmentRecord,
    EventType,
    PerturbationEngine,
    Perturbation,
    PerturbationType,
    BaselineState,
)


@dataclass
class ExperimentConfig:
    """EXP-EPIST-001 configuration."""
    max_steps: int = 500
    observation_interval: int = 5     # Steps between observations
    hypothesis_interval: int = 20     # Steps between hypothesis generation
    prediction_interval: int = 10     # Steps between predictions
    rule_check_interval: int = 50     # Steps between rule discovery
    report_interval: int = 100        # Steps between reports
    
    # Environment parameters
    signal_frequency: float = 0.1    # Frequency of environmental signal
    noise_level: float = 0.1        # Noise in observations
    pattern_change_step: int = 250   # Step at which pattern changes


class SimulatedEnvironment:
    """Simulated environment with learnable patterns."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._step = 0
        self._phase = 0  # 0: stable, 1: pattern change
    
    def observe(self) -> RawObservation:
        """Generate observation from environment."""
        self._step += 1
        
        # Base signal
        signal = np.sin(2 * np.pi * self.config.signal_frequency * self._step)
        
        # Pattern change at midpoint
        if self._step >= self.config.pattern_change_step:
            self._phase = 1
            signal = np.cos(2 * np.pi * self.config.signal_frequency * self._step)
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_level)
        observed_value = signal + noise
        
        # Additional features
        trend = 1.0 if self._step % 100 < 50 else -1.0
        periodic = np.sin(2 * np.pi * self._step / 50)
        
        return RawObservation(
            data={
                "signal": observed_value,
                "trend": trend,
                "periodic": periodic,
                "step": self._step,
                "phase": self._phase,
            },
            modality="simulation",
            source="simulated_environment",
            timestamp=time.time(),
        )
    
    def get_true_state(self) -> Dict[str, Any]:
        """Get true state of environment (for verification)."""
        signal = np.sin(2 * np.pi * self.config.signal_frequency * self._step)
        if self._phase == 1:
            signal = np.cos(2 * np.pi * self.config.signal_frequency * self._step)
        
        return {
            "signal": signal,
            "phase": self._phase,
            "trend": 1.0 if self._step % 100 < 50 else -1.0,
        }


class EpistAgent:
    """Agent using epistemology layer to learn from observations."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Epistemology components
        self.extractor = FeatureExtractor()
        self.observation_memory = ObservationMemory()
        self.hypothesis_space = HypothesisSpace()
        self.prediction_engine = PredictionEngine()
        self.prediction_memory = PredictionMemory()
        self.rule_engine = RuleDiscoveryEngine()
        self.development_record = DevelopmentRecord()
        
        # State
        self._step = 0
        self._last_features: Optional[FeatureSet] = None
        
        # Metrics
        self._hypothesis_count = 0
        self._prediction_count = 0
        self._rule_count = 0
    
    def process_observation(self, raw: RawObservation) -> FeatureSet:
        """Process raw observation into features."""
        features = self.extractor.extract(raw)
        self.observation_memory.record(features)
        self._last_features = features
        
        # Record in development record
        self.development_record.record_observation(
            self._step,
            f"Observed signal={raw.data.get('signal', 0):.3f}",
            details=features.to_dict(),
        )
        
        return features
    
    def generate_hypotheses(self, features: FeatureSet) -> List[Hypothesis]:
        """Generate hypotheses from features."""
        hypotheses = []
        
        # Check for patterns in recent observations
        recent = self.observation_memory.get_recent(20)
        
        if len(recent) >= 10:
            # Hypothesis: signal is periodic
            signal_trend = self.observation_memory.detect_trend("signal")
            
            if signal_trend == "stable":
                h = self.hypothesis_space.generate(
                    description="Signal is stable with small oscillations",
                    confidence=0.6,
                    source="trend_analysis",
                )
                h.add_evidence(Evidence(
                    description="Signal trend is stable",
                    supports=True,
                    strength=0.7,
                    source="trend_detection",
                ))
                hypotheses.append(h)
            
            elif signal_trend == "increasing":
                h = self.hypothesis_space.generate(
                    description="Signal is increasing",
                    confidence=0.5,
                    source="trend_analysis",
                )
                h.add_evidence(Evidence(
                    description="Signal trend is increasing",
                    supports=True,
                    strength=0.6,
                    source="trend_detection",
                ))
                hypotheses.append(h)
            
            elif signal_trend == "decreasing":
                h = self.hypothesis_space.generate(
                    description="Signal is decreasing",
                    confidence=0.5,
                    source="trend_analysis",
                )
                h.add_evidence(Evidence(
                    description="Signal trend is decreasing",
                    supports=True,
                    strength=0.6,
                    source="trend_detection",
                ))
                hypotheses.append(h)
        
        # Record in development record
        for h in hypotheses:
            self.development_record.record_hypothesis(
                self._step,
                f"Generated hypothesis: {h.description[:50]}",
                h.id,
                h.confidence,
            )
            self._hypothesis_count += 1
        
        return hypotheses
    
    def generate_predictions(self, hypotheses: List[Hypothesis],
                            features: FeatureSet) -> List[PredictionRecord]:
        """Generate predictions from hypotheses."""
        predictions = []
        
        for h in hypotheses:
            preds = self.prediction_engine.generate_from_hypothesis(
                h, features, time_horizon=20
            )
            predictions.extend(preds)
            
            # Record in development record
            for p in preds:
                self.development_record.record_prediction(
                    self._step,
                    f"Prediction: {p.description[:50]}",
                    p.id,
                    h.id,
                )
                self._prediction_count += 1
        
        # Also generate predictions from trends
        if self._last_features:
            trend = self.observation_memory.detect_trend("signal")
            if trend:
                trend_preds = self.prediction_engine.generate_from_features(
                    self._last_features, trend
                )
                predictions.extend(trend_preds)
        
        # Record predictions
        for p in predictions:
            self.prediction_memory.record(p)
        
        return predictions
    
    def verify_predictions(self, actual_outcome: Dict[str, Any]):
        """Verify pending predictions against actual outcome."""
        pending = self.prediction_memory.get_pending(self._step)
        
        for prediction in pending:
            status = prediction.verify(actual_outcome)
            
            # Record in development record
            success = status in (PredictionStatus.CONFIRMED, PredictionStatus.PARTIAL)
            self.development_record.record_outcome(
                self._step,
                prediction.id,
                success,
                actual_outcome,
                prediction.expected,
            )
            
            # Update hypothesis based on prediction outcome
            if prediction.source_hypothesis:
                hypothesis = self.hypothesis_space.get(prediction.source_hypothesis)
                if hypothesis:
                    evidence = Evidence(
                        description=f"Prediction {prediction.id}: {status.value}",
                        supports=success,
                        strength=0.7 if success else 0.8,
                        source="prediction_verification",
                    )
                    hypothesis.add_evidence(evidence)
    
    def discover_rules(self):
        """Discover rules from observation history."""
        if self._last_features:
            self.rule_engine.observe(self._last_features)
        
        new_rules = self.rule_engine.discover_rules()
        
        for rule in new_rules:
            self.development_record.record_rule_learned(
                self._step,
                rule.id,
                f"Discovered rule: {rule.description[:50]}",
            )
            self._rule_count += 1
    
    def tick(self):
        """Advance agent state."""
        self._step += 1
        
        # Take periodic snapshots
        if self._step % 50 == 0:
            self.development_record.take_snapshot(
                self._step,
                {h.id: h.confidence for h in self.hypothesis_space.get_active()},
                {r.id: r.confidence for r in self.rule_engine.get_rules()},
                self.prediction_memory.get_recent_accuracy(),
            )


def run_experiment():
    """Run EXP-EPIST-001."""
    config = ExperimentConfig(
        max_steps=500,
        observation_interval=5,
        hypothesis_interval=20,
        prediction_interval=10,
        report_interval=100,
    )
    
    env = SimulatedEnvironment(config)
    agent = EpistAgent(config)
    
    print(f"\n{'='*60}")
    print(f"EXP-EPIST-001: Epistemology Layer Validation")
    print(f"{'='*60}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Pattern Change: Step {config.pattern_change_step}")
    print(f"{'='*60}\n")
    
    for step in range(1, config.max_steps + 1):
        agent._step = step
        
        # 1. Observe environment
        raw = env.observe()
        features = agent.process_observation(raw)
        
        # 2. Generate hypotheses periodically
        if step % config.hypothesis_interval == 0:
            hypotheses = agent.generate_hypotheses(features)
        
        # 3. Generate predictions periodically
        if step % config.prediction_interval == 0:
            active_hypotheses = agent.hypothesis_space.get_active()
            predictions = agent.generate_predictions(active_hypotheses, features)
        
        # 4. Verify predictions
        true_state = env.get_true_state()
        agent.verify_predictions(true_state)
        
        # 5. Discover rules periodically
        if step % config.rule_check_interval == 0:
            agent.discover_rules()
        
        # 6. Report
        if step % config.report_interval == 0:
            accuracy_stats = agent.prediction_memory.get_accuracy_stats()
            print(f"\n--- Step {step} ---")
            print(f"  Hypotheses: {len(agent.hypothesis_space.get_active())}")
            print(f"  Predictions: {accuracy_stats['total']} verified, "
                  f"accuracy={accuracy_stats['accuracy']:.3f}")
            print(f"  Rules: {len(agent.rule_engine.get_rules())}")
            print(f"  True State: signal={true_state['signal']:.3f}, phase={true_state['phase']}")
        
        # 7. Agent tick
        agent.tick()
    
    # Final report
    print(f"\n{'='*60}")
    print(f"EXP-EPIST-001: Final Report")
    print(f"{'='*60}")
    
    accuracy_stats = agent.prediction_memory.get_accuracy_stats()
    print(f"\nPrediction Accuracy:")
    print(f"  Total verified: {accuracy_stats['total']}")
    print(f"  Confirmed: {accuracy_stats['confirmed']}")
    print(f"  Failed: {accuracy_stats['failed']}")
    print(f"  Overall accuracy: {accuracy_stats['accuracy']:.3f}")
    
    print(f"\nHypothesis Space:")
    print(f"  Total generated: {agent._hypothesis_count}")
    print(f"  Active: {len(agent.hypothesis_space.get_active())}")
    print(f"  Confirmed: {len([h for h in agent.hypothesis_space._hypotheses.values() if h.confidence > 0.8])}")
    
    print(f"\nRule Discovery:")
    print(f"  Rules discovered: {agent._rule_count}")
    print(f"  Validated rules: {len(agent.rule_engine.get_validated_rules())}")
    
    print(f"\nDevelopment Record:")
    summary = agent.development_record.get_summary()
    print(f"  Total events: {summary['total_events']}")
    print(f"  Failed predictions (valuable data): {summary['failed_predictions']}")
    print(f"  Significant events: {summary['significant_events']}")
    
    # Show learning trajectory
    trajectory = agent.development_record.get_learning_trajectory()
    if trajectory:
        print(f"\nLearning Trajectory:")
        for t in trajectory[::max(1, len(trajectory)//5)]:  # Sample 5 points
            print(f"  Step {t['step']}: confidence={t['confidence']:.3f}, "
                  f"accuracy={t['accuracy']:.3f}, rules={t['rules']}")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"{'='*60}")
    
    return {
        "accuracy": accuracy_stats,
        "hypotheses": agent._hypothesis_count,
        "rules": agent._rule_count,
        "events": summary['total_events'],
    }


if __name__ == "__main__":
    results = run_experiment()
    print(f"\nResults: {results}")