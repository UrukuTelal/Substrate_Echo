"""EXP-EPIST-002: Confidence Calibration.

Goal: Determine if confidence matches reality.
A system saying 90% confidence should be correct ~90% of the time.

Measures:
  - Prediction confidence vs actual success rate
  - Calibration error over time
  - Effect of council interventions on calibration

Architecture:
  Multiple Simulated Environments
       |
  Parallel Agents with Different Confidence Profiles
       |
  Prediction Tracking
       |
  Calibration Analysis
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
    HypothesisSpace,
    PredictionEngine,
    PredictionMemory,
    PredictionStatus,
    DevelopmentRecord,
    EventType,
    CalibrationCouncil,
)


@dataclass
class ExperimentConfig:
    """EXP-EPIST-002 configuration."""
    max_steps: int = 500
    n_agents: int = 3
    observation_interval: int = 5
    prediction_interval: int = 10
    calibration_interval: int = 50
    report_interval: int = 100
    
    # Environment parameters
    signal_noise: float = 0.1
    pattern_change_step: int = 250


class CalibrationAgent:
    """Agent with confidence tracking for calibration analysis."""
    
    def __init__(self, agent_id: str, confidence_modifier: float = 1.0):
        self.agent_id = agent_id
        self.confidence_modifier = confidence_modifier  # >1 = overconfident, <1 = underconfident
        
        self.extractor = FeatureExtractor()
        self.observation_memory = ObservationMemory()
        self.hypothesis_space = HypothesisSpace()
        self.prediction_engine = PredictionEngine()
        self.prediction_memory = PredictionMemory()
        self.development_record = DevelopmentRecord()
        
        self._step = 0
        self._last_features = None
        
        # Calibration tracking
        self._confidence_history: List[float] = []
        self._accuracy_history: List[float] = []
    
    def process_observation(self, raw_data: Dict[str, float]):
        """Process observation and generate features."""
        from substrate_echo.epistemology.observation import RawObservation
        
        raw = RawObservation(
            data=raw_data,
            modality="simulation",
            source=self.agent_id,
            timestamp=time.time(),
        )
        
        features = self.extractor.extract(raw)
        self.observation_memory.record(features)
        self._last_features = features
        
        return features
    
    def generate_predictions(self, features):
        """Generate predictions from current state."""
        if not features or not self._last_features:
            return []
        
        # Generate hypothesis
        trend = self.observation_memory.detect_trend("signal")
        if trend:
            h = self.hypothesis_space.generate(
                description=f"Signal is {trend}",
                confidence=0.5 * self.confidence_modifier,
                source="calibration_test",
            )
            
            # Generate predictions
            predictions = self.prediction_engine.generate_from_hypothesis(
                h, features, time_horizon=20
            )
            
            # Apply confidence modifier
            for p in predictions:
                p.confidence = min(1.0, p.confidence * self.confidence_modifier)
                self.prediction_memory.record(p)
            
            return predictions
        
        return []
    
    def verify_predictions(self, actual_outcome: Dict[str, float]):
        """Verify predictions and track calibration."""
        pending = self.prediction_memory.get_pending(self._step)
        
        for prediction in pending:
            status = prediction.verify(actual_outcome)
            
            # Track confidence vs accuracy
            self._confidence_history.append(prediction.confidence)
            self._accuracy_history.append(1.0 if status == PredictionStatus.CONFIRMED else 0.0)
            
            # Record in development record
            success = status in (PredictionStatus.CONFIRMED, PredictionStatus.PARTIAL)
            self.development_record.record_outcome(
                self._step,
                prediction.id,
                success,
                actual_outcome,
                prediction.expected,
            )
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Calculate calibration metrics."""
        if not self._confidence_history:
            return {"error": 0.0, "n": 0}
        
        # Group by confidence bins
        n_bins = 10
        bins = {}
        
        for conf, acc in zip(self._confidence_history, self._accuracy_history):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            if bin_idx not in bins:
                bins[bin_idx] = {"confidences": [], "accuracies": []}
            bins[bin_idx]["confidences"].append(conf)
            bins[bin_idx]["accuracies"].append(acc)
        
        # Calculate Expected Calibration Error (ECE)
        total_error = 0.0
        total_samples = len(self._confidence_history)
        
        bin_details = {}
        for bin_idx, data in bins.items():
            avg_confidence = np.mean(data["confidences"])
            avg_accuracy = np.mean(data["accuracies"])
            bin_count = len(data["accuracies"])
            
            bin_error = abs(avg_confidence - avg_accuracy)
            total_error += bin_error * bin_count
            
            bin_details[bin_idx] = {
                "confidence": round(avg_confidence, 3),
                "accuracy": round(avg_accuracy, 3),
                "error": round(bin_error, 3),
                "count": bin_count,
            }
        
        ece = total_error / total_samples if total_samples > 0 else 0.0
        
        return {
            "ece": round(ece, 4),
            "n_predictions": total_samples,
            "avg_confidence": round(np.mean(self._confidence_history), 3),
            "avg_accuracy": round(np.mean(self._accuracy_history), 3),
            "bins": bin_details,
        }


class SimulatedEnvironment:
    """Environment with learnable patterns."""
    
    def __init__(self, noise: float = 0.1):
        self._step = 0
        self._phase = 0
        self._noise = noise
    
    def observe(self) -> Dict[str, float]:
        """Generate observation."""
        self._step += 1
        
        # Base signal
        signal = np.sin(2 * np.pi * 0.1 * self._step)
        
        # Pattern change at midpoint
        if self._step >= 250:
            self._phase = 1
            signal = np.cos(2 * np.pi * 0.1 * self._step)
        
        # Add noise
        noise = np.random.normal(0, self._noise)
        
        return {
            "signal": signal + noise,
            "trend": 1.0 if self._step % 100 < 50 else -1.0,
            "periodic": np.sin(2 * np.pi * self._step / 50),
        }
    
    def get_true_state(self) -> Dict[str, float]:
        """Get true state for verification."""
        signal = np.sin(2 * np.pi * 0.1 * self._step)
        if self._phase == 1:
            signal = np.cos(2 * np.pi * 0.1 * self._step)
        
        return {"signal": signal}


def run_experiment():
    """Run EXP-EPIST-002."""
    config = ExperimentConfig(
        max_steps=500,
        n_agents=3,
        report_interval=100,
    )
    
    print(f"\n{'='*60}")
    print(f"EXP-EPIST-002: Confidence Calibration")
    print(f"{'='*60}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Agents: {config.n_agents}")
    print(f"{'='*60}\n")
    
    # Create agents with different confidence profiles
    agents = [
        CalibrationAgent("calibrated", confidence_modifier=1.0),
        CalibrationAgent("overconfident", confidence_modifier=1.5),
        CalibrationAgent("underconfident", confidence_modifier=0.6),
    ]
    
    env = SimulatedEnvironment(noise=config.signal_noise)
    
    for step in range(1, config.max_steps + 1):
        # Observe environment
        raw = env.observe()
        true_state = env.get_true_state()
        
        # Process observations and generate predictions for each agent
        for agent in agents:
            agent._step = step
            features = agent.process_observation(raw)
            
            if step % config.prediction_interval == 0:
                agent.generate_predictions(features)
            
            agent.verify_predictions(true_state)
        
        # Report
        if step % config.report_interval == 0:
            print(f"\n--- Step {step} ---")
            for agent in agents:
                metrics = agent.get_calibration_metrics()
                print(f"  {agent.agent_id}: ECE={metrics['ece']:.4f}, "
                      f"n={metrics['n_predictions']}, "
                      f"avg_conf={metrics.get('avg_confidence', 0):.3f}, "
                      f"avg_acc={metrics.get('avg_accuracy', 0):.3f}")
    
    # Final report
    print(f"\n{'='*60}")
    print(f"EXP-EPIST-002: Final Report")
    print(f"{'='*60}")
    
    for agent in agents:
        metrics = agent.get_calibration_metrics()
        print(f"\n{agent.agent_id}:")
        print(f"  ECE (calibration error): {metrics['ece']:.4f}")
        print(f"  Total predictions: {metrics['n_predictions']}")
        print(f"  Average confidence: {metrics.get('avg_confidence', 0):.3f}")
        print(f"  Average accuracy: {metrics.get('avg_accuracy', 0):.3f}")
        print(f"  Confidence-Accuracy gap: {abs(metrics.get('avg_confidence', 0) - metrics.get('avg_accuracy', 0)):.3f}")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"{'='*60}")
    
    # Return results for comparison
    results = {}
    for agent in agents:
        results[agent.agent_id] = agent.get_calibration_metrics()
    
    return results


if __name__ == "__main__":
    results = run_experiment()