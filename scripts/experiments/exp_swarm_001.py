"""EXP-SWARM-001: Knowledge Migration.

Question:
    Can knowledge discovered by one agent improve another agent
    that never experienced the original event?

Setup:
    1. Agent A encounters novel scenario.
    2. Agent A forms discovery.
    3. Discovery propagates through exchange protocol.
    4. Agent B encounters same scenario.
    5. Compare:

Without swarm knowledge:
    B learns from scratch

With swarm knowledge:
    B starts with cultural prior

Measure:
    - learning speed
    - prediction accuracy
    - confidence calibration
    - discovery retention

This is the moment where it stops being "multiple agents" and starts
becoming a **collective intelligence system**.
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
    SwarmDevelopmentRecord,
    AgentEpistemicState,
    CompressedDiscovery,
    DiscoveryType,
    DiscoveryExchangeProtocol,
    DiscoveryLineageSystem,
)


@dataclass
class ExperimentConfig:
    """EXP-SWARM-001 configuration."""
    max_steps: int = 500
    learning_phase_steps: int = 200
    migration_phase_steps: int = 200
    validation_phase_steps: int = 100
    report_interval: int = 50
    
    # Environment parameters
    pattern_complexity: float = 0.7
    noise_level: float = 0.1


class SimulatedAgent:
    """An agent that can learn and exchange discoveries."""
    
    def __init__(self, agent_id: str, has_swarm_knowledge: bool = False):
        self.agent_id = agent_id
        self.has_swarm_knowledge = has_swarm_knowledge
        
        # Epistemology components
        self.extractor = FeatureExtractor()
        self.observation_memory = ObservationMemory()
        self.hypothesis_space = HypothesisSpace()
        self.prediction_engine = PredictionEngine()
        self.prediction_memory = PredictionMemory()
        self.development_record = DevelopmentRecord()
        
        # Swarm integration
        self.epistemic_state = AgentEpistemicState(agent_id=agent_id)
        
        # Learning metrics
        self._step = 0
        self._predictions_made = 0
        self._predictions_correct = 0
        self._discoveries_formed = 0
        self._confidences: List[float] = []
    
    def process_observation(self, raw_data: Dict[str, float]) -> bool:
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
        
        return len(features.features) > 0
    
    def generate_prediction(self, features, true_state: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate and verify a prediction."""
        if not features:
            return None
        
        # Generate hypothesis
        trend = self.observation_memory.detect_trend("signal")
        if not trend:
            return None
        
        h = self.hypothesis_space.generate(
            description=f"Signal is {trend}",
            confidence=0.5,
            source=self.agent_id,
        )
        
        # Generate prediction using features
        predictions = self.prediction_engine.generate_from_features(
            features, trend=trend
        )
        
        if not predictions:
            return None
        
        prediction = predictions[0]
        self._predictions_made += 1
        self._confidences.append(prediction.confidence)
        
        # Create actual outcome from true state with matching keys
        actual_outcome = {}
        for key in prediction.expected.keys():
            if key in true_state:
                actual_outcome[key] = true_state[key]
            elif "signal" in key.lower():
                actual_outcome[key] = true_state.get("signal", 0.0)
            else:
                # Use the first numeric value from true state
                actual_outcome[key] = list(true_state.values())[0] if true_state else 0.0
        
        # Verify prediction
        status = prediction.verify(actual_outcome)
        success = status in (PredictionStatus.CONFIRMED, PredictionStatus.PARTIAL)
        
        if success:
            self._predictions_correct += 1
        
        # Record in development record
        self.development_record.record_outcome(
            self._step,
            prediction.id,
            success,
            actual_outcome,
            prediction.expected,
        )
        
        return {
            "success": success,
            "confidence": prediction.confidence,
            "status": status.value,
        }
    
    def form_discovery(self, pattern: Dict[str, Any], confidence: float) -> CompressedDiscovery:
        """Form a discovery from accumulated experience."""
        self._discoveries_formed += 1
        
        discovery = CompressedDiscovery(
            discovery_id=f"{self.agent_id}_disc_{self._discoveries_formed}",
            discovery_type=DiscoveryType.PATTERN,
            description=f"Pattern discovered by {self.agent_id}",
            pattern=pattern,
            confidence=confidence,
            evidence_count=self._predictions_correct,
            discovered_at=time.time(),
            discovered_by=self.agent_id,
        )
        
        # Add to epistemic state
        self.epistemic_state.discoveries.append(discovery)
        
        return discovery
    
    def ingest_swarm_knowledge(self, discovery: CompressedDiscovery,
                               source_trust: float = 0.7) -> bool:
        """Ingest knowledge from the swarm."""
        if not self.has_swarm_knowledge:
            return False
        
        return self.epistemic_state.ingest_discovery(discovery, source_trust)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get learning metrics."""
        accuracy = (
            self._predictions_correct / self._predictions_made
            if self._predictions_made > 0
            else 0.0
        )
        
        avg_confidence = (
            np.mean(self._confidences)
            if self._confidences
            else 0.0
        )
        
        return {
            "predictions_made": self._predictions_made,
            "predictions_correct": self._predictions_correct,
            "accuracy": round(accuracy, 3),
            "avg_confidence": round(avg_confidence, 3),
            "discoveries_formed": self._discoveries_formed,
            "swarm_discoveries": len(self.epistemic_state.discoveries),
        }


class SwarmEnvironment:
    """Environment with learnable patterns for swarm testing."""
    
    def __init__(self, complexity: float = 0.7, noise: float = 0.1):
        self._step = 0
        self._complexity = complexity
        self._noise = noise
        self._pattern_phase = 0
    
    def observe(self) -> Dict[str, float]:
        """Generate observation."""
        self._step += 1
        
        # Base signal with complexity
        signal = np.sin(2 * np.pi * 0.1 * self._step)
        
        # Add complexity based on phase
        if self._step % 200 < 100:
            self._pattern_phase = 0
            signal += 0.3 * np.sin(2 * np.pi * 0.05 * self._step)
        else:
            self._pattern_phase = 1
            signal += 0.3 * np.cos(2 * np.pi * 0.05 * self._step)
        
        # Add noise
        noise = np.random.normal(0, self._noise)
        
        return {
            "signal": signal + noise,
            "trend": 1.0 if self._step % 100 < 50 else -1.0,
            "periodic": np.sin(2 * np.pi * self._step / 50),
            "phase": self._pattern_phase,
        }
    
    def get_true_state(self) -> Dict[str, float]:
        """Get true state for verification."""
        signal = np.sin(2 * np.pi * 0.1 * self._step)
        
        if self._pattern_phase == 0:
            signal += 0.3 * np.sin(2 * np.pi * 0.05 * self._step)
        else:
            signal += 0.3 * np.cos(2 * np.pi * 0.05 * self._step)
        
        return {"signal": signal}


def run_experiment():
    """Run EXP-SWARM-001: Knowledge Migration."""
    config = ExperimentConfig(
        max_steps=500,
        learning_phase_steps=200,
        migration_phase_steps=200,
        validation_phase_steps=100,
        report_interval=50,
    )
    
    print(f"\n{'='*60}")
    print(f"EXP-SWARM-001: Knowledge Migration")
    print(f"{'='*60}")
    print(f"Question: Can knowledge discovered by one agent improve")
    print(f"another agent that never experienced the original event?")
    print(f"{'='*60}\n")
    
    # Create agents
    agent_a = SimulatedAgent("agent_a", has_swarm_knowledge=False)
    agent_b_no_swarm = SimulatedAgent("agent_b_no_swarm", has_swarm_knowledge=False)
    agent_b_with_swarm = SimulatedAgent("agent_b_with_swarm", has_swarm_knowledge=True)
    
    # Create swarm infrastructure
    swarm_record = SwarmDevelopmentRecord()
    exchange_protocol = DiscoveryExchangeProtocol()
    lineage_system = DiscoveryLineageSystem()
    
    # Register agents
    swarm_record.register_agent("agent_a")
    swarm_record.register_agent("agent_b_no_swarm")
    swarm_record.register_agent("agent_b_with_swarm")
    
    # Create environment
    env = SwarmEnvironment(
        complexity=config.pattern_complexity,
        noise=config.noise_level,
    )
    
    print(f"Phase 1: Learning ({config.learning_phase_steps} steps)")
    print(f"Agent A learns the pattern...")
    
    # Phase 1: Agent A learns
    discoveries_a = []
    for step in range(1, config.learning_phase_steps + 1):
        agent_a._step = step
        agent_b_no_swarm._step = step
        agent_b_with_swarm._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        # Agent A processes
        agent_a.process_observation(raw)
        if step % 10 == 0:
            # Get most recent features
            recent_features = agent_a.observation_memory.get_recent(1)
            if recent_features:
                agent_a.generate_prediction(
                    recent_features[0],
                    true_state
                )
        
        # Form discovery at midpoint
        if step == config.learning_phase_steps // 2:
            pattern = {
                "domain": "signal_prediction",
                "phase_0_behavior": "sinusoidal",
                "phase_1_behavior": "cosinusoidal",
                "transition_at": 100,
            }
            discovery = agent_a.form_discovery(pattern, confidence=0.85)
            discoveries_a.append(discovery)
            
            # Submit to swarm
            swarm_record.submit_discovery("agent_a", discovery)
            
            # Start lineage
            lineage_system.register_discovery(
                discovery.discovery_id,
                "agent_a",
                "Signal phase transition pattern",
                confidence=0.85
            )
        
        # Report
        if step % config.report_interval == 0:
            metrics_a = agent_a.get_metrics()
            print(f"  Step {step}: Agent A accuracy={metrics_a['accuracy']:.3f}, "
                  f"confidence={metrics_a['avg_confidence']:.3f}")
    
    # Distribute discoveries
    agents_dict = {
        "agent_b_no_swarm": agent_b_no_swarm.epistemic_state,
        "agent_b_with_swarm": agent_b_with_swarm.epistemic_state,
    }
    swarm_record.distribute_to_agents(agents_dict)
    
    print(f"\nPhase 2: Migration ({config.migration_phase_steps} steps)")
    print(f"Agent B (no swarm) and Agent B (with swarm) learn the same pattern...")
    
    # Phase 2: Both B agents learn
    for step in range(config.learning_phase_steps + 1,
                      config.learning_phase_steps + config.migration_phase_steps + 1):
        agent_b_no_swarm._step = step
        agent_b_with_swarm._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        # Both B agents process
        agent_b_no_swarm.process_observation(raw)
        agent_b_with_swarm.process_observation(raw)
        
        if step % 10 == 0:
            # Get most recent features
            recent_b1 = agent_b_no_swarm.observation_memory.get_recent(1)
            recent_b2 = agent_b_with_swarm.observation_memory.get_recent(1)
            
            if recent_b1:
                agent_b_no_swarm.generate_prediction(
                    recent_b1[0],
                    true_state
                )
            if recent_b2:
                agent_b_with_swarm.generate_prediction(
                    recent_b2[0],
                    true_state
                )
        
        # Report
        if step % config.report_interval == 0:
            metrics_b1 = agent_b_no_swarm.get_metrics()
            metrics_b2 = agent_b_with_swarm.get_metrics()
            print(f"  Step {step}: B(no swarm)={metrics_b1['accuracy']:.3f}, "
                  f"B(with swarm)={metrics_b2['accuracy']:.3f}")
    
    # Phase 3: Validation
    print(f"\nPhase 3: Validation ({config.validation_phase_steps} steps)")
    
    for step in range(config.learning_phase_steps + config.migration_phase_steps + 1,
                      config.max_steps + 1):
        agent_b_no_swarm._step = step
        agent_b_with_swarm._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        agent_b_no_swarm.process_observation(raw)
        agent_b_with_swarm.process_observation(raw)
        
        if step % 10 == 0:
            # Get most recent features
            recent_b1 = agent_b_no_swarm.observation_memory.get_recent(1)
            recent_b2 = agent_b_with_swarm.observation_memory.get_recent(1)
            
            if recent_b1:
                agent_b_no_swarm.generate_prediction(
                    recent_b1[0],
                    true_state
                )
            if recent_b2:
                agent_b_with_swarm.generate_prediction(
                    recent_b2[0],
                    true_state
                )
    
    # Final Report
    print(f"\n{'='*60}")
    print(f"EXP-SWARM-001: Final Report")
    print(f"{'='*60}")
    
    metrics_a = agent_a.get_metrics()
    metrics_b1 = agent_b_no_swarm.get_metrics()
    metrics_b2 = agent_b_with_swarm.get_metrics()
    
    print(f"\nAgent A (Teacher):")
    print(f"  Predictions: {metrics_a['predictions_made']}")
    print(f"  Accuracy: {metrics_a['accuracy']:.3f}")
    print(f"  Discoveries: {metrics_a['discoveries_formed']}")
    
    print(f"\nAgent B (No Swarm Knowledge):")
    print(f"  Predictions: {metrics_b1['predictions_made']}")
    print(f"  Accuracy: {metrics_b1['accuracy']:.3f}")
    print(f"  Swarm discoveries ingested: {metrics_b1['swarm_discoveries']}")
    
    print(f"\nAgent B (With Swarm Knowledge):")
    print(f"  Predictions: {metrics_b2['predictions_made']}")
    print(f"  Accuracy: {metrics_b2['accuracy']:.3f}")
    print(f"  Swarm discoveries ingested: {metrics_b2['swarm_discoveries']}")
    
    # Calculate improvement
    if metrics_b1['accuracy'] > 0:
        improvement = (metrics_b2['accuracy'] - metrics_b1['accuracy']) / metrics_b1['accuracy']
        print(f"\nKnowledge Migration Effect:")
        print(f"  Accuracy improvement: {improvement:.1%}")
        print(f"  {'SUCCESS' if improvement > 0 else 'NEEDS INVESTIGATION'}")
    
    # Swarm state
    print(f"\nSwarm Development Record:")
    swarm_metrics = swarm_record.get_knowledge_summary()
    print(f"  What we know: {swarm_metrics['what_we_know']}")
    print(f"  What we don't know: {swarm_metrics['what_we_dont_know']}")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"{'='*60}")
    
    return {
        "agent_a": metrics_a,
        "agent_b_no_swarm": metrics_b1,
        "agent_b_with_swarm": metrics_b2,
        "swarm": swarm_record.get_knowledge_summary(),
    }


if __name__ == "__main__":
    results = run_experiment()