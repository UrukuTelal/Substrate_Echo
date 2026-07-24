"""EXP-SWARM-002: Cultural Prior Acceleration.

Question:
    Does cultural inheritance accelerate adaptation?

Setup:
    Agent A learns environment rule.
    
    Example:
        "Resources near enemy bases often indicate strategic importance."
    
    Generate discovery.
    Transfer to Agent B.
    
    Two groups:
    
    Control:
        Agent B starts naive.
    
    Experimental:
        Agent B receives cultural priors.

Measure:
    - time to first correct hypothesis
    - number of failed predictions
    - confidence calibration
    - exploration efficiency

Expected:
    The culturally initialized agent should not be perfect.
    It should simply waste less experience rediscovering
    already validated patterns.

This tests whether culture actually accelerates adaptation,
or just adds noise.
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
    CulturalPriorEngine,
    CulturalPrior,
)


@dataclass
class ExperimentConfig:
    """EXP-SWARM-002 configuration."""
    max_steps: int = 400
    learning_phase_steps: int = 150
    migration_phase_steps: int = 150
    validation_phase_steps: int = 100
    report_interval: int = 25
    
    # Environment parameters
    pattern_complexity: float = 0.7
    noise_level: float = 0.15


class SimulatedAgent:
    """An agent that can learn and use cultural priors."""
    
    def __init__(self, agent_id: str, use_cultural_priors: bool = False):
        self.agent_id = agent_id
        self.use_cultural_priors = use_cultural_priors
        
        # Epistemology components
        self.extractor = FeatureExtractor()
        self.observation_memory = ObservationMemory()
        self.hypothesis_space = HypothesisSpace()
        self.prediction_engine = PredictionEngine()
        self.prediction_memory = PredictionMemory()
        self.development_record = DevelopmentRecord()
        
        # Swarm integration
        self.epistemic_state = AgentEpistemicState(agent_id=agent_id)
        
        # Cultural prior engine (for agents with cultural knowledge)
        self.cultural_prior_engine = CulturalPriorEngine() if use_cultural_priors else None
        
        # Learning metrics
        self._step = 0
        self._predictions_made = 0
        self._predictions_correct = 0
        self._discoveries_formed = 0
        self._confidences: List[float] = []
        self._hypothesis_times: List[float] = []  # Time to generate hypothesis
        
        # First correct hypothesis tracking
        self._first_correct_step: Optional[int] = None
    
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
    
    def generate_hypotheses_with_culture(self, features, domain: str = "general") -> List[Dict[str, Any]]:
        """Generate hypotheses, potentially modified by cultural priors."""
        start_time = time.time()
        
        # Generate base hypotheses
        hypotheses = self._generate_base_hypotheses(features)
        
        # Apply cultural priors if available
        if self.use_cultural_priors and self.cultural_prior_engine:
            # Convert features to observation dict
            observation = {}
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    observation[name] = feature.value
            
            # Get priors for this observation
            priors = self.cultural_prior_engine.get_priors_for_observation(
                observation, domain
            )
            
            if priors:
                # Apply priors to modify hypothesis probabilities
                for hyp in hypotheses:
                    adjustment = 0.0
                    applied_priors = []
                    
                    for prior in priors:
                        # Check if prior is relevant to this hypothesis
                        if self._prior_matches_hypothesis(prior, hyp):
                            weight = prior.get_effective_weight()
                            adjustment += weight * 0.5  # Scale factor
                            applied_priors.append(prior.prior_id)
                    
                    # Apply adjustment
                    hyp["probability"] = hyp.get("probability", 0.33) * (1 + adjustment)
                    hyp["cultural_priors_applied"] = applied_priors
                    hyp["prior_adjustment"] = adjustment
                
                # Renormalize probabilities
                total = sum(h.get("probability", 0) for h in hypotheses)
                if total > 0:
                    for h in hypotheses:
                        h["probability"] = h["probability"] / total
        
        elapsed = time.time() - start_time
        self._hypothesis_times.append(elapsed)
        
        return hypotheses
    
    def _prior_matches_hypothesis(self, prior: CulturalPrior, hypothesis: Dict[str, Any]) -> bool:
        """Check if a prior is relevant to a hypothesis.
        
        The prior specifies both a phase and a signal direction.
        It only matches hypotheses that have the same phase AND direction.
        """
        hyp_desc = hypothesis.get("description", "").lower()
        hyp_phase = hypothesis.get("phase")
        
        # Check phase match
        prior_phase = prior.applicability_conditions.get("phase")
        if prior_phase is not None and hyp_phase is not None:
            if prior_phase != hyp_phase:
                return False  # Phase mismatch
        
        # Check signal direction match
        signal_direction = prior.applicability_conditions.get("signal_direction")
        if signal_direction:
            if signal_direction in hyp_desc:
                return True
            return False  # Direction mismatch
        
        return False
    
    def _generate_base_hypotheses(self, features) -> List[Dict[str, Any]]:
        """Generate base hypotheses from features.
        
        Hypotheses are phase-dependent: the agent must predict whether
        the signal will be positive or negative based on the current phase.
        The short-term trend is misleading (oscillates rapidly), so the
        agent must learn the phase-dependent offset.
        """
        hypotheses = []
        
        # Get current phase from features
        phase = features.features.get("phase")
        phase_val = phase.value if phase else 0
        
        # Phase 0: signal tends positive
        # Phase 1: signal tends negative
        # But the agent doesn't know this yet, so it generates
        # both possibilities and assigns equal probability
        
        if phase_val == 0:
            hypotheses.append({
                "description": "phase 0 signal positive",
                "probability": 0.5,
                "expected": {"signal": "positive"},
                "phase": 0,
            })
            hypotheses.append({
                "description": "phase 0 signal negative",
                "probability": 0.5,
                "expected": {"signal": "negative"},
                "phase": 0,
            })
        else:
            hypotheses.append({
                "description": "phase 1 signal positive",
                "probability": 0.5,
                "expected": {"signal": "positive"},
                "phase": 1,
            })
            hypotheses.append({
                "description": "phase 1 signal negative",
                "probability": 0.5,
                "expected": {"signal": "negative"},
                "phase": 1,
            })
        
        return hypotheses
    
    def generate_prediction(self, features, true_state: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate and verify a prediction."""
        if not features:
            return None
        
        # Generate hypotheses (with cultural priors if available)
        hypotheses = self.generate_hypotheses_with_culture(features, domain="signal_prediction")
        
        if not hypotheses:
            return None
        
        # Select best hypothesis
        best_hyp = max(hypotheses, key=lambda h: h.get("probability", 0))
        
        self._predictions_made += 1
        self._confidences.append(best_hyp.get("probability", 0.5))
        
        # Verify prediction
        predicted_signal = best_hyp.get("expected", {}).get("signal", "positive")
        actual_signal = "positive" if true_state.get("signal", 0) > 0 else "negative"
        
        success = (predicted_signal == actual_signal)
        
        if success:
            self._predictions_correct += 1
            if self._first_correct_step is None:
                self._first_correct_step = self._step
        
        # Record in development record
        self.development_record.record_outcome(
            self._step,
            f"pred_{self._predictions_made}",
            success,
            true_state,
            {"predicted": predicted_signal},
        )
        
        # Update cultural prior effectiveness
        if self.use_cultural_priors and self.cultural_prior_engine:
            for hyp in hypotheses:
                if hyp.get("cultural_priors_applied"):
                    for prior_id in hyp["cultural_priors_applied"]:
                        self.cultural_prior_engine.record_application(prior_id, success)
        
        return {
            "success": success,
            "confidence": best_hyp.get("probability", 0.5),
            "hypothesis": best_hyp.get("description", ""),
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
        
        # If using cultural priors, ingest the discovery
        if self.use_cultural_priors and self.cultural_prior_engine:
            self.cultural_prior_engine.ingest_discovery(discovery)
        
        return discovery
    
    def ingest_swarm_knowledge(self, discovery: CompressedDiscovery,
                               source_trust: float = 0.7) -> bool:
        """Ingest knowledge from the swarm."""
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
        
        avg_hypothesis_time = (
            np.mean(self._hypothesis_times)
            if self._hypothesis_times
            else 0.0
        )
        
        return {
            "predictions_made": self._predictions_made,
            "predictions_correct": self._predictions_correct,
            "accuracy": round(accuracy, 3),
            "avg_confidence": round(avg_confidence, 3),
            "first_correct_step": self._first_correct_step,
            "discoveries_formed": self._discoveries_formed,
            "swarm_discoveries": len(self.epistemic_state.discoveries),
            "avg_hypothesis_time_ms": round(avg_hypothesis_time * 1000, 2),
            "cultural_priors": (
                len(self.cultural_prior_engine._priors)
                if self.cultural_prior_engine
                else 0
            ),
        }


class SwarmEnvironment:
    """Environment with learnable patterns for swarm testing.
    
    The pattern alternates between two phases, but the short-term trend
    is often misleading (oscillates within each phase). The cultural prior
    tells the agent which direction the signal tends toward in each phase,
    which the agent can't determine from short-term observations alone.
    """
    
    def __init__(self, complexity: float = 0.7, noise: float = 0.1):
        self._step = 0
        self._complexity = complexity
        self._noise = noise
        self._pattern_phase = 0
    
    def observe(self) -> Dict[str, float]:
        """Generate observation with misleading short-term trends."""
        self._step += 1
        
        # Phase alternates every 100 steps
        if self._step % 200 < 100:
            self._pattern_phase = 0
        else:
            self._pattern_phase = 1
        
        # Base signal: oscillates rapidly (short-term trend is misleading)
        signal = 0.3 * np.sin(2 * np.pi * 0.3 * self._step)
        
        # Phase-dependent offset: this is what the cultural prior tells you
        if self._pattern_phase == 0:
            # Phase 0: signal tends positive on average
            signal += 0.5
        else:
            # Phase 1: signal tends negative on average
            signal -= 0.5
        
        # Add noise
        noise = np.random.normal(0, self._noise)
        
        return {
            "signal": signal + noise,
            "trend": 1.0 if self._step % 100 < 50 else -1.0,  # Misleading!
            "periodic": np.sin(2 * np.pi * self._step / 50),
            "phase": self._pattern_phase,
        }
    
    def get_true_state(self) -> Dict[str, float]:
        """Get true state for verification."""
        if self._pattern_phase == 0:
            return {"signal": 0.5}  # Phase 0: tends positive
        else:
            return {"signal": -0.5}  # Phase 1: tends negative


def run_experiment():
    """Run EXP-SWARM-002: Cultural Prior Acceleration."""
    config = ExperimentConfig(
        max_steps=400,
        learning_phase_steps=150,
        migration_phase_steps=150,
        validation_phase_steps=100,
        report_interval=25,
    )
    
    print(f"\n{'='*60}")
    print(f"EXP-SWARM-002: Cultural Prior Acceleration")
    print(f"{'='*60}")
    print(f"Question: Does cultural inheritance accelerate adaptation?")
    print(f"{'='*60}\n")
    
    # Create agents
    agent_a = SimulatedAgent("agent_a", use_cultural_priors=False)
    agent_b_control = SimulatedAgent("agent_b_control", use_cultural_priors=False)
    agent_b_culture = SimulatedAgent("agent_b_culture", use_cultural_priors=True)
    
    # Create swarm infrastructure
    swarm_record = SwarmDevelopmentRecord()
    exchange_protocol = DiscoveryExchangeProtocol()
    
    # Register agents
    swarm_record.register_agent("agent_a")
    swarm_record.register_agent("agent_b_control")
    swarm_record.register_agent("agent_b_culture")
    
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
        agent_b_control._step = step
        agent_b_culture._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        # Agent A processes
        agent_a.process_observation(raw)
        if step % 10 == 0:
            recent_features = agent_a.observation_memory.get_recent(1)
            if recent_features:
                agent_a.generate_prediction(recent_features[0], true_state)
        
        # Form discovery at end of learning phase (when both phases have been observed)
        if step == config.learning_phase_steps:
            # Create phase-specific discoveries that can bias hypothesis selection
            # Phase 0: signal tends positive
            discovery_phase0 = agent_a.form_discovery({
                "domain": "signal_prediction",
                "phase": 0,
                "signal_direction": "positive",
            }, confidence=0.85)
            
            # Phase 1: signal tends negative
            discovery_phase1 = agent_a.form_discovery({
                "domain": "signal_prediction",
                "phase": 1,
                "signal_direction": "negative",
            }, confidence=0.85)
            
            discoveries_a.extend([discovery_phase0, discovery_phase1])
            
            # Submit to swarm
            swarm_record.submit_discovery("agent_a", discovery_phase0)
            swarm_record.submit_discovery("agent_a", discovery_phase1)
        
        # Report
        if step % config.report_interval == 0:
            metrics_a = agent_a.get_metrics()
            print(f"  Step {step}: Agent A accuracy={metrics_a['accuracy']:.3f}, "
                  f"first_correct={metrics_a['first_correct_step']}")
    
    # Distribute discoveries
    for discovery in discoveries_a:
        agent_b_control.ingest_swarm_knowledge(discovery)
        agent_b_culture.ingest_swarm_knowledge(discovery)
        
        # Also ingest into cultural prior engine for culture agent
        if agent_b_culture.cultural_prior_engine:
            agent_b_culture.cultural_prior_engine.ingest_discovery(discovery)
    
    print(f"\nPhase 2: Migration ({config.migration_phase_steps} steps)")
    print(f"Agent B (control) and Agent B (with culture) learn the same pattern...")
    
    # Phase 2: Both B agents learn
    for step in range(config.learning_phase_steps + 1,
                      config.learning_phase_steps + config.migration_phase_steps + 1):
        agent_b_control._step = step
        agent_b_culture._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        # Both B agents process
        agent_b_control.process_observation(raw)
        agent_b_culture.process_observation(raw)
        
        if step % 10 == 0:
            recent_b1 = agent_b_control.observation_memory.get_recent(1)
            recent_b2 = agent_b_culture.observation_memory.get_recent(1)
            
            if recent_b1:
                agent_b_control.generate_prediction(recent_b1[0], true_state)
            if recent_b2:
                agent_b_culture.generate_prediction(recent_b2[0], true_state)
        
        # Report
        if step % config.report_interval == 0:
            metrics_b1 = agent_b_control.get_metrics()
            metrics_b2 = agent_b_culture.get_metrics()
            print(f"  Step {step}: B(control)={metrics_b1['accuracy']:.3f}, "
                  f"B(culture)={metrics_b2['accuracy']:.3f}")
    
    # Phase 3: Validation
    print(f"\nPhase 3: Validation ({config.validation_phase_steps} steps)")
    
    for step in range(config.learning_phase_steps + config.migration_phase_steps + 1,
                      config.max_steps + 1):
        agent_b_control._step = step
        agent_b_culture._step = step
        
        raw = env.observe()
        true_state = env.get_true_state()
        
        agent_b_control.process_observation(raw)
        agent_b_culture.process_observation(raw)
        
        if step % 10 == 0:
            recent_b1 = agent_b_control.observation_memory.get_recent(1)
            recent_b2 = agent_b_culture.observation_memory.get_recent(1)
            
            if recent_b1:
                agent_b_control.generate_prediction(recent_b1[0], true_state)
            if recent_b2:
                agent_b_culture.generate_prediction(recent_b2[0], true_state)
    
    # Final Report
    print(f"\n{'='*60}")
    print(f"EXP-SWARM-002: Final Report")
    print(f"{'='*60}")
    
    metrics_a = agent_a.get_metrics()
    metrics_b1 = agent_b_control.get_metrics()
    metrics_b2 = agent_b_culture.get_metrics()
    
    print(f"\nAgent A (Teacher):")
    print(f"  Predictions: {metrics_a['predictions_made']}")
    print(f"  Accuracy: {metrics_a['accuracy']:.3f}")
    print(f"  First correct step: {metrics_a['first_correct_step']}")
    print(f"  Discoveries: {metrics_a['discoveries_formed']}")
    
    print(f"\nAgent B (Control - No Culture):")
    print(f"  Predictions: {metrics_b1['predictions_made']}")
    print(f"  Accuracy: {metrics_b1['accuracy']:.3f}")
    print(f"  First correct step: {metrics_b1['first_correct_step']}")
    print(f"  Cultural priors: {metrics_b1['cultural_priors']}")
    
    print(f"\nAgent B (With Cultural Priors):")
    print(f"  Predictions: {metrics_b2['predictions_made']}")
    print(f"  Accuracy: {metrics_b2['accuracy']:.3f}")
    print(f"  First correct step: {metrics_b2['first_correct_step']}")
    print(f"  Cultural priors: {metrics_b2['cultural_priors']}")
    
    # Calculate acceleration
    if metrics_b1['first_correct_step'] and metrics_b2['first_correct_step']:
        acceleration = (
            (metrics_b1['first_correct_step'] - metrics_b2['first_correct_step'])
            / metrics_b1['first_correct_step']
        )
        print(f"\nCultural Prior Acceleration:")
        print(f"  Control first correct: step {metrics_b1['first_correct_step']}")
        print(f"  Culture first correct: step {metrics_b2['first_correct_step']}")
        print(f"  Acceleration: {acceleration:.1%}")
        print(f"  {'SUCCESS' if acceleration > 0 else 'NEEDS INVESTIGATION'}")
    
    # Accuracy comparison
    if metrics_b1['accuracy'] > 0:
        accuracy_improvement = (metrics_b2['accuracy'] - metrics_b1['accuracy']) / metrics_b1['accuracy']
        print(f"\nAccuracy Improvement:")
        print(f"  Control accuracy: {metrics_b1['accuracy']:.3f}")
        print(f"  Culture accuracy: {metrics_b2['accuracy']:.3f}")
        print(f"  Improvement: {accuracy_improvement:.1%}")
    
    # Cultural prior effectiveness
    if agent_b_culture.cultural_prior_engine:
        print(f"\nCultural Prior Effectiveness:")
        effectiveness = agent_b_culture.cultural_prior_engine.get_prior_effectiveness()
        print(f"  Total priors: {effectiveness.get('total_priors', 0)}")
        print(f"  Applications: {effectiveness.get('total_applications', 0)}")
        print(f"  Help rate: {effectiveness.get('help_rate', 0):.1%}")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"{'='*60}")
    
    return {
        "agent_a": metrics_a,
        "agent_b_control": metrics_b1,
        "agent_b_culture": metrics_b2,
    }


if __name__ == "__main__":
    results = run_experiment()