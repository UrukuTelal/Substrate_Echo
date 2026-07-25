"""Epistemic Chain Recorder — Full observation→action→outcome trace.

Records every epistemic transition in the cognition pipeline so that
post-game analysis can answer: "Show me the reasoning chain that led to defeat."

Architecture:
    Observation (raw)
        ↓
    Feature Extraction (encoded 16D)
        ↓
    Hypothesis Formation
        ↓
    Prediction
        ↓
    Action Selection
        ↓
    Outcome (next observation)
        ↓
    Belief Update

Each transition is recorded as a ChainLink with:
    - tick
    - input state
    - output state
    - confidence
    - latency
    - anomalies
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time


class TransitionType(Enum):
    """Types of epistemic transitions."""
    OBSERVATION = "observation"
    FEATURE_EXTRACTION = "feature_extraction"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    PREDICTION = "prediction"
    ACTION_SELECTION = "action_selection"
    OUTCOME = "outcome"
    BELIEF_UPDATE = "belief_update"


class AnomalyType(Enum):
    """Types of anomalies detected in the chain."""
    STATE_MISMATCH = "state_mismatch"       # observation != expected
    PREDICTION_FAILURE = "prediction_failure"  # prediction != outcome
    CONFIDENCE_STALE = "confidence_stale"    # confidence unchanged after failure
    ACTION_DEGENERATE = "action_degenerate"  # same action repeated N times
    EPISTEMIC_SILENCE = "epistemic_silence"  # no hypothesis/prediction generated
    REWARD_STALE = "reward_stale"          # reward not applied
    OBSERVATION_GAP = "observation_gap"      # data source disagreement


@dataclass
class ChainLink:
    """A single step in the epistemic chain."""
    tick: int
    transition: TransitionType
    timestamp: float = 0.0

    # Input/output data
    input_state: Dict[str, Any] = field(default_factory=dict)
    output_state: Dict[str, Any] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0

    # Latency
    latency_ms: float = 0.0

    # Anomalies detected at this link
    anomalies: List[AnomalyType] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "transition": self.transition.value,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "anomalies": [a.value for a in self.anomalies],
            "input": self.input_state,
            "output": self.output_state,
            "metadata": self.metadata,
        }


@dataclass
class PredictionRecord:
    """A prediction made at some tick, to be verified later."""
    tick: int
    prediction: Dict[str, Any]
    confidence: float
    expected_tick: int  # when we expect to verify


@dataclass
class OutcomeRecord:
    """The actual outcome at some tick."""
    tick: int
    actual: Dict[str, Any]


class EpistemicChainRecorder:
    """Records the full epistemic chain for post-game analysis.

    Usage:
        recorder = EpistemicChainRecorder()

        # At each tick, record transitions
        recorder.record_observation(tick, raw_game_state, encoded_vector)
        recorder.record_features(tick, features)
        recorder.record_hypothesis(tick, hypothesis, confidence)
        recorder.record_prediction(tick, prediction, confidence, verify_at_tick)
        recorder.record_action(tick, action_type, action_vector)
        recorder.record_outcome(tick, actual_outcome)
        recorder.record_belief_update(tick, old_confidence, new_confidence, reason)

        # Detect anomalies
        anomalies = recorder.detect_anomalies()

        # Post-game analysis
        chain = recorder.get_chain()
        failure_chain = recorder.get_failure_chain(tick)
        summary = recorder.get_summary()
    """

    def __init__(self):
        self._chain: List[ChainLink] = []
        self._predictions: List[PredictionRecord] = []
        self._outcomes: List[OutcomeRecord] = []
        self._anomalies: List[Tuple[int, AnomalyType, str]] = []

        # Tracking for anomaly detection
        self._last_action: Optional[str] = None
        self._action_repeat_count: int = 0
        self._last_hypothesis_tick: int = -100
        self._last_prediction_tick: int = -100
        self._confidence_history: Dict[str, List[Tuple[int, float]]] = {}

    def record_observation(self, tick: int,
                           raw_state: Dict[str, Any],
                           encoded_vector: List[float],
                           source: str = "unknown"):
        """Record raw observation entering the pipeline."""
        link = ChainLink(
            tick=tick,
            transition=TransitionType.OBSERVATION,
            timestamp=time.time(),
            input_state={
                "raw_minerals": raw_state.get("minerals", 0),
                "raw_workers": raw_state.get("workers", 0),
                "raw_army": raw_state.get("army", 0),
                "raw_supply_used": raw_state.get("supply_used", 0),
                "raw_supply_cap": raw_state.get("supply_cap", 0),
                "source": source,
            },
            output_state={
                "vector_norm": float(sum(v**2 for v in encoded_vector) ** 0.5),
                "vector_dimensions": len(encoded_vector),
            },
        )
        self._chain.append(link)

    def record_features(self, tick: int,
                        features: Dict[str, Any]):
        """Record feature extraction output."""
        link = ChainLink(
            tick=tick,
            transition=TransitionType.FEATURE_EXTRACTION,
            timestamp=time.time(),
            input_state=features.get("input", {}),
            output_state=features.get("output", {}),
            metadata=features.get("metadata", {}),
        )
        self._chain.append(link)

    def record_hypothesis(self, tick: int,
                          hypothesis: str,
                          confidence: float,
                          supporting_evidence: int = 0,
                          contradicting_evidence: int = 0):
        """Record hypothesis formation."""
        self._last_hypothesis_tick = tick

        anomalies = []
        if tick - self._last_hypothesis_tick > 50:
            anomalies.append(AnomalyType.EPISTEMIC_SILENCE)

        link = ChainLink(
            tick=tick,
            transition=TransitionType.HYPOTHESIS_FORMATION,
            timestamp=time.time(),
            input_state={"previous_tick": self._last_hypothesis_tick},
            output_state={
                "hypothesis": hypothesis,
                "supporting_evidence": supporting_evidence,
                "contradicting_evidence": contradicting_evidence,
            },
            confidence=confidence,
            anomalies=anomalies,
        )
        self._chain.append(link)

    def record_prediction(self, tick: int,
                          prediction: Dict[str, Any],
                          confidence: float,
                          verify_at_tick: int):
        """Record a prediction to be verified later."""
        self._last_prediction_tick = tick

        self._predictions.append(PredictionRecord(
            tick=tick,
            prediction=prediction,
            confidence=confidence,
            expected_tick=verify_at_tick,
        ))

        link = ChainLink(
            tick=tick,
            transition=TransitionType.PREDICTION,
            timestamp=time.time(),
            output_state=prediction,
            confidence=confidence,
        )
        self._chain.append(link)

    def record_action(self, tick: int,
                      action_type: str,
                      action_vector: List[float],
                      decision_source: str = "kernel"):
        """Record action selection."""
        anomalies = []

        # Check for degenerate action (same action repeated)
        if action_type == self._last_action:
            self._action_repeat_count += 1
            if self._action_repeat_count >= 20:
                anomalies.append(AnomalyType.ACTION_DEGENERATE)
        else:
            self._action_repeat_count = 0
        self._last_action = action_type

        link = ChainLink(
            tick=tick,
            transition=TransitionType.ACTION_SELECTION,
            timestamp=time.time(),
            input_state={"decision_source": decision_source},
            output_state={
                "action_type": action_type,
                "action_norm": float(sum(v**2 for v in action_vector) ** 0.5) if action_vector else 0.0,
            },
            anomalies=anomalies,
        )
        self._chain.append(link)

    def record_outcome(self, tick: int,
                       actual_state: Dict[str, Any]):
        """Record the actual outcome (next observation)."""
        self._outcomes.append(OutcomeRecord(
            tick=tick,
            actual=actual_state,
        ))

        # Check predictions against outcome
        anomalies = []
        for pred in self._predictions:
            if pred.expected_tick <= tick:
                # Verify prediction
                match = self._verify_prediction(pred.prediction, actual_state)
                if not match:
                    anomalies.append(AnomalyType.PREDICTION_FAILURE)
                    self._anomalies.append((
                        tick, AnomalyType.PREDICTION_FAILURE,
                        f"Prediction from tick {pred.tick} failed at tick {tick}"
                    ))

        # Remove verified predictions
        self._predictions = [p for p in self._predictions if p.expected_tick > tick]

        link = ChainLink(
            tick=tick,
            transition=TransitionType.OUTCOME,
            timestamp=time.time(),
            output_state=actual_state,
            anomalies=anomalies,
        )
        self._chain.append(link)

    def record_belief_update(self, tick: int,
                             old_confidence: float,
                             new_confidence: float,
                             reason: str = ""):
        """Record a belief update (hypothesis revision)."""
        anomalies = []

        # Check for stale confidence after prediction failure
        if abs(new_confidence - old_confidence) < 0.01:
            recent_failures = [
                a for a in self._anomalies
                if a[0] >= tick - 50 and a[1] == AnomalyType.PREDICTION_FAILURE
            ]
            if recent_failures:
                anomalies.append(AnomalyType.CONFIDENCE_STALE)
                self._anomalies.append((
                    tick, AnomalyType.CONFIDENCE_STALE,
                    f"Confidence unchanged ({old_confidence:.3f}→{new_confidence:.3f}) "
                    f"after prediction failure"
                ))

        link = ChainLink(
            tick=tick,
            transition=TransitionType.BELIEF_UPDATE,
            timestamp=time.time(),
            input_state={"old_confidence": old_confidence, "reason": reason},
            output_state={"new_confidence": new_confidence},
            confidence=new_confidence,
            anomalies=anomalies,
        )
        self._chain.append(link)

    def _verify_prediction(self, prediction: Dict[str, Any],
                           actual: Dict[str, Any]) -> bool:
        """Check if prediction matches actual outcome."""
        # Simple: check if key predicted values are within tolerance
        for key, predicted_val in prediction.items():
            if key in actual:
                actual_val = actual[key]
                if isinstance(predicted_val, (int, float)) and isinstance(actual_val, (int, float)):
                    tolerance = abs(predicted_val) * 0.2 + 0.1  # 20% tolerance
                    if abs(predicted_val - actual_val) > tolerance:
                        return False
        return True

    def detect_anomalies(self) -> List[Tuple[int, AnomalyType, str]]:
        """Detect anomalies across the full chain."""
        anomalies = list(self._anomalies)

        # Check for state mismatches between consecutive observations
        observations = [l for l in self._chain if l.transition == TransitionType.OBSERVATION]
        for i in range(1, len(observations)):
            prev = observations[i-1].input_state
            curr = observations[i].input_state
            prev_workers = prev.get("raw_workers", 0)
            curr_workers = curr.get("raw_workers", 0)
            if prev_workers > 0 and curr_workers == 0:
                anomalies.append((
                    observations[i].tick,
                    AnomalyType.STATE_MISMATCH,
                    f"Workers dropped from {prev_workers} to {curr_workers}"
                ))

        self._anomalies = anomalies
        return anomalies

    def get_chain(self) -> List[ChainLink]:
        """Get the full epistemic chain."""
        return list(self._chain)

    def get_chain_for_tick(self, tick: int,
                           window: int = 5) -> List[ChainLink]:
        """Get chain links around a specific tick."""
        return [l for l in self._chain
                if tick - window <= l.tick <= tick + window]

    def get_failure_chain(self, failure_tick: int) -> List[ChainLink]:
        """Get the reasoning chain leading up to a failure."""
        # Find the last prediction before failure
        last_pred_tick = 0
        for link in self._chain:
            if link.transition == TransitionType.PREDICTION and link.tick <= failure_tick:
                last_pred_tick = link.tick

        # Return chain from last prediction to failure
        return [l for l in self._chain
                if last_pred_tick <= l.tick <= failure_tick]

    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken."""
        counts = {}
        for link in self._chain:
            if link.transition == TransitionType.ACTION_SELECTION:
                action = link.output_state.get("action_type", "unknown")
                counts[action] = counts.get(action, 0) + 1
        return counts

    @property
    def anomalies(self) -> List[Tuple[int, AnomalyType, str]]:
        """Public alias for internal anomaly list."""
        return self._anomalies

    def get_anomaly_summary(self) -> Dict[str, int]:
        """Get count of each anomaly type."""
        counts = {}
        for _, anomaly_type, _ in self._anomalies:
            counts[anomaly_type.value] = counts.get(anomaly_type.value, 0) + 1
        return counts

    def get_summary(self) -> Dict[str, Any]:
        """Get full summary of the epistemic chain."""
        anomalies = self.detect_anomalies()
        action_dist = self.get_action_distribution()

        transitions = {}
        for link in self._chain:
            t = link.transition.value
            transitions[t] = transitions.get(t, 0) + 1

        return {
            "total_ticks": self._chain[-1].tick if self._chain else 0,
            "total_links": len(self._chain),
            "transitions": transitions,
            "actions": action_dist,
            "anomalies": self.get_anomaly_summary(),
            "total_anomalies": len(anomalies),
            "predictions_made": len([l for l in self._chain
                                     if l.transition == TransitionType.PREDICTION]),
            "hypotheses_formed": len([l for l in self._chain
                                      if l.transition == TransitionType.HYPOTHESIS_FORMATION]),
            "belief_updates": len([l for l in self._chain
                                   if l.transition == TransitionType.BELIEF_UPDATE]),
        }

    def render_summary(self) -> str:
        """Render a human-readable summary."""
        summary = self.get_summary()
        anomalies = self.detect_anomalies()

        lines = []
        lines.append("=" * 60)
        lines.append("Epistemic Chain Summary")
        lines.append("=" * 60)
        lines.append(f"Total ticks:       {summary['total_ticks']}")
        lines.append(f"Total chain links: {summary['total_links']}")
        lines.append("")
        lines.append("Transitions:")
        for t, count in summary["transitions"].items():
            lines.append(f"  {t:30s}: {count}")
        lines.append("")
        lines.append("Actions:")
        for a, count in summary["actions"].items():
            pct = count / max(1, sum(summary["actions"].values())) * 100
            lines.append(f"  {a:30s}: {count:5d} ({pct:.1f}%)")
        lines.append("")
        lines.append(f"Anomalies detected: {summary['total_anomalies']}")
        for atype, count in summary["anomalies"].items():
            lines.append(f"  {atype:30s}: {count}")
        lines.append("")
        lines.append(f"Predictions made:   {summary['predictions_made']}")
        lines.append(f"Hypotheses formed:  {summary['hypotheses_formed']}")
        lines.append(f"Belief updates:     {summary['belief_updates']}")

        if anomalies:
            lines.append("")
            lines.append("Anomaly Log (last 10):")
            lines.append("-" * 60)
            for tick, atype, desc in anomalies[-10:]:
                lines.append(f"  Tick {tick:5d}: [{atype.value}] {desc}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def render_tick(self, tick: int) -> str:
        """Render all chain links for a specific tick."""
        links = self.get_chain_for_tick(tick, window=0)
        if not links:
            return f"  Tick {tick}: (no recorded events)"

        lines = []
        lines.append(f"  Tick {tick}:")
        for link in links:
            anomalies_str = ""
            if link.anomalies:
                anomalies_str = f" ANOMALIES: {[a.value for a in link.anomalies]}"
            lines.append(
                f"    {link.transition.value:30s} "
                f"conf={link.confidence:.3f} "
                f"latency={link.latency_ms:.1f}ms"
                f"{anomalies_str}"
            )
            if link.output_state:
                for k, v in link.output_state.items():
                    lines.append(f"      {k}: {v}")
        return "\n".join(lines)
