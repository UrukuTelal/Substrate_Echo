"""Meta-Cognition — P8

The self-correcting layer: "I know that I know."

Core idea: the agent doesn't just have confidence in its models — it
tracks whether its confidence is *calibrated*. If it consistently
over-estimates confidence, it learns to be more cautious. If
under-confident, it learns to trust itself more.

This is the layer that makes reasoning self-correcting.

Architecture:

    Confidence Sources:
    - DynamicsMemory: prediction confidence
    - Self Model: capability confidence
    - Theory of Mind: belief confidence
    - Counterfactual: outcome uncertainty
    - Curiosity: exploration uncertainty

    Aggregation:
    source_confidences → weighted_sum → meta_confidence

    Calibration:
    predicted_confidence vs actual_accuracy → calibration_error

    Self-Correction:
    if calibration_error > threshold → adjust confidence_weights

    Disagreement Detection:
    if models disagree → flag uncertainty → prefer cautious action

Usage:
    mc = MetaCognition()

    # Update with prediction outcome
    mc.update(
        predicted_confidence=0.8,
        actual_outcome_correct=True,
        source="dynamics_memory",
    )

    # Get calibrated confidence
    confidence = mc.get_confidence()

    # Check if models disagree
    disagreement = mc.check_disagreement({
        "dynamics_memory": 0.8,
        "theory_of_mind": 0.3,
        "self_model": 0.9,
    })

    # Should I trust myself?
    trust = mc.should_trust_myself()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict
from enum import Enum


class ConfidenceSource(Enum):
    """Sources of confidence in the cognitive architecture."""
    DYNAMICS_MEMORY = "dynamics_memory"
    SELF_MODEL = "self_model"
    THEORY_OF_MIND = "theory_of_mind"
    COUNTERFACTUAL = "counterfactual"
    CURIOSITY = "curiosity"
    HABIT = "habit"
    EPISODIC = "episodic"
    PLANNER = "planner"


@dataclass
class CalibrationRecord:
    """A single prediction-outcome pair for calibration."""
    source: str
    predicted_confidence: float
    actual_correct: bool
    tick: int = 0
    context_hash: str = ""


@dataclass
class MetaState:
    """Aggregate meta-cognitive state."""
    # Overall calibrated confidence
    calibrated_confidence: float = 0.5

    # Per-source trust weights
    source_trust: dict[str, float] = field(default_factory=dict)

    # Calibration metrics
    overconfidence: float = 0.0  # how much we over-estimate
    underconfidence: float = 0.0  # how much we under-estimate
    calibration_error: float = 0.0  # brier-like score

    # Disagreement
    model_disagreement: float = 0.0  # 0 = all agree, 1 = max conflict
    has_disagreement: bool = False

    # Self-correction
    correction_active: bool = False
    correction_magnitude: float = 0.0

    @property
    def should_be_cautious(self) -> bool:
        """True if meta-cognition recommends caution."""
        return (self.model_disagreement > 0.4 or
                self.calibration_error > 0.3 or
                self.calibrated_confidence < 0.3)

    @property
    def summary(self) -> str:
        if self.should_be_cautious:
            return "cautious"
        elif self.calibrated_confidence > 0.7:
            return "confident"
        else:
            return "uncertain"


@dataclass
class MetaCognitionConfig:
    """Configuration for meta-cognition."""
    history_size: int = 500
    calibration_window: int = 50
    overconfidence_threshold: float = 0.15
    learning_rate: float = 0.05
    default_trust: float = 0.5
    disagreement_threshold: float = 0.4
    min_records_for_calibration: int = 10


class MetaCognition:
    """Self-correcting confidence layer.

    Tracks whether the agent's confidence is well-calibrated,
    detects disagreements between internal models, and adjusts
    trust weights accordingly.

    Usage:
        mc = MetaCognition()

        # After each prediction
        mc.update(predicted_confidence=0.8, actual_outcome_correct=True,
                  source="dynamics_memory")

        # Get calibrated state
        state = mc.get_meta_state()

        # Should I trust this prediction?
        if state.should_be_cautious:
            prefer safer action
    """

    def __init__(self, config: Optional[MetaCognitionConfig] = None):
        self.config = config or MetaCognitionConfig()
        self._history: list[CalibrationRecord] = []
        self._source_history: dict[str, list[CalibrationRecord]] = defaultdict(list)

        # Per-source trust weights (start equal)
        self._source_trust: dict[str, float] = {
            src.value: self.config.default_trust
            for src in ConfidenceSource
        }

        # Calibration tracking
        self._recent_predictions: list[float] = []
        self._recent_outcomes: list[bool] = []

    def update(self, predicted_confidence: float,
               actual_outcome_correct: bool,
               source: str,
               tick: int = 0,
               context_hash: str = "") -> None:
        """Record a prediction and its outcome for calibration.

        Args:
            predicted_confidence: how confident the model was (0-1)
            actual_outcome_correct: whether prediction was correct
            source: which model made the prediction
            tick: current tick
            context_hash: optional context identifier
        """
        record = CalibrationRecord(
            source=source,
            predicted_confidence=np.clip(predicted_confidence, 0, 1),
            actual_correct=actual_outcome_correct,
            tick=tick,
            context_hash=context_hash,
        )

        self._history.append(record)
        self._source_history[source].append(record)

        # Maintain history size
        if len(self._history) > self.config.history_size:
            self._history.pop(0)
        if len(self._source_history[source]) > self.config.history_size:
            self._source_history[source].pop(0)

        # Update calibration tracking
        self._recent_predictions.append(predicted_confidence)
        self._recent_outcomes.append(actual_outcome_correct)
        if len(self._recent_predictions) > self.config.calibration_window:
            self._recent_predictions.pop(0)
            self._recent_outcomes.pop(0)

        # Recalibrate
        self._recalibrate(source)

    def update_multi(self, confidences: dict[str, float],
                     actual_outcome_correct: bool,
                     tick: int = 0) -> None:
        """Update with multiple source confidences at once.

        Args:
            confidences: dict of source_name → predicted_confidence
            actual_outcome_correct: whether the combined prediction was correct
            tick: current tick
        """
        for source, confidence in confidences.items():
            self.update(confidence, actual_outcome_correct, source, tick)

    def get_meta_state(self) -> MetaState:
        """Get current meta-cognitive state."""
        # Calibrated confidence
        calibrated = self._compute_calibrated_confidence()

        # Disagreement
        disagreement = self._compute_disagreement()

        # Calibration error
        cal_error = self._compute_calibration_error()

        # Over/under confidence
        over, under = self._compute_confidence_bias()

        # Correction
        correction_active = over > self.config.overconfidence_threshold
        correction_mag = over if correction_active else 0.0

        return MetaState(
            calibrated_confidence=calibrated,
            source_trust=dict(self._source_trust),
            overconfidence=over,
            underconfidence=under,
            calibration_error=cal_error,
            model_disagreement=disagreement,
            has_disagreement=disagreement > self.config.disagreement_threshold,
            correction_active=correction_active,
            correction_magnitude=correction_mag,
        )

    def get_confidence(self, source: Optional[str] = None) -> float:
        """Get calibrated confidence, optionally for a specific source."""
        if source:
            trust = self._source_trust.get(source, self.config.default_trust)
            recent = self._source_history.get(source, [])
            if recent:
                recent_conf = np.mean([r.predicted_confidence for r in recent[-20:]])
                return float(recent_conf * trust)
            return trust

        state = self.get_meta_state()
        return state.calibrated_confidence

    def should_trust_myself(self) -> bool:
        """Should the agent trust its own predictions?

        Returns True if confidence is calibrated and models agree.
        """
        state = self.get_meta_state()
        return (not state.should_be_cautious and
                state.calibrated_confidence > 0.4)

    def check_disagreement(self, confidences: dict[str, float]) -> float:
        """Check how much models disagree about confidence.

        Returns disagreement score 0-1.
        """
        if len(confidences) < 2:
            return 0.0

        values = list(confidences.values())
        return float(np.std(values) * 2)  # scale to 0-1 roughly

    def get_source_reliability(self, source: str) -> float:
        """Get reliability score for a specific source (0-1)."""
        records = self._source_history.get(source, [])
        if len(records) < 5:
            return self.config.default_trust

        recent = records[-50:]
        correct = sum(1 for r in recent if r.actual_correct)
        return correct / len(recent)

    def summary(self) -> dict:
        """Summary of meta-cognitive state."""
        state = self.get_meta_state()
        return {
            "n_predictions": len(self._history),
            "calibrated_confidence": state.calibrated_confidence,
            "calibration_error": state.calibration_error,
            "overconfidence": state.overconfidence,
            "underconfidence": state.underconfidence,
            "model_disagreement": state.model_disagreement,
            "should_be_cautious": state.should_be_cautious,
            "should_trust": self.should_trust_myself(),
            "meta_state": state.summary,
        }

    # ── Private methods ──────────────────────────────────────

    def _recalibrate(self, source: str) -> None:
        """Recalibrate trust weights based on recent accuracy."""
        if len(self._recent_predictions) < self.config.min_records_for_calibration:
            return

        # Compute per-source accuracy
        for src in self._source_trust:
            records = self._source_history.get(src, [])
            if len(records) < 10:
                continue

            recent = records[-50:]
            correct = sum(1 for r in recent if r.actual_correct)
            accuracy = correct / len(recent)

            # Mean predicted confidence for this source
            mean_conf = np.mean([r.predicted_confidence for r in recent])

            # Calibration: if accuracy > mean_conf, we're underconfident
            # If accuracy < mean_conf, we're overconfident
            bias = accuracy - mean_conf

            # Update trust: increase if well-calibrated, decrease if not
            current_trust = self._source_trust[src]
            if abs(bias) < 0.1:
                # Well calibrated — maintain or slightly increase trust
                new_trust = current_trust + self.config.learning_rate * 0.5
            elif bias < 0:
                # Overconfident — decrease trust
                new_trust = current_trust + self.config.learning_rate * bias
            else:
                # Underconfident — increase trust
                new_trust = current_trust + self.config.learning_rate * bias * 0.5

            self._source_trust[src] = float(np.clip(new_trust, 0.1, 1.0))

    def _compute_calibrated_confidence(self) -> float:
        """Compute overall calibrated confidence."""
        if not self._recent_predictions:
            return 0.5

        # Weighted average of recent predictions by source trust
        weighted_sum = 0.0
        weight_total = 0.0

        for record in list(self._history)[-50:]:
            trust = self._source_trust.get(record.source, 0.5)
            weighted_sum += record.predicted_confidence * trust
            weight_total += trust

        if weight_total == 0:
            return 0.5

        return float(np.clip(weighted_sum / weight_total, 0, 1))

    def _compute_disagreement(self) -> float:
        """Compute disagreement between recent source confidences."""
        if not self._recent_predictions or len(self._recent_predictions) < 2:
            return 0.0

        # Group by source
        source_confidences = defaultdict(list)
        for record in list(self._history)[-50:]:
            source_confidences[record.source].append(record.predicted_confidence)

        if len(source_confidences) < 2:
            return 0.0

        # Mean confidence per source
        means = [np.mean(confs) for confs in source_confidences.values()
                 if confs]

        if len(means) < 2:
            return 0.0

        return float(np.std(means) * 2)

    def _compute_calibration_error(self) -> float:
        """Compute calibration error (Brier-like score)."""
        if len(self._recent_predictions) < self.config.min_records_for_calibration:
            return 0.0

        predictions = np.array(self._recent_predictions)
        outcomes = np.array(self._recent_outcomes, dtype=float)

        # Brier score: mean((predicted - actual)^2)
        brier = float(np.mean((predictions - outcomes) ** 2))
        return brier

    def _compute_confidence_bias(self) -> tuple[float, float]:
        """Compute overconfidence and underconfidence."""
        if len(self._recent_predictions) < self.config.min_records_for_calibration:
            return 0.0, 0.0

        predictions = np.array(self._recent_predictions)
        outcomes = np.array(self._recent_outcomes, dtype=float)

        mean_pred = float(np.mean(predictions))
        mean_outcome = float(np.mean(outcomes))

        bias = mean_pred - mean_outcome

        if bias > 0:
            return bias, 0.0  # overconfident
        else:
            return 0.0, -bias  # underconfident
