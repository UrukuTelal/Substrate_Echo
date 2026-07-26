"""Affordance Model — Learned predictive models of action effects.

Each action type has a model that predicts:
  - How it changes need satisfaction levels (delta)
  - How uncertain those predictions are (variance)
  - How confident we are overall (calibration)

The model starts ignorant (delta=0, variance=high).
After each execution, it observes the actual state change and updates.

Architecture:
    Action executed
         |
         v
    State delta observed (need satisfaction after - before)
         |
         v
    Prediction error computed (predicted - observed)
         |
         v
    Model updated (delta moves toward observed, variance shrinks)

Over time, the model becomes a calibrated predictor of action effects.
This is the core of epistemic affordance reasoning.

Usage:
    model = AffordanceModel("build_economy")

    # Predict
    predicted = model.predict()
    # {"economy": 0.023, "military": 0.0, ...}

    # After execution, observe
    actual = {"economy": 0.018, "military": 0.0, ...}
    model.update(actual)

    # Confidence
    conf = model.confidence  # 0.82
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class AffordanceModel:
    """Learned predictive model of an action's effects.

    Tracks predicted state delta, variance, and confidence.
    Updates through prediction error after each execution.
    """
    action_type: str

    # Predicted mean delta per need type
    predicted_delta: Dict[str, float] = field(default_factory=dict)

    # Predicted variance per need type (uncertainty)
    predicted_variance: Dict[str, float] = field(default_factory=dict)

    # Overall confidence in this model [0, 1]
    confidence: float = 0.1

    # Number of times this action has been observed
    n_observations: int = 0

    # History of (predicted, observed) pairs for analysis
    prediction_history: List[Tuple[Dict[str, float], Dict[str, float]]] = field(
        default_factory=list
    )

    # Running mean prediction error per need
    _mean_error: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize variance to high uncertainty
        for need in ["minerals", "gas", "supply", "military", "intel", "defense",
                      "expansion", "technology"]:
            if need not in self.predicted_variance:
                self.predicted_variance[need] = 0.25  # high initial uncertainty
            if need not in self.predicted_delta:
                self.predicted_delta[need] = 0.0
            if need not in self._mean_error:
                self._mean_error[need] = 0.0

    def predict(self) -> Dict[str, float]:
        """Return predicted state delta for each need."""
        return dict(self.predicted_delta)

    def predict_with_uncertainty(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (predicted_delta, predicted_variance)."""
        return dict(self.predicted_delta), dict(self.predicted_variance)

    def information_gain(self) -> float:
        """Total uncertainty across all needs.

        High variance = high information gain potential.
        This is what makes uncertain actions worth exploring.
        """
        return sum(self.predicted_variance.values())

    def expected_deficit_reduction(
        self,
        deficits: Dict[str, float],
        need_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Predicted weighted deficit reduction from this action.

        deficits: current need deficits (positive = unsatisfied)
        need_weights: optional weights per need type

        Returns: sum of predicted_delta * deficit * weight
        Positive = action reduces deficits (good).
        Negative = action increases deficits (bad).
        """
        total = 0.0
        for need, delta in self.predicted_delta.items():
            deficit = deficits.get(need, 0.0)
            weight = (need_weights or {}).get(need, 1.0)
            # If delta is positive (need improves) and deficit is high, this is valuable
            total += delta * deficit * weight
        return total

    def update(self, observed_delta: Dict[str, float], learning_rate: float = 0.1):
        """Update model based on observed state change.

        Moves predicted_delta toward observed_delta.
        Shrinks variance based on prediction error.
        """
        self.n_observations += 1

        # Store prediction历史
        predicted = dict(self.predicted_delta)
        self.prediction_history.append((predicted, dict(observed_delta)))
        if len(self.prediction_history) > 200:
            self.prediction_history.pop(0)

        # Update predicted delta (exponential moving average)
        for need in self.predicted_delta:
            observed = observed_delta.get(need, 0.0)
            old_pred = self.predicted_delta[need]
            error = observed - old_pred

            # Move prediction toward observation
            self.predicted_delta[need] = old_pred + learning_rate * error

            # Update running mean absolute error
            old_mae = self._mean_error.get(need, 0.0)
            self._mean_error[need] = old_mae + learning_rate * (abs(error) - old_mae)

            # Update variance (shrinks as we get more accurate)
            # Variance tracks how wrong we typically are
            old_var = self.predicted_variance.get(need, 0.25)
            # If error is smaller than current variance, variance shrinks
            # If error is larger, variance grows
            error_contribution = error * error
            self.predicted_variance[need] = old_var + learning_rate * (
                error_contribution - old_var
            )

        # Update confidence based on consistency
        self._update_confidence()

    def _update_confidence(self):
        """Update confidence based on prediction accuracy and sample size.

        Confidence increases with:
          - More observations
          - Lower prediction error
          - More consistent predictions
        """
        if self.n_observations < 3:
            self.confidence = 0.1
            return

        # Factor 1: sample size (diminishing returns)
        sample_factor = min(1.0, math.log(1 + self.n_observations) / math.log(50))

        # Factor 2: prediction accuracy (lower error = higher confidence)
        total_error = sum(self._mean_error.values())
        n_needs = max(1, len(self._mean_error))
        avg_error = total_error / n_needs
        # Map error to confidence: error=0 → 1.0, error=0.5 → 0.5, error=1.0 → 0.1
        accuracy_factor = max(0.1, 1.0 - avg_error * 1.8)

        # Factor 3: variance consistency (lower variance = more confident)
        total_var = sum(self.predicted_variance.values())
        avg_var = total_var / n_needs
        variance_factor = max(0.1, 1.0 - avg_var * 2.0)

        self.confidence = min(0.95, sample_factor * accuracy_factor * variance_factor)

    def get_recent_accuracy(self, window: int = 20) -> float:
        """Mean prediction accuracy over recent observations."""
        if len(self.prediction_history) < 2:
            return 0.0
        recent = self.prediction_history[-window:]
        errors = []
        for predicted, observed in recent:
            for need in predicted:
                p = predicted.get(need, 0.0)
                o = observed.get(need, 0.0)
                errors.append(abs(p - o))
        if not errors:
            return 0.0
        mean_error = sum(errors) / len(errors)
        return max(0.0, 1.0 - mean_error * 2.0)

    def save_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            "action_type": self.action_type,
            "predicted_delta": self.predicted_delta,
            "predicted_variance": self.predicted_variance,
            "confidence": self.confidence,
            "n_observations": self.n_observations,
            "_mean_error": self._mean_error,
            "prediction_history": [
                (p, o) for p, o in self.prediction_history[-50:]
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffordanceModel":
        """Deserialize from dict."""
        model = cls(action_type=data["action_type"])
        model.predicted_delta = data["predicted_delta"]
        model.predicted_variance = data["predicted_variance"]
        model.confidence = data["confidence"]
        model.n_observations = data["n_observations"]
        model._mean_error = data.get("_mean_error", {})
        model.prediction_history = [
            (p, o) for p, o in data.get("prediction_history", [])
        ]
        return model

    def render(self) -> str:
        lines = [f"  AffordanceModel: {self.action_type}"]
        lines.append(f"    Observations: {self.n_observations}")
        lines.append(f"    Confidence:   {self.confidence:.3f}")
        lines.append(f"    Info gain:    {self.information_gain():.4f}")
        lines.append(f"    Recent acc:   {self.get_recent_accuracy():.3f}")
        lines.append(f"    {'Need':12s} {'Delta':>8s} {'Var':>8s} {'Error':>8s}")
        lines.append(f"    {'-'*40}")
        for need in sorted(self.predicted_delta.keys()):
            d = self.predicted_delta[need]
            v = self.predicted_variance.get(need, 0.25)
            e = self._mean_error.get(need, 0.0)
            lines.append(f"    {need:12s} {d:+8.4f} {v:8.4f} {e:8.4f}")
        return "\n".join(lines)


class AffordanceModelPool:
    """Collection of AffordanceModels, one per action type.

    Provides a unified interface for prediction, scoring, and learning.
    """

    def __init__(self, action_types: List[str], need_types: List[str]):
        self.action_types = list(action_types)
        self.need_types = list(need_types)
        self.models: Dict[str, AffordanceModel] = {}
        for action in action_types:
            self.models[action] = AffordanceModel(action_type=action)
            # Initialize need types in the model
            for need in need_types:
                if need not in self.models[action].predicted_delta:
                    self.models[action].predicted_delta[need] = 0.0
                if need not in self.models[action].predicted_variance:
                    self.models[action].predicted_variance[need] = 0.25

        self.total_updates = 0
        self.epsilon = 0.25  # exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

    def get_model(self, action_type: str) -> AffordanceModel:
        return self.models.get(action_type, AffordanceModel(action_type=action_type))

    def predict(self, action_type: str) -> Dict[str, float]:
        """Get predicted delta for an action."""
        return self.get_model(action_type).predict()

    def score_action(
        self,
        action_type: str,
        deficits: Dict[str, float],
        need_weights: Optional[Dict[str, float]] = None,
        beta_exploration: float = 0.3,
    ) -> float:
        """Score an action = expected deficit reduction + exploration bonus.

        beta_exploration: weight for information gain (exploration).
        """
        model = self.get_model(action_type)
        expected_reduction = model.expected_deficit_reduction(deficits, need_weights)
        info_gain = model.information_gain()
        confidence = model.confidence

        # Core score: weighted deficit reduction
        score = expected_reduction * 100.0  # scale to [0, ~100]

        # Exploration bonus: high variance actions get a boost
        exploration_bonus = info_gain * beta_exploration * 50.0

        # Confidence dampening: low-confidence predictions are penalized slightly
        confidence_mod = 0.5 + 0.5 * confidence

        return score * confidence_mod + exploration_bonus

    def rank_actions(
        self,
        deficits: Dict[str, float],
        need_weights: Optional[Dict[str, float]] = None,
        beta_exploration: float = 0.3,
    ) -> List[Tuple[str, float, float]]:
        """Rank all actions by score. Returns [(action, score, confidence)]."""
        rankings = []
        for action in self.action_types:
            score = self.score_action(action, deficits, need_weights, beta_exploration)
            conf = self.get_model(action).confidence
            rankings.append((action, score, conf))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def update(
        self,
        action_type: str,
        observed_delta: Dict[str, float],
        learning_rate: float = 0.1,
    ):
        """Update a model based on observed outcome."""
        if action_type in self.models:
            self.models[action_type].update(observed_delta, learning_rate)
            self.total_updates += 1

            # Decay exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def should_explore(self) -> bool:
        """Should we pick a random action?"""
        import random
        return random.random() < self.epsilon

    def random_action(self) -> str:
        import random
        return random.choice(self.action_types)

    def save(self, path: str):
        """Persist all models to JSON."""
        data = {
            "action_types": self.action_types,
            "need_types": self.need_types,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "models": {
                action: model.save_dict()
                for action, model in self.models.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load all models from JSON. Returns True if successful."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.total_updates = data["total_updates"]
            self.epsilon = data["epsilon"]
            for action, model_data in data["models"].items():
                if action in self.models:
                    self.models[action] = AffordanceModel.from_dict(model_data)
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def render(self) -> str:
        lines = ["  Affordance Model Pool:"]
        lines.append(f"    Total updates: {self.total_updates}")
        lines.append(f"    Exploration:   {self.epsilon:.3f}")
        lines.append("")

        # Header
        lines.append(f"    {'Action':15s} {'Obs':>4s} {'Conf':>5s} {'InfoGain':>9s} "
                     + " ".join(f"{n[:6]:>7s}" for n in self.need_types))
        lines.append("    " + "-" * (15 + 4 + 5 + 9 + 7 * len(self.need_types)))

        for action in self.action_types:
            model = self.models[action]
            obs = model.n_observations
            conf = model.confidence
            ig = model.information_gain()
            deltas = " ".join(
                f"{model.predicted_delta.get(n, 0.0):+7.3f}"
                for n in self.need_types
            )
            lines.append(f"    {action:15s} {obs:4d} {conf:5.2f} {ig:9.3f} {deltas}")

        return "\n".join(lines)
