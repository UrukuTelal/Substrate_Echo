"""Epistemic Action Bridge — Converts beliefs into action weights.

The missing link between "I believe X" and "I do Y."

Architecture:
    Hypothesis (confidence, evidence)
         |
         v
    Prediction (expected outcome)
         |
         v
    Utility Estimate (expected reward)
         |
         v
    Action Weight = reward × confidence × prediction_accuracy × affordance_strength

The key principle:
    A low-confidence belief cannot dominate behavior.
    A high-confidence belief that contradicts evidence cannot drive action.

Usage:
    bridge = EpistemicActionBridge()
    
    # Score a candidate action
    score = bridge.score_action(
        candidate=affordance_candidate,
        entity_model=entity_model,
        prediction_accuracy=0.73,
    )
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class ActionScore:
    """Complete score breakdown for an action."""
    action_type: str
    description: str = ""

    # Component scores
    base_reward: float = 0.0       # from affordance candidate
    confidence_weight: float = 0.5 # from entity model / belief confidence
    prediction_weight: float = 0.5 # from prediction accuracy
    affordance_weight: float = 0.5 # from affordance strength
    uncertainty_penalty: float = 0.0

    # Final score
    final_score: float = 0.0

    # Metadata
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_type,
            "base_reward": round(self.base_reward, 3),
            "confidence": round(self.confidence_weight, 3),
            "prediction": round(self.prediction_weight, 3),
            "affordance": round(self.affordance_weight, 3),
            "uncertainty_penalty": round(self.uncertainty_penalty, 3),
            "final_score": round(self.final_score, 3),
            "reasoning": self.reasoning,
        }


class EpistemicActionBridge:
    """Converts beliefs into action weights.
    
    Formula:
        ActionScore = (base_reward × confidence × prediction_accuracy × affordance)
                    - uncertainty_penalty
    
    Where:
        - base_reward comes from the affordance candidate
        - confidence comes from the entity model / hypothesis confidence
        - prediction_accuracy comes from historical prediction accuracy
        - affordance_weight comes from the affordance's success probability
        - uncertainty_penalty increases when beliefs are uncertain
    
    Usage:
        bridge = EpistemicActionBridge()
        
        # Update prediction accuracy
        bridge.update_accuracy(True)   # prediction was correct
        bridge.update_accuracy(False)  # prediction was wrong
        
        # Score actions
        scores = bridge.score_candidates(candidates, entity_model)
        best = scores[0]  # highest epistemic score
    """

    def __init__(self, confidence_weight: float = 0.4,
                 prediction_weight: float = 0.3,
                 affordance_weight: float = 0.3):
        # Tunable weights
        self.confidence_weight = confidence_weight
        self.prediction_weight = prediction_weight
        self.affordance_weight = affordance_weight

        # Prediction accuracy tracking
        self._predictions_made: int = 0
        self._predictions_correct: int = 0
        self._accuracy_history: List[bool] = []

        # Score history
        self._score_history: List[ActionScore] = []

    @property
    def prediction_accuracy(self) -> float:
        """Historical prediction accuracy."""
        if self._predictions_made == 0:
            return 0.5  # prior: neutral
        return self._predictions_correct / self._predictions_made

    def update_accuracy(self, correct: bool):
        """Record a prediction outcome."""
        self._predictions_made += 1
        if correct:
            self._predictions_correct += 1
        self._accuracy_history.append(correct)
        # Keep last 100
        if len(self._accuracy_history) > 100:
            self._accuracy_history.pop(0)

    def get_recent_accuracy(self, window: int = 20) -> float:
        """Get accuracy over recent window."""
        if not self._accuracy_history:
            return 0.5
        recent = self._accuracy_history[-window:]
        return sum(recent) / len(recent)

    def score_action(self, candidate: Any,
                     entity_confidence: float = 0.5,
                     prediction_accuracy: Optional[float] = None) -> ActionScore:
        """Score a single action candidate."""
        if prediction_accuracy is None:
            prediction_accuracy = self.prediction_accuracy

        # Base reward from affordance
        base_reward = candidate.expected_reward if hasattr(candidate, 'expected_reward') else 0.0
        affordance_strength = candidate.success_probability if hasattr(candidate, 'success_probability') else 0.5
        risk = candidate.risk if hasattr(candidate, 'risk') else 0.0

        # Uncertainty penalty: high when confidence is low and action is expensive
        cost_level = 0
        if hasattr(candidate, 'cost_level'):
            cost_level = candidate.cost_level.value if hasattr(candidate.cost_level, 'value') else 0

        uncertainty_penalty = (1.0 - entity_confidence) * cost_level * 0.1

        # Final score
        reward_component = base_reward * self.confidence_weight * entity_confidence
        prediction_component = base_reward * self.prediction_weight * prediction_accuracy
        affordance_component = base_reward * self.affordance_weight * affordance_strength

        final_score = (
            reward_component
            + prediction_component
            + affordance_component
            - uncertainty_penalty
            - risk * base_reward * 0.2
        )

        # Build reasoning
        reasoning_parts = []
        if entity_confidence < 0.3:
            reasoning_parts.append(f"LOW CONFIDENCE ({entity_confidence:.2f})")
        if prediction_accuracy < 0.4:
            reasoning_parts.append(f"POOR PREDICTION ({prediction_accuracy:.2f})")
        if risk > 0.5:
            reasoning_parts.append(f"HIGH RISK ({risk:.2f})")
        if uncertainty_penalty > 0.2:
            reasoning_parts.append(f"UNCERTAINTY PENALTY ({uncertainty_penalty:.2f})")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "standard"

        action_type = candidate.action_type.value if hasattr(candidate, 'action_type') else "unknown"
        description = candidate.description if hasattr(candidate, 'description') else ""

        score = ActionScore(
            action_type=action_type,
            description=description,
            base_reward=base_reward,
            confidence_weight=entity_confidence,
            prediction_weight=prediction_accuracy,
            affordance_weight=affordance_strength,
            uncertainty_penalty=uncertainty_penalty,
            final_score=final_score,
            reasoning=reasoning,
        )

        self._score_history.append(score)
        return score

    def score_candidates(self, candidates: List[Any],
                         entity_confidence: float = 0.5,
                         prediction_accuracy: Optional[float] = None) -> List[ActionScore]:
        """Score all candidate actions and return ranked."""
        scores = []
        for c in candidates:
            score = self.score_action(c, entity_confidence, prediction_accuracy)
            scores.append(score)

        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores

    def get_confidence_modifier(self, base_confidence: float,
                                recent_failures: int = 0) -> float:
        """Modify confidence based on recent prediction failures.
        
        After consecutive failures, dampen confidence more aggressively.
        """
        accuracy = self.prediction_accuracy

        # Start with base confidence
        modified = base_confidence

        # Apply accuracy modifier
        modified *= (0.5 + 0.5 * accuracy)

        # Apply failure dampening
        if recent_failures > 0:
            dampening = max(0.1, 1.0 - recent_failures * 0.15)
            modified *= dampening

        return max(0.05, min(0.95, modified))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "predictions_made": self._predictions_made,
            "predictions_correct": self._predictions_correct,
            "accuracy": round(self.prediction_accuracy, 4),
            "recent_accuracy": round(self.get_recent_accuracy(), 4),
            "scores_generated": len(self._score_history),
            "weights": {
                "confidence": self.confidence_weight,
                "prediction": self.prediction_weight,
                "affordance": self.affordance_weight,
            },
        }

    def render(self, scores: List[ActionScore]) -> str:
        """Render scored actions."""
        lines = []
        lines.append("Epistemic Action Scores:")
        lines.append("-" * 70)
        lines.append(f"  Prediction accuracy: {self.prediction_accuracy:.1%}")
        lines.append(f"  Recent accuracy:     {self.get_recent_accuracy():.1%}")
        lines.append("")
        for i, s in enumerate(scores):
            marker = ">>>" if i == 0 else "   "
            lines.append(
                f"  {marker} {s.action_type:12s} "
                f"final={s.final_score:7.2f} "
                f"conf={s.confidence_weight:.2f} "
                f"pred={s.prediction_weight:.2f} "
                f"aff={s.affordance_weight:.2f} "
                f"penalty={s.uncertainty_penalty:.2f}"
            )
            if s.reasoning != "standard":
                lines.append(f"       REASON: {s.reasoning}")
        return "\n".join(lines)
