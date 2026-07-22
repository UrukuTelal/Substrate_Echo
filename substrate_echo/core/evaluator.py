"""Evaluator — scores future states via pillar-based utility.

Different agents weight pillars differently, creating personality
without changing the physics. The evaluator answers:
"Is this predicted future good or bad?"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .world_model import WorldModel


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]


@dataclass
class UtilityWeights:
    """Per-agent utility weights over 16 pillars + meta-signals.
    
    Different weight vectors create different agent personalities.
    Positive weights = utility gained when pillar is high.
    Negative weights = utility gained when pillar is low (e.g., Harm).
    """
    # Pillar weights (indexed by pillar position)
    pillar_weights: list[float] = field(default_factory=lambda: [
        0.3,   # [0]  Awareness — perceptual openness
        0.4,   # [1]  Willpower — goal-directed energy
        0.2,   # [2]  Force — executive power
        0.1,   # [3]  Influence — outward impact
        0.2,   # [4]  Resistance — boundary integrity
        1.0,   # [5]  Integrity — internal coherence
        0.6,   # [6]  Cohesion — binding force
        0.3,   # [7]  Relation — social connection
        0.2,   # [8]  Presence — grounding
        0.1,   # [9]  Warmth — affiliative tone
        0.5,   # [10] Memory — retention capacity
        0.1,   # [11] Attraction — gravitational pull
        -2.0,  # [12] Harm — damage (negative = avoid)
        -1.0,  # [13] Distortion — deviation (negative = avoid)
        -0.3,  # [14] Flux — instability (mildly avoid)
        0.2,   # [15] Depth — processing complexity
    ])
    
    # Meta-utility signals
    stability_weight: float = 0.5
    novelty_weight: float = 0.3
    information_weight: float = 0.2
    coherence_weight: float = 0.3
    
    # Task-specific
    task_reward: float = 0.0
    
    def pillar_utility(self, state: np.ndarray) -> float:
        """Scalar utility from pillar values."""
        weights = np.array(self.pillar_weights[:len(state)])
        return float(np.dot(weights, state))
    
    @staticmethod
    def cautious() -> 'UtilityWeights':
        return UtilityWeights(
            pillar_weights=[0.1, 0.2, 0.1, 0.0, 0.5, 2.0, 0.8, 0.1,
                           0.1, 0.0, 0.3, 0.0, -3.0, -2.0, -1.0, 0.1],
            stability_weight=1.0, novelty_weight=0.0, information_weight=0.0,
        )
    
    @staticmethod
    def curious() -> 'UtilityWeights':
        return UtilityWeights(
            pillar_weights=[0.8, 0.3, 0.1, 0.1, 0.1, 0.5, 0.2, 0.2,
                           0.3, 0.1, 0.8, 0.2, -1.0, 0.5, 0.3, 0.8],
            stability_weight=0.0, novelty_weight=1.0, information_weight=1.0,
        )
    
    @staticmethod
    def social() -> 'UtilityWeights':
        return UtilityWeights(
            pillar_weights=[0.2, 0.2, 0.1, 0.3, 0.2, 0.5, 1.5, 2.0,
                           0.2, 1.0, 0.3, 0.2, -2.0, -0.5, -0.2, 0.1],
            stability_weight=0.3, novelty_weight=0.2, information_weight=0.1,
        )
    
    @staticmethod
    def achiever() -> 'UtilityWeights':
        return UtilityWeights(
            pillar_weights=[0.3, 2.0, 1.5, 0.5, 0.3, 1.0, 0.3, 0.1,
                           0.2, 0.0, 1.0, 0.3, -2.0, -1.0, -0.5, 0.5],
            stability_weight=0.5, novelty_weight=0.1, information_weight=0.2,
            task_reward=3.0,
        )


@dataclass
class EvalResult:
    """Evaluation result for a state or trajectory."""
    utility: float
    pillar_contributions: np.ndarray  # per-pillar utility contribution
    stability_score: float
    novelty_score: float
    information_score: float
    coherence_score: float
    breakdown: dict
    
    def __repr__(self):
        return (f"EvalResult(U={self.utility:.3f}, "
                f"stability={self.stability_score:.3f}, "
                f"novelty={self.novelty_score:.3f})")


class Evaluator:
    """Evaluates future states using pillar-based utility.
    
    The evaluator is the bridge between prediction and decision.
    It takes a predicted future and answers: "Is this good?"
    
    Different agents use different UtilityWeights, creating
    personality through value alignment, not behavioral rules.
    """
    
    def __init__(self, weights: Optional[UtilityWeights] = None,
                 world_model: Optional[WorldModel] = None):
        self.weights = weights or UtilityWeights()
        self.world_model = world_model
        self._visited_states: list[np.ndarray] = []
    
    def evaluate(self, state: np.ndarray, lightweight: bool = False) -> EvalResult:
        """Evaluate a single state.
        
        lightweight=True skips expensive meta-scores (stability Jacobian, etc.)
        and only computes pillar utility. Use for intermediate trajectory points.
        """
        state = np.asarray(state, dtype=np.float64)
        
        # Pillar utility
        pillar_util = self.weights.pillar_utility(state)
        pillar_contributions = np.array(self.weights.pillar_weights[:len(state)]) * state
        
        if lightweight:
            return EvalResult(
                utility=pillar_util + self.weights.task_reward,
                pillar_contributions=pillar_contributions,
                stability_score=0.0,
                novelty_score=0.0,
                information_score=0.0,
                coherence_score=0.0,
                breakdown={"pillar_utility": pillar_util, "task_reward": self.weights.task_reward},
            )
        
        # Meta-utility signals
        stability = self._compute_stability_score(state)
        novelty = self._compute_novelty_score(state)
        information = self._compute_information_score(state)
        coherence = self._compute_coherence_score(state)
        
        # Combined utility
        utility = (pillar_util
                   + self.weights.stability_weight * stability
                   + self.weights.novelty_weight * novelty
                   + self.weights.information_weight * information
                   + self.weights.coherence_weight * coherence
                   + self.weights.task_reward)
        
        breakdown = {
            "pillar_utility": pillar_util,
            "stability": stability,
            "novelty": novelty,
            "information": information,
            "coherence": coherence,
            "task_reward": self.weights.task_reward,
        }
        
        return EvalResult(
            utility=utility,
            pillar_contributions=pillar_contributions,
            stability_score=stability,
            novelty_score=novelty,
            information_score=information,
            coherence_score=coherence,
            breakdown=breakdown,
        )
    
    def evaluate_trajectory(self, trajectory: list[np.ndarray]) -> EvalResult:
        """Evaluate a full trajectory (sum of per-step utilities with discount).
        
        Intermediate points use lightweight evaluation (pillar utility only).
        Final point gets full evaluation (stability, novelty, etc.).
        """
        if not trajectory:
            return EvalResult(0.0, np.zeros(16), 0.0, 0.0, 0.0, 0.0, {})
        
        discount = 0.95
        total_utility = 0.0
        total_pillars = np.zeros(len(trajectory[0]))
        final_eval = None
        
        for t, state in enumerate(trajectory):
            # Full evaluation only for final state; lightweight for intermediates
            is_final = (t == len(trajectory) - 1)
            step_eval = self.evaluate(state, lightweight=not is_final)
            weight = discount ** t
            total_utility += weight * step_eval.utility
            total_pillars += weight * step_eval.pillar_contributions
            if is_final:
                final_eval = step_eval
        
        if final_eval is None:
            final_eval = self.evaluate(trajectory[-1])
        
        return EvalResult(
            utility=total_utility,
            pillar_contributions=total_pillars,
            stability_score=final_eval.stability_score,
            novelty_score=final_eval.novelty_score,
            information_score=final_eval.information_score,
            coherence_score=final_eval.coherence_score,
            breakdown={
                "trajectory_length": len(trajectory),
                "discounted_sum": total_utility,
                "final_state_eval": final_eval.utility,
            },
        )
    
    def record_visit(self, state: np.ndarray):
        """Record that we've visited this state (for novelty computation)."""
        self._visited_states.append(state.copy())
        if len(self._visited_states) > 500:
            self._visited_states = self._visited_states[-500:]
    
    def reset_visits(self):
        self._visited_states.clear()
    
    # ── Meta-Utility Components ─────────────────────────────────
    
    def _compute_stability_score(self, state: np.ndarray) -> float:
        """Higher score for more stable states."""
        if self.world_model is None:
            return 0.0
        
        stability = self.world_model.get_stability(state)
        classification = stability['classification']
        
        if classification == 'attractor':
            # More negative eigenvalue = more stable
            max_real = stability.get('max_eigenvalue_real', 0.0)
            return float(min(1.0, max(0.0, -max_real)))
        elif classification == 'saddle':
            return -0.3
        elif classification == 'repellor':
            return -0.5
        else:
            return 0.0
    
    def _compute_novelty_score(self, state: np.ndarray) -> float:
        """Higher score for states unlike previously visited ones."""
        if not self._visited_states:
            return 1.0  # everything is novel at first
        
        states_arr = np.array(self._visited_states)
        dists = np.linalg.norm(states_arr - state, axis=1)
        min_dist = float(np.min(dists))
        
        # Novelty decays with proximity to visited states
        return float(np.clip(1.0 - np.exp(-min_dist / 0.2), 0.0, 1.0))
    
    def _compute_information_score(self, state: np.ndarray) -> float:
        """Higher score for states with high information gain potential.

        Uses DynamicsMemory.novelty() when available (distance to training
        data — the correct info gain signal). Falls back to prediction
        confidence when DynamicsMemory is unavailable.
        """
        if self.world_model is None:
            return 0.0
        
        # Prefer DynamicsMemory novelty (correct info gain signal)
        dm = getattr(self.world_model, 'memory', None)
        if dm is not None and hasattr(dm, 'novelty') and dm._fitted and dm._states:
            novelty = dm.novelty(state)
            # Normalize by average training data spacing
            if len(dm._states) > 1:
                states_arr = np.array(dm._states)
                sample_idx = np.random.choice(len(dm._states),
                                               min(50, len(dm._states)),
                                               replace=False)
                dists = []
                for i in sample_idx:
                    d = np.linalg.norm(states_arr - states_arr[i], axis=1)
                    if len(d) > 1:
                        dists.append(np.sort(d)[1])
                avg_spacing = np.mean(dists) if dists else 1.0
                return float(np.clip(novelty / (avg_spacing * 3), 0.0, 1.0))
            return float(np.clip(novelty, 0.0, 1.0))
        
        # Fallback: prediction confidence
        confidence = self.world_model.prediction_confidence(state)
        return float(1.0 - confidence)
    
    def _compute_coherence_score(self, state: np.ndarray) -> float:
        """Higher score for states with low variance across pillars."""
        std = float(np.std(state))
        # Coherence peaks when std is moderate (not too flat, not too wild)
        # Optimal around 0.15
        return float(np.exp(-((std - 0.15) ** 2) / (2 * 0.1 ** 2)))
