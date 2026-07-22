"""Self-Model — P7.4

Model of own capabilities, limitations, confidence, and expected future state.

Core capabilities:
1. Capability assessment: what can I do?
2. Limitation awareness: what can't I do?
3. Confidence estimation: how sure am I about my predictions?
4. Future projection: what will I be able to do next?
5. Meta-cognition: monitoring my own cognitive processes

This enables deciding WHETHER to attempt something, not just HOW.

Architecture:
- SelfModel maintains a profile of the agent's own capabilities
- Tracks success/failure history for different action types
- Estimates confidence from DynamicsMemory coverage
- Projects future capabilities from current trajectory
- Monitors cognitive load and decision quality

Usage:
    sm = SelfModel(dim=16)
    sm.update(pillars=current_psv, action_taken="approach", success=True)
    can_do = sm.can_do("navigate_to", target=np.array([5,5,0]))
    confidence = sm.confidence_in("predict_dynamics")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict


@dataclass
class Capability:
    """A specific capability the agent has assessed."""
    name: str
    proficiency: float = 0.0     # 0-1, how good at this
    confidence: float = 0.0      # 0-1, how sure about proficiency estimate
    attempts: int = 0
    successes: int = 0
    last_attempt_tick: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)


@dataclass
class CognitiveLoad:
    """Current cognitive load assessment."""
    planning_complexity: float = 0.0    # how complex current planning is
    memory_utilization: float = 0.0     # how much memory is being used
    prediction_uncertainty: float = 0.0 # how uncertain predictions are
    social_complexity: float = 0.0      # how many social dynamics to track

    @property
    def total_load(self) -> float:
        return (self.planning_complexity + self.memory_utilization +
                self.prediction_uncertainty + self.social_complexity) / 4.0


@dataclass
class SelfModelConfig:
    """Configuration for self-model."""
    proficiency_smoothing: float = 0.3   # EMA for proficiency updates
    confidence_decay: float = 0.005      # per-tick confidence decay
    min_attempts_for_confidence: int = 5
    cognitive_load_window: int = 20      # ticks to average over
    capability_names: list[str] = field(default_factory=lambda: [
        "navigate", "predict_dynamics", "social_interact",
        "explore", "avoid_harm", "learn", "plan",
    ])


class SelfModel:
    """Models own capabilities, limitations, and cognitive state.

    The self-model answers questions like:
    - "Can I navigate to that location?" (capability assessment)
    - "How confident am I in my predictions?" (confidence estimation)
    - "What will I be able to do next?" (future projection)
    - "Am I overloaded?" (cognitive load monitoring)

    This is the foundation for autonomous decision-making:
    an agent that knows its limitations can make better choices
    than one that overestimates itself.

    Usage:
        sm = SelfModel()
        sm.update(pillars=current_psv, action_taken="approach", success=True)
        sm.update_prediction_confidence(0.8)
        can_do = sm.can_do("navigate", target=np.array([5,5,0]))
    """

    def __init__(self, config: Optional[SelfModelConfig] = None, dim: int = 16):
        self.config = config or SelfModelConfig()
        self.dim = dim

        # Capabilities
        self._capabilities: dict[str, Capability] = {}
        for name in self.config.capability_names:
            self._capabilities[name] = Capability(name=name)

        # Current state
        self._current_pillars: Optional[np.ndarray] = None
        self._current_confidence: float = 0.5
        self._cognitive_load = CognitiveLoad()

        # History
        self._action_history: list[dict] = []
        self._confidence_history: list[float] = []
        self._load_history: list[float] = []

        # Self-assessment
        self._strengths: list[str] = []
        self._weaknesses: list[str] = []

    def update(self, pillars: Optional[np.ndarray] = None,
               action_taken: Optional[str] = None,
               success: Optional[bool] = None,
               tick: int = 0) -> None:
        """Update self-model with new observation.

        Args:
            pillars: current PSV state
            action_taken: name of action just taken
            success: whether the action succeeded
            tick: current tick number
        """
        if pillars is not None:
            self._current_pillars = np.asarray(pillars, dtype=np.float64)

        # Update capability from action outcome
        if action_taken is not None and success is not None:
            # Map action to capability
            action_to_cap = {
                "approach": "navigate",
                "retreat": "navigate",
                "observe": "explore",
                "investigate": "explore",
                "communicate": "social_interact",
                "defend": "avoid_harm",
                "grasp": "plan",
            }
            cap_name = action_to_cap.get(action_taken, action_taken)
            if cap_name not in self._capabilities:
                self._capabilities[cap_name] = Capability(name=cap_name)

            cap = self._capabilities[cap_name]
            cap.attempts += 1
            cap.last_attempt_tick = tick
            if success:
                cap.successes += 1

            # Update proficiency with EMA
            new_proficiency = 1.0 if success else 0.0
            cap.proficiency = cap.proficiency * (1 - self.config.proficiency_smoothing) + \
                               new_proficiency * self.config.proficiency_smoothing

            # Update confidence based on attempt count
            if cap.attempts >= self.config.min_attempts_for_confidence:
                cap.confidence = min(1.0, cap.attempts / 50.0)

            self._action_history.append({
                "action": action_taken,
                "success": success,
                "tick": tick,
            })

        # Decay confidence over time
        for cap in self._capabilities.values():
            cap.confidence *= (1 - self.config.confidence_decay)

        # Update self-assessment
        self._update_self_assessment()

        # Record history
        self._confidence_history.append(self._current_confidence)
        if len(self._confidence_history) > 100:
            self._confidence_history.pop(0)
        self._load_history.append(self._cognitive_load.total_load)
        if len(self._load_history) > 100:
            self._load_history.pop(0)

    def can_do(self, capability_name: str,
               target: Optional[np.ndarray] = None) -> bool:
        """Check if the agent can do something.

        Returns True if proficiency > 0.3 and confidence > 0.1.
        """
        cap = self._capabilities.get(capability_name)
        if cap is None:
            return False
        return cap.proficiency > 0.3 and cap.confidence > 0.1

    def confidence_in(self, capability_name: str) -> float:
        """Get confidence in a specific capability."""
        cap = self._capabilities.get(capability_name)
        if cap is None:
            return 0.0
        return cap.confidence

    def proficiency_in(self, capability_name: str) -> float:
        """Get proficiency in a specific capability."""
        cap = self._capabilities.get(capability_name)
        if cap is None:
            return 0.0
        return cap.proficiency

    def update_prediction_confidence(self, confidence: float) -> None:
        """Update confidence in world model predictions."""
        self._current_confidence = np.clip(confidence, 0.0, 1.0)
        self._cognitive_load.prediction_uncertainty = 1.0 - confidence

    def update_cognitive_load(self, **kwargs) -> None:
        """Update cognitive load components."""
        for key, value in kwargs.items():
            if hasattr(self._cognitive_load, key):
                setattr(self._cognitive_load, key, np.clip(value, 0.0, 1.0))

    def project_future(self, steps: int = 10) -> dict:
        """Project what the agent will be able to do in the future.

        Based on current trajectory and capability trends.
        """
        # Capability trends
        improving = []
        declining = []
        stable = []

        recent_actions = self._action_history[-20:]
        if recent_actions:
            success_rate_recent = sum(1 for a in recent_actions if a["success"]) / len(recent_actions)
            success_rate_all = sum(1 for a in self._action_history if a["success"]) / max(1, len(self._action_history))

            if success_rate_recent > success_rate_all + 0.1:
                improving.append("overall")
            elif success_rate_recent < success_rate_all - 0.1:
                declining.append("overall")
            else:
                stable.append("overall")

        # Projection
        projected_capabilities = {}
        for name, cap in self._capabilities.items():
            trend = 0.0
            if name in improving:
                trend = 0.05 * steps
            elif name in declining:
                trend = -0.05 * steps
            projected = np.clip(cap.proficiency + trend, 0.0, 1.0)
            projected_capabilities[name] = projected

        return {
            "current_confidence": self._current_confidence,
            "cognitive_load": self._cognitive_load.total_load,
            "projected_capabilities": projected_capabilities,
            "improving": improving,
            "declining": declining,
            "stable": stable,
            "recommended_focus": self._identify_weakness(),
        }

    def _update_self_assessment(self) -> None:
        """Update strengths and weaknesses from capability assessment."""
        self._strengths = []
        self._weaknesses = []

        for name, cap in self._capabilities.items():
            if cap.attempts < 3:
                continue
            if cap.proficiency > 0.6 and cap.confidence > 0.1:
                self._strengths.append(name)
            elif cap.proficiency < 0.4 and cap.confidence > 0.1:
                self._weaknesses.append(name)

    def _identify_weakness(self) -> Optional[str]:
        """Identify the most important weakness to address."""
        if not self._weaknesses:
            return None
        # Return weakness with lowest proficiency
        weakest = min(self._weaknesses,
                       key=lambda n: self._capabilities[n].proficiency)
        return weakest

    def should_attempt(self, capability_name: str,
                       risk_tolerance: float = 0.5) -> bool:
        """Decide whether to attempt something based on self-model.

        Args:
            capability_name: what to attempt
            risk_tolerance: 0=risk-averse, 1=risk-seeking
        """
        cap = self._capabilities.get(capability_name)
        if cap is None:
            return risk_tolerance > 0.7  # only if risk-seeking

        # Expected success probability
        p_success = cap.proficiency

        # Cognitive load penalty
        load_penalty = self._cognitive_load.total_load * 0.3

        # Adjusted probability
        adjusted = p_success * (1 - load_penalty)

        # Threshold based on risk tolerance
        threshold = 0.5 - risk_tolerance * 0.3

        return adjusted > threshold

    def get_all_capabilities(self) -> dict[str, Capability]:
        """Get all capability assessments."""
        return dict(self._capabilities)

    def stats(self) -> dict:
        """Summary statistics."""
        caps = self._capabilities
        return {
            "n_capabilities": len(caps),
            "avg_proficiency": np.mean([c.proficiency for c in caps.values()]) if caps else 0,
            "avg_confidence": np.mean([c.confidence for c in caps.values()]) if caps else 0,
            "total_attempts": sum(c.attempts for c in caps.values()),
            "strengths": self._strengths,
            "weaknesses": self._weaknesses,
            "cognitive_load": self._cognitive_load.total_load,
            "current_confidence": self._current_confidence,
        }
