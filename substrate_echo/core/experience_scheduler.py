"""Experience Scheduler — Idle-time autonomous experience acquisition.

When the agent predicts unused cognitive capacity, it allocates that
time toward experiences selected by expected information gain.

The scheduler:
1. Detects upcoming idle periods
2. Analyzes knowledge weaknesses
3. Selects an appropriate training environment
4. Allocates simulation budget
5. Feeds acquired experience back into general cognition

Usage:
    scheduler = ExperienceScheduler(meta_cognition, self_model)

    # Each tick, check if idle
    if scheduler.should_activate():
        env = scheduler.select_environment()
        budget = scheduler.allocate_budget()
        adapter = scheduler.create_adapter(env)
        # ... run simulation ...
        scheduler.integrate_experience(adapter.export_experience())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from collections import defaultdict
from enum import Enum


class EnvironmentType(Enum):
    """Available training environments."""
    SPATIAL_NAVIGATION = "spatial_navigation"    # Minecraft-like
    ADVERSARIAL_PLANNING = "adversarial_planning"  # StarCraft-like
    SOCIAL_SIMULATION = "social_simulation"      # Civilization-like
    RESOURCE_MANAGEMENT = "resource_management"
    EXPLORATION = "exploration"
    GENERIC = "generic"


@dataclass
class EnvironmentProfile:
    """What an environment trains and its characteristics."""
    env_type: EnvironmentType
    name: str
    description: str = ""

    # Training targets: which modules benefit
    trains_modules: list[str] = field(default_factory=list)

    # Difficulty and characteristics
    difficulty: float = 0.5       # 0-1
    social_density: float = 0.0   # 0-1, how many other agents
    spatial_complexity: float = 0.0
    adversarial: bool = False
    resource_rich: bool = True

    # Expected information gain per module
    expected_gain: dict[str, float] = field(default_factory=dict)


@dataclass
class IdlePrediction:
    """Prediction of upcoming idle period."""
    predicted_duration: int = 0      # ticks
    confidence: float = 0.0          # 0-1
    current_load: float = 0.0        # 0-1, current cognitive load
    pending_goals: int = 0
    pending_tasks: int = 0


@dataclass
class WeaknessAnalysis:
    """Analysis of current knowledge weaknesses."""
    module_confidences: dict[str, float] = field(default_factory=dict)
    weakest_modules: list[str] = field(default_factory=list)
    curiosity_drive: float = 0.0
    information_gaps: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class ExperienceSchedulerConfig:
    """Configuration for the experience scheduler."""
    idle_threshold: float = 0.3        # cognitive load below this = idle
    min_idle_duration: int = 50        # minimum ticks to activate
    max_session_length: int = 500      # max ticks per session
    budget_per_tick: float = 1.0       # compute budget per tick
    exploration_rate: float = 0.2      # probability of exploring new env
    confidence_threshold: float = 0.5  # below this = weakness


# ── Environment Catalog ─────────────────────────────

ENVIRONMENT_CATALOG: dict[EnvironmentType, EnvironmentProfile] = {
    EnvironmentType.SPATIAL_NAVIGATION: EnvironmentProfile(
        env_type=EnvironmentType.SPATIAL_NAVIGATION,
        name="spatial_navigation",
        description="3D spatial environment with objects and resources",
        trains_modules=["dynamics_memory", "spatial_memory", "habit_formation",
                        "hierarchical_planner", "self_model"],
        difficulty=0.4,
        social_density=0.1,
        spatial_complexity=0.8,
        resource_rich=True,
        expected_gain={
            "dynamics_memory": 0.8,
            "habit_formation": 0.7,
            "hierarchical_planner": 0.6,
            "self_model": 0.5,
        },
    ),
    EnvironmentType.ADVERSARIAL_PLANNING: EnvironmentProfile(
        env_type=EnvironmentType.ADVERSARIAL_PLANNING,
        name="adversarial_planning",
        description="Strategic game with opponent modeling",
        trains_modules=["counterfactual", "hierarchical_planner",
                        "meta_cognition", "theory_of_mind"],
        difficulty=0.8,
        social_density=0.3,
        adversarial=True,
        expected_gain={
            "counterfactual": 0.9,
            "hierarchical_planner": 0.8,
            "meta_cognition": 0.7,
            "theory_of_mind": 0.5,
        },
    ),
    EnvironmentType.SOCIAL_SIMULATION: EnvironmentProfile(
        env_type=EnvironmentType.SOCIAL_SIMULATION,
        name="social_simulation",
        description="Multi-agent society with diplomacy and alliances",
        trains_modules=["theory_of_mind", "emotional_contagion",
                        "episodic_memory", "hierarchical_planner"],
        difficulty=0.6,
        social_density=0.9,
        expected_gain={
            "theory_of_mind": 0.9,
            "emotional_contagion": 0.8,
            "episodic_memory": 0.6,
            "hierarchical_planner": 0.5,
        },
    ),
    EnvironmentType.RESOURCE_MANAGEMENT: EnvironmentProfile(
        env_type=EnvironmentType.RESOURCE_MANAGEMENT,
        name="resource_management",
        description="Economy simulation with supply chains",
        trains_modules=["hierarchical_planner", "dynamics_memory",
                        "counterfactual", "self_model"],
        difficulty=0.5,
        resource_rich=True,
        expected_gain={
            "hierarchical_planner": 0.8,
            "dynamics_memory": 0.7,
            "counterfactual": 0.6,
        },
    ),
    EnvironmentType.EXPLORATION: EnvironmentProfile(
        env_type=EnvironmentType.EXPLORATION,
        name="exploration",
        description="Unknown environment with discovery incentives",
        trains_modules=["dynamics_memory", "curiosity", "episodic_memory",
                        "self_model"],
        difficulty=0.3,
        spatial_complexity=0.6,
        expected_gain={
            "dynamics_memory": 0.7,
            "episodic_memory": 0.7,
            "self_model": 0.5,
        },
    ),
}


class ExperienceScheduler:
    """Decides when and how to use idle time for learning.

    Monitors cognitive load, identifies weaknesses, selects training
    environments, and integrates acquired experience.

    Usage:
        scheduler = ExperienceScheduler(meta_cognition, self_model)

        if scheduler.should_activate():
            plan = scheduler.create_session()
            for tick in range(plan.budget):
                adapter = scheduler.get_adapter(plan)
                frame = adapter.observe(...)
                action = agent.think(frame)
                result = adapter.record_outcome(action, reward)
            scheduler.complete_session(plan, adapter)
    """

    def __init__(self,
                 meta_cognition=None,
                 self_model=None,
                 dynamics_memory=None,
                 config: Optional[ExperienceSchedulerConfig] = None):
        self.config = config or ExperienceSchedulerConfig()
        self._mc = meta_cognition
        self._sm = self_model
        self._dm = dynamics_memory

        self._session_active = False
        self._session_ticks = 0
        self._current_env: Optional[EnvironmentType] = None

        # History of sessions
        self._sessions: list[dict] = []

    def should_activate(self) -> bool:
        """Should the scheduler activate an idle learning session?

        Returns True if cognitive load is low enough and an idle
        period is predicted.
        """
        if self._session_active:
            return False

        load = self._get_cognitive_load()
        if load > self.config.idle_threshold:
            return False

        prediction = self.predict_idle_period()
        if prediction.predicted_duration < self.config.min_idle_duration:
            return False

        if prediction.confidence < 0.3:
            return False

        return True

    def predict_idle_period(self) -> IdlePrediction:
        """Predict upcoming idle period duration."""
        load = self._get_cognitive_load()
        pending = self._get_pending_work()

        # Simple heuristic: lower load + fewer pending = longer idle
        duration = int((1.0 - load) * 200 * (1.0 / max(1, pending)))
        confidence = max(0, min(1, 1.0 - load))

        return IdlePrediction(
            predicted_duration=min(duration, self.config.max_session_length),
            confidence=confidence,
            current_load=load,
            pending_goals=pending,
        )

    def analyze_weaknesses(self) -> WeaknessAnalysis:
        """Analyze current knowledge weaknesses across modules."""
        confidences = {}

        # Meta-cognition provides per-source trust
        if self._mc:
            meta_state = self._mc.get_meta_state()
            for source, trust in meta_state.source_trust.items():
                confidences[source] = trust

        # Self-model provides capability confidence
        if self._sm:
            strengths = self._sm._strengths
            weaknesses = self._sm._weaknesses
            for s in strengths:
                confidences[f"self_{s}"] = 0.8
            for w in weaknesses:
                confidences[f"self_{w}"] = 0.3

        # Information gaps from dynamics memory
        info_gaps = {}
        if self._dm:
            try:
                # Use novelty as information gap proxy
                test_state = np.random.randn(16) * 0.5
                novelty = self._dm.novelty(test_state)
                info_gaps["dynamics_novelty"] = novelty
            except Exception:
                pass

        # Find weakest modules
        if confidences:
            sorted_modules = sorted(confidences.items(), key=lambda x: x[1])
            weakest = [m for m, c in sorted_modules
                      if c < self.config.confidence_threshold]
        else:
            weakest = ["general"]

        # Generate recommendation
        recommendation = self._generate_recommendation(weakest, info_gaps)

        return WeaknessAnalysis(
            module_confidences=confidences,
            weakest_modules=weakest,
            curiosity_drive=1.0 - np.mean(list(confidences.values())) if confidences else 0.5,
            information_gaps=info_gaps,
            recommendation=recommendation,
        )

    def select_environment(self) -> EnvironmentType:
        """Select the best environment based on weaknesses."""
        weakness = self.analyze_weaknesses()

        # Score each environment by expected gain for weak modules
        scores = {}
        for env_type, profile in ENVIRONMENT_CATALOG.items():
            score = 0.0
            for module in weakness.weakest_modules:
                gain = profile.expected_gain.get(module, 0.0)
                score += gain

            # Bonus for exploration rate
            if np.random.random() < self.config.exploration_rate:
                score += 0.3

            scores[env_type] = score

        if not scores:
            return EnvironmentType.GENERIC

        return max(scores, key=scores.get)

    def create_session(self) -> dict:
        """Create a learning session plan."""
        env_type = self.select_environment()
        weakness = self.analyze_weaknesses()
        prediction = self.predict_idle_period()

        budget = min(prediction.predicted_duration,
                     self.config.max_session_length)

        session = {
            "env_type": env_type,
            "budget": budget,
            "weakness": weakness,
            "prediction": prediction,
            "target_modules": weakness.weakest_modules[:3],
        }

        self._session_active = True
        self._session_ticks = 0
        self._current_env = env_type

        return session

    def get_adapter(self, session: dict):
        """Get an ExperienceAdapter for the session's environment."""
        from substrate_echo.core.experience_adapter import ExperienceAdapter
        env_type = session["env_type"]
        profile = ENVIRONMENT_CATALOG.get(env_type)
        if profile is None:
            profile = ENVIRONMENT_CATALOG[EnvironmentType.EXPLORATION]
        return ExperienceAdapter(env_name=profile.name)

    def tick_session(self) -> bool:
        """Tick the active session. Returns False when budget exhausted."""
        if not self._session_active:
            return False

        self._session_ticks += 1
        if self._current_env:
            profile = ENVIRONMENT_CATALOG.get(self._current_env)
            if profile and self._session_ticks >= profile.difficulty * 100:
                return True  # still running

        return self._session_ticks < self.config.max_session_length

    def complete_session(self, session: dict, adapter=None) -> dict:
        """Complete a learning session and integrate experience."""
        result = {
            "env_type": session["env_type"],
            "ticks_run": self._session_ticks,
            "budget": session["budget"],
            "target_modules": session["target_modules"],
        }

        # Export experience if adapter available
        if adapter:
            experiences = adapter.export_experience()
            result["experiences_acquired"] = len(experiences)
            result["information_gaps"] = adapter.get_information_gap()

        self._sessions.append(result)
        self._session_active = False
        self._session_ticks = 0
        self._current_env = None

        return result

    def get_session_history(self) -> list[dict]:
        """Get history of completed sessions."""
        return list(self._sessions)

    def stats(self) -> dict:
        """Scheduler statistics."""
        return {
            "sessions_completed": len(self._sessions),
            "session_active": self._session_active,
            "current_env": self._current_env.value if self._current_env else None,
        }

    # ── Private helpers ──────────────────────────────────────

    def _get_cognitive_load(self) -> float:
        """Estimate current cognitive load (0-1)."""
        if self._mc:
            meta_state = self._mc.get_meta_state()
            # High disagreement = high load
            load = meta_state.model_disagreement * 0.5
            # High overconfidence = some load
            load += meta_state.overconfidence * 0.3
            # Low confidence = processing load
            load += (1.0 - meta_state.calibrated_confidence) * 0.2
            return min(1.0, load)
        return 0.3  # default moderate load

    def _get_pending_work(self) -> int:
        """Count pending goals/tasks."""
        # Simple heuristic
        return 1  # assume at least one pending thing

    def _generate_recommendation(self, weakest: list[str],
                                 info_gaps: dict[str, float]) -> str:
        """Generate human-readable recommendation."""
        if not weakest or weakest == ["general"]:
            return "No specific weaknesses detected. Explore freely."

        # Map weaknesses to environments
        module_to_env = {
            "dynamics_memory": "spatial_navigation",
            "habit_formation": "spatial_navigation",
            "counterfactual": "adversarial_planning",
            "hierarchical_planner": "resource_management",
            "theory_of_mind": "social_simulation",
            "emotional_contagion": "social_simulation",
            "self_model": "exploration",
            "meta_cognition": "adversarial_planning",
        }

        env_scores = defaultdict(float)
        for module in weakest:
            env = module_to_env.get(module, "generic")
            env_scores[env] += 1.0

        best_env = max(env_scores, key=env_scores.get) if env_scores else "generic"
        return f"Weakest: {', '.join(weakest[:3])}. Recommend: {best_env}."
