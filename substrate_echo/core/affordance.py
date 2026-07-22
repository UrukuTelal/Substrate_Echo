"""Affordance & Intent Translation Layer.

The critical bridge between external perception and internal reasoning.

Perception answers: "What exists?"
Reasoning answers: "What should I do?"
This layer answers: "What does this mean relative to my own goals,
memories, and possible actions?"

The perception isn't complete until it has been translated into
action possibilities. This closes the perception-cognition-action loop:

    Environment → Perception → World Model → AFFORDANCE TRANSLATION
    → Pillar Space → Planning → Action → Environment

Every perceived entity is translated into an Affordance — a structured
representation of what actions are possible and what they mean for the
agent's cognitive state (PSV).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


# ── Entity Types ─────────────────────────────────────────────────

class EntityType(Enum):
    """Types of entities the companion can perceive."""
    HUMAN = auto()
    ANIMAL = auto()
    PLANT = auto()
    MYCELIUM = auto()
    OBJECT = auto()
    TERRAIN = auto()
    AGENT = auto()       # artificial agent (robot, AI)
    UNKNOWN = auto()


class ActionType(Enum):
    """Actions the AI can take toward an entity."""
    APPROACH = auto()
    RETREAT = auto()
    GRASP = auto()
    RELEASE = auto()
    HAND_OVER = auto()
    REFILL = auto()
    OBSERVE = auto()
    COMMUNICATE = auto()
    ASSIST = auto()
    DEFEND = auto()
    AVOID = auto()
    MONITOR = auto()
    IGNORE = auto()
    INVESTIGATE = auto()


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class PropertyMap:
    """Properties of a perceived entity.
    
    These are raw observations, not interpretations.
    """
    temperature: float = 0.0
    moisture: float = 0.0
    mass: float = 0.0
    toxicity: float = 0.0
    luminance: float = 0.0
    sound_level: float = 0.0
    motion_speed: float = 0.0
    distance: float = 0.0
    ownership: bool = False       # does the human own this?
    familiar: bool = False        # have we seen this before?
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.temperature, self.moisture, self.mass,
            self.toxicity, self.luminance, self.sound_level,
            self.motion_speed, self.distance,
            float(self.ownership), float(self.familiar),
        ])


@dataclass
class GoalEstimate:
    """Estimated goal of a perceived entity."""
    description: str = ""
    confidence: float = 0.0       # [0, 1]
    target_state: Optional[np.ndarray] = None  # estimated goal state
    trajectory: list[np.ndarray] = field(default_factory=list)  # recent path


@dataclass
class ActionAffordance:
    """A possible action toward a perceived entity."""
    action: ActionType
    feasibility: float = 0.0      # [0, 1] how feasible given current state
    cost: float = 0.0             # [0, 1] energy/risk cost
    benefit: float = 0.0          # [0, 1] expected gain toward goals
    
    @property
    def score(self) -> float:
        """Net value of this action."""
        return self.benefit * self.feasibility - self.cost * 0.3


@dataclass
class PSVDelta:
    """A proposed change to the agent's Pillar State Vector.
    
    This is the core output of the translation layer — it converts
    external observations into the agent's native cognitive language.
    """
    pillar_idx: int
    delta_theta: float    # rotation amount (positive = toward north pole)
    reason: str = ""      # human-readable explanation
    
    def __repr__(self):
        PILLAR_NAMES = [
            "Awareness", "Willpower", "Force", "Influence",
            "Resistance", "Integrity", "Cohesion", "Relation",
            "Presence", "Warmth", "Memory", "Attraction",
            "Harm", "Distortion", "Flux", "Depth",
        ]
        name = PILLAR_NAMES[self.pillar_idx] if self.pillar_idx < 16 else f"P{self.pillar_idx}"
        direction = "↑" if self.delta_theta > 0 else "↓"
        return f"PSV({name} {direction} {abs(self.delta_theta):.3f}: {self.reason})"


@dataclass
class Affordance:
    """Structured representation of what an observed entity means.
    
    An affordance bridges perception and action. It says:
    - What exists (entity, properties, position)
    - What I can do (possible_actions)
    - What it means (inferred goal, social intent)
    - How it changes me (PSV deltas)
    """
    entity_id: int
    entity_type: EntityType
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    properties: PropertyMap = field(default_factory=PropertyMap)
    
    # What I can do
    possible_actions: list[ActionAffordance] = field(default_factory=list)
    
    # What it means
    inferred_goal: Optional[GoalEstimate] = None
    social_intent: float = 0.0     # [0, 1] is this directed at me?
    requires_response: bool = False
    
    # How it changes me
    pillar_deltas: list[PSVDelta] = field(default_factory=list)
    
    @property
    def best_action(self) -> Optional[ActionAffordance]:
        """The highest-scoring possible action."""
        if not self.possible_actions:
            return None
        return max(self.possible_actions, key=lambda a: a.score)
    
    @property
    def total_psv_magnitude(self) -> float:
        """Total magnitude of PSV changes."""
        return sum(abs(d.delta_theta) for d in self.pillar_deltas)


# ── Translation Rules ────────────────────────────────────────────

# Default PSV translation rules: (entity_type, action_type) → [(pillar_idx, delta, reason)]
DEFAULT_TRANSLATION_RULES: dict[tuple, list[tuple[int, float, str]]] = {
    # Human approaching
    (EntityType.HUMAN, ActionType.APPROACH): [
        (7, 0.15, "social connection increasing"),
        (8, 0.10, "human presence salient"),
    ],
    # Human communicating
    (EntityType.HUMAN, ActionType.COMMUNICATE): [
        (7, 0.20, "communication intent detected"),
        (8, 0.15, "social presence"),
        (3, 0.10, "influence opportunity"),
    ],
    # Human needs assistance
    (EntityType.HUMAN, ActionType.ASSIST): [
        (9, 0.20, "helping intent"),
        (7, 0.15, "social bond strengthening"),
        (2, 0.10, "executive action"),
    ],
    # Animal approaching
    (EntityType.ANIMAL, ActionType.APPROACH): [
        (7, 0.10, "animal social contact"),
        (0, 0.05, "awareness increase"),
    ],
    # Animal needs attention
    (EntityType.ANIMAL, ActionType.MONITOR): [
        (0, 0.10, "heightened awareness"),
        (9, 0.05, "care intent"),
    ],
    # Plant needs water
    (EntityType.PLANT, ActionType.ASSIST): [
        (9, 0.15, "nurturing intent"),
        (2, 0.10, "action required"),
        (10, 0.05, "remembering care history"),
    ],
    # Plant observation
    (EntityType.PLANT, ActionType.OBSERVE): [
        (0, 0.05, "perceptual openness"),
    ],
    # Mycelium network activity
    (EntityType.MYCELIUM, ActionType.MONITOR): [
        (0, 0.10, "network awareness"),
        (14, 0.05, "information flow"),
    ],
    # Threat detected
    (EntityType.UNKNOWN, ActionType.DEFEND): [
        (4, 0.20, "defensive response"),
        (0, 0.15, "threat awareness"),
        (12, 0.10, "harm detection"),
    ],
    # Obstacle detected
    (EntityType.TERRAIN, ActionType.AVOID): [
        (4, 0.10, "boundary enforcement"),
        (0, 0.10, "spatial awareness"),
    ],
    # New territory
    (EntityType.TERRAIN, ActionType.INVESTIGATE): [
        (15, 0.10, "latent exploration"),
        (14, 0.05, "information flow"),
    ],
    # Safe familiar environment
    (EntityType.OBJECT, ActionType.OBSERVE): [
        (5, 0.05, "stability reinforced"),
    ],
    # Object interaction
    (EntityType.OBJECT, ActionType.GRASP): [
        (2, 0.10, "executive action"),
        (8, 0.05, "physical presence"),
    ],
}


# ── Intent Translator ────────────────────────────────────────────

class IntentTranslator:
    """Translates perceived entities into affordances and PSV deltas.
    
    The translation pipeline:
    1. Raw entity → EntityObservation (properties, position, behavior)
    2. EntityObservation → Affordance (what can I do?)
    3. Affordance → PSVDelta (how does this change my cognitive state?)
    
    This closes the perception-cognition-action loop by projecting
    external observations into the same latent space where reasoning
    occurs.
    
    The translator is configurable via translation rules and can be
    extended with domain-specific logic (plant care, animal behavior,
    human interaction patterns).
    """
    
    def __init__(self, dim: int = 16,
                 translation_rules: Optional[dict] = None,
                 world_model: Optional[object] = None):
        self.dim = dim
        self.rules = translation_rules or DEFAULT_TRANSLATION_RULES
        self.world_model = world_model
        
        # History for goal inference
        self._entity_histories: dict[int, list[np.ndarray]] = {}
        self._max_history = 20
    
    def translate(self, entity_id: int, entity_type: EntityType,
                  position: np.ndarray, properties: Optional[PropertyMap] = None,
                  social_intent_hint: float = 0.0,
                  hsv_state: Optional[object] = None) -> Affordance:
        """Translate a perceived entity into an Affordance.
        
        Args:
            entity_id: unique identifier for this entity
            entity_type: what kind of entity is this
            position: 3D position of the entity
            properties: observed properties (temperature, moisture, etc.)
            social_intent_hint: external hint about social intent
            hsv_state: optional HSVState from HumanStateEstimator — when
                provided, HSV dimensions weight the PSV deltas. Same
                behavior (human approaches) can mean friendly or threatening;
                HSV provides the context to disambiguate.
        
        Returns:
            Affordance with possible_actions, inferred_goal, and pillar_deltas
        """
        position = np.asarray(position, dtype=np.float64)
        properties = properties or PropertyMap()
        
        # Update entity history for goal inference
        self._update_history(entity_id, position)
        
        # Step 1: Evaluate possible actions
        possible_actions = self._evaluate_actions(
            entity_type, properties, social_intent_hint)
        
        # Step 1b: Boost INVESTIGATE benefit based on novelty at entity position
        possible_actions = self._boost_investigate_by_novelty(
            possible_actions, position)
        
        # Step 2: Infer goal from trajectory
        inferred_goal = self._infer_goal(entity_id, entity_type)
        
        # Step 3: Detect social intent
        social_intent = self._detect_social_intent(
            entity_type, properties, social_intent_hint)
        
        # Step 3b: HSV modulation — if human state estimate available,
        # weight social_intent and deltas by HSV dimensions
        hsv_confidence = 0.0
        if hsv_state is not None and entity_type == EntityType.HUMAN:
            social_intent = self._hsv_modulate_social_intent(
                social_intent, hsv_state)
            hsv_confidence = hsv_state.confidence
        
        # Step 4: Generate PSV deltas
        pillar_deltas = self._generate_psv_deltas(
            entity_type, possible_actions, social_intent, properties)
        
        # Step 4b: HSV-weighted delta scaling — confidence in the human
        # state estimate scales how much we let it influence our PSV
        if hsv_state is not None and entity_type == EntityType.HUMAN:
            pillar_deltas = self._hsv_scale_deltas(
                pillar_deltas, hsv_state)
        
        # Step 5: Determine if response required
        requires_response = (
            social_intent > 0.5 or
            (possible_actions and possible_actions[0].score > 0.5) or
            properties.toxicity > 0.5 or
            properties.motion_speed > 0.8
        )
        
        return Affordance(
            entity_id=entity_id,
            entity_type=entity_type,
            position=position,
            properties=properties,
            possible_actions=possible_actions,
            inferred_goal=inferred_goal,
            social_intent=social_intent,
            requires_response=requires_response,
            pillar_deltas=pillar_deltas,
        )
    
    def translate_batch(self, observations: list[dict]) -> list[Affordance]:
        """Translate multiple observations at once.
        
        Each observation is a dict with keys:
            entity_id, entity_type, position, properties (optional),
            social_intent_hint (optional)
        """
        return [
            self.translate(
                entity_id=obs["entity_id"],
                entity_type=obs["entity_type"],
                position=obs["position"],
                properties=obs.get("properties"),
                social_intent_hint=obs.get("social_intent_hint", 0.0),
            )
            for obs in observations
        ]
    
    def _evaluate_actions(self, entity_type: EntityType,
                          properties: PropertyMap,
                          social_hint: float) -> list[ActionAffordance]:
        """Evaluate what actions are possible toward this entity."""
        actions = []
        
        if entity_type == EntityType.HUMAN:
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE, feasibility=1.0, cost=0.0, benefit=0.2))
            actions.append(ActionAffordance(
                action=ActionType.COMMUNICATE,
                feasibility=0.8 if social_hint > 0.3 else 0.3,
                cost=0.1, benefit=0.4 if social_hint > 0.3 else 0.1))
            actions.append(ActionAffordance(
                action=ActionType.ASSIST,
                feasibility=0.6, cost=0.3, benefit=0.5))
            actions.append(ActionAffordance(
                action=ActionType.APPROACH,
                feasibility=0.7, cost=0.1, benefit=0.3))
            actions.append(ActionAffordance(
                action=ActionType.RETREAT,
                feasibility=0.8, cost=0.05, benefit=0.1))
        
        elif entity_type == EntityType.ANIMAL:
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE, feasibility=1.0, cost=0.0, benefit=0.2))
            actions.append(ActionAffordance(
                action=ActionType.APPROACH,
                feasibility=0.5, cost=0.15, benefit=0.3))
            actions.append(ActionAffordance(
                action=ActionType.MONITOR,
                feasibility=0.9, cost=0.0, benefit=0.2))
            actions.append(ActionAffordance(
                action=ActionType.RETREAT,
                feasibility=0.8, cost=0.05, benefit=0.1))
        
        elif entity_type == EntityType.PLANT:
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE, feasibility=1.0, cost=0.0, benefit=0.1))
            actions.append(ActionAffordance(
                action=ActionType.ASSIST,
                feasibility=0.7, cost=0.2,
                benefit=0.4 if properties.moisture < 0.3 else 0.1))
            actions.append(ActionAffordance(
                action=ActionType.MONITOR,
                feasibility=0.9, cost=0.0, benefit=0.15))
        
        elif entity_type == EntityType.MYCELIUM:
            actions.append(ActionAffordance(
                action=ActionType.MONITOR, feasibility=1.0, cost=0.0, benefit=0.2))
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE, feasibility=1.0, cost=0.0, benefit=0.15))
        
        elif entity_type == EntityType.TERRAIN:
            actions.append(ActionAffordance(
                action=ActionType.INVESTIGATE,
                feasibility=0.7, cost=0.2, benefit=0.3))
            actions.append(ActionAffordance(
                action=ActionType.AVOID,
                feasibility=0.9, cost=0.05,
                benefit=0.3 if properties.toxicity > 0.3 else 0.05))
            actions.append(ActionAffordance(
                action=ActionType.APPROACH,
                feasibility=0.8, cost=0.1, benefit=0.2))
        
        elif entity_type == EntityType.OBJECT:
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE, feasibility=1.0, cost=0.0, benefit=0.1))
            actions.append(ActionAffordance(
                action=ActionType.GRASP,
                feasibility=0.6, cost=0.15, benefit=0.3))
            actions.append(ActionAffordance(
                action=ActionType.IGNORE,
                feasibility=1.0, cost=0.0, benefit=0.0))
        
        elif entity_type == EntityType.UNKNOWN:
            actions.append(ActionAffordance(
                action=ActionType.INVESTIGATE,
                feasibility=0.5, cost=0.3, benefit=0.4))
            actions.append(ActionAffordance(
                action=ActionType.DEFEND,
                feasibility=0.7, cost=0.2, benefit=0.3))
            actions.append(ActionAffordance(
                action=ActionType.OBSERVE,
                feasibility=1.0, cost=0.0, benefit=0.2))
        
        # Sort by score
        actions.sort(key=lambda a: a.score, reverse=True)
        return actions
    
    def _boost_investigate_by_novelty(self, actions: list[ActionAffordance],
                                       position: np.ndarray) -> list[ActionAffordance]:
        """Boost INVESTIGATE/OBSERVE benefit based on DynamicsMemory novelty.

        When the entity is in a novel region (far from training data),
        investigating it provides more information gain.
        """
        if self.world_model is None:
            return actions
        dm = getattr(self.world_model, 'memory', None)
        if dm is None or not hasattr(dm, 'novelty') or not dm._fitted or not dm._states:
            return actions
        
        novelty = dm.novelty(position)
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
            novelty_normalized = min(1.0, novelty / max(avg_spacing * 2, 1e-6))
        else:
            novelty_normalized = min(1.0, novelty)
        
        if novelty_normalized > 0.3:
            for action in actions:
                if action.action in (ActionType.INVESTIGATE, ActionType.OBSERVE):
                    action.benefit = min(1.0, action.benefit + novelty_normalized * 0.4)
        
        return actions
    
    def _infer_goal(self, entity_id: int,
                    entity_type: EntityType) -> Optional[GoalEstimate]:
        """Infer what the entity is trying to achieve from its trajectory."""
        history = self._entity_histories.get(entity_id, [])
        if len(history) < 3:
            return None
        
        recent = np.array(history[-5:])
        
        # Simple trajectory analysis: direction and speed
        if len(recent) >= 2:
            velocity = recent[-1] - recent[-2]
            speed = np.linalg.norm(velocity)
            direction = velocity / (speed + 1e-10)
            
            # Confidence increases with consistent direction
            if len(recent) >= 3:
                v_prev = recent[-2] - recent[-3]
                cos_sim = np.dot(direction, v_prev / (np.linalg.norm(v_prev) + 1e-10))
                confidence = max(0.0, min(1.0, (cos_sim + 1) / 2 * speed * 10))
            else:
                confidence = min(1.0, speed * 5)
            
            goal_desc = f"moving toward ({direction[0]:.2f}, {direction[1]:.2f})"
            if speed < 0.01:
                goal_desc = "stationary"
                confidence = max(0.3, confidence)
            
            return GoalEstimate(
                description=goal_desc,
                confidence=confidence,
                trajectory=history[-5:],
            )
        
        return None
    
    def _detect_social_intent(self, entity_type: EntityType,
                              properties: PropertyMap,
                              hint: float) -> float:
        """Detect if entity's behavior is directed at the AI.
        
        Social intent is high when:
        - Entity is close (small distance)
        - Entity is moving toward us
        - External hints suggest communication
        """
        base = hint
        
        # Distance modulation
        if properties.distance < 0.3:
            base += 0.2
        elif properties.distance < 0.6:
            base += 0.1
        
        # Motion modulation
        if properties.motion_speed > 0.5:
            base += 0.1
        
        # Sound modulation
        if properties.sound_level > 0.5:
            base += 0.15
        
        return max(0.0, min(1.0, base))
    
    def _generate_psv_deltas(self, entity_type: EntityType,
                             actions: list[ActionAffordance],
                             social_intent: float,
                             properties: PropertyMap) -> list[PSVDelta]:
        """Generate PSV changes based on the entity and best action.
        
        This is the core of the translation layer — it converts
        external observations into the agent's native cognitive language.
        """
        deltas = []
        
        if not actions:
            return deltas
        
        best = actions[0]
        rule_key = (entity_type, best.action)
        
        # Look up translation rules
        if rule_key in self.rules:
            for pillar_idx, delta, reason in self.rules[rule_key]:
                if pillar_idx < self.dim:
                    deltas.append(PSVDelta(
                        pillar_idx=pillar_idx,
                        delta_theta=delta,
                        reason=reason,
                    ))
        
        # Social intent adds Relation and Presence
        if social_intent > 0.3:
            deltas.append(PSVDelta(
                pillar_idx=7,  # Relation
                delta_theta=social_intent * 0.15,
                reason=f"social intent detected ({social_intent:.2f})",
            ))
            deltas.append(PSVDelta(
                pillar_idx=8,  # Presence
                delta_theta=social_intent * 0.10,
                reason="social presence increasing",
            ))
        
        # High toxicity/danger adds defensive pillars
        if properties.toxicity > 0.5:
            deltas.append(PSVDelta(
                pillar_idx=4,  # Resistance
                delta_theta=properties.toxicity * 0.15,
                reason="hazard detected",
            ))
            deltas.append(PSVDelta(
                pillar_idx=0,  # Awareness
                delta_theta=properties.toxicity * 0.10,
                reason="threat awareness",
            ))
        
        # Familiar entities slightly boost Memory
        if properties.familiar:
            deltas.append(PSVDelta(
                pillar_idx=10,  # Memory
                delta_theta=0.05,
                reason="familiar entity recognized",
            ))
        
        return deltas
    
    def _update_history(self, entity_id: int, position: np.ndarray):
        """Track entity position history for goal inference."""
        if entity_id not in self._entity_histories:
            self._entity_histories[entity_id] = []
        self._entity_histories[entity_id].append(position.copy())
        # Keep bounded
        if len(self._entity_histories[entity_id]) > self._max_history:
            self._entity_histories[entity_id] = \
                self._entity_histories[entity_id][-self._max_history:]
    
    def _hsv_modulate_social_intent(self, base_social: float,
                                     hsv_state) -> float:
        """Modulate social_intent using HSV social_openness and arousal.
        
        High social_openness + moderate arousal → social intent increases.
        Low social_openness or high fatigue → social intent decreases.
        High arousal + low valence → ambiguous (could be threat).
        """
        so = hsv_state.social_openness.mean
        arousal = hsv_state.arousal.mean
        fatigue = hsv_state.fatigue.mean
        valence = hsv_state.valence.mean
        conf = hsv_state.confidence
        
        # Social openness amplifies or suppresses social intent
        openness_mod = (so - 0.5) * 0.4  # range: [-0.2, +0.2]
        
        # Fatigue suppresses social engagement
        fatigue_mod = -(fatigue - 0.5) * 0.2  # range: [-0.1, +0.1]
        
        # High arousal + low valence = potential threat, reduce social
        threat_mod = 0.0
        if arousal > 0.6 and valence < 0.4:
            threat_mod = -0.15
        
        modulated = base_social + conf * (openness_mod + fatigue_mod + threat_mod)
        return max(0.0, min(1.0, modulated))
    
    def _hsv_scale_deltas(self, deltas: list[PSVDelta],
                          hsv_state) -> list[PSVDelta]:
        """Scale PSV deltas by HSV state.
        
        High arousal → boost Force/Willpower deltas, dampen Warmth.
        High social_openness → boost Relation/Warmth, dampen Resistance.
        High fatigue → dampen all deltas (less capacity to respond).
        Low stability → dampen deltas (volatile state = cautious).
        """
        arousal = hsv_state.arousal.mean
        so = hsv_state.social_openness.mean
        fatigue = hsv_state.fatigue.mean
        stability = hsv_state.stability.mean
        conf = hsv_state.confidence
        
        scaled = []
        for delta in deltas:
            scale = 1.0
            
            # Pillar-specific HSV modulation
            if delta.pillar_idx in (2, 1):  # Force, Willpower
                # Arousal boosts action-oriented pillars
                scale += conf * (arousal - 0.5) * 0.3
            
            elif delta.pillar_idx in (9, 7):  # Warmth, Relation
                # Social openness boosts social pillars
                scale += conf * (so - 0.5) * 0.3
            
            elif delta.pillar_idx == 4:  # Resistance
                # Social openness dampens defensiveness
                scale -= conf * (so - 0.5) * 0.2
            
            # Global fatigue damping
            scale *= (1.0 - fatigue * 0.3 * conf)
            
            # Global stability modulation
            scale *= (0.7 + stability * 0.3 * conf)
            
            scaled.append(PSVDelta(
                pillar_idx=delta.pillar_idx,
                delta_theta=delta.delta_theta * scale,
                reason=delta.reason,
            ))
        
        return scaled
    
    def apply_deltas(self, state: np.ndarray,
                     affordances: list[Affordance]) -> np.ndarray:
        """Apply all PSV deltas from affordances to the current state.
        
        This projects external observations into the agent's cognitive
        state — the final step of the perception-cognition loop.
        """
        new_state = state.copy()
        
        for affordance in affordances:
            for delta in affordance.pillar_deltas:
                if 0 <= delta.pillar_idx < len(new_state):
                    # Simple additive (could be Bloch rotation in full impl)
                    new_state[delta.pillar_idx] += delta.delta_theta
                    new_state[delta.pillar_idx] = np.clip(
                        new_state[delta.pillar_idx], 0.0, 1.0)
        
        return new_state
