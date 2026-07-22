"""Communicative Intent Detection — Stage 5 of the developmental architecture.

"Which behaviors are communication? Which are private? Which invite
interaction?"

This module analyzes behavioral signals from observed entities and
classifies their communicative intent. It distinguishes between:

  - PRIVATE actions: scratching head, self-grooming, looking around
    (not directed at anyone)
  - DIRECTED communication: pointing at me, calling me, waving
    (explicitly directed at the AI)
  - AMBIGUOUS signals: looking in my direction, moving toward me
    (could be communicative or coincidental)

The detector outputs a CommunicativeSignal with classification and
confidence, which the IntentTranslator and Planner consume.

Key signals:
  - Gaze direction relative to AI position
  - Speech directed at AI (speech_level * gaze_co-occurrence)
  - Gestural intent (arm/hand motion patterns)
  - Proximity approach pattern
  - Temporal regularity (repeated signals = intentional)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


class CommunicativeIntent(Enum):
    """Classification of communicative behavior."""
    NONE = auto()
    REQUEST = auto()         # "I need something from you"
    GREETING = auto()        # "I acknowledge your presence"
    WARNING = auto()         # "Danger / attention"
    INFORMATION = auto()     # "I'm sharing this with you"
    BID_FOR_ATTENTION = auto()  # "Look at me"
    INVITATION = auto()      # "Come here / join me"
    DIRECTIVE = auto()       # "Do this"


@dataclass
class BehavioralSignals:
    """Raw behavioral observations for communicative intent analysis.
    
    These are the inputs to the detector. They come from the
    perception layer (camera, microphone, sensors).
    """
    # Gaze
    gaze_direction: np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # unit vector
    gaze_confidence: float = 0.0   # [0, 1]
    
    # Speech
    speech_level: float = 0.0      # [0, 1] volume / intensity
    speech_duration: float = 0.0   # seconds of current utterance
    
    # Gestures
    gesture_speed: float = 0.0     # [0, 1] hand/arm motion speed
    gesture_direction: np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # where gestures point
    gesture_repetition: int = 0    # repeated gesture count
    
    # Body orientation
    facing_toward_me: bool = False
    body_angle_to_me: float = 90.0  # degrees from facing direction to AI
    
    # Approach
    distance: float = 1.0
    approach_speed: float = 0.0    # [0, 1] speed toward AI
    
    # Temporal
    signal_duration: float = 0.0   # how long this signal pattern has persisted
    signal_repetition: int = 0     # how many times this pattern has recurred
    
    # Context
    time_of_day: float = 0.5       # [0, 1] normalized time
    environment: str = "indoor"    # indoor, outdoor, forest


@dataclass
class CommunicativeSignal:
    """Result of communicative intent analysis."""
    intent: CommunicativeIntent = CommunicativeIntent.NONE
    confidence: float = 0.0        # [0, 1]
    is_directed_at_me: bool = False
    signal_strength: float = 0.0   # [0, 1]
    
    # Evidence breakdown
    gaze_evidence: float = 0.0
    speech_evidence: float = 0.0
    gesture_evidence: float = 0.0
    proximity_evidence: float = 0.0
    
    # Classification details
    is_request: bool = False
    is_warning: bool = False
    is_greeting: bool = False
    requires_response: bool = False
    
    @property
    def evidence_total(self) -> float:
        return self.gaze_evidence + self.speech_evidence + \
               self.gesture_evidence + self.proximity_evidence


class CommunicativeIntentDetector:
    """Detects whether an observed entity's behavior is communicative.
    
    Usage:
        detector = CommunicativeIntentDetector(ai_position=my_pos)
        signals = BehavioralSignals(
            gaze_direction=np.array([0.8, 0.0, 0.0]),
            gaze_confidence=0.9,
            speech_level=0.5,
            facing_toward_me=True,
            distance=1.5,
        )
        result = detector.analyze(signals)
        if result.is_directed_at_me and result.confidence > 0.5:
            # This entity is communicating with me
            pass
    """
    
    def __init__(self, ai_position: Optional[np.ndarray] = None):
        self.ai_position = ai_position if ai_position is not None else np.zeros(3)
        self._history: list[CommunicativeSignal] = []
        self._max_history = 20
    
    def set_ai_position(self, pos: np.ndarray):
        self.ai_position = np.asarray(pos, dtype=np.float64)
    
    def analyze(self, signals: BehavioralSignals,
                entity_position: Optional[np.ndarray] = None,
                entity_velocity: Optional[np.ndarray] = None) -> CommunicativeSignal:
        """Analyze behavioral signals for communicative intent.
        
        Args:
            signals: observed behavioral signals
            entity_position: where the entity is (for gaze validation)
            entity_velocity: entity's velocity (for approach detection)
        
        Returns:
            CommunicativeSignal with classification and confidence
        """
        # Step 1: Compute evidence components
        gaze_ev = self._gaze_evidence(signals, entity_position)
        speech_ev = self._speech_evidence(signals)
        gesture_ev = self._gesture_evidence(signals, entity_position)
        proximity_ev = self._proximity_evidence(signals, entity_velocity)
        
        # Step 2: Determine if directed at me
        directed = self._is_directed_at_me(
            signals, gaze_ev, entity_position)
        
        # Step 3: Classify intent type
        intent, intent_confidence = self._classify_intent(
            signals, gaze_ev, speech_ev, gesture_ev, proximity_ev, directed)
        
        # Step 4: Determine signal strength
        strength = min(1.0, gaze_ev * 0.3 + speech_ev * 0.3 +
                       gesture_ev * 0.25 + proximity_ev * 0.15)
        
        # Step 5: Build result
        result = CommunicativeSignal(
            intent=intent,
            confidence=intent_confidence,
            is_directed_at_me=directed,
            signal_strength=strength,
            gaze_evidence=gaze_ev,
            speech_evidence=speech_ev,
            gesture_evidence=gesture_ev,
            proximity_evidence=proximity_ev,
        )
        
        # Step 6: Refine with temporal pattern
        self._apply_temporal_refinement(result)
        
        # Step 7: Classify specific types
        result.is_request = (
            intent == CommunicativeIntent.REQUEST and result.confidence > 0.4)
        result.is_warning = (
            intent == CommunicativeIntent.WARNING and result.confidence > 0.4)
        result.is_greeting = (
            intent == CommunicativeIntent.GREETING and result.confidence > 0.4)
        result.requires_response = (
            result.is_directed_at_me and result.confidence > 0.3 and
            intent in (CommunicativeIntent.REQUEST,
                       CommunicativeIntent.WARNING,
                       CommunicativeIntent.INVITATION,
                       CommunicativeIntent.DIRECTIVE))
        
        # Store history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return result
    
    def _gaze_evidence(self, signals: BehavioralSignals,
                       entity_position: Optional[np.ndarray]) -> float:
        """How strongly is the entity looking at the AI?"""
        if signals.gaze_confidence < 0.1:
            return 0.0
        
        if entity_position is not None:
            # Compute expected gaze direction to AI
            to_ai = self.ai_position - entity_position
            dist = np.linalg.norm(to_ai)
            if dist > 1e-6:
                expected_gaze = to_ai / dist
                # Cosine similarity between gaze and expected direction
                cos_sim = np.dot(signals.gaze_direction, expected_gaze)
                alignment = max(0.0, cos_sim)  # 1 = looking at me, 0 = away
                return alignment * signals.gaze_confidence
        
        # Fallback: use gaze confidence as proxy
        return signals.gaze_confidence * 0.5
    
    def _speech_evidence(self, signals: BehavioralSignals) -> float:
        """How much does speech suggest communication?"""
        if signals.speech_level < 0.1:
            return 0.0
        
        # Speech + facing = communicative
        base = signals.speech_level * 0.5
        
        # Longer speech = more likely communicative
        duration_bonus = min(0.3, signals.speech_duration * 0.1)
        
        return min(1.0, base + duration_bonus)
    
    def _gesture_evidence(self, signals: BehavioralSignals,
                          entity_position: Optional[np.ndarray]) -> float:
        """How much do gestures suggest communication?"""
        if signals.gesture_speed < 0.1:
            return 0.0
        
        # Gesture directed at me?
        if entity_position is not None:
            to_ai = self.ai_position - entity_position
            dist = np.linalg.norm(to_ai)
            if dist > 1e-6:
                expected = to_ai / dist
                alignment = max(0.0, np.dot(signals.gesture_direction, expected))
                gesture_dir_ev = alignment * signals.gesture_speed
            else:
                gesture_dir_ev = signals.gesture_speed * 0.5
        else:
            gesture_dir_ev = signals.gesture_speed * 0.5
        
        # Repetition = intentional
        repetition_bonus = min(0.3, signals.gesture_repetition * 0.1)
        
        return min(1.0, gesture_dir_ev + repetition_bonus)
    
    def _proximity_evidence(self, signals: BehavioralSignals,
                            entity_velocity: Optional[np.ndarray]) -> float:
        """How much does approach pattern suggest communication?"""
        if signals.distance > 2.0:
            return 0.0
        
        # Close proximity = more likely communicative
        proximity = max(0.0, 1.0 - signals.distance / 2.0)
        
        # Approaching = more likely
        approach_bonus = signals.approach_speed * 0.3
        
        return min(1.0, proximity * 0.5 + approach_bonus)
    
    def _is_directed_at_me(self, signals: BehavioralSignals,
                           gaze_ev: float,
                           entity_position: Optional[np.ndarray]) -> bool:
        """Is this behavior explicitly directed at the AI?"""
        # Strong gaze at me + close = directed
        if gaze_ev > 0.5 and signals.distance < 2.0:
            return True
        
        # Facing me + speaking = directed
        if signals.facing_toward_me and signals.speech_level > 0.3:
            return True
        
        # Gesture at me + facing me = directed
        if signals.facing_toward_me and signals.gesture_speed > 0.3:
            return True
        
        # Body angle close to facing me
        if signals.body_angle_to_me < 30 and signals.speech_level > 0.2:
            return True
        
        return False
    
    def _classify_intent(self, signals: BehavioralSignals,
                         gaze_ev: float, speech_ev: float,
                         gesture_ev: float, proximity_ev: float,
                         directed: bool) -> tuple[CommunicativeIntent, float]:
        """Classify the type of communicative intent."""
        if not directed:
            return CommunicativeIntent.NONE, 0.1
        
        # Score each intent type
        scores = {}
        
        # REQUEST: gesture + speech + close proximity
        scores[CommunicativeIntent.REQUEST] = (
            gesture_ev * 0.3 + speech_ev * 0.3 + proximity_ev * 0.2 +
            (0.2 if signals.gesture_repetition > 0 else 0.0))
        
        # GREETING: gaze + facing + brief
        scores[CommunicativeIntent.GREETING] = (
            gaze_ev * 0.4 + (0.3 if signals.facing_toward_me else 0.0) +
            (0.2 if signals.signal_duration < 3.0 else 0.0) +
            (0.1 if signals.speech_level > 0.1 else 0.0))
        
        # WARNING: high speech + gesture urgency
        scores[CommunicativeIntent.WARNING] = (
            speech_ev * 0.4 + gesture_ev * 0.3 +
            (0.3 if signals.gesture_speed > 0.7 else 0.0) +
            (0.2 if signals.speech_level > 0.6 else 0.0))
        
        # INFORMATION: gaze + facing + moderate speech
        scores[CommunicativeIntent.INFORMATION] = (
            gaze_ev * 0.3 + (0.3 if signals.facing_toward_me else 0.0) +
            speech_ev * 0.2 + proximity_ev * 0.1)
        
        # BID_FOR_ATTENTION: gesture + gaze + approach
        scores[CommunicativeIntent.BID_FOR_ATTENTION] = (
            gesture_ev * 0.4 + gaze_ev * 0.2 + proximity_ev * 0.2 +
            signals.approach_speed * 0.2)
        
        # INVITATION: approach + facing + gesture
        scores[CommunicativeIntent.INVITATION] = (
            proximity_ev * 0.3 + signals.approach_speed * 0.3 +
            (0.2 if signals.facing_toward_me else 0.0) +
            gesture_ev * 0.2)
        
        # DIRECTIVE: strong speech + gesture + facing
        scores[CommunicativeIntent.DIRECTIVE] = (
            speech_ev * 0.4 + gesture_ev * 0.3 +
            (0.2 if signals.facing_toward_me else 0.0) +
            (0.1 if signals.gesture_repetition > 2 else 0.0))
        
        # Find best
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Must exceed threshold
        if best_score < 0.2:
            return CommunicativeIntent.NONE, best_score
        
        return best_intent, min(1.0, best_score)
    
    def _apply_temporal_refinement(self, result: CommunicativeSignal):
        """Refine result based on temporal patterns.
        
        Repeated signals increase confidence. Single ambiguous signals
        are more likely noise.
        """
        if len(self._history) < 2:
            return
        
        # Check if same intent was seen recently
        recent = self._history[-5:]
        same_intent_count = sum(
            1 for h in recent if h.intent == result.intent)
        
        if same_intent_count >= 2:
            # Repetition increases confidence
            result.confidence = min(1.0,
                result.confidence + 0.1 * same_intent_count)
        
        # Consistent gaze direction = intentional
        if len(recent) >= 3:
            gaze_values = [h.gaze_evidence for h in recent]
            if all(g > 0.3 for g in gaze_values):
                result.confidence = min(1.0, result.confidence + 0.15)
    
    def get_history(self) -> list[CommunicativeSignal]:
        return list(self._history)
