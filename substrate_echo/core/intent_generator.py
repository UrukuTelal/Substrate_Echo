"""IntentGenerator — maps personality + situation to symbolic goals.

Different agents assess the same situation differently and produce
different intents, all from the same code. Personality = different
weights on intrinsic drives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .intent import Intent, Situation, IntentProposal
from .world_model import WorldModel


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]


@dataclass
class AgentPersonality:
    """Defines agent's intrinsic drives via drive strengths."""
    exploration_drive: float = 0.3    # tendency to seek novelty
    safety_drive: float = 0.5         # tendency to avoid harm
    social_drive: float = 0.3         # tendency to cooperate
    achievement_drive: float = 0.4    # tendency to complete tasks
    curiosity_drive: float = 0.2      # tendency to investigate uncertainty
    stability_drive: float = 0.4      # tendency to maintain equilibrium
    
    @staticmethod
    def cautious() -> 'AgentPersonality':
        return AgentPersonality(
            exploration_drive=0.1, safety_drive=0.9, social_drive=0.3,
            achievement_drive=0.3, curiosity_drive=0.1, stability_drive=0.9,
        )
    
    @staticmethod
    def explorer() -> 'AgentPersonality':
        return AgentPersonality(
            exploration_drive=0.9, safety_drive=0.2, social_drive=0.2,
            achievement_drive=0.3, curiosity_drive=0.8, stability_drive=0.1,
        )
    
    @staticmethod
    def socialite() -> 'AgentPersonality':
        return AgentPersonality(
            exploration_drive=0.3, safety_drive=0.4, social_drive=0.9,
            achievement_drive=0.2, curiosity_drive=0.3, stability_drive=0.5,
        )
    
    @staticmethod
    def achiever() -> 'AgentPersonality':
        return AgentPersonality(
            exploration_drive=0.2, safety_drive=0.3, social_drive=0.2,
            achievement_drive=0.9, curiosity_drive=0.4, stability_drive=0.5,
        )


class IntentGenerator:
    """Generates symbolic intents from personality + situation.
    
    The generator:
    1. Assesses the current situation (what's happening?)
    2. Maps situation × personality → candidate intents
    3. Selects the highest-priority intent
    """
    
    def __init__(self, personality: AgentPersonality,
                 world_model: Optional[WorldModel] = None):
        self.personality = personality
        self.world_model = world_model
    
    def generate_intent(self, state: np.ndarray,
                        situation: Optional[Situation] = None) -> IntentProposal:
        """Generate an intent based on current state and personality."""
        state = np.asarray(state, dtype=np.float64)
        
        if situation is None:
            situation = self._assess_situation(state)
        
        candidates = self._map_situation_to_intents(state, situation)
        
        # Add information gain candidate if world model has DynamicsMemory
        if self.world_model is not None and hasattr(self.world_model, 'memory'):
            dm = self.world_model.memory
            if hasattr(dm, 'novelty') and dm._fitted:
                novelty = dm.novelty(state)
                # Normalize novelty
                if dm._states and len(dm._states) > 10:
                    states_arr = np.array(dm._states)
                    sample_idx = np.random.choice(len(dm._states),
                                                   min(50, len(dm._states)),
                                                   replace=False)
                    dists = []
                    for i in sample_idx:
                        d = np.linalg.norm(states_arr - states_arr[i], axis=1)
                        dists.append(np.sort(d)[1] if len(d) > 1 else 1.0)
                    avg_spacing = np.mean(dists)
                    novelty_normalized = min(1.0, novelty / max(avg_spacing * 2, 1e-6))
                else:
                    novelty_normalized = min(1.0, novelty)
                
                # Only add if novelty is significant
                if novelty_normalized > 0.3:
                    candidates.append(IntentProposal(
                        intent=Intent.INFORMATION_GAIN,
                        priority=self.personality.curiosity_drive * novelty_normalized,
                        confidence=novelty_normalized,
                        reasoning=f"novelty={novelty:.3f}, seeking information gain",
                    ))
        
        if not candidates:
            return IntentProposal(
                intent=Intent.DEFEND,
                priority=0.5,
                confidence=0.3,
                reasoning="no candidates, defaulting to DEFEND",
            )
        
        return max(candidates, key=lambda c: c.score)
    
    def _assess_situation(self, state: np.ndarray) -> Situation:
        """Assess the current situation from pillar state."""
        harm = state[12] if len(state) > 12 else 0.0
        distortion = state[13] if len(state) > 13 else 0.0
        flux = state[14] if len(state) > 14 else 0.0
        integrity = state[5] if len(state) > 5 else 0.5
        
        # Check stability
        if self.world_model is not None:
            stability = self.world_model.get_stability(state)
            classification = stability['classification']
            confidence = self.world_model.prediction_confidence(state)
        else:
            classification = 'marginal'
            confidence = 0.5
        
        # Situation assessment
        if harm > 0.7 or distortion > 0.7:
            return Situation.THREATENED
        elif classification == 'repellor':
            return Situation.UNSTABLE
        elif classification == 'saddle':
            return Situation.UNSTABLE
        elif confidence < 0.3:
            return Situation.NOVEL
        elif classification == 'attractor' and flux < 0.3:
            return Situation.STABLE
        elif integrity > 0.7 and flux < 0.2:
            return Situation.OPPORTUNITY
        else:
            return Situation.STABLE
    
    def _map_situation_to_intents(self, state: np.ndarray,
                                   situation: Situation) -> list[IntentProposal]:
        """Map situation to candidate intents weighted by personality."""
        p = self.personality
        candidates = []
        
        if situation == Situation.THREATENED:
            candidates.append(IntentProposal(
                intent=Intent.AVOID_HARM,
                priority=p.safety_drive,
                confidence=0.9,
                reasoning="harm or distortion is high",
            ))
            candidates.append(IntentProposal(
                intent=Intent.RESTORE_INTEGRITY,
                priority=p.safety_drive * 0.7,
                confidence=0.8,
                reasoning="integrity compromised",
            ))
            candidates.append(IntentProposal(
                intent=Intent.DEFEND,
                priority=p.safety_drive * 0.5,
                confidence=0.7,
                reasoning="defensive posture",
            ))
        
        elif situation == Situation.UNSTABLE:
            candidates.append(IntentProposal(
                intent=Intent.MAINTAIN_STABILITY,
                priority=p.stability_drive,
                confidence=0.8,
                reasoning="unstable region detected",
            ))
            candidates.append(IntentProposal(
                intent=Intent.REDUCE_FLUX,
                priority=p.stability_drive * 0.6,
                confidence=0.7,
                reasoning="flux is high",
            ))
        
        elif situation == Situation.NOVEL:
            candidates.append(IntentProposal(
                intent=Intent.EXPLORE,
                priority=p.exploration_drive,
                confidence=0.6,
                reasoning="unfamiliar territory",
            ))
            candidates.append(IntentProposal(
                intent=Intent.LEARN,
                priority=p.curiosity_drive,
                confidence=0.7,
                reasoning="low prediction confidence",
            ))
            candidates.append(IntentProposal(
                intent=Intent.INVESTIGATE,
                priority=p.curiosity_drive * 0.8,
                confidence=0.6,
                reasoning="uncertainty detected",
            ))
        
        elif situation == Situation.STABLE:
            candidates.append(IntentProposal(
                intent=Intent.DEFEND,
                priority=p.stability_drive * 0.5,
                confidence=0.5,
                reasoning="stable, maintain equilibrium",
            ))
            if p.exploration_drive > 0.5:
                candidates.append(IntentProposal(
                    intent=Intent.SEEK_NOVELTY,
                    priority=p.exploration_drive * 0.3,
                    confidence=0.4,
                    reasoning="stable but want to explore",
                ))
            if p.achievement_drive > 0.5:
                candidates.append(IntentProposal(
                    intent=Intent.TASK_COMPLETE,
                    priority=p.achievement_drive * 0.5,
                    confidence=0.5,
                    reasoning="stable, pursue objectives",
                ))
        
        elif situation == Situation.OPPORTUNITY:
            candidates.append(IntentProposal(
                intent=Intent.TASK_COMPLETE,
                priority=p.achievement_drive,
                confidence=0.8,
                reasoning="favorable conditions",
            ))
            candidates.append(IntentProposal(
                intent=Intent.INCREASE_COHESION,
                priority=p.social_drive * 0.5,
                confidence=0.6,
                reasoning="good state for bonding",
            ))
        
        elif situation == Situation.SOCIAL:
            candidates.append(IntentProposal(
                intent=Intent.COOPERATE,
                priority=p.social_drive,
                confidence=0.8,
                reasoning="other agents present",
            ))
        
        return candidates
    
    def assess_situation(self, state: np.ndarray) -> Situation:
        """Public accessor for situation assessment."""
        return self._assess_situation(state)
