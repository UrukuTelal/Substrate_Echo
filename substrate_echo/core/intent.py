"""Intent — symbolic goals that drive agent behavior.

Intents are the highest level of the control hierarchy.
They represent WHAT the agent wants to achieve, without
specifying HOW. The Planner converts intents into PSV trajectories.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


class Intent(Enum):
    """Symbolic goals an agent can pursue."""
    EXPLORE = auto()            # seek novel states
    COOPERATE = auto()          # align with other agents
    DEFEND = auto()             # maintain current state
    LEARN = auto()              # increase knowledge/predictability
    INVESTIGATE = auto()        # examine anomaly or uncertainty
    INCREASE_COHESION = auto()  # bind pillars together
    REDUCE_DISTORTION = auto()  # lower deviation from baseline
    MAINTAIN_STABILITY = auto() # stay in current basin
    SEEK_NOVELTY = auto()       # find unfamiliar states
    AVOID_HARM = auto()         # minimize Harm pillar
    RESTORE_INTEGRITY = auto()  # raise Integrity pillar
    REDUCE_FLUX = auto()        # stabilize dynamics
    TASK_COMPLETE = auto()      # achieve external objective
    INFORMATION_GAIN = auto()   # maximize expected learning (curiosity-driven)


class Situation(Enum):
    """Assessment of current state."""
    STABLE = auto()        # in attractor, nothing happening
    UNSTABLE = auto()      # near saddle/repellor
    NOVEL = auto()         # unfamiliar state
    THREATENED = auto()    # harm/distortion high
    SOCIAL = auto()        # other agents nearby
    OPPORTUNITY = auto()   # favorable conditions
    CONFUSED = auto()      # low prediction confidence


@dataclass
class IntentProposal:
    """A proposed intent with priority and confidence."""
    intent: Intent
    priority: float        # 0-1, how important this intent is
    confidence: float      # 0-1, how sure we are about the situation
    reasoning: str = ""
    target_state: Optional[np.ndarray] = None  # desired PSV if applicable
    
    @property
    def score(self) -> float:
        return self.priority * self.confidence
