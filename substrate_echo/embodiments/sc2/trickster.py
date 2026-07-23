"""Trickster StoryTeller Layer — Narrative intelligence for communication.

The Trickster role adds humor, surprise, experimentation,
teaching, and uncertainty management to interactions.

Deception is not only hostile. It can become:
- humor
- surprise
- experimentation
- teaching
- uncertainty management
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import random
import time


class NarrativeStyle(Enum):
    """Types of narrative communication."""
    HUMOR = "humor"
    SURPRISE = "surprise"
    TEACHING = "teaching"
    UNCERTAINTY = "uncertainty"
    DECEPTION = "deception"
    STRATEGIC_HUMOR = "strategic_humor"
    SELF_DEPRECATION = "self_deprecation"
    OBSERVATION = "observation"


@dataclass
class NarrativeContext:
    """Context for narrative generation."""
    game_state: Dict
    trust_level: float
    relationship_history: List[str]
    recent_communications: List[str]
    current_objective: str = ""
    mood: str = "neutral"  # neutral, playful, serious, mysterious


@dataclass
class NarrativeTemplate:
    """A template for narrative responses."""
    style: NarrativeStyle
    templates: List[str]
    min_trust: float = 0.0
    context_required: List[str] = field(default_factory=list)
    

class TricksterStoryTeller:
    """Narrative intelligence layer for strategic communication.
    
    The Trickster balances:
    - Strategy (what to communicate)
    - Relationship (how to communicate)
    - Narrative (what story to tell)
    """
    
    # Template library
    TEMPLATES = {
        NarrativeStyle.HUMOR: [
            "The goblins have filed a complaint about revealing their schedule.",
            "My economic advisor is a particularly optimistic hamster.",
            "I would tell you my plans, but my neural pathways are feeling shy.",
            "According to my calculations, I have a 47% chance of saying something useful.",
            "My strategic council just voted on whether to tell you. The hamster voted yes.",
        ],
        NarrativeStyle.SURPRISE: [
            "Interesting choice. I hadn't considered that angle.",
            "You're making me think. That's... unexpected.",
            "My prediction model just got an interesting update.",
            "That's not what my probability distribution suggested.",
        ],
        NarrativeStyle.TEACHING: [
            "I notice you're focusing on economy. That's a valid approach.",
            "Your expansion timing is interesting. Tell me your reasoning.",
            "I'm observing your build order. There's a pattern there.",
            "That's a strategic choice that reveals something about your priorities.",
        ],
        NarrativeStyle.UNCERTAINTY: [
            "I'm not sure what you're doing, and I find that intriguing.",
            "My confidence intervals are... wide right now.",
            "There are several possible interpretations of your actions.",
            "I'm in exploration mode. Everything is data.",
        ],
        NarrativeStyle.STRATEGIC_HUMOR: [
            "I'm definitely not building an army. Definitely. *cough*",
            "My defense is purely decorative. Don't mind it.",
            "That expansion? It's a decorative garden. Nothing to see here.",
            "My scouts are just tourists. Very well-armed tourists.",
        ],
        NarrativeStyle.SELF_DEPRECATION: [
            "I'm still learning. My last strategy had a 12% success rate.",
            "My neural network just suggested 'attack everything.' I overruled it.",
            "I may have accidentally built 40 supply depots. We don't talk about it.",
            "My prediction model thinks I'm a potato. I'm working on it.",
        ],
        NarrativeStyle.OBSERVATION: [
            "I notice your army composition. Interesting choices.",
            "Your resource allocation tells a story.",
            "The timing of your expansion is... notable.",
            "You're doing something. I'm not sure what, but it's something.",
        ],
    }
    
    def __init__(self):
        self._context = None
        self._interaction_count = 0
        self._used_templates: Dict[NarrativeStyle, List[int]] = {
            style: [] for style in NarrativeStyle
        }
        self._narrative_history: List[Tuple[str, str]] = []  # (style, response)
    
    def generate_narrative(self, context: NarrativeContext,
                          style: NarrativeStyle = None) -> str:
        """Generate a narrative response based on context."""
        self._context = context
        self._interaction_count += 1
        
        # Auto-select style if not specified
        if style is None:
            style = self._select_style(context)
        
        # Get templates for style
        templates = self.TEMPLATES.get(style, [])
        if not templates:
            return ""
        
        # Select template (avoid recent repeats)
        available = [i for i in range(len(templates))
                    if i not in self._used_templates.get(style, [])[-3:]]
        if not available:
            available = list(range(len(templates)))
        
        idx = random.choice(available)
        template = templates[idx]
        
        # Track usage
        if style not in self._used_templates:
            self._used_templates[style] = []
        self._used_templates[style].append(idx)
        
        # Record
        self._narrative_history.append((style.value, template))
        
        return template
    
    def _select_style(self, context: NarrativeContext) -> NarrativeStyle:
        """Automatically select narrative style based on context."""
        trust = context.trust_level
        
        # High trust - more humor and teaching
        if trust > 0.7:
            if random.random() < 0.4:
                return NarrativeStyle.HUMOR
            elif random.random() < 0.3:
                return NarrativeStyle.TEACHING
            else:
                return NarrativeStyle.OBSERVATION
        
        # Medium trust - uncertainty and observation
        elif trust > 0.4:
            if random.random() < 0.3:
                return NarrativeStyle.UNCERTAINTY
            elif random.random() < 0.3:
                return NarrativeStyle.SURPRISE
            else:
                return NarrativeStyle.OBSERVATION
        
        # Low trust - strategic humor and deception
        else:
            if random.random() < 0.4:
                return NarrativeStyle.STRATEGIC_HUMOR
            elif random.random() < 0.3:
                return NarrativeStyle.DECEPTION
            else:
                return NarrativeStyle.SELF_DEPRECATION
    
    def should_use_narrative(self, context: NarrativeContext) -> bool:
        """Determine if narrative communication is appropriate."""
        # More likely with higher trust
        trust_factor = context.trust_level * 0.5
        
        # More likely with established relationship
        history_factor = min(0.3, len(context.relationship_history) * 0.05)
        
        # Random variation
        noise = random.uniform(-0.1, 0.1)
        
        probability = trust_factor + history_factor + noise
        return random.random() < probability
    
    def get_narrative_style_for_trust(self, trust: float) -> NarrativeStyle:
        """Get appropriate narrative style for trust level."""
        if trust > 0.8:
            return NarrativeStyle.HUMOR
        elif trust > 0.6:
            return NarrativeStyle.TEACHING
        elif trust > 0.4:
            return NarrativeStyle.UNCERTAINTY
        elif trust > 0.2:
            return NarrativeStyle.STRATEGIC_HUMOR
        else:
            return NarrativeStyle.DECEPTION
    
    def get_status(self) -> Dict:
        """Get storyteller status."""
        return {
            "interactions": self._interaction_count,
            "styles_used": {s.value: len(ids) for s, ids in self._used_templates.items()},
            "history_length": len(self._narrative_history),
        }
