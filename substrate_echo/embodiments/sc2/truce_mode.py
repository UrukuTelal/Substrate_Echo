"""Truce Mode — Non-aggressive optimization landscape.

A different objective function for cooperative play:

Maximize:
+ economy growth
+ map understanding
+ defensive stability
+ prediction accuracy
+ cooperation value

Minimize:
- unnecessary engagements
- resource waste
- instability
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class TruceMode(Enum):
    """Game mode."""
    COMPETITIVE = "competitive"
    TRUCE = "truce"
    COOPERATIVE = "cooperative"
    SANDBOX = "sandbox"


@dataclass
class OptimizationObjective:
    """An objective in the optimization landscape."""
    name: str
    weight: float  # Importance [0, 1]
    current_value: float = 0.0
    target_value: float = 1.0
    is_minimized: bool = False
    
    def score(self) -> float:
        """Calculate objective score."""
        if self.is_minimized:
            return (1.0 - self.current_value) * self.weight
        else:
            return self.current_value * self.weight


@dataclass
class TruceState:
    """State of truce/cooperation."""
    mode: TruceMode = TruceMode.COMPETITIVE
    trust_level: float = 0.5
    cooperation_value: float = 0.0
    last_violation: int = 0
    violation_count: int = 0
    
    def is_truce_active(self) -> bool:
        """Check if truce is active."""
        return self.mode in (TruceMode.TRUCE, TruceMode.COOPERATIVE)
    
    def check_violation(self, action: Dict) -> bool:
        """Check if action violates truce."""
        if not self.is_truce_active():
            return False
        
        # Check for aggressive actions
        if action.get("type") in ("attack", "harass"):
            return True
        
        return False


class TruceModeOptimizer:
    """Optimization landscape for different game modes.
    
    Adjusts objective weights based on mode:
    - Competitive: victory through elimination
    - Truce: stable coexistence
    - Cooperative: shared success
    - Sandbox: exploration only
    """
    
    def __init__(self):
        self._state = TruceState()
        self._objectives: Dict[TruceMode, List[OptimizationObjective]] = {
            TruceMode.COMPETITIVE: self._competitive_objectives(),
            TruceMode.TRUCE: self._truce_objectives(),
            TruceMode.COOPERATIVE: self._cooperative_objectives(),
            TruceMode.SANDBOX: self._sandbox_objectives(),
        }
        self._step: int = 0
    
    def _competitive_objectives(self) -> List[OptimizationObjective]:
        """Objectives for competitive play."""
        return [
            OptimizationObjective("enemy_elimination", weight=0.8, is_minimized=False),
            OptimizationObjective("army_strength", weight=0.7, is_minimized=False),
            OptimizationObjective("expansion_rate", weight=0.5, is_minimized=False),
            OptimizationObjective("resource_efficiency", weight=0.4, is_minimized=False),
            OptimizationObjective("casualties", weight=0.6, is_minimized=True),
        ]
    
    def _truce_objectives(self) -> List[OptimizationObjective]:
        """Objectives for truce mode."""
        return [
            OptimizationObjective("economy_growth", weight=0.7, is_minimized=False),
            OptimizationObjective("map_understanding", weight=0.6, is_minimized=False),
            OptimizationObjective("defensive_stability", weight=0.8, is_minimized=False),
            OptimizationObjective("prediction_accuracy", weight=0.5, is_minimized=False),
            OptimizationObjective("cooperation_value", weight=0.9, is_minimized=False),
            OptimizationObjective("unnecessary_engagements", weight=0.8, is_minimized=True),
            OptimizationObjective("resource_waste", weight=0.6, is_minimized=True),
            OptimizationObjective("instability", weight=0.7, is_minimized=True),
        ]
    
    def _cooperative_objectives(self) -> List[OptimizationObjective]:
        """Objectives for cooperative play."""
        return [
            OptimizationObjective("shared_victory", weight=1.0, is_minimized=False),
            OptimizationObjective("communication_quality", weight=0.8, is_minimized=False),
            OptimizationObjective("resource_sharing", weight=0.7, is_minimized=False),
            OptimizationObjective("mutual_defense", weight=0.9, is_minimized=False),
            OptimizationObjective("trust_building", weight=0.8, is_minimized=False),
        ]
    
    def _sandbox_objectives(self) -> List[OptimizationObjective]:
        """Objectives for sandbox/exploration mode."""
        return [
            OptimizationObjective("exploration_coverage", weight=0.9, is_minimized=False),
            OptimizationObjective("pattern_discovery", weight=0.8, is_minimized=False),
            OptimizationObjective("strategy_testing", weight=0.7, is_minimized=False),
            OptimizationObjective("learning_rate", weight=0.6, is_minimized=False),
        ]
    
    def set_mode(self, mode: TruceMode):
        """Change game mode."""
        self._state.mode = mode
    
    def update_objective(self, name: str, value: float):
        """Update an objective's current value."""
        objectives = self._objectives.get(self._state.mode, [])
        for obj in objectives:
            if obj.name == name:
                obj.current_value = np.clip(value, 0.0, 1.0)
                break
    
    def calculate_score(self) -> float:
        """Calculate total score for current mode."""
        objectives = self._objectives.get(self._state.mode, [])
        if not objectives:
            return 0.0
        
        return sum(obj.score() for obj in objectives)
    
    def should_engage(self, threat_level: float) -> bool:
        """Determine if engagement is appropriate."""
        if self._state.mode == TruceMode.COMPETITIVE:
            return threat_level > 0.3
        elif self._state.mode == TruceMode.TRUCE:
            return threat_level > 0.7  # Only engage if threatened
        elif self._state.mode == TruceMode.COOPERATIVE:
            return threat_level > 0.8  # Very defensive
        else:  # SANDBOX
            return False
    
    def get_recommendation(self) -> Dict:
        """Get recommendation based on current state."""
        score = self.calculate_score()
        mode = self._state.mode
        
        if mode == TruceMode.COMPETITIVE:
            return {
                "action": "attack" if score > 0.6 else "defend",
                "reasoning": f"Competitive score: {score:.2f}",
                "priority": "high" if score > 0.8 else "medium",
            }
        elif mode == TruceMode.TRUCE:
            return {
                "action": "observe" if score > 0.5 else "defend",
                "reasoning": f"Truce score: {score:.2f}",
                "priority": "low",
            }
        elif mode == TruceMode.COOPERATIVE:
            return {
                "action": "share" if score > 0.6 else "observe",
                "reasoning": f"Cooperative score: {score:.2f}",
                "priority": "medium",
            }
        else:
            return {
                "action": "explore",
                "reasoning": f"Sandbox score: {score:.2f}",
                "priority": "low",
            }
    
    def tick(self):
        """Update optimization state."""
        self._step += 1
        # Natural decay of objectives toward baseline
        for objectives in self._objectives.values():
            for obj in objectives:
                if obj.current_value > 0.5:
                    obj.current_value *= 0.99
                elif obj.current_value < 0.5:
                    obj.current_value = 0.5 + (obj.current_value - 0.5) * 0.99
    
    def get_status(self) -> Dict:
        """Get truce mode status."""
        return {
            "mode": self._state.mode.value,
            "score": self.calculate_score(),
            "trust_level": self._state.trust_level,
            "violations": self._state.violation_count,
        }
