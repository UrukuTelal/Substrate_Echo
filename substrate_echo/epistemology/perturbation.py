"""Perturbation Experiments — Causal discovery through intervention.

Instead of only "Does the system work?", perturbation experiments
ask "What changes when one variable changes?" This creates causal
discovery rather than just correlation detection.

Architecture:
    Baseline State
         |
    Perturbation Applied
         |
    System Response
         |
    Comparison with Baseline
         |
    Causal Inference
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time


class PerturbationType(Enum):
    """Types of perturbations."""
    PARAMETER = "parameter"       # Change a system parameter
    INPUT = "input"               # Change input data
    TIMING = "timing"             # Change timing of events
    STRUCTURE = "structure"       # Change system structure
    NOISE = "noise"               # Add noise
    ABSENCE = "absence"           # Remove something


@dataclass
class Perturbation:
    """A perturbation to apply to the system."""
    name: str
    perturbation_type: PerturbationType
    target: str                   # What to perturb
    magnitude: float = 0.5       # How much to change [0, 1]
    duration: int = 10           # How many steps to maintain
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineState:
    """State of the system before perturbation."""
    step: int
    timestamp: float
    
    # System state
    metrics: Dict[str, float] = field(default_factory=dict)
    hypotheses: Dict[str, float] = field(default_factory=dict)
    predictions_accuracy: float = 0.0
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "metrics": self.metrics,
            "hypotheses": len(self.hypotheses),
            "prediction_accuracy": self.predictions_accuracy,
            "confidence": self.confidence,
        }


@dataclass
class SystemResponse:
    """System response after perturbation."""
    step: int
    timestamp: float
    
    # Observed changes
    metrics: Dict[str, float] = field(default_factory=dict)
    recovery_time: Optional[int] = None  # Steps to recover
    collapsed: bool = False
    
    # Comparison with baseline
    metric_changes: Dict[str, float] = field(default_factory=dict)
    confidence_change: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "metric_changes": self.metric_changes,
            "confidence_change": self.confidence_change,
            "recovery_time": self.recovery_time,
            "collapsed": self.collapsed,
        }


@dataclass
class CausalInference:
    """Inferred causal relationship from perturbation."""
    cause: str
    effect: str
    strength: float               # How strong is the causal link [0, 1]
    direction: str                # "positive" or "negative"
    confidence: float = 0.5
    evidence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "direction": self.direction,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }


class PerturbationEngine:
    """Applies perturbations and measures system response.
    
    Creates causal discovery by systematically varying
    one variable at a time.
    """
    
    def __init__(self):
        self._perturbations: List[Perturbation] = []
        self._experiments: List[Dict[str, Any]] = []
        self._causal_inferences: List[CausalInference] = []
        self._active_perturbation: Optional[Perturbation] = None
        self._perturbation_start: int = 0
    
    def apply(self, perturbation: Perturbation, current_step: int) -> bool:
        """Apply a perturbation."""
        if self._active_perturbation is not None:
            return False  # Already have active perturbation
        
        self._active_perturbation = perturbation
        self._perturbation_start = current_step
        self._perturbations.append(perturbation)
        
        return True
    
    def is_active(self) -> bool:
        """Check if a perturbation is active."""
        return self._active_perturbation is not None
    
    def get_active(self) -> Optional[Perturbation]:
        """Get the active perturbation."""
        return self._active_perturbation
    
    def check_expiration(self, current_step: int) -> bool:
        """Check if perturbation should end."""
        if self._active_perturbation is None:
            return False
        
        elapsed = current_step - self._perturbation_start
        if elapsed >= self._active_perturbation.duration:
            self._active_perturbation = None
            return True
        
        return False
    
    def measure_response(self, baseline: BaselineState,
                         current_metrics: Dict[str, float],
                         current_step: int) -> SystemResponse:
        """Measure system response to perturbation."""
        metric_changes = {}
        for key, current_val in current_metrics.items():
            if key in baseline.metrics:
                baseline_val = baseline.metrics[key]
                if abs(baseline_val) > 1e-6:
                    metric_changes[key] = (current_val - baseline_val) / abs(baseline_val)
                else:
                    metric_changes[key] = current_val - baseline_val
        
        # Check for collapse
        collapsed = any(abs(v) > 0.5 for v in metric_changes.values())
        
        # Estimate recovery time
        recovery_time = None
        if collapsed:
            recovery_time = current_step - baseline.step
        
        return SystemResponse(
            step=current_step,
            timestamp=time.time(),
            metrics=current_metrics,
            metric_changes=metric_changes,
            recovery_time=recovery_time,
            collapsed=collapsed,
        )
    
    def record_experiment(self, perturbation: Perturbation,
                         baseline: BaselineState,
                         response: SystemResponse):
        """Record a complete experiment."""
        self._experiments.append({
            "perturbation": perturbation.__dict__,
            "baseline": baseline.to_dict(),
            "response": response.to_dict(),
            "timestamp": time.time(),
        })
        
        # Update causal inferences
        self._update_causal_inferences(perturbation, baseline, response)
    
    def _update_causal_inferences(self, perturbation: Perturbation,
                                   baseline: BaselineState,
                                   response: SystemResponse):
        """Update causal inferences based on experiment."""
        # Look for existing inference
        existing = None
        for inf in self._causal_inferences:
            if inf.cause == perturbation.target:
                for effect_key in response.metric_changes:
                    if inf.effect == effect_key:
                        existing = inf
                        break
        
        if existing:
            # Update existing inference
            existing.evidence_count += 1
            
            # Update strength based on response magnitude
            avg_change = np.mean([abs(v) for v in response.metric_changes.values()])
            existing.strength = existing.strength * 0.7 + avg_change * 0.3
            
            # Update confidence with more evidence
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            # Create new inference
            for effect_key, change in response.metric_changes.items():
                if abs(change) > 0.1:  # Significant change
                    inference = CausalInference(
                        cause=perturbation.target,
                        effect=effect_key,
                        strength=abs(change),
                        direction="positive" if change > 0 else "negative",
                        confidence=0.3,
                        evidence_count=1,
                    )
                    self._causal_inferences.append(inference)
    
    def get_causal_inferences(self, min_confidence: float = 0.5) -> List[CausalInference]:
        """Get causal inferences above confidence threshold."""
        return [
            i for i in self._causal_inferences
            if i.confidence >= min_confidence
        ]
    
    def generate_perturbations(self) -> List[Perturbation]:
        """Generate perturbation suggestions based on current knowledge."""
        suggestions = []
        
        # Suggest perturbations for weakly understood relationships
        for inference in self._causal_inferences:
            if inference.confidence < 0.7:
                suggestions.append(Perturbation(
                    name=f"Test {inference.cause} -> {inference.effect}",
                    perturbation_type=PerturbationType.PARAMETER,
                    target=inference.cause,
                    magnitude=0.3,
                    duration=20,
                ))
        
        return suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_experiments": len(self._experiments),
            "causal_inferences": [i.to_dict() for i in self._causal_inferences],
            "perturbations_applied": len(self._perturbations),
        }