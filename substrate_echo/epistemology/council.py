"""Epistemology Council — Specialized roles for belief evaluation.

The Epistemology Council provides metacognitive oversight of the
belief formation process. It evaluates hypotheses, verifies
predictions, and decides when patterns become durable rules.

Architecture:
    Hypothesis Space → Model Council → Confidence Adjustments
    Prediction Memory → Reality Council → Belief Updates
    Rule Candidates → Memory Council → Rule Consolidation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class CouncilRole(Enum):
    """Roles in the Epistemology Council."""
    MODEL = "model"           # Evaluates competing hypotheses
    REALITY = "reality"       # Evaluates prediction outcomes
    MEMORY = "memory"         # Decides when patterns become rules
    CALIBRATION = "calibration"  # Checks confidence calibration


@dataclass
class CouncilVerdict:
    """A verdict from a council role."""
    role: CouncilRole
    target_id: str
    verdict: str              # "support", "reject", "adjust", "consolidate"
    confidence_adjustment: float = 0.0
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = 0.0


class ModelCouncil:
    """Evaluates competing hypotheses.
    
    Question: "What explanations fit?"
    
    Generates competing hypotheses and evaluates their relative
    strengths based on evidence.
    """
    
    def __init__(self):
        self._verdicts: List[CouncilVerdict] = []
    
    def evaluate(self, hypothesis_space: 'HypothesisSpace') -> List[CouncilVerdict]:
        """Evaluate all active hypotheses."""
        verdicts = []
        
        active = hypothesis_space.get_active()
        if len(active) < 2:
            return verdicts
        
        # Find best and worst hypotheses
        sorted_hyps = sorted(active, key=lambda h: h.confidence, reverse=True)
        best = sorted_hyps[0]
        worst = sorted_hyps[-1]
        
        # Reward well-supported hypotheses
        for h in active:
            evidence_summary = h.get_evidence_summary()
            
            # Good evidence balance
            if evidence_summary["supporting"] > evidence_summary["contradicting"] * 2:
                verdict = CouncilVerdict(
                    role=CouncilRole.MODEL,
                    target_id=h.id,
                    verdict="support",
                    confidence_adjustment=0.05,
                    reasoning=f"Strong evidence balance: {evidence_summary['supporting']} vs {evidence_summary['contradicting']}",
                    timestamp=time.time(),
                )
                verdicts.append(verdict)
            
            # Poor evidence balance
            elif evidence_summary["contradicting"] > evidence_summary["supporting"] * 2:
                verdict = CouncilVerdict(
                    role=CouncilRole.MODEL,
                    target_id=h.id,
                    verdict="adjust",
                    confidence_adjustment=-0.1,
                    reasoning=f"Weak evidence balance: {evidence_summary['supporting']} vs {evidence_summary['contradicting']}",
                    recommendations=["Gather more evidence", "Consider alternative explanations"],
                    timestamp=time.time(),
                )
                verdicts.append(verdict)
        
        # Competition: if one hypothesis dominates, suppress others
        if best.confidence > 0.8 and len(active) > 3:
            for h in active[1:]:
                if h.confidence > 0.6:
                    verdict = CouncilVerdict(
                        role=CouncilRole.MODEL,
                        target_id=h.id,
                        verdict="adjust",
                        confidence_adjustment=-0.05,
                        reasoning=f"Dominated by {best.id} (conf={best.confidence:.2f})",
                        timestamp=time.time(),
                    )
                    verdicts.append(verdict)
        
        self._verdicts.extend(verdicts)
        return verdicts


class RealityCouncil:
    """Evaluates prediction outcomes.
    
    Question: "Was the prediction correct?"
    
    Compares predictions with actual outcomes and updates
    beliefs based on reality.
    """
    
    def __init__(self):
        self._verdicts: List[CouncilVerdict] = []
        self._accuracy_history: List[float] = []
    
    def evaluate(self, prediction_memory: 'PredictionMemory',
                 hypothesis_space: 'HypothesisSpace') -> List[CouncilVerdict]:
        """Evaluate prediction outcomes and update beliefs."""
        verdicts = []
        
        stats = prediction_memory.get_accuracy_stats()
        recent_accuracy = prediction_memory.get_recent_accuracy(50)
        
        # Track accuracy trend
        self._accuracy_history.append(recent_accuracy)
        if len(self._accuracy_history) > 100:
            self._accuracy_history.pop(0)
        
        # Check for systematic failures
        systematic_errors = prediction_memory.detect_systematic_errors()
        
        for error in systematic_errors:
            source_id = error["source"]
            hypothesis = hypothesis_space.get(source_id)
            
            if hypothesis:
                verdict = CouncilVerdict(
                    role=CouncilRole.REALITY,
                    target_id=source_id,
                    verdict="reject",
                    confidence_adjustment=-0.2,
                    reasoning=f"Systematic prediction failures: {error['failure_count']}",
                    recommendations=["Consider alternative hypothesis", "Check assumptions"],
                    timestamp=time.time(),
                )
                verdicts.append(verdict)
        
        # Calibrate confidence based on accuracy
        if len(self._accuracy_history) >= 20:
            avg_accuracy = sum(self._accuracy_history[-20:]) / 20
            
            # If accuracy is high, slightly boost confident hypotheses
            if avg_accuracy > 0.8:
                for h in hypothesis_space.get_by_confidence(0.7):
                    verdict = CouncilVerdict(
                        role=CouncilRole.REALITY,
                        target_id=h.id,
                        verdict="support",
                        confidence_adjustment=0.02,
                        reasoning=f"High prediction accuracy: {avg_accuracy:.2f}",
                        timestamp=time.time(),
                    )
                    verdicts.append(verdict)
            
            # If accuracy is low, suppress confident hypotheses
            elif avg_accuracy < 0.5:
                for h in hypothesis_space.get_by_confidence(0.6):
                    verdict = CouncilVerdict(
                        role=CouncilRole.REALITY,
                        target_id=h.id,
                        verdict="adjust",
                        confidence_adjustment=-0.05,
                        reasoning=f"Low prediction accuracy: {avg_accuracy:.2f}",
                        timestamp=time.time(),
                    )
                    verdicts.append(verdict)
        
        self._verdicts.extend(verdicts)
        return verdicts


class MemoryCouncil:
    """Decides when patterns become durable rules.
    
    Question: "Should this become a learned rule?"
    
    Evaluates rule candidates and decides which ones to promote
    to durable memory.
    """
    
    def __init__(self, min_confidence: float = 0.7, min_applications: int = 3):
        self._min_confidence = min_confidence
        self._min_applications = min_applications
        self._verdicts: List[CouncilVerdict] = []
        self._consolidated_rules: List[str] = []
    
    def evaluate(self, rule_engine: 'RuleDiscoveryEngine') -> List[CouncilVerdict]:
        """Evaluate rule candidates for consolidation."""
        verdicts = []
        
        rules = rule_engine.get_rules(min_confidence=0.0)
        
        for rule in rules:
            # Promote high-confidence rules
            if (rule.confidence >= self._min_confidence and 
                rule.application_count >= self._min_applications):
                
                verdict = CouncilVerdict(
                    role=CouncilRole.MEMORY,
                    target_id=rule.id,
                    verdict="consolidate",
                    reasoning=f"Rule meets criteria: conf={rule.confidence:.2f}, apps={rule.application_count}",
                    timestamp=time.time(),
                )
                verdicts.append(verdict)
                self._consolidated_rules.append(rule.id)
            
            # Reject low-confidence rules
            elif rule.confidence < 0.3 and rule.application_count >= 5:
                verdict = CouncilVerdict(
                    role=CouncilRole.MEMORY,
                    target_id=rule.id,
                    verdict="reject",
                    reasoning=f"Rule failed validation: conf={rule.confidence:.2f}, apps={rule.application_count}",
                    timestamp=time.time(),
                )
                verdicts.append(verdict)
        
        self._verdicts.extend(verdicts)
        return verdicts


class CalibrationCouncil:
    """Checks confidence calibration.
    
    Question: "Does confidence match reality?"
    
    A system saying 90% confidence should be correct ~90% of the time.
    """
    
    def __init__(self):
        self._calibration_data: List[Dict[str, Any]] = []
    
    def check_calibration(self, prediction_memory: 'PredictionMemory') -> Dict[str, Any]:
        """Check if confidence matches actual accuracy."""
        verified = [
            p for p in prediction_memory._predictions
            if p.status.value in ("confirmed", "failed", "partial")
        ]
        
        if not verified:
            return {"calibrated": True, "error": 0.0}
        
        # Group by confidence bins
        bins = {}
        for p in verified:
            conf_bin = round(p.confidence * 10) / 10  # Round to nearest 0.1
            if conf_bin not in bins:
                bins[conf_bin] = {"total": 0, "correct": 0}
            bins[conf_bin]["total"] += 1
            if p.status.value == "confirmed":
                bins[conf_bin]["correct"] += 1
        
        # Calculate calibration error
        total_error = 0.0
        for conf_bin, data in bins.items():
            if data["total"] > 0:
                actual_accuracy = data["correct"] / data["total"]
                total_error += abs(conf_bin - actual_accuracy) * data["total"]
        
        avg_error = total_error / len(verified) if verified else 0.0
        
        result = {
            "calibrated": avg_error < 0.1,
            "error": avg_error,
            "bins": bins,
            "n_verified": len(verified),
        }
        
        self._calibration_data.append(result)
        return result


class EpistemologyCouncil:
    """Combined epistemology council with all specialized roles."""
    
    def __init__(self):
        self.model_council = ModelCouncil()
        self.reality_council = RealityCouncil()
        self.memory_council = MemoryCouncil()
        self.calibration_council = CalibrationCouncil()
    
    def audit(self, hypothesis_space: 'HypothesisSpace',
              prediction_memory: 'PredictionMemory',
              rule_engine: 'RuleDiscoveryEngine') -> Dict[str, Any]:
        """Run full epistemology audit."""
        # Model evaluation
        model_verdicts = self.model_council.evaluate(hypothesis_space)
        
        # Reality evaluation
        reality_verdicts = self.reality_council.evaluate(prediction_memory, hypothesis_space)
        
        # Memory evaluation
        memory_verdicts = self.memory_council.evaluate(rule_engine)
        
        # Apply verdicts
        for verdict in model_verdicts + reality_verdicts:
            hypothesis = hypothesis_space.get(verdict.target_id)
            if hypothesis:
                hypothesis.confidence = max(0.0, min(1.0, 
                    hypothesis.confidence + verdict.confidence_adjustment))
        
        for verdict in memory_verdicts:
            if verdict.verdict == "consolidate":
                rule = rule_engine.get_rules()
                for r in rule:
                    if r.id == verdict.target_id:
                        r.metadata["consolidated"] = True
        
        # Calibration check
        calibration = self.calibration_council.check_calibration(prediction_memory)
        
        return {
            "model_verdicts": len(model_verdicts),
            "reality_verdicts": len(reality_verdicts),
            "memory_verdicts": len(memory_verdicts),
            "calibration": calibration,
            "total_verdicts": len(model_verdicts) + len(reality_verdicts) + len(memory_verdicts),
        }