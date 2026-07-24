"""Epistemic Curiosity — Active knowledge acquisition.

The swarm doesn't only answer "What do we know?"
It asks "What knowledge would most improve future decisions?"

Architecture:
    Uncertainty Map
          |
    Impact Assessment
          |
    Research Goal Generation
          |
    Experiment Proposal
          |
    Discovery + Lineage
          |
    Cultural Update

This is the computational equivalent of science:
the swarm generates its own research agenda.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import numpy as np
import time
import uuid


class GapType(Enum):
    """Types of knowledge gaps."""
    PREDICTION = "prediction"       # Low prediction accuracy in a domain
    CAUSAL = "causal"               # Unknown causal relationships
    BOUNDARY = "boundary"           # Unknown domain boundaries
    CALIBRATION = "calibration"     # Confidence doesn't match accuracy
    NOVEL = "novel"                 # Completely unexplored territory


class GapPriority(Enum):
    """Priority levels for knowledge gaps."""
    CRITICAL = "critical"    # Immediately impacts decisions
    HIGH = "high"            # Significant improvement potential
    MEDIUM = "medium"        # Moderate improvement potential
    LOW = "low"              # Nice to know, not urgent


@dataclass
class KnowledgeGap:
    """A specific gap in the swarm's knowledge."""
    gap_id: str
    gap_type: GapType
    domain: str
    description: str
    
    # Impact assessment
    current_confidence: float = 0.5    # How confident are we now
    expected_improvement: float = 0.0  # Expected improvement from filling gap
    impact_score: float = 0.0          # Impact on decisions [0, 1]
    
    # Feasibility
    estimated_experiments: int = 1     # How many experiments to fill
    difficulty: float = 0.5           # How hard to investigate [0, 1]
    
    # Priority
    priority: GapPriority = GapPriority.MEDIUM
    
    # Related knowledge
    related_discoveries: List[str] = field(default_factory=list)
    related_agents: List[str] = field(default_factory=list)
    
    # Metadata
    identified_at: float = 0.0
    last_assessed: float = 0.0
    
    def get_expected_value(self) -> float:
        """Calculate expected value of filling this gap."""
        if self.difficulty <= 0:
            return self.impact_score
        return self.impact_score / max(0.1, self.difficulty)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.gap_id,
            "type": self.gap_type.value,
            "domain": self.domain,
            "description": self.description,
            "current_confidence": round(self.current_confidence, 3),
            "impact_score": round(self.impact_score, 3),
            "expected_improvement": round(self.expected_improvement, 3),
            "priority": self.priority.value,
            "expected_value": round(self.get_expected_value(), 3),
        }


@dataclass
class ResearchGoal:
    """A prioritized goal for knowledge acquisition."""
    goal_id: str
    gap_id: str
    domain: str
    description: str
    
    # Scoring
    priority_score: float = 0.0    # Overall priority [0, 1]
    feasibility: float = 0.5       # How feasible [0, 1]
    expected_value: float = 0.0    # Expected information gain
    
    # Plan
    experiment_count: int = 1
    estimated_steps: int = 100
    
    # Status
    status: str = "pending"        # pending, active, completed, abandoned
    results: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.goal_id,
            "domain": self.domain,
            "description": self.description,
            "priority_score": round(self.priority_score, 3),
            "feasibility": round(self.feasibility, 3),
            "expected_value": round(self.expected_value, 3),
            "status": self.status,
        }


@dataclass
class ImpactAssessment:
    """Assessment of how much a knowledge gap impacts decisions."""
    gap_id: str
    domain: str
    
    # Decision impact
    decisions_affected: int = 0          # How many decisions are affected
    decision_confidence_delta: float = 0  # How much confidence would change
    
    # Performance impact
    accuracy_potential: float = 0.0       # Potential accuracy improvement
    efficiency_potential: float = 0.0     # Potential efficiency improvement
    
    # Risk assessment
    risk_of_ignoring: float = 0.0        # What happens if we don't learn this
    cost_of_investigation: float = 0.5   # What it costs to learn
    
    def get_net_impact(self) -> float:
        """Calculate net impact of filling this gap."""
        benefit = self.accuracy_potential + self.efficiency_potential
        cost = self.cost_of_investigation
        return benefit - cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "domain": self.domain,
            "decisions_affected": self.decisions_affected,
            "accuracy_potential": round(self.accuracy_potential, 3),
            "efficiency_potential": round(self.efficiency_potential, 3),
            "risk_of_ignoring": round(self.risk_of_ignoring, 3),
            "net_impact": round(self.get_net_impact(), 3),
        }


class UncertaintyMap:
    """Maps the swarm's knowledge gaps across domains.
    
    Scans multiple knowledge sources to identify where the swarm
    is uncertain, what's missing, and where improvement would
    have the highest impact.
    """
    
    def __init__(self):
        self._gaps: Dict[str, KnowledgeGap] = {}
        self._domain_gaps: Dict[str, List[str]] = {}  # domain -> gap_ids
        self._assessment_cache: Dict[str, ImpactAssessment] = {}
    
    def scan_prediction_gaps(self, domain_accuracies: Dict[str, float],
                              threshold: float = 0.7) -> List[KnowledgeGap]:
        """Identify domains where prediction accuracy is low."""
        gaps = []
        
        for domain, accuracy in domain_accuracies.items():
            if accuracy < threshold:
                gap = KnowledgeGap(
                    gap_id=str(uuid.uuid4()),
                    gap_type=GapType.PREDICTION,
                    domain=domain,
                    description=f"Low prediction accuracy in {domain}: {accuracy:.2f}",
                    current_confidence=accuracy,
                    expected_improvement=threshold - accuracy,
                    impact_score=1.0 - accuracy,
                    priority=self._classify_priority(1.0 - accuracy),
                    identified_at=time.time(),
                )
                self._register_gap(gap)
                gaps.append(gap)
        
        return gaps
    
    def scan_calibration_gaps(self, confidence_accuracy_pairs: List[Tuple[float, float]],
                              domain: str = "general") -> List[KnowledgeGap]:
        """Identify where confidence doesn't match accuracy."""
        gaps = []
        
        if not confidence_accuracy_pairs:
            return gaps
        
        confidences = [c for c, _ in confidence_accuracy_pairs]
        accuracies = [a for _, a in confidence_accuracy_pairs]
        
        avg_confidence = np.mean(confidences)
        avg_accuracy = np.mean(accuracies)
        
        miscalibration = abs(avg_confidence - avg_accuracy)
        
        if miscalibration > 0.15:
            gap = KnowledgeGap(
                gap_id=str(uuid.uuid4()),
                gap_type=GapType.CALIBRATION,
                domain=domain,
                description=f"Miscalibration in {domain}: confidence={avg_confidence:.2f} vs accuracy={avg_accuracy:.2f}",
                current_confidence=1.0 - miscalibration,
                expected_improvement=miscalibration,
                impact_score=miscalibration * 0.8,
                priority=self._classify_priority(miscalibration * 0.8),
                identified_at=time.time(),
            )
            self._register_gap(gap)
            gaps.append(gap)
        
        return gaps
    
    def scan_open_questions(self, open_questions: List[Any],
                             domain: str = "general") -> List[KnowledgeGap]:
        """Convert open questions into knowledge gaps."""
        gaps = []
        
        for question in open_questions:
            if hasattr(question, 'importance') and question.importance > 0.5:
                gap = KnowledgeGap(
                    gap_id=str(uuid.uuid4()),
                    gap_type=GapType.CAUSAL,
                    domain=domain,
                    description=question.description,
                    current_confidence=getattr(question, 'best_confidence', 0.0),
                    impact_score=question.importance,
                    priority=self._classify_priority(question.importance),
                    identified_at=time.time(),
                )
                self._register_gap(gap)
                gaps.append(gap)
        
        return gaps
    
    def scan_failed_assumptions(self, failed_assumptions: List[Dict[str, Any]],
                                 domain: str = "general") -> List[KnowledgeGap]:
        """Convert failed assumptions into knowledge gaps."""
        gaps = []
        
        for assumption in failed_assumptions:
            gap = KnowledgeGap(
                gap_id=str(uuid.uuid4()),
                gap_type=GapType.BOUNDARY,
                domain=domain,
                description=f"Failed assumption: {assumption.get('assumption', 'unknown')}",
                current_confidence=0.0,
                impact_score=0.7,
                priority=GapPriority.HIGH,
                identified_at=time.time(),
            )
            self._register_gap(gap)
            gaps.append(gap)
        
        return gaps
    
    def _register_gap(self, gap: KnowledgeGap):
        """Register a gap in the map."""
        self._gaps[gap.gap_id] = gap
        
        if gap.domain not in self._domain_gaps:
            self._domain_gaps[gap.domain] = []
        self._domain_gaps[gap.domain].append(gap.gap_id)
    
    def _classify_priority(self, impact: float) -> GapPriority:
        """Classify priority based on impact score."""
        if impact >= 0.8:
            return GapPriority.CRITICAL
        elif impact >= 0.6:
            return GapPriority.HIGH
        elif impact >= 0.3:
            return GapPriority.MEDIUM
        else:
            return GapPriority.LOW
    
    def get_gap(self, gap_id: str) -> Optional[KnowledgeGap]:
        """Get a specific gap."""
        return self._gaps.get(gap_id)
    
    def get_gaps_by_domain(self, domain: str) -> List[KnowledgeGap]:
        """Get all gaps for a domain."""
        gap_ids = self._domain_gaps.get(domain, [])
        return [self._gaps[gid] for gid in gap_ids if gid in self._gaps]
    
    def get_gaps_by_priority(self, priority: GapPriority) -> List[KnowledgeGap]:
        """Get all gaps with a specific priority."""
        return [g for g in self._gaps.values() if g.priority == priority]
    
    def get_all_gaps(self) -> List[KnowledgeGap]:
        """Get all gaps sorted by expected value."""
        return sorted(self._gaps.values(), 
                      key=lambda g: g.get_expected_value(), 
                      reverse=True)
    
    def get_top_gaps(self, n: int = 5) -> List[KnowledgeGap]:
        """Get top N gaps by expected value."""
        return self.get_all_gaps()[:n]
    
    def get_domain_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Get coverage summary by domain."""
        coverage = {}
        
        for domain, gap_ids in self._domain_gaps.items():
            gaps = [self._gaps[gid] for gid in gap_ids if gid in self._gaps]
            coverage[domain] = {
                "gap_count": len(gaps),
                "avg_confidence": np.mean([g.current_confidence for g in gaps]) if gaps else 0,
                "avg_impact": np.mean([g.impact_score for g in gaps]) if gaps else 0,
                "critical_gaps": sum(1 for g in gaps if g.priority == GapPriority.CRITICAL),
            }
        
        return coverage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gaps": len(self._gaps),
            "domains": len(self._domain_gaps),
            "by_priority": {
                p.value: len([g for g in self._gaps.values() if g.priority == p])
                for p in GapPriority
            },
            "by_type": {
                t.value: len([g for g in self._gaps.values() if g.gap_type == t])
                for t in GapType
            },
        }


class ImpactAssessor:
    """Assesses the impact of knowledge gaps on decision quality.
    
    Considers:
    - How many decisions are affected
    - How much accuracy would improve
    - What's the risk of not knowing
    - What's the cost of learning
    """
    
    def __init__(self):
        self._assessments: Dict[str, ImpactAssessment] = {}
        self._decision_history: List[Dict[str, Any]] = []
    
    def record_decision(self, domain: str, decision: str,
                        confidence: float, outcome: Optional[bool] = None):
        """Record a decision for impact assessment."""
        self._decision_history.append({
            "domain": domain,
            "decision": decision,
            "confidence": confidence,
            "outcome": outcome,
            "timestamp": time.time(),
        })
    
    def assess_gap(self, gap: KnowledgeGap,
                    domain_decisions: Optional[List[Dict[str, Any]]] = None,
                    domain_accuracy: float = 0.5) -> ImpactAssessment:
        """Assess the impact of filling a knowledge gap."""
        # Count decisions in this domain
        decisions = domain_decisions or [
            d for d in self._decision_history if d["domain"] == gap.domain
        ]
        decisions_affected = len(decisions)
        
        # Estimate accuracy potential
        accuracy_potential = gap.expected_improvement * (decisions_affected / max(1, len(self._decision_history)))
        
        # Estimate efficiency potential
        efficiency_potential = gap.impact_score * 0.3  # Heuristic
        
        # Risk of ignoring
        risk_of_ignoring = gap.impact_score * (1.0 - domain_accuracy)
        
        # Cost of investigation
        cost_of_investigation = gap.difficulty * gap.estimated_experiments * 0.1
        
        assessment = ImpactAssessment(
            gap_id=gap.gap_id,
            domain=gap.domain,
            decisions_affected=decisions_affected,
            decision_confidence_delta=gap.expected_improvement,
            accuracy_potential=accuracy_potential,
            efficiency_potential=efficiency_potential,
            risk_of_ignoring=risk_of_ignoring,
            cost_of_investigation=cost_of_investigation,
        )
        
        self._assessments[gap.gap_id] = assessment
        return assessment
    
    def get_assessment(self, gap_id: str) -> Optional[ImpactAssessment]:
        """Get assessment for a gap."""
        return self._assessments.get(gap_id)
    
    def rank_by_impact(self, gaps: List[KnowledgeGap]) -> List[Tuple[KnowledgeGap, ImpactAssessment]]:
        """Rank gaps by their impact assessment."""
        ranked = []
        
        for gap in gaps:
            assessment = self._assessments.get(gap.gap_id)
            if assessment is None:
                assessment = self.assess_gap(gap)
            ranked.append((gap, assessment))
        
        # Sort by net impact
        ranked.sort(key=lambda x: x[1].get_net_impact(), reverse=True)
        
        return ranked
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_assessments": len(self._assessments),
            "total_decisions": len(self._decision_history),
            "avg_net_impact": (
                np.mean([a.get_net_impact() for a in self._assessments.values()])
                if self._assessments else 0
            ),
        }


class EpistemicCuriosityEngine:
    """The swarm's curiosity engine — generates its own research agenda.
    
    This is the connective tissue that turns knowledge gaps into
    research goals. It scans uncertainty, assesses impact, and
    prioritizes what the swarm should learn next.
    
    Architecture:
        UncertaintyMap (what don't we know?)
              |
        ImpactAssessor (what would knowing improve?)
              |
        ResearchGoalGenerator (what should we investigate?)
              |
        ExperimentPlanner (how should we investigate?)
    
    The engine doesn't run experiments itself — it generates
    proposals that can be executed by PerturbationEngine.
    """
    
    def __init__(self):
        self.uncertainty_map = UncertaintyMap()
        self.impact_assessor = ImpactAssessor()
        
        self._research_goals: Dict[str, ResearchGoal] = {}
        self._completed_goals: List[ResearchGoal] = []
        
        # Configuration
        self._min_impact_threshold: float = 0.3
        self._max_active_goals: int = 5
        self._goal_decay_rate: float = 0.01
    
    def scan_swarm_knowledge(self, swarm_record: Any,
                              prediction_accuracies: Optional[Dict[str, float]] = None,
                              confidence_accuracy_pairs: Optional[List[Tuple[float, float]]] = None,
                              domain: str = "general") -> List[KnowledgeGap]:
        """Comprehensive scan of swarm knowledge for gaps.
        
        Args:
            swarm_record: SwarmDevelopmentRecord
            prediction_accuracies: Domain -> accuracy mapping
            confidence_accuracy_pairs: (confidence, accuracy) pairs for calibration
            domain: Default domain for uncategorized gaps
        
        Returns:
            List of identified knowledge gaps
        """
        all_gaps = []
        
        # 1. Scan prediction gaps
        if prediction_accuracies:
            gaps = self.uncertainty_map.scan_prediction_gaps(prediction_accuracies)
            all_gaps.extend(gaps)
        
        # 2. Scan calibration gaps
        if confidence_accuracy_pairs:
            gaps = self.uncertainty_map.scan_calibration_gaps(
                confidence_accuracy_pairs, domain
            )
            all_gaps.extend(gaps)
        
        # 3. Scan open questions
        if hasattr(swarm_record, 'get_open_questions'):
            open_questions = swarm_record.get_open_questions()
            gaps = self.uncertainty_map.scan_open_questions(open_questions, domain)
            all_gaps.extend(gaps)
        
        # 4. Scan failed assumptions
        if hasattr(swarm_record, 'get_failed_assumptions'):
            failed = swarm_record.get_failed_assumptions()
            gaps = self.uncertainty_map.scan_failed_assumptions(failed, domain)
            all_gaps.extend(gaps)
        
        # 5. Assess impact for all gaps
        for gap in all_gaps:
            self.impact_assessor.assess_gap(gap)
        
        return all_gaps
    
    def generate_research_goals(self, max_goals: int = 5) -> List[ResearchGoal]:
        """Generate prioritized research goals from identified gaps.
        
        Returns goals sorted by expected value (impact / difficulty).
        """
        gaps = self.uncertainty_map.get_all_gaps()
        
        # Filter by minimum impact
        gaps = [g for g in gaps if g.impact_score >= self._min_impact_threshold]
        
        # Rank by expected value
        gaps.sort(key=lambda g: g.get_expected_value(), reverse=True)
        
        goals = []
        for gap in gaps[:max_goals]:
            goal = ResearchGoal(
                goal_id=str(uuid.uuid4()),
                gap_id=gap.gap_id,
                domain=gap.domain,
                description=f"Investigate: {gap.description}",
                priority_score=gap.impact_score,
                feasibility=1.0 - gap.difficulty,
                expected_value=gap.get_expected_value(),
                experiment_count=gap.estimated_experiments,
                estimated_steps=gap.estimated_experiments * 50,
                created_at=time.time(),
            )
            
            self._research_goals[goal.goal_id] = goal
            goals.append(goal)
        
        return goals
    
    def get_active_goals(self) -> List[ResearchGoal]:
        """Get currently active research goals."""
        return [g for g in self._research_goals.values() if g.status == "active"]
    
    def get_pending_goals(self) -> List[ResearchGoal]:
        """Get pending research goals sorted by priority."""
        pending = [g for g in self._research_goals.values() if g.status == "pending"]
        return sorted(pending, key=lambda g: -g.priority_score)
    
    def start_goal(self, goal_id: str) -> bool:
        """Activate a research goal."""
        if goal_id not in self._research_goals:
            return False
        
        goal = self._research_goals[goal_id]
        if goal.status != "pending":
            return False
        
        # Check max active
        if len(self.get_active_goals()) >= self._max_active_goals:
            return False
        
        goal.status = "active"
        return True
    
    def complete_goal(self, goal_id: str, results: List[str]) -> bool:
        """Mark a research goal as completed."""
        if goal_id not in self._research_goals:
            return False
        
        goal = self._research_goals[goal_id]
        goal.status = "completed"
        goal.results = results
        
        self._completed_goals.append(goal)
        del self._research_goals[goal_id]
        
        return True
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research activity."""
        return {
            "active_goals": len(self.get_active_goals()),
            "pending_goals": len(self.get_pending_goals()),
            "completed_goals": len(self._completed_goals),
            "gaps_identified": len(self.uncertainty_map._gaps),
            "assessments_made": len(self.impact_assessor._assessments),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainty_map": self.uncertainty_map.to_dict(),
            "impact_assessor": self.impact_assessor.to_dict(),
            "research_goals": self.get_research_summary(),
        }
