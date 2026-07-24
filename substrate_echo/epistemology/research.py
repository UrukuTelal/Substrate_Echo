"""Research Planning — Turning goals into experiments.

The research planner converts abstract research goals into
concrete experiment proposals that can be executed by the
PerturbationEngine.

Architecture:
    Research Goal
          |
    Experiment Design
          |
    Proposal Generation
          |
    Execution Plan
          |
    Integration Plan
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time
import uuid


class ExperimentType(Enum):
    """Types of experiments the swarm can run."""
    PERTURBATION = "perturbation"    # Change a variable, observe effect
    OBSERVATION = "observation"      # Watch without interfering
    ABLATION = "ablation"            # Remove a component
    AUGMENTATION = "augmentation"    # Add a component
    STRESS = "stress"                # Push to limits
    TRANSFER = "transfer"            # Test across domains


class ProposalStatus(Enum):
    """Status of an experiment proposal."""
    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class ExperimentDesign:
    """Design for a specific experiment."""
    experiment_type: ExperimentType
    target: str                     # What to experiment on
    variables: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    duration: int = 50              # Steps to run
    sample_size: int = 1            # Number of replications
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.experiment_type.value,
            "target": self.target,
            "variables": self.variables,
            "controls": self.controls,
            "metrics": self.metrics,
            "duration": self.duration,
        }


@dataclass
class ExperimentProposal:
    """A concrete proposal for investigating a knowledge gap."""
    proposal_id: str
    goal_id: str
    domain: str
    description: str
    
    # Design
    design: ExperimentDesign
    
    # Expected outcomes
    expected_information_gain: float = 0.0
    expected_accuracy_delta: float = 0.0
    
    # Requirements
    required_agents: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: ProposalStatus = ProposalStatus.DRAFT
    
    # Results (filled after execution)
    actual_results: Dict[str, Any] = field(default_factory=dict)
    discoveries_generated: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: float = 0.0
    executed_at: float = 0.0
    completed_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.proposal_id,
            "goal_id": self.goal_id,
            "domain": self.domain,
            "description": self.description,
            "design": self.design.to_dict(),
            "status": self.status.value,
            "expected_information_gain": round(self.expected_information_gain, 3),
        }


@dataclass
class ExecutionPlan:
    """Step-by-step plan for executing an experiment."""
    plan_id: str
    proposal_id: str
    
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    
    # Timing
    estimated_duration: int = 0
    actual_duration: int = 0
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    def add_step(self, action: str, target: str, 
                 parameters: Optional[Dict[str, Any]] = None):
        """Add a step to the plan."""
        self.steps.append({
            "step": len(self.steps) + 1,
            "action": action,
            "target": target,
            "parameters": parameters or {},
            "status": "pending",
        })
    
    def advance(self) -> bool:
        """Advance to next step."""
        if self.current_step >= len(self.steps):
            return False
        
        self.steps[self.current_step]["status"] = "completed"
        self.current_step += 1
        return True
    
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return self.current_step >= len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "proposal_id": self.proposal_id,
            "steps": len(self.steps),
            "current_step": self.current_step,
            "complete": self.is_complete(),
        }


class ExperimentPlanner:
    """Converts research goals into executable experiment proposals.
    
    The planner considers:
    - What type of experiment best addresses the gap
    - What variables to manipulate
    - What metrics to measure
    - How to integrate findings back into knowledge
    """
    
    def __init__(self):
        self._proposals: Dict[str, ExperimentProposal] = {}
        self._execution_plans: Dict[str, ExecutionPlan] = {}
        self._completed_proposals: List[ExperimentProposal] = []
    
    def create_proposal(self, goal: Any, 
                         domain_context: Optional[Dict[str, Any]] = None) -> ExperimentProposal:
        """Create an experiment proposal from a research goal.
        
        Args:
            goal: ResearchGoal to investigate
            domain_context: Optional context about the domain
        
        Returns:
            ExperimentProposal ready for approval
        """
        context = domain_context or {}
        
        # Determine experiment type based on gap type
        experiment_type = self._select_experiment_type(goal)
        
        # Design the experiment
        design = self._design_experiment(goal, experiment_type, context)
        
        # Estimate information gain
        info_gain = self._estimate_information_gain(goal, design)
        
        proposal = ExperimentProposal(
            proposal_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            domain=goal.domain,
            description=goal.description,
            design=design,
            expected_information_gain=info_gain,
            expected_accuracy_delta=goal.expected_value * 0.3,
            required_agents=self._identify_required_agents(goal, context),
            created_at=time.time(),
        )
        
        self._proposals[proposal.proposal_id] = proposal
        return proposal
    
    def _select_experiment_type(self, goal: Any) -> ExperimentType:
        """Select the best experiment type for a goal."""
        # Map goal characteristics to experiment types
        if hasattr(goal, 'gap_type'):
            gap_type = goal.gap_type if hasattr(goal.gap_type, 'value') else str(goal.gap_type)
            
            if 'prediction' in str(gap_type).lower():
                return ExperimentType.OBSERVATION
            elif 'causal' in str(gap_type).lower():
                return ExperimentType.PERTURBATION
            elif 'boundary' in str(gap_type).lower():
                return ExperimentType.TRANSFER
            elif 'calibration' in str(gap_type).lower():
                return ExperimentType.STRESS
        
        return ExperimentType.PERTURBATION
    
    def _design_experiment(self, goal: Any, experiment_type: ExperimentType,
                            context: Dict[str, Any]) -> ExperimentDesign:
        """Design the specific experiment."""
        domain = goal.domain if hasattr(goal, 'domain') else "general"
        
        if experiment_type == ExperimentType.PERTURBATION:
            return ExperimentDesign(
                experiment_type=experiment_type,
                target=domain,
                variables={"perturbation_magnitude": 0.3},
                controls={"baseline_runs": 3},
                metrics=["accuracy", "confidence", "calibration"],
                duration=50,
            )
        elif experiment_type == ExperimentType.OBSERVATION:
            return ExperimentDesign(
                experiment_type=experiment_type,
                target=domain,
                variables={"observation_count": 100},
                controls={},
                metrics=["pattern_detection", "accuracy"],
                duration=100,
            )
        elif experiment_type == ExperimentType.TRANSFER:
            return ExperimentDesign(
                experiment_type=experiment_type,
                target=domain,
                variables={"source_domain": context.get("source_domain", "general")},
                controls={"target_domain": domain},
                metrics=["transfer_accuracy", "generalization"],
                duration=75,
            )
        else:
            return ExperimentDesign(
                experiment_type=experiment_type,
                target=domain,
                variables={},
                controls={},
                metrics=["accuracy"],
                duration=50,
            )
    
    def _estimate_information_gain(self, goal: Any, design: ExperimentDesign) -> float:
        """Estimate how much information the experiment will provide."""
        base_gain = goal.expected_value if hasattr(goal, 'expected_value') else 0.5
        
        # Adjust based on experiment type
        type_multiplier = {
            ExperimentType.PERTURBATION: 1.0,
            ExperimentType.OBSERVATION: 0.7,
            ExperimentType.ABLATION: 0.9,
            ExperimentType.AUGMENTATION: 0.8,
            ExperimentType.STRESS: 0.6,
            ExperimentType.TRANSFER: 0.8,
        }
        
        return base_gain * type_multiplier.get(design.experiment_type, 0.5)
    
    def _identify_required_agents(self, goal: Any, 
                                   context: Dict[str, Any]) -> List[str]:
        """Identify which agents should participate."""
        agents = []
        
        # Get domain experts from context
        if "domain_experts" in context:
            agents.extend(context["domain_experts"])
        
        # Ensure at least one agent
        if not agents:
            agents.append("default_agent")
        
        return agents
    
    def create_execution_plan(self, proposal: ExperimentProposal) -> ExecutionPlan:
        """Create a step-by-step execution plan for a proposal."""
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            proposal_id=proposal.proposal_id,
            estimated_duration=proposal.design.duration,
        )
        
        # Add steps based on experiment type
        design = proposal.design
        
        if design.experiment_type == ExperimentType.PERTURBATION:
            plan.add_step("establish_baseline", design.target, 
                         {"metrics": design.metrics, "duration": 10})
            plan.add_step("apply_perturbation", design.target,
                         {"variables": design.variables, "duration": design.duration // 2})
            plan.add_step("measure_response", design.target,
                         {"metrics": design.metrics})
            plan.add_step("compare_with_baseline", design.target, {})
            plan.add_step("record_discovery", design.target, {})
        
        elif design.experiment_type == ExperimentType.OBSERVATION:
            plan.add_step("start_observation", design.target,
                         {"variables": design.variables})
            plan.add_step("collect_data", design.target,
                         {"duration": design.duration})
            plan.add_step("analyze_patterns", design.target, {})
            plan.add_step("record_discovery", design.target, {})
        
        else:
            plan.add_step("prepare", design.target, {})
            plan.add_step("execute", design.target, 
                         {"duration": design.duration})
            plan.add_step("analyze", design.target, {})
            plan.add_step("record_discovery", design.target, {})
        
        self._execution_plans[plan.plan_id] = plan
        return plan
    
    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal for execution."""
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        if proposal.status != ProposalStatus.DRAFT:
            return False
        
        proposal.status = ProposalStatus.APPROVED
        return True
    
    def start_execution(self, proposal_id: str) -> bool:
        """Mark a proposal as executing."""
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        if proposal.status != ProposalStatus.APPROVED:
            return False
        
        proposal.status = ProposalStatus.EXECUTING
        proposal.executed_at = time.time()
        return True
    
    def complete_proposal(self, proposal_id: str, 
                           results: Dict[str, Any],
                           discoveries: Optional[List[str]] = None) -> bool:
        """Mark a proposal as completed with results."""
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        proposal.status = ProposalStatus.COMPLETED
        proposal.actual_results = results
        proposal.discoveries_generated = discoveries or []
        proposal.completed_at = time.time()
        
        self._completed_proposals.append(proposal)
        del self._proposals[proposal_id]
        
        return True
    
    def get_proposal(self, proposal_id: str) -> Optional[ExperimentProposal]:
        """Get a specific proposal."""
        return self._proposals.get(proposal_id)
    
    def get_pending_proposals(self) -> List[ExperimentProposal]:
        """Get proposals awaiting approval."""
        return [p for p in self._proposals.values() 
                if p.status == ProposalStatus.DRAFT]
    
    def get_approved_proposals(self) -> List[ExperimentProposal]:
        """Get approved proposals ready for execution."""
        return [p for p in self._proposals.values()
                if p.status == ProposalStatus.APPROVED]
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research planning activity."""
        return {
            "total_proposals": len(self._proposals) + len(self._completed_proposals),
            "pending": len(self.get_pending_proposals()),
            "approved": len(self.get_approved_proposals()),
            "completed": len(self._completed_proposals),
            "execution_plans": len(self._execution_plans),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposals": len(self._proposals),
            "completed": len(self._completed_proposals),
            "execution_plans": len(self._execution_plans),
            "summary": self.get_research_summary(),
        }


class ResearchAgenda:
    """Manages the swarm's overall research agenda.
    
    Coordinates between EpistemicCuriosityEngine (what to learn)
    and ExperimentPlanner (how to learn it).
    
    The agenda ensures:
    - Research is aligned with knowledge gaps
    - Experiments are well-designed
    - Results are integrated back into knowledge
    """
    
    def __init__(self):
        self.curiosityEngine: Optional[Any] = None
        self.experimentPlanner = ExperimentPlanner()
        
        self._active_research: Dict[str, Dict[str, Any]] = {}
        self._completed_research: List[Dict[str, Any]] = []
    
    def set_curiosityEngine(self, engine: Any):
        """Connect to the epistemic curiosity engine."""
        self.curiosityEngine = engine
    
    def generate_research_cycle(self) -> List[ExperimentProposal]:
        """Run a complete research cycle:
        1. Identify gaps (from curiosity engine)
        2. Generate goals
        3. Create proposals
        4. Return for approval
        """
        if self.curiosityEngine is None:
            return []
        
        # Generate research goals
        goals = self.curiosityEngine.generate_research_goals(max_goals=3)
        
        # Create proposals for each goal
        proposals = []
        for goal in goals:
            proposal = self.experimentPlanner.create_proposal(goal)
            proposals.append(proposal)
        
        return proposals
    
    def integrate_results(self, proposal: ExperimentProposal,
                           discoveries: List[Any]) -> bool:
        """Integrate experiment results back into knowledge.
        
        This closes the loop:
        Gap → Goal → Experiment → Discovery → Knowledge Update
        """
        if self.curiosityEngine is None:
            return False
        
        # Mark goal as completed
        goal_id = proposal.goal_id
        if goal_id in self.curiosityEngine._research_goals:
            self.curiosityEngine.complete_goal(
                goal_id,
                results=[d.description for d in discoveries] if discoveries else []
            )
        
        # Record in completed research
        self._completed_research.append({
            "proposal": proposal.to_dict(),
            "discoveries": [d.to_dict() for d in discoveries] if discoveries else [],
            "timestamp": time.time(),
        })
        
        return True
    
    def get_agenda_summary(self) -> Dict[str, Any]:
        """Get summary of the research agenda."""
        return {
            "active_research": len(self._active_research),
            "completed_research": len(self._completed_research),
            "planner": self.experimentPlanner.get_research_summary(),
            "curiosityEngine": (
                self.curiosityEngine.get_research_summary()
                if self.curiosityEngine else None
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.get_agenda_summary(),
            "completed_research": len(self._completed_research),
        }
