"""Resource Allocator - Arbitrates council proposals against budget."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import time

from substrate_echo.blackboard import Blackboard, BlackboardChannel, CouncilProposal
from substrate_echo.blackboard.proposal import Proposal


class AllocationStrategy(Enum):
    PRIORITY = "priority"
    UTILITY = "utility"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BALANCED = "balanced"


@dataclass
class AllocationResult:
    accepted: List[CouncilProposal] = field(default_factory=list)
    rejected: List[CouncilProposal] = field(default_factory=list)
    deferred: List[CouncilProposal] = field(default_factory=list)
    reasoning: Dict[str, str] = field(default_factory=dict)
    budget_used: Dict[str, int] = field(default_factory=dict)


class ResourceAllocator:
    def __init__(self, blackboard: Blackboard, strategy: AllocationStrategy = AllocationStrategy.BALANCED):
        self.blackboard = blackboard
        self.strategy = strategy
        self._budget: Dict[str, float] = {
            "minerals": 10000,
            "vespene": 5000,
            "supply": 200,
        }
        self._spent: Dict[str, float] = {
            "minerals": 0,
            "vespene": 0,
            "supply": 0,
        }
        self._council_weights: Dict[str, float] = {
            "logistics": 1.2,
            "economy": 1.1,
            "capability": 1.0,
            "reconnaissance": 0.9,
            "counter_intelligence": 0.8,
            "military_industrial": 1.0,
            "technology": 0.9,
            "strategy": 0.7,
        }
        self._allocation_history: List[AllocationResult] = []

    def set_budget(self, minerals: float, vespene: float, supply: float) -> None:
        self._budget["minerals"] = minerals
        self._budget["vespene"] = vespene
        self._budget["supply"] = supply

    def update_budget(self, minerals: float, vespene: float, supply_cap: int) -> None:
        self._budget["minerals"] = minerals
        self._budget["vespene"] = vespene
        self._budget["supply"] = supply_cap
        self._spent = {"minerals": 0, "vespene": 0, "supply": 0}

    def allocate(self, proposals: List[CouncilProposal], game_state: Dict[str, Any]) -> AllocationResult:
        if not proposals:
            return AllocationResult()

        minerals = game_state.get("minerals", 0)
        vespene = game_state.get("vespene", 0)
        supply_cap = game_state.get("supply_cap", 200)
        supply_used = game_state.get("supply_used", 0)
        supply_available = supply_cap - supply_used

        scored = []
        for p in proposals:
            weight = self._council_weights.get(p.council_name, 1.0)
            score = self._score_proposal(p, weight)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        result = AllocationResult()
        budget_remaining = {
            "minerals": minerals,
            "vespene": vespene,
            "supply": supply_available,
        }

        for score, proposal in scored:
            content = proposal.content
            cost_minerals = content.get("minerals_cost", 0)
            cost_vespene = content.get("vespene_cost", 0)
            cost_supply = content.get("supply_cost", 0)

            if (budget_remaining["minerals"] >= cost_minerals and
                budget_remaining["vespene"] >= cost_vespene and
                budget_remaining["supply"] >= cost_supply):
                result.accepted.append(proposal)
                budget_remaining["minerals"] -= cost_minerals
                budget_remaining["vespene"] -= cost_vespene
                budget_remaining["supply"] -= cost_supply
                result.reasoning[proposal.proposal_type] = f"accepted (score={score:.2f})"
            elif proposal.priority >= 0.8:
                result.deferred.append(proposal)
                result.reasoning[proposal.proposal_type] = "deferred (high priority, insufficient budget)"
            else:
                result.rejected.append(proposal)
                result.reasoning[proposal.proposal_type] = f"rejected (score={score:.2f})"

        result.budget_used = {
            "minerals": minerals - budget_remaining["minerals"],
            "vespene": vespene - budget_remaining["vespene"],
            "supply": supply_available - budget_remaining["supply"],
        }

        self._allocation_history.append(result)
        if len(self._allocation_history) > 100:
            self._allocation_history.pop(0)

        for p in result.accepted:
            self.blackboard.accept_proposal(p)
        for p in result.rejected:
            self.blackboard.reject_proposal(p, result.reasoning.get(p.proposal_type, ""))

        return result

    def _score_proposal(self, proposal: CouncilProposal, weight: float) -> float:
        utility = proposal.content.get("utility", 0.5)
        confidence = proposal.confidence
        priority = proposal.priority

        if self.strategy == AllocationStrategy.PRIORITY:
            return priority * weight
        elif self.strategy == AllocationStrategy.UTILITY:
            return utility * weight
        elif self.strategy == AllocationStrategy.CONFIDENCE_WEIGHTED:
            return utility * confidence * weight
        else:
            return (priority * 0.4 + utility * 0.3 + confidence * 0.3) * weight

    def get_history(self) -> List[AllocationResult]:
        return list(self._allocation_history)
