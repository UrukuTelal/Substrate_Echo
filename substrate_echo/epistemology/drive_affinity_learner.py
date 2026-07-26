"""Drive Affinity Learner — Learns which actions satisfy which needs.

No domain knowledge. Starts ignorant. Discovers through experimentation.

Architecture:
    Action executed → observe need deficits before/after → update affinity matrix

The learner maintains a matrix: action_type × need_type → estimated utility.
Initially all 0.5 (neutral). After each action, if a need's deficit decreased,
the action-need affinity is reinforced. If deficit increased, it's weakened.

Exploration: with probability epsilon, try a random action instead of the
"best" one. This discovers new action-need relationships.

Persistence: save/load to JSON so learning carries across runs.

Usage:
    learner = DriveAffinityLearner(action_types, need_types)

    # Before action
    deficits_before = learner.get_deficit_snapshot(drive_manager)

    # Score a candidate
    affinities = learner.get_affinities("build_economy")
    utility = learner.dot_product("build_economy", deficits_before)

    # After action
    learner.update("build_economy", deficits_before, drive_manager)

    # Exploration
    if learner.should_explore():
        action = learner.random_action()
"""
from __future__ import annotations
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class DriveAffinityLearner:
    """Learns action→need affinities through experimentation.

    Pure cognitive module. No domain knowledge.
    """

    def __init__(
        self,
        action_types: List[str],
        need_types: List[str],
        learning_rate: float = 0.05,
        exploration_rate: float = 0.25,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
    ):
        self.action_types = list(action_types)
        self.need_types = list(need_types)
        self.lr = learning_rate
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration

        # Affinity matrix: action → need → utility [0, 1]
        self.affinities: Dict[str, Dict[str, float]] = {
            a: {n: 0.5 for n in need_types}
            for a in action_types
        }

        # Visit counts
        self.visits: Dict[str, int] = {a: 0 for a in action_types}
        self.total_updates = 0

        # Outcome history: action → list of (deficit_delta per need)
        self.outcome_history: Dict[str, List[Dict[str, float]]] = {
            a: [] for a in action_types
        }

    def get_affinities(self, action_type: str) -> Dict[str, float]:
        """Get learned affinities for an action type."""
        return self.affinities.get(action_type, {n: 0.5 for n in self.need_types})

    def dot_product(self, action_type: str, deficits: Dict[str, float]) -> float:
        """Compute drive utility = deficit · affinity for an action."""
        affs = self.get_affinities(action_type)
        if not affs or not deficits:
            return 0.0
        total = 0.0
        for need_str, affinity in affs.items():
            deficit = deficits.get(need_str, 0.0)
            total += deficit * affinity
        return total

    def should_explore(self) -> bool:
        """Should we try a random action?"""
        return random.random() < self.epsilon

    def random_action(self) -> str:
        """Pick a random action type."""
        return random.choice(self.action_types)

    def update(
        self,
        action_type: str,
        deficits_before: Dict[str, float],
        deficits_after: Dict[str, float],
    ):
        """Update affinities based on observed outcome.

        If deficit decreased (action helped need), reinforce.
        If deficit increased (action hurt need), weaken.
        """
        if action_type not in self.affinities:
            return

        deltas = {}
        for need_str in self.need_types:
            before = deficits_before.get(need_str, 0.0)
            after = deficits_after.get(need_str, 0.0)
            # Positive delta = deficit decreased = action helped
            delta = before - after
            deltas[need_str] = delta

            # Update affinity
            old = self.affinities[action_type][need_str]
            new = old + self.lr * delta
            self.affinities[action_type][need_str] = max(0.0, min(1.0, new))

        self.visits[action_type] = self.visits.get(action_type, 0) + 1
        self.total_updates += 1

        # Record outcome
        self.outcome_history[action_type].append(deltas)
        if len(self.outcome_history[action_type]) > 200:
            self.outcome_history[action_type].pop(0)

        # Decay exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_confidence(self, action_type: str) -> float:
        """How confident are we in this action's affinities?

        Based on number of observations and variance of outcomes.
        """
        visits = self.visits.get(action_type, 0)
        if visits < 5:
            return 0.2  # low confidence — few observations
        if visits < 20:
            return 0.5
        if visits < 50:
            return 0.7
        return min(0.95, 0.7 + visits * 0.005)

    def get_most_surprising(self) -> Optional[str]:
        """Which action has the most surprising affinities?

        High surprise = big deviation from 0.5 (initial) = learned something.
        """
        best_action = None
        best_surprise = 0.0
        for action, affs in self.affinities.items():
            surprise = sum(abs(v - 0.5) for v in affs.values())
            if surprise > best_surprise:
                best_surprise = surprise
                best_action = action
        return best_action

    def get_action_ranking(self, deficits: Dict[str, float]) -> List[tuple]:
        """Rank actions by drive utility for current deficits."""
        ranking = []
        for action in self.action_types:
            utility = self.dot_product(action, deficits)
            confidence = self.get_confidence(action)
            ranking.append((action, utility, confidence))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def save(self, path: str):
        """Persist learner state to JSON."""
        data = {
            "affinities": self.affinities,
            "visits": self.visits,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "outcome_history": {
                a: hist[-50:] for a, hist in self.outcome_history.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load learner state from JSON. Returns True if successful."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.affinities = data["affinities"]
            self.visits = data["visits"]
            self.total_updates = data["total_updates"]
            self.epsilon = data["epsilon"]
            self.outcome_history = data.get("outcome_history", {
                a: [] for a in self.action_types
            })
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def render(self) -> str:
        """Human-readable summary of what the bot has learned."""
        lines = ["  Drive Affinity Learner:"]
        lines.append(f"    Total updates: {self.total_updates}")
        lines.append(f"    Exploration rate: {self.epsilon:.3f}")
        lines.append("")
        lines.append(f"    {'Action':15s} {'Visits':>6s} {'Conf':>5s} ", )
        # Need headers
        header = "    " + " " * 15 + " "
        for n in self.need_types:
            header += f"{n[:6]:>7s}"
        lines.append(header)
        lines.append("    " + "-" * (15 + 6 + 5 + 7 * len(self.need_types)))

        for action in self.action_types:
            visits = self.visits.get(action, 0)
            conf = self.get_confidence(action)
            affs = self.affinities.get(action, {})
            aff_str = " ".join(f"{affs.get(n, 0.5):7.3f}" for n in self.need_types)
            lines.append(f"    {action:15s} {visits:6d} {conf:5.2f} {aff_str}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affinities": self.affinities,
            "visits": self.visits,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
        }
