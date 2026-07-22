"""Planner — model-based planning: simulate → evaluate → select best.

The planner takes an intent and produces a concrete plan:
a sequence of actions that will achieve the intent, evaluated
by the Evaluator and constrained by the Controller.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .intent import Intent, IntentProposal
from .simulator import Simulator, ActionDelta, SimResult, SimConfig
from .evaluator import Evaluator, EvalResult, UtilityWeights
from .controller import Controller, ControlConfig, ControlOutput
from .world_model import WorldModel


# Pillar indices for intent-to-action mapping
PILLAR_MAP = {
    Intent.INCREASE_COHESION: 6,     # Cohesion
    Intent.REDUCE_DISTORTION: 13,    # Distortion (suppress)
    Intent.AVOID_HARM: 12,           # Harm (suppress)
    Intent.RESTORE_INTEGRITY: 5,     # Integrity
    Intent.REDUCE_FLUX: 14,          # Flux (suppress)
    Intent.SEEK_NOVELTY: 13,         # Distortion (boost — controlled chaos)
}


@dataclass
class PlannerConfig:
    """Configuration for the planner."""
    n_candidates: int = 20        # candidate actions per planning step
    lookahead_steps: int = 10     # simulation horizon per candidate
    beam_width: int = 5           # top candidates to extend
    exploration_rate: float = 0.1 # fraction of random candidates
    max_plan_length: int = 5      # max actions in a plan
    action_magnitudes: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])


class Plan:
    """A concrete plan: sequence of actions toward an intent."""
    
    def __init__(self, intent: IntentProposal,
                 actions: list[ActionDelta],
                 sim_results: list[SimResult],
                 eval_results: list[EvalResult]):
        self.intent = intent
        self.actions = actions
        self.sim_results = sim_results
        self.eval_results = eval_results
    
    @property
    def total_utility(self) -> float:
        """Total discounted utility of the plan."""
        if not self.eval_results:
            return 0.0
        discount = 0.9
        return sum(discount ** i * e.utility for i, e in enumerate(self.eval_results))
    
    @property
    def final_state(self) -> Optional[np.ndarray]:
        if self.sim_results:
            return self.sim_results[-1].final_state
        return None
    
    @property
    def confidence(self) -> float:
        if not self.sim_results:
            return 0.0
        return float(np.mean([s.confidence for s in self.sim_results]))
    
    @property
    def basin_transitions(self) -> list[int]:
        """All basin IDs visited across the plan."""
        basins = []
        for sr in self.sim_results:
            basins.extend(sr.basin_transitions)
        return basins
    
    def __repr__(self):
        return (f"Plan(intent={self.intent.intent.name}, "
                f"actions={len(self.actions)}, "
                f"U={self.total_utility:.3f}, "
                f"conf={self.confidence:.3f})")


class Planner:
    """Model-based planner using simulation + evaluation.
    
    Planning algorithm:
    1. Generate candidate actions based on intent
    2. Simulate each candidate
    3. Evaluate each simulation result
    4. Select best candidates (beam search)
    5. Extend top candidates with further actions
    6. Return highest-utility plan
    """
    
    def __init__(self, simulator: Simulator, evaluator: Evaluator,
                 controller: Optional[Controller] = None,
                 config: Optional[PlannerConfig] = None):
        self.simulator = simulator
        self.evaluator = evaluator
        self.controller = controller or Controller()
        self.config = config or PlannerConfig()
    
    def plan(self, state: np.ndarray,
             intent: IntentProposal) -> Plan:
        """Generate a plan for the given intent."""
        state = np.asarray(state, dtype=np.float64)
        
        # Generate candidate actions
        candidates = self.generate_candidates(state, intent)
        
        if not candidates:
            return Plan(intent=intent, actions=[], sim_results=[], eval_results=[])
        
        # Simulate and evaluate each candidate
        scored = []
        for action in candidates:
            sim_result = self.simulator.simulate(state, action, steps=self.config.lookahead_steps, diagnostics=False)
            eval_result = self.evaluator.evaluate_trajectory(sim_result.trajectory)
            scored.append((eval_result.utility, action, sim_result, eval_result))
        
        # Sort by utility
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Beam search: extend top candidates
        best_plans = []
        for _, action, sim_result, eval_result in scored[:self.config.beam_width]:
            # Single-step plan for simplicity (extend later)
            plan = Plan(
                intent=intent,
                actions=[action],
                sim_results=[sim_result],
                eval_results=[eval_result],
            )
            best_plans.append(plan)
        
        if not best_plans:
            return Plan(intent=intent, actions=[], sim_results=[], eval_results=[])
        
        return max(best_plans, key=lambda p: p.total_utility)
    
    def plan_sequence(self, state: np.ndarray,
                      intent: IntentProposal,
                      n_steps: int = 3) -> Plan:
        """Multi-step planning: plan n_steps ahead."""
        state = np.asarray(state, dtype=np.float64)
        
        all_actions = []
        all_sim_results = []
        all_eval_results = []
        current_state = state.copy()
        
        for step in range(n_steps):
            candidates = self.generate_candidates(current_state, intent)
            if not candidates:
                break
            
            scored = []
            for action in candidates:
                sim_result = self.simulator.simulate(current_state, action,
                                                     steps=self.config.lookahead_steps,
                                                     diagnostics=False)
                eval_result = self.evaluator.evaluate_trajectory(sim_result.trajectory)
                scored.append((eval_result.utility, action, sim_result, eval_result))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            best_utility, best_action, best_sim, best_eval = scored[0]
            
            all_actions.append(best_action)
            all_sim_results.append(best_sim)
            all_eval_results.append(best_eval)
            current_state = best_sim.final_state
        
        return Plan(
            intent=intent,
            actions=all_actions,
            sim_results=all_sim_results,
            eval_results=all_eval_results,
        )
    
    def plan_to_basin(self, state: np.ndarray,
                      target_basin: int) -> Plan:
        """Plan a transition to a specific basin."""
        attractors = self.simulator.world_model.get_attractors()
        if target_basin >= len(attractors):
            return Plan(
                intent=IntentProposal(intent=Intent.DEFEND, priority=0.0, confidence=0.0),
                actions=[], sim_results=[], eval_results=[],
            )
        
        target = attractors[target_basin]
        intent = IntentProposal(
            intent=Intent.EXPLORE,
            priority=0.8,
            confidence=0.7,
            reasoning=f"plan transition to basin {target_basin}",
            target_state=target,
        )
        
        # Generate candidates biased toward target
        candidates = []
        
        # Direct approach
        delta = target - state
        mag = np.linalg.norm(delta)
        if mag > 0.01:
            for scale in [0.3, 0.5, 0.8, 1.0]:
                d = delta * min(scale, 0.2 / max(mag, 0.01))
                candidates.append(ActionDelta(
                    delta=d, description=f"toward_basin_{target_basin}"
                ))
        
        # Random exploration
        for _ in range(self.config.n_candidates // 2):
            candidates.append(ActionDelta.random(
                dim=len(state), magnitude=0.1
            ))
        
        # Simulate and evaluate
        scored = []
        for action in candidates:
            sim_result = self.simulator.simulate(state, action,
                                                 steps=self.config.lookahead_steps * 2,
                                                 diagnostics=False)
            # Bonus for reaching target basin
            final_basin = self.simulator.world_model.get_basin(sim_result.final_state)
            basin_bonus = 2.0 if final_basin == target_basin else 0.0
            
            eval_result = self.evaluator.evaluate_trajectory(sim_result.trajectory)
            eval_result.utility += basin_bonus
            scored.append((eval_result.utility, action, sim_result, eval_result))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if scored:
            _, best_action, best_sim, best_eval = scored[0]
            return Plan(
                intent=intent,
                actions=[best_action],
                sim_results=[best_sim],
                eval_results=[best_eval],
            )
        
        return Plan(intent=intent, actions=[], sim_results=[], eval_results=[])
    
    def generate_candidates(self, state: np.ndarray,
                            intent: IntentProposal) -> list[ActionDelta]:
        """Generate candidate actions based on intent."""
        candidates = []
        rng = np.random.RandomState()
        
        # Intent-specific actions
        if intent.intent in PILLAR_MAP:
            pillar_idx = PILLAR_MAP[intent.intent]
            for mag in self.config.action_magnitudes:
                if intent.intent in (Intent.AVOID_HARM, Intent.REDUCE_DISTORTION,
                                     Intent.REDUCE_FLUX):
                    candidates.append(ActionDelta.pillar_suppress(pillar_idx, mag, len(state)))
                else:
                    candidates.append(ActionDelta.pillar_boost(pillar_idx, mag, len(state)))
        
        # Target-directed actions
        if intent.target_state is not None:
            candidates.append(ActionDelta.toward_target(
                intent.target_state, state, max_magnitude=0.2
            ))
        
        # DEFEND: small perturbations (test stability)
        if intent.intent == Intent.DEFEND:
            for _ in range(5):
                candidates.append(ActionDelta.random(len(state), magnitude=0.02, rng=rng))
        
        # EXPLORE/SEEK_NOVELTY: larger random actions
        if intent.intent in (Intent.EXPLORE, Intent.SEEK_NOVELTY, Intent.INVESTIGATE):
            for mag in [0.05, 0.1, 0.15, 0.2]:
                candidates.append(ActionDelta.random(len(state), magnitude=mag, rng=rng))
        
        # Random exploration
        n_random = max(1, int(self.config.n_candidates * self.config.exploration_rate))
        for _ in range(n_random):
            candidates.append(ActionDelta.random(len(state), magnitude=0.1, rng=rng))
        
        return candidates
    
    def is_safe(self, state: np.ndarray) -> bool:
        """Check if a state is safe (harm < threshold, integrity > threshold)."""
        harm = state[12] if len(state) > 12 else 0.0
        integrity = state[5] if len(state) > 5 else 0.5
        return harm < 0.8 and integrity > 0.2
    
    def find_safe_action(self, state: np.ndarray) -> ActionDelta:
        """Find an action that leads to a safe state."""
        intent = IntentProposal(
            intent=Intent.AVOID_HARM,
            priority=1.0,
            confidence=1.0,
            reasoning="safety critical",
        )
        plan = self.plan(state, intent)
        if plan.actions:
            return plan.actions[0]
        # Fallback: suppress harm pillar
        return ActionDelta.pillar_suppress(12, 0.15, len(state))
