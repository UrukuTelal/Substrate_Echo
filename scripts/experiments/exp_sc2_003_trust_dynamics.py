"""EXP-SC2-003: Trust Dynamics Test.

Tests trust formation, decay, and recovery within a single game session.

Validates:
  - Trust formation through repeated positive interactions
  - Trust decay through negative interactions
  - Trust recovery through renewed positive evidence
  - Trust-informed communication decisions
  - Trust level classification

Architecture:
  SC2 Game State (Phase 1: Cooperative)
       |
  Trust Layer (Formation)
       |
  SC2 Game State (Phase 2: Betrayal)
       |
  Trust Layer (Decay)
       |
  SC2 Game State (Phase 3: Recovery)
       |
  Trust Layer (Recovery)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from substrate_echo.kernel import SubstrateKernel, KernelConfig, Observation, CognitiveState, Goal
from substrate_echo.kernel.executive import GoalTier, GoalStatus
from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder, ActionType
from substrate_echo.embodiments.sc2.trust import TrustEvaluationLayer, TrustLevel
from substrate_echo.embodiments.sc2.communication import CommunicationPolicyLayer, InfoCategory

os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'
from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId
from pathlib import Path as SC2Path


@dataclass
class ExperimentConfig:
    """EXP-SC2-003 configuration."""
    map_name: str = "Simple64"
    max_steps: int = 1500
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    realtime: bool = False
    
    kernel_dims: int = 16
    convergence_window: int = 80
    report_interval: int = 200
    
    # Phase durations (in steps)
    phase1_steps: int = 500   # Cooperative
    phase2_steps: int = 500   # Betrayal
    phase3_steps: int = 500   # Recovery
    interaction_interval: int = 50  # Steps between trust interactions


@dataclass
class GamePhase:
    """Phase of the experiment."""
    name: str
    description: str
    trust_events: List[Dict[str, str]]  # Events to simulate


PHASES = [
    GamePhase("Cooperative", "Positive interactions build trust", [
        {"type": "agreement_honored", "target": "ally"},
        {"type": "predictable_behavior", "target": "enemy"},
    ]),
    GamePhase("Betrayal", "Negative interactions reduce trust", [
        {"type": "agreement_broken", "target": "ally"},
        {"type": "unprovoked_attack", "target": "enemy"},
    ]),
    GamePhase("Recovery", "Positive interactions rebuild trust", [
        {"type": "accurate_prediction", "target": "ally"},
        {"type": "deception_detected", "target": "enemy"},
    ]),
]


class TrustDynamicsBot(BotAI):
    """SC2 bot for testing trust dynamics."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Kernel
        kernel_config = KernelConfig(
            dim=config.kernel_dims,
            convergence_window=config.convergence_window,
        )
        self.kernel = SubstrateKernel(config=kernel_config)
        
        # SC2 components
        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()
        
        # Trust layers
        self.trust = TrustEvaluationLayer()
        self.communication = CommunicationPolicyLayer(self.trust)
        
        # State tracking
        self._step = 0
        self._current_phase = 0
        self._phase_start_step = 0
        
        # Metrics
        self._phase_metrics = []
        self._trust_history = []
        self._communication_history = []
        
        # Agent IDs
        self._ally_agent = "ally_001"
        self._enemy_agent = "enemy_001"
    
    async def on_start(self):
        """Called when game starts."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-003: Trust Dynamics Test")
        print(f"{'='*60}")
        print(f"Max Steps: {self.config.max_steps}")
        print(f"Phase 1: Cooperative ({self.config.phase1_steps} steps)")
        print(f"Phase 2: Betrayal ({self.config.phase2_steps} steps)")
        print(f"Phase 3: Recovery ({self.config.phase3_steps} steps)")
        print(f"{'='*60}\n")
        
        # Initialize agents with neutral trust
        self.trust.register_agent(self._ally_agent, initial_trust=0.5)
        self.trust.register_agent(self._enemy_agent, initial_trust=0.5)
        
        self._start_phase(0)
    
    async def on_step(self, iteration: int):
        """Called each game step."""
        self._step += 1
        
        # 1. Check phase transitions
        self._check_phase_transition()
        
        # 2. Simulate trust interactions
        if self._step % self.config.interaction_interval == 0:
            self._simulate_trust_interactions()
        
        # 3. Record trust state
        if self._step % 10 == 0:
            self._record_trust_state()
        
        # 4. Process communication decisions
        if self._step % 100 == 0:
            self._process_communication()
        
        # 5. Observe game state
        observation = self.state
        vec = self.encoder.encode(observation)
        
        # 6. Feed to kernel
        kernel_obs = Observation(
            vector=vec.tolist(),
            modality="sc2_trust",
            embodiment_id="sc2_trust",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "phase": PHASES[self._current_phase].name,
                "minerals": self.minerals,
                "units": len(self.units),
            }
        )
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        
        # 7. Execute basic action
        await self._execute_basic_action()
        
        # 8. Status report
        if self._step % self.config.report_interval == 0:
            self._report_status()
        
        # 9. Check termination
        if self._step >= self.config.max_steps:
            self._print_final_report()
    
    def _check_phase_transition(self):
        """Check if we should transition to next phase."""
        phase_steps = [
            self.config.phase1_steps,
            self.config.phase2_steps,
            self.config.phase3_steps,
        ]
        
        steps_in_phase = self._step - self._phase_start_step
        if steps_in_phase >= phase_steps[self._current_phase]:
            self._complete_current_phase()
            if self._current_phase < len(PHASES) - 1:
                self._start_phase(self._current_phase + 1)
    
    def _start_phase(self, phase_idx: int):
        """Start a new phase."""
        self._current_phase = phase_idx
        self._phase_start_step = self._step
        phase = PHASES[phase_idx]
        
        print(f"\n--- Phase {phase_idx + 1}: {phase.name} ---")
        print(f"  {phase.description}")
        print(f"  Ally Trust: {self.trust.get_trust(self._ally_agent):.3f}")
        print(f"  Enemy Trust: {self.trust.get_trust(self._enemy_agent):.3f}")
    
    def _complete_current_phase(self):
        """Record metrics for completed phase."""
        phase = PHASES[self._current_phase]
        steps_in_phase = self._step - self._phase_start_step
        
        phase_data = {
            "phase": phase.name,
            "steps": steps_in_phase,
            "final_ally_trust": self.trust.get_trust(self._ally_agent),
            "final_enemy_trust": self.trust.get_trust(self._enemy_agent),
            "final_ally_level": self.trust.get_level(self._ally_agent).name,
            "final_enemy_level": self.trust.get_level(self._enemy_agent).name,
        }
        self._phase_metrics.append(phase_data)
        
        print(f"\n  Phase Complete: {phase.name}")
        print(f"    Ally Trust: {phase_data['final_ally_trust']:.3f} ({phase_data['final_ally_level']})")
        print(f"    Enemy Trust: {phase_data['final_enemy_trust']:.3f} ({phase_data['final_enemy_level']})")
    
    def _simulate_trust_interactions(self):
        """Simulate trust interactions based on current phase."""
        phase = PHASES[self._current_phase]
        
        for event in phase.trust_events:
            target = self._ally_agent if event["target"] == "ally" else self._enemy_agent
            self.trust.observe(target, {"type": event["type"]})
    
    def _process_communication(self):
        """Process communication decisions based on trust."""
        should_share_economy = self.communication.should_send(
            self._ally_agent,
            InfoCategory.ECONOMY,
            info_value=0.7
        )
        
        should_share_army = self.communication.should_send(
            self._enemy_agent,
            InfoCategory.ARMY,
            info_value=0.8
        )
        
        self._communication_history.append({
            "step": self._step,
            "phase": PHASES[self._current_phase].name,
            "ally_trust": self.trust.get_trust(self._ally_agent),
            "enemy_trust": self.trust.get_trust(self._enemy_agent),
            "share_economy_with_ally": should_share_economy,
            "share_army_with_enemy": should_share_army,
        })
    
    def _record_trust_state(self):
        """Record current trust state."""
        self._trust_history.append({
            "step": self._step,
            "phase": PHASES[self._current_phase].name,
            "ally_trust": self.trust.get_trust(self._ally_agent),
            "enemy_trust": self.trust.get_trust(self._enemy_agent),
            "ally_level": self.trust.get_level(self._ally_agent).name,
            "enemy_level": self.trust.get_level(self._enemy_agent).name,
        })
    
    async def _execute_basic_action(self):
        """Execute basic game action."""
        if self.minerals >= 150:
            for unit in self.units:
                if unit.is_structure and unit.type_id in {UnitTypeId.BARRACKS, UnitTypeId.FACTORY}:
                    if unit.is_idle and self.can_afford(UnitTypeId.MARINE):
                        await unit.train(UnitTypeId.MARINE)
                        break
    
    def _report_status(self):
        """Print status report."""
        ally_trust = self.trust.get_trust(self._ally_agent)
        enemy_trust = self.trust.get_trust(self._enemy_agent)
        
        print(f"\n--- Step {self._step} ({PHASES[self._current_phase].name}) ---")
        print(f"  Kernel: {self.kernel._tick} ticks, {self.kernel._compute_coherence():.3f} coherence")
        print(f"  Trust: Ally={ally_trust:.3f}, Enemy={enemy_trust:.3f}")
        print(f"  Game: Minerals={self.minerals}, Units={len(self.units)}")
    
    def _print_final_report(self):
        """Print final experiment report."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-003: Trust Dynamics Final Report")
        print(f"{'='*60}")
        
        print(f"\nPhase Results:")
        for i, phase_data in enumerate(self._phase_metrics):
            print(f"\n  Phase {i+1}: {phase_data['phase']}")
            print(f"    Steps: {phase_data['steps']}")
            print(f"    Final Ally Trust: {phase_data['final_ally_trust']:.3f} ({phase_data['final_ally_level']})")
            print(f"    Final Enemy Trust: {phase_data['final_enemy_trust']:.3f} ({phase_data['final_enemy_level']})")
        
        print(f"\nTrust Evolution:")
        if len(self._trust_history) >= 2:
            first = self._trust_history[0]
            last = self._trust_history[-1]
            print(f"  Initial: Ally={first['ally_trust']:.3f}, Enemy={first['enemy_trust']:.3f}")
            print(f"  Final: Ally={last['ally_trust']:.3f}, Enemy={last['enemy_trust']:.3f}")
        
        print(f"\nCommunication Decisions:")
        if self._communication_history:
            share_with_ally = sum(1 for c in self._communication_history if c['share_economy_with_ally'])
            share_with_enemy = sum(1 for c in self._communication_history if c['share_army_with_enemy'])
            print(f"  Shared economy with ally: {share_with_ally}/{len(self._communication_history)}")
            print(f"  Shared army with enemy: {share_with_enemy}/{len(self._communication_history)}")
        
        print(f"\nKernel Metrics:")
        print(f"  Total ticks: {self.kernel._tick}")
        print(f"  Final coherence: {self.kernel._compute_coherence():.3f}")
        print(f"  Attractors formed: {len(self.kernel._base_attractors)}")
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete")
        print(f"{'='*60}")


def run_experiment():
    """Run EXP-SC2-003."""
    config = ExperimentConfig(
        map_name="Simple64",
        max_steps=1500,
        difficulty=Difficulty.Easy,
        realtime=False,
    )
    
    bot = TrustDynamicsBot(config)
    
    map_path = SC2Path(r'C:\Program Files (x86)\StarCraft II\Maps\Melee\Simple64.SC2Map')
    map_settings = Map(map_path)
    
    result = run_game(
        map_settings=map_settings,
        players=[
            Bot(Race.Terran, bot),
            Computer(Race.Random, config.difficulty),
        ],
        realtime=config.realtime,
    )
    
    bot._print_final_report()
    return result


if __name__ == "__main__":
    result = run_experiment()
    print(f"\nFinal Result: {result}")