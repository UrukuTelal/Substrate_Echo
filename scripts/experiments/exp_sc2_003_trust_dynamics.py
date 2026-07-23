"""EXP-SC2-003: Trust Dynamics Test.

Tests trust formation, decay, and recovery through multi-game sessions.

Validates:
  - Trust formation through repeated interactions
  - Trust decay over time without evidence
  - Trust recovery after negative events
  - Multi-game session persistence
  - Trust-informed communication decisions

Architecture:
  SC2 Game State (Game 1: Cooperative)
       |
  Observation Encoder
       |
  Trust Layer (Formation Phase)
       |
  SC2 Game State (Game 2: Betrayal)
       |
  Trust Layer (Betrayal Phase)
       |
  SC2 Game State (Game 3: Recovery)
       |
  Trust Layer (Recovery Phase)
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
    steps_per_game: int = 500  # Steps per game session
    num_games: int = 3         # Number of game sessions
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    realtime: bool = False
    
    kernel_dims: int = 16
    convergence_window: int = 80
    report_interval: int = 100


@dataclass
class GamePhase:
    """Phase of the experiment."""
    name: str
    trust_action: str  # "positive", "negative", "neutral"
    trust_delta: float  # Expected trust change direction
    description: str


# Experiment phases
PHASES = [
    GamePhase("Cooperative", "positive", 0.3, "Positive interactions build trust"),
    GamePhase("Betrayal", "negative", -0.4, "Negative interactions reduce trust"),
    GamePhase("Recovery", "positive", 0.2, "Positive interactions rebuild trust"),
]


class TrustDynamicsBot(BotAI):
    """SC2 bot for testing trust dynamics across game sessions."""
    
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
        self._game_step = 0
        self._current_phase = 0
        
        # Metrics
        self._phase_metrics = []
        self._trust_history = []
        self._communication_history = []
        
        # Agent IDs for trust testing
        self._ally_agent = "ally_001"
        self._enemy_agent = "enemy_001"
    
    async def on_start(self):
        """Called when game starts."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-003: Trust Dynamics Test")
        print(f"{'='*60}")
        print(f"Games: {self.config.num_games}")
        print(f"Steps per Game: {self.config.steps_per_game}")
        print(f"Difficulty: {self.config.difficulty}")
        print(f"{'='*60}\n")
        
        # Initialize agents
        self.trust.register_agent(self._ally_agent, initial_trust=0.5)
        self.trust.register_agent(self._enemy_agent, initial_trust=0.5)
        
        # Start first phase
        self._start_phase(0)
    
    async def on_step(self, iteration: int):
        """Called each game step."""
        self._step += 1
        self._game_step += 1
        
        # 1. Observe game state
        observation = self._observe()
        
        # 2. Encode to 16D vector
        vec = self.encoder.encode(observation)
        
        # 3. Feed to kernel
        kernel_obs = Observation(
            vector=vec.tolist(),
            modality="sc2_trust",
            embodiment_id="sc2_trust",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "game_step": self._game_step,
                "phase": PHASES[self._current_phase].name,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
                "units": len(self.units),
            }
        )
        
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        
        # 4. Trust dynamics based on phase
        self._process_trust_dynamics()
        
        # 5. Communication decisions based on trust
        self._process_communication()
        
        # 6. Execute action
        action = self._extract_action(cognitive_state)
        if action:
            await self._execute_action(action)
        
        # 7. Record trust state
        if self._game_step % 10 == 0:
            self._record_trust_state()
        
        # 8. Status report
        if self._game_step % self.config.report_interval == 0:
            self._report_status()
        
        # 9. Check game completion
        if self._game_step >= self.config.steps_per_game:
            self._complete_game()
    
    def _observe(self) -> Any:
        """Get current game observation."""
        return self.state
    
    def _start_phase(self, phase_idx: int):
        """Start a new experiment phase."""
        if phase_idx >= len(PHASES):
            print(f"\n{'='*60}")
            print(f"All phases complete!")
            print(f"{'='*60}")
            return
        
        self._current_phase = phase_idx
        self._game_step = 0
        phase = PHASES[phase_idx]
        
        print(f"\n--- Phase {phase_idx + 1}: {phase.name} ---")
        print(f"  {phase.description}")
        print(f"  Trust delta direction: {'+' if phase.trust_delta > 0 else ''}{phase.trust_delta}")
    
    def _process_trust_dynamics(self):
        """Process trust updates based on current phase."""
        phase = PHASES[self._current_phase]
        
        # Simulate interactions based on phase
        if phase.name == "Cooperative":
            # Positive interactions
            if self._game_step % 50 == 0:
                self.trust.observe(self._ally_agent, {
                    "type": "agreement_honored",
                    "outcome": "positive"
                })
                self.trust.observe(self._enemy_agent, {
                    "type": "predictable_behavior",
                    "outcome": "neutral"
                })
        
        elif phase.name == "Betrayal":
            # Negative interactions
            if self._game_step % 50 == 0:
                self.trust.observe(self._ally_agent, {
                    "type": "agreement_broken",
                    "outcome": "negative"
                })
                self.trust.observe(self._enemy_agent, {
                    "type": "unprovoked_attack",
                    "outcome": "negative"
                })
        
        elif phase.name == "Recovery":
            # Mixed interactions to test recovery
            if self._game_step % 50 == 0:
                self.trust.observe(self._ally_agent, {
                    "type": "accurate_prediction",
                    "outcome": "positive"
                })
                self.trust.observe(self._enemy_agent, {
                    "type": "deception_detected",
                    "outcome": "negative"
                })
    
    def _process_communication(self):
        """Process communication decisions based on trust."""
        # Test information sharing with different trust levels
        if self._game_step % 100 == 0:
            # Economy info sharing
            should_share_economy = self.communication.should_send(
                self._ally_agent,
                InfoCategory.ECONOMY,
                info_value=0.7
            )
            
            # Army info sharing
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
        ally_trust = self.trust.get_trust(self._ally_agent)
        enemy_trust = self.trust.get_trust(self._enemy_agent)
        
        self._trust_history.append({
            "step": self._step,
            "game_step": self._game_step,
            "phase": PHASES[self._current_phase].name,
            "ally_trust": ally_trust,
            "enemy_trust": enemy_trust,
            "ally_level": self.trust.get_level(self._ally_agent).name,
            "enemy_level": self.trust.get_level(self._enemy_agent).name,
        })
    
    def _extract_action(self, cognitive_state: CognitiveState) -> Optional[Dict[str, Any]]:
        """Extract SC2 action from cognitive state."""
        if cognitive_state.action and cognitive_state.action.vector:
            action_vec = np.array(cognitive_state.action.vector)
            
            action_types = [
                ActionType.EXPAND,
                ActionType.BUILD_ARMY,
                ActionType.DEFEND,
                ActionType.ATTACK,
                ActionType.SCOUT,
                ActionType.HOLD,
            ]
            
            if len(action_vec) >= len(action_types):
                action_idx = np.argmax(action_vec[:len(action_types)])
                action_type = action_types[action_idx]
                
                return {
                    "type": action_type.value,
                    "confidence": cognitive_state.action.confidence,
                }
        
        # Default action based on phase
        phase = PHASES[self._current_phase]
        if phase.name == "Cooperative":
            return {"type": "build_army"} if self.minerals >= 150 else {"type": "hold"}
        elif phase.name == "Betrayal":
            return {"type": "defend"} if len(self.units) > 5 else {"type": "build_army"}
        else:  # Recovery
            return {"type": "hold"}
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute action in SC2."""
        action_type = action.get("type", "hold")
        
        if action_type == "build_army":
            for unit in self.units:
                if unit.is_structure and unit.type_id in {UnitTypeId.BARRACKS, UnitTypeId.FACTORY}:
                    if unit.is_idle and self.can_afford(UnitTypeId.MARINE):
                        await unit.train(UnitTypeId.MARINE)
        
        elif action_type == "hold":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army:
                army.hold_position()
        
        elif action_type == "defend":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army and self.townhalls:
                army.move(self.townhalls.first.position)
    
    def _complete_game(self):
        """Complete current game and record metrics."""
        phase = PHASES[self._current_phase]
        
        # Record phase metrics
        phase_data = {
            "phase": phase.name,
            "steps": self._game_step,
            "final_ally_trust": self.trust.get_trust(self._ally_agent),
            "final_enemy_trust": self.trust.get_trust(self._enemy_agent),
            "final_ally_level": self.trust.get_level(self._ally_agent).name,
            "final_enemy_level": self.trust.get_level(self._enemy_agent).name,
            "communication_decisions": len(self._communication_history),
        }
        self._phase_metrics.append(phase_data)
        
        print(f"\n--- Game {self._current_phase + 1} Complete ---")
        print(f"  Ally Trust: {phase_data['final_ally_trust']:.3f} ({phase_data['final_ally_level']})")
        print(f"  Enemy Trust: {phase_data['final_enemy_trust']:.3f} ({phase_data['final_enemy_level']})")
        print(f"  Communication Decisions: {phase_data['communication_decisions']}")
        
        # Start next phase
        self._start_phase(self._current_phase + 1)
    
    def _report_status(self):
        """Print status report."""
        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active_goals = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]
        
        ally_trust = self.trust.get_trust(self._ally_agent)
        enemy_trust = self.trust.get_trust(self._enemy_agent)
        
        print(f"\n--- Step {self._step} (Game Step {self._game_step}) ---")
        print(f"  Phase: {PHASES[self._current_phase].name}")
        print(f"  Kernel: {self.kernel._tick} ticks, {self.kernel._compute_coherence():.3f} coherence")
        print(f"  Trust: Ally={ally_trust:.3f}, Enemy={enemy_trust:.3f}")
        print(f"  Game: Minerals={self.minerals}, Units={len(self.units)}")
    
    def print_final_report(self):
        """Print final experiment report."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-003: Trust Dynamics Final Report")
        print(f"{'='*60}")
        
        print(f"\nPhase Results:")
        for i, phase_data in enumerate(self._phase_metrics):
            print(f"\n  Game {i+1}: {phase_data['phase']}")
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
        steps_per_game=500,
        num_games=3,
        difficulty=Difficulty.Easy,
        realtime=False,
    )
    
    bot = TrustDynamicsBot(config)
    
    # Create map settings
    map_path = SC2Path(r'C:\Program Files (x86)\StarCraft II\Maps\Melee\Simple64.SC2Map')
    map_settings = Map(map_path)
    
    # Run all games in sequence
    results = []
    for game_idx in range(config.num_games):
        print(f"\nStarting Game {game_idx + 1}/{config.num_games}")
        
        # Reset bot for new game
        bot._game_step = 0
        bot._current_phase = game_idx
        
        result = run_game(
            map_settings=map_settings,
            players=[
                Bot(Race.Terran, bot),
                Computer(Race.Random, config.difficulty),
            ],
            realtime=config.realtime,
        )
        results.append(result)
    
    bot.print_final_report()
    return results


if __name__ == "__main__":
    results = run_experiment()
    print(f"\nAll Results: {results}")