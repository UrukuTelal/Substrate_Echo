"""EXP-SC2-001: Kernel Integration Test.

Connects SC2 observation loop to Substrate Kernel.

Validates:
  - Real-time cognition loop
  - Goal management in game context
  - Attractor updates from game state
  - Executive decisions under game pressure
  - Action translation from kernel to SC2

Architecture:
  SC2 Game State
       |
  Observation Encoder (16D)
       |
  Substrate Kernel
       |
  CognitiveState (action, prediction, goals)
       |
  Action Decoder
       |
  SC2 Commands
"""
from __future__ import annotations
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from substrate_echo.kernel import SubstrateKernel, KernelConfig, Observation, CognitiveState, Goal
from substrate_echo.kernel.executive import GoalTier, GoalStatus
from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder, AbstractAction, ActionType
from substrate_echo.embodiments.sc2.trust import TrustEvaluationLayer
from substrate_echo.embodiments.sc2.communication import CommunicationPolicyLayer, InfoCategory
from substrate_echo.embodiments.sc2.trickster import TricksterStoryTeller, NarrativeContext, NarrativeStyle
from substrate_echo.embodiments.sc2.truce_mode import TruceModeOptimizer, TruceMode

# SC2 imports
os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'
from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId
from pathlib import Path as SC2Path


# ── Configuration ────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """EXP-SC2-001 configuration."""
    map_name: str = "Simple64"
    max_steps: int = 1000
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    realtime: bool = False
    
    # Kernel settings
    kernel_dims: int = 16
    convergence_window: int = 80
    
    # Experiment settings
    goal_interval: int = 100  # Steps between goal injection
    report_interval: int = 100  # Steps between status reports


# ── SC2 Bot with Kernel Integration ──────────────────────────────

class KernelIntegratedBot(BotAI):
    """SC2 bot that uses Substrate Kernel for decisions."""
    
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
        
        # Social/cognitive layers
        self.trust = TrustEvaluationLayer()
        self.communication = CommunicationPolicyLayer(self.trust)
        self.trickster = TricksterStoryTeller()
        self.truce = TruceModeOptimizer()
        
        # State tracking
        self._step = 0
        self._cognitive_states: List[CognitiveState] = []
        self._actions_taken: List[Dict] = []
        self._observations: List[np.ndarray] = []
        
        # Metrics
        self._metrics = {
            "total_actions": 0,
            "attractors_formed": 0,
            "goals_managed": 0,
            "council_reports": 0,
        }
    
    async def on_start(self):
        """Called when game starts."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-001: Kernel Integration Test")
        print(f"{'='*60}")
        print(f"Map: {self.config.map_name}")
        print(f"Race: {self.config.race}")
        print(f"Difficulty: {self.config.difficulty}")
        print(f"Max Steps: {self.config.max_steps}")
        print(f"{'='*60}\n")
        
        # Set initial goals
        self._inject_game_goals()
    
    async def on_step(self, iteration: int):
        """Called each game step."""
        self._step += 1
        
        # 1. Observe game state
        observation = self._observe()
        
        # 2. Encode to 16D vector
        vec = self.encoder.encode(observation)
        self._observations.append(vec.copy())
        
        # 3. Feed to kernel
        kernel_obs = Observation(
            vector=vec.tolist(),
            modality="sc2_game",
            embodiment_id="sc2",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
                "units": len(self.units),
            }
        )
        
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        self._cognitive_states.append(cognitive_state)
        
        # 4. Extract action from kernel
        action = self._extract_action(cognitive_state)
        
        # 5. Execute action
        if action:
            await self._execute_action(action)
            self._actions_taken.append({
                "step": self._step,
                "type": action.get("type", "unknown"),
                "cognitive_energy": cognitive_state.cognitive_energy,
                "n_attractors": cognitive_state.n_attractors,
            })
        
        # 6. Periodic goal injection
        if self._step % self.config.goal_interval == 0:
            self._inject_game_goals()
        
        # 7. Status report
        if self._step % self.config.report_interval == 0:
            self._report_status()
        
        # 8. Check termination
        if self._step >= self.config.max_steps:
            print(f"\n{'='*60}")
            print(f"Experiment Complete: {self._step} steps")
            print(f"{'='*60}")
            self._print_final_report()
    
    def _observe(self) -> Any:
        """Get current game observation."""
        return self.state
    
    def _extract_action(self, cognitive_state: CognitiveState) -> Optional[Dict[str, Any]]:
        """Extract SC2 action from kernel's cognitive state."""
        # Use kernel's action vector if available
        if cognitive_state.action and cognitive_state.action.vector:
            action_vec = np.array(cognitive_state.action.vector)
            
            # Interpret action vector
            # [expand, build_army, defend, attack, scout, hold, ...]
            action_types = [
                ActionType.EXPAND,
                ActionType.BUILD_ARMY,
                ActionType.DEFEND,
                ActionType.ATTACK,
                ActionType.SCOUT,
                ActionType.HOLD,
            ]
            
            # Get action with highest activation
            if len(action_vec) >= len(action_types):
                action_idx = np.argmax(action_vec[:len(action_types)])
                action_type = action_types[action_idx]
                
                # Find target position from action vector
                target = None
                if len(action_vec) > len(action_types):
                    # Use remaining dimensions as target coordinates
                    target = (
                        float(action_vec[len(action_types)] * 100),
                        float(action_vec[len(action_types) + 1] * 100) if len(action_vec) > len(action_types) + 1 else 50.0,
                    )
                
                return {
                    "type": action_type.value,
                    "target": target,
                    "confidence": cognitive_state.action.confidence,
                }
        
        # Default: build army if we have resources
        if self.minerals >= 150:
            return {"type": "build_army"}
        
        return {"type": "hold"}
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute action in SC2."""
        action_type = action.get("type", "hold")
        target = action.get("target")
        
        if action_type == "expand":
            await self._action_expand(target)
        elif action_type == "build_army":
            await self._action_build_army()
        elif action_type == "defend":
            await self._action_defend(target)
        elif action_type == "attack":
            await self._action_attack(target)
        elif action_type == "scout":
            await self._action_scout(target)
        elif action_type == "hold":
            await self._action_hold()
    
    async def _action_expand(self, target=None):
        """Build a new command center."""
        if self.townhalls and self.can_afford(UnitTypeId.COMMANDCENTER):
            location = target or self._find_expansion_location()
            if location:
                from sc2.position import Point2
                if isinstance(location, tuple):
                    location = Point2(location)
                await self.build(UnitTypeId.COMMANDCENTER, near=location)
    
    async def _action_build_army(self):
        """Build military units."""
        production_types = {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}
        for unit in self.units:
            if unit.is_structure and unit.type_id in production_types:
                if unit.is_idle and self.can_afford(UnitTypeId.MARINE):
                    await unit.train(UnitTypeId.MARINE)
    
    async def _action_defend(self, target=None):
        """Move army to defensive position."""
        army = self.units.of_type([
            UnitTypeId.MARINE,
            UnitTypeId.MARAUDER,
        ])
        if army and target:
            from sc2.position import Point2
            if isinstance(target, tuple):
                target = Point2(target)
            army.move(target)
    
    async def _action_attack(self, target=None):
        """Attack position."""
        army = self.units.of_type([
            UnitTypeId.MARINE,
            UnitTypeId.MARAUDER,
        ])
        if army and target:
            from sc2.position import Point2
            if isinstance(target, tuple):
                target = Point2(target)
            army.attack(target)
    
    async def _action_scout(self, target=None):
        """Send a unit to scout."""
        workers = self.units.of_type(UnitTypeId.SCV)
        if workers and target:
            from sc2.position import Point2
            if isinstance(target, tuple):
                target = Point2(target)
            workers.first.move(target)
    
    async def _action_hold(self):
        """Hold current position."""
        army = self.units.of_type([
            UnitTypeId.MARINE,
            UnitTypeId.MARAUDER,
        ])
        if army:
            army.hold_position()
    
    def _find_expansion_location(self):
        """Find a location for expansion."""
        if self.expansion_locations:
            return list(self.expansion_locations.keys())[0]
        return None
    
    def _inject_game_goals(self):
        """Inject game-relevant goals into kernel."""
        # Economy goal
        self.kernel.publish_goal(Goal(
            target=[0.8, 0.2, 0.5, 0.3] + [0.0] * 12,
            priority=0.7,
            description="Expand economy",
            embodiment_id="sc2",
        ))
        
        # Military goal
        self.kernel.publish_goal(Goal(
            target=[0.2, 0.8, 0.5, 0.7] + [0.0] * 12,
            priority=0.6,
            description="Build military",
            embodiment_id="sc2",
        ))
        
        # Scout goal
        self.kernel.publish_goal(Goal(
            target=[0.5, 0.5, 0.8, 0.2] + [0.0] * 12,
            priority=0.4,
            description="Scout opponent",
            embodiment_id="sc2",
        ))
    
    def _report_status(self):
        """Print status report."""
        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active_goals = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]
        
        print(f"\n--- Step {self._step} ---")
        print(f"  Kernel: {self.kernel._tick} ticks, {self.kernel._compute_coherence():.3f} coherence")
        print(f"  Attractors: {len(self.kernel._base_attractors)}")
        print(f"  Executive: {len(active_goals)} active goals")
        print(f"  Game: Minerals={self.minerals}, Units={len(self.units)}")
    
    def _print_final_report(self):
        """Print final experiment report."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-001: Final Report")
        print(f"{'='*60}")
        
        # Kernel metrics
        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active_goals = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]
        
        print(f"\nKernel Metrics:")
        print(f"  Total ticks: {self.kernel._tick}")
        print(f"  Final coherence: {self.kernel._compute_coherence():.3f}")
        print(f"  Attractors formed: {len(self.kernel._base_attractors)}")
        print(f"  Basin balance: {topo.basin_balance:.3f}")
        print(f"  Volume entropy: {topo.volume_entropy:.3f}")
        
        print(f"\nExecutive Metrics:")
        print(f"  Goals managed: {len(active_goals)}")
        print(f"  Total created: {len(goals)}")
        
        print(f"\nGame Metrics:")
        print(f"  Steps completed: {self._step}")
        print(f"  Actions taken: {len(self._actions_taken)}")
        print(f"  Final minerals: {self.minerals}")
        print(f"  Final units: {len(self.units)}")
        
        # Action breakdown
        action_counts = {}
        for a in self._actions_taken:
            action_type = a.get("type", "unknown")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        print(f"\nAction Distribution:")
        for action_type, count in sorted(action_counts.items()):
            print(f"  {action_type}: {count}")
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete")
        print(f"{'='*60}")


# ── Main ─────────────────────────────────────────────────────────

def run_experiment():
    """Run EXP-SC2-001."""
    config = ExperimentConfig(
        map_name="Simple64",
        max_steps=1000,
        difficulty=Difficulty.Easy,
        realtime=False,
    )
    
    bot = KernelIntegratedBot(config)
    
    # Create map settings
    map_path = SC2Path(r'C:\Program Files (x86)\StarCraft II\Maps\Melee\Simple64.SC2Map')
    map_settings = Map(map_path)
    
    # Run game
    result = run_game(
        map_settings=map_settings,
        players=[
            Bot(Race.Terran, bot),
            Computer(Race.Random, config.difficulty),
        ],
        realtime=config.realtime,
    )
    
    return result, bot._metrics


if __name__ == "__main__":
    result, metrics = run_experiment()
    print(f"\nFinal Result: {result}")
    print(f"Metrics: {metrics}")
