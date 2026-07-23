"""EXP-SC2-004: Companion Mode Test.

Tests human-substrate interaction with intent-based commands.

Validates:
  - Intent interpretation from human commands
  - Substrate response to natural language instructions
  - Adaptive behavior based on human feedback
  - Shared understanding between human and substrate
  - Performance under human supervision

Architecture:
  Human Command (Intent)
       |
  Intent Interpreter
       |
  Substrate Kernel (Interpretation)
       |
  Action Execution
       |
  Human Feedback
       |
  Learning Update
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
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


class IntentType(Enum):
    """Types of human intent."""
    ECONOMY = "economy"
    MILITARY = "military"
    DEFENSE = "defense"
    SCOUT = "scout"
    EXPAND = "expand"
    HOLD = "hold"
    ATTACK = "attack"
    RETREAT = "retreat"


@dataclass
class HumanCommand:
    """Simulated human command."""
    intent: IntentType
    priority: float
    description: str
    confidence: float = 0.8
    timestamp: int = 0


@dataclass
class ExperimentConfig:
    """EXP-SC2-004 configuration."""
    map_name: str = "Simple64"
    max_steps: int = 1000
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    realtime: bool = False
    
    kernel_dims: int = 16
    convergence_window: int = 80
    report_interval: int = 100
    
    # Human command simulation
    command_interval: int = 50  # Steps between commands
    command_variance: float = 0.2  # Variance in command priority


# Simulated human commands
COMMAND_SEQUENCE = [
    HumanCommand(IntentType.ECONOMY, 0.8, "Focus on economy"),
    HumanCommand(IntentType.MILITARY, 0.7, "Build army"),
    HumanCommand(IntentType.SCOUT, 0.6, "Scout the map"),
    HumanCommand(IntentType.DEFENSE, 0.75, "Defend base"),
    HumanCommand(IntentType.EXPAND, 0.65, "Expand territory"),
    HumanCommand(IntentType.ATTACK, 0.9, "Attack enemy"),
    HumanCommand(IntentType.HOLD, 0.5, "Hold position"),
    HumanCommand(IntentType.RETREAT, 0.8, "Retreat units"),
]


class CompanionBot(BotAI):
    """SC2 bot for testing human-substrate interaction."""
    
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
        self._command_idx = 0
        self._human_agent = "human_operator"
        
        # Metrics
        self._command_history = []
        self._response_history = []
        self._performance_history = []
        
        # Intent interpretation
        self._intent_weights = {intent: 0.5 for intent in IntentType}
    
    async def on_start(self):
        """Called when game starts."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-004: Companion Mode Test")
        print(f"{'='*60}")
        print(f"Max Steps: {self.config.max_steps}")
        print(f"Command Interval: {self.config.command_interval}")
        print(f"Difficulty: {self.config.difficulty}")
        print(f"{'='*60}\n")
        
        # Register human operator
        self.trust.register_agent(self._human_agent, initial_trust=0.7)
    
    async def on_step(self, iteration: int):
        """Called each game step."""
        self._step += 1
        
        # 1. Process human command if due
        if self._step % self.config.command_interval == 0:
            command = self._get_next_command()
            if command:
                self._process_human_command(command)
        
        # 2. Observe game state
        observation = self._observe()
        
        # 3. Encode to 16D vector
        vec = self.encoder.encode(observation)
        
        # 4. Feed to kernel with intent context
        kernel_obs = Observation(
            vector=vec.tolist(),
            modality="sc2_companion",
            embodiment_id="sc2_companion",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
                "units": len(self.units),
                "intent_weights": dict(self._intent_weights),
            }
        )
        
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        
        # 5. Extract action based on intent
        action = self._extract_companion_action(cognitive_state)
        
        # 6. Execute action
        if action:
            await self._execute_action(action)
            self._response_history.append({
                "step": self._step,
                "action": action.get("type", "unknown"),
                "intent": action.get("intent", "unknown"),
                "confidence": action.get("confidence", 0.0),
            })
        
        # 7. Record performance
        if self._step % 50 == 0:
            self._record_performance()
        
        # 8. Status report
        if self._step % self.config.report_interval == 0:
            self._report_status()
        
        # 9. Check termination
        if self._step >= self.config.max_steps:
            print(f"\n{'='*60}")
            print(f"Companion Mode Complete: {self._step} steps")
            print(f"{'='*60}")
            self._print_final_report()
    
    def _observe(self) -> Any:
        """Get current game observation."""
        return self.state
    
    def _get_next_command(self) -> Optional[HumanCommand]:
        """Get next command from sequence."""
        if self._command_idx >= len(COMMAND_SEQUENCE):
            return None
        
        command = COMMAND_SEQUENCE[self._command_idx]
        self._command_idx += 1
        
        # Add timestamp
        command.timestamp = self._step
        
        # Add variance to priority
        variance = np.random.uniform(-self.config.command_variance, self.config.command_variance)
        command.priority = max(0.0, min(1.0, command.priority + variance))
        
        self._command_history.append(command)
        return command
    
    def _process_human_command(self, command: HumanCommand):
        """Process human command and update intent weights."""
        print(f"\n--- Human Command at Step {self._step} ---")
        print(f"  Intent: {command.intent.value}")
        print(f"  Priority: {command.priority:.2f}")
        print(f"  Description: {command.description}")
        
        # Update intent weights based on command
        for intent in IntentType:
            if intent == command.intent:
                # Increase weight for commanded intent
                self._intent_weights[intent] = min(1.0, self._intent_weights[intent] + 0.2)
            else:
                # Decrease weight for other intents
                self._intent_weights[intent] = max(0.0, self._intent_weights[intent] - 0.05)
        
        # Normalize weights
        total = sum(self._intent_weights.values())
        if total > 0:
            self._intent_weights = {k: v/total for k, v in self._intent_weights.items()}
        
        # Update trust based on command quality
        if command.priority > 0.7:
            self.trust.observe(self._human_agent, {
                "type": "accurate_prediction",
                "outcome": "positive"
            })
        
        print(f"  Updated weights: { {k.value: f'{v:.2f}' for k, v in self._intent_weights.items()} }")
    
    def _extract_companion_action(self, cognitive_state: CognitiveState) -> Optional[Dict[str, Any]]:
        """Extract action based on intent weights."""
        # Get dominant intent
        dominant_intent = max(self._intent_weights.items(), key=lambda x: x[1])
        intent_type, intent_weight = dominant_intent
        
        # Map intent to action type
        intent_to_action = {
            IntentType.ECONOMY: "expand",
            IntentType.MILITARY: "build_army",
            IntentType.DEFENSE: "defend",
            IntentType.SCOUT: "scout",
            IntentType.EXPAND: "expand",
            IntentType.HOLD: "hold",
            IntentType.ATTACK: "attack",
            IntentType.RETREAT: "retreat",
        }
        
        action_type = intent_to_action.get(intent_type, "hold")
        
        # Adjust based on resources
        if action_type == "expand" and self.minerals < 400:
            action_type = "build_army"
        elif action_type == "build_army" and self.minerals < 150:
            action_type = "hold"
        
        return {
            "type": action_type,
            "intent": intent_type.value,
            "confidence": intent_weight,
            "cognitive_energy": cognitive_state.cognitive_energy,
        }
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute action in SC2."""
        action_type = action.get("type", "hold")
        
        if action_type == "expand":
            if self.townhalls and self.can_afford(UnitTypeId.COMMANDCENTER):
                loc = self._find_expansion()
                if loc:
                    from sc2.position import Point2
                    await self.build(UnitTypeId.COMMANDCENTER, near=loc)
        
        elif action_type == "build_army":
            for unit in self.units:
                if unit.is_structure and unit.type_id in {UnitTypeId.BARRACKS, UnitTypeId.FACTORY}:
                    if unit.is_idle and self.can_afford(UnitTypeId.MARINE):
                        await unit.train(UnitTypeId.MARINE)
        
        elif action_type == "defend":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army and self.townhalls:
                army.move(self.townhalls.first.position)
        
        elif action_type == "attack":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army:
                # Attack toward enemy base (approximate)
                attack_point = self._find_enemy_direction()
                if attack_point:
                    from sc2.position import Point2
                    army.attack(Point2(attack_point))
        
        elif action_type == "hold":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army:
                army.hold_position()
        
        elif action_type == "retreat":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army and self.townhalls:
                army.move(self.townhalls.first.position)
    
    def _find_expansion(self):
        """Find expansion location."""
        if self.expansion_locations:
            return list(self.expansion_locations.keys())[0]
        return None
    
    def _find_enemy_direction(self):
        """Find approximate enemy direction."""
        # Simple heuristic: attack toward map center
        return (64.0, 64.0)
    
    def _record_performance(self):
        """Record performance metrics."""
        self._performance_history.append({
            "step": self._step,
            "minerals": self.minerals,
            "vespene": self.vespene,
            "units": len(self.units),
            "supply_used": self.supply_used,
            "supply_cap": self.supply_cap,
            "intent_weights": dict(self._intent_weights),
        })
    
    def _report_status(self):
        """Print status report."""
        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active_goals = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]
        
        dominant_intent = max(self._intent_weights.items(), key=lambda x: x[1])
        
        print(f"\n--- Step {self._step} ---")
        print(f"  Kernel: {self.kernel._tick} ticks, {self.kernel._compute_coherence():.3f} coherence")
        print(f"  Dominant Intent: {dominant_intent[0].value} ({dominant_intent[1]:.2f})")
        print(f"  Trust: {self.trust.get_trust(self._human_agent):.3f}")
        print(f"  Game: Minerals={self.minerals}, Units={len(self.units)}")
        print(f"  Commands Processed: {len(self._command_history)}")
    
    def _print_final_report(self):
        """Print final experiment report."""
        print(f"\n{'='*60}")
        print(f"EXP-SC2-004: Companion Mode Final Report")
        print(f"{'='*60}")
        
        print(f"\nCommand Analysis:")
        print(f"  Total commands: {len(self._command_history)}")
        if self._command_history:
            intent_counts = {}
            for cmd in self._command_history:
                intent_counts[cmd.intent.value] = intent_counts.get(cmd.intent.value, 0) + 1
            print(f"  Command distribution:")
            for intent, count in sorted(intent_counts.items()):
                print(f"    {intent}: {count}")
        
        print(f"\nResponse Analysis:")
        print(f"  Total responses: {len(self._response_history)}")
        if self._response_history:
            action_counts = {}
            for resp in self._response_history:
                action_counts[resp["action"]] = action_counts.get(resp["action"], 0) + 1
            print(f"  Action distribution:")
            for action, count in sorted(action_counts.items()):
                print(f"    {action}: {count}")
        
        print(f"\nIntent Evolution:")
        if self._performance_history:
            first_perf = self._performance_history[0]
            last_perf = self._performance_history[-1]
            print(f"  Initial intent weights: { {k.value: f'{v:.2f}' for k, v in first_perf['intent_weights'].items()} }")
            print(f"  Final intent weights: { {k.value: f'{v:.2f}' for k, v in last_perf['intent_weights'].items()} }")
        
        print(f"\nTrust Dynamics:")
        print(f"  Final human trust: {self.trust.get_trust(self._human_agent):.3f}")
        print(f"  Human trust level: {self.trust.get_level(self._human_agent).name}")
        
        print(f"\nKernel Metrics:")
        print(f"  Total ticks: {self.kernel._tick}")
        print(f"  Final coherence: {self.kernel._compute_coherence():.3f}")
        print(f"  Attractors formed: {len(self.kernel._base_attractors)}")
        
        print(f"\nGame Performance:")
        if self._performance_history:
            final_perf = self._performance_history[-1]
            print(f"  Final minerals: {final_perf['minerals']}")
            print(f"  Final units: {final_perf['units']}")
            print(f"  Final supply: {final_perf['supply_used']}/{final_perf['supply_cap']}")
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete")
        print(f"{'='*60}")


def run_experiment():
    """Run EXP-SC2-004."""
    config = ExperimentConfig(
        map_name="Simple64",
        max_steps=1000,
        difficulty=Difficulty.Easy,
        realtime=False,
    )
    
    bot = CompanionBot(config)
    
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
    
    bot._print_final_report()
    return result


if __name__ == "__main__":
    result = run_experiment()
    print(f"\nFinal Result: {result}")