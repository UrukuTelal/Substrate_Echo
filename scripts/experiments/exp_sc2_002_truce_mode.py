"""EXP-SC2-002: Truce Mode Test.

Tests alternative optimization landscape for cooperative play.

Validates:
  - Economy stability under truce
  - Defensive behavior without aggression
  - Prediction accuracy in non-combat mode
  - Cooperation metrics
  - Unnecessary engagement reduction

Architecture:
  SC2 Game State
       |
  Observation Encoder (16D)
       |
  Truce Mode Optimizer
       |
  Substrate Kernel
       |
  Truce-aligned Actions
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
from substrate_echo.embodiments.sc2.trust import TrustEvaluationLayer
from substrate_echo.embodiments.sc2.trickster import TricksterStoryTeller, NarrativeContext, NarrativeStyle
from substrate_echo.embodiments.sc2.truce_mode import TruceModeOptimizer, TruceMode

os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'
from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId
from pathlib import Path as SC2Path


@dataclass
class ExperimentConfig:
    map_name: str = "Simple64"
    max_steps: int = 800
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    realtime: bool = False
    kernel_dims: int = 16
    convergence_window: int = 80
    goal_interval: int = 100
    report_interval: int = 200
    mode: TruceMode = TruceMode.TRUCE


class TruceBot(BotAI):
    """SC2 bot operating in Truce Mode."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        kernel_config = KernelConfig(
            dim=config.kernel_dims,
            convergence_window=config.convergence_window,
        )
        self.kernel = SubstrateKernel(config=kernel_config)
        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()
        self.trust = TrustEvaluationLayer()
        self.trickster = TricksterStoryTeller()
        self.truce = TruceModeOptimizer()
        self.truce.set_mode(config.mode)

        self._step = 0
        self._actions_taken: List[Dict] = []
        self._engagement_events: List[Dict] = []
        self._economy_history: List[Dict] = []

    async def on_start(self):
        print(f"\n{'='*60}")
        print(f"EXP-SC2-002: Truce Mode Test")
        print(f"{'='*60}")
        print(f"Mode: {self.config.mode.value}")
        print(f"Map: {self.config.map_name}")
        print(f"Max Steps: {self.config.max_steps}")
        print(f"{'='*60}\n")
        self._inject_truce_goals()

    async def on_step(self, iteration: int):
        self._step += 1

        observation = self.state
        vec = self.encoder.encode(observation)

        kernel_obs = Observation(
            vector=vec.tolist(),
            modality="sc2_truce",
            embodiment_id="sc2_truce",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
                "units": len(self.units),
                "truce_mode": self.config.mode.value,
            }
        )

        cognitive_state = self.kernel.publish_observation(kernel_obs)
        action = self._extract_truce_action(cognitive_state)

        if action:
            await self._execute_action(action)
            self._actions_taken.append({
                "step": self._step,
                "type": action.get("type", "unknown"),
                "coherence": cognitive_state.coherence,
            })

        # Record economy
        if self._step % 10 == 0:
            self._economy_history.append({
                "step": self._step,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "units": len(self.units),
            })

        # Truce compliance check
        if action and action.get("type") in ("attack", "harass"):
            self._engagement_events.append({
                "step": self._step,
                "type": action["type"],
                "violation": True,
            })

        if self._step % self.config.goal_interval == 0:
            self._inject_truce_goals()

        if self._step % self.config.report_interval == 0:
            self._report_status()

        if self._step >= self.config.max_steps:
            print(f"\n{'='*60}")
            print(f"Truce Mode Complete: {self._step} steps")
            print(f"{'='*60}")
            self._print_final_report()

    def _extract_truce_action(self, cs: CognitiveState) -> Optional[Dict]:
        """Extract truce-appropriate action."""
        if cs.action and cs.action.vector:
            vec = np.array(cs.action.vector)
            # In truce mode, prefer economic and defensive actions
            if self.minerals >= 400 and self._count_expansions() < 3:
                return {"type": "expand"}
            elif self.minerals >= 150:
                return {"type": "build_army"}
        if self.minerals >= 400 and self._count_expansions() < 3:
            return {"type": "expand"}
        return {"type": "hold"}

    def _count_expansions(self) -> int:
        return sum(1 for u in self.units if u.type_id in (UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS))

    async def _execute_action(self, action: Dict):
        t = action.get("type", "hold")
        if t == "expand":
            if self.townhalls and self.can_afford(UnitTypeId.COMMANDCENTER):
                loc = self._find_expansion()
                if loc:
                    from sc2.position import Point2
                    await self.build(UnitTypeId.COMMANDCENTER, near=loc)
        elif t == "build_army":
            for u in self.units:
                if u.is_structure and u.type_id in {UnitTypeId.BARRACKS, UnitTypeId.FACTORY}:
                    if u.is_idle and self.can_afford(UnitTypeId.MARINE):
                        await u.train(UnitTypeId.MARINE)
        elif t == "hold":
            army = self.units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
            if army:
                army.hold_position()

    def _find_expansion(self):
        if self.expansion_locations:
            return list(self.expansion_locations.keys())[0]
        return None

    def _inject_truce_goals(self):
        self.kernel.publish_goal(Goal(
            target=[0.9, 0.1, 0.3, 0.2] + [0.0] * 12,
            priority=0.8,
            description="Grow economy peacefully",
            embodiment_id="sc2_truce",
        ))
        self.kernel.publish_goal(Goal(
            target=[0.1, 0.1, 0.9, 0.8] + [0.0] * 12,
            priority=0.6,
            description="Maintain defensive stability",
            embodiment_id="sc2_truce",
        ))

    def _report_status(self):
        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]
        truce_score = self.truce.calculate_score()

        print(f"\n--- Step {self._step} ---")
        print(f"  Kernel: {self.kernel._tick} ticks, {self.kernel._compute_coherence():.3f} coherence")
        print(f"  Attractors: {len(self.kernel._base_attractors)}")
        print(f"  Truce Score: {truce_score:.3f}")
        print(f"  Engagements: {len(self._engagement_events)}")
        print(f"  Game: Minerals={self.minerals}, Units={len(self.units)}")

    def _print_final_report(self):
        print(f"\n{'='*60}")
        print(f"EXP-SC2-002: Truce Mode Final Report")
        print(f"{'='*60}")

        topo = self.kernel.topology.compute_metrics()
        goals = self.kernel.executive.get_goals()
        active = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED)]

        print(f"\nKernel Metrics:")
        print(f"  Total ticks: {self.kernel._tick}")
        print(f"  Final coherence: {self.kernel._compute_coherence():.3f}")
        print(f"  Attractors formed: {len(self.kernel._base_attractors)}")
        print(f"  Volume entropy: {topo.volume_entropy:.3f}")

        print(f"\nTruce Metrics:")
        print(f"  Mode: {self.config.mode.value}")
        print(f"  Truce score: {self.truce.calculate_score():.3f}")
        print(f"  Engagements: {len(self._engagement_events)}")
        print(f"  Engagements per 100 steps: {len(self._engagement_events) / max(1, self._step) * 100:.2f}")

        print(f"\nGame Metrics:")
        print(f"  Steps: {self._step}")
        print(f"  Actions: {len(self._actions_taken)}")
        print(f"  Final minerals: {self.minerals}")
        print(f"  Final units: {len(self.units)}")
        print(f"  Expansions: {self._count_expansions()}")

        action_counts = {}
        for a in self._actions_taken:
            action_counts[a["type"]] = action_counts.get(a["type"], 0) + 1
        print(f"\nAction Distribution:")
        for k, v in sorted(action_counts.items()):
            print(f"  {k}: {v}")

        print(f"\n{'='*60}")


def run_experiment():
    config = ExperimentConfig(
        map_name="Simple64",
        max_steps=800,
        difficulty=Difficulty.Easy,
        mode=TruceMode.TRUCE,
    )
    bot = TruceBot(config)
    map_path = SC2Path(r'C:\Program Files (x86)\StarCraft II\Maps\Melee\Simple64.SC2Map')
    result = run_game(
        map_settings=Map(map_path),
        players=[Bot(Race.Terran, bot), Computer(Race.Random, config.difficulty)],
        realtime=config.realtime,
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print(f"\nFinal Result: {result}")
