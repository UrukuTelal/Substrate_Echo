"""SC2 Bot Adapter — Connects SC2 game to Substrate Kernel.

This is the main bot class that:
1. Connects to SC2 game
2. Observes game state
3. Feeds observations to kernel
4. Receives abstract actions
5. Translates to game commands
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import asyncio

# SC2 imports
from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId

# Substrate imports
from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder, AbstractAction, ActionType


@dataclass
class SC2Config:
    """SC2 bot configuration."""
    map_name: str = "Simple64"
    difficulty: Difficulty = Difficulty.Easy
    race: Race = Race.Terran
    sc2_path: str = r"C:\Program Files (x86)\StarCraft II"
    realtime: bool = False
    max_steps: int = 1000


class SC2Bot(BotAI):
    """StarCraft II bot that uses Substrate Kernel for decisions.

    This bot:
    1. Observes game state via SC2 API
    2. Encodes observations into 16D vectors
    3. Passes vectors to Substrate Kernel
    4. Receives abstract actions
    5. Translates to game commands
    """

    def __init__(self, config: SC2Config):
        super().__init__()
        self.config = config
        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()
        self._step = 0
        self._kernel = None  # SubstrateKernel instance
        self._done = False

    def set_kernel(self, kernel):
        """Set the Substrate Kernel instance."""
        self._kernel = kernel

    async def on_start(self):
        """Called when game starts."""
        print(f"[SC2Bot] Game started on {self.config.map_name}")
        print(f"[SC2Bot] Playing as {self.config.race}")
        print(f"[SC2Bot] Starting workers: {len(self.units)}")

    async def on_step(self, iteration: int):
        """Called each game step."""
        self._step += 1

        # 1. Observe
        observation = self.state

        # 2. Encode
        vec = self.encoder.encode(observation)

        # 3. Feed to kernel (if available)
        if self._kernel is not None:
            action = await self._kernel.observe({
                "raw": observation,
                "vector": vec.tolist(),
                "step": self._step,
                "embodiment": "sc2",
            })
        else:
            # Default action if no kernel
            action = self._default_action(observation)

        # 4. Execute
        if action is not None:
            await self._execute_action(action)

        # 5. Check termination
        if self._step >= self.config.max_steps and not self._done:
            print(f"[SC2Bot] Max steps reached: {self.config.max_steps}")
            self._done = True

    async def _execute_action(self, action: Dict[str, Any]):
        """Execute an action in the game."""
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
        if self.townhalls:
            if self.can_afford(UnitTypeId.COMMANDCENTER):
                location = target or self._find_expansion_location()
                if location:
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
            UnitTypeId.SIEGETANK,
        ])
        if army and target:
            army.move(target)

    async def _action_attack(self, target=None):
        """Attack position."""
        army = self.units.of_type([
            UnitTypeId.MARINE,
            UnitTypeId.MARAUDER,
            UnitTypeId.SIEGETANK,
        ])
        if army and target:
            army.attack(target)

    async def _action_scout(self, target=None):
        """Send a unit to scout."""
        workers = self.units.of_type(UnitTypeId.SCV)
        if workers and target:
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

    def _default_action(self, observation) -> Dict[str, Any]:
        """Default action when no kernel is connected."""
        # Use BotAI attributes directly
        workers = len(self.units.of_type(UnitTypeId.SCV))

        if self.minerals >= 150 and workers < 22:
            return {"type": "build_army"}
        elif self.townhalls:
            townhall = self.townhalls.first
            if townhall.is_idle:
                return {"type": "build_army"}

        return {"type": "hold"}


def run_sc2_bot(config: SC2Config = None) -> Any:
    """Run the SC2 bot with optional kernel integration."""
    if config is None:
        config = SC2Config()

    # Set SC2PATH if not set
    if 'SC2PATH' not in os.environ:
        os.environ['SC2PATH'] = config.sc2_path

    bot = SC2Bot(config)

    # Create map settings
    map_path = Path(config.sc2_path) / "Maps" / "Melee" / f"{config.map_name}.SC2Map"
    map_settings = Map(map_path)

    # Run game
    result = run_game(
        map_settings=map_settings,
        players=[
            Bot(config.race, bot),
            Computer(Race.Random, config.difficulty),
        ],
        realtime=config.realtime,
    )

    return result
