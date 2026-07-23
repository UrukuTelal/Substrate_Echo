"""Test SC2 Connection — Verify bot can connect and run.

Usage:
    python -m substrate_echo.embodiments.sc2.test_connection
"""
from __future__ import annotations
import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from substrate_echo.embodiments.sc2.sc2_bot import SC2Bot, SC2Config


async def test_connection():
    """Test basic SC2 connection."""
    print("Testing SC2 connection...")
    print(f"SC2 Path: {SC2Config().sc2_path}")

    config = SC2Config(
        map_name="Simple64",
        realtime=False,
    )

    bot = SC2Bot(config)

    print("Starting game...")
    try:
        from sc2.main import run_game
        from sc2.maps import Map
        from sc2.player import Bot, Computer
        from sc2 import Race, Difficulty

        result = await run_game(
            map=Map(config.map_name),
            players=[
                Bot(config.race, bot),
                Computer(Race.Random, Difficulty.Easy),
            ],
            realtime=False,
        )

        print(f"Game finished: {result}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
