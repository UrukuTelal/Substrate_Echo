"""SC2 Live Game — Runs the substrate kernel against a real SC2 game.

Launches SC2, connects the bot with full kernel integration,
and plays a complete game. Reports cognitive metrics at the end.

Usage:
    python scripts/sc2_live.py
    python scripts/sc2_live.py --realtime
    python scripts/sc2_live.py --steps 2000 --difficulty Hard
"""
from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'

from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId

from substrate_echo.kernel import SubstrateKernel, KernelConfig, Observation, CognitiveState
from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder, AbstractAction, ActionType
from substrate_echo.epistemology.trust import EpistemicTrustSystem
from substrate_echo.epistemology.observatory import EpistemicObservatory
from substrate_echo.epistemology.chain_recorder import EpistemicChainRecorder, AnomalyType
from substrate_echo.epistemology.entity_model import EntityModel, EvidenceType, RelationshipType
from substrate_echo.epistemology.affordance_tracer import AffordanceTracer
from substrate_echo.epistemology.action_bridge import EpistemicActionBridge
from substrate_echo.epistemology.governance_gate import GovernanceGate


# ── Bot ───────────────────────────────────────────────────────────

class LiveBot(BotAI):
    """SC2 bot with full substrate kernel + epistemology integration."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Kernel
        kernel_config = KernelConfig(
            dim=16,
            convergence_window=80,
        )
        self.kernel = SubstrateKernel(config=kernel_config)

        # SC2 components
        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()

        # Epistemology
        self.trust = EpistemicTrustSystem()
        self.observatory = EpistemicObservatory()
        self.chain = EpistemicChainRecorder()
        self.entity_model = EntityModel()
        self.affordance_tracer = AffordanceTracer()
        self.action_bridge = EpistemicActionBridge()
        self.governance_gate = GovernanceGate()

        # State
        self._step = 0
        self._cognitive_states: List[CognitiveState] = []
        self._actions_log: List[Dict] = []
        self._start_time = 0.0
        self._prev_observation: Optional[Dict] = None

    def _validate_kernel(self):
        required = [
            "entity_model",
            "affordance_tracer",
            "action_bridge",
            "governance_gate",
            "chain",
        ]
        missing = [x for x in required if not hasattr(self, x) or getattr(self, x) is None]
        if missing:
            raise RuntimeError(f"Epistemic kernel missing: {missing}")

    def on_start(self):
        self._validate_kernel()
        self._start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  SUBSTRATE ECHO — SC2 LIVE GAME")
        print(f"{'='*60}")
        print(f"  Map:     {self.config['map_name']}")
        print(f"  Race:    Terran")
        print(f"  Enemy:   Random ({self.config['difficulty']})")
        print(f"  Steps:   {self.config['max_steps']}")
        print(f"  Realtime: {self.config['realtime']}")
        print(f"{'='*60}\n")

        # Health check — all epistemic components must be initialized
        components = {
            "Kernel":      self.kernel,
            "EntityModel": self.entity_model,
            "Affordances": self.affordance_tracer,
            "ActionBridge": self.action_bridge,
            "Governance":  self.governance_gate,
            "Recorder":    self.chain,
            "Encoder":     self.encoder,
        }
        all_ok = True
        print("  Epistemic Kernel Health Check:")
        for name, obj in components.items():
            status = "OK" if obj is not None else "MISSING"
            if obj is None:
                all_ok = False
            print(f"    {name:16s} {status}")
        if not all_ok:
            print("\n  WARNING: Some components missing — agent will fallback to default policy")
        print()

        # Inject initial goals
        from substrate_echo.kernel import Goal
        self.kernel.publish_goal(Goal(
            target=[0.0] * 16,
            priority=0.8,
            description="Establish economy",
            embodiment_id="sc2",
        ))

    async def on_step(self, iteration: int):
        self._step += 1

        # ── 1. OBSERVE ──
        # Use encode_from_botai() for own-unit counts (fog-of-war independent)
        raw_vec = self.encoder.encode_from_botai(self)

        # BotAI persistent knowledge for metadata (correct counts)
        workers_botai = len(self.units.of_type(UnitTypeId.SCV))
        army_botai = len(self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE}))
        bases_botai = len(self.units.of_type(UnitTypeId.COMMANDCENTER))

        # Record observation in chain
        self.chain.record_observation(
            tick=self._step,
            raw_state={
                "minerals": self.minerals,
                "workers": workers_botai,
                "army": army_botai,
                "bases": bases_botai,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
            },
            encoded_vector=raw_vec.tolist(),
            source="botai",
        )

        # ── 2. LAYERED TELEMETRY: detect observation mismatch ──
        encoder_workers = self.encoder.economy.workers
        if encoder_workers != workers_botai:
            self.chain.anomalies.append((
                self._step,
                AnomalyType.OBSERVATION_GAP,
                f"Encoder sees {encoder_workers} workers, BotAI has {workers_botai}"
            ))

        # ── 3. FEED TO KERNEL ──
        kernel_obs = Observation(
            vector=raw_vec.tolist(),
            modality="sc2_game",
            embodiment_id="sc2",
            timestamp=time.time(),
            metadata={
                "step": self._step,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
                "workers": workers_botai,
                "army": army_botai,
                "bases": bases_botai,
            }
        )

        cognitive_state = self.kernel.publish_observation(kernel_obs)
        self._cognitive_states.append(cognitive_state)

        # ── 4. ENTITY MODEL: track enemy ──
        if self._step == 1:
            self.entity_model.create_entity("enemy", embodiment="sc2")

        # Simple enemy observation: if we see enemy units
        if hasattr(self, 'enemy_units') and self.enemy_units:
            enemy = self.entity_model.get_entity("enemy")
            if enemy:
                enemy.add_evidence(
                    EvidenceType.OBSERVED_BEHAVIOR,
                    f"Enemy units visible: {len(self.enemy_units)}",
                    tick=self._step,
                )

        # ── 5. ACTION SELECTION ──
        action = self._interpret_action(cognitive_state)
        action_type = action.get("type", "hold") if action else "hold"

        # Record action in chain
        self.chain.record_action(
            tick=self._step,
            action_type=action_type,
            action_vector=cognitive_state.action.vector if cognitive_state.action else [],
            decision_source="interpret_action",
        )

        if action:
            await self._execute(action)

        # ── 6. OUTCOME RECORDING ──
        if self._prev_observation is not None:
            self.chain.record_outcome(
                tick=self._step,
                actual_state={
                    "minerals": self.minerals,
                    "workers": workers_botai,
                    "army": army_botai,
                    "bases": bases_botai,
                    "supply_used": self.supply_used,
                },
            )

        self._prev_observation = {
            "minerals": self.minerals,
            "workers": workers_botai,
            "army": army_botai,
        }

        # ── 7. PERIODIC GOALS ──
        if self._step % 200 == 0:
            self._inject_goal()

        # ── 8. PERIODIC ENTITY UPDATE ──
        if self._step % 100 == 0:
            self.entity_model.update_all(self._step)

        # ── 9. STATUS ──
        if self._step % 500 == 0:
            self._status()

        # ── 10. OBSERVATORY ──
        if self._step % 100 == 0:
            self.observatory.record_observation(self._step, {
                "step": self._step,
                "minerals": self.minerals,
                "army": army_botai,
                "coherence": cognitive_state.coherence,
                "n_attractors": cognitive_state.n_attractors,
                "anomalies": len(self.chain.anomalies),
            })

    def _interpret_action(self, cs: CognitiveState) -> Optional[Dict[str, Any]]:
        """Convert kernel action vector to SC2 action."""
        if not cs.action or not cs.action.vector:
            return self._fallback_action()

        vec = np.array(cs.action.vector)
        # Use highest-scoring dimension from the action vector
        # Dimensions map to: expand, build_army, defend, attack, scout, hold
        labels = ["expand", "build_army", "defend", "attack", "scout", "hold"]

        if len(vec) >= len(labels):
            best_idx = int(np.argmax(vec[:len(labels)]))
            best_score = vec[best_idx]
        elif len(vec) > 0:
            best_idx = int(np.argmax(vec))
            best_score = vec[best_idx]
        else:
            return self._fallback_action()

        # Use fallback if kernel output is too uncertain
        if best_score < 0.1:
            return self._fallback_action()

        return {"type": labels[best_idx]}

    def _fallback_action(self) -> Dict[str, Any]:
        """Simple rule-based fallback."""
        workers = len(self.units.of_type(UnitTypeId.SCV))
        army = len(self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE}))
        bases = len(self.units.of_type(UnitTypeId.COMMANDCENTER))

        if not self.townhalls:
            return {"type": "hold"}

        if self.supply_used >= self.supply_cap - 2 and self.minerals >= 100:
            return {"type": "build_army"}
        elif self.townhalls.first.is_idle and self.minerals >= 50:
            return {"type": "build_army"}
        elif bases < 2 and self.minerals >= 400 and workers >= 14:
            return {"type": "expand"}
        elif army >= 20:
            return {"type": "attack"}
        return {"type": "hold"}

    async def _execute(self, action: Dict[str, Any]):
        """Execute a decoded action in-game."""
        action_type = action.get("type", "hold")

        if action_type == "expand":
            await self._do_expand()
        elif action_type == "build_army":
            await self._do_build_army()
        elif action_type == "defend":
            await self._do_defend()
        elif action_type == "attack":
            await self._do_attack()
        elif action_type == "scout":
            await self._do_scout()
        # hold = do nothing

        self._actions_log.append({"step": self._step, "type": action_type})

    async def _do_expand(self):
        if self.townhalls and self.can_afford(UnitTypeId.COMMANDCENTER):
            loc = self._find_expansion()
            if loc:
                await self.build(UnitTypeId.COMMANDCENTER, near=loc)

    async def _do_build_army(self):
        if not self.townhalls:
            return

        # Build SCVs if needed
        workers = self.units.of_type(UnitTypeId.SCV)
        if len(workers) < 22:
            tc = self.townhalls.first
            if tc.is_idle and self.can_afford(UnitTypeId.SCV):
                tc.train(UnitTypeId.SCV)
                return

        # Build supply if needed
        if self.supply_used >= self.supply_cap - 3:
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first.position)
                return

        # Build barracks if none
        barracks = self.units.of_type(UnitTypeId.BARRACKS)
        if not barracks and self.can_afford(UnitTypeId.BARRACKS):
            await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first.position)
            return

        # Train marines
        for b in barracks:
            if b.is_idle and self.can_afford(UnitTypeId.MARINE):
                b.train(UnitTypeId.MARINE)

    async def _do_defend(self):
        army = self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE})
        if army and self.townhalls:
            army.move(self.townhalls.first.position)

    async def _do_attack(self):
        army = self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE})
        if army and self.enemy_start_locations:
            army.attack(self.enemy_start_locations[0])

    async def _do_scout(self):
        scvs = self.units.of_type(UnitTypeId.SCV)
        if scvs and self.enemy_start_locations:
            scvs.first.move(self.enemy_start_locations[0])

    def _find_expansion(self):
        if self.expansion_locations:
            taken = set()
            for th in self.townhalls:
                taken.add(th.position)
            for loc in self.expansion_locations:
                if all(loc.distance_to(t) > 10 for t in taken):
                    return loc
        return None

    def _inject_goal(self):
        from substrate_echo.kernel import Goal
        workers = len(self.units.of_type(UnitTypeId.SCV))
        army = len(self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE}))

        if army < 10:
            desc = "Build military forces"
            pri = 0.7
        elif workers < 30:
            desc = "Expand economy"
            pri = 0.6
        else:
            desc = "Seek engagement"
            pri = 0.8

        self.kernel.publish_goal(Goal(
            target=[0.0] * 16,
            priority=pri,
            description=desc,
            embodiment_id="sc2",
        ))

    def _status(self):
        elapsed = time.time() - self._start_time
        workers = len(self.units.of_type(UnitTypeId.SCV))
        army = len(self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE}))
        bases = len(self.units.of_type(UnitTypeId.COMMANDCENTER))
        cs = self._cognitive_states[-1] if self._cognitive_states else None

        print(f"  [Step {self._step:5d}] "
              f"Workers={workers:3d} Army={army:3d} Bases={bases} "
              f"Minerals={self.minerals:5d} Gas={self.vespene:5d} "
              f"Supply={self.supply_used}/{self.supply_cap} "
              f"| Coherence={cs.coherence:.3f} "
              f"Attractors={cs.n_attractors} "
              f"Energy={cs.cognitive_energy:.3f} "
              f"| {elapsed:.1f}s elapsed")

    def on_end(self, game_result):
        elapsed = time.time() - self._start_time
        workers = len(self.units.of_type(UnitTypeId.SCV))
        army = len(self.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE}))
        bases = len(self.units.of_type(UnitTypeId.COMMANDCENTER))

        print(f"\n{'='*60}")
        print(f"  GAME OVER -- {game_result}")
        print(f"{'='*60}")
        print(f"  Duration:    {elapsed:.1f}s ({self._step} steps)")
        print(f"  Workers:     {workers}")
        print(f"  Army:        {army}")
        print(f"  Bases:       {bases}")
        print(f"  Minerals:    {self.minerals}")
        print(f"  Gas:         {self.vespene}")
        print(f"  Supply:      {self.supply_used}/{self.supply_cap}")
        print()

        # Kernel metrics
        abs_summary = self.kernel.abstraction.summary()
        print(f"\n  Kernel Metrics:")
        print(f"    Total ticks:        {self.kernel._tick}")
        print(f"    Attractors formed:  {len(self.kernel._base_attractors)}")
        print(f"    Meta-attractors:    {abs_summary.get('n_meta', 0)}")
        topo = self.kernel.topology.compute_metrics()
        print(f"    Mean depth:         {topo.mean_depth:.4f}")
        print(f"    Volume entropy:     {topo.volume_entropy:.4f}")
        print(f"    Basin balance:      {topo.basin_balance:.4f}")

        # Action distribution
        action_counts = {}
        for a in self._actions_log:
            t = a["type"]
            action_counts[t] = action_counts.get(t, 0) + 1
        print(f"\n  Action Distribution:")
        for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / max(1, len(self._actions_log)) * 100
            print(f"    {action_type:12s}: {count:4d} ({pct:.1f}%)")

        # Epistemology
        rules = self.kernel.rule_engine._rules if hasattr(self.kernel.rule_engine, '_rules') else []
        hyps = self.kernel.hypothesis_space._hypotheses if hasattr(self.kernel.hypothesis_space, '_hypotheses') else []
        preds = self.kernel.prediction_memory._records if hasattr(self.kernel.prediction_memory, '_records') else []
        print(f"\n  Epistemology:")
        print(f"    Hypotheses:  {len(hyps)}")
        print(f"    Predictions: {len(preds)}")
        print(f"    Rules:       {len(rules)}")

        # ── EPISTEMIC CHAIN DIAGNOSTICS ──
        print(f"\n{'='*60}")
        print(f"  EPISTEMIC CHAIN DIAGNOSTICS")
        print(f"{'='*60}")
        print(self.chain.render_summary())

        # ── ENTITY MODEL ──
        print()
        print(self.entity_model.render())

        # ── OBSERVATION GAP DETECTION ──
        gaps = [
            (t, desc) for t, atype, desc in self.chain.detect_anomalies()
            if atype == AnomalyType.OBSERVATION_GAP
        ]
        if gaps:
            print(f"\n  Observation Gaps Detected: {len(gaps)}")
            print("  " + "-" * 56)
            for tick, desc in gaps[:10]:
                print(f"    Tick {tick:5d}: {desc}")

        # ── ACTION DEGENERACY ──
        degenerate = [
            (t, desc) for t, atype, desc in self.chain.detect_anomalies()
            if atype == AnomalyType.ACTION_DEGENERATE
        ]
        if degenerate:
            print(f"\n  Action Degeneracy Detected: {len(degenerate)} instances")
            print("  " + "-" * 56)
            for tick, desc in degenerate[:5]:
                print(f"    Tick {tick:5d}: {desc}")

        print(f"\n{'='*60}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Substrate Echo — SC2 Live Game")
    parser.add_argument("--steps", type=int, default=2000, help="Max game steps")
    parser.add_argument("--realtime", action="store_true", help="Run in realtime")
    parser.add_argument("--difficulty", default="Easy",
                        choices=["Easy", "Medium", "Hard", "VeryHard", "CheatVision",
                                 "CheatMoney", "CheatInsane"],
                        help="AI difficulty")
    parser.add_argument("--map", default="Simple64", help="Map name")
    parser.add_argument("--race", default="Terran",
                        choices=["Terran", "Zerg", "Protoss"],
                        help="Player race")
    args = parser.parse_args()

    difficulty_map = {
        "Easy": Difficulty.Easy,
        "Medium": Difficulty.Medium,
        "Hard": Difficulty.Hard,
        "VeryHard": Difficulty.VeryHard,
        "CheatVision": Difficulty.CheatVision,
        "CheatMoney": Difficulty.CheatMoney,
        "CheatInsane": Difficulty.CheatInsane,
    }
    race_map = {
        "Terran": Race.Terran,
        "Zerg": Race.Zerg,
        "Protoss": Race.Protoss,
    }

    config = {
        "map_name": args.map,
        "max_steps": args.steps,
        "realtime": args.realtime,
        "difficulty": args.difficulty,
    }

    bot = LiveBot(config)

    map_path = Path(os.environ['SC2PATH']) / "Maps" / "Melee" / f"{args.map}.SC2Map"
    map_settings = Map(map_path)

    print(f"\nLaunching SC2...")
    print(f"  Map:      {map_path}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Steps:    {args.steps}")
    print(f"  Realtime: {args.realtime}")
    print()

    result = run_game(
        map_settings=map_settings,
        players=[
            Bot(race_map[args.race], bot),
            Computer(Race.Random, difficulty_map[args.difficulty]),
        ],
        realtime=args.realtime,
    )

    print(f"\nResult: {result}")
    return result


if __name__ == "__main__":
    main()
