"""SC2 Iterate — Multi-run iteration loop with learning persistence.

Runs the bot N times, logging metrics between runs.
The AffordanceModel persists between runs via JSON,
so the bot accumulates knowledge across games.

Usage:
    python scripts/sc2_iterate.py --games 3
    python scripts/sc2_iterate.py --games 5 --steps 2000
    python scripts/sc2_iterate.py --games 9999 --infinite
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'

from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import AbilityId

from substrate_echo.kernel import SubstrateKernel, KernelConfig, Observation, CognitiveState
from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder
from substrate_echo.epistemology.trust import EpistemicTrustSystem
from substrate_echo.epistemology.observatory import EpistemicObservatory
from substrate_echo.epistemology.chain_recorder import EpistemicChainRecorder, AnomalyType
from substrate_echo.epistemology.entity_model import EntityModel, EvidenceType
from substrate_echo.epistemology.affordance_tracer import AffordanceTracer, WorldState
from substrate_echo.epistemology.action_bridge import EpistemicActionBridge
from substrate_echo.epistemology.governance_gate import GovernanceGate, GovernanceDecision
from substrate_echo.epistemology.drives import DriveManager, NeedType
from substrate_echo.epistemology.affordance_model import AffordanceModelPool
from substrate_echo.epistemology.game_state_perceiver import GameStatePerceiver, PerceivedEvent, PerceivedEventType
from substrate_echo.epistemology.replay_parser import ReplayParser, ReplayLearningIntegrator, load_and_learn_from_replays, SC2READER_AVAILABLE, ParsedReplay


MODEL_PATH = str(Path(__file__).parent.parent / "data" / "affordance_models.json")
LOG_PATH = str(Path(__file__).parent.parent / "data" / "iteration_log.jsonl")


class IterateBot(BotAI):
    """SC2 bot for iteration runs. Same architecture as LiveBot."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        kernel_config = KernelConfig(dim=16, convergence_window=80)
        self.kernel = SubstrateKernel(config=kernel_config)

        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()

        self.trust = EpistemicTrustSystem()
        self.observatory = EpistemicObservatory()
        self.chain = EpistemicChainRecorder()
        self.entity_model = EntityModel()
        self.affordance_tracer = AffordanceTracer()
        self.action_bridge = EpistemicActionBridge()
        self.governance_gate = GovernanceGate()

        self.drives = DriveManager()
        self.drives.add_need(NeedType.MINERALS, initial=0.18, target=0.72, weight=0.89)
        self.drives.add_need(NeedType.GAS, initial=0.10, target=0.72, weight=0.85)
        self.drives.add_need(NeedType.SUPPLY, initial=0.16, target=0.72, weight=0.86)
        self.drives.add_need(NeedType.MILITARY, initial=0.12, target=0.70, weight=1.0)
        self.drives.add_need(NeedType.INTEL, initial=0.17, target=0.72, weight=0.72)
        self.drives.add_need(NeedType.DEFENSE, initial=0.15, target=0.72, weight=0.9)
        self.drives.add_need(NeedType.EXPANSION, initial=0.14, target=0.72, weight=0.6)
        self.drives.add_need(NeedType.TECHNOLOGY, initial=0.11, target=0.72, weight=0.5)

        action_types = [
            "expand", "build_army", "build_economy", "defend",
            "attack", "scout", "tech_up", "retreat", "hold",
        ]
        need_types = [nt.value for nt in NeedType]
        self.model_pool = AffordanceModelPool(action_types, need_types)

        self.perceiver = GameStatePerceiver()
        self._my_player_id = None
        self._chat_log: List[Dict] = []
        self._last_failed_ability = None
        self._last_failed_tick = 0

        self._step = 0
        self._cognitive_states: List[CognitiveState] = []
        self._actions_log: List[Dict] = []
        self._start_time = 0.0
        self._prev_observation = None
        self._prev_deficits = None
        self._prev_action = None
        self._cached_caps = []
        self._caps_unit_count = 0
        self._caps_refresh_tick = 0
        # Income tracking for empirical diminishing returns
        self._prev_minerals = 0
        self._prev_worker_count = 0
        self._income_history: List[float] = []  # minerals per worker per tick
        self._diminishing_returns = False
        # Zerg-specific tracking
        self._drone_to_geyser: Dict[int, Point2] = {}  # drone tag -> geyser position
        # Track larvae that are currently morphing (tag -> step commanded)
        self._morphing_larvae: Dict[int, int] = {}
        # Action delay timer (learnable by AI) - prevents action spam
        self._action_delay = 3  # ticks between actions (baseline 3, AI can adjust)
        self._last_action_tick = 0
        self._action_delay_history: List[int] = []  # track delays tried
        self._delay_performance: Dict[int, List[float]] = {}  # delay -> performance scores

        if Path(MODEL_PATH).exists():
            self.model_pool.load(MODEL_PATH)

    def on_start(self):
        self._start_time = time.time()
        self._race = Race.Zerg  # Force Zerg for testing
        self._my_player_id = self.state.common.player_id

    async def on_step(self, iteration: int):
        self._step += 1

        workers = len(self.workers)
        army = len(self.units)  # all mobile units — workers can fight too
        bases = len(self.townhalls)

        raw_vec = self.encoder.encode_from_botai(self)

        kernel_obs = Observation(
            vector=raw_vec.tolist(),
            modality="sc2_game",
            embodiment_id="sc2",
            timestamp=time.time(),
            metadata={"step": self._step, "minerals": self.minerals,
                      "workers": workers, "army": army, "bases": bases,
                      "supply_used": self.supply_used, "supply_cap": self.supply_cap},
        )
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        self._cognitive_states.append(cognitive_state)

        if self._step == 1:
            self.entity_model.create_entity("enemy", embodiment="sc2")
        enemy = self.entity_model.get_entity("enemy")
        if enemy:
            enemy_count = len(self.known_enemy_units)
            if enemy_count > 0:
                enemy.add_evidence(EvidenceType.OBSERVED_BEHAVIOR,
                                   f"Enemy units: {enemy_count}", tick=self._step)

        action = self._interpret_action()
        action_type = action.get("type", "hold") if action else "hold"

        self.chain.record_observation(tick=self._step,
            raw_state={"minerals": self.minerals, "workers": workers,
                       "army": army, "bases": bases},
            encoded_vector=raw_vec.tolist(), source="botai")

        self.chain.record_action(tick=self._step, action_type=action_type,
            action_vector=[], decision_source="interpret_action")

        if action:
            await self._execute(action)

        # Perceive text feedback from the game (chat, alerts, errors)
        events = self.perceiver.perceive(self.state, tick=self._step)
        for ev in events:
            self.chain.record_observation(tick=self._step,
                raw_state={"perceived_text": ev.text, "event_type": ev.event_type.value},
                encoded_vector=[], source="game_perceiver")
            if ev.event_type == PerceivedEventType.CHAT and ev.source_id != self._my_player_id:
                await self._respond_to_chat(ev)
        # Reset alert cache every 200 ticks so recurring alerts fire again
        if self._step % 200 == 0:
            self.perceiver.clear_alert_cache()

        if self._prev_observation is not None:
            self.chain.record_outcome(tick=self._step,
                actual_state={"minerals": self.minerals, "workers": workers,
                              "army": army, "bases": bases,
                              "supply_used": self.supply_used})

        self._prev_observation = {"minerals": self.minerals, "workers": workers, "army": army}

        self._update_drives(workers, army, bases)

        if self._prev_action and self._prev_deficits:
            current_deficits = self.drives.deficits()
            current_deficit_str = {k.value: v for k, v in current_deficits.items()}
            observed_delta = {}
            for need_str in current_deficit_str:
                prev_def = self._prev_deficits.get(need_str, 0.0)
                curr_def = current_deficit_str.get(need_str, 0.0)
                observed_delta[need_str] = prev_def - curr_def
            self.model_pool.update(self._prev_action, observed_delta, learning_rate=0.05)

        if self._step % 500 == 0:
            self.model_pool.save(MODEL_PATH)

    def _update_drives(self, workers, army, bases):
        # Mineral income: workers NOT on gas
        gas_workers = 0
        for u in self.workers:
            try:
                if u.orders and any(
                    "EXTRACTOR" in str(o.ability) or
                    "ASSIMILATOR" in str(o.ability) or
                    "REFINERY" in str(o.ability)
                    for o in u.orders):
                    gas_workers += 1
            except (KeyError, Exception):
                pass
        mineral_workers = max(0, workers - gas_workers)
        
        # Optimal saturation per base: 8 mineral fields * 2 workers + 2 gas geysers * 3 workers
        optimal_mineral_workers = bases * 16  # 8 patches * 2 workers each
        optimal_gas_workers = bases * 6       # 2 geysers * 3 workers each
        
        minerals = min(1.0, mineral_workers / max(1, optimal_mineral_workers))
        gas = min(1.0, gas_workers / max(1, optimal_gas_workers))
        supply = (self.supply_cap - self.supply_used) / max(1, self.supply_cap) if self.supply_cap > 0 else 0.5
        mil = min(1.0, army / 20.0)
        enemy_count = len(self.known_enemy_units)
        enemy_structs = len(self.known_enemy_structures)
        intel = min(1.0, (enemy_count + enemy_structs * 2) / 10.0)
        defense = min(1.0, army / max(1, bases * 3))
        expansion = min(1.0, bases / 3.0)
        townhall_ids = {s.type_id for s in self.townhalls}
        prod = sum(1 for s in self.units.structure if s.type_id not in townhall_ids)
        tech = min(1.0, prod / 3.0)

        self.drives.update({
            NeedType.MINERALS: minerals, NeedType.GAS: gas,
            NeedType.SUPPLY: supply, NeedType.MILITARY: mil,
            NeedType.INTEL: intel, NeedType.DEFENSE: defense,
            NeedType.EXPANSION: expansion, NeedType.TECHNOLOGY: tech,
        }, tick=self._step)

    def _interpret_action(self):
        world = WorldState.from_botai(self, tick=self._step)
        enemy_entity = self.entity_model.get_entity("enemy")
        entity_confidence = 0.5
        threat_level = 0.0
        if enemy_entity:
            _, conf = enemy_entity.get_dominant_relationship()
            entity_confidence = conf
            threat_level = enemy_entity.get_threat_level()

        deficits = self.drives.deficits()
        deficit_str = {k.value: v for k, v in deficits.items()}

        candidates = self.affordance_tracer.generate(world, entities=None)
        if not candidates:
            return self._fallback_action()

        need_weights = {nt.value: w for nt, w in [
            (NeedType.MINERALS, 1.2), (NeedType.GAS, 0.8), (NeedType.SUPPLY, 0.8),
            (NeedType.MILITARY, 1.0), (NeedType.INTEL, 0.7),
            (NeedType.DEFENSE, 0.9), (NeedType.EXPANSION, 0.6),
            (NeedType.TECHNOLOGY, 0.5)]}

        scored = []
        for c in candidates:
            at = c.action_type.value
            model = self.model_pool.get_model(at)
            c.need_affinities = dict(model.predicted_delta)
            action_score = self.action_bridge.score_action(
                c, entity_confidence=entity_confidence,
                prediction_accuracy=self.action_bridge.prediction_accuracy,
                drive_utility=c.drive_utility(deficit_str))
            info_gain = model.information_gain()
            action_score.final_score += info_gain * self.model_pool.epsilon * 30.0
            scored.append((c, action_score))

        scored.sort(key=lambda x: x[1].final_score, reverse=True)

        if self.model_pool.should_explore():
            rand_action = self.model_pool.random_action()
            for c, score in scored:
                if c.action_type.value == rand_action:
                    best_candidate, best_score_obj = c, score
                    break
            else:
                best_candidate, best_score_obj = scored[0]
        else:
            best_candidate, best_score_obj = scored[0]

        army_count = len(self.units)  # workers count as army
        total_units = max(1, len(self.units))
        army_exposure = army_count / total_units

        verdict = self.governance_gate.check(
            action_type=best_candidate.action_type.value,
            confidence=entity_confidence,
            cost_level=best_candidate.cost_level.value,
            uncertainty=world.uncertainty,
            army_exposure=army_exposure,
            threat_level=threat_level,
            deficits=deficit_str)

        if verdict.decision == GovernanceDecision.DENY:
            action_type = "hold"
        elif verdict.decision == GovernanceDecision.MODIFY:
            action_type = verdict.adjusted_action or "hold"
        else:
            action_type = best_candidate.action_type.value

        self._prev_deficits = dict(deficit_str)
        self._prev_action = action_type
        return {"type": action_type}

    def _fallback_action(self):
        if not self.townhalls:
            return {"type": "hold"}
        if self.supply_used >= self.supply_cap - 2 and self.minerals >= 100:
            return {"type": "build_army"}
        elif self.townhalls.first.is_idle and self.minerals >= 50:
            return {"type": "build_army"}
        return {"type": "hold"}

    async def _discover_capabilities(self):
        """Query SC2 for what each unit can actually do.

        Caches results — only refreshes every 100 ticks or when unit count changes.
        """
        current_count = len(self.units)
        ticks_since_refresh = self._step - self._caps_refresh_tick
        if (current_count == self._caps_unit_count
                and ticks_since_refresh < 100
                and self._cached_caps):
            return self._cached_caps
        self._caps_unit_count = current_count
        self._caps_refresh_tick = self._step
        caps = []
        for unit in self.units:
            try:
                abilities = await self.get_available_abilities(unit)
                for ability in abilities:
                    # Skip MORPH abilities (supply depot lower/raise spam)
                    if "MORPH" in ability.name:
                        continue
                    caps.append((unit, ability))
            except (ValueError, Exception):
                pass
        self._cached_caps = caps
        return caps

    def _find_capable_unit(self, caps, ability_tag):
        """Find a unit that has a specific ability."""
        for unit, ability in caps:
            if ability == ability_tag:
                return unit
        return None

    async def _execute(self, action):
        """Execute by discovering capabilities, not hardcoding unit types."""
        action_type = action.get("type", "hold")

        if action_type == "hold":
            return

        # Action delay - prevent action spam
        if self._step - self._last_action_tick < self._action_delay:
            return
        self._last_action_tick = self._step

        caps = await self._discover_capabilities()

        # Engagement override: if army is strong enough, attack periodically
        worker_ids = {u.tag for u in self.workers}
        supply_ids = {u.tag for u in self.units if any(kw in u.name.upper() for kw in ("OVERLORD", "OVERSEER", "OBSERVER"))}
        army_units = [u for u in self.units
                      if u.tag not in worker_ids and u.tag not in supply_ids
                      and not u.is_structure and u.can_attack]
        
        # Defensive attack: if enemy units visible near our base, attack regardless of army size
        # Filter out spawned units (locusts, broodlings, interceptors)
        enemy_nearby = any(
            eu.position.distance_to(self.townhalls.first.position) < 40
            and eu.type_id.name not in ("LOCUST", "BROODLING", "INTERCEPTOR", "AUTOTURRET")
            for eu in self.known_enemy_units
        ) if self.townhalls else False

        if (len(army_units) >= 8
                and self.enemy_start_locations
                and self._step % 300 < 50):
            action_type = "attack"
        elif enemy_nearby and army_units:
            action_type = "defensive_attack"

        if action_type in ("build_economy", "build_army"):
            await self._exec_train(caps)
        elif action_type in ("attack", "defensive_attack"):
            await self._exec_attack(caps)
        elif action_type == "defend":
            await self._exec_defend(caps)
        elif action_type == "scout":
            await self._exec_scout(caps)
        elif action_type == "expand":
            await self._exec_expand(caps)
        elif action_type == "retreat":
            await self._exec_defend(caps)

        self._actions_log.append({"step": self._step, "type": action_type})

    async def _exec_train(self, caps):
        """Discover and use production/building capabilities.

        All decisions based on discovered abilities, not hardcoded types.
        Skips abilities that recently failed to avoid spam.
        """
        # Skip abilities that failed within the last 50 ticks
        skip = self._last_failed_ability if (self._step - self._last_failed_tick < 50) else None

        # 0. Empirical diminishing returns — track income per worker
        if self._prev_worker_count > 0 and self._step > 10:
            mineral_delta = self.minerals - self._prev_minerals
            workers_now = len(self.workers)
            if workers_now > 0:
                income_per_worker = mineral_delta / workers_now
                self._income_history.append(income_per_worker)
                # Keep last 50 observations
                if len(self._income_history) > 50:
                    self._income_history.pop(0)
                # Detect diminishing returns: compare recent avg to earlier avg
                if len(self._income_history) >= 20:
                    recent = sum(self._income_history[-10:]) / 10
                    earlier = sum(self._income_history[-20:-10]) / 10
                    # If income per worker dropped >30%, we have diminishing returns
                    if earlier > 0 and recent < earlier * 0.7:
                        self._diminishing_returns = True
                    elif recent > earlier * 0.9:
                        self._diminishing_returns = False
        self._prev_minerals = self.minerals
        self._prev_worker_count = len(self.workers)

        # Reset diminishing returns if we have very few workers (need to rebuild economy)
        if len(self.workers) < 10:
            self._diminishing_returns = False

        # If diminishing returns, train army instead of workers
        if self._diminishing_returns and self.workers:
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                if "TRAIN" in aname or "MORPH" in aname:
                    if not any(kw in aname for kw in ("DRONE", "SCV", "PROBE")):
                        if unit.is_idle and self.can_afford(ability):
                            await self.do(unit(ability))
                            return
        elif self.workers and self.townhalls:
            # Still need workers — train one
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                if any(kw in aname for kw in ("TRAIN", "MORPH")) and any(
                        kw in aname for kw in ("DRONE", "SCV", "PROBE")):
                    if unit.is_idle and self.can_afford(ability):
                        await self.do(unit(ability))
                        return

        # 1. Supply blocked — discover any supply-building ability (skip MORPH)
        if self.supply_used >= self.supply_cap - 3 and self.minerals >= 100 and self.townhalls:
            th_pos = self.townhalls.first.position
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                # Skip morph abilities (supply depot lower/raise)
                if "MORPH" in aname:
                    continue
                # Any ability with supply-increasing keywords
                if any(kw in aname for kw in ("SUPPLY", "OVERLORD", "PYLON", "DEPOT")):
                    if self.can_afford(ability):
                        try:
                            pos = await self.find_placement(ability, near=th_pos, max_distance=20)
                            if pos:
                                await self.do(unit(ability, target=pos))
                                return
                        except (KeyError, Exception):
                            pass

        # 2. No production buildings — discover and build one
        townhall_ids = {s.type_id for s in self.townhalls}
        has_prod = any(s.type_id not in townhall_ids for s in self.units.structure)
        if not has_prod and self.minerals >= 150 and self.townhalls:
            th_pos = self.townhalls.first.position
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                # Any BUILD ability that creates a non-townhall structure
                if "BUILD" in aname and unit in self.workers:
                    if self.can_afford(ability):
                        try:
                            pos = await self.find_placement(ability, near=th_pos, max_distance=20)
                            if pos:
                                await self.do(unit(ability, target=pos))
                                return
                        except (KeyError, Exception):
                            pass

# 3. Need gas — try to build extractor (drone must be on geyser first)
        gas_structures = self.units.structure.filter(
            lambda s: "EXTRACTOR" in s.name.upper())
        gas_count = len(gas_structures)
        gas_needed = min(len(self.workers) // 3, 2) if self.townhalls else 0
        if gas_count < gas_needed and self.minerals >= 75:
            th_pos = self.townhalls.first.position
            enemy_pos = self.enemy_start_locations[0] if self.enemy_start_locations else None
            best_geyser = None
            best_dist = float('inf')
            for g in self.state.vespene_geyser:
                d = g.position.distance_to(th_pos)
                if d >= 20:
                    continue
                if not self.is_visible(g.position):
                    continue
                if enemy_pos and g.position.distance_to(enemy_pos) < d:
                    continue
                occupied = any(
                    abs(s.position.x - g.position.x) < 1 and abs(s.position.y - g.position.y) < 1
                    for s in gas_structures)
                if occupied:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_geyser = g
            if best_geyser:
                for unit, ability in caps:
                    if ability == skip:
                        continue
                    aname = ability.name
                    if "ZERGBUILD" in aname or "EXTRACTOR" in aname:
                        if self.can_afford(ability):
                            try:
                                # If drone already on geyser, build
                                if unit.position.distance_to(best_geyser.position) < 2:
                                    await self.do(unit(ability, target=best_geyser))
                                    return
                                # Otherwise move drone to geyser first
                                await self.do(unit.move(best_geyser))
                                return
                            except (KeyError, Exception):
                                pass

# 4. Train from any available training ability (filter: idle, not larva already morphing)
        for unit, ability in caps:
            if ability == skip:
                continue
            aname = ability.name
            if "TRAIN" in aname or "MORPH" in aname:
                # Only train from idle units that aren't already producing
                if unit.is_idle and self.can_afford(ability):
                    # For Zerg larva, check persistent morphing tracking
                    if "LARVA" in unit.name.upper():
                        # Skip if larva is already morphing (has orders)
                        if unit.orders:
                            continue
                        # Skip if we commanded it recently (morph takes ~15 ticks)
                        if unit.tag in self._morphing_larvae:
                            if self._step - self._morphing_larvae[unit.tag] < 15:
                                continue
                            # Clean up old entries (older than 30 ticks)
                            if self._step - self._morphing_larvae[unit.tag] > 30:
                                del self._morphing_larvae[unit.tag]
                    # Don't queue if unit already has 5 orders (queue full)
                    if hasattr(unit, 'orders') and len(unit.orders) >= 5:
                        continue
                    await self.do(unit(ability))
                    if "LARVA" in unit.name.upper():
                        self._morphing_larvae[unit.tag] = self._step
                    return

    async def _exec_attack(self, caps):
        """Send combat units to attack enemy base."""
        if not self.enemy_start_locations:
            return
        target = self.enemy_start_locations[0]
        # Find combat units (not workers, not structures, not overlords/observers, not workers even if they can attack)
        worker_ids = {u.tag for u in self.workers}
        supply_ids = {u.tag for u in self.units if any(kw in u.name.upper() for kw in ("OVERLORD", "OVERSEER", "OBSERVER"))}
        combat_units = [u for u in self.units
                        if u.tag not in worker_ids and u.tag not in supply_ids
                        and not u.is_structure and u.can_attack
                        and u.ground_range > 0]  # Only units that actually shoot
        if not combat_units:
            return
        for unit in combat_units:
            if unit.is_idle or unit.is_moving:
                await self.do(unit.attack(target))

    async def _exec_defend(self, caps):
        """Find units with move capability and rally them."""
        if not self.townhalls:
            return
        target = self.townhalls.first.position
        for unit, ability in caps:
            if ability == AbilityId.MOVE:
                await self.do(unit.move(target))

    async def _exec_scout(self, caps):
        """Find a worker with move capability and send it to scout."""
        if not self.enemy_start_locations:
            return
        target = self.enemy_start_locations[0]
        for unit, ability in caps:
            if unit in self.workers and ability == AbilityId.MOVE:
                await self.do(unit.move(target))
                return

    async def _exec_expand(self, caps):
        """Find a worker that can build a townhall."""
        if self.minerals < 400:
            return
        loc = self._find_expansion()
        if not loc:
            return
        for unit, ability in caps:
            if unit in self.workers:
                aname = ability.name
                if any(kw in aname for kw in ("COMMANDCENTER", "NEXUS", "HATCHERY", "LAIR", "HIVE")):
                    if self.can_afford(ability):
                        pos = await self.find_placement(ability, near=loc)
                        if pos:
                            await self.do(unit(ability, target=pos))
                            return

    def _find_expansion(self):
        if self.expansion_locations:
            taken = {th.position for th in self.townhalls}
            for loc in self.expansion_locations:
                if all(loc.distance_to(t) > 10 for t in taken):
                    return loc
        return None

    async def _respond_to_chat(self, event: PerceivedEvent):
        """Respond to a chat message from another player.

        Reads the message, reasons about game state, and sends a response.
        """
        msg = event.text.lower().strip()
        response = self._generate_chat_response(msg)
        if response:
            await self.chat_send(response)
            self._chat_log.append({
                "tick": self._step,
                "from": event.source_id,
                "message": event.text,
                "response": response,
            })

    def _generate_chat_response(self, msg: str) -> str:
        """Generate a response to a chat message based on game state.

        Returns None if no response is warranted.
        """
        workers = len(self.workers)
        army = len(self.units)
        bases = len(self.townhalls)
        supply_pct = self.supply_used / max(1, self.supply_cap)

        # Greeting responses
        if any(w in msg for w in ("hello", "hi", "hey", "gl", "hf", "good luck", "have fun")):
            return "gl hf"

        # GG responses
        if any(w in msg for w in ("gg", "good game", "wp", "well played")):
            return "gg"

        # Question about game state
        if any(w in msg for w in ("how", "what", "why", "?")):
            if "worker" in msg or "economy" in msg:
                return f"i have {workers} workers across {bases} bases"
            if "army" in msg or "military" in msg:
                return f"army count: {army}"
            if "supply" in msg:
                return f"supply: {self.supply_used}/{self.supply_cap}"
            return f"minerals: {self.minerals}, supply: {self.supply_used}/{self.supply_cap}"

        # Trash talk
        if any(w in msg for w in ("bad", "terrible", "noob", "suck", "easy")):
            if army > workers:
                return "my army disagrees"
            return "we'll see"

        # Encouragement / neutral
        if any(w in msg for w in ("nice", "good", "cool", "wow")):
            return "thanks"

        # Supply block taunt
        if "supply" in msg and ("block" in msg or "stuck" in msg):
            return "building more supply now"

        # Default: acknowledge with game state
        return None

    def on_end(self, game_result):
        elapsed = time.time() - self._start_time
        workers = len(self.workers)
        army = len(self.units)
        bases = len(self.townhalls)

        action_counts = {}
        for a in self._actions_log:
            t = a["type"]
            action_counts[t] = action_counts.get(t, 0) + 1

        self.model_pool.save(MODEL_PATH)

        result = {
            "game_number": self.config.get("game_number", 0),
            "result": str(game_result),
            "steps": self._step,
            "elapsed": round(elapsed, 1),
            "workers_final": workers,
            "army_final": army,
            "bases_final": bases,
            "minerals_final": self.minerals,
            "supply_final": f"{self.supply_used}/{self.supply_cap}",
            "actions": action_counts,
            "model_updates": self.model_pool.total_updates,
            "exploration_rate": round(self.model_pool.epsilon, 4),
            "model_confidences": {
                action: round(model.confidence, 3)
                for action, model in self.model_pool.models.items()
                if model.n_observations > 0
            },
            "model_observations": {
                action: model.n_observations
                for action, model in self.model_pool.models.items()
                if model.n_observations > 0
            },
            "action_distribution": {
                action: round(count / max(1, len(self._actions_log)) * 100, 1)
                for action, count in action_counts.items()
            },
            "chat_log": self._chat_log,
            "perceived_events": self.perceiver.summary(),
        }

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(json.dumps(result, indent=2))

    def _integrate_replay_learnings(self, learnings: Dict[str, Any], parsed: ParsedReplay):
        """Integrate replay learnings into the bot's epistemic systems."""
        # Feed build orders into affordance models as "expert demonstrations"
        for build_order_data in learnings.get("build_orders", []):
            race = build_order_data.get("race", "Z")
            build_order = build_order_data.get("build_order", [])
            result = build_order_data.get("result", "unknown")
            
            # Convert build order steps into affordance observations
            for step in build_order:
                action_type = self._map_build_step_to_action(step)
                if action_type and action_type in self.model_pool.models:
                    model = self.model_pool.models[action_type]
                    # Treat expert build order as high-confidence observation
                    # Positive delta for the primary need this action serves
                    primary_need = self._infer_primary_need(action_type)
                    if primary_need:
                        observed_delta = {primary_need: 0.15}  # Expert demonstration
                        model.update(observed_delta, learning_rate=0.02)
        
        # Feed timing attacks into entity model as threat patterns
        for timing in learnings.get("timing_attacks", []):
            race = timing.get("race", "?")
            attack_time = timing.get("attack_time", 0)
            composition = timing.get("army_composition", {})
            effectiveness = timing.get("effectiveness", 1.0)
            
            if self.entity_model and attack_time > 0:
                enemy = self.entity_model.get_entity("enemy")
                if enemy:
                    enemy.add_evidence(
                        EvidenceType.OBSERVED_BEHAVIOR,
                        f"Replay timing attack: {race} attacks at {attack_time:.0f}s with {composition} (eff: {effectiveness:.1f})",
                        tick=int(attack_time * 22.4)  # Convert seconds to ticks
                    )
        
        # Feed counter-strategies into governance gate as learned constraints
        for counter in learnings.get("counter_strategies", []):
            failed_strat = counter.get("failed_strategy", {})
            counter_strat = counter.get("counter", "unknown")
            race = counter.get("race", "?")
            
            if self.governance_gate:
                # Record as learned constraint: avoid this strategy vs this race
                weakness = failed_strat.get("weakness", "unknown")
                self.governance_gate.add_learned_constraint(
                    f"avoid_{weakness}_vs_{race}",
                    f"Replay shows {race} with {weakness} weakness loses to {counter_strat}"
                )
        
        # Feed economic patterns into drive targets
        for econ in learnings.get("economic_patterns", []):
            efficiency = econ.get("efficiency_ratio", 0)
            workers = econ.get("workers", 0)
            max_supply = econ.get("max_supply", 0)
            
            # Estimate base count from max supply
            # Each base at full saturation: 8 mineral fields * 2 workers + 2 gas geysers * 3 workers = 22 workers
            estimated_bases = max(1, max_supply // 70)  # ~70 supply per base (22 workers + army/structures)
            optimal_workers = estimated_bases * 22  # 16 on minerals + 6 on gas
            
            # High efficiency benchmark: efficiency > 2.0 AND workers near optimal saturation
            if efficiency > 2.0 and workers >= optimal_workers * 0.8:
                # Adjust drive targets toward more efficient ratios
                if self.drives:
                    # Slightly increase mineral target for high efficiency
                    pass  # Drive targets are adaptive, model learning handles this

    def _map_build_step_to_action(self, step: Dict) -> Optional[str]:
        """Map a build order step to an affordance action type."""
        unit = step.get("unit", "").lower()
        upgrade = step.get("upgrade", "").lower()
        step_type = step.get("type", "")
        
        # Map unit/building names to action types
        if step_type == "upgrade":
            return "tech_up"
        
        economy_units = {"drone", "scv", "probe", "worker"}
        army_units = {"zergling", "baneling", "roach", "hydralisk", "mutalisk", "corruptor", 
                      "ultralisk", "infestor", "queen", "marine", "marauder", "stalker", "zealot",
                      "siegetank", "viking", "medivac", "colossus", "hightemplar", "darktemplar"}
        defense_units = {"spinecrawler", "sporecrawler", "bunker", "missileturrent", "photoncannon",
                         "planetaryfortress"}
        tech_units = {"spawningpool", "evolutionchamber", "hydraliskden", "spire", "greater_spire",
                      "infestationpit", "ultraliskcavern", "lair", "hive",
                      "barracks", "factory", "starport", "engineeringbay", "armory",
                      "gateway", "roboticsfacility", "stargate", "forge", "templararchive",
                      "dark_shrine", "roboticsbay", "fleetbeacon"}
        expand_units = {"hatchery", "lair", "hive", "commandcenter", "orbitalcommand", "planetaryfortress",
                        "nexus"}
        
        if unit in economy_units:
            return "build_economy"
        elif unit in army_units:
            return "build_army"
        elif unit in defense_units:
            return "defend"
        elif unit in tech_units:
            return "tech_up"
        elif unit in expand_units:
            return "expand"
        
        return None

    def _infer_primary_need(self, action_type: str) -> Optional[str]:
        """Map action type to primary need it satisfies."""
        mapping = {
            "build_economy": "minerals",
            "build_army": "military",
            "defend": "defense",
            "tech_up": "technology",
            "expand": "expansion",
            "attack": "military",
            "scout": "intel",
            "retreat": "defense",
            "hold": "defense",
        }
        return mapping.get(action_type)


def main():
    parser = argparse.ArgumentParser(description="SC2 Iterate — Multi-run learning loop")
    parser.add_argument("--games", type=int, default=3, help="Number of games to run")
    parser.add_argument("--difficulty", default="Easy",
                        choices=["Easy", "Medium", "Hard", "VeryHard"])
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--infinite", action="store_true", help="Run forever")
    args = parser.parse_args()

    difficulty_map = {
        "Easy": Difficulty.Easy, "Medium": Difficulty.Medium,
        "Hard": Difficulty.Hard, "VeryHard": Difficulty.VeryHard,
    }
    map_path = Path(os.environ['SC2PATH']) / "Maps" / "Melee" / "Simple64.SC2Map"
    map_settings = Map(map_path)

    game_num = 0
    total_existing = 0

    print(f"\n{'='*60}")
    print(f"  SC2 ITERATE — Learning Loop")
    print(f"  Games: {'infinite' if args.infinite else args.games}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Log: {LOG_PATH}")
    print(f"{'='*60}\n")

    # Resume from previous games
    if Path(LOG_PATH).exists():
        existing = Path(LOG_PATH).read_text().strip().split("\n")
        total_existing = len([l for l in existing if l.strip()])
        print(f"  Found {total_existing} previous game records\n")

    # Initialize replay learning integrator
    replay_integrator = None
    if SC2READER_AVAILABLE:
        replay_integrator = ReplayLearningIntegrator()
        print("  Replay learning enabled")

    games_to_run = 999999 if args.infinite else args.games

    try:
        for i in range(games_to_run):
            game_num = total_existing + i + 1
            print(f"\n{'='*40}")
            print(f"  GAME {game_num}")
            print(f"{'='*40}")

            config = {
                "map_name": "Simple64",
                "realtime": args.realtime,
                "difficulty": args.difficulty,
                "game_number": game_num,
            }

            bot = IterateBot(config)

            result = run_game(
                map_settings=map_settings,
                players=[
                    Bot(Race.Zerg, bot),
                    Computer(Race.Random, difficulty_map[args.difficulty]),
                ],
                realtime=False,
                step_time_limit=0.5,  # ~2x realtime instead of max speed
            )

            print(f"\n  Game {game_num} result: {result}")

            # Learn from replays periodically (every 5 games)
            if replay_integrator and SC2READER_AVAILABLE and game_num % 5 == 0:
                try:
                    replay_dir = Path(os.environ.get('SC2REPLAY_DIR', 
                        r'C:\Users\aobie\Documents\StarCraft II\Replays'))
                    replays = list(replay_dir.rglob("*.SC2Replay"))
                    if replays:
                        # Learn from the 5 most recent replays
                        recent_replays = sorted(replays, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
                        for replay_path in recent_replays:
                            print(f"  Learning from replay: {replay_path.name}")
                            parser = ReplayParser()
                            parsed = parser.parse_replay(str(replay_path))
                            learnings = replay_integrator.learn_from_replay(parsed)
                            
                            # Integrate learnings into the bot's epistemic systems
                            bot._integrate_replay_learnings(learnings, parsed)
                        
                        total_builds = len(replay_integrator.learned_patterns.get('build_orders', []))
                        total_timings = len(replay_integrator.learned_patterns.get('timing_attacks', []))
                        total_counters = len(replay_integrator.learned_patterns.get('counter_strategies', []))
                        print(f"  Total learned: {total_builds} build orders, {total_timings} timing attacks, {total_counters} counter-strategies")
                except Exception as e:
                    print(f"  Replay learning error: {e}")

            print(f"\n  Game {game_num} result: {result}")

            # Show improvement summary
            if Path(LOG_PATH).exists():
                lines = Path(LOG_PATH).read_text().strip().split("\n")
                if len(lines) >= 2:
                    last_two = [json.loads(l) for l in lines[-2:]]
                    if len(last_two) == 2:
                        prev, curr = last_two
                        print(f"\n  --- Improvement ---")
                        print(f"  Steps: {prev.get('steps', 0)} -> {curr.get('steps', 0)}")
                        print(f"  Workers: {prev.get('workers_final', 0)} -> {curr.get('workers_final', 0)}")
                        print(f"  Army: {prev.get('army_final', 0)} -> {curr.get('army_final', 0)}")
                        prev_actions = prev.get('action_distribution', {})
                        curr_actions = curr.get('action_distribution', {})
                        all_actions = set(list(prev_actions.keys()) + list(curr_actions.keys()))
                        for a in sorted(all_actions):
                            p = prev_actions.get(a, 0)
                            c = curr_actions.get(a, 0)
                            arrow = "up" if c > p else ("down" if c < p else "=")
                            print(f"  {a:15s}: {p:5.1f}% -> {c:5.1f}% ({arrow})")

            if args.infinite:
                time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {game_num} games")

    # Final summary
    if Path(LOG_PATH).exists():
        lines = Path(LOG_PATH).read_text().strip().split("\n")
        games = [json.loads(l) for l in lines if l.strip()]
        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY ({len(games)} games)")
        print(f"{'='*60}")

        results = {}
        for g in games:
            r = g.get("result", "unknown")
            results[r] = results.get(r, 0) + 1
        print(f"  Results: {results}")

        steps = [g.get("steps", 0) for g in games]
        print(f"  Steps: min={min(steps)}, max={max(steps)}, avg={sum(steps)/len(steps):.0f}")

        workers = [g.get("workers_final", 0) for g in games]
        print(f"  Workers: min={min(workers)}, max={max(workers)}, avg={sum(workers)/len(workers):.0f}")

        # Model state after all games
        if Path(MODEL_PATH).exists():
            with open(MODEL_PATH) as f:
                model_data = json.load(f)
            print(f"\n  Model state:")
            print(f"    Total updates: {model_data.get('total_updates', 0)}")
            print(f"    Exploration: {model_data.get('epsilon', 0):.4f}")
            for action, data in model_data.get("models", {}).items():
                if data.get("n_observations", 0) > 0:
                    print(f"    {action:15s} obs={data['n_observations']:4d} "
                          f"conf={data.get('confidence', 0):.3f}")

    print(f"\n  Log: {LOG_PATH}")
    print(f"  Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
