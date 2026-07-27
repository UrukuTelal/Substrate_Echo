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
import asyncio
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'

from sc2 import BotAI, Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.constants import AbilityId
from sc2.position import Point2

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
from substrate_echo.embodiments.sc2.unit_classifier import (
    UnitClassifier, Role, Movement, AttackCapability,
)
from substrate_echo.epistemology.tactical_brain import TacticalBrain, BattleState
from substrate_echo.epistemology.replay_auditor import ReplayAuditor, EpistemicLedger, StrategyContext
from substrate_echo.representational import (
    Ontology, StateGraph, SemanticInterpreter, CausalGraph,
    FrameSystem, NarrativeLayer, Perspective,
    PERSPECTIVE_EARLY_GAME, PERSPECTIVE_MID_GAME,
)


MODEL_PATH = str(Path(__file__).parent.parent / "data" / "affordance_models.json")
LOG_PATH = str(Path(__file__).parent.parent / "data" / "iteration_log.jsonl")
TACTICAL_BRAIN_PATH = str(Path(__file__).parent.parent / "data" / "tactical_brain.json")
LEDGER_PATH = str(Path(__file__).parent.parent / "data" / "epistemic_ledger.json")
REPRESENTATIONAL_PATH = str(Path(__file__).parent.parent / "data" / "representational")


class IterateBot(BotAI):
    """SC2 bot for iteration runs. Same architecture as LiveBot."""

    # Race-agnostic name sets — used for unit filtering across all methods
    SUPPLY_NAMES = {"OVERLORD", "OVERSEER", "OBSERVER", "MEDIVAC", "WARP_PRISM", "COLossus"}
    SPAWNED_NAMES = {"LOCUST", "BROODLING", "INTERCEPTOR", "AUTOTURRET", "LARVA", "EGG", "COCOON"}
    WORKER_KEYWORDS = ("DRONE", "SCV", "PROBE")
    SUPPLY_KEYWORDS = ("SUPPLY", "OVERLORD", "PYLON", "DEPOT")
    TOWNHALL_KEYWORDS = ("COMMANDCENTER", "NEXUS", "HATCHERY", "LAIR", "HIVE", "ORBITALCOMMAND", "PLANETARYFORTRESS")
    GAS_KEYWORDS = ("EXTRACTOR", "ASSIMILATOR", "REFINERY")
    GAS_BUILD_KEYWORDS = ("EXTRACTOR", "ASSIMILATOR", "REFINERY", "ZERGBUILD", "PROTOSSBUILD", "TERRANBUILD")

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
        self.drives.add_need(NeedType.MINERALS, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.GAS, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.SUPPLY, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.MILITARY, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.INTEL, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.DEFENSE, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.EXPANSION, initial=0.01, target=0.99, weight=0.1)
        self.drives.add_need(NeedType.TECHNOLOGY, initial=0.01, target=0.99, weight=0.1)

        action_types = [
            "expand", "build_army", "build_economy", "defend",
            "attack", "scout", "tech_up", "retreat", "hold", 
        ]
        need_types = [nt.value for nt in NeedType]
        self.model_pool = AffordanceModelPool(action_types, need_types)

        self.perceiver = GameStatePerceiver()
        self._unit_classifier = UnitClassifier()
        self.tactical_brain = TacticalBrain()
        self.auditor = ReplayAuditor()
        self.ledger = EpistemicLedger()
        if Path(LEDGER_PATH).exists():
            self.ledger.load(LEDGER_PATH)
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
        self._spawned_names = self.SPAWNED_NAMES
        self._supply_names = self.SUPPLY_NAMES
        self._cached_caps = []
        self._caps_unit_count = 0
        self._caps_refresh_tick = 0
        # Income tracking for empirical diminishing returns
        self._prev_minerals = 0
        self._prev_worker_count = 0
        self._income_history: List[float] = []  # minerals per worker per tick
        self._diminishing_returns = False
        # Track larvae that are currently morphing (tag -> step commanded)
        self._morphing_larvae: Dict[int, int] = {}
        # Action delay timer (learnable by AI) - prevents action spam
        self._action_delay = 8  # ticks between actions (baseline 3, AI can adjust)
        self._last_action_tick = 0
        self._action_delay_history: List[int] = []  # track delays tried
        self._delay_performance: Dict[int, List[float]] = {}  # delay -> performance scores
        self._tactical_state: Optional[BattleState] = None
        self._last_attack_units: Dict[int, str] = {}  # tag -> unit name
        self._last_attack_enemies: Dict[int, str] = {}  # tag -> unit name
        self._last_attack_step: int = 0

        if Path(MODEL_PATH).exists():
            self.model_pool.load(MODEL_PATH)

        if Path(TACTICAL_BRAIN_PATH).exists():
            self.tactical_brain.load(TACTICAL_BRAIN_PATH)

        # Representational Layer — shared semantic substrate
        self.ontology = Ontology()
        self.state_graph = StateGraph()
        self.interpreter = SemanticInterpreter(self.ontology, self.state_graph)
        self.causal_graph = CausalGraph()
        self.frame_system = FrameSystem(self.state_graph, self.causal_graph)
        self.narrative_layer = NarrativeLayer(self.causal_graph)
        self._last_representational_tick = 0
        # Metrics tracking for representational layer and kernel
        self._frame_queries_executed = 0
        self._interpreter_entities_processed = 0
        self._kernel_coherence_values: List[float] = []
        self._kernel_volume_entropy_values: List[float] = []

    def on_start(self):
        self._start_time = time.time()
        self._race = self.config.get("race", "Zerg")
        self._my_player_id = self.state.common.player_id
        self.ledger.begin_game(self.config.get("game_number", 0))

    def _count_army(self):
        """Count actual combat units, excluding workers, supply, structures, and spawned."""
        worker_ids = {u.tag for u in self.workers}
        count = 0
        for u in self.units:
            if u.tag in worker_ids:
                continue
            if u.is_structure:
                continue
            if u.name.upper() in self._supply_names:
                continue
            if u.name.upper() in self._spawned_names:
                continue
            if u.can_attack:
                count += 1
        return count

    async def on_step(self, iteration: int):
        self._step += 1

        workers = len(self.workers)
        army = self._count_army()
        bases = len(self.townhalls)

        # Kernel observation (throttled internally)
        raw_vec = self.encoder.encode_from_botai(self)

        # Tactical brain: full state capture + battle analysis (every 5 ticks)
        if self._step % 5 == 0:
            self._tactical_state = self.tactical_brain.capture_state(self, self._step)
            battle = self.tactical_brain.analyze_battles(self, self._step)
            if battle:
                # Update experiment progress
                exp = self.tactical_brain.get_active_experiment()
                if exp and exp.unit_type:
                    current_count = sum(
                        1 for u in self.units
                        if u.name.upper() == exp.unit_type.upper()
                        and not u.is_structure
                    )
                    self.tactical_brain.update_experiment_progress(current_count, self._step)
        else:
            self._tactical_state = None

        kernel_obs = Observation(
            vector=raw_vec.tolist(),
            modality="sc2_game",
            embodiment_id="sc2",
            timestamp=time.time(),
            metadata={"step": self._step, "minerals": self.minerals,
                      "workers": workers, "army": army, "bases": bases,
                      "supply_used": self.supply_used, "supply_cap": self.supply_cap,
                      "map_revealed": self.encoder.information.map_revealed,
                      "terrain_complexity": self.encoder.information.terrain_complexity,
                      "cliff_density": self.encoder.information.cliff_density,
                      "visibility_advantage": self.encoder.information.visibility_advantage},
        )
        cognitive_state = self.kernel.publish_observation(kernel_obs)
        self._kernel_coherence_values.append(cognitive_state.coherence)
        self._kernel_volume_entropy_values.append(cognitive_state.volume_entropy)

        # ── Representational Layer tick ──
        if self._step % 3 == 0:  # every 3 ticks to avoid overhead
            # Interpret game state into EntityDescriptors
            prev_entity_count = len(self.state_graph._entities)
            self.interpreter.interpret_tick(self, self._step)
            self._interpreter_entities_processed += len(self.state_graph._entities) - prev_entity_count

            # Record significant events in causal graph
            self._record_causal_events(workers, army, bases)

            # Update frame system and narrative layer
            self.frame_system.set_perspective(self._select_perspective())
            # Frame queries are not yet wired (H14 pending) — count perspective updates
            self._frame_queries_executed += 1
            self.narrative_layer.process_tick(self._step)

        # Entity model (every tick lightweight, evidence every 50)
        if self._step == 1:
            self.entity_model.create_entity("enemy", embodiment="sc2")
        if self._step % 50 == 0:
            enemy = self.entity_model.get_entity("enemy")
            if enemy:
                enemy_count = len(self.known_enemy_units)
                if enemy_count > 0:
                    enemy.add_evidence(EvidenceType.OBSERVED_BEHAVIOR,
                                       f"Enemy units: {enemy_count}", tick=self._step)

        action = self._interpret_action()
        action_type = action.get("type", "hold") if action else "hold"

        # Auditor: record tick for failure analysis
        self.auditor.record_tick(self._step, self, action_type)

        # Populate epistemic ledger with evidence from this tick
        self.auditor.populate_ledger(self.ledger, self)

        # Chain recording (throttled — every 5 ticks)
        if self._step % 5 == 0:
            self.chain.record_observation(tick=self._step,
                raw_state={"minerals": self.minerals, "workers": workers,
                           "army": army, "bases": bases},
                encoded_vector=raw_vec.tolist(), source="botai")
            self.chain.record_action(tick=self._step, action_type=action_type,
                action_vector=[], decision_source="interpret_action")

        if action:
            await self._execute(action)

        # Perceive text feedback (throttled — every 10 ticks)
        if self._step % 10 == 0:
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

        if self._prev_observation is not None and self._step % 5 == 0:
            self.chain.record_outcome(tick=self._step,
                actual_state={"minerals": self.minerals, "workers": workers,
                              "army": army, "bases": bases,
                              "supply_used": self.supply_used})

        self._prev_observation = {"minerals": self.minerals, "workers": workers, "army": army}

        # Drives update + model learning (synced — every 10 ticks)
        if self._step % 10 == 0:
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
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.model_pool.save, MODEL_PATH)

        # Battle outcome tracking: evaluate results 20-40 ticks after attack
        if (self._last_attack_units and self._last_attack_step > 0
                and self._step - self._last_attack_step in (20, 30, 40)):
            # Check how many of our attack units survived
            our_survivors = sum(
                1 for tag in self._last_attack_units
                if any(u.tag == tag for u in self.units)
            )
            our_killed = len(self._last_attack_units) - our_survivors

            # Check how many enemy units we killed (were alive at attack, now gone)
            enemy_killed = sum(
                1 for tag, name in self._last_attack_enemies.items()
                if not any(eu.tag == tag for eu in self.known_enemy_units)
            )
            enemy_survivors = len(self._last_attack_enemies) - enemy_killed

            total_engaged = len(self._last_attack_units) + len(self._last_attack_enemies)
            if total_engaged > 0 and self._tactical_state:
                outcome = "won" if enemy_killed > our_killed else "lost"
                our_comp = {}
                for tag in self._last_attack_units:
                    name = self._last_attack_units[tag]
                    our_comp[name] = our_comp.get(name, 0) + 1
                enemy_comp = {}
                for tag in self._last_attack_enemies:
                    name = self._last_attack_enemies[tag]
                    enemy_comp[name] = enemy_comp.get(name, 0) + 1
                self.tactical_brain.record_battle_outcome(
                    my_comp=our_comp,
                    enemy_comp=enemy_comp,
                    won=(outcome == "won"),
                    tick=self._step,
                )

                # Record hypothesis outcome in epistemic ledger
                comp_key = "+".join(sorted(set(self._last_attack_units.values())))
                hyp_name = f"attack_{comp_key}"
                ctx = StrategyContext(
                    tick=self._step,
                    army_value=sum(
                        u.health + getattr(u, 'shield', 0)
                        for u in self.units if u.tag in self._last_attack_units
                    ),
                    army_count=len(self._last_attack_units),
                    worker_count=len(self.workers),
                    base_count=len(self.townhalls),
                    minerals=self.minerals,
                    vespene=self.vespene,
                    supply_used=self.supply_used,
                    supply_cap=self.supply_cap,
                    enemy_visible=len(self.known_enemy_units),
                    enemy_army_value=snap.enemy_army_value if (snap := self.auditor._snapshots[-1] if self.auditor._snapshots else None) else 0.0,
                    terrain_complexity=self.encoder.information.terrain_complexity,
                    cliff_density=self.encoder.information.cliff_density,
                    visibility_advantage=self.encoder.information.visibility_advantage,
                )
                self.ledger.record_hypothesis_outcome(
                    name=hyp_name,
                    outcome=outcome,
                    tick=self._step,
                    evidence_data={
                        "our_composition": list(self._last_attack_units.values()),
                        "enemy_composition": list(self._last_attack_enemies.values()),
                        "units_lost": our_killed,
                        "enemies_killed": enemy_killed,
                    },
                    context=ctx,
                )

            # Clear tracking after evaluation
            if self._step - self._last_attack_step > 40:
                self._last_attack_units = {}
                self._last_attack_enemies = {}

        # Save tactical brain every 500 ticks
        if self._step % 500 == 0:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.tactical_brain.save, TACTICAL_BRAIN_PATH)

    def _update_drives(self, workers, army, bases):
        # Use tactical brain for real per-base saturation from SC2 API
        if self._tactical_state:
            sat = self.tactical_brain.get_saturation_status(self._tactical_state)
            optimal_mineral_workers = sat["total_ideal"]
            # Count workers on gas from actual assignments
            gas_workers = 0
            for u in self.workers:
                try:
                    if u.orders and any(
                        any(kw in str(o.ability) for kw in self.GAS_KEYWORDS)
                        for o in u.orders):
                        gas_workers += 1
                except (KeyError, Exception):
                    pass
        else:
            # Fallback: estimate from API if tactical state unavailable
            gas_workers = 0
            for u in self.workers:
                try:
                    if u.orders and any(
                        any(kw in str(o.ability) for kw in self.GAS_KEYWORDS)
                        for o in u.orders):
                        gas_workers += 1
                except (KeyError, Exception):
                    pass
            optimal_mineral_workers = bases * 16

        mineral_workers = max(0, workers - gas_workers)
        optimal_gas_workers = bases * 6

        minerals = min(1.0, mineral_workers / max(1, optimal_mineral_workers))
        # Gas satisfaction = min of (worker allocation, actual reserves)
        gas_worker_ratio = gas_workers / max(1, optimal_gas_workers)
        gas_reserve_ratio = min(1.0, self.vespene / max(1, bases * 150))  # 150 gas per base is healthy
        gas = min(gas_worker_ratio, gas_reserve_ratio)
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

    def _record_causal_events(self, workers, army, bases):
        """Record significant game events in the causal graph."""
        from substrate_echo.representational.causal_graph import EventType

        # Track army changes for causal events
        if not hasattr(self, '_prev_army'):
            self._prev_army = army
            self._prev_workers = workers
            self._prev_bases = bases

        army_delta = army - self._prev_army
        if army_delta < -3:
            self.causal_graph.record_event(
                EventType.UNIT_DAMAGED, self._step, "own_army", "Army",
                f"Army lost {abs(army_delta)} units",
                0.9, {"delta": army_delta})
        elif army_delta > 3:
            self.causal_graph.record_event(
                EventType.UNIT_CREATED, self._step, "own_army", "Army",
                f"Army grew by {army_delta} units",
                0.9, {"delta": army_delta})

        if workers != self._prev_workers:
            self.causal_graph.record_event(
                EventType.RESOURCE_CHANGED, self._step, "economy", "Economy",
                f"Workers: {self._prev_workers} -> {workers}",
                0.8, {"from": self._prev_workers, "to": workers})

        if bases != self._prev_bases:
            self.causal_graph.record_event(
                EventType.EXPANSION_STARTED if bases > self._prev_bases else EventType.STRUCTURE_DESTROYED,
                self._step, "bases", "Bases",
                f"Bases: {self._prev_bases} -> {bases}",
                1.0, {"from": self._prev_bases, "to": bases})

        self._prev_army = army
        self._prev_workers = workers
        self._prev_bases = bases

    def _select_perspective(self):
        """Select the appropriate frame perspective based on game state."""
        army = self._count_army()
        enemy_count = len(self.known_enemy_units)

        if self._step < 500:
            return PERSPECTIVE_EARLY_GAME
        elif enemy_count > 0 and army > 0:
            # Check if we're under pressure
            threat_ratio = enemy_count / max(1, army)
            if threat_ratio > 1.5:
                from substrate_echo.representational.frames import PERSPECTIVE_UNDER_ATTACK
                return PERSPECTIVE_UNDER_ATTACK
            elif threat_ratio < 0.5:
                from substrate_echo.representational.frames import PERSPECTIVE_ATTACKING
                return PERSPECTIVE_ATTACKING

        return PERSPECTIVE_MID_GAME

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
            # Boost exploration for under-observed actions
            obs_boost = max(1.0, 10.0 / max(1, model.n_observations))
            action_score.final_score += info_gain * self.model_pool.epsilon * 100.0 * obs_boost
            scored.append((c, action_score))

        scored.sort(key=lambda x: x[1].final_score, reverse=True)

        # Diversity penalty: reduce score of over-represented actions
        recent_window = 50
        recent_actions = [a["type"] for a in self._actions_log[-recent_window:]]
        if recent_actions:
            from collections import Counter
            freq = Counter(recent_actions)
            for c, score_obj in scored:
                action_name = c.action_type.value
                count = freq.get(action_name, 0)
                dominance = count / len(recent_actions)
                if dominance > 0.6:
                    penalty = (dominance - 0.6) * 50.0
                    score_obj.final_score -= penalty

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

        army_count = self._count_army()
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

        Caches results — only refreshes every 200 ticks, when unit count changes,
        or if any cached unit tag is no longer active.
        Skips larvae and other mass-produced units to reduce API calls.
        """
        current_count = len(self.units)
        ticks_since_refresh = self._step - self._caps_refresh_tick
        current_tags = {u.tag for u in self.units}
        if (current_count == self._caps_unit_count
                and ticks_since_refresh < 200
                and self._cached_caps
                and all(u.tag in current_tags for u, _ in self._cached_caps)):
            return self._cached_caps
        self._caps_unit_count = current_count
        self._caps_refresh_tick = self._step
        caps = []
        skip_names = {"LOCUST", "BROODLING"}
        for unit in self.units:
            if unit.name.upper() in skip_names:
                continue
            try:
                abilities = await self.get_available_abilities(unit)
                for ability in abilities:
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
            action_type = "defend"  # fallback to defend if no other action 
            

        # Action delay - prevent action spam
        if self._step - self._last_action_tick < self._action_delay:
            return
        self._last_action_tick = self._step

        caps = await self._discover_capabilities()

        # Engagement override: if army is strong enough, attack periodically
        worker_ids = {u.tag for u in self.workers}
        supply_ids = {u.tag for u in self.units if any(kw in u.name.upper() for kw in self.SUPPLY_NAMES)}
        spawned = self.SPAWNED_NAMES
        army_units = []
        for u in self.units:
            if u.tag in worker_ids or u.tag in supply_ids:
                continue
            if u.is_structure or not u.can_attack:
                continue
            if u.name.upper() in spawned:
                continue
            info = self._unit_classifier.classify(u)
            if info and (Role.ARMY in info.roles or Role.SCOUT in info.roles):
                army_units.append(u)
        
        # Defensive attack: if enemy units visible near our base, attack regardless of army size
        # Filter out spawned units
        enemy_nearby = any(
            eu.position.distance_to(self.townhalls.first.position) < 40
            and eu.type_id.name.upper() not in self.SPAWNED_NAMES
            for eu in self.known_enemy_units
        ) if self.townhalls else False

        # Visibility-aware engagement: only push out when we have adequate vision
        vis_adv = self.encoder.information.visibility_advantage
        if (len(army_units) >= 8
                and self.enemy_start_locations
                and self._step % 300 < 50
                and vis_adv > 0.3):
            action_type = "attack"
        elif enemy_nearby and army_units:
            action_type = "attack"

        if action_type == "build_army":
            await self._exec_train(caps, force_army=True)
        elif action_type == "build_economy":
            await self._exec_train(caps)
        elif action_type in ("attack", "defensive_attack"):
            await self._exec_attack(caps)
        elif action_type == "defend":
            await self._exec_defend(caps)
        elif action_type == "scout":
            await self._exec_scout(caps)
        elif action_type == "expand":
            await self._exec_expand(caps)
        elif action_type == "tech_up":
            await self._exec_tech_up(caps)
        elif action_type == "retreat":
            await self._exec_defend(caps)

        self._actions_log.append({"step": self._step, "type": action_type})

    async def _exec_train(self, caps, force_army=False):
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
                # Keep last 500 observations (~20 seconds of game time)
                if len(self._income_history) > 500:
                    self._income_history.pop(0)
                # Detect diminishing returns over a ~5-second smoothing window
                if len(self._income_history) >= 200:
                    recent = sum(self._income_history[-112:]) / 112
                    earlier = sum(self._income_history[-224:-112]) / 112
                    if earlier > 0 and recent < earlier * 0.7:
                        self._diminishing_returns = True
                    elif recent > earlier * 0.9:
                        self._diminishing_returns = False
        self._prev_minerals = self.minerals
        self._prev_worker_count = len(self.workers)

        # Reset diminishing returns if we have very few workers (need to rebuild economy)
        if len(self.workers) < 10:
            self._diminishing_returns = False

        # If action is build_army OR diminishing returns, train army instead of workers
        build_army_now = force_army or self._diminishing_returns
        if build_army_now and self.workers:
            # Counter-unit override: if tactical brain suggests a specific unit, try it first
            suggested_type = None
            if self._tactical_state and self._tactical_state.enemy.unit_counts:
                suggestion = self.tactical_brain.suggest_counter_unit(
                    self._tactical_state.enemy.unit_counts, self._step)
                if suggestion:
                    suggested_type = suggestion

            if suggested_type:
                # Try to build the suggested counter-unit
                for unit, ability in caps:
                    if ability == skip:
                        continue
                    aname = ability.name
                    if "TRAIN" in aname or "MORPH" in aname:
                        if suggested_type.upper() in aname:
                            if unit.is_idle and self.can_afford(ability):
                                await self.do(unit(ability))
                                return

            # Search ALL capabilities for the best army unit to train
            army_candidates = []
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                if "TRAIN" in aname or "MORPH" in aname:
                    # Exclude workers AND supply units
                    if not any(kw in aname for kw in self.WORKER_KEYWORDS):
                        if not any(kw in aname for kw in self.SUPPLY_KEYWORDS):
                            if unit.is_idle and self.can_afford(ability):
                                army_candidates.append((unit, ability, aname))
            
            # Train the first viable army unit
            if army_candidates:
                unit, ability, aname = army_candidates[0]
                print(f"[DEBUG build_army] Training: {aname} from {unit.name} (tag={unit.tag})")
                await self.do(unit(ability))
                return
            # If force_army and no army available, do NOT fall back to workers
            if force_army:
                return
        elif self.workers and self.townhalls:
            # Still need workers — train one
            for unit, ability in caps:
                if ability == skip:
                    continue
                aname = ability.name
                if any(kw in aname for kw in ("TRAIN", "MORPH")) and any(
                        kw in aname for kw in self.WORKER_KEYWORDS):
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
                if any(kw in aname for kw in self.SUPPLY_KEYWORDS):
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

        # 3. Need gas — try to build extractor (worker must be on geyser first)
        gas_structures = self.units.structure.filter(
            lambda s: any(kw in s.name.upper() for kw in self.GAS_KEYWORDS))
        gas_count = len(gas_structures)
        gas_needed = min(len(self.workers) // 3, 2) if self.townhalls else 0
        # Also consider: if we have lots of minerals but low gas, need more gas
        if self.minerals > 300 and self.vespene < 50:
            gas_needed = max(gas_needed, 2)
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
                    if unit in self.workers and any(kw in aname for kw in self.GAS_KEYWORDS):
                        if self.can_afford(ability):
                            try:
                                # SC2 API handles worker movement automatically
                                await self.do(unit(ability, target=best_geyser))
                                return
                            except (KeyError, Exception):
                                pass

    async def _exec_attack(self, caps):
        """Send combat units to attack enemy base.

        Uses UnitClassifier for role-aware unit selection:
        - If enemy has visible air units, prioritize anti-air + dual-attack units
        - Dual-attack units (can hit both air and ground) are always included
        - Workers, overlords, and spawned units excluded
        - On cliff-heavy maps, prioritize cliff-traversable units (Reaper, Colossus)
        - On low-visibility maps, send scouts ahead

        Tracks battle composition for TacticalBrain hypothesis generation.
        """
        if not self.enemy_start_locations:
            return
        target = self.enemy_start_locations[0]

        worker_ids = {u.tag for u in self.workers}
        supply_ids = {u.tag for u in self.units if any(
            kw in u.name.upper() for kw in self.SUPPLY_NAMES)}
        spawned = self.SPAWNED_NAMES

        combat_units = []
        for u in self.units:
            if u.tag in worker_ids or u.tag in supply_ids:
                continue
            if u.is_structure or not u.can_attack:
                continue
            if u.name.upper() in spawned:
                continue
            info = self._unit_classifier.classify(u)
            if info and (Role.ARMY in info.roles or Role.SCOUT in info.roles):
                combat_units.append(u)

        if not combat_units:
            return

        # Detect if enemy has visible air units
        enemy_air = [
            eu for eu in self.known_enemy_units
            if not eu.is_structure
            and eu.type_id.name.upper() not in spawned
            and self._unit_classifier.classify(eu)
            and self._unit_classifier.classify(eu).movement == Movement.AIR
        ]

        # Record pre-battle composition for hypothesis tracking
        if self._tactical_state and len(combat_units) >= 3:
            self._last_attack_units = {u.tag: u.name.upper() for u in combat_units}
            self._last_attack_step = self._step

        if enemy_air:
            # Prioritize: dual-attack units first, then anti-air-only
            dual = self._unit_classifier.filter_dual_attack(combat_units)
            anti_air_only = [
                u for u in combat_units
                if u not in dual
                and self._unit_classifier.classify(u)
                and (AttackCapability.GVA in self._unit_classifier.classify(u).attack_caps
                     or AttackCapability.AVA in self._unit_classifier.classify(u).attack_caps)
            ]
            # If we have anti-air, engage; otherwise still attack with what we have
            attack_group = dual + anti_air_only if (dual or anti_air_only) else combat_units
        else:
            attack_group = combat_units

        # Terrain-aware ordering: on cliff-heavy maps, send cliff-traversable
        # units (Reaper, Colossus) first — they navigate terrain faster
        cliff_density = self.encoder.information.cliff_density
        if cliff_density > 0.15 and len(attack_group) > 1:
            cliff_units = self._unit_classifier.filter_cliff_traversable(attack_group)
            other_units = [u for u in attack_group if u not in cliff_units]
            if cliff_units:
                attack_group = cliff_units + other_units

        # Record which enemies are engaged for outcome tracking
        if self._tactical_state and self.known_enemy_units:
            visible_enemy = [
                eu for eu in self.known_enemy_units
                if not eu.is_structure
                and eu.type_id.name.upper() not in spawned
                and eu.position.distance_to(target) < 40
            ]
            if visible_enemy:
                self._last_attack_enemies = {eu.tag: eu.name.upper() for eu in visible_enemy}

        for unit in attack_group:
            if unit.is_idle or unit.is_moving:
                await self.do(unit.attack(target))

    async def _exec_defend(self, caps):
        """Rally combat units to base defense position.

        Uses UnitClassifier to select army-role units.
        If enemy air is visible near base, prioritizes anti-air.
        If terrain has high cliff density, prioritizes cliff-traversable units.
        """
        if not self.townhalls:
            return
        target = self.townhalls.first.position

        worker_ids = {u.tag for u in self.workers}
        supply_ids = {u.tag for u in self.units if any(
            kw in u.name.upper() for kw in self.SUPPLY_NAMES)}
        spawned = self.SPAWNED_NAMES

        combat_units = []
        for u in self.units:
            if u.tag in worker_ids or u.tag in supply_ids:
                continue
            if u.is_structure or not u.can_attack:
                continue
            if u.name.upper() in spawned:
                continue
            info = self._unit_classifier.classify(u)
            if info and (Role.ARMY in info.roles or Role.SUPPORT in info.roles):
                combat_units.append(u)

        if not combat_units:
            return

        # Check for enemy air near base
        enemy_air_near = [
            eu for eu in self.known_enemy_units
            if eu.position.distance_to(target) < 30
            and not eu.is_structure
            and self._unit_classifier.classify(eu)
            and self._unit_classifier.classify(eu).movement == Movement.AIR
        ]

        if enemy_air_near:
            dual = self._unit_classifier.filter_dual_attack(combat_units)
            anti_air_only = [
                u for u in combat_units
                if u not in dual
                and self._unit_classifier.classify(u)
                and (AttackCapability.GVA in self._unit_classifier.classify(u).attack_caps
                     or AttackCapability.AVA in self._unit_classifier.classify(u).attack_caps)
            ]
            defend_group = dual + anti_air_only if (dual or anti_air_only) else combat_units
        else:
            defend_group = combat_units

        # Terrain-aware: on cliff-heavy maps, prioritize units that can
        # traverse cliffs to reach threats faster
        cliff_density = self.encoder.information.cliff_density
        if cliff_density > 0.15:
            cliff_units = self._unit_classifier.filter_cliff_traversable(defend_group)
            other_units = [u for u in defend_group if u not in cliff_units]
            defend_group = cliff_units + other_units

        for unit in defend_group:
            if unit.is_idle or unit.is_moving:
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
                if any(kw in aname for kw in self.TOWNHALL_KEYWORDS):
                    if self.can_afford(ability):
                        pos = await self.find_placement(ability, near=loc)
                        if pos:
                            await self.do(unit(ability, target=pos))
                            return

    async def _exec_tech_up(self, caps):
        """Build tech structures and research upgrades.

        Substrate-agnostic: discovers available BUILD and RESEARCH abilities
        from capabilities rather than hardcoding race-specific structures.
        """
        skip = self._last_failed_ability if (self._step - self._last_failed_tick < 50) else None

        townhall_types = {s.type_id for s in self.townhalls}
        existing_structs = {s.type_id for s in self.units.structure if s.type_id not in townhall_types}

        # Phase 1: Build a tech structure we don't already have
        # Find any BUILD ability that creates a non-townhall, non-supply, non-gas structure
        for unit, ability in caps:
            if ability == skip:
                continue
            if unit not in self.workers:
                continue
            aname = ability.name
            if "BUILD" not in aname:
                continue
            # Skip supply and gas structures (handled elsewhere)
            if any(kw in aname for kw in self.SUPPLY_KEYWORDS):
                continue
            if any(kw in aname for kw in self.GAS_KEYWORDS):
                continue
            # Skip townhall builds (handled by expand)
            if any(kw in aname for kw in self.TOWNHALL_KEYWORDS):
                continue
            if self.can_afford(ability):
                try:
                    th_pos = self.townhalls.first.position if self.townhalls else None
                    if th_pos:
                        pos = await self.find_placement(ability, near=th_pos, max_distance=20)
                        if pos:
                            print(f"[DEBUG tech_up] Building: {aname} at {pos}")
                            await self.do(unit(ability, target=pos))
                            return
                except (KeyError, Exception):
                    pass

        # Phase 2: Research upgrades from existing tech structures
        # Find any structure that has RESEARCH abilities
        for struct in self.units.structure:
            if struct.is_idle and struct.type_id not in townhall_types:
                for unit, ability in caps:
                    if ability == skip:
                        continue
                    if unit.tag != struct.tag:
                        continue
                    aname = ability.name
                    if "RESEARCH" in aname and self.can_afford(ability):
                        await self.do(unit(ability))
                        return

        # Phase 3: Morph advanced townhalls (Lair, Hive, Orbital, Planetary)
        for unit, ability in caps:
            if ability == skip:
                continue
            aname = ability.name
            if "MORPH" in aname and any(kw in aname for kw in self.TOWNHALL_KEYWORDS):
                if unit.is_idle and self.can_afford(ability):
                    await self.do(unit(ability))
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
        army = self._count_army()
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
        try:
            self._on_end_impl(game_result)
        except Exception as e:
            print(f"  on_end error (non-fatal): {e}")
            # Still save what we can
            try:
                self.model_pool.save(MODEL_PATH)
                self.tactical_brain.save(TACTICAL_BRAIN_PATH)
            except Exception:
                pass

    def _on_end_impl(self, game_result):
        if game_result is None:
            game_result = "Result.Victory"
        elapsed = time.time() - self._start_time
        workers = len(self.workers)
        army = self._count_army()
        bases = len(self.townhalls)

        action_counts = {}
        for a in self._actions_log:
            t = a["type"]
            action_counts[t] = action_counts.get(t, 0) + 1

        self.model_pool.save(MODEL_PATH)
        self.tactical_brain.save(TACTICAL_BRAIN_PATH)

        # Run auditor
        audit_report = self.auditor.analyze(
            game_result=str(game_result),
            game_number=self.config.get("game_number", 0),
        )

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
            },
            "model_observations": {
                action: model.n_observations
                for action, model in self.model_pool.models.items()
            },
            "action_distribution": {
                action: round(count / max(1, len(self._actions_log)) * 100, 1)
                for action, count in action_counts.items()
            },
            "chat_log": self._chat_log,
            "perceived_events": self.perceiver.summary(),
            "tactical_brain": {
                "battles_recorded": len(self.tactical_brain._battle_log),
                "hypotheses_generated": len(self.tactical_brain._hypotheses),
                "hypotheses_validated": len([h for h in self.tactical_brain._hypotheses.values()
                                            if h.status.name == "VALIDATED"]),
                "hypotheses_active": len([h for h in self.tactical_brain._hypotheses.values()
                                          if h.status.name == "ACTIVE"]),
                "experiments_run": len(self.tactical_brain._experiments),
                "active_experiment": {
                    "unit_type": self.tactical_brain.get_active_experiment().unit_type,
                    "outcome": self.tactical_brain.get_active_experiment().outcome,
                } if self.tactical_brain.get_active_experiment() else None,
                "hypothesis_details": [
                    {"id": h.id, "counter": h.counter_unit,
                     "targets": h.targets_enemy, "confidence": round(h.confidence, 3),
                     "wins": h.wins, "losses": h.losses, "status": h.status.name}
                    for h in self.tactical_brain._hypotheses.values()
                ],
            },
            "audit": audit_report.to_dict(),
            "kernel": {
                "avg_coherence": round(sum(self._kernel_coherence_values) / max(1, len(self._kernel_coherence_values)), 4),
                "coherence_variance": round(
                    sum((c - sum(self._kernel_coherence_values) / max(1, len(self._kernel_coherence_values))) ** 2
                        for c in self._kernel_coherence_values) / max(1, len(self._kernel_coherence_values)), 6
                ) if self._kernel_coherence_values else 0.0,
                "avg_volume_entropy": round(sum(self._kernel_volume_entropy_values) / max(1, len(self._kernel_volume_entropy_values)), 4),
                "n_attractors": len(self.kernel._base_attractors),
                "cognitive_energy": self.kernel._cognitive_energy,
                "ticks_processed": self.kernel._tick,
            },
            "representational": {
                "state_graph": {
                    "entities": len(self.state_graph._entities),
                    "edges": len(self.state_graph._edges),
                },
                "causal_graph": {
                    "events": len(self.causal_graph._events),
                    "consequences": len(self.causal_graph._consequences),
                },
                "narrative_layer": {
                    "active": len(self.narrative_layer._active),
                    "completed": len(self.narrative_layer._completed),
                },
                "frame_queries_executed": self._frame_queries_executed,
                "interpreter_entities_processed": self._interpreter_entities_processed,
                "ontology_concepts": len(self.ontology.concepts),
            },
        }

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Finalize epistemic ledger
        self.ledger.end_game(
            game_number=self.config.get("game_number", 0),
            result=str(game_result),
            tick_count=self._step,
            capabilities_used=[
                cap_name.split(":")[-1]
                for cap_name in self.ledger.capabilities
                if self.ledger.capabilities[cap_name].last_seen_tick == self._step
            ],
        )
        self.ledger.save(LEDGER_PATH)

        # Save representational layer
        rep_dir = REPRESENTATIONAL_PATH
        os.makedirs(rep_dir, exist_ok=True)
        self.state_graph.save(os.path.join(rep_dir, "state_graph.json"))
        self.causal_graph.save(os.path.join(rep_dir, "causal_graph.json"))
        self.narrative_layer.save(os.path.join(rep_dir, "narratives.json"))

        print(json.dumps(result, indent=2))
        print(audit_report.summary_text())
        print(self.ledger.summary_text())
        print(self.state_graph.summary_text())
        print(self.causal_graph.summary_text())
        print(self.narrative_layer.summary_text())

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
                weakness = failed_strat.get("weakness", "unknown")
                rule_id = f"replay_{weakness}_{race}_{counter_strat}"
                desc = (f"Replay shows {race} with {weakness} "
                        f"weakness loses to {counter_strat}")
                from substrate_echo.epistemology.governance_gate import (
                    GovernanceRule, RulePriority,
                )
                def _make_check(avoid_action, fallback):
                    def _check(action_type, **kw):
                        from substrate_echo.epistemology.governance_gate import (
                            GovernanceVerdict, GovernanceDecision,
                        )
                        if action_type == avoid_action:
                            return GovernanceVerdict(
                                decision=GovernanceDecision.MODIFY,
                                rule_id=rule_id,
                                reason=desc,
                                original_action=action_type,
                                adjusted_action=fallback,
                            )
                        return None
                    return _check
                avoid_action = weakness if weakness != "unknown" else "attack"
                self.governance_gate.add_rule(GovernanceRule(
                    rule_id=rule_id,
                    description=desc,
                    priority=RulePriority.LOW,
                    check=_make_check(avoid_action, "defend"),
                ))
        
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

        # Persist updated models and constraints to disk for the next game iteration
        self.model_pool.save(MODEL_PATH)
        if hasattr(self.governance_gate, 'save'):
            self.governance_gate.save()
        if hasattr(self.entity_model, 'save'):
            self.entity_model.save()

    def _map_build_step_to_action(self, step: Dict) -> Optional[str]:
        """Map a build order step to an affordance action type."""
        unit = step.get("unit", "").lower()
        upgrade = step.get("upgrade", "").lower()
        step_type = step.get("type", "")
        
        # Map unit/building names to action types
        if step_type == "upgrade":
            return "tech_up"
        
        economy_units = {"drone", "scv", "probe", "worker", "medivac", "overlord", "overseer", "mule"}
        army_units = {"zergling", "baneling", "roach", "hydralisk", "mutalisk", "corruptor", 
                      "ultralisk", "infestor", "queen", "marine", "marauder", "stalker", "zealot",
                      "siegetank", "viking",  "colossus", "hightemplar", "darktemplar"}
        defense_units = {"spinecrawler", "sporecrawler", "bunker", "missileturrent", "photoncannon",
                         "planetaryfortress"}
        tech_units = {"spawningpool", "evolutionchamber", "hydraliskden", "spire", "greater_spire",
                      "infestationpit", "ultraliskcavern", "lair", "hive",
                      "barracks", "factory", "starport", "engineeringbay", "armory",
                      "gateway", "roboticsfacility", "stargate", "forge", "templararchive",
                      "dark_shrine", "roboticsbay", "fleetbeacon"}
        expand_units = {"hatchery", "lair", "hive", "commandcenter", "orbitalcommand", "planetaryfortress",
                        "nexus"}
        
        if unit in tech_units:
            return "tech_up"
        elif unit in army_units:
            return "build_army"
        elif unit in economy_units:
            return "build_economy"
        elif unit in defense_units:
            return "defend"
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
    parser.add_argument("--race", default="Zerg",
                        choices=["Zerg", "Terran", "Protoss", "Random"],
                        help="Race to play")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--infinite", action="store_true", help="Run forever")
    parser.add_argument("--size", default="64",
                        help="Map size: 32, 48, 64, 96, 128, or 'maxsize' to randomize each game")
    args = parser.parse_args()

    difficulty_map = {
        "Easy": Difficulty.Easy, "Medium": Difficulty.Medium,
        "Hard": Difficulty.Hard, "VeryHard": Difficulty.VeryHard,
    }

    SC2_MAPS_DIR = Path(os.environ['SC2PATH']) / "Maps" / "Melee"
    ALL_MAPS = {
        32: "Flat32", 48: "Flat48", 64: "Flat64",
        96: "Flat96", 128: "Flat128",
    }
    ALL_SIZES = sorted(ALL_MAPS.keys())

    def pick_map():
        if args.size == "maxsize":
            nonlocal _size_pool
            if not _size_pool:
                _size_pool = list(ALL_SIZES)
                random.shuffle(_size_pool)
            size = _size_pool.pop()
        else:
            size = int(args.size)
        map_name = ALL_MAPS.get(size, f"Simple{size}")
        return Map(SC2_MAPS_DIR / f"{map_name}.SC2Map"), map_name, size

    def time_limit_for_size(size):
        """Exponential time limit: larger maps = exponentially more time per step."""
        return 30.0 * (size / 32) ** 2.0

    _size_pool: List[int] = []

    game_num = 0
    total_existing = 0

    print(f"\n{'='*60}")
    print(f"  SC2 ITERATE — Learning Loop")
    print(f"  Games: {'infinite' if args.infinite else args.games}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Size: {args.size}")
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

            map_settings, map_name, map_size = pick_map()
            ttl = time_limit_for_size(map_size)
            print(f"  Map: {map_name} ({map_size}x{map_size}) — {ttl:.0f}s/step")

            config = {
                "map_name": map_name,
                "realtime": args.realtime,
                "difficulty": args.difficulty,
                "race": args.race,
                "game_number": game_num,
            }

            bot = IterateBot(config)

            try:
                bot_race = Race.Zerg if args.race == "Zerg" else (
                    Race.Terran if args.race == "Terran" else (
                    Race.Protoss if args.race == "Protoss" else Race.Random))
                result = run_game(
                    map_settings=map_settings,
                    players=[
                        Bot(bot_race, bot),
                        Computer(Race.Random, difficulty_map[args.difficulty]),
                    ],
                    realtime=args.realtime,
                    step_time_limit=None if args.realtime else ttl,
                )
            except Exception as e:
                print(f"\n  Game {game_num} crashed: {e}")
                result = None

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
                        # Audit comparison
                        prev_audit = prev.get("audit", {})
                        curr_audit = curr.get("audit", {})
                        if prev_audit and curr_audit:
                            prev_fp = len(prev_audit.get("failure_points", []))
                            curr_fp = len(curr_audit.get("failure_points", []))
                            print(f"  Failures: {prev_fp} -> {curr_fp}")
                            prev_sb = prev_audit.get("supply_blocked_ticks", 0)
                            curr_sb = curr_audit.get("supply_blocked_ticks", 0)
                            print(f"  Supply blocked: {prev_sb} -> {curr_sb} ticks")

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