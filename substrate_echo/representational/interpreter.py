"""Semantic Interpreter — translates raw SC2 observations into EntityDescriptors.

This is the bridge between perception (what we see) and semantics
(what it means). It takes raw game state and produces EntityDescriptors
enriched with taxonomy, capabilities, affordances, and relationships
using the Ontology.

Architecture:
    Raw SC2 observation
          |
          v
    Semantic Interpreter
      - Classifies each unit via UnitClassifier/BuildingClassifier
      - Looks up taxonomy, capabilities, affordances from Ontology
      - Creates/updates EntityDescriptors
      - Populates StateGraph
          |
          v
    StateGraph (populated with semantic-rich entities)
          |
          v
    Frame System / Causal Graph / Narrative Layer
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from substrate_echo.representational.ontology import Ontology, Concept, ConceptCategory, TaxonomyNode
from substrate_echo.representational.entity_descriptor import (
    EntityDescriptor, EntityIdentity, EntityEmbodiment, EntityClassification,
    EntityComposition, EntityCapabilities, EntityAffordances,
    EntityRelationships, EntityState, EntityObservation, EntityCausality,
)
from substrate_echo.representational.state_graph import StateGraph, GraphEdge


# ── SC2 Knowledge Base ────────────────────────────────────────────
# Static knowledge that the interpreter uses to enrich observations.
# This is the "meaning" part — what each unit type means in terms
# of capabilities, affordances, and relationships.

# Unit type → (taxonomy path, capabilities, creates, enables, invites, denies)
SC2_UNIT_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Marine": {
        "taxonomy": ["Entity", "Unit", "Biological", "Infantry", "Ground", "Combat"],
        "roles": ["army", "defender"],
        "capabilities": ["move", "attack_ground", "attack_air", "stimpack", "guardian_shield"],
        "cost": {"minerals": 50},
        "creates": ["sustained_ranged_damage", "anti_air", "bio_ball"],
        "enables": ["stim_push", "bioball_timing"],
        "invites": ["splash_damage", "aoe_attacks"],
        "counters": ["Zergling", "Baneling"],
        "countered_by": ["Baneling", "Colossus", "SiegeTank"],
    },
    "Marauder": {
        "taxonomy": ["Entity", "Unit", "Biological", "Infantry", "Ground", "Combat"],
        "roles": ["army"],
        "capabilities": ["move", "attack_ground", "concussive_shells"],
        "cost": {"minerals": 100, "vespene": 25},
        "creates": ["armored_damage", "anti_armor"],
        "enables": ["bio_tank_push"],
        "invites": ["air_attacks"],
        "counters": ["Stalker", "Roach"],
        "countered_by": ["Stalker", "VoidRay"],
    },
    "Reaper": {
        "taxonomy": ["Entity", "Unit", "Biological", "Infantry", "Ground", "Combat"],
        "roles": ["scout", "army"],
        "capabilities": ["move", "attack_ground", "cliff_jump", "heal_out_of_combat"],
        "cost": {"minerals": 50, "vespene": 50},
        "creates": ["early_scout", "harassment"],
        "enables": ["early_intel", "worker_harass"],
        "invites": ["early_aggression_detection"],
        "counters": [],
        "countered_by": ["Queen", "Stalker"],
    },
    "SiegeTank": {
        "taxonomy": ["Entity", "Unit", "Mechanical", "Vehicle", "Ground", "Combat"],
        "roles": ["army", "defender"],
        "capabilities": ["move", "attack_ground", "siege_mode", "splash_damage"],
        "cost": {"minerals": 150, "vespene": 125},
        "creates": ["zone_control", "defensive_line", "splash_damage"],
        "enables": ["containment", "defensive_play"],
        "invites": ["flanking", "air_attacks", "drops"],
        "counters": ["Baneling", "Mutalisk"],
        "countered_by": ["Mutalisk", "Drop"],
    },
    "Medivac": {
        "taxonomy": ["Entity", "Unit", "Mechanical", "Transport", "Air", "Support"],
        "roles": ["support", "transport"],
        "capabilities": ["fly", "heal", "boost", "drop"],
        "cost": {"minerals": 100, "vespene": 100},
        "creates": ["biological_healing", "drop_play"],
        "enables": ["bio_sustain", "harassment_drops", "mobility"],
        "invites": ["anti_air"],
        "counters": [],
        "countered_by": ["Mutalisk", "Viper"],
    },
    "Zergling": {
        "taxonomy": ["Entity", "Unit", "Biological", "Infantry", "Ground", "Combat"],
        "roles": ["army"],
        "capabilities": ["move", "attack_ground", "burrow", "speed_upgrade"],
        "cost": {"minerals": 25},
        "creates": ["swarm", "map_control", "early_pressure"],
        "enables": [" surround_tactics", "runby"],
        "invites": ["splash_damage"],
        "counters": ["Baneling", "Marine"],
        "countered_by": ["Baneling", "Colossus", "Hellion"],
    },
    "Roach": {
        "taxonomy": ["Entity", "Unit", "Biological", "Armored", "Ground", "Combat"],
        "roles": ["army"],
        "capabilities": ["move", "attack_ground", "burrow", "ranged"],
        "cost": {"minerals": 75, "vespene": 25},
        "creates": ["durable_frontline", "anti_armor"],
        "enables": ["roach_rush", "defensive_frontline"],
        "invites": ["anti_armor"],
        "counters": ["Marauder", "Immortal"],
        "countered_by": ["Marauder", "Immortal", "VoidRay"],
    },
    "Mutalisk": {
        "taxonomy": ["Entity", "Unit", "Biological", "Air", "Combat"],
        "roles": ["army", "scout"],
        "capabilities": ["fly", "attack_ground", "attack_air", "harassment"],
        "cost": {"minerals": 100, "vespene": 100},
        "creates": ["air_superiority", "harassment", "map_control"],
        "enables": ["mutalisk_cloud", "worker_harass"],
        "invites": ["anti_air", "thors"],
        "counters": ["Marine", "Stalker"],
        "countered_by": ["Thor", "Phoenix", "Hydralisk"],
    },
    "Colossus": {
        "taxonomy": ["Entity", "Unit", "Mechanical", "Walker", "Ground", "Combat"],
        "roles": ["army"],
        "capabilities": ["move", "attack_ground", "cliff_walk", "splash_damage"],
        "cost": {"minerals": 300, "vespene": 200},
        "creates": ["massive_splash", "bio_annihilation"],
        "enables": ["deathball", "anti_bio"],
        "invites": ["anti_massive", "air_attacks"],
        "counters": ["Marine", "Zergling"],
        "countered_by": ["Viper", "Corruptor", "Immortal"],
    },
    "SCV": {
        "taxonomy": ["Entity", "Unit", "Mechanical", "Worker", "Ground", "Economy"],
        "roles": ["economy"],
        "capabilities": ["move", "harvest", "build", "repair"],
        "cost": {"minerals": 50},
        "creates": ["resource_income", "construction"],
        "enables": ["base_expansion", "infrastructure"],
        "invites": ["harassment"],
        "counters": [],
        "countered_by": [],
    },
    "Drone": {
        "taxonomy": ["Entity", "Unit", "Biological", "Worker", "Ground", "Economy"],
        "roles": ["economy"],
        "capabilities": ["move", "harvest", "build", "morph_structure"],
        "cost": {"minerals": 50},
        "creates": ["resource_income", "construction"],
        "enables": ["base_expansion", "infrastructure"],
        "invites": ["harassment"],
        "counters": [],
        "countered_by": [],
    },
    "Probe": {
        "taxonomy": ["Entity", "Unit", "Mechanical", "Worker", "Ground", "Economy"],
        "roles": ["economy"],
        "capabilities": ["move", "harvest", "build", "charge_shield"],
        "cost": {"minerals": 50},
        "creates": ["resource_income", "construction"],
        "enables": ["base_expansion", "infrastructure"],
        "invites": ["harassment"],
        "counters": [],
        "countered_by": [],
    },
    "Overlord": {
        "taxonomy": ["Entity", "Unit", "Biological", "Supply", "Air", "Support"],
        "roles": ["support", "supply"],
        "capabilities": ["fly", "generate_supply", "morph_overseer"],
        "cost": {"minerals": 100},
        "creates": ["supply_cap", "aerial_vision"],
        "enables": ["unit_production"],
        "invites": ["anti_air"],
        "counters": [],
        "countered_by": ["Viking", "Phoenix"],
    },
    "Pylon": {
        "taxonomy": ["Entity", "Structure", "Psionic", "Supply", "Ground", "Support"],
        "roles": ["support", "supply"],
        "capabilities": ["generate_supply", "power_buildings"],
        "cost": {"minerals": 100},
        "creates": ["supply_cap", "power_field"],
        "enables": ["building_placement", "warp_in"],
        "invites": ["harassment"],
        "counters": [],
        "countered_by": [],
    },
}

# Structure type → knowledge
SC2_STRUCTURE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "CommandCenter": {
        "taxonomy": ["Entity", "Structure", "Command", "Ground", "Economy"],
        "roles": ["economy", "command"],
        "capabilities": ["produce_worker", "orbital_command"],
        "cost": {"minerals": 400},
        "creates": ["resource_collection", "base_of_operations"],
        "enables": ["worker_production", "supply"],
        "invites": ["harassment"],
    },
    "Barracks": {
        "taxonomy": ["Entity", "Structure", "Production", "Ground", "Military"],
        "roles": ["production"],
        "capabilities": ["produce_infantry", "tech_lab", "reactor"],
        "cost": {"minerals": 150},
        "creates": ["infantry_production"],
        "enables": ["bio_army"],
        "invites": ["harassment"],
    },
    "Hatchery": {
        "taxonomy": ["Entity", "Structure", "Command", "Ground", "Economy"],
        "roles": ["economy", "command", "production"],
        "capabilities": ["produce_worker", "morph_larva", "creep_spread"],
        "cost": {"minerals": 300},
        "creates": ["resource_collection", "larva_production", "creep"],
        "enables": ["zerg_production", "expansion"],
        "invites": ["harassment"],
    },
    "Nexus": {
        "taxonomy": ["Entity", "Structure", "Command", "Ground", "Economy"],
        "roles": ["economy", "command"],
        "capabilities": ["produce_worker", "chronoboost", "warp_in"],
        "cost": {"minerals": 400},
        "creates": ["resource_collection", "chronoboost_energy"],
        "enables": ["probe_production", "warp_gate_tech"],
        "invites": ["harassment"],
    },
}


class SemanticInterpreter:
    """Translates raw SC2 observations into semantic-rich EntityDescriptors.

    This is the bridge between perception and meaning. It uses:
      - UnitClassifier for mechanical classification
      - Ontology for taxonomic and relational knowledge
      - Static SC2 knowledge base for capabilities/affordances
      - StateGraph as the shared world model it populates

    The interpreter runs every tick and:
      1. Scans all visible units and structures
      2. Creates or updates EntityDescriptors
      3. Updates edges in the StateGraph
      4. Marks missing entities as potentially gone
    """

    def __init__(self, ontology: Ontology, state_graph: StateGraph):
        self.ontology = ontology
        self.graph = state_graph
        self._tick: int = 0
        self._seen_this_tick: set = set()

        # Inject SC2 knowledge into ontology if empty
        self._ensure_ontology_populated()

    def _ensure_ontology_populated(self) -> None:
        """Populate ontology with SC2 concepts if not already present."""
        if len(self.ontology.concepts) > 0:
            return  # already populated

        for name, knowledge in SC2_UNIT_KNOWLEDGE.items():
            self.ontology.add_concept(Concept(
                name=name,
                category=ConceptCategory.UNIT,
                aliases=[],
                description=f"SC2 unit: {name}",
            ))
            # Add taxonomy edges
            taxonomy = knowledge.get("taxonomy", [])
            for i, tax in enumerate(taxonomy):
                parent = taxonomy[i + 1] if i + 1 < len(taxonomy) else None
                if parent and tax not in self.ontology.taxonomy:
                    self.ontology.add_taxonomy_node(TaxonomyNode(tax, parent=parent))

        for name, knowledge in SC2_STRUCTURE_KNOWLEDGE.items():
            self.ontology.add_concept(Concept(
                name=name,
                category=ConceptCategory.STRUCTURE,
                aliases=[],
                description=f"SC2 structure: {name}",
            ))

    def interpret_tick(self, bot: Any, tick: int) -> None:
        """Main entry point: interpret all visible game state at this tick.

        Args:
            bot: BotAI instance with units, known_enemy_units, etc.
            tick: current game tick
        """
        self._tick = tick
        self._seen_this_tick = set()

        # ── Own units ──
        for unit in bot.units:
            self._interpret_unit(unit, is_own=True, bot=bot)

        # ── Own structures ──
        for structure in bot.units.structure:
            self._interpret_structure(structure, is_own=True, bot=bot)

        # ── Known enemy units ──
        for unit in bot.known_enemy_units:
            self._interpret_unit(unit, is_own=False, bot=bot)

        # ── Known enemy structures ──
        for structure in bot.known_enemy_structures:
            self._interpret_structure(structure, is_own=False, bot=bot)

        # ── Update graph edges (relationships between entities) ──
        self._update_edges(bot)

        # ── Mark unseen entities ──
        self._mark_unseen()

        # ── Advance graph clock ──
        self.graph.tick(tick)

    def _interpret_unit(self, unit: Any, is_own: bool, bot: Any) -> None:
        """Create or update EntityDescriptor for a unit."""
        tag = unit.tag
        unit_name = unit.name if hasattr(unit, 'name') else str(unit.unit_type)
        controller = "self" if is_own else "enemy"

        # Build identity
        eid = f"{'own' if is_own else 'enemy'}_{unit_name}_{tag}"
        identity = EntityIdentity(
            id=eid,
            name=unit_name,
            aliases=[str(unit.unit_type)],
        )

        # Look up knowledge
        knowledge = SC2_UNIT_KNOWLEDGE.get(unit_name, {})

        # Classification
        taxonomy = knowledge.get("taxonomy", ["Entity", "Unit", unit_name])
        roles = knowledge.get("roles", [])

        # Capabilities from knowledge + mechanical properties
        capabilities = list(knowledge.get("capabilities", []))
        if hasattr(unit, 'can_attack') and unit.can_attack:
            if "attack" not in capabilities:
                capabilities.append("attack")
        if hasattr(unit, 'is_flying') and unit.is_flying:
            if "fly" not in capabilities:
                capabilities.append("fly")

        # Affordances from knowledge
        affordances = EntityAffordances(
            creates=list(knowledge.get("creates", [])),
            enables=list(knowledge.get("enables", [])),
            invites=list(knowledge.get("invites", [])),
        )

        # Relationships
        relationships = EntityRelationships(
            allies=[],
            counters=list(knowledge.get("counters", [])),
            countered_by=list(knowledge.get("countered_by", [])),
        )

        # State from live unit data
        state = EntityState(
            location=(unit.position.x, unit.position.y) if hasattr(unit, 'position') else None,
            health=unit.health if hasattr(unit, 'health') else None,
            shields=unit.shield if hasattr(unit, 'shield') else None,
            energy=unit.energy if hasattr(unit, 'energy') else None,
        )

        # Observation
        observation = EntityObservation(
            tick=self._tick,
            source="sc2_api",
            observation="visible_unit",
            confidence=1.0,
            data={"tag": tag, "is_own": is_own, "type_id": unit.unit_type},
        )

        # Create or update descriptor
        existing = self.graph.get_entity(eid)
        if existing:
            existing.state = state
            existing.observations.append(observation)
            existing.history.updated_tick = self._tick
            # Record state snapshot
            existing.history.record(self._tick, state.to_dict())
        else:
            descriptor = EntityDescriptor(
                identity=identity,
                embodiment=EntityEmbodiment(
                    environment="StarCraft II",
                    controller=controller,
                    simulation_tick=self._tick,
                ),
                classification=EntityClassification(
                    taxonomy=taxonomy,
                    roles=roles,
                ),
                composition=EntityComposition(
                    cost=knowledge.get("cost", {}),
                ),
                capabilities=EntityCapabilities(abilities=capabilities),
                affordances=affordances,
                relationships=relationships,
                state=state,
                observations=[observation],
            )
            descriptor.history.created_tick = self._tick
            descriptor.history.updated_tick = self._tick
            self.graph.add_entity(descriptor)
            self.graph.register_tag(tag, eid)

        self._seen_this_tick.add(eid)

    def _interpret_structure(self, structure: Any, is_own: bool, bot: Any) -> None:
        """Create or update EntityDescriptor for a structure."""
        tag = structure.tag
        struct_name = structure.name if hasattr(structure, 'name') else str(structure.unit_type)
        controller = "self" if is_own else "enemy"

        eid = f"{'own' if is_own else 'enemy'}_{struct_name}_{tag}"
        knowledge = SC2_STRUCTURE_KNOWLEDGE.get(struct_name, {})

        taxonomy = knowledge.get("taxonomy", ["Entity", "Structure", struct_name])
        roles = knowledge.get("roles", [])
        capabilities = list(knowledge.get("capabilities", []))

        affordances = EntityAffordances(
            creates=list(knowledge.get("creates", [])),
            enables=list(knowledge.get("enables", [])),
            invites=list(knowledge.get("invites", [])),
        )

        state = EntityState(
            location=(structure.position.x, structure.position.y) if hasattr(structure, 'position') else None,
            health=structure.health if hasattr(structure, 'health') else None,
            shields=structure.shield if hasattr(structure, 'shield') else None,
        )

        observation = EntityObservation(
            tick=self._tick,
            source="sc2_api",
            observation="visible_structure",
            confidence=1.0,
            data={"tag": tag, "is_own": is_own, "type_id": structure.unit_type},
        )

        existing = self.graph.get_entity(eid)
        if existing:
            existing.state = state
            existing.observations.append(observation)
            existing.history.updated_tick = self._tick
        else:
            descriptor = EntityDescriptor(
                identity=EntityIdentity(id=eid, name=struct_name, aliases=[str(structure.unit_type)]),
                embodiment=EntityEmbodiment(
                    environment="StarCraft II",
                    controller=controller,
                    simulation_tick=self._tick,
                ),
                classification=EntityClassification(taxonomy=taxonomy, roles=roles),
                composition=EntityComposition(cost=knowledge.get("cost", {})),
                capabilities=EntityCapabilities(abilities=capabilities),
                affordances=affordances,
                state=state,
                observations=[observation],
            )
            descriptor.history.created_tick = self._tick
            descriptor.history.updated_tick = self._tick
            self.graph.add_entity(descriptor)
            self.graph.register_tag(tag, eid)

        self._seen_this_tick.add(eid)

    def _update_edges(self, bot: Any) -> None:
        """Update relationship edges between entities.

        Adds counter/enemy edges based on the knowledge base.
        """
        own_entities = [e for e in self.graph.all_entities()
                        if e.embodiment.controller == "self"]
        enemy_entities = [e for e in self.graph.all_entities()
                          if e.embodiment.controller == "enemy"]

        # Add "enemy" edges between own and enemy units
        for own in own_entities:
            for enemy in enemy_entities:
                # Check if this pair already has an edge
                existing = self.graph.get_edges_from(
                    own.identity.id, "enemy")
                if not any(e.target_id == enemy.identity.id for e in existing):
                    # Determine edge weight from counter relationships
                    weight = 0.5  # default
                    if enemy.identity.name in own.relationships.counters:
                        weight = 0.8  # we counter them
                    elif own.identity.name in enemy.relationships.counters:
                        weight = 0.3  # they counter us

                    self.graph.add_edge(GraphEdge(
                        source_id=own.identity.id,
                        target_id=enemy.identity.id,
                        relation="enemy",
                        weight=weight,
                        confidence=1.0,
                        tick_created=self._tick,
                    ))

    def _mark_unseen(self) -> None:
        """Entities not seen this tick get reduced confidence."""
        for entity in self.graph.all_entities():
            if entity.identity.id not in self._seen_this_tick:
                age = self._tick - entity.history.updated_tick
                # Decay health confidence for unseen entities
                if age > 0:
                    entity.state.health_confidence = max(
                        0.1, entity.state.health_confidence * 0.95)

    def get_terrain_entity(self, terrain_metrics: Dict[str, float]) -> EntityDescriptor:
        """Create a terrain EntityDescriptor from computed metrics.

        Terrain is also an entity — it has capabilities (blocks movement,
        provides high ground) and affordances (chokes, defensive positions).
        """
        cliff_density = terrain_metrics.get("cliff_density", 0.0)
        terrain_complexity = terrain_metrics.get("terrain_complexity", 0.0)
        map_revealed = terrain_metrics.get("map_revealed", 0.0)

        # Terrain capabilities based on features
        capabilities = ["static", "blocks_movement"]
        if cliff_density > 0.1:
            capabilities.append("cliffs_present")
        if terrain_complexity > 0.3:
            capabilities.append("complex_pathing")

        # Terrain affordances
        creates = []
        enables = []
        if cliff_density > 0.05:
            creates.append("defensive_terrain")
            enables.append("cliff_jump_tactics")
        if terrain_complexity > 0.2:
            creates.append("choke_points")
            enables.append("funnel_defense")

        return EntityDescriptor(
            identity=EntityIdentity(id="terrain_map", name="Terrain"),
            classification=EntityClassification(
                taxonomy=["Entity", "Environment", "Terrain"],
                roles=["terrain"],
            ),
            capabilities=EntityCapabilities(abilities=capabilities),
            affordances=EntityAffordances(creates=creates, enables=enables),
            state=EntityState(
                confidence={
                    "cliff_density": cliff_density,
                    "terrain_complexity": terrain_complexity,
                    "map_revealed": map_revealed,
                }
            ),
        )
