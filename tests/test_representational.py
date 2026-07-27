"""Tests for the Representational Layer.

Covers: Ontology, EntityDescriptor, StateGraph, SemanticInterpreter,
CausalGraph, FrameSystem, NarrativeLayer.
"""
import os
import tempfile
import pytest
from substrate_echo.representational import (
    Concept, ConceptCategory, TaxonomyNode, Rule, Constraint,
    PropertySchema, Ontology,
    EntityDescriptor, EntityIdentity, EntityEmbodiment, EntityClassification,
    EntityComposition, EntityCapabilities, EntityAffordances,
    EntityRelationships, EntityState, EntityObservation, EntityHypothesis,
    EntityEvidence, EntityCausality, EntityHistory,
    StateGraph, GraphEdge,
    CausalGraph, CausalEvent, Consequence, EventChain,
    EventType, ConsequenceType,
    FrameSystem, FrameResult, Perspective, FrameType,
    PERSPECTIVE_EARLY_GAME, PERSPECTIVE_MID_GAME,
    PERSPECTIVE_UNDER_ATTACK, PERSPECTIVE_ATTACKING,
    NarrativeLayer, Narrative, NarrativeEvent, NarrativeType, NarrativeArc,
)


# ── Ontology ───────────────────────────────────────────────────────

class TestOntology:
    def test_concept_creation(self):
        c = Concept("Marine", ConceptCategory.UNIT, aliases=["bio", "infantry"])
        assert c.name == "Marine"
        assert c.category == ConceptCategory.UNIT
        assert "bio" in c.aliases

    def test_add_concepts(self):
        o = Ontology()
        o.add_concept(Concept("Marine", ConceptCategory.UNIT))
        o.add_concept(Concept("Drone", ConceptCategory.UNIT))
        assert len(o.concepts) == 2

    def test_taxonomy_is_a(self):
        o = Ontology()
        o.add_taxonomy_node(TaxonomyNode("Marine", parent="GroundUnit"))
        o.add_taxonomy_node(TaxonomyNode("GroundUnit", parent="Unit"))
        o.add_taxonomy_node(TaxonomyNode("Unit", parent="Entity"))
        assert o.is_a("Marine", "Entity")
        assert o.is_a("Marine", "GroundUnit")
        assert not o.is_a("Marine", "Structure")

    def test_rules(self):
        o = Ontology()
        o.add_rule(Rule("supply_block", "supply_used >= supply_cap", "production_blocked"))
        assert len(o.rules) == 1
        assert o.rules[0].rule_id == "supply_block"

    def test_constraints(self):
        o = Ontology()
        o.add_constraint(Constraint("drone_no_air", "Drone", "cannot", "attack_air"))
        assert o.is_constrained("Drone", "cannot", "attack_air")
        assert not o.is_constrained("Marine", "cannot", "attack_air")

    def test_save_load(self):
        o = Ontology()
        o.add_concept(Concept("Marine", ConceptCategory.UNIT))
        o.add_taxonomy_node(TaxonomyNode("Marine", parent="Unit"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            o.save(path)
            o2 = Ontology()
            o2.load(path)
            assert len(o2.concepts) == 1
        finally:
            os.unlink(path)


# ── EntityDescriptor ────────────────────────────────────────────────

class TestEntityDescriptor:
    def test_basic_creation(self):
        e = EntityDescriptor(
            identity=EntityIdentity(id="marine_001", name="Marine"),
            classification=EntityClassification(
                taxonomy=["Entity", "Unit", "Infantry"],
                roles=["army", "defender"],
            ),
            capabilities=EntityCapabilities(abilities=["move", "attack_ground"]),
            affordances=EntityAffordances(
                creates=["sustained_ranged_damage"],
                enables=["bio_ball"],
            ),
            state=EntityState(location=(100.0, 50.0), health=45.0),
        )
        assert e.identity.name == "Marine"
        assert e.classification.is_a("Entity")
        assert e.classification.is_a("Infantry")
        assert not e.classification.is_a("Vehicle")
        assert e.capabilities.has("move")
        assert not e.capabilities.has("fly")
        assert "sustained_ranged_damage" in e.affordances.creates

    def test_serialization_roundtrip(self):
        e = EntityDescriptor(
            identity=EntityIdentity(id="marine_001", name="Marine"),
            classification=EntityClassification(taxonomy=["Entity", "Unit"]),
            capabilities=EntityCapabilities(abilities=["move"]),
            state=EntityState(health=45.0),
            observations=[EntityObservation(tick=100, source="sc2_api", observation="visible")],
            hypotheses=[EntityHypothesis(statement="is_damaged", confidence=0.7)],
        )
        d = e.to_dict()
        e2 = EntityDescriptor.from_dict(d)
        assert e2.identity.name == "Marine"
        assert e2.state.health == 45.0
        assert len(e2.observations) == 1
        assert len(e2.hypotheses) == 1
        assert e2.hypotheses[0].confidence == 0.7

    def test_relationships(self):
        e = EntityDescriptor(
            relationships=EntityRelationships(
                allies=["Marauder"],
                counters=["Zergling"],
                countered_by=["Baneling"],
            ),
        )
        assert "Marauder" in e.relationships.allies
        assert "Zergling" in e.relationships.counters
        assert "Baneling" in e.relationships.countered_by
        targets = e.relationships.all_targets()
        assert "Marauder" in targets
        assert "Zergling" in targets

    def test_history_record(self):
        h = EntityHistory()
        h.record(100, {"health": 45.0})
        h.record(101, {"health": 40.0})
        assert h.created_tick == 0
        assert h.updated_tick == 101
        assert len(h.previous_states) == 2


# ── StateGraph ──────────────────────────────────────────────────────

class TestStateGraph:
    def _make_entity(self, name, roles=None, taxonomy=None):
        return EntityDescriptor(
            identity=EntityIdentity(id=f"{name}_0", name=name),
            classification=EntityClassification(
                taxonomy=taxonomy or ["Entity", "Unit"],
                roles=roles or ["army"],
            ),
            capabilities=EntityCapabilities(abilities=["move"]),
        )

    def test_add_get_entity(self):
        g = StateGraph()
        e = self._make_entity("Marine")
        g.add_entity(e)
        assert g.entity_count() == 1
        assert g.get_entity("Marine_0").identity.name == "Marine"

    def test_query_by_role(self):
        g = StateGraph()
        g.add_entity(self._make_entity("Marine", roles=["army"]))
        g.add_entity(self._make_entity("SCV", roles=["economy"]))
        army = g.query_by_role("army")
        assert len(army) == 1
        assert army[0].identity.name == "Marine"

    def test_query_by_taxonomy(self):
        g = StateGraph()
        g.add_entity(self._make_entity("Marine", taxonomy=["Entity", "Unit", "Infantry"]))
        g.add_entity(self._make_entity("SCV", taxonomy=["Entity", "Unit", "Worker"]))
        infantry = g.query_by_taxonomy("Infantry")
        assert len(infantry) == 1

    def test_edges(self):
        g = StateGraph()
        g.add_entity(self._make_entity("Marine"))
        g.add_entity(self._make_entity("Zergling"))
        g.add_edge(GraphEdge("Marine_0", "Zergling_0", "enemy", weight=0.5))
        neighbors = g.neighbors("Marine_0")
        assert "Zergling_0" in neighbors
        connected = g.connected_by("Marine_0", "enemy")
        assert "Zergling_0" in connected

    def test_remove_entity(self):
        g = StateGraph()
        g.add_entity(self._make_entity("Marine"))
        g.add_edge(GraphEdge("Marine_0", "Zergling_0", "enemy"))
        assert g.remove_entity("Marine_0")
        assert g.entity_count() == 0
        assert g.get_entity("Marine_0") is None

    def test_save_load(self):
        g = StateGraph()
        g.add_entity(self._make_entity("Marine"))
        g.tick(42)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            g.save(path)
            g2 = StateGraph()
            g2.load(path)
            assert g2.entity_count() == 1
            assert g2.current_tick == 42
        finally:
            os.unlink(path)


# ── CausalGraph ─────────────────────────────────────────────────────

class TestCausalGraph:
    def test_record_events(self):
        cg = CausalGraph()
        e1 = cg.record_event(EventType.UNIT_CREATED, 100, "u1", "Marine", "Marine created")
        e2 = cg.record_event(EventType.ARMY_ENGAGED, 150, "u1", "Marine", "Marine attacks")
        assert len(cg._events) == 2
        assert e1.tick == 100

    def test_record_convenience(self):
        cg = CausalGraph()
        cg.record_unit_destroyed("u1", "Marine", 200, killed_by="Zergling")
        assert len(cg._events) == 1
        event = list(cg._events.values())[0]
        assert event.event_type == EventType.UNIT_DESTROYED

    def test_auto_consequences(self):
        cg = CausalGraph()
        cg.record_unit_destroyed("u1", "Marine", 200)
        # Should auto-generate a consequence
        assert len(cg._consequences) >= 1

    def test_get_events_at_tick(self):
        cg = CausalGraph()
        cg.record_event(EventType.UNIT_CREATED, 100, "u1", "A", "created")
        cg.record_event(EventType.UNIT_CREATED, 100, "u2", "B", "created")
        cg.record_event(EventType.UNIT_CREATED, 200, "u3", "C", "created")
        tick100 = cg.get_events_at_tick(100)
        assert len(tick100) == 2

    def test_get_events_for_entity(self):
        cg = CausalGraph()
        cg.record_event(EventType.UNIT_CREATED, 100, "u1", "A", "created")
        cg.record_event(EventType.UNIT_DAMAGED, 150, "u1", "A", "damaged")
        cg.record_event(EventType.UNIT_DESTROYED, 200, "u1", "A", "destroyed")
        events = cg.get_events_for_entity("u1")
        assert len(events) == 3

    def test_save_load(self):
        cg = CausalGraph()
        cg.record_event(EventType.UNIT_CREATED, 100, "u1", "Marine", "created")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cg.save(path)
            cg2 = CausalGraph()
            cg2.load(path)
            assert len(cg2._events) == 1
        finally:
            os.unlink(path)


# ── FrameSystem ─────────────────────────────────────────────────────

class TestFrameSystem:
    def _setup_graph(self):
        g = StateGraph()
        # Own army
        marine = EntityDescriptor(
            identity=EntityIdentity(id="own_Marine_0", name="Marine"),
            embodiment=EntityEmbodiment(controller="self"),
            classification=EntityClassification(
                taxonomy=["Entity", "Unit"], roles=["army"]),
            capabilities=EntityCapabilities(abilities=["attack_air"]),
            affordances=EntityAffordances(creates=["sustained_ranged_damage"]),
            relationships=EntityRelationships(countered_by=["Baneling"]),
        )
        g.add_entity(marine)
        # Enemy
        baneling = EntityDescriptor(
            identity=EntityIdentity(id="enemy_Baneling_0", name="Baneling"),
            embodiment=EntityEmbodiment(controller="enemy"),
            classification=EntityClassification(
                taxonomy=["Entity", "Unit"], roles=["army"]),
            relationships=EntityRelationships(counters=["Marine"]),
        )
        g.add_entity(baneling)
        # Terrain
        terrain = EntityDescriptor(
            identity=EntityIdentity(id="terrain_map", name="Terrain"),
            classification=EntityClassification(
                taxonomy=["Entity", "Environment"], roles=["terrain"]),
            capabilities=EntityCapabilities(abilities=["cliffs_present"]),
            state=EntityState(confidence={"cliff_density": 0.15, "terrain_complexity": 0.3, "map_revealed": 0.4}),
        )
        g.add_entity(terrain)
        return g

    def test_danger_frame(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_danger()
        assert result.frame_type == FrameType.DANGER
        # Baneling counters Marine → should detect threat
        assert "Baneling" in result.answer or len(result.entities) > 0

    def test_opportunity_frame(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_opportunity()
        assert result.frame_type == FrameType.OPPORTUNITY

    def test_composition_frame(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_composition()
        assert "Marine" in result.answer

    def test_terrain_frame(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_terrain()
        assert result.frame_type == FrameType.TERRAIN
        assert "cliff" in result.answer.lower() or "terrain" in result.answer.lower()

    def test_uncertainty_frame(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_uncertainty()
        assert result.frame_type == FrameType.UNCERTAINTY

    def test_threat_assessment(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        result = fs.query_threat_assessment()
        assert result.frame_type == FrameType.THREAT_ASSESSMENT
        # 1 enemy unit, 1 own unit
        assert result.data["enemy_army"] == 1
        assert result.data["own_army"] == 1

    def test_query_all(self):
        g = self._setup_graph()
        fs = FrameSystem(g)
        results = fs.query_all()
        assert len(results) == 8  # all frame types
        # Each result should have a relevance score from perspective
        for r in results:
            assert 0 <= r.relevance <= 1.0

    def test_perspective_weighting(self):
        p = Perspective(
            name="test",
            weights={ft: 1.0 for ft in FrameType},
        )
        result = FrameResult(
            frame_type=FrameType.DANGER,
            query="test", answer="test",
            confidence=1.0, relevance=1.0,
        )
        score = p.score_frame(result)
        assert score == 1.0

    def test_perspectives_exist(self):
        assert PERSPECTIVE_EARLY_GAME.name == "early_game"
        assert PERSPECTIVE_MID_GAME.name == "mid_game"
        assert PERSPECTIVE_UNDER_ATTACK.name == "under_attack"
        assert PERSPECTIVE_ATTACKING.name == "attacking"


# ── NarrativeLayer ──────────────────────────────────────────────────

class TestNarrativeLayer:
    def test_process_events(self):
        cg = CausalGraph()
        nl = NarrativeLayer(cg)

        cg.record_event(EventType.UNIT_CREATED, 100, "u1", "Marine", "Marine created")
        cg.record_event(EventType.ARMY_ENGAGED, 150, "u1", "Marine", "Marine attacks")
        cg.record_unit_destroyed("u1", "Marine", 200, "Zergling")

        events = cg.get_events_in_range(100, 200)
        nl.process_tick(200, events)

        active = nl.get_active_narratives()
        assert len(active) >= 1

    def test_narrative_serialization(self):
        n = Narrative(
            narrative_id="n1",
            narrative_type=NarrativeType.ENGAGEMENT,
            title="Test Battle",
            events=[
                NarrativeEvent("e1", 100, "Setup", 0.3, NarrativeArc.SETUP),
                NarrativeEvent("e2", 150, "Climax", 0.9, NarrativeArc.CLIMAX),
            ],
            start_tick=100,
            end_tick=200,
            outcome="victory",
        )
        d = n.to_dict()
        n2 = Narrative.from_dict(d)
        assert n2.title == "Test Battle"
        assert n2.outcome == "victory"
        assert len(n2.events) == 2
        assert n2.peak_moment().description == "Climax"

    def test_narrative_arc(self):
        n = Narrative(
            narrative_id="n1",
            narrative_type=NarrativeType.ENGAGEMENT,
            title="Test",
        )
        assert NarrativeArc.SETUP.value == "setup"
        assert NarrativeArc.CLIMAX.value == "climax"

    def test_narrative_layer_summary(self):
        nl = NarrativeLayer()
        summary = nl.summary_text()
        assert "NARRATIVE LAYER" in summary


# ── SemanticInterpreter ─────────────────────────────────────────────

class TestSemanticInterpreter:
    def test_ontology_populated(self):
        """Interpreter should auto-populate ontology with SC2 concepts."""
        o = Ontology()
        g = StateGraph()
        from substrate_echo.representational.interpreter import SemanticInterpreter
        si = SemanticInterpreter(o, g)
        assert len(o.concepts) > 0
        # Should have unit concepts
        concept_names = [c.name for c in o.concepts.values()]
        assert "Marine" in concept_names

    def test_terrain_entity(self):
        o = Ontology()
        g = StateGraph()
        from substrate_echo.representational.interpreter import SemanticInterpreter
        si = SemanticInterpreter(o, g)
        terrain = si.get_terrain_entity({
            "cliff_density": 0.15,
            "terrain_complexity": 0.3,
            "map_revealed": 0.5,
        })
        assert terrain.identity.name == "Terrain"
        assert "cliffs_present" in terrain.capabilities.abilities
        assert "defensive_terrain" in terrain.affordances.creates
