"""Frame System — queries over the world model through different lenses.

Frames are not static templates. They are queries that dynamically
extract relevant information from the StateGraph. Each frame answers
a different strategic question:

    "What could kill me?"
    "What am I uncertain about?"
    "What are my strongest units?"
    "Where should I attack?"
    "What is the enemy doing?"

Frames transform the world model into actionable perspectives.
The Perspective class weights which frames dominate reasoning.

Architecture:
    StateGraph + CausalGraph
          |
          v
    Frame System (query engine)
      - DangerFrame: what threatens me?
      - OpportunityFrame: where are the openings?
      - UncertaintyFrame: what don't I know?
      - CompositionFrame: what do I/we have?
      - TerrainFrame: where are the terrain advantages?
      - TemporalFrame: what patterns are emerging?
          |
          v
    FrameResult (structured answer)
          |
          v
    Perspective (weights which frames matter right now)
          |
          v
    Kernel / Governance / Action selection
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class FrameType(Enum):
    """Types of queries over the world model."""
    DANGER = "danger"
    OPPORTUNITY = "opportunity"
    UNCERTAINTY = "uncertainty"
    COMPOSITION = "composition"
    TERRAIN = "terrain"
    TEMPORAL = "temporal"
    ECONOMIC = "economic"
    THREAT_ASSESSMENT = "threat_assessment"


@dataclass
class FrameResult:
    """The answer to a frame query.

    Each result includes:
      - The extracted information
      - Confidence in the answer
      - Which entities are relevant
      - A relevance score (how important right now)
    """
    frame_type: FrameType
    query: str
    answer: str
    confidence: float = 0.5
    relevance: float = 0.5          # how important right now
    entities: List[str] = field(default_factory=list)  # relevant entity_ids
    data: Dict[str, Any] = field(default_factory=dict)
    tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_type": self.frame_type.value,
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "relevance": round(self.relevance, 3),
            "entities": self.entities,
            "data": self.data,
            "tick": self.tick,
        }


@dataclass
class Perspective:
    """A weighting over frames — what matters right now.

    Perspectives shift over time:
      - Early game: economic and uncertainty frames dominate
      - Mid game: composition and terrain frames dominate
      - Under attack: danger frame dominates
      - Attacking: opportunity frame dominates

    The perspective weights which frame results feed into decision-making.
    """
    name: str = "default"
    weights: Dict[FrameType, float] = field(default_factory=lambda: {
        FrameType.DANGER: 0.5,
        FrameType.OPPORTUNITY: 0.5,
        FrameType.UNCERTAINTY: 0.5,
        FrameType.COMPOSITION: 0.5,
        FrameType.TERRAIN: 0.5,
        FrameType.TEMPORAL: 0.3,
        FrameType.ECONOMIC: 0.5,
        FrameType.THREAT_ASSESSMENT: 0.5,
    })
    tick: int = 0

    def score_frame(self, result: FrameResult) -> float:
        """How much should this frame result influence decisions?"""
        weight = self.weights.get(result.frame_type, 0.5)
        return weight * result.relevance * result.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weights": {k.value: v for k, v in self.weights.items()},
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Perspective":
        weights = {FrameType(k): v for k, v in d.get("weights", {}).items()}
        return cls(name=d.get("name", "default"), weights=weights, tick=d.get("tick", 0))


# ── Built-in Perspectives ─────────────────────────────────────────

PERSPECTIVE_EARLY_GAME = Perspective(
    name="early_game",
    weights={
        FrameType.DANGER: 0.3,
        FrameType.OPPORTUNITY: 0.4,
        FrameType.UNCERTAINTY: 0.8,  # lots we don't know
        FrameType.COMPOSITION: 0.3,
        FrameType.TERRAIN: 0.2,
        FrameType.TEMPORAL: 0.3,
        FrameType.ECONOMIC: 0.9,     # economy matters most early
        FrameType.THREAT_ASSESSMENT: 0.4,
    },
)

PERSPECTIVE_MID_GAME = Perspective(
    name="mid_game",
    weights={
        FrameType.DANGER: 0.6,
        FrameType.OPPORTUNITY: 0.7,
        FrameType.UNCERTAINTY: 0.5,
        FrameType.COMPOSITION: 0.8,
        FrameType.TERRAIN: 0.7,
        FrameType.TEMPORAL: 0.5,
        FrameType.ECONOMIC: 0.5,
        FrameType.THREAT_ASSESSMENT: 0.7,
    },
)

PERSPECTIVE_UNDER_ATTACK = Perspective(
    name="under_attack",
    weights={
        FrameType.DANGER: 1.0,       # survival first
        FrameType.OPPORTUNITY: 0.2,
        FrameType.UNCERTAINTY: 0.3,
        FrameType.COMPOSITION: 0.7,
        FrameType.TERRAIN: 0.6,
        FrameType.TEMPORAL: 0.2,
        FrameType.ECONOMIC: 0.3,
        FrameType.THREAT_ASSESSMENT: 0.9,
    },
)

PERSPECTIVE_ATTACKING = Perspective(
    name="attacking",
    weights={
        FrameType.DANGER: 0.4,
        FrameType.OPPORTUNITY: 1.0,   # maximize openings
        FrameType.UNCERTAINTY: 0.4,
        FrameType.COMPOSITION: 0.7,
        FrameType.TERRAIN: 0.8,
        FrameType.TEMPORAL: 0.4,
        FrameType.ECONOMIC: 0.3,
        FrameType.THREAT_ASSESSMENT: 0.6,
    },
)


class FrameSystem:
    """Queries the world model through different strategic lenses.

    Each frame is a function that extracts specific information from
    the StateGraph. The Perspective weights which frames dominate
    the current reasoning.

    Usage:
        frame_system = FrameSystem(state_graph, causal_graph)
        perspective = Perspective("mid_game", ...)
        results = frame_system.query_all(perspective, tick=current_tick)
    """

    def __init__(self, state_graph, causal_graph=None):
        self.graph = state_graph
        self.causal = causal_graph
        self._perspective = PERSPECTIVE_MID_GAME

    def set_perspective(self, perspective: Perspective) -> None:
        self._perspective = perspective

    def get_perspective(self) -> Perspective:
        return self._perspective

    # ── Query Methods ────────────────────────────────────────────

    def query_all(self, perspective: Optional[Perspective] = None,
                  tick: int = 0) -> List[FrameResult]:
        """Run all frame queries and return results weighted by perspective."""
        perspective = perspective or self._perspective
        results = [
            self.query_danger(tick),
            self.query_opportunity(tick),
            self.query_uncertainty(tick),
            self.query_composition(tick),
            self.query_terrain(tick),
            self.query_temporal(tick),
            self.query_economic(tick),
            self.query_threat_assessment(tick),
        ]
        # Weight each result by perspective
        for r in results:
            r.relevance = perspective.score_frame(r)
        return results

    def query_danger(self, tick: int = 0) -> FrameResult:
        """What could kill me?"""
        own = [e for e in self.graph.all_entities()
               if e.embodiment.controller == "self"
               and e.classification.is_a("Unit")]
        enemies = [e for e in self.graph.all_entities()
                   if e.embodiment.controller == "enemy"
                   and e.classification.is_a("Unit")]

        threats = []
        threat_ids = []
        for enemy in enemies:
            for own_unit in own:
                if enemy.identity.name in own_unit.relationships.countered_by:
                    threats.append(
                        f"{enemy.identity.name} counters {own_unit.identity.name}")
                    threat_ids.append(enemy.identity.id)

        danger_count = len(threats)
        confidence = min(1.0, danger_count / max(1, len(own)))

        return FrameResult(
            frame_type=FrameType.DANGER,
            query="What could kill me?",
            answer=f"{danger_count} counter-threats identified" if threats else "No immediate threats",
            confidence=confidence,
            relevance=min(1.0, danger_count / 5),
            entities=list(set(threat_ids)),
            data={"threats": threats[:10]},
            tick=tick,
        )

    def query_opportunity(self, tick: int = 0) -> FrameResult:
        """Where are the openings?"""
        own = [e for e in self.graph.all_entities()
               if e.embodiment.controller == "self"]
        enemies = [e for e in self.graph.all_entities()
                   if e.embodiment.controller == "enemy"]

        openings = []
        opening_ids = []
        for own_unit in own:
            for enemy in enemies:
                if own_unit.identity.name in enemy.relationships.countered_by:
                    openings.append(
                        f"{own_unit.identity.name} counters {enemy.identity.name}")
                    opening_ids.append(own_unit.identity.id)

        # Terrain-based opportunities
        terrain = self.graph.get_entity("terrain_map")
        if terrain and "cliffs_present" in terrain.capabilities.abilities:
            cliff_jumpers = [e for e in own
                             if e.capabilities.has("cliff_jump")
                             or e.capabilities.has("fly")]
            if cliff_jumpers:
                openings.append(f"{len(cliff_jumpers)} units can exploit terrain")
                opening_ids.extend(e.identity.id for e in cliff_jumpers)

        return FrameResult(
            frame_type=FrameType.OPPORTUNITY,
            query="Where are the openings?",
            answer=f"{len(openings)} opportunities found" if openings else "No clear openings",
            confidence=min(1.0, len(openings) / 3),
            relevance=min(1.0, len(openings) / 5),
            entities=list(set(opening_ids)),
            data={"openings": openings[:10]},
            tick=tick,
        )

    def query_uncertainty(self, tick: int = 0) -> FrameResult:
        """What don't I know?"""
        enemies = [e for e in self.graph.all_entities()
                   if e.embodiment.controller == "enemy"]

        unknown_count = 0
        low_confidence = []
        for e in enemies:
            if e.state.health_confidence < 0.5:
                unknown_count += 1
                low_confidence.append(e.identity.name)

        # Check if we have low map visibility
        terrain = self.graph.get_entity("terrain_map")
        map_revealed = 0.0
        if terrain:
            map_revealed = terrain.state.confidence.get("map_revealed", 0.0)

        uncertainty_score = (unknown_count / max(1, len(enemies)) +
                           max(0, 1.0 - map_revealed)) / 2

        return FrameResult(
            frame_type=FrameType.UNCERTAINTY,
            query="What don't I know?",
            answer=f"{unknown_count} low-confidence entities, {map_revealed:.0%} map revealed",
            confidence=1.0 - uncertainty_score,  # inverted — less uncertain = more confident
            relevance=uncertainty_score,
            entities=[],
            data={
                "low_confidence_entities": low_confidence[:10],
                "map_revealed": map_revealed,
            },
            tick=tick,
        )

    def query_composition(self, tick: int = 0) -> FrameResult:
        """What do I/we have?"""
        own = [e for e in self.graph.all_entities()
               if e.embodiment.controller == "self"
               and e.classification.is_a("Unit")]

        composition: Dict[str, int] = {}
        for e in own:
            name = e.identity.name
            composition[name] = composition.get(name, 0) + 1

        total = sum(composition.values())
        dominant = max(composition.items(), key=lambda x: x[1]) if composition else ("none", 0)

        return FrameResult(
            frame_type=FrameType.COMPOSITION,
            query="What do I have?",
            answer=f"{total} units: {', '.join(f'{k}({v})' for k, v in composition.items())}",
            confidence=0.9,
            relevance=0.6,
            entities=[e.identity.id for e in own],
            data={"composition": composition, "dominant": dominant[0]},
            tick=tick,
        )

    def query_terrain(self, tick: int = 0) -> FrameResult:
        """Where are the terrain advantages?"""
        terrain = self.graph.get_entity("terrain_map")
        if not terrain:
            return FrameResult(
                frame_type=FrameType.TERRAIN,
                query="Where are terrain advantages?",
                answer="No terrain data available",
                confidence=0.1, relevance=0.1, tick=tick,
            )

        cliff_density = terrain.state.confidence.get("cliff_density", 0.0)
        complexity = terrain.state.confidence.get("terrain_complexity", 0.0)

        advantages = []
        if cliff_density > 0.05:
            advantages.append(f"Cliff terrain ({cliff_density:.0%} density)")
        if complexity > 0.2:
            advantages.append(f"Complex pathing ({complexity:.0%})")

        own = [e for e in self.graph.all_entities()
               if e.embodiment.controller == "self"]
        cliff_capable = [e for e in own if e.capabilities.has("cliff_jump")
                        or e.capabilities.has("fly")
                        or e.capabilities.has("cliff_walk")]

        if cliff_capable:
            advantages.append(f"{len(cliff_capable)} units can exploit terrain")

        return FrameResult(
            frame_type=FrameType.TERRAIN,
            query="Where are terrain advantages?",
            answer="; ".join(advantages) if advantages else "Flat terrain, no advantages",
            confidence=0.7,
            relevance=min(1.0, cliff_density + complexity),
            entities=[e.identity.id for e in cliff_capable],
            data={"cliff_density": cliff_density, "complexity": complexity},
            tick=tick,
        )

    def query_temporal(self, tick: int = 0) -> FrameResult:
        """What patterns are emerging?"""
        if not self.causal:
            return FrameResult(
                frame_type=FrameType.TEMPORAL,
                query="What patterns emerging?",
                answer="No causal data available",
                confidence=0.1, relevance=0.2, tick=tick,
            )

        recent_events = self.causal.get_events_in_range(
            max(0, tick - 200), tick)

        # Detect patterns
        event_types: Dict[str, int] = {}
        for e in recent_events:
            event_types[e.event_type.value] = event_types.get(e.event_type.value, 0) + 1

        patterns = []
        if event_types.get("unit_destroyed", 0) > 5:
            patterns.append(f"High attrition ({event_types['unit_destroyed']} losses)")
        if event_types.get("army_engaged", 0) > 3:
            patterns.append(f"Frequent engagements ({event_types['army_engaged']})")
        if event_types.get("structure_created", 0) > 5:
            patterns.append(f"Active construction ({event_types['structure_created']})")

        return FrameResult(
            frame_type=FrameType.TEMPORAL,
            query="What patterns emerging?",
            answer="; ".join(patterns) if patterns else "No strong patterns",
            confidence=0.5,
            relevance=min(1.0, len(recent_events) / 20),
            entities=[],
            data={"event_types": event_types, "total_recent": len(recent_events)},
            tick=tick,
        )

    def query_economic(self, tick: int = 0) -> FrameResult:
        """How is the economy?"""
        own = [e for e in self.graph.all_entities()
               if e.embodiment.controller == "self"]

        workers = [e for e in own if e.classification.is_a("Worker")]
        bases = [e for e in own if "command" in e.classification.roles
                 or "economy" in e.classification.roles]

        return FrameResult(
            frame_type=FrameType.ECONOMIC,
            query="How is the economy?",
            answer=f"{len(workers)} workers, {len(bases)} bases",
            confidence=0.8,
            relevance=0.5,
            entities=[e.identity.id for e in workers + bases],
            data={"workers": len(workers), "bases": len(bases)},
            tick=tick,
        )

    def query_threat_assessment(self, tick: int = 0) -> FrameResult:
        """Overall threat level."""
        own_army = [e for e in self.graph.all_entities()
                    if e.embodiment.controller == "self"
                    and "army" in e.classification.roles]
        enemy_army = [e for e in self.graph.all_entities()
                      if e.embodiment.controller == "enemy"
                      and "army" in e.classification.roles]

        own_count = len(own_army)
        enemy_count = len(enemy_army)

        if own_count + enemy_count == 0:
            ratio = 0.5
        else:
            ratio = enemy_count / (own_count + enemy_count)

        if ratio > 0.7:
            level = "HIGH"
        elif ratio > 0.5:
            level = "MEDIUM"
        else:
            level = "LOW"

        return FrameResult(
            frame_type=FrameType.THREAT_ASSESSMENT,
            query="What is the threat level?",
            answer=f"{level} — {enemy_count} enemy vs {own_count} own units",
            confidence=0.7,
            relevance=ratio,
            entities=[e.identity.id for e in enemy_army],
            data={"own_army": own_count, "enemy_army": enemy_count, "ratio": ratio},
            tick=tick,
        )

    # ── Summary ──────────────────────────────────────────────────

    def summary_text(self, results: Optional[List[FrameResult]] = None) -> str:
        if results is None:
            results = self.query_all()
        lines = ["=== FRAME SYSTEM ==="]
        lines.append(f"  Perspective: {self._perspective.name}")
        for r in sorted(results, key=lambda x: -x.relevance):
            lines.append(f"  [{r.frame_type.value}] (rel={r.relevance:.2f}) {r.answer}")
        return "\n".join(lines)
