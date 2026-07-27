"""Entity Descriptor — the universal cognitive object.

Everything that can be reasoned about is represented through this
structure: units, terrain, resources, strategies, goals, rules.
The descriptor represents our BELIEFS about an entity, not the
entity itself. Uncertainty is always explicit.

Design Principles:
    1. Everything becomes an entity.
    2. Evidence never overwritten by hypotheses.
    3. Uncertainty is always explicit (confidence fields).
    4. Capabilities ≠ Affordances.
       Capabilities: what CAN it do?
       Affordances: what opportunities does it CREATE?
    5. Roles are contextual and multiple.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time


# ── Sub-structures ───────────────────────────────────────────────


@dataclass
class EntityIdentity:
    """The persistent identity of an entity."""
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "aliases": self.aliases}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityIdentity":
        return cls(id=d["id"], name=d["name"], aliases=d.get("aliases", []))


@dataclass
class EntityEmbodiment:
    """Where this entity exists. Separates cognition from physical substrate.

    An AI could migrate between embodiments while remaining the same entity.
    """
    environment: str = ""       # "StarCraft II", "robotics", "simulation"
    controller: str = ""        # who controls this entity
    simulation_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "controller": self.controller,
            "simulation_tick": self.simulation_tick,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityEmbodiment":
        return cls(**{k: d.get(k, v) for k, v in {
            "environment": "", "controller": "", "simulation_tick": 0,
        }.items()})


@dataclass
class EntityClassification:
    """How this entity is classified — taxonomy, not a single type.

    Uses is_a hierarchy from the Ontology. A Marine is_a GroundUnit
    is_a Unit is_a Entity. Inheritance applies transitively.
    """
    taxonomy: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"taxonomy": self.taxonomy, "roles": self.roles}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityClassification":
        return cls(taxonomy=d.get("taxonomy", []), roles=d.get("roles", []))

    def is_a(self, ancestor: str) -> bool:
        return ancestor in self.taxonomy


@dataclass
class EntityComposition:
    """What this entity is made of. Answers: what is it made of?"""
    cost: Dict[str, int] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    attributes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cost": self.cost,
            "requirements": self.requirements,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityComposition":
        return cls(
            cost=d.get("cost", {}),
            requirements=d.get("requirements", []),
            attributes=d.get("attributes", {}),
        )


@dataclass
class EntityCapabilities:
    """What this entity CAN do. Separate from role (what it's doing now).

    Capabilities are intrinsic to the entity type.
    """
    abilities: List[str] = field(default_factory=list)

    def has(self, capability: str) -> bool:
        return capability in self.abilities

    def to_dict(self) -> Dict[str, Any]:
        return {"abilities": self.abilities}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityCapabilities":
        return cls(abilities=d.get("abilities", []))


@dataclass
class EntityAffordances:
    """What opportunities this entity CREATES for others.

    Affordances are relational — they depend on who's observing.
    A cliff affords climbing only if something can climb.
    A mineral patch affords income only if workers exist.

    This is the field that turns entity descriptions into a knowledge graph.
    """
    creates: List[str] = field(default_factory=list)    # what it provides
    enables: List[str] = field(default_factory=list)    # what becomes possible
    invites: List[str] = field(default_factory=list)    # what it attracts
    denies: List[str] = field(default_factory=list)     # what it blocks
    competes_for: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "creates": self.creates,
            "enables": self.enables,
            "invites": self.invites,
            "denies": self.denies,
            "competes_for": self.competes_for,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityAffordances":
        return cls(
            creates=d.get("creates", []),
            enables=d.get("enables", []),
            invites=d.get("invites", []),
            denies=d.get("denies", []),
            competes_for=d.get("competes_for", []),
        )


@dataclass
class EntityRelationships:
    """How this entity connects to others. Becomes the knowledge graph."""
    allies: List[str] = field(default_factory=list)
    counters: List[str] = field(default_factory=list)
    countered_by: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    synergizes_with: List[str] = field(default_factory=list)
    adjacent_to: List[str] = field(default_factory=list)
    contains: List[str] = field(default_factory=list)
    part_of: List[str] = field(default_factory=list)
    custom: Dict[str, List[str]] = field(default_factory=dict)

    def all_targets(self) -> List[str]:
        """All entity names this entity relates to."""
        targets = set()
        for field_name in ("allies", "counters", "countered_by", "requires",
                           "synergizes_with", "adjacent_to", "contains", "part_of"):
            targets.update(getattr(self, field_name))
        for v in self.custom.values():
            targets.update(v)
        return list(targets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allies": self.allies, "counters": self.counters,
            "countered_by": self.countered_by, "requires": self.requires,
            "synergizes_with": self.synergizes_with,
            "adjacent_to": self.adjacent_to, "contains": self.contains,
            "part_of": self.part_of, "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityRelationships":
        return cls(**{k: d.get(k, v) for k, v in {
            "allies": [], "counters": [], "countered_by": [],
            "requires": [], "synergizes_with": [], "adjacent_to": [],
            "contains": [], "part_of": [], "custom": {},
        }.items()})


@dataclass
class EntityState:
    """Dynamic state — changes every tick. Our BELIEFS about current state.

    Every field carries a confidence because we may not know the true value.
    """
    location: Optional[Tuple[float, float]] = None
    health: Optional[float] = None
    health_confidence: float = 1.0
    shields: Optional[float] = None
    energy: Optional[float] = None
    velocity: Optional[Tuple[float, float]] = None
    current_order: str = ""
    target: str = ""
    status: List[str] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "confidence": self.confidence,
            "current_order": self.current_order,
            "target": self.target,
        }
        if self.location is not None:
            d["location"] = list(self.location)
        if self.health is not None:
            d["health"] = self.health
            d["health_confidence"] = self.health_confidence
        if self.shields is not None:
            d["shields"] = self.shields
        if self.energy is not None:
            d["energy"] = self.energy
        if self.velocity is not None:
            d["velocity"] = list(self.velocity)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityState":
        loc = tuple(d["location"]) if "location" in d else None
        vel = tuple(d["velocity"]) if "velocity" in d else None
        return cls(
            location=loc, health=d.get("health"),
            health_confidence=d.get("health_confidence", 1.0),
            shields=d.get("shields"), energy=d.get("energy"),
            velocity=vel, current_order=d.get("current_order", ""),
            target=d.get("target", ""), status=d.get("status", []),
            confidence=d.get("confidence", {}),
        )


@dataclass
class EntityObservation:
    """A raw measurement. Not a fact — a measurement with confidence."""
    tick: int
    source: str
    observation: str
    confidence: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick, "source": self.source,
            "observation": self.observation,
            "confidence": self.confidence, "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityObservation":
        return cls(**{k: d.get(k, v) for k, v in {
            "tick": 0, "source": "", "observation": "",
            "confidence": 1.0, "data": {},
        }.items()})


@dataclass
class EntityHypothesis:
    """A belief about this entity. Never overwrites evidence."""
    statement: str
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    opposing_evidence: List[str] = field(default_factory=list)
    tickCreated: int = 0
    tickLastUpdated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "opposing_evidence": self.opposing_evidence,
            "tickCreated": self.tickCreated,
            "tickLastUpdated": self.tickLastUpdated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityHypothesis":
        return cls(**{k: d.get(k, v) for k, v in {
            "statement": "", "confidence": 0.0,
            "supporting_evidence": [], "opposing_evidence": [],
            "tickCreated": 0, "tickLastUpdated": 0,
        }.items()})


@dataclass
class EntityEvidence:
    """Accumulated evidence. Separate from hypotheses so they never overwrite."""
    observations: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    proven: List[str] = field(default_factory=list)
    disproven: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "experiments": self.experiments,
            "counterexamples": self.counterexamples,
            "proven": self.proven,
            "disproven": self.disproven,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityEvidence":
        return cls(**{k: d.get(k, v) for k, v in {
            "observations": [], "experiments": [], "counterexamples": [],
            "proven": [], "disproven": [],
        }.items()})


@dataclass
class EntityCausality:
    """Causal links: what causes what. Enables forward reasoning."""
    causes: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"causes": self.causes, "effects": self.effects}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityCausality":
        return cls(causes=d.get("causes", []), effects=d.get("effects", []))


@dataclass
class EntityHistory:
    """Temporal record. Enables narrative and temporal reasoning."""
    created_tick: int = 0
    updated_tick: int = 0
    previous_states: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    def record(self, tick: int, state_snapshot: Dict[str, Any]) -> None:
        self.updated_tick = tick
        self.previous_states.append(state_snapshot)
        if len(self.previous_states) > self.max_history:
            self.previous_states = self.previous_states[-self.max_history:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_tick": self.created_tick,
            "updated_tick": self.updated_tick,
            "previous_states": self.previous_states[-20:],  # keep last 20
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityHistory":
        return cls(
            created_tick=d.get("created_tick", 0),
            updated_tick=d.get("updated_tick", 0),
            previous_states=d.get("previous_states", []),
        )


# ── Entity Descriptor ────────────────────────────────────────────


@dataclass
class EntityDescriptor:
    """The universal cognitive object. Our representation of a thing.

    Everything that can be reasoned about — units, terrain, resources,
    strategies, goals, rules — is represented through this structure.

    The actual entity is in the world. This is our mind's model of it.
    That distinction is fundamental: we reason over our BELIEFS about
    entities, not the entities themselves.
    """
    identity: EntityIdentity = field(default_factory=lambda: EntityIdentity("", ""))
    embodiment: EntityEmbodiment = field(default_factory=EntityEmbodiment)
    classification: EntityClassification = field(default_factory=EntityClassification)
    composition: EntityComposition = field(default_factory=EntityComposition)
    capabilities: EntityCapabilities = field(default_factory=EntityCapabilities)
    affordances: EntityAffordances = field(default_factory=EntityAffordances)
    relationships: EntityRelationships = field(default_factory=EntityRelationships)
    state: EntityState = field(default_factory=EntityState)
    observations: List[EntityObservation] = field(default_factory=list)
    hypotheses: List[EntityHypothesis] = field(default_factory=list)
    evidence: EntityEvidence = field(default_factory=EntityEvidence)
    causality: EntityCausality = field(default_factory=EntityCausality)
    history: EntityHistory = field(default_factory=EntityHistory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "embodiment": self.embodiment.to_dict(),
            "classification": self.classification.to_dict(),
            "composition": self.composition.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "affordances": self.affordances.to_dict(),
            "relationships": self.relationships.to_dict(),
            "state": self.state.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "evidence": self.evidence.to_dict(),
            "causality": self.causality.to_dict(),
            "history": self.history.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityDescriptor":
        return cls(
            identity=EntityIdentity.from_dict(d.get("identity", {})),
            embodiment=EntityEmbodiment.from_dict(d.get("embodiment", {})),
            classification=EntityClassification.from_dict(d.get("classification", {})),
            composition=EntityComposition.from_dict(d.get("composition", {})),
            capabilities=EntityCapabilities.from_dict(d.get("capabilities", {})),
            affordances=EntityAffordances.from_dict(d.get("affordances", {})),
            relationships=EntityRelationships.from_dict(d.get("relationships", {})),
            state=EntityState.from_dict(d.get("state", {})),
            observations=[EntityObservation.from_dict(o) for o in d.get("observations", [])],
            hypotheses=[EntityHypothesis.from_dict(h) for h in d.get("hypotheses", [])],
            evidence=EntityEvidence.from_dict(d.get("evidence", {})),
            causality=EntityCausality.from_dict(d.get("causality", {})),
            history=EntityHistory.from_dict(d.get("history", {})),
        )

    def __repr__(self) -> str:
        return f"EntityDescriptor({self.identity.name}, {self.classification.taxonomy})"
