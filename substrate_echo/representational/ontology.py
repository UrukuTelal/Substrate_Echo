"""Ontology — Static knowledge about what exists and how it behaves.

The ontology is intentionally small. It contains only what's needed
for reasoning — not an encyclopedia. It changes slowly (game patches,
new discoveries) while the world model changes every tick.

Layers:
    Concept         — what things are
    TaxonomyNode    — how things are classified (is_a hierarchy)
    PropertySchema  — what properties things can have
    Rule            — how the world behaves (causal)
    Constraint      — what cannot happen
    Ontology        — the unified static knowledge store
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class ConceptCategory(Enum):
    """Broad categories of things that can exist."""
    UNIT = "unit"
    STRUCTURE = "structure"
    TERRAIN = "terrain"
    RESOURCE = "resource"
    TECHNOLOGY = "technology"
    STRATEGY = "strategy"
    GOAL = "goal"
    RULE = "rule"
    CONSTRAINT = "constraint"
    AGENT = "agent"
    ABSTRACTION = "abstraction"


@dataclass
class Concept:
    """A thing that exists in the world.

    Concepts are the atoms of the ontology. They answer:
    > What is this?

    Concepts are deliberately minimal. They carry identity and
    classification, not state. State lives in EntityDescriptor.
    """
    name: str
    category: ConceptCategory
    aliases: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "aliases": self.aliases,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concept":
        return cls(
            name=d["name"],
            category=ConceptCategory(d["category"]),
            aliases=d.get("aliases", []),
            description=d.get("description", ""),
        )


@dataclass
class TaxonomyNode:
    """A node in the is_a hierarchy.

    Taxonomy enables reasoning by inheritance. If GroundUnit cannot
    cross cliff, then Marine (which is_a GroundUnit) cannot cross cliff.

    > How are things classified?
    """
    name: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    properties_inherited: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "children": self.children,
            "properties_inherited": self.properties_inherited,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaxonomyNode":
        return cls(**d)


@dataclass
class PropertySchema:
    """Defines a property that entities can have.

    Properties are typed and bounded. They answer:
    > What can things have?

    Example: health (float, 0-1), position (2D), cost_minerals (int, 0+)
    """
    name: str
    dtype: str  # "float", "int", "bool", "str", "vec2"
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PropertySchema":
        return cls(**{k: d.get(k, v) for k, v in {
            "name": "", "dtype": "float", "default": None,
            "min_value": None, "max_value": None, "unit": "", "description": "",
        }.items()})


@dataclass
class Rule:
    """A causal relationship: condition → consequence.

    Rules answer:
    > How does the world behave?

    Rules are composable. Multiple rules chain into causal graphs.
    Rules have confidence because some are learned from observation,
    not hardcoded.
    """
    rule_id: str
    condition: str          # e.g. "supply_used >= supply_cap"
    consequence: str        # e.g. "production_blocked"
    confidence: float = 1.0
    source: str = "hardcoded"  # "hardcoded", "learned", "inferred"
    description: str = ""
    tick_first_observed: int = 0
    observation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "condition": self.condition,
            "consequence": self.consequence,
            "confidence": self.confidence,
            "source": self.source,
            "description": self.description,
            "tick_first_observed": self.tick_first_observed,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rule":
        return cls(**{k: d.get(k, v) for k, v in {
            "rule_id": "", "condition": "", "consequence": "",
            "confidence": 1.0, "source": "hardcoded", "description": "",
            "tick_first_observed": 0, "observation_count": 0,
        }.items()})


@dataclass
class Constraint:
    """What cannot happen. Constraints define the boundaries of reality.

    Constraints answer:
    > What is impossible?

    Constraints dramatically shrink the search space. Instead of
    evaluating "should Drone attack air?", the system knows it
    cannot and never considers it.
    """
    constraint_id: str
    subject: str        # e.g. "Drone", "GroundUnit", "any"
    predicate: str      # "cannot", "must", "requires"
    object: str         # e.g. "attack_air", "move", "Barracks"
    scope: str = "always"  # "always", "unless_", "when_"
    scope_detail: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "scope": self.scope,
            "scope_detail": self.scope_detail,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Constraint":
        return cls(**{k: d.get(k, v) for k, v in {
            "constraint_id": "", "subject": "", "predicate": "",
            "object": "", "scope": "always", "scope_detail": "",
            "description": "",
        }.items()})


class Ontology:
    """The unified static knowledge store.

    All subsystems query the ontology for definitions, taxonomy,
    properties, rules, and constraints. The ontology changes slowly
    — new concepts and rules are added when the bot discovers them,
    but the core structure is stable.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.taxonomy: Dict[str, TaxonomyNode] = {}
        self.properties: Dict[str, PropertySchema] = {}
        self.rules: List[Rule] = []
        self.constraints: List[Constraint] = []

    # ── Concepts ─────────────────────────────────────────────────

    def add_concept(self, concept: Concept) -> None:
        self.concepts[concept.name] = concept
        for alias in concept.aliases:
            if alias not in self.concepts:
                self.concepts[alias] = concept

    def get_concept(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)

    def concepts_in_category(self, category: ConceptCategory) -> List[Concept]:
        return [c for c in self.concepts.values()
                if c.category == category and c.name == c.name]

    # ── Taxonomy ─────────────────────────────────────────────────

    def add_taxonomy_node(self, node: TaxonomyNode) -> None:
        self.taxonomy[node.name] = node
        if node.parent and node.parent in self.taxonomy:
            parent = self.taxonomy[node.parent]
            if node.name not in parent.children:
                parent.children.append(node.name)

    def is_a(self, child: str, parent: str) -> bool:
        """Check if child is_a parent (transitive)."""
        if child == parent:
            return True
        node = self.taxonomy.get(child)
        if not node:
            return False
        if node.parent == parent:
            return True
        return self.is_a(node.parent, parent) if node.parent else False

    def ancestors(self, name: str) -> List[str]:
        """Return the chain from name up to root."""
        chain = []
        node = self.taxonomy.get(name)
        while node and node.parent:
            chain.append(node.parent)
            node = self.taxonomy.get(node.parent)
        return chain

    def descendants(self, name: str) -> List[str]:
        """Return all descendants (recursive)."""
        node = self.taxonomy.get(name)
        if not node:
            return []
        result = []
        for child in node.children:
            result.append(child)
            result.extend(self.descendants(child))
        return result

    # ── Properties ───────────────────────────────────────────────

    def add_property_schema(self, prop: PropertySchema) -> None:
        self.properties[prop.name] = prop

    def get_property_schema(self, name: str) -> Optional[PropertySchema]:
        return self.properties.get(name)

    # ── Rules ────────────────────────────────────────────────────

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def rules_for(self, concept_name: str) -> List[Rule]:
        """Find rules where concept_name appears in condition or consequence."""
        return [r for r in self.rules
                if concept_name in r.condition or concept_name in r.consequence]

    def active_rules(self, min_confidence: float = 0.5) -> List[Rule]:
        return [r for r in self.rules if r.confidence >= min_confidence]

    # ── Constraints ──────────────────────────────────────────────

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def constraints_for(self, subject: str) -> List[Constraint]:
        """Find constraints that apply to a subject (transitive via taxonomy)."""
        subjects = [subject] + self.ancestors(subject)
        return [c for c in self.constraints if c.subject in subjects]

    def is_constrained(self, subject: str, predicate: str, obj: str) -> bool:
        """Check if a specific action is forbidden."""
        for c in self.constraints_for(subject):
            if c.predicate == predicate and c.object == obj:
                return True
            if c.predicate == predicate and c.object == "any":
                return True
        return False

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()
                         if v.name == k},  # deduplicate aliases
            "taxonomy": {k: v.to_dict() for k, v in self.taxonomy.items()},
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "rules": [r.to_dict() for r in self.rules],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            self.concepts = {
                k: Concept.from_dict(v)
                for k, v in data.get("concepts", {}).items()
            }
            self.taxonomy = {
                k: TaxonomyNode.from_dict(v)
                for k, v in data.get("taxonomy", {}).items()
            }
            self.properties = {
                k: PropertySchema.from_dict(v)
                for k, v in data.get("properties", {}).items()
            }
            self.rules = [Rule.from_dict(r) for r in data.get("rules", [])]
            self.constraints = [
                Constraint.from_dict(c) for c in data.get("constraints", [])
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # ── Summary ──────────────────────────────────────────────────

    def summary_text(self) -> str:
        lines = ["=== ONTOLOGY ==="]
        lines.append(f"  Concepts: {len(self.concepts)}")
        cats: Dict[str, int] = {}
        for c in self.concepts.values():
            if c.name in self.concepts and self.concepts[c.name] is c:
                cats[c.category.value] = cats.get(c.category.value, 0) + 1
        for cat, count in sorted(cats.items()):
            lines.append(f"    {cat}: {count}")
        lines.append(f"  Taxonomy nodes: {len(self.taxonomy)}")
        lines.append(f"  Property schemas: {len(self.properties)}")
        lines.append(f"  Rules: {len(self.rules)}")
        lines.append(f"  Constraints: {len(self.constraints)}")
        return "\n".join(lines)
